#!/usr/bin/env python3
"""Run-preserving test of item-specific correspondence beyond global severity.

The script reads existing model outputs only. It makes no API or LLM calls and
writes exclusively to its --output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy
from scipy.stats import rankdata


MODEL_SPECS = OrderedDict(
    [
        ("GPT-4o", "prompts_1_11"),
        ("Gemini-1.5pro", "prompts_1_12"),
        ("Llama-3.2-90b", "prompts_1_13"),
        ("MedLM-Large-1.5", "prompts_1_17"),
    ]
)

QUESTIONNAIRES = OrderedDict(
    [
        (
            "GAD-7",
            {
                "prefix": "GAD7",
                "score_label": "anxiety",
                "n_items": 7,
                "manuscript_model_means": {
                    "GPT-4o": 59.2,
                    "Gemini-1.5pro": 59.4,
                    "Llama-3.2-90b": 61.2,
                    "MedLM-Large-1.5": 59.6,
                },
            },
        ),
        (
            "PHQ-9",
            {
                "prefix": "PHQ9",
                "score_label": "depression",
                "n_items": 9,
                "manuscript_model_means": {
                    "GPT-4o": 71.7,
                    "Gemini-1.5pro": 70.3,
                    "Llama-3.2-90b": 71.4,
                    "MedLM-Large-1.5": 69.8,
                },
            },
        ),
    ]
)

N_RUNS = 10
N_PROMPTS = 5
RUN_SEEDS = [100 * N_PROMPTS + run for run in range(N_RUNS)]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).resolve().parent
    default_data = default_output.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12012026)
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def score_column(frame: pd.DataFrame, score_label: str) -> str:
    marker = f"\n{score_label} scores\n"
    matches = [column for column in frame.columns if marker in column]
    if len(matches) != 1:
        raise ValueError(f"Expected one column containing {marker!r}; got {matches}")
    return matches[0]


def split_score_vectors(series: pd.Series, n_items: int) -> np.ndarray:
    vectors = series.astype(str).str.split(",")
    lengths = vectors.str.len()
    if not lengths.eq(n_items).all():
        counts = lengths.value_counts().to_dict()
        raise ValueError(f"Unexpected item-vector lengths: {counts}")
    matrix = np.asarray(vectors.tolist(), dtype=float)
    if not np.isin(matrix, [0.0, 1.0, 2.0, 3.0]).all():
        raise ValueError("Predicted item responses include values outside 0--3")
    return matrix


def select_prompt_sets(trainsets: list[str]) -> list[list[str]]:
    series = pl.Series("trainsets", sorted(trainsets))
    return [series.sample(n=N_PROMPTS, seed=seed).to_list() for seed in RUN_SEEDS]


def load_data(data_root: Path):
    questionnaire_path = data_root / "questionnaire_data_clean.csv"
    questionnaire = pd.read_csv(questionnaire_path)

    input_manifest = [
        {
            "source": str(questionnaire_path),
            "bytes": questionnaire_path.stat().st_size,
            "sha256": sha256_file(questionnaire_path),
        }
    ]

    model_frames: dict[str, pd.DataFrame] = {}
    model_arrays: dict[str, dict[str, np.ndarray]] = {}
    audit_rows = []
    common_participants: set[int] | None = None
    reference_trainsets: list[str] | None = None

    for model, directory in MODEL_SPECS.items():
        result_path = data_root / directory / "results.csv"
        frame = pd.read_csv(
            result_path,
            dtype={"train_participant_ids": "string"},
        )
        required = {"val_train_participant_id", "val_participant_id", "train_participant_ids"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{result_path} lacks required columns: {sorted(missing)}")
        if frame["val_train_participant_id"].duplicated().any():
            raise ValueError(f"Duplicate prompt identifiers in {result_path}")

        participants = set(frame["val_participant_id"].astype(int))
        common_participants = (
            participants if common_participants is None else common_participants.intersection(participants)
        )
        trainsets = sorted(frame["train_participant_ids"].astype(str).unique().tolist())
        if reference_trainsets is None:
            reference_trainsets = trainsets
        elif trainsets != reference_trainsets:
            raise ValueError("Models do not share the same prompt-configuration identifiers")

        arrays = {}
        for questionnaire_name, spec in QUESTIONNAIRES.items():
            column = score_column(frame, spec["score_label"])
            arrays[questionnaire_name] = split_score_vectors(frame[column], spec["n_items"])

        model_frames[model] = frame[
            ["val_train_participant_id", "val_participant_id", "train_participant_ids"]
        ].copy()
        model_arrays[model] = arrays
        audit_rows.append(
            {
                "model": model,
                "directory": directory,
                "rows": len(frame),
                "participants": len(participants),
                "prompt_configurations": len(trainsets),
                "duplicate_prompt_ids": int(frame["val_train_participant_id"].duplicated().sum()),
            }
        )
        input_manifest.append(
            {
                "source": str(result_path),
                "bytes": result_path.stat().st_size,
                "sha256": sha256_file(result_path),
            }
        )

    assert common_participants is not None
    assert reference_trainsets is not None
    participants = sorted(common_participants)
    if len(participants) != 106:
        raise ValueError(f"Expected 106 common evaluated participants; found {len(participants)}")

    truth = {}
    q_indexed = questionnaire.set_index("participant_id")
    for questionnaire_name, spec in QUESTIONNAIRES.items():
        columns = [f'{spec["prefix"]}_Q{i}' for i in range(1, spec["n_items"] + 1)]
        missing_columns = [column for column in columns if column not in q_indexed.columns]
        if missing_columns:
            raise ValueError(f"Ground truth lacks columns: {missing_columns}")
        matrix = q_indexed.loc[participants, columns].to_numpy(dtype=float)
        if not np.isin(matrix, [0.0, 1.0, 2.0, 3.0]).all():
            raise ValueError(f"{questionnaire_name} ground truth includes values outside 0--3")
        score_column_name = f'{spec["prefix"]}_score'
        recorded_total = q_indexed.loc[participants, score_column_name].to_numpy(dtype=float)
        if not np.array_equal(matrix.sum(axis=1), recorded_total):
            raise ValueError(f"{questionnaire_name} item sums differ from recorded totals")
        truth[questionnaire_name] = matrix

    selected_sets = select_prompt_sets(reference_trainsets)
    return (
        participants,
        truth,
        model_frames,
        model_arrays,
        selected_sets,
        pd.DataFrame(audit_rows),
        input_manifest,
    )


def build_prediction_matrices(
    participants: list[int],
    model_frames: dict[str, pd.DataFrame],
    model_arrays: dict[str, dict[str, np.ndarray]],
    selected_sets: list[list[str]],
):
    pipeline_predictions: dict[str, list[np.ndarray]] = {
        questionnaire: [] for questionnaire in QUESTIONNAIRES
    }
    all_prompt_predictions: dict[str, list[np.ndarray]] = {
        questionnaire: [] for questionnaire in QUESTIONNAIRES
    }
    count_rows = []

    for model in MODEL_SPECS:
        identifiers = model_frames[model].copy()
        identifiers["val_participant_id"] = identifiers["val_participant_id"].astype(int)
        identifiers["train_participant_ids"] = identifiers["train_participant_ids"].astype(str)

        for questionnaire_name, spec in QUESTIONNAIRES.items():
            item_columns = [f"item_{i}" for i in range(1, spec["n_items"] + 1)]
            score_frame = pd.concat(
                [
                    identifiers[["val_participant_id", "train_participant_ids"]].reset_index(drop=True),
                    pd.DataFrame(model_arrays[model][questionnaire_name], columns=item_columns),
                ],
                axis=1,
            )

            all_grouped = (
                score_frame.groupby("val_participant_id", sort=True)[item_columns]
                .mean()
                .reindex(participants)
            )
            if all_grouped.isna().any().any():
                raise ValueError(f"Missing all-prompt prediction for {model}, {questionnaire_name}")
            all_prompt_predictions[questionnaire_name].append(all_grouped.to_numpy(dtype=float))

            run_matrices = []
            for run, selected in enumerate(selected_sets):
                subset = score_frame[score_frame["train_participant_ids"].isin(selected)]
                grouped = (
                    subset.groupby("val_participant_id", sort=True)[item_columns]
                    .mean()
                    .reindex(participants)
                )
                counts = subset.groupby("val_participant_id").size().reindex(participants, fill_value=0)
                if grouped.isna().any().any() or (counts == 0).any():
                    raise ValueError(
                        f"Missing pipeline prediction for {model}, {questionnaire_name}, run {run}"
                    )
                run_matrices.append(grouped.to_numpy(dtype=float))
                count_rows.append(
                    {
                        "model": model,
                        "questionnaire": questionnaire_name,
                        "run": run,
                        "seed": RUN_SEEDS[run],
                        "minimum_available_prompts_per_participant": int(counts.min()),
                        "maximum_available_prompts_per_participant": int(counts.max()),
                        "mean_available_prompts_per_participant": float(counts.mean()),
                    }
                )
            pipeline_predictions[questionnaire_name].append(np.stack(run_matrices, axis=0))

    pipeline_predictions = {
        questionnaire: np.stack(model_matrices, axis=0)
        for questionnaire, model_matrices in pipeline_predictions.items()
    }
    all_prompt_predictions = {
        questionnaire: np.stack(model_matrices, axis=0)[:, None, :, :]
        for questionnaire, model_matrices in all_prompt_predictions.items()
    }
    return pipeline_predictions, all_prompt_predictions, pd.DataFrame(count_rows)


def safe_corr(x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
    x_centered = x - np.mean(x, axis=axis, keepdims=True)
    y_centered = y - np.mean(y, axis=axis, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=axis)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=axis) * np.sum(y_centered**2, axis=axis)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > EPS,
    )


def partial_item_correlations(
    predicted_ranks: np.ndarray,
    true_ranks: np.ndarray,
    severity_ranks: np.ndarray,
) -> np.ndarray:
    """Return partial correlations with shape model x run x item."""
    y = true_ranks[None, None, :, :]
    c = severity_ranks[None, None, :, :]
    r_xy = safe_corr(predicted_ranks, y, axis=2)
    r_xc = safe_corr(predicted_ranks, c, axis=2)
    r_yc = safe_corr(y, c, axis=2)
    denominator = np.sqrt(np.maximum((1.0 - r_xc**2) * (1.0 - r_yc**2), 0.0))
    numerator = r_xy - r_xc * r_yc
    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > EPS,
    )
    return np.clip(result, -0.999999, 0.999999)


def aggregate_item_correlations(item_correlations: np.ndarray):
    z = np.arctanh(np.clip(item_correlations, -0.999999, 0.999999))
    model_run = np.tanh(np.mean(z, axis=2))
    model = np.tanh(np.mean(z, axis=(1, 2)))
    omnibus = float(np.tanh(np.mean(z)))
    return model_run, model, omnibus


def double_center(matrix: np.ndarray) -> np.ndarray:
    return (
        matrix
        - np.mean(matrix, axis=-1, keepdims=True)
        - np.mean(matrix, axis=-2, keepdims=True)
        + np.mean(matrix, axis=(-2, -1), keepdims=True)
    )


def profile_correlations(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return two-way residualized profile correlations as model x run."""
    pred_residual = double_center(predicted)
    true_residual = double_center(truth[None, None, :, :])
    pred_flat = pred_residual.reshape(predicted.shape[0], predicted.shape[1], -1)
    true_flat = true_residual.reshape(1, 1, -1)
    return np.clip(safe_corr(pred_flat, true_flat, axis=2), -0.999999, 0.999999)


def project_profiles_within_total_strata(
    matrix: np.ndarray, total_scores: np.ndarray
) -> np.ndarray:
    """Remove participant means, then exact-total-stratum x item means."""
    residual = matrix - np.mean(matrix, axis=-1, keepdims=True)
    return center_within_strata(residual, total_scores, axis=residual.ndim - 2)


def exact_total_profile_correlations(
    predicted: np.ndarray, truth: np.ndarray, total_scores: np.ndarray
) -> np.ndarray:
    """Return exact-total-stratified profile correlations as model x run."""
    pred_residual = project_profiles_within_total_strata(predicted, total_scores)
    true_residual = project_profiles_within_total_strata(
        truth[None, None, :, :], total_scores
    )
    pred_flat = pred_residual.reshape(predicted.shape[0], predicted.shape[1], -1)
    true_flat = true_residual.reshape(1, 1, -1)
    return np.clip(safe_corr(pred_flat, true_flat, axis=2), -0.999999, 0.999999)


def aggregate_profile_correlations(run_correlations: np.ndarray):
    z = np.arctanh(np.clip(run_correlations, -0.999999, 0.999999))
    model = np.tanh(np.mean(z, axis=1))
    omnibus = float(np.tanh(np.mean(z)))
    return model, omnibus


def batched_exact_total_profile_statistics(
    predicted: np.ndarray,
    truth: np.ndarray,
    total_scores: np.ndarray,
    indices: np.ndarray,
    mode: str,
    batch_size: int,
) -> np.ndarray:
    """Return strict profile statistics as resample x (models + omnibus)."""
    outputs = []
    for start in range(0, len(indices), batch_size):
        index_batch = indices[start : start + batch_size]
        batch_count = len(index_batch)
        x = predicted[:, :, index_batch, :]
        if mode == "bootstrap":
            y = truth[index_batch, :]
        elif mode == "permutation":
            y = np.broadcast_to(truth[None, :, :], (batch_count,) + truth.shape)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        y = y[None, None, :, :, :].copy()
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
        for value in np.unique(total_scores):
            positions = np.flatnonzero(total_scores == value)
            x[:, :, :, positions, :] = x[:, :, :, positions, :] - np.mean(
                x[:, :, :, positions, :], axis=3, keepdims=True
            )
            y[:, :, :, positions, :] = y[:, :, :, positions, :] - np.mean(
                y[:, :, :, positions, :], axis=3, keepdims=True
            )

        x_flat = x.reshape(x.shape[0], x.shape[1], batch_count, -1)
        y_flat = y.reshape(1, 1, batch_count, -1)
        run_r = safe_corr(x_flat, y_flat, axis=3)
        z = np.arctanh(np.clip(run_r, -0.999999, 0.999999))
        model = np.tanh(np.mean(z, axis=1)).T
        omnibus = np.tanh(np.mean(z, axis=(0, 1)))[:, None]
        outputs.append(np.concatenate([model, omnibus], axis=1))
    return np.concatenate(outputs, axis=0)


def prepare_rank_arrays(predicted: np.ndarray, truth: np.ndarray):
    predicted_ranks = rankdata(predicted, method="average", axis=2)
    true_ranks = rankdata(truth, method="average", axis=0)
    leave_one_out = truth.sum(axis=1, keepdims=True) - truth
    severity_ranks = rankdata(leave_one_out, method="average", axis=0)
    return predicted_ranks, true_ranks, severity_ranks


def center_within_strata(matrix: np.ndarray, strata: np.ndarray, axis: int) -> np.ndarray:
    """Remove exact stratum means along a participant axis."""
    residual = matrix.copy()
    for value in np.unique(strata):
        positions = np.flatnonzero(strata == value)
        selection = [slice(None)] * residual.ndim
        selection[axis] = positions
        selection = tuple(selection)
        residual[selection] = residual[selection] - np.mean(
            residual[selection], axis=axis, keepdims=True
        )
    return residual


def stratified_bootstrap_indices(
    strata: np.ndarray, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    indices = np.tile(np.arange(len(strata), dtype=np.int16), (replicates, 1))
    for value in np.unique(strata):
        positions = np.flatnonzero(strata == value)
        draws = rng.choice(positions, size=(replicates, len(positions)), replace=True)
        indices[:, positions] = draws
    return indices


def stratified_permutation_indices(
    strata: np.ndarray, replicates: int, rng: np.random.Generator
) -> np.ndarray:
    indices = np.tile(np.arange(len(strata), dtype=np.int16), (replicates, 1))
    for replicate in range(replicates):
        for value in np.unique(strata):
            positions = np.flatnonzero(strata == value)
            indices[replicate, positions] = rng.permutation(positions)
    return indices


def batched_partial_statistics(
    predicted_ranks: np.ndarray,
    true_ranks: np.ndarray,
    severity_ranks: np.ndarray,
    indices: np.ndarray,
    mode: str,
    batch_size: int,
) -> np.ndarray:
    """Return resample x (models + omnibus) statistics."""
    outputs = []
    for start in range(0, len(indices), batch_size):
        index_batch = indices[start : start + batch_size]
        batch_count = len(index_batch)
        x = predicted_ranks[:, :, index_batch, :]
        if mode == "bootstrap":
            y = true_ranks[index_batch, :]
            c = severity_ranks[index_batch, :]
        elif mode == "permutation":
            y = np.broadcast_to(
                true_ranks[None, :, :], (batch_count,) + true_ranks.shape
            )
            c = np.broadcast_to(
                severity_ranks[None, :, :], (batch_count,) + severity_ranks.shape
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        y = y[None, None, :, :, :]
        c = c[None, None, :, :, :]
        r_xy = safe_corr(x, y, axis=3)
        r_xc = safe_corr(x, c, axis=3)
        r_yc = safe_corr(y, c, axis=3)
        denominator = np.sqrt(np.maximum((1.0 - r_xc**2) * (1.0 - r_yc**2), 0.0))
        partial = np.divide(
            r_xy - r_xc * r_yc,
            denominator,
            out=np.zeros_like(r_xy, dtype=float),
            where=denominator > EPS,
        )
        z = np.arctanh(np.clip(partial, -0.999999, 0.999999))
        model = np.tanh(np.mean(z, axis=(1, 3))).T
        omnibus = np.tanh(np.mean(z, axis=(0, 1, 3)))[:, None]
        outputs.append(np.concatenate([model, omnibus], axis=1))
    return np.concatenate(outputs, axis=0)


def batched_profile_statistics(
    predicted: np.ndarray,
    truth: np.ndarray,
    indices: np.ndarray,
    mode: str,
    batch_size: int,
) -> np.ndarray:
    """Return resample x (models + omnibus) profile statistics."""
    outputs = []
    for start in range(0, len(indices), batch_size):
        index_batch = indices[start : start + batch_size]
        batch_count = len(index_batch)
        x = predicted[:, :, index_batch, :]
        if mode == "bootstrap":
            y = truth[index_batch, :]
        elif mode == "permutation":
            y = np.broadcast_to(truth[None, :, :], (batch_count,) + truth.shape)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        x_residual = double_center(x)
        y_residual = double_center(y)[None, None, :, :, :]
        x_flat = x_residual.reshape(x.shape[0], x.shape[1], batch_count, -1)
        y_flat = y_residual.reshape(1, 1, batch_count, -1)
        run_r = safe_corr(x_flat, y_flat, axis=3)
        z = np.arctanh(np.clip(run_r, -0.999999, 0.999999))
        model = np.tanh(np.mean(z, axis=1)).T
        omnibus = np.tanh(np.mean(z, axis=(0, 1)))[:, None]
        outputs.append(np.concatenate([model, omnibus], axis=1))
    return np.concatenate(outputs, axis=0)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for position, original_index in enumerate(order):
        candidate = min(1.0, (total - position) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted


def p_value_from_permutations(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (len(null) + 1))


def summarize_inference(
    questionnaire: str,
    metric: str,
    observed_models: np.ndarray,
    observed_omnibus: float,
    bootstrap: np.ndarray,
    permutations: np.ndarray,
) -> list[dict]:
    rows = []
    names = list(MODEL_SPECS) + ["All four models"]
    observed = np.concatenate([observed_models, [observed_omnibus]])
    for index, name in enumerate(names):
        rows.append(
            {
                "metric": metric,
                "questionnaire": questionnaire,
                "level": "omnibus" if index == len(names) - 1 else "model",
                "model": name,
                "estimate_r": observed[index],
                "ci_95_low": np.quantile(bootstrap[:, index], 0.025),
                "ci_95_high": np.quantile(bootstrap[:, index], 0.975),
                "permutation_p_two_sided": p_value_from_permutations(
                    observed[index], permutations[:, index]
                ),
            }
        )
    return rows


def item_accuracy_outputs(predictions: dict[str, np.ndarray], truth: dict[str, np.ndarray]):
    run_rows = []
    summary_rows = []
    for questionnaire, predicted in predictions.items():
        rounded = np.rint(predicted).astype(int)
        accuracy = np.mean(rounded == truth[questionnaire][None, None, :, :], axis=2)
        for model_index, model in enumerate(MODEL_SPECS):
            for run in range(N_RUNS):
                for item in range(accuracy.shape[-1]):
                    run_rows.append(
                        {
                            "model": model,
                            "questionnaire": questionnaire,
                            "run": run,
                            "seed": RUN_SEEDS[run],
                            "item": item + 1,
                            "exact_accuracy_percent": 100.0 * accuracy[model_index, run, item],
                        }
                    )
            item_means = accuracy[model_index].mean(axis=0) * 100.0
            item_sds = accuracy[model_index].std(axis=0, ddof=1) * 100.0
            for item in range(accuracy.shape[-1]):
                summary_rows.append(
                    {
                        "model": model,
                        "questionnaire": questionnaire,
                        "item": item + 1,
                        "mean_exact_accuracy_percent": item_means[item],
                        "sd_across_runs_percent": item_sds[item],
                        "minimum_run_percent": 100.0 * accuracy[model_index, :, item].min(),
                        "maximum_run_percent": 100.0 * accuracy[model_index, :, item].max(),
                        "model_mean_across_items_percent": 100.0
                        * accuracy[model_index].mean(),
                        "manuscript_reported_model_mean_percent": QUESTIONNAIRES[questionnaire][
                            "manuscript_model_means"
                        ][model],
                    }
                )
    return pd.DataFrame(run_rows), pd.DataFrame(summary_rows)


def observed_statistic_outputs(
    predictions: dict[str, np.ndarray], truth: dict[str, np.ndarray]
):
    run_rows = []
    item_rows = []
    global_rows = []
    observed_cache = {}

    for questionnaire, predicted in predictions.items():
        true = truth[questionnaire]
        pred_rank, true_rank, severity_rank = prepare_rank_arrays(predicted, true)
        item_partial = partial_item_correlations(pred_rank, true_rank, severity_rank)
        partial_run, partial_model, partial_omnibus = aggregate_item_correlations(item_partial)
        total_scores = true.sum(axis=1)
        strict_profile_run = exact_total_profile_correlations(
            np.rint(predicted), true, total_scores
        )
        strict_profile_model, strict_profile_omnibus = aggregate_profile_correlations(
            strict_profile_run
        )
        profile_run = profile_correlations(predicted, true)
        profile_model, profile_omnibus = aggregate_profile_correlations(profile_run)
        observed_cache[questionnaire] = {
            "pred_rank": pred_rank,
            "true_rank": true_rank,
            "severity_rank": severity_rank,
            "total_scores": total_scores,
            "partial_model": partial_model,
            "partial_omnibus": partial_omnibus,
            "strict_profile_model": strict_profile_model,
            "strict_profile_omnibus": strict_profile_omnibus,
            "profile_model": profile_model,
            "profile_omnibus": profile_omnibus,
        }

        for model_index, model in enumerate(MODEL_SPECS):
            for run in range(predicted.shape[1]):
                predicted_total = predicted[model_index, run].sum(axis=1)
                true_total = true.sum(axis=1)
                total_spearman = safe_corr(
                    rankdata(predicted_total)[None, :],
                    rankdata(true_total)[None, :],
                    axis=1,
                )[0]
                run_rows.extend(
                    [
                        {
                            "model": model,
                            "questionnaire": questionnaire,
                            "run": run,
                            "metric": "severity_adjusted_partial_spearman",
                            "estimate_r": partial_run[model_index, run],
                        },
                        {
                            "model": model,
                            "questionnaire": questionnaire,
                            "run": run,
                            "metric": "exact_total_stratum_profile_correlation",
                            "estimate_r": strict_profile_run[model_index, run],
                        },
                        {
                            "model": model,
                            "questionnaire": questionnaire,
                            "run": run,
                            "metric": "double_centered_profile_correlation",
                            "estimate_r": profile_run[model_index, run],
                        },
                    ]
                )
                global_rows.append(
                    {
                        "model": model,
                        "questionnaire": questionnaire,
                        "run": run,
                        "predicted_vs_true_total_spearman_r": total_spearman,
                    }
                )
            for item in range(predicted.shape[-1]):
                per_run = item_partial[model_index, :, item]
                item_rows.append(
                    {
                        "model": model,
                        "questionnaire": questionnaire,
                        "item": item + 1,
                        "severity_adjusted_partial_spearman_r": np.tanh(
                            np.mean(np.arctanh(np.clip(per_run, -0.999999, 0.999999)))
                        ),
                        "minimum_run_r": per_run.min(),
                        "maximum_run_r": per_run.max(),
                    }
                )

    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(item_rows),
        pd.DataFrame(global_rows),
        observed_cache,
    )


def all_prompt_robustness_outputs(
    predictions: dict[str, np.ndarray], truth: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for questionnaire, predicted in predictions.items():
        true = truth[questionnaire]
        pred_rank, true_rank, severity_rank = prepare_rank_arrays(predicted, true)
        item_partial = partial_item_correlations(pred_rank, true_rank, severity_rank)
        _, partial_model, partial_omnibus = aggregate_item_correlations(item_partial)
        strict_profile_run = exact_total_profile_correlations(
            predicted, true, true.sum(axis=1)
        )
        strict_profile_model, strict_profile_omnibus = aggregate_profile_correlations(
            strict_profile_run
        )
        profile_run = profile_correlations(predicted, true)
        profile_model, profile_omnibus = aggregate_profile_correlations(profile_run)
        for model_index, model in enumerate(MODEL_SPECS):
            rows.extend(
                [
                    {
                        "model": model,
                        "questionnaire": questionnaire,
                        "metric": "severity_adjusted_partial_spearman",
                        "estimate_r": partial_model[model_index],
                    },
                    {
                        "model": model,
                        "questionnaire": questionnaire,
                        "metric": "exact_total_stratum_profile_correlation",
                        "estimate_r": strict_profile_model[model_index],
                    },
                    {
                        "model": model,
                        "questionnaire": questionnaire,
                        "metric": "double_centered_profile_correlation",
                        "estimate_r": profile_model[model_index],
                    },
                ]
            )
        rows.extend(
            [
                {
                    "model": "All four models",
                    "questionnaire": questionnaire,
                    "metric": "severity_adjusted_partial_spearman",
                    "estimate_r": partial_omnibus,
                },
                {
                    "model": "All four models",
                    "questionnaire": questionnaire,
                    "metric": "exact_total_stratum_profile_correlation",
                    "estimate_r": strict_profile_omnibus,
                },
                {
                    "model": "All four models",
                    "questionnaire": questionnaire,
                    "metric": "double_centered_profile_correlation",
                    "estimate_r": profile_omnibus,
                },
            ]
        )
    return pd.DataFrame(rows)


def apply_multiplicity(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["holm_family"] = ""
    results["holm_adjusted_p"] = np.nan
    families = [
        (
            "primary_omnibus_2_questionnaires",
            (results["metric"] == "severity_adjusted_partial_spearman")
            & (results["level"] == "omnibus"),
        ),
        (
            "primary_model_specific_8_tests",
            (results["metric"] == "severity_adjusted_partial_spearman")
            & (results["level"] == "model"),
        ),
        (
            "strict_sensitivity_omnibus_2_questionnaires",
            (results["metric"] == "exact_total_stratum_profile_correlation")
            & (results["level"] == "omnibus"),
        ),
        (
            "strict_sensitivity_model_specific_8_tests",
            (results["metric"] == "exact_total_stratum_profile_correlation")
            & (results["level"] == "model"),
        ),
        (
            "secondary_omnibus_2_questionnaires",
            (results["metric"] == "double_centered_profile_correlation")
            & (results["level"] == "omnibus"),
        ),
        (
            "secondary_model_specific_8_tests",
            (results["metric"] == "double_centered_profile_correlation")
            & (results["level"] == "model"),
        ),
    ]
    for family, mask in families:
        indices = results.index[mask]
        results.loc[indices, "holm_family"] = family
        results.loc[indices, "holm_adjusted_p"] = holm_adjust(
            results.loc[indices, "permutation_p_two_sided"].to_numpy()
        )
    return results


def write_selected_sets(output: Path, selected_sets: list[list[str]]) -> None:
    rows = []
    for run, selected in enumerate(selected_sets):
        for position, prompt_set in enumerate(selected, start=1):
            rows.append(
                {
                    "run": run,
                    "seed": RUN_SEEDS[run],
                    "position": position,
                    "train_participant_ids": prompt_set,
                }
            )
    pd.DataFrame(rows).to_csv(output / "selected_prompt_sets.csv", index=False)


def plot_primary_results(results: pd.DataFrame, output: Path) -> None:
    primary = results[results["metric"] == "severity_adjusted_partial_spearman"].copy()
    labels = list(MODEL_SPECS) + ["All four models"]
    colors = {"GAD-7": "#2563eb", "PHQ-9": "#dc2626"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True, sharey=True)
    for axis, questionnaire in zip(axes, QUESTIONNAIRES):
        subset = primary[primary["questionnaire"] == questionnaire].set_index("model").loc[labels]
        y = np.arange(len(labels))
        axis.errorbar(
            subset["estimate_r"],
            y,
            xerr=np.vstack(
                [
                    subset["estimate_r"] - subset["ci_95_low"],
                    subset["ci_95_high"] - subset["estimate_r"],
                ]
            ),
            fmt="o",
            color=colors[questionnaire],
            capsize=3,
            linewidth=1.5,
        )
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(questionnaire)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel("Severity-adjusted partial Spearman r")
    fig.suptitle("Item-specific correspondence beyond leave-one-item-out severity")
    fig.tight_layout()
    fig.savefig(output / "severity_adjusted_item_correspondence.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "severity_adjusted_item_correspondence.pdf", bbox_inches="tight")
    plt.close(fig)


def format_p(value: float) -> str:
    return f"{value:.4f}" if value >= 0.0001 else "<0.0001"


def write_report(
    output: Path,
    results: pd.DataFrame,
    accuracy_summary: pd.DataFrame,
    run_results: pd.DataFrame,
    global_results: pd.DataFrame,
    stratum_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    n_participants: int,
    n_bootstrap: int,
    n_permutations: int,
) -> None:
    informative_counts = (
        stratum_summary[stratum_summary["contributes_within_stratum_contrast"]]
        .groupby("questionnaire")["participants"]
        .sum()
        .to_dict()
    )
    lines = [
        "# Run-preserving item-analysis results",
        "",
        "This report is an exploratory decision aid. It was generated from existing",
        "item-level outputs only; no LLM calls were made.",
        "",
        "## Analysis population and reconstruction",
        "",
        f"- Independent participants: {n_participants}.",
        "- Models: GPT-4o, Gemini-1.5pro, Llama-3.2-90b, and MedLM-Large-1.5.",
        "- Ten five-prompt ensembles per model, using the same seeds and sampling",
        "  procedure as the manuscript notebook.",
        f"- Participant-cluster bootstrap replicates: {n_bootstrap:,}.",
        f"- Joint participant-label permutations: {n_permutations:,}.",
        "",
        "## Global-severity channel",
        "",
        "Predicted questionnaire totals strongly track ground-truth totals. These",
        "associations show that global severity is an",
        "important channel, but do not determine whether it is the only channel.",
        "",
        "| Questionnaire | Model | Mean Spearman rho | Run range |",
        "|---|---|---:|---:|",
    ]
    global_grouped = global_results.groupby(["questionnaire", "model"])[
        "predicted_vs_true_total_spearman_r"
    ].agg(["mean", "min", "max"])
    for questionnaire in QUESTIONNAIRES:
        for model in MODEL_SPECS:
            row = global_grouped.loc[(questionnaire, model)]
            lines.append(
                f"| {questionnaire} | {model} | {row['mean']:.3f} | "
                f"{row['min']:.3f}--{row['max']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Primary result: severity-adjusted matching-item correspondence",
            "",
            "The effect is the Fisher-averaged partial Spearman correlation between each",
            "predicted item and its matching true item after controlling for the true sum",
            "of all other questionnaire items.",
            "",
            "| Questionnaire | Model | r | 95% CI | Permutation p | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    primary = results[results["metric"] == "severity_adjusted_partial_spearman"]
    for questionnaire in QUESTIONNAIRES:
        for model in list(MODEL_SPECS) + ["All four models"]:
            row = primary[
                (primary["questionnaire"] == questionnaire) & (primary["model"] == model)
            ].iloc[0]
            lines.append(
                f"| {questionnaire} | {model} | {row.estimate_r:.3f} | "
                f"[{row.ci_95_low:.3f}, {row.ci_95_high:.3f}] | "
                f"{format_p(row.permutation_p_two_sided)} | "
                f"{format_p(row.holm_adjusted_p)} |"
            )

    lines.extend(
        [
            "",
            "## Stricter sensitivity: correspondence within identical total scores",
            "",
            "This post-plan sensitivity analysis removes participant mean severity and",
            "exact full-total-score-stratum x item means, then permutes entire predicted",
            "profiles only within identical ground-truth total-score strata. It is the",
            "analysis most directly targeted at a global-severity-only mechanism.",
            "",
            f"The exact-score stratification retained {informative_counts['GAD-7']}/106 GAD-7 participants and",
            f"{informative_counts['PHQ-9']}/106 PHQ-9 participants in strata containing at least two participants;",
            "singleton strata contribute zero within-stratum contrast.",
            "",
            "| Questionnaire | Model | r | 95% CI | Stratified permutation p | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    strict = results[
        results["metric"] == "exact_total_stratum_profile_correlation"
    ]
    for questionnaire in QUESTIONNAIRES:
        for model in list(MODEL_SPECS) + ["All four models"]:
            row = strict[
                (strict["questionnaire"] == questionnaire) & (strict["model"] == model)
            ].iloc[0]
            lines.append(
                f"| {questionnaire} | {model} | {row.estimate_r:.3f} | "
                f"[{row.ci_95_low:.3f}, {row.ci_95_high:.3f}] | "
                f"{format_p(row.permutation_p_two_sided)} | "
                f"{format_p(row.holm_adjusted_p)} |"
            )

    lines.extend(
        [
            "",
            "## Secondary result: within-person symptom-profile concordance",
            "",
            "This correlation uses two-way centered participant-by-item matrices,",
            "removing participant mean severity and item mean prevalence.",
            "",
            "| Questionnaire | Model | r | 95% CI | Permutation p | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    secondary = results[results["metric"] == "double_centered_profile_correlation"]
    for questionnaire in QUESTIONNAIRES:
        for model in list(MODEL_SPECS) + ["All four models"]:
            row = secondary[
                (secondary["questionnaire"] == questionnaire) & (secondary["model"] == model)
            ].iloc[0]
            lines.append(
                f"| {questionnaire} | {model} | {row.estimate_r:.3f} | "
                f"[{row.ci_95_low:.3f}, {row.ci_95_high:.3f}] | "
                f"{format_p(row.permutation_p_two_sided)} | "
                f"{format_p(row.holm_adjusted_p)} |"
            )

    lines.extend(["", "## Pipeline-run stability", ""])
    for metric, label in [
        ("severity_adjusted_partial_spearman", "Primary partial correlation"),
        (
            "exact_total_stratum_profile_correlation",
            "Strict exact-total-stratum sensitivity correlation",
        ),
        ("double_centered_profile_correlation", "Secondary profile correlation"),
    ]:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Questionnaire | Model | Minimum run r | Maximum run r |")
        lines.append("|---|---|---:|---:|")
        subset = run_results[run_results["metric"] == metric]
        grouped = subset.groupby(["questionnaire", "model"])["estimate_r"].agg(["min", "max"])
        for questionnaire in QUESTIONNAIRES:
            for model in MODEL_SPECS:
                row = grouped.loc[(questionnaire, model)]
                lines.append(
                    f"| {questionnaire} | {model} | {row['min']:.3f} | {row['max']:.3f} |"
                )
        lines.append("")

    lines.extend(
        [
            "## Reproduction check",
            "",
            "The reconstructed rounded five-prompt ensembles reproduce the manuscript's",
            "model-wise item-accuracy means as follows:",
            "",
            "| Questionnaire | Model | Recomputed % | Manuscript % | Difference (pp) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    accuracy_models = (
        accuracy_summary.groupby(["questionnaire", "model"], as_index=False)
        .first()[
            [
                "questionnaire",
                "model",
                "model_mean_across_items_percent",
                "manuscript_reported_model_mean_percent",
            ]
        ]
    )
    for questionnaire in QUESTIONNAIRES:
        for model in MODEL_SPECS:
            row = accuracy_models[
                (accuracy_models["questionnaire"] == questionnaire)
                & (accuracy_models["model"] == model)
            ].iloc[0]
            recomputed = row.model_mean_across_items_percent
            manuscript = row.manuscript_reported_model_mean_percent
            lines.append(
                f"| {questionnaire} | {model} | {recomputed:.2f} | {manuscript:.1f} | "
                f"{recomputed - manuscript:+.2f} |"
            )

    lines.extend(
        [
            "",
            "## All-prompt robustness point estimates",
            "",
            "These use each participant's mean across every available prompt configuration",
            "and have no additional inferential status.",
            "",
            "| Questionnaire | Model | Metric | r |",
            "|---|---|---|---:|",
        ]
    )
    for questionnaire in QUESTIONNAIRES:
        for model in list(MODEL_SPECS) + ["All four models"]:
            for metric in [
                "severity_adjusted_partial_spearman",
                "exact_total_stratum_profile_correlation",
                "double_centered_profile_correlation",
            ]:
                row = robustness[
                    (robustness["questionnaire"] == questionnaire)
                    & (robustness["model"] == model)
                    & (robustness["metric"] == metric)
                ].iloc[0]
                lines.append(f"| {questionnaire} | {model} | {metric} | {row.estimate_r:.3f} |")

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A positive severity-adjusted association would be evidence that matching item",
            "outputs retain some item-specific predictive correspondence beyond measured",
            "overall severity. It would not prove that the item outputs are independently",
            "validated measurements, that model rationales are faithful, or that the model",
            "used item-level evidence internally. Those limitations remain even when the",
            "association is statistically distinguishable from zero.",
            "",
        ]
    )
    (output / "statistical_results.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    (
        participants,
        truth,
        model_frames,
        model_arrays,
        selected_sets,
        audit,
        input_manifest,
    ) = load_data(data_root)
    predictions, all_prompt_predictions, prompt_counts = build_prediction_matrices(
        participants, model_frames, model_arrays, selected_sets
    )

    write_selected_sets(output, selected_sets)
    audit.to_csv(output / "data_audit.csv", index=False)
    prompt_counts.to_csv(output / "pipeline_prompt_counts.csv", index=False)
    stratum_rows = []
    for questionnaire, matrix in truth.items():
        totals, counts = np.unique(matrix.sum(axis=1).astype(int), return_counts=True)
        for total, count in zip(totals, counts):
            stratum_rows.append(
                {
                    "questionnaire": questionnaire,
                    "total_score": int(total),
                    "participants": int(count),
                    "contributes_within_stratum_contrast": bool(count >= 2),
                }
            )
    stratum_summary = pd.DataFrame(stratum_rows)
    stratum_summary.to_csv(output / "total_score_strata.csv", index=False)

    accuracy_run, accuracy_summary = item_accuracy_outputs(predictions, truth)
    accuracy_run.to_csv(output / "item_accuracy_by_run.csv", index=False)
    accuracy_summary.to_csv(output / "item_accuracy_summary.csv", index=False)

    run_results, item_results, global_results, observed_cache = observed_statistic_outputs(
        predictions, truth
    )
    run_results.to_csv(output / "run_level_item_specificity.csv", index=False)
    item_results.to_csv(output / "item_level_partial_correlations.csv", index=False)
    global_results.to_csv(output / "global_total_score_associations.csv", index=False)

    rng_bootstrap = np.random.default_rng(args.seed)
    rng_permutation = np.random.default_rng(args.seed + 1)
    n = len(participants)
    bootstrap_indices = rng_bootstrap.integers(0, n, size=(args.bootstrap, n), dtype=np.int16)
    permutation_indices = np.stack(
        [rng_permutation.permutation(n) for _ in range(args.permutations)], axis=0
    ).astype(np.int16)

    inference_rows = []
    distribution_archive = {}
    for questionnaire_index, questionnaire in enumerate(QUESTIONNAIRES):
        true = truth[questionnaire]
        predicted = predictions[questionnaire]
        cached = observed_cache[questionnaire]

        partial_bootstrap = batched_partial_statistics(
            cached["pred_rank"],
            cached["true_rank"],
            cached["severity_rank"],
            bootstrap_indices,
            "bootstrap",
            args.batch_size,
        )
        partial_permutations = batched_partial_statistics(
            cached["pred_rank"],
            cached["true_rank"],
            cached["severity_rank"],
            permutation_indices,
            "permutation",
            args.batch_size,
        )
        inference_rows.extend(
            summarize_inference(
                questionnaire,
                "severity_adjusted_partial_spearman",
                cached["partial_model"],
                cached["partial_omnibus"],
                partial_bootstrap,
                partial_permutations,
            )
        )

        strict_bootstrap_indices = stratified_bootstrap_indices(
            cached["total_scores"],
            args.bootstrap,
            np.random.default_rng(args.seed + 100 + questionnaire_index),
        )
        strict_permutation_indices = stratified_permutation_indices(
            cached["total_scores"],
            args.permutations,
            np.random.default_rng(args.seed + 200 + questionnaire_index),
        )
        strict_bootstrap = batched_exact_total_profile_statistics(
            np.rint(predicted),
            true,
            cached["total_scores"],
            strict_bootstrap_indices,
            "bootstrap",
            args.batch_size,
        )
        strict_permutations = batched_exact_total_profile_statistics(
            np.rint(predicted),
            true,
            cached["total_scores"],
            strict_permutation_indices,
            "permutation",
            args.batch_size,
        )
        inference_rows.extend(
            summarize_inference(
                questionnaire,
                "exact_total_stratum_profile_correlation",
                cached["strict_profile_model"],
                cached["strict_profile_omnibus"],
                strict_bootstrap,
                strict_permutations,
            )
        )

        profile_bootstrap = batched_profile_statistics(
            predicted,
            true,
            bootstrap_indices,
            "bootstrap",
            args.batch_size,
        )
        profile_permutations = batched_profile_statistics(
            predicted,
            true,
            permutation_indices,
            "permutation",
            args.batch_size,
        )
        inference_rows.extend(
            summarize_inference(
                questionnaire,
                "double_centered_profile_correlation",
                cached["profile_model"],
                cached["profile_omnibus"],
                profile_bootstrap,
                profile_permutations,
            )
        )

        key = questionnaire.lower().replace("-", "")
        distribution_archive[f"{key}_partial_bootstrap"] = partial_bootstrap
        distribution_archive[f"{key}_partial_permutations"] = partial_permutations
        distribution_archive[f"{key}_strict_stratum_bootstrap"] = strict_bootstrap
        distribution_archive[f"{key}_strict_stratum_permutations"] = strict_permutations
        distribution_archive[f"{key}_profile_bootstrap"] = profile_bootstrap
        distribution_archive[f"{key}_profile_permutations"] = profile_permutations

    results = apply_multiplicity(pd.DataFrame(inference_rows))
    results.to_csv(output / "inference_results.csv", index=False)
    np.savez_compressed(output / "inference_distributions.npz", **distribution_archive)

    robustness = all_prompt_robustness_outputs(all_prompt_predictions, truth)
    robustness.to_csv(output / "all_prompt_robustness.csv", index=False)

    plot_primary_results(results, output)
    write_report(
        output,
        results,
        accuracy_summary,
        run_results,
        global_results,
        stratum_summary,
        robustness,
        len(participants),
        args.bootstrap,
        args.permutations,
    )

    manifest = {
        "created": "2026-08-05",
        "script": str(Path(__file__).resolve()),
        "data_root": str(data_root),
        "output": str(output),
        "participants": len(participants),
        "participant_ids": participants,
        "models": list(MODEL_SPECS),
        "runs": N_RUNS,
        "prompts_per_run": N_PROMPTS,
        "run_seeds": RUN_SEEDS,
        "bootstrap_replicates": args.bootstrap,
        "permutations": args.permutations,
        "random_seed": args.seed,
        "inputs": input_manifest,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "polars": pl.__version__,
            "scipy": scipy.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Analysis complete. Results written to {output}")


if __name__ == "__main__":
    main()
