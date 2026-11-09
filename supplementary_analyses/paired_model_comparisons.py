#!/usr/bin/env python
"""Paired statistical comparisons of the four LLMs reported in the paper.

The primary analyses reproduce and compare the balanced-accuracy endpoints in
Table 1:

* Diagnosis (binary): exact Friedman omnibus tests (all rank configurations
  enumerated; the chi-square approximation is unreliable at five blocks) and
  exact paired sign-permutation post-hoc tests over the five matched
  neural-network aggregation runs.
* Severity (multiclass): participant-blocked randomization tests on mean
  balanced accuracy over the ten mean-aggregation runs. Each permutation keeps
  a participant's complete ten-run prediction vector together while exchanging
  model labels within that participant.

An ordinal sensitivity analysis uses Friedman omnibus tests and permutation
Wilcoxon signed-rank post-hoc tests on each participant's mean absolute
severity-level error. Pairwise p-values are Holm-adjusted within each six-pair
family; a second Holm adjustment across all 24 primary comparisons is also
reported. Primary omnibus p-values are Holm-adjusted across the four Table 1
panels.

The script reads frozen artifacts without modifying them and refuses to
overwrite an existing output directory. Only aggregate results are written.
Polars 1.19.0 is required because it reproduces the published seeded samples;
newer Polars releases may draw different samples for the same seed.

Usage:
    python supplementary_analyses/paired_model_comparisons.py [data_dir]
    python supplementary_analyses/paired_model_comparisons.py [data_dir] \
        --output-dir PATH
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy
from scipy import stats


MODELS = {
    "GPT-4o": "prompts_1_11",
    "Gemini-1.5-pro": "prompts_1_12",
    "Llama-3.2-90b": "prompts_1_13",
    "MedLM-Large-1.5": "prompts_1_17",
}
CONDITIONS = ("anxiety", "depression")
N_SAMPLES = 5
N_MEAN_RUNS = 10
N_NN_RUNS = 5
DEFAULT_RESAMPLES = 1_000_000
DEFAULT_SEED = 20260805
PERMUTATION_BATCH = 10_000

BIN_MAP = {
    "minimal": 0,
    "mild": 1,
    "moderate": 1,
    "moderately-severe": 1,
    "severe": 1,
}
MC_MAP = {
    "minimal": 0,
    "mild": 1,
    "moderate": 2,
    "moderately-severe": 3,
    "severe": 3,
}

TABLE1_DIAGNOSIS = {
    ("GPT-4o", "anxiety"): (86.2, 5.5),
    ("Gemini-1.5-pro", "anxiety"): (87.4, 3.3),
    ("Llama-3.2-90b", "anxiety"): (86.0, 5.4),
    ("MedLM-Large-1.5", "anxiety"): (87.7, 3.8),
    ("GPT-4o", "depression"): (85.4, 3.3),
    ("Gemini-1.5-pro", "depression"): (85.8, 2.0),
    ("Llama-3.2-90b", "depression"): (89.1, 6.5),
    ("MedLM-Large-1.5", "depression"): (88.2, 5.0),
}
TABLE1_SEVERITY = {
    ("GPT-4o", "anxiety"): (53.5, 3.2),
    ("Gemini-1.5-pro", "anxiety"): (45.3, 4.5),
    ("Llama-3.2-90b", "anxiety"): (58.2, 3.1),
    ("MedLM-Large-1.5", "anxiety"): (47.5, 3.1),
    ("GPT-4o", "depression"): (61.2, 4.0),
    ("Gemini-1.5-pro", "depression"): (59.1, 4.5),
    ("Llama-3.2-90b", "depression"): (55.1, 6.4),
    ("MedLM-Large-1.5", "depression"): (56.7, 8.9),
}

assert pl.__version__ == "1.19.0", (
    f"need polars 1.19.0 for the published seeded draws; got {pl.__version__}"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=Path.home() / "Downloads" / "mental_health",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=("new output directory (default: DATA_DIR/"
              "supplementary_analysis_outputs/paired_model_comparisons)"),
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
        help=f"Monte Carlo permutations (default: {DEFAULT_RESAMPLES:,})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    if args.output_dir is None:
        args.output_dir = (
            args.data_dir
            / "supplementary_analysis_outputs"
            / "paired_model_comparisons"
        )
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(prompts_dir):
    """Load the four columns needed to reconstruct the published runs."""
    prompt_id = prompts_dir.name.replace("prompts", "prompt")
    frame = pl.read_csv(prompts_dir / "results.csv")
    prediction_columns = {
        parts[1]: column
        for column in frame.columns
        if len(parts := column.split("\n")) == 3
        and parts[2] == prompt_id
        and parts[1] in CONDITIONS
    }
    assert set(prediction_columns) == set(CONDITIONS), (
        f"missing prediction columns in {prompts_dir / 'results.csv'}"
    )
    return frame.select(
        "val_participant_id",
        "train_participant_ids",
        "anxiety",
        "depression",
        anxiety_pd=prediction_columns["anxiety"],
        depression_pd=prediction_columns["depression"],
    )


def per_run_predictions(frame, mapping):
    """Reconstruct the ten mean-aggregation runs as participant x run arrays."""
    mapped = frame.with_columns(
        pl.col("anxiety").replace_strict(mapping),
        pl.col("depression").replace_strict(mapping),
        pl.col("anxiety_pd").replace_strict(mapping),
        pl.col("depression_pd").replace_strict(mapping),
    )
    train_ids = mapped["train_participant_ids"].unique().sort()
    runs = []
    for run in range(N_MEAN_RUNS):
        selected = train_ids.sample(N_SAMPLES, seed=100 * N_SAMPLES + run)
        aggregated = (
            mapped.filter(pl.col("train_participant_ids").is_in(selected))
            .group_by("val_participant_id")
            .agg(
                pl.col("anxiety").first(),
                pl.col("depression").first(),
                pl.col("anxiety_pd").mean().round().cast(pl.Int64),
                pl.col("depression_pd").mean().round().cast(pl.Int64),
            )
            .sort("val_participant_id")
        )
        runs.append(aggregated)

    participant_ids = runs[0]["val_participant_id"].to_numpy()
    ground_truth = {condition: runs[0][condition].to_numpy() for condition in CONDITIONS}
    predictions = {
        condition: np.column_stack(
            [run[f"{condition}_pd"].to_numpy() for run in runs]
        )
        for condition in CONDITIONS
    }
    return participant_ids, ground_truth, predictions


def balanced_accuracy_by_run(ground_truth, predictions):
    """Balanced accuracy for each prediction column."""
    classes = np.unique(ground_truth)
    return np.array(
        [
            np.mean(
                [
                    np.mean(predictions[ground_truth == label, run] == label)
                    for label in classes
                ]
            )
            for run in range(predictions.shape[1])
        ]
    )


def balanced_accuracy_contributions(ground_truth, predictions):
    """Participant contributions whose sum is mean balanced accuracy over runs."""
    classes, counts = np.unique(ground_truth, return_counts=True)
    count_by_class = dict(zip(classes, counts))
    weights = np.array(
        [1.0 / (len(classes) * count_by_class[label]) for label in ground_truth]
    )
    correct_rate = np.mean(predictions == ground_truth[:, None], axis=1)
    return weights * correct_rate


def holm(p_values):
    """Holm step-down adjusted p-values, returned in input order."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def exact_friedman(matrix):
    """Exact Friedman test by enumerating every within-block rank configuration.

    Returns (chi-square statistic, exact p). Requires no within-block ties and
    a small enough layout to enumerate (here 4!**5 = 7,962,624 configurations);
    the chi-square approximation is unreliable at five blocks."""
    n_blocks, n_treatments = matrix.shape
    assert all(len(np.unique(row)) == n_treatments for row in matrix), (
        "within-block ties present; exact no-tie null distribution is invalid"
    )
    assert math.factorial(n_treatments) ** n_blocks <= 20_000_000, (
        "layout too large to enumerate exactly"
    )
    rank_rows = tuple(itertools.permutations(range(1, n_treatments + 1)))
    rank_sum_counts = Counter({(0,) * n_treatments: 1})
    for _ in range(n_blocks):
        next_counts = Counter()
        for current_sums, count in rank_sum_counts.items():
            for rank_row in rank_rows:
                next_counts[
                    tuple(total + rank for total, rank in zip(current_sums, rank_row))
                ] += count
        rank_sum_counts = next_counts
    scale = 12.0 / (n_blocks * n_treatments * (n_treatments + 1))
    offset = 3.0 * n_blocks * (n_treatments + 1)
    observed_sums = np.apply_along_axis(stats.rankdata, 1, matrix).sum(axis=0)
    observed = scale * (observed_sums ** 2).sum() - offset
    extreme = sum(
        count
        for rank_sums, count in rank_sum_counts.items()
        if scale * sum(rank_sum**2 for rank_sum in rank_sums) - offset
        >= observed - 1e-9
    )
    total_configurations = math.factorial(n_treatments) ** n_blocks
    assert sum(rank_sum_counts.values()) == total_configurations
    return float(observed), float(extreme / total_configurations)


def exact_paired_sign_pvalue(first, second):
    """Two-sided exact sign-permutation p-value for a paired mean difference."""
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    observed = abs(differences.sum())
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(differences))))
    null_statistics = np.abs(signs @ differences)
    return float(np.mean(null_statistics >= observed - 1e-12))


def participant_blocked_tests(contributions, n_resamples, rng_omnibus, rng_pairwise):
    """Randomize model labels within participants for omnibus and pairwise tests."""
    values = np.asarray(contributions, dtype=float)
    n_participants, n_models = values.shape
    pairs = list(itertools.combinations(range(n_models), 2))

    observed_scores = values.sum(axis=0)
    observed_omnibus = np.sum((observed_scores - observed_scores.mean()) ** 2)
    model_permutations = np.array(
        list(itertools.permutations(range(n_models))), dtype=np.int8
    )
    omnibus_extreme = 0
    for start in range(0, n_resamples, PERMUTATION_BATCH):
        batch = min(PERMUTATION_BATCH, n_resamples - start)
        choices = rng_omnibus.integers(
            0, len(model_permutations), size=(batch, n_participants)
        )
        randomized_scores = np.empty((batch, n_models), dtype=float)
        participant_index = np.arange(n_participants)[None, :]
        for model in range(n_models):
            source_models = model_permutations[choices, model]
            randomized_scores[:, model] = values[
                participant_index, source_models
            ].sum(axis=1)
        null_statistics = np.sum(
            (randomized_scores - randomized_scores.mean(axis=1, keepdims=True)) ** 2,
            axis=1,
        )
        omnibus_extreme += np.count_nonzero(
            null_statistics >= observed_omnibus - 1e-15
        )
    omnibus_p = (omnibus_extreme + 1) / (n_resamples + 1)

    differences = np.column_stack(
        [values[:, first] - values[:, second] for first, second in pairs]
    )
    observed_pairwise = np.abs(differences.sum(axis=0))
    pairwise_extreme = np.zeros(len(pairs), dtype=np.int64)
    for start in range(0, n_resamples, PERMUTATION_BATCH):
        batch = min(PERMUTATION_BATCH, n_resamples - start)
        signs = rng_pairwise.integers(0, 2, size=(batch, n_participants), dtype=np.int8)
        signs = signs * 2 - 1
        null_statistics = np.abs(signs @ differences)
        pairwise_extreme += np.count_nonzero(
            null_statistics >= observed_pairwise[None, :] - 1e-15, axis=0
        )
    pairwise_p = (pairwise_extreme + 1) / (n_resamples + 1)
    return observed_scores, observed_omnibus, float(omnibus_p), pairwise_p


def permutation_wilcoxon_pvalues(values, pairs, n_resamples, rng):
    """Two-sided permutation Wilcoxon p-values with Pratt handling of zeros."""
    signed_ranks = []
    observed = []
    for first, second in pairs:
        differences = values[:, first] - values[:, second]
        ranks = stats.rankdata(np.abs(differences), method="average")
        ranked_differences = ranks * np.sign(differences)
        signed_ranks.append(np.abs(ranked_differences))
        observed.append(abs(ranked_differences.sum()))
    signed_ranks = np.column_stack(signed_ranks)
    observed = np.asarray(observed)

    extreme = np.zeros(len(pairs), dtype=np.int64)
    for start in range(0, n_resamples, PERMUTATION_BATCH):
        batch = min(PERMUTATION_BATCH, n_resamples - start)
        signs = rng.integers(0, 2, size=(batch, values.shape[0]), dtype=np.int8)
        signs = signs * 2 - 1
        null_statistics = np.abs(signs @ signed_ranks)
        extreme += np.count_nonzero(
            null_statistics >= observed[None, :] - 1e-12, axis=0
        )
    return (extreme + 1) / (n_resamples + 1)


def load_nn_run_scores(data_dir, names):
    """Reproduce Table 1a/b: best fold-mean NN variant within each run."""
    scores = {condition: {} for condition in CONDITIONS}
    reference_fold_signature = None
    reference_run_ids = {}
    for name in names:
        metrics_path = data_dir / MODELS[name] / "ml_algos_binary" / "metrics.pkl"
        with metrics_path.open("rb") as stream:
            records = pickle.load(stream)
        records = [
            record
            for record in records
            if record["n_samples"] == N_SAMPLES
            and record["algo"].startswith("NeuralNetFastAI")
        ]
        fold_signature = {
            (record["run"], record["val_set"], record["anx_dep"]): (
                record["train_example_count_0"],
                record["train_example_count_1"],
            )
            for record in records
        }
        if reference_fold_signature is None:
            reference_fold_signature = fold_signature
        else:
            assert fold_signature == reference_fold_signature, (
                f"neural-network run/fold composition differs for {name}"
            )
        for condition in CONDITIONS:
            by_run = {}
            for record in (r for r in records if r["anx_dep"] == condition):
                by_run.setdefault(record["run"], {}).setdefault(record["algo"], []).append(
                    record["val_balacc"]
                )
            run_ids = tuple(sorted(by_run))
            if condition not in reference_run_ids:
                reference_run_ids[condition] = run_ids
            else:
                assert run_ids == reference_run_ids[condition], (
                    f"neural-network run identifiers differ for {name}/{condition}"
                )
            run_scores = np.array(
                [
                    100 * max(np.mean(folds) for folds in variants.values())
                    for _, variants in sorted(by_run.items())
                ]
            )
            assert len(run_scores) == N_NN_RUNS, (
                f"expected {N_NN_RUNS} NN runs for {name}/{condition}; "
                f"found {len(run_scores)}"
            )
            scores[condition][name] = run_scores
    return scores


def format_p(value):
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.6f}"


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")

    required_files = []
    for directory in MODELS.values():
        required_files.extend(
            [
                data_dir / directory / "results.csv",
                data_dir / directory / "ml_algos_binary" / "metrics.pkl",
            ]
        )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise SystemExit("missing input files:\n" + "\n".join(missing))

    names = list(MODELS)
    pairs = list(itertools.combinations(range(len(names)), 2))
    pair_labels = [f"{names[first]}|{names[second]}" for first, second in pairs]
    data = {}
    reference_ids = None
    for name, directory in MODELS.items():
        frame = load_model(data_dir / directory)
        ids_binary, gt_binary, pred_binary = per_run_predictions(frame, BIN_MAP)
        ids_multiclass, gt_multiclass, pred_multiclass = per_run_predictions(frame, MC_MAP)
        assert np.array_equal(ids_binary, ids_multiclass)
        if reference_ids is None:
            reference_ids = ids_binary
        assert np.array_equal(ids_binary, reference_ids), (
            f"participant set/order differs for {name}"
        )
        data[name] = {
            "binary": (gt_binary, pred_binary),
            "multiclass": (gt_multiclass, pred_multiclass),
        }

    lines = []
    rows = []
    primary_pair_rows = []
    primary_omnibus_rows = []

    def emit(message=""):
        print(message)
        lines.append(message)

    emit("Paired statistical comparisons of the four LLMs")
    emit(f"participants common to all model outputs: {len(reference_ids)}")
    emit(f"models: {', '.join(names)}")
    emit(f"Monte Carlo permutations: {args.resamples:,}; seed: {args.seed}")
    emit(f"software: Python {sys.version.split()[0]}, NumPy {np.__version__}, "
         f"SciPy {scipy.__version__}, Polars {pl.__version__}")

    # Primary diagnosis endpoint: the five run-level balanced accuracies in Table 1a/b.
    nn_scores = load_nn_run_scores(data_dir, names)
    emit("validation: neural-network run identifiers and fold compositions match across models")
    for condition_index, condition in enumerate(CONDITIONS):
        matrix = np.column_stack([nn_scores[condition][name] for name in names])
        emit(f"\nPRIMARY: diagnosis balanced accuracy, {condition}")
        emit("unit: five matched neural-network aggregation runs")
        for model_index, name in enumerate(names):
            mean = matrix[:, model_index].mean()
            standard_deviation = matrix[:, model_index].std(ddof=1)
            expected_mean, expected_sd = TABLE1_DIAGNOSIS[(name, condition)]
            assert np.isclose(mean, expected_mean, atol=0.05)
            assert np.isclose(standard_deviation, expected_sd, atol=0.05)
            emit(f"  {name}: {mean:.1f} +/- {standard_deviation:.1f} "
                 f"(Table 1: {expected_mean:.1f} +/- {expected_sd:.1f})")

        friedman_statistic, omnibus_p = exact_friedman(matrix)
        raw_p = np.array(
            [
                exact_paired_sign_pvalue(matrix[:, first], matrix[:, second])
                for first, second in pairs
            ]
        )
        adjusted_p = holm(raw_p)
        emit(f"  exact Friedman omnibus: chi2={friedman_statistic:.6f}, p={format_p(omnibus_p)}")
        for label, (first, second), p_value, p_adjusted in zip(
            pair_labels, pairs, raw_p, adjusted_p
        ):
            difference = matrix[:, first].mean() - matrix[:, second].mean()
            emit(f"  {label}: mean difference={difference:+.3f} percentage points, "
                 f"exact paired sign-permutation p={format_p(p_value)}, "
                 f"Holm p={format_p(p_adjusted)}")
            row = {
                "endpoint": "diagnosis_balanced_accuracy",
                "condition": condition,
                "analysis": "primary",
                "unit": "matched aggregation run",
                "comparison": label,
                "test": "exact paired sign-permutation",
                "effect": difference,
                "effect_unit": "balanced-accuracy percentage points",
                "p_raw": float(p_value),
                "p_holm_within_family": float(p_adjusted),
                "p_holm_all_primary": None,
            }
            rows.append(row)
            primary_pair_rows.append(row)
        omnibus_row = {
            "endpoint": "diagnosis_balanced_accuracy",
            "condition": condition,
            "analysis": "primary",
            "unit": "matched aggregation run",
            "comparison": "omnibus",
            "test": "exact Friedman",
            "effect": float(friedman_statistic),
            "effect_unit": "chi-square",
            "p_raw": float(omnibus_p),
            "p_holm_within_family": None,
            "p_holm_all_primary": None,
        }
        rows.append(omnibus_row)
        primary_omnibus_rows.append(omnibus_row)

    # Primary severity endpoint: Table 1c/d balanced accuracy, preserving each
    # participant's complete prediction vector as the randomization block.
    for condition_index, condition in enumerate(CONDITIONS):
        ground_truth = data[names[0]]["multiclass"][0][condition]
        assert all(
            np.array_equal(data[name]["multiclass"][0][condition], ground_truth)
            for name in names
        )
        contributions = np.column_stack(
            [
                balanced_accuracy_contributions(
                    ground_truth, data[name]["multiclass"][1][condition]
                )
                for name in names
            ]
        )
        run_balanced_accuracies = {
            name: 100
            * balanced_accuracy_by_run(
                ground_truth, data[name]["multiclass"][1][condition]
            )
            for name in names
        }
        observed_scores, omnibus_statistic, omnibus_p, raw_p = participant_blocked_tests(
            contributions,
            args.resamples,
            np.random.default_rng(args.seed + 1000 + 10 * condition_index),
            np.random.default_rng(args.seed + 1001 + 10 * condition_index),
        )
        adjusted_p = holm(raw_p)

        emit(f"\nPRIMARY: severity balanced accuracy, {condition}")
        emit("unit: participant; each block retains its complete ten-run prediction vector")
        for model_index, name in enumerate(names):
            mean = 100 * observed_scores[model_index]
            standard_deviation = run_balanced_accuracies[name].std(ddof=1)
            expected_mean, expected_sd = TABLE1_SEVERITY[(name, condition)]
            assert np.isclose(mean, expected_mean, atol=0.05)
            assert np.isclose(standard_deviation, expected_sd, atol=0.05)
            emit(f"  {name}: {mean:.1f} +/- {standard_deviation:.1f} "
                 f"(Table 1: {expected_mean:.1f} +/- {expected_sd:.1f})")
        emit(f"  participant-blocked omnibus permutation: statistic="
             f"{omnibus_statistic:.8f}, p={format_p(omnibus_p)}")
        for label, (first, second), p_value, p_adjusted in zip(
            pair_labels, pairs, raw_p, adjusted_p
        ):
            difference = 100 * (observed_scores[first] - observed_scores[second])
            emit(f"  {label}: mean difference={difference:+.3f} percentage points, "
                 f"participant-blocked permutation p={format_p(p_value)}, "
                 f"Holm p={format_p(p_adjusted)}")
            row = {
                "endpoint": "severity_balanced_accuracy",
                "condition": condition,
                "analysis": "primary",
                "unit": "participant block retaining ten runs",
                "comparison": label,
                "test": "participant-blocked permutation",
                "effect": difference,
                "effect_unit": "balanced-accuracy percentage points",
                "p_raw": float(p_value),
                "p_holm_within_family": float(p_adjusted),
                "p_holm_all_primary": None,
            }
            rows.append(row)
            primary_pair_rows.append(row)
        omnibus_row = {
            "endpoint": "severity_balanced_accuracy",
            "condition": condition,
            "analysis": "primary",
            "unit": "participant block retaining ten runs",
            "comparison": "omnibus",
            "test": "participant-blocked permutation",
            "effect": float(omnibus_statistic),
            "effect_unit": "between-model sum of squares",
            "p_raw": float(omnibus_p),
            "p_holm_within_family": None,
            "p_holm_all_primary": None,
        }
        rows.append(omnibus_row)
        primary_omnibus_rows.append(omnibus_row)

    all_primary_adjusted = holm([row["p_raw"] for row in primary_pair_rows])
    for row, adjusted in zip(primary_pair_rows, all_primary_adjusted):
        row["p_holm_all_primary"] = float(adjusted)
    all_primary_omnibus_adjusted = holm(
        [row["p_raw"] for row in primary_omnibus_rows]
    )
    for row, adjusted in zip(primary_omnibus_rows, all_primary_omnibus_adjusted):
        row["p_holm_all_primary"] = float(adjusted)
    emit("\nMULTIPLICITY SENSITIVITY")
    emit("Holm adjustment across all 24 primary pairwise comparisons: "
         f"minimum adjusted p={format_p(all_primary_adjusted.min())}")
    emit("Holm adjustment across all four primary omnibus tests:")
    for row in primary_omnibus_rows:
        emit(
            f"  {row['endpoint']}, {row['condition']}: "
            f"raw p={format_p(row['p_raw'])}, "
            f"Holm p={format_p(row['p_holm_all_primary'])}"
        )

    # Ordinal sensitivity: participant-level mean absolute severity error.
    for condition_index, condition in enumerate(CONDITIONS):
        errors = np.column_stack(
            [
                np.abs(
                    data[name]["multiclass"][1][condition]
                    - data[name]["multiclass"][0][condition][:, None]
                ).mean(axis=1)
                for name in names
            ]
        )
        friedman_statistic, omnibus_p = stats.friedmanchisquare(
            *[errors[:, model] for model in range(len(names))]
        )
        raw_p = permutation_wilcoxon_pvalues(
            errors,
            pairs,
            args.resamples,
            np.random.default_rng(args.seed + 2000 + condition_index),
        )
        adjusted_p = holm(raw_p)
        emit(f"\nSENSITIVITY: ordinal mean absolute severity error, {condition}")
        emit("unit: participant; error averaged over ten runs")
        emit(f"  Friedman omnibus: chi2={friedman_statistic:.6f}, p={format_p(omnibus_p)}")
        for label, (first, second), p_value, p_adjusted in zip(
            pair_labels, pairs, raw_p, adjusted_p
        ):
            difference = errors[:, first].mean() - errors[:, second].mean()
            emit(f"  {label}: mean difference={difference:+.6f} severity levels, "
                 f"permutation Wilcoxon p={format_p(p_value)}, "
                 f"Holm p={format_p(p_adjusted)}")
            rows.append(
                {
                    "endpoint": "severity_mean_absolute_error",
                    "condition": condition,
                    "analysis": "ordinal sensitivity",
                    "unit": "participant",
                    "comparison": label,
                    "test": "permutation Wilcoxon signed-rank",
                    "effect": float(difference),
                    "effect_unit": "severity levels",
                    "p_raw": float(p_value),
                    "p_holm_within_family": float(p_adjusted),
                    "p_holm_all_primary": None,
                }
            )
        rows.append(
            {
                "endpoint": "severity_mean_absolute_error",
                "condition": condition,
                "analysis": "ordinal sensitivity",
                "unit": "participant",
                "comparison": "omnibus",
                "test": "Friedman",
                "effect": float(friedman_statistic),
                "effect_unit": "chi-square",
                "p_raw": float(omnibus_p),
                "p_holm_within_family": None,
                "p_holm_all_primary": None,
            }
        )

    output_dir.mkdir(parents=True)
    input_hashes = {
        str(path.relative_to(data_dir)): sha256(path) for path in required_files
    }
    metadata = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "participants": int(len(reference_ids)),
        "models": names,
        "mean_aggregation_runs": N_MEAN_RUNS,
        "neural_network_aggregation_runs": N_NN_RUNS,
        "monte_carlo_resamples": args.resamples,
        "seed": args.seed,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "polars": pl.__version__,
        },
        "input_sha256": input_hashes,
        "results": rows,
    }
    (output_dir / "analysis_results.txt").write_text("\n".join(lines) + "\n")
    pl.DataFrame(rows).write_csv(output_dir / "paired_tests.csv")
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"\nwritten without participant-level data: {output_dir}")


if __name__ == "__main__":
    main()
