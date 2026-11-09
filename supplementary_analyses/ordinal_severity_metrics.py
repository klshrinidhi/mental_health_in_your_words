#!/usr/bin/env python
"""Ordinal severity metrics and controlled label-collapse analysis.

This analysis reconstructs the ten published mean-aggregation runs from the
four frozen LLM ``results.csv`` artifacts. It does not make LLM calls and does
not modify source artifacts. It writes aggregate-only outputs and refuses to
overwrite an existing output directory.

The primary descriptive ordinal metrics are:

* participant-weighted mean absolute severity-level error (MAE);
* class-balanced MAE (the mean of class-specific MAEs);
* the proportion of predictions within one severity level; and
* its class-balanced counterpart (the mean of class-specific proportions).

A controlled label-collapse analysis uses the identical prompt selections and
mean-aggregation procedure but maps each prompt-level prediction to
minimal-versus-above-minimal before aggregation. This isolates the reduction
in label granularity without introducing the neural-network binary aggregator.

Usage:
    python supplementary_analyses/ordinal_severity_metrics.py
    python supplementary_analyses/ordinal_severity_metrics.py DATA_DIR \
        --output-dir OUTPUT_DIR
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl


MODELS = {
    "GPT-4o": "prompts_1_11",
    "Gemini-1.5-pro": "prompts_1_12",
    "Llama-3.2-90b": "prompts_1_13",
    "MedLM-Large-1.5": "prompts_1_17",
}
CONDITIONS = ("anxiety", "depression")
N_SAMPLES = 5
N_RUNS = 10
MANUSCRIPT_COHORT_COUNT = 110

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
    f"need polars 1.19.0 to reproduce the published seeded draws; got {pl.__version__}"
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
        default=(Path.home() / "Downloads" / "mental_health"
                 / "supplementary_analysis_outputs" / "ordinal_severity_metrics"),
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(prompts_dir):
    """Load only columns needed to reconstruct the published runs."""
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
    """Reconstruct participant-by-run predictions using published sampling."""
    mapped = frame.with_columns(
        pl.col("anxiety").replace_strict(mapping),
        pl.col("depression").replace_strict(mapping),
        pl.col("anxiety_pd").replace_strict(mapping),
        pl.col("depression_pd").replace_strict(mapping),
    )
    train_ids = mapped["train_participant_ids"].unique().sort()
    runs = []
    selected_ids = []
    for run in range(N_RUNS):
        selected = train_ids.sample(N_SAMPLES, seed=100 * N_SAMPLES + run)
        selected_ids.append(tuple(int(value) for value in selected.to_list()))
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
    for run in runs[1:]:
        assert np.array_equal(run["val_participant_id"].to_numpy(), participant_ids)
        for condition in CONDITIONS:
            assert np.array_equal(run[condition].to_numpy(), ground_truth[condition])
    return participant_ids, ground_truth, predictions, selected_ids


def balanced_accuracy(ground_truth, predictions):
    classes = np.unique(ground_truth)
    return float(
        np.mean(
            [
                np.mean(predictions[ground_truth == label] == label)
                for label in classes
            ]
        )
    )


def class_balanced_mean(values, ground_truth):
    return float(
        np.mean(
            [np.mean(values[ground_truth == label]) for label in np.unique(ground_truth)]
        )
    )


def summarize(rows, group_fields, value_fields):
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    summaries = []
    for key in sorted(grouped):
        group = grouped[key]
        summary = dict(zip(group_fields, key))
        summary["runs"] = len(group)
        if "artifact_participants" in group[0]:
            counts = {row["artifact_participants"] for row in group}
            assert len(counts) == 1
            summary["artifact_participants"] = counts.pop()
        if "class_n" in group[0]:
            counts = {row["class_n"] for row in group}
            assert len(counts) == 1
            summary["class_n"] = counts.pop()
        for field in value_fields:
            values = np.asarray([row[field] for row in group], dtype=float)
            summary[f"{field}_mean"] = float(values.mean())
            summary[f"{field}_sd"] = float(values.std(ddof=1))
            summary[f"{field}_min"] = float(values.min())
            summary[f"{field}_max"] = float(values.max())
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    assert rows
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rounded_for_json(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: rounded_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded_for_json(item) for item in value]
    return value


def format_mean_sd(mean, standard_deviation, digits=3):
    return f"{mean:.{digits}f} +/- {standard_deviation:.{digits}f}"


def main():
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")

    input_paths = [data_dir / directory / "results.csv" for directory in MODELS.values()]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise SystemExit("missing input files:\n" + "\n".join(missing))

    reconstructed = {}
    reference_ids = None
    reference_ground_truth = None
    reference_selected_ids = None
    for model, directory in MODELS.items():
        frame = load_model(data_dir / directory)
        ids_mc, ground_truth_mc, predictions_mc, selected_ids_mc = per_run_predictions(
            frame, MC_MAP
        )
        ids_bin, ground_truth_bin, predictions_bin, selected_ids_bin = per_run_predictions(
            frame, BIN_MAP
        )
        assert np.array_equal(ids_mc, ids_bin)
        assert selected_ids_mc == selected_ids_bin
        for condition in CONDITIONS:
            assert np.array_equal(
                ground_truth_bin[condition], (ground_truth_mc[condition] > 0).astype(int)
            )
            assert predictions_mc[condition].shape == (len(ids_mc), N_RUNS)
            assert predictions_bin[condition].shape == (len(ids_mc), N_RUNS)
            assert set(np.unique(predictions_mc[condition])).issubset({0, 1, 2, 3})
            assert set(np.unique(predictions_bin[condition])).issubset({0, 1})

        if reference_ids is None:
            reference_ids = ids_mc
            reference_ground_truth = ground_truth_mc
            reference_selected_ids = selected_ids_mc
        else:
            assert np.array_equal(ids_mc, reference_ids)
            assert selected_ids_mc == reference_selected_ids
            for condition in CONDITIONS:
                assert np.array_equal(ground_truth_mc[condition], reference_ground_truth[condition])

        reconstructed[model] = {
            "ground_truth_mc": ground_truth_mc,
            "predictions_mc": predictions_mc,
            "ground_truth_bin": ground_truth_bin,
            "predictions_bin": predictions_bin,
        }

    artifact_participants = len(reference_ids)
    ordinal_rows = []
    error_distance_rows = []
    per_class_rows = []
    collapse_rows = []
    parity_rows = []

    for model in MODELS:
        model_data = reconstructed[model]
        for condition in CONDITIONS:
            ground_truth = model_data["ground_truth_mc"][condition]
            predictions = model_data["predictions_mc"][condition]
            ground_truth_bin = model_data["ground_truth_bin"][condition]
            predictions_bin = model_data["predictions_bin"][condition]
            classes = np.unique(ground_truth)
            assert np.array_equal(classes, np.array([0, 1, 2, 3]))

            run_balanced_accuracies = []
            for run in range(N_RUNS):
                predicted = predictions[:, run]
                errors = predicted - ground_truth
                absolute_errors = np.abs(errors)
                exact = absolute_errors == 0
                within_one = absolute_errors <= 1

                severity_ba = 100 * balanced_accuracy(ground_truth, predicted)
                run_balanced_accuracies.append(severity_ba)
                ordinal_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "run": run,
                        "artifact_participants": artifact_participants,
                        "accuracy_pct": 100 * float(exact.mean()),
                        "balanced_accuracy_pct": severity_ba,
                        "mae_levels": float(absolute_errors.mean()),
                        "class_balanced_mae_levels": class_balanced_mean(
                            absolute_errors, ground_truth
                        ),
                        "within_one_pct": 100 * float(within_one.mean()),
                        "class_balanced_within_one_pct": 100
                        * class_balanced_mean(within_one, ground_truth),
                        "signed_error_levels": float(errors.mean()),
                        "class_balanced_signed_error_levels": class_balanced_mean(
                            errors, ground_truth
                        ),
                        "underestimation_pct": 100 * float((errors < 0).mean()),
                        "overestimation_pct": 100 * float((errors > 0).mean()),
                    }
                )

                overall_distance_sum = 0.0
                balanced_distance_sum = 0.0
                for distance in range(4):
                    at_distance = absolute_errors == distance
                    overall_pct = 100 * float(at_distance.mean())
                    balanced_pct = 100 * class_balanced_mean(at_distance, ground_truth)
                    overall_distance_sum += overall_pct
                    balanced_distance_sum += balanced_pct
                    error_distance_rows.append(
                        {
                            "model": model,
                            "condition": condition,
                            "run": run,
                            "absolute_error_levels": distance,
                            "artifact_participants": artifact_participants,
                            "overall_pct": overall_pct,
                            "class_balanced_pct": balanced_pct,
                        }
                    )
                assert math.isclose(overall_distance_sum, 100.0, abs_tol=1e-10)
                assert math.isclose(balanced_distance_sum, 100.0, abs_tol=1e-10)

                for severity_class in classes:
                    mask = ground_truth == severity_class
                    class_errors = errors[mask]
                    class_absolute_errors = absolute_errors[mask]
                    per_class_rows.append(
                        {
                            "model": model,
                            "condition": condition,
                            "run": run,
                            "true_severity_class": int(severity_class),
                            "class_n": int(mask.sum()),
                            "exact_accuracy_pct": 100
                            * float((class_absolute_errors == 0).mean()),
                            "mae_levels": float(class_absolute_errors.mean()),
                            "within_one_pct": 100
                            * float((class_absolute_errors <= 1).mean()),
                            "signed_error_levels": float(class_errors.mean()),
                            "underestimation_pct": 100 * float((class_errors < 0).mean()),
                            "overestimation_pct": 100 * float((class_errors > 0).mean()),
                        }
                    )

                predicted_bin = predictions_bin[:, run]
                collapsed_ba = 100 * balanced_accuracy(ground_truth_bin, predicted_bin)
                collapse_rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "run": run,
                        "artifact_participants": artifact_participants,
                        "severity_balanced_accuracy_pct": severity_ba,
                        "collapsed_binary_balanced_accuracy_pct": collapsed_ba,
                        "balanced_accuracy_gain_pct_points": collapsed_ba - severity_ba,
                        "collapsed_binary_accuracy_pct": 100
                        * float((predicted_bin == ground_truth_bin).mean()),
                        "minimal_recall_pct": 100
                        * float(
                            (predicted_bin[ground_truth_bin == 0] == 0).mean()
                        ),
                        "above_minimal_recall_pct": 100
                        * float(
                            (predicted_bin[ground_truth_bin == 1] == 1).mean()
                        ),
                    }
                )

            observed_mean = float(np.mean(run_balanced_accuracies))
            observed_sd = float(np.std(run_balanced_accuracies, ddof=1))
            expected_mean, expected_sd = TABLE1_SEVERITY[(model, condition)]
            assert np.isclose(observed_mean, expected_mean, atol=0.05)
            assert np.isclose(observed_sd, expected_sd, atol=0.05)
            parity_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "observed_balanced_accuracy_mean_pct": observed_mean,
                    "observed_balanced_accuracy_sd_pct": observed_sd,
                    "table1_balanced_accuracy_mean_pct": expected_mean,
                    "table1_balanced_accuracy_sd_pct": expected_sd,
                    "mean_difference_pct_points": observed_mean - expected_mean,
                    "sd_difference_pct_points": observed_sd - expected_sd,
                    "parity_within_0_05_pct_points": True,
                }
            )

    ordinal_value_fields = [
        "accuracy_pct",
        "balanced_accuracy_pct",
        "mae_levels",
        "class_balanced_mae_levels",
        "within_one_pct",
        "class_balanced_within_one_pct",
        "signed_error_levels",
        "class_balanced_signed_error_levels",
        "underestimation_pct",
        "overestimation_pct",
    ]
    ordinal_summary = summarize(
        ordinal_rows, ["model", "condition"], ordinal_value_fields
    )
    error_distance_summary = summarize(
        error_distance_rows,
        ["model", "condition", "absolute_error_levels"],
        ["overall_pct", "class_balanced_pct"],
    )
    per_class_summary = summarize(
        per_class_rows,
        ["model", "condition", "true_severity_class"],
        [
            "exact_accuracy_pct",
            "mae_levels",
            "within_one_pct",
            "signed_error_levels",
            "underestimation_pct",
            "overestimation_pct",
        ],
    )
    collapse_summary = summarize(
        collapse_rows,
        ["model", "condition"],
        [
            "severity_balanced_accuracy_pct",
            "collapsed_binary_balanced_accuracy_pct",
            "balanced_accuracy_gain_pct_points",
            "collapsed_binary_accuracy_pct",
            "minimal_recall_pct",
            "above_minimal_recall_pct",
        ],
    )

    class_counts = []
    for condition in CONDITIONS:
        ground_truth = reference_ground_truth[condition]
        for severity_class in range(4):
            class_counts.append(
                {
                    "condition": condition,
                    "severity_class": severity_class,
                    "artifact_participants": int((ground_truth == severity_class).sum()),
                }
            )

    by_model_condition = {
        (row["model"], row["condition"]): row for row in ordinal_summary
    }
    collapse_by_model_condition = {
        (row["model"], row["condition"]): row for row in collapse_summary
    }
    ranges = {}
    for condition in CONDITIONS:
        condition_rows = [
            by_model_condition[(model, condition)] for model in MODELS
        ]
        condition_collapse_rows = [
            collapse_by_model_condition[(model, condition)] for model in MODELS
        ]
        ranges[condition] = {
            "balanced_accuracy_mean_pct": [
                min(row["balanced_accuracy_pct_mean"] for row in condition_rows),
                max(row["balanced_accuracy_pct_mean"] for row in condition_rows),
            ],
            "mae_mean_levels": [
                min(row["mae_levels_mean"] for row in condition_rows),
                max(row["mae_levels_mean"] for row in condition_rows),
            ],
            "class_balanced_mae_mean_levels": [
                min(row["class_balanced_mae_levels_mean"] for row in condition_rows),
                max(row["class_balanced_mae_levels_mean"] for row in condition_rows),
            ],
            "class_balanced_within_one_mean_pct": [
                min(
                    row["class_balanced_within_one_pct_mean"]
                    for row in condition_rows
                ),
                max(
                    row["class_balanced_within_one_pct_mean"]
                    for row in condition_rows
                ),
            ],
            "collapsed_binary_balanced_accuracy_mean_pct": [
                min(
                    row["collapsed_binary_balanced_accuracy_pct_mean"]
                    for row in condition_collapse_rows
                ),
                max(
                    row["collapsed_binary_balanced_accuracy_pct_mean"]
                    for row in condition_collapse_rows
                ),
            ],
            "balanced_accuracy_gain_mean_pct_points": [
                min(
                    row["balanced_accuracy_gain_pct_points_mean"]
                    for row in condition_collapse_rows
                ),
                max(
                    row["balanced_accuracy_gain_pct_points_mean"]
                    for row in condition_collapse_rows
                ),
            ],
        }

    output_dir.mkdir(parents=True)
    output_files = {
        "ordinal_metrics_by_run.csv": ordinal_rows,
        "ordinal_metrics_summary.csv": ordinal_summary,
        "error_distance_by_run.csv": error_distance_rows,
        "error_distance_summary.csv": error_distance_summary,
        "per_class_metrics_by_run.csv": per_class_rows,
        "per_class_metrics_summary.csv": per_class_summary,
        "controlled_collapse_by_run.csv": collapse_rows,
        "controlled_collapse_summary.csv": collapse_summary,
        "table1_parity.csv": parity_rows,
        "class_counts.csv": class_counts,
    }
    for filename, rows in output_files.items():
        write_csv(output_dir / filename, rows)

    metadata = {
        "analysis": "Ordinal severity metrics and controlled label collapse",
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "manuscript_cohort_count_context": MANUSCRIPT_COHORT_COUNT,
        "participant_ids_in_evaluation_artifacts": artifact_participants,
        "models": list(MODELS),
        "conditions": list(CONDITIONS),
        "mean_aggregation_runs": N_RUNS,
        "prompts_per_run": N_SAMPLES,
        "run_sampling_seeds": [100 * N_SAMPLES + run for run in range(N_RUNS)],
        "metric_definitions": {
            "balanced_accuracy": "Mean exact-class recall across the four ground-truth severity classes.",
            "mae_levels": "Participant-weighted mean absolute difference between predicted and ground-truth severity class indices on the 0-3 scale.",
            "class_balanced_mae_levels": "Mean of the four ground-truth-class-specific MAEs.",
            "within_one_pct": "Participant-weighted percentage with absolute severity-class error <= 1.",
            "class_balanced_within_one_pct": "Mean of the four ground-truth-class-specific percentages with absolute severity-class error <= 1.",
            "controlled_collapse": "The same five prompt outputs in each run are mapped to minimal=0 versus above-minimal=1 before mean aggregation; no neural-network aggregator is used.",
        },
        "validation": {
            "identical_participant_order_across_models": True,
            "identical_ground_truth_across_models": True,
            "identical_prompt_samples_across_models_and_endpoints": True,
            "all_eight_table1_severity_mean_sd_cells_reproduced_within_0_05_pct_points": True,
            "aggregate_outputs_only": True,
            "source_artifacts_modified": False,
            "llm_calls_made": False,
        },
        "table1_parity": parity_rows,
        "ranges_across_models": ranges,
        "class_counts_in_evaluation_artifacts": class_counts,
        "input_sha256": {
            str(path.relative_to(data_dir)): sha256(path) for path in input_paths
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "polars": pl.__version__,
        },
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "output_files": sorted(
            [*output_files, "analysis_results.txt", "analysis_summary.json"]
        ),
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(rounded_for_json(metadata), indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "Ordinal severity metrics and controlled label collapse",
        f"Evaluation participant IDs in frozen outputs: {artifact_participants}",
        f"Manuscript cohort count context (not modified): {MANUSCRIPT_COHORT_COUNT}",
        f"Runs: {N_RUNS}; prompts per run: {N_SAMPLES}",
        "All eight Table 1 severity balanced-accuracy mean/SD cells reproduced within 0.05 percentage points.",
        "No LLM calls; aggregate-only outputs; source artifacts opened read-only.",
        "",
        "ORDINAL METRICS (mean +/- SD across ten runs)",
    ]
    for condition in CONDITIONS:
        lines.append(f"\n{condition.upper()}")
        for model in MODELS:
            row = by_model_condition[(model, condition)]
            lines.append(
                f"  {model}: BA {format_mean_sd(row['balanced_accuracy_pct_mean'], row['balanced_accuracy_pct_sd'], 2)}%; "
                f"MAE {format_mean_sd(row['mae_levels_mean'], row['mae_levels_sd'])} levels; "
                f"class-balanced MAE {format_mean_sd(row['class_balanced_mae_levels_mean'], row['class_balanced_mae_levels_sd'])} levels; "
                f"within-one {format_mean_sd(row['within_one_pct_mean'], row['within_one_pct_sd'], 2)}%; "
                f"class-balanced within-one {format_mean_sd(row['class_balanced_within_one_pct_mean'], row['class_balanced_within_one_pct_sd'], 2)}%; "
                f"signed error {format_mean_sd(row['signed_error_levels_mean'], row['signed_error_levels_sd'])} levels"
            )
        condition_range = ranges[condition]
        lines.append(
            "  Across-model ranges of run means: "
            f"MAE {condition_range['mae_mean_levels'][0]:.3f}-{condition_range['mae_mean_levels'][1]:.3f} levels; "
            f"class-balanced within-one {condition_range['class_balanced_within_one_mean_pct'][0]:.1f}-{condition_range['class_balanced_within_one_mean_pct'][1]:.1f}%"
        )

    lines.extend(["", "CONTROLLED MINIMAL/ABOVE-MINIMAL COLLAPSE (mean +/- SD across ten runs)"])
    for condition in CONDITIONS:
        lines.append(f"\n{condition.upper()}")
        for model in MODELS:
            row = collapse_by_model_condition[(model, condition)]
            lines.append(
                f"  {model}: four-class BA {format_mean_sd(row['severity_balanced_accuracy_pct_mean'], row['severity_balanced_accuracy_pct_sd'], 2)}%; "
                f"collapsed binary BA {format_mean_sd(row['collapsed_binary_balanced_accuracy_pct_mean'], row['collapsed_binary_balanced_accuracy_pct_sd'], 2)}%; "
                f"increase {format_mean_sd(row['balanced_accuracy_gain_pct_points_mean'], row['balanced_accuracy_gain_pct_points_sd'], 2)} percentage points"
            )
        condition_range = ranges[condition]
        lines.append(
            "  Across-model ranges of run means: "
            f"collapsed binary BA {condition_range['collapsed_binary_balanced_accuracy_mean_pct'][0]:.1f}-{condition_range['collapsed_binary_balanced_accuracy_mean_pct'][1]:.1f}%; "
            f"increase {condition_range['balanced_accuracy_gain_mean_pct_points'][0]:.1f}-{condition_range['balanced_accuracy_gain_mean_pct_points'][1]:.1f} percentage points"
        )

    lines.extend(["", "CLASS COUNTS IN FROZEN EVALUATION ARTIFACTS"])
    for condition in CONDITIONS:
        counts = [
            row["artifact_participants"]
            for row in class_counts
            if row["condition"] == condition
        ]
        lines.append(f"  {condition}: " + ", ".join(f"class {index}={count}" for index, count in enumerate(counts)))

    result_text = "\n".join(lines) + "\n"
    (output_dir / "analysis_results.txt").write_text(result_text, encoding="utf-8")
    print(result_text, end="")
    print(f"\nWrote aggregate outputs to {output_dir}")


if __name__ == "__main__":
    main()
