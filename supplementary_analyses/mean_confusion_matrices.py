#!/usr/bin/env python
"""Regenerate the confusion matrices from all ten mean-aggregation runs.

This script reconstructs the canonical ten S=5, N=1 runs, computes a
row-normalized confusion matrix within each run, and averages corresponding
cells across runs.  Because every run evaluates the same participants, this is
also equivalent to pooling the ten predictions per participant before row
normalization.

The canonical reconstruction uses Polars 1.19 seeded sampling and its
round-half-away-from-zero behavior.  These are the runs that reproduce the
published Table 1c/d severity means and standard deviations.  The binary
panels continue to describe mean aggregation, as stated in the figure caption;
the main binary results use the separately reported neural-network aggregator.

Only aggregate outputs are written.  The output directory must not already
exist, and the source results.csv files are checked against the SHA-256
manifest from the independently validated paired-comparison analysis.

Usage:
    python supplementary_analyses/mean_confusion_matrices.py \
        --output-dir OUTPUT_DIR
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


MODELS = {
    "GPT-4o": ("prompts_1_11", "gpt4o"),
    "Gemini-1.5-pro": ("prompts_1_12", "gemini"),
    "Llama-3.2-90b": ("prompts_1_13", "llama"),
    "MedLM-Large-1.5": ("prompts_1_17", "medlm"),
}
CONDITIONS = {
    "anxiety": "anx",
    "depression": "dep",
}
ENDPOINTS = {
    "binary": {
        "mapping": {
            "minimal": 0,
            "mild": 1,
            "moderate": 1,
            "moderately-severe": 1,
            "severe": 1,
        },
        "labels": [0, 1],
        "ticklabels": ["Negative", "Positive"],
        "figsize": (5, 4),
        "ticksize": 20,
        "annotationsize": 24,
    },
    "multiclass": {
        "mapping": {
            "minimal": 0,
            "mild": 1,
            "moderate": 2,
            "moderately-severe": 3,
            "severe": 3,
        },
        "labels": [0, 1, 2, 3],
        "ticklabels": ["Minimal", "Mild", "Moderate", "Severe"],
        "figsize": (10, 8),
        "ticksize": 24,
        "annotationsize": 28,
    },
}
N_PROMPTS = 5
N_RUNS = 10
EXPECTED_PARTICIPANTS = 106
EXPECTED_CLASS_COUNTS = {
    "binary": {
        "anxiety": (60, 46),
        "depression": (68, 38),
    },
    "multiclass": {
        "anxiety": (60, 24, 12, 10),
        "depression": (68, 27, 8, 3),
    },
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
    f"need Polars 1.19.0 to reproduce the published seeded draws; "
    f"got {pl.__version__}"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / "Downloads" / "mental_health",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_input_hashes(data_dir):
    manifest_path = (
        data_dir / "supplementary_analysis_outputs"
        / "paired_model_comparisons" / "analysis_summary.json"
    )
    manifest = json.loads(manifest_path.read_text())["input_sha256"]
    hashes = {}
    for prompts_dir, _ in MODELS.values():
        relative = f"{prompts_dir}/results.csv"
        observed = sha256(data_dir / relative)
        expected = manifest[relative]
        assert observed == expected, f"SHA-256 mismatch: {relative}"
        hashes[relative] = observed
    return manifest_path, hashes


def load_model(path):
    prompt_id = path.name.replace("prompts", "prompt")
    frame = pl.read_csv(path / "results.csv")
    prediction_columns = {
        parts[1]: column
        for column in frame.columns
        if len(parts := column.split("\n")) == 3
        and parts[2] == prompt_id
        and parts[1] in CONDITIONS
    }
    assert set(prediction_columns) == set(CONDITIONS), (
        f"missing prediction columns in {path / 'results.csv'}"
    )
    return frame.select(
        "val_participant_id",
        "train_participant_ids",
        "anxiety",
        "depression",
        anxiety_pd=prediction_columns["anxiety"],
        depression_pd=prediction_columns["depression"],
    )


def reconstruct_runs(frame, mapping):
    mapped = frame.with_columns(
        pl.col("anxiety").replace_strict(mapping),
        pl.col("depression").replace_strict(mapping),
        pl.col("anxiety_pd").replace_strict(mapping),
        pl.col("depression_pd").replace_strict(mapping),
    )
    train_ids = mapped["train_participant_ids"].unique().sort()
    runs = []
    for run in range(N_RUNS):
        selected = train_ids.sample(N_PROMPTS, seed=100 * N_PROMPTS + run)
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
        assert aggregated.height == EXPECTED_PARTICIPANTS
        runs.append(aggregated)

    participant_ids = runs[0]["val_participant_id"].to_numpy()
    assert all(
        np.array_equal(run["val_participant_id"].to_numpy(), participant_ids)
        for run in runs[1:]
    )
    ground_truth = {
        condition: runs[0][condition].to_numpy() for condition in CONDITIONS
    }
    predictions = {
        condition: np.column_stack(
            [run[f"{condition}_pd"].to_numpy() for run in runs]
        )
        for condition in CONDITIONS
    }
    return ground_truth, predictions


def normalized_confusion_matrix(ground_truth, prediction, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for true_index, true_label in enumerate(labels):
        mask = ground_truth == true_label
        assert mask.any()
        for predicted_index, predicted_label in enumerate(labels):
            matrix[true_index, predicted_index] = np.mean(
                prediction[mask] == predicted_label
            )
    assert np.allclose(matrix.sum(axis=1), 1.0)
    return matrix


def format_cell(value):
    return format(value, ".2g")


def save_heatmap(matrix, config, output_path):
    fig, axis = plt.subplots(figsize=config["figsize"])
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_ticks(np.linspace(0, 1, 5 if len(matrix) == 2 else 6))
    colorbar.ax.tick_params(labelsize=config["ticksize"])

    axis.set_xticks(np.arange(len(matrix)), labels=config["ticklabels"])
    axis.set_yticks(np.arange(len(matrix)), labels=config["ticklabels"])
    axis.tick_params(
        axis="x", bottom=True, labelsize=config["ticksize"], rotation=0
    )
    axis.tick_params(
        axis="y", left=True, labelsize=config["ticksize"], rotation=90
    )
    axis.set_xticks(np.arange(-0.5, len(matrix), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(matrix), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.5, alpha=0.35)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for row in range(len(matrix)):
        for column in range(len(matrix)):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                format_cell(value),
                ha="center",
                va="center",
                color="white" if value >= 0.6 else "#222222",
                fontsize=config["annotationsize"],
            )

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True)
    manifest_path, input_hashes = load_input_hashes(args.data_dir)

    per_run_rows = []
    matrix_rows = []
    summary = {
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "participants": EXPECTED_PARTICIPANTS,
        "prompts_per_run": N_PROMPTS,
        "runs": N_RUNS,
        "protocol": (
            "mean of corresponding cells from ten row-normalized confusion "
            "matrices reconstructed with the canonical Table 1 seeded draws"
        ),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "polars": pl.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "input_manifest": str(manifest_path),
        "input_sha256": input_hashes,
        "results": {},
    }

    maximum_selection_gap = 0.0
    for model, (prompts_dir, file_stem) in MODELS.items():
        frame = load_model(args.data_dir / prompts_dir)
        for endpoint, config in ENDPOINTS.items():
            ground_truth, predictions = reconstruct_runs(frame, config["mapping"])
            for condition, condition_stem in CONDITIONS.items():
                truth = ground_truth[condition]
                predicted = predictions[condition]
                counts = tuple(
                    int(np.sum(truth == label)) for label in config["labels"]
                )
                assert counts == EXPECTED_CLASS_COUNTS[endpoint][condition]

                run_matrices = [
                    normalized_confusion_matrix(
                        truth, predicted[:, run], config["labels"]
                    )
                    for run in range(N_RUNS)
                ]
                mean_matrix = np.mean(run_matrices, axis=0)
                run_balanced_accuracy = np.array(
                    [100 * np.trace(matrix) / len(matrix) for matrix in run_matrices]
                )
                mean_balanced_accuracy = float(run_balanced_accuracy.mean())
                mean_matrix_balanced_accuracy = float(
                    100 * np.trace(mean_matrix) / len(mean_matrix)
                )
                assert np.isclose(
                    mean_balanced_accuracy, mean_matrix_balanced_accuracy, atol=1e-12
                )

                if endpoint == "multiclass":
                    expected_mean, expected_sd = TABLE1_SEVERITY[(model, condition)]
                    assert np.isclose(
                        mean_balanced_accuracy, expected_mean, atol=0.05
                    ), (model, condition, mean_balanced_accuracy, expected_mean)
                    assert np.isclose(
                        run_balanced_accuracy.std(ddof=1), expected_sd, atol=0.05
                    ), (
                        model,
                        condition,
                        run_balanced_accuracy.std(ddof=1),
                        expected_sd,
                    )

                best_run = int(np.argmax(run_balanced_accuracy))
                selection_gap = float(
                    run_balanced_accuracy[best_run] - mean_balanced_accuracy
                )
                maximum_selection_gap = max(maximum_selection_gap, selection_gap)
                key = f"{model}|{condition}|{endpoint}"
                summary["results"][key] = {
                    "ground_truth_class_counts": list(counts),
                    "balanced_accuracy_by_run": [
                        float(value) for value in run_balanced_accuracy
                    ],
                    "balanced_accuracy_mean": mean_balanced_accuracy,
                    "balanced_accuracy_sd": float(
                        run_balanced_accuracy.std(ddof=1)
                    ),
                    "averaged_matrix_diagonal_mean": (
                        mean_matrix_balanced_accuracy
                    ),
                    "best_run": best_run,
                    "best_run_balanced_accuracy": float(
                        run_balanced_accuracy[best_run]
                    ),
                    "best_minus_mean_percentage_points": selection_gap,
                    "mean_confusion_matrix": mean_matrix.tolist(),
                }

                for run, balanced_accuracy in enumerate(run_balanced_accuracy):
                    per_run_rows.append(
                        [model, condition, endpoint, run, balanced_accuracy]
                    )
                for true_index, true_label in enumerate(config["labels"]):
                    for predicted_index, predicted_label in enumerate(
                        config["labels"]
                    ):
                        matrix_rows.append(
                            [
                                model,
                                condition,
                                endpoint,
                                true_label,
                                predicted_label,
                                mean_matrix[true_index, predicted_index],
                            ]
                        )

                output_name = (
                    f"{file_stem}_{endpoint}_{condition_stem}_cm.pdf"
                )
                save_heatmap(mean_matrix, config, args.output_dir / output_name)

    summary["maximum_best_minus_mean_percentage_points"] = maximum_selection_gap
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (args.output_dir / "per_run_balanced_accuracy.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["model", "condition", "endpoint", "run", "balanced_accuracy"]
        )
        writer.writerows(per_run_rows)
    with (args.output_dir / "mean_confusion_matrices.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "model",
                "condition",
                "endpoint",
                "true_label",
                "predicted_label",
                "mean_row_proportion",
            ]
        )
        writer.writerows(matrix_rows)

    print(f"wrote aggregate analysis and 16 figure panels to {args.output_dir}")
    print(
        "validation: every averaged-matrix diagonal mean equals the ten-run "
        "mean balanced accuracy"
    )
    print(
        "validation: all eight severity means/SDs reproduce Table 1c/d "
        "to the reported precision"
    )
    print(
        f"largest best-run minus all-run-mean gap: "
        f"{maximum_selection_gap:.2f} percentage points"
    )


if __name__ == "__main__":
    main()
