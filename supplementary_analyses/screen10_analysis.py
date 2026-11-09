"""Screen-10 analysis at GAD-7 or PHQ-9 total score >= 10.

This script computes screen-positivity performance from the frozen per-prompt
outputs (results.csv) over the ten canonical
mean-aggregation runs, without any model rerun or aggregator retraining.

Operationalizations of the >= 10 binarization (per run):
* indicator_mean (PRIMARY): map each prompt-level predicted total to a 0/1
  screen-positive indicator (total >= 10), then apply the same training-free
  mean aggregation used by the manuscript's severity analysis and the
  controlled label-collapse analysis (mean over the run's
  prompt-level values, rounded half away from zero).
* class_threshold: threshold the run-level aggregated severity class
  (>= moderate) from the reported Table 1c,d reconstruction. Since class
  >= moderate corresponds to total >= 10 for both instruments, this collapses
  the aggregated ordinal prediction instead of the prompt-level indicators.
* total_mean: threshold the run-level mean predicted total at >= 10
  (aggregate first, then binarize).

Metrics per run: balanced accuracy, sensitivity, specificity, positive
predictive value, and predicted-positive count; reported as mean and standard
deviation (ddof=1) over the ten runs. ROC-AUC uses the run-level mean
predicted total as a continuous score (threshold-free, tie-corrected
Mann-Whitney formulation). Stratified participant-bootstrap percentile
intervals quantify sampling uncertainty in balanced accuracy, sensitivity,
and specificity averaged across the ten fixed runs. A draw-free sensitivity
aggregates each participant over all 200 in-context combinations.

Validation gates:
1. Input results.csv SHA-256 digests must match the paired-comparison manifest.
2. Reconstructed ten-run mean +/- std multiclass balanced accuracy must match
   the reported Table 1c,d values (anchors this script to the reported
   record; the class_threshold operationalization reuses this reconstruction).
3. Ground-truth premise, verified against questionnaire_data_clean.csv for
   the evaluated participants: the merged severity class derived from the
   ground-truth questionnaire total (bins at 5/10/15 for both instruments,
   moderately-severe merged into severe) must equal the results.csv label,
   and (total >= 10) must equal (merged class >= moderate) participant by
   participant, with exactly 22 anxiety and 11 depression positives.
4. Ground-truth class counts 60/24/12/10 (anxiety) and 68/27/8/3 (depression)
   over the 106 evaluated participants; predicted totals within instrument
   ranges (GAD-7 0-21, PHQ-9 0-27).

Data quirk handled: the Gemini results.csv names its depression total column
'depression totalscore' (no hyphen); total columns are therefore matched by
substring rather than exact name.

Reconstruction functions are shared with paired_model_comparisons.py
(the digit-perfect Table 1c,d reproduction). Polars 1.19.0 is required because
it reproduces the reported seeded draws.

Usage:
    python supplementary_analyses/screen10_analysis.py [data_dir] \
        [--output-dir output_dir]
"""

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata


MODELS = {
    "GPT-4o": "prompts_1_11",
    "Gemini-1.5-pro": "prompts_1_12",
    "Llama-3.2-90b": "prompts_1_13",
    "MedLM-Large-1.5": "prompts_1_17",
}
CONDITIONS = ("anxiety", "depression")
N_SAMPLES = 5
N_MEAN_RUNS = 10
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 22_022
THRESHOLD = 10
OPERATIONALIZATIONS = ("indicator_mean", "class_threshold", "total_mean")

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
GT_CLASS_COUNTS = {
    "anxiety": (60, 24, 12, 10),
    "depression": (68, 27, 8, 3),
}
EXPECTED_POSITIVES = {"anxiety": 22, "depression": 11}
SCORE_MAX = {"anxiety": 21, "depression": 27}
GT_SCORE_COLUMN = {"anxiety": "GAD7_score", "depression": "PHQ9_score"}

DOWNLOADS = Path.home() / "Downloads" / "mental_health"
OUTPUT_BASE = DOWNLOADS / "supplementary_analysis_outputs" / "screen10"

assert pl.__version__ == "1.19.0", (
    f"need polars 1.19.0 for the reported seeded draws; got {pl.__version__}"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=DOWNLOADS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_BASE,
        help="base directory in which a new versioned output folder is created",
    )
    return parser.parse_args()


def versioned_output_dir(base):
    for version in itertools.count(1):
        candidate = base / f"v{version:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(prompts_dir):
    """Load labels and predicted totals needed to reconstruct the runs."""
    prompt_id = prompts_dir.name.replace("prompts", "prompt")
    frame = pl.read_csv(prompts_dir / "results.csv")
    label_columns, total_columns = {}, {}
    for column in frame.columns:
        parts = column.split("\n")
        if len(parts) != 3 or parts[2] != prompt_id:
            continue
        if parts[1] in CONDITIONS:
            label_columns[parts[1]] = column
        elif "total" in parts[1]:
            total_columns[parts[1].split()[0]] = column
    assert set(label_columns) == set(CONDITIONS), label_columns
    assert set(total_columns) == set(CONDITIONS), total_columns
    return frame.select(
        "val_participant_id",
        "train_participant_ids",
        "anxiety",
        "depression",
        anxiety_pd=label_columns["anxiety"],
        depression_pd=label_columns["depression"],
        anxiety_total=total_columns["anxiety"],
        depression_total=total_columns["depression"],
    )


def per_run_class_predictions(frame, mapping):
    """Reconstruct the ten mean-aggregation runs as participant x run arrays.

    Shared with paired_model_comparisons.py (digit-perfect Table 1c,d
    reproduction).
    """
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


def per_run_totals(frame):
    """Per-run screen indicators and mean totals from prompt-level totals."""
    with_indicators = frame.with_columns(
        (pl.col("anxiety_total") >= THRESHOLD).alias("anxiety_ind"),
        (pl.col("depression_total") >= THRESHOLD).alias("depression_ind"),
    )
    train_ids = with_indicators["train_participant_ids"].unique().sort()
    indicator_mean, total_mean = {c: [] for c in CONDITIONS}, {c: [] for c in CONDITIONS}
    for run in range(N_MEAN_RUNS):
        selected = train_ids.sample(N_SAMPLES, seed=100 * N_SAMPLES + run)
        aggregated = (
            with_indicators.filter(pl.col("train_participant_ids").is_in(selected))
            .group_by("val_participant_id")
            .agg(
                pl.col("anxiety_ind").mean(),
                pl.col("depression_ind").mean(),
                pl.col("anxiety_total").mean(),
                pl.col("depression_total").mean(),
            )
            .sort("val_participant_id")
        )
        for condition in CONDITIONS:
            indicator_mean[condition].append(
                aggregated[f"{condition}_ind"].to_numpy()
            )
            total_mean[condition].append(
                aggregated[f"{condition}_total"].to_numpy()
            )
    return (
        {c: np.column_stack(v) for c, v in indicator_mean.items()},
        {c: np.column_stack(v) for c, v in total_mean.items()},
    )


def all_combo_totals(frame):
    """Draw-free sensitivity: indicators and totals over all 200 combinations."""
    with_indicators = frame.with_columns(
        (pl.col("anxiety_total") >= THRESHOLD).alias("anxiety_ind"),
        (pl.col("depression_total") >= THRESHOLD).alias("depression_ind"),
    )
    aggregated = (
        with_indicators.group_by("val_participant_id")
        .agg(
            pl.col("anxiety_ind").mean(),
            pl.col("depression_ind").mean(),
            pl.col("anxiety_total").mean(),
            pl.col("depression_total").mean(),
        )
        .sort("val_participant_id")
    )
    return aggregated


def balanced_accuracy_by_run(ground_truth, predictions):
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


def screen_metrics(gt_positive, predicted_positive):
    """Balanced accuracy, sensitivity, specificity, PPV, predicted positives."""
    sensitivity = float(np.mean(predicted_positive[gt_positive]))
    specificity = float(np.mean(~predicted_positive[~gt_positive]))
    n_predicted = int(predicted_positive.sum())
    ppv = (
        float(np.mean(gt_positive[predicted_positive])) if n_predicted else np.nan
    )
    return {
        "balanced_accuracy": 100 * (sensitivity + specificity) / 2,
        "sensitivity": 100 * sensitivity,
        "specificity": 100 * specificity,
        "ppv": 100 * ppv,
        "predicted_positives": n_predicted,
    }


def roc_auc(gt_positive, scores):
    """Tie-corrected Mann-Whitney ROC-AUC for a continuous score."""
    ranks = rankdata(scores)
    n_pos = int(gt_positive.sum())
    n_neg = len(gt_positive) - n_pos
    rank_sum = float(ranks[gt_positive].sum())
    return 100 * (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def mean_sd(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def bootstrap_intervals(predictions, positive_draws, negative_draws):
    """Stratified participant-bootstrap intervals for ten-run mean metrics."""
    assert predictions.shape[1] == N_MEAN_RUNS
    assert positive_draws.shape[0] == N_BOOTSTRAP
    assert negative_draws.shape[0] == N_BOOTSTRAP

    # Each array is bootstrap replicate x sampled participant x fixed run.
    positive_predictions = predictions[positive_draws, :]
    negative_predictions = predictions[negative_draws, :]
    sensitivity = positive_predictions.mean(axis=1)
    specificity = (~negative_predictions).mean(axis=1)
    balanced_accuracy = (sensitivity + specificity) / 2

    replicate_metrics = {
        "balanced_accuracy": 100 * balanced_accuracy.mean(axis=1),
        "sensitivity": 100 * sensitivity.mean(axis=1),
        "specificity": 100 * specificity.mean(axis=1),
    }
    return {
        metric: tuple(
            float(value) for value in np.percentile(values, [2.5, 97.5])
        )
        for metric, values in replicate_metrics.items()
    }


def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = versioned_output_dir(args.output_dir)
    manifest_path = (
        data_dir / "supplementary_analysis_outputs"
        / "paired_model_comparisons" / "analysis_summary.json"
    )
    questionnaire_path = data_dir / "questionnaire_data_clean.csv"

    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("Screen-10 performance (total >= 10) over the ten "
         "canonical mean-aggregation runs")
    emit(f"data_dir: {data_dir}")
    emit(f"output_dir: {output_dir}")
    emit(f"polars {pl.__version__}, numpy {np.__version__}, "
         f"python {sys.version.split()[0]}")

    manifest = json.loads(manifest_path.read_text())["input_sha256"]
    input_sha = {}
    for name, prompts_dir in MODELS.items():
        key = f"{prompts_dir}/results.csv"
        digest = sha256(data_dir / prompts_dir / "results.csv")
        assert digest == manifest[key], (
            f"{key} SHA-256 mismatch vs paired-comparison manifest"
        )
        input_sha[key] = digest
    input_sha["questionnaire_data_clean.csv"] = sha256(questionnaire_path)
    emit("gate 1 OK: all four results.csv SHA-256 digests match the "
         "paired-comparison manifest")

    questionnaire = pl.read_csv(questionnaire_path).select(
        "participant_id", "GAD7_score", "PHQ9_score"
    )

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "participants": None,
        "threshold": THRESHOLD,
        "mean_aggregation_runs": N_MEAN_RUNS,
        "bootstrap": {
            "method": (
                "stratified participant bootstrap within screen-positive and "
                "screen-negative classes; statistic is the mean metric across "
                "the ten fixed runs"
            ),
            "replicates": N_BOOTSTRAP,
            "base_seed": BOOTSTRAP_SEED,
            "interval": "2.5th and 97.5th percentiles",
        },
        "ground_truth_positives": {},
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "polars": pl.__version__,
        },
        "input_sha256": input_sha,
        "screen_positivity": {},
        "all200_sensitivity": {},
    }
    csv_rows = []
    bootstrap_rows = []
    reference_ids = None
    reference_ground_truth = {}
    bootstrap_draws = {}

    gt_checked = False
    for name, prompts_dir in MODELS.items():
        frame = load_model(data_dir / prompts_dir)
        ids, gt_mc, pred_mc = per_run_class_predictions(frame, MC_MAP)
        assert len(ids) == 106, f"{name}: expected 106 participants, got {len(ids)}"
        if reference_ids is None:
            reference_ids = ids.copy()
        else:
            assert np.array_equal(ids, reference_ids), (
                f"{name}: evaluated participant IDs differ from the first model"
            )
        summary["participants"] = int(len(ids))
        indicator_mean, total_mean = per_run_totals(frame)
        all200 = all_combo_totals(frame)
        assert np.array_equal(all200["val_participant_id"].to_numpy(), ids)

        gt_scores = (
            pl.DataFrame({"participant_id": ids})
            .join(questionnaire, on="participant_id", how="left")
            .sort("participant_id")
        )
        assert np.array_equal(gt_scores["participant_id"].to_numpy(), ids)

        for condition in CONDITIONS:
            counts = tuple(
                int(np.sum(gt_mc[condition] == label)) for label in range(4)
            )
            assert counts == GT_CLASS_COUNTS[condition], (
                f"{name} {condition}: ground-truth class counts {counts}"
            )

            scores = gt_scores[GT_SCORE_COLUMN[condition]].to_numpy()
            assert not np.isnan(scores.astype(float)).any()
            derived_class = np.digitize(scores, [5, 10, 15])
            assert np.array_equal(derived_class, gt_mc[condition]), (
                f"{name} {condition}: results.csv labels disagree with "
                "questionnaire-score-derived classes"
            )
            gt_positive = scores >= THRESHOLD
            assert np.array_equal(gt_positive, gt_mc[condition] >= 2)
            assert int(gt_positive.sum()) == EXPECTED_POSITIVES[condition]
            if condition not in reference_ground_truth:
                reference_ground_truth[condition] = gt_positive.copy()
            else:
                assert np.array_equal(
                    gt_positive, reference_ground_truth[condition]
                ), f"{name} {condition}: ground truth differs from the first model"
            summary["ground_truth_positives"][condition] = int(gt_positive.sum())

            if condition not in bootstrap_draws:
                positive_indices = np.flatnonzero(gt_positive)
                negative_indices = np.flatnonzero(~gt_positive)
                rng = np.random.default_rng(
                    BOOTSTRAP_SEED + CONDITIONS.index(condition)
                )
                bootstrap_draws[condition] = (
                    rng.choice(
                        positive_indices,
                        size=(N_BOOTSTRAP, len(positive_indices)),
                        replace=True,
                    ),
                    rng.choice(
                        negative_indices,
                        size=(N_BOOTSTRAP, len(negative_indices)),
                        replace=True,
                    ),
                )

            balacc = balanced_accuracy_by_run(gt_mc[condition], pred_mc[condition]) * 100
            expected_mean, expected_sd = TABLE1_SEVERITY[(name, condition)]
            assert np.isclose(balacc.mean(), expected_mean, atol=0.05)
            assert np.isclose(balacc.std(ddof=1), expected_sd, atol=0.05)

            prompt_totals = frame[f"{condition}_total"].to_numpy()
            assert prompt_totals.min() >= 0 and prompt_totals.max() <= SCORE_MAX[condition], (
                f"{name} {condition}: predicted totals outside instrument range"
            )

            if not gt_checked:
                emit(f"gate 3 OK ({condition}): questionnaire totals reproduce "
                     f"the results.csv classes; (total >= 10) == (class >= "
                     f"moderate); positives "
                     f"{int(gt_positive.sum())}/{len(ids)} "
                     f"({100 * gt_positive.mean():.1f}%)")

            predicted = {
                "indicator_mean": np.floor(indicator_mean[condition] + 0.5).astype(bool),
                "class_threshold": pred_mc[condition] >= 2,
                "total_mean": total_mean[condition] >= THRESHOLD,
            }
            emit(f"\n{name}, {condition} screen positivity (total >= {THRESHOLD})")
            emit(f"  [gate 2 OK] multiclass reconstruction matches Table 1: "
                 f"{balacc.mean():.1f} +/- {balacc.std(ddof=1):.1f}")
            block = {}
            for operationalization in OPERATIONALIZATIONS:
                run_metrics = [
                    screen_metrics(gt_positive, predicted[operationalization][:, run])
                    for run in range(N_MEAN_RUNS)
                ]
                entry = {}
                for metric in ("balanced_accuracy", "sensitivity", "specificity",
                               "ppv", "predicted_positives"):
                    mean, sd = mean_sd([m[metric] for m in run_metrics])
                    entry[f"{metric}_mean"] = mean
                    entry[f"{metric}_sd"] = sd
                intervals = bootstrap_intervals(
                    predicted[operationalization], *bootstrap_draws[condition]
                )
                entry["bootstrap_95ci"] = {
                    metric: {"lower": lower, "upper": upper}
                    for metric, (lower, upper) in intervals.items()
                }
                entry["by_run"] = run_metrics
                block[operationalization] = entry
                emit(f"  [{operationalization:15s}] BA "
                     f"{entry['balanced_accuracy_mean']:5.1f} +/- "
                     f"{entry['balanced_accuracy_sd']:4.1f}  sens "
                     f"{entry['sensitivity_mean']:5.1f} +/- "
                     f"{entry['sensitivity_sd']:4.1f}  spec "
                     f"{entry['specificity_mean']:5.1f} +/- "
                     f"{entry['specificity_sd']:4.1f}  PPV "
                     f"{entry['ppv_mean']:5.1f}  pred-pos "
                     f"{entry['predicted_positives_mean']:4.1f}")
                emit(" " * 20 + "bootstrap 95% CI: " + ", ".join(
                    f"{metric} [{lower:.1f}, {upper:.1f}]"
                    for metric, (lower, upper) in intervals.items()
                ))
                for metric, (lower, upper) in intervals.items():
                    bootstrap_rows.append(
                        {
                            "model": name,
                            "condition": condition,
                            "operationalization": operationalization,
                            "metric": metric,
                            "ten_run_mean": entry[f"{metric}_mean"],
                            "bootstrap_95ci_lower": lower,
                            "bootstrap_95ci_upper": upper,
                            "bootstrap_replicates": N_BOOTSTRAP,
                            "bootstrap_seed": (
                                BOOTSTRAP_SEED + CONDITIONS.index(condition)
                            ),
                        }
                    )
                for run, metrics_run in enumerate(run_metrics):
                    csv_rows.append(
                        [name, condition, operationalization, run,
                         f"{metrics_run['balanced_accuracy']:.6f}",
                         f"{metrics_run['sensitivity']:.6f}",
                         f"{metrics_run['specificity']:.6f}",
                         f"{metrics_run['ppv']:.6f}",
                         metrics_run["predicted_positives"], ""]
                    )

            auc_by_run = [
                roc_auc(gt_positive, total_mean[condition][:, run])
                for run in range(N_MEAN_RUNS)
            ]
            auc_mean, auc_sd = mean_sd(auc_by_run)
            block["roc_auc_mean_total"] = {
                "mean": auc_mean, "sd": auc_sd, "by_run": auc_by_run,
            }
            emit(f"  [roc-auc mean-total score] {auc_mean:.1f} +/- {auc_sd:.1f}")
            for run, auc in enumerate(auc_by_run):
                csv_rows.append(
                    [name, condition, "roc_auc_mean_total", run,
                     "", "", "", "", "", f"{auc:.6f}"]
                )
            summary["screen_positivity"][f"{name}|{condition}"] = block

            all200_ind = np.floor(
                all200[f"{condition}_ind"].to_numpy() + 0.5
            ).astype(bool)
            all200_total = all200[f"{condition}_total"].to_numpy() >= THRESHOLD
            entry = {
                "indicator_mean": screen_metrics(gt_positive, all200_ind),
                "total_mean": screen_metrics(gt_positive, all200_total),
                "roc_auc_mean_total": roc_auc(
                    gt_positive, all200[f"{condition}_total"].to_numpy()
                ),
            }
            summary["all200_sensitivity"][f"{name}|{condition}"] = entry
            emit(f"  [all200 draw-free] indicator BA "
                 f"{entry['indicator_mean']['balanced_accuracy']:.1f}, "
                 f"total BA {entry['total_mean']['balanced_accuracy']:.1f}, "
                 f"AUC {entry['roc_auc_mean_total']:.1f}")
        gt_checked = True

    emit("\nRanges across models of ten-run mean values (primary "
         "indicator_mean operationalization):")
    for condition in CONDITIONS:
        for metric in ("balanced_accuracy", "sensitivity", "specificity"):
            values = [
                summary["screen_positivity"][f"{name}|{condition}"]
                ["indicator_mean"][f"{metric}_mean"]
                for name in MODELS
            ]
            emit(f"  {condition} {metric}: "
                 + " ".join(f"{value:.1f}" for value in values)
                 + f" -> range {min(values):.1f}-{max(values):.1f}")
        auc_values = [
            summary["screen_positivity"][f"{name}|{condition}"]
            ["roc_auc_mean_total"]["mean"]
            for name in MODELS
        ]
        emit(f"  {condition} roc_auc: "
             + " ".join(f"{value:.1f}" for value in auc_values)
             + f" -> range {min(auc_values):.1f}-{max(auc_values):.1f}")
    emit(f"\nSensitivity resolution: one participant = "
         f"{100 / EXPECTED_POSITIVES['anxiety']:.1f} points (anxiety), "
         f"{100 / EXPECTED_POSITIVES['depression']:.1f} points (depression)")

    table_d10_rows = []
    for condition in CONDITIONS:
        for name in MODELS:
            block = summary["screen_positivity"][f"{name}|{condition}"]
            primary = block["indicator_mean"]
            row = {"condition": condition, "model": name}
            for metric in ("balanced_accuracy", "sensitivity", "specificity"):
                row[f"{metric}_mean"] = primary[f"{metric}_mean"]
                row[f"{metric}_sd"] = primary[f"{metric}_sd"]
                row[f"{metric}_bootstrap_95ci_lower"] = primary[
                    "bootstrap_95ci"
                ][metric]["lower"]
                row[f"{metric}_bootstrap_95ci_upper"] = primary[
                    "bootstrap_95ci"
                ][metric]["upper"]
            row["roc_auc_mean"] = block["roc_auc_mean_total"]["mean"]
            row["roc_auc_sd"] = block["roc_auc_mean_total"]["sd"]
            table_d10_rows.append(row)

    (output_dir / "analysis_results.txt").write_text("\n".join(lines) + "\n")
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with (output_dir / "per_run_stats.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["model", "condition", "operationalization", "run",
             "balanced_accuracy", "sensitivity", "specificity", "ppv",
             "predicted_positives", "roc_auc"]
        )
        writer.writerows(csv_rows)
    for filename, rows in (
        ("bootstrap_intervals.csv", bootstrap_rows),
        ("table_d10_values.csv", table_d10_rows),
    ):
        with (output_dir / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nwrote {output_dir}/analysis_results.txt, analysis_summary.json, "
          f"per_run_stats.csv, bootstrap_intervals.csv, and table_d10_values.csv")


if __name__ == "__main__":
    main()
