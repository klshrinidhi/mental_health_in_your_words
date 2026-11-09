"""Disentangle in-context example selection from
prompt-ensemble aggregation in the random-vs-nearest-neighbor (RAG) comparison.

Design facts this analysis rests on (verified 2026-08-06 against the public
compiler ``public/compile_results.py`` and the RAG prompt generator
``src/fewshot_perclass_prompts_system_rag.py``):

* Random arm (Appendix Table ``tab:fewshot_mean``, N=1 row): five prompts per
  participant, each with a different randomly drawn in-context example per
  class; stored labels are mapped first, then mean-aggregated per participant
  with half-away rounding; the table reports the mean over ten seeded runs
  (run r samples 5 of the 200 stored combos globally with polars seed
  ``100*5+r``).
* RAG arm (Appendix Table ``tab:rag_fewshot_mean``): retrieval is
  deterministic, one fixed prompt per participant, scored from a single stored
  call with no aggregation of any kind (``print_metrics_nosampling``; dirs
  prompts_1_09/1_10 for GPT-4o, prompts_1_15/1_16 for Gemini, not on this
  laptop -- published table values are transcribed below as constants).
* Model snapshots match across arms (gpt-4o-2024-08-06, gemini_1.5_pro_001),
  so single-prompt comparisons are within-snapshot.

This script computes, from the frozen per-(participant x combo) outputs of the
two RAG-relevant models, the distribution of SINGLE-PROMPT performance across
all 200 stored random combos. This is a descriptive one-prompt/one-call
reference for nearest-neighbor selection; only RAG-per-class also matches the
random prompt's number of examples per class:

    Gap A (descriptive placement) = single-random mean - published RAG value
    Gap B (aggregation contrast)  = published aggregated value - single-random mean

It also makes a same-participant comparison within each of the ten published
five-prompt runs. Expected S=1 is the exact balanced accuracy expected under
uniform selection of one available constituent prompt output per participant;
the S=5 endpoint is the published mean-and-round procedure. A complete-case
sensitivity analysis restricts to participants with all five outputs and
averages over every prompt subset of sizes S=1,...,5.

Gate before any single-prompt number is trusted: the ten-run aggregated
reconstruction must reproduce all 32 published fewshot_mean N=1 cells
(Acc/F1/Pre/Rec, binary and multiclass, both models) to display rounding.

Cohort note: the RAG arm evaluates all 106 participants; each random combo
evaluates the cohort minus that combo's members (n = 102-103), because a
participant never appears in their own prompt.
"""

import itertools
import json
import sys

import numpy as np
import polars as pl
import sklearn.metrics as skm

from screen10_analysis import (
    CONDITIONS,
    DOWNLOADS,
    GT_CLASS_COUNTS,
    MC_MAP,
    N_MEAN_RUNS,
    N_SAMPLES,
    load_model,
    sha256,
    versioned_output_dir,
)

RAG_MODELS = {
    "GPT-4o": "prompts_1_11",
    "Gemini-1.5-pro": "prompts_1_12",
}
BIN_MAP = {
    "minimal": 0,
    "mild": 1,
    "moderate": 1,
    "moderately-severe": 1,
    "severe": 1,
}
TASKS = {"binary": BIN_MAP, "multiclass": MC_MAP}
TASK_LABELS = {"binary": [0, 1], "multiclass": [0, 1, 2, 3]}
METRICS = ("balacc", "f1", "pre", "rec")
EXPECTED_ROWS = 20_401
EXPECTED_COMBOS = 200
OUTPUT_BASE = DOWNLOADS / "supplementary_analysis_outputs" / "individual_prompt_icl"

# Appendix Table tab:fewshot_mean, N=1 rows (mean over 10 runs), transcribed
# from Supplementary Table 1.
PUBLISHED_AGGREGATED = {
    ("GPT-4o", "binary"): {
        "anxiety": (82.8, 83.2, 83.3, 83.2),
        "depression": (82.0, 84.4, 84.5, 84.6),
    },
    ("Gemini-1.5-pro", "binary"): {
        "anxiety": (83.4, 84.1, 84.4, 84.2),
        "depression": (81.6, 83.7, 83.7, 83.9),
    },
    ("GPT-4o", "multiclass"): {
        "anxiety": (53.5, 67.7, 72.0, 68.0),
        "depression": (61.2, 76.9, 77.0, 77.5),
    },
    ("Gemini-1.5-pro", "multiclass"): {
        "anxiety": (45.3, 64.7, 67.1, 66.2),
        "depression": (59.1, 74.6, 74.2, 75.2),
    },
}

# Appendix Table tab:rag_fewshot_mean balanced accuracies (single deterministic
# prompt, single call), transcribed from sn-article-edited.tex lines 693-723.
PUBLISHED_RAG = {
    "per_class": {
        ("GPT-4o", "binary"): {"anxiety": 78.8, "depression": 75.0},
        ("Gemini-1.5-pro", "binary"): {"anxiety": 83.6, "depression": 81.8},
        ("GPT-4o", "multiclass"): {"anxiety": 48.5, "depression": 44.6},
        ("Gemini-1.5-pro", "multiclass"): {"anxiety": 47.5, "depression": 43.2},
    },
    "all": {
        ("GPT-4o", "binary"): {"anxiety": 79.1, "depression": 81.5},
        ("Gemini-1.5-pro", "binary"): {"anxiety": 86.4, "depression": 77.2},
        ("GPT-4o", "multiclass"): {"anxiety": 59.6, "depression": 49.5},
        ("Gemini-1.5-pro", "multiclass"): {"anxiety": 44.4, "depression": 45.4},
    },
}


def mapped_frame(frame, mapping):
    return frame.with_columns(
        pl.col("anxiety").replace_strict(mapping),
        pl.col("depression").replace_strict(mapping),
        pl.col("anxiety_pd").replace_strict(mapping),
        pl.col("depression_pd").replace_strict(mapping),
    )


def metric_row(gt, pd, labels):
    """Acc/F1/Pre/Rec with the public compiler's exact sklearn kwargs."""
    kw = {"average": "weighted", "labels": labels, "zero_division": np.nan}
    return {
        "balacc": skm.balanced_accuracy_score(gt, pd),
        "f1": skm.f1_score(gt, pd, **kw),
        "pre": skm.precision_score(gt, pd, **kw),
        "rec": skm.recall_score(gt, pd, **kw),
    }


def reconstruct_aggregated(frame, task):
    """Ten-run mean-aggregation endpoint, identical to compute_*_mean_metrics.

    Returns per-condition dicts of per-run metric arrays plus the set of
    combos drawn across the published runs.
    """
    mapped = mapped_frame(frame, TASKS[task])
    train_ids = mapped["train_participant_ids"].unique().sort()
    per_run = {condition: [] for condition in CONDITIONS}
    drawn_combos = set()
    for run in range(N_MEAN_RUNS):
        selected = train_ids.sample(N_SAMPLES, seed=100 * N_SAMPLES + run)
        drawn_combos.update(selected.to_list())
        aggregated = (
            mapped.filter(pl.col("train_participant_ids").is_in(selected))
            .group_by("val_participant_id")
            .agg(
                pl.col("anxiety").first(),
                pl.col("depression").first(),
                pl.col("anxiety_pd").mean().round().cast(pl.Int64),
                pl.col("depression_pd").mean().round().cast(pl.Int64),
            )
        )
        for condition in CONDITIONS:
            per_run[condition].append(
                metric_row(
                    aggregated[condition],
                    aggregated[f"{condition}_pd"],
                    TASK_LABELS[task],
                )
            )
    return per_run, drawn_combos


def gate_aggregated(model, task, per_run):
    """All 32 published N=1 cells must reproduce to display rounding."""
    failures = []
    for condition in CONDITIONS:
        published = PUBLISHED_AGGREGATED[(model, task)][condition]
        for metric, expected in zip(METRICS, published):
            values = [row[metric] for row in per_run[condition]]
            got = round(float(np.mean(values)) * 100, 1)
            if abs(got - expected) > 0.05:
                failures.append(
                    f"{model} {task} {condition} {metric}: "
                    f"reconstructed {got} vs published {expected}"
                )
    assert not failures, "\n".join(failures)


def single_prompt_rows(model, frame, task):
    """Per-combo single-prompt metrics: one row per (condition, combo)."""
    mapped = mapped_frame(frame, TASKS[task])
    rows = []
    for (combo,), sub in mapped.group_by("train_participant_ids"):
        for condition in CONDITIONS:
            gt = sub[condition].to_numpy()
            pd = sub[f"{condition}_pd"].to_numpy()
            row = {
                "model": model,
                "task": task,
                "condition": condition,
                "combo": combo,
                "n_participants": len(gt),
                "n_gt_classes_present": int(len(np.unique(gt))),
            }
            row.update(
                {k: v * 100 for k, v in metric_row(gt, pd, TASK_LABELS[task]).items()}
            )
            rows.append(row)
    return rows


def distribution(values):
    values = np.asarray(values, dtype=float)
    pct = np.percentile(values, [2.5, 25, 50, 75, 97.5])
    return {
        "n_combos": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p2.5": float(pct[0]),
        "p25": float(pct[1]),
        "median": float(pct[2]),
        "p75": float(pct[3]),
        "p97.5": float(pct[4]),
        "max": float(values.max()),
    }


def placement(values, published):
    values = np.asarray(values, dtype=float)
    return {
        "value": published,
        "pct_combos_below": float((values < published).mean() * 100),
        "pct_combos_at_or_below": float((values <= published).mean() * 100),
    }


def fractional_balanced_accuracy(ground_truth, correctness, labels):
    """Balanced accuracy allowing fractional per-participant correctness."""
    ground_truth = np.asarray(ground_truth)
    correctness = np.asarray(correctness, dtype=float)
    class_means = []
    for label in labels:
        mask = ground_truth == label
        assert mask.any(), f"ground-truth class {label} absent"
        class_means.append(float(correctness[mask].mean()))
    return float(np.mean(class_means) * 100)


def same_participant_rows(model, frame, task):
    """Compare S=5 with exact expected S=1 on the same participants."""
    mapped = mapped_frame(frame, TASKS[task])
    train_ids = mapped["train_participant_ids"].unique().sort()
    labels = TASK_LABELS[task]
    rows = []
    for run in range(N_MEAN_RUNS):
        selected = train_ids.sample(N_SAMPLES, seed=100 * N_SAMPLES + run)
        selected_frame = mapped.filter(
            pl.col("train_participant_ids").is_in(selected)
        )
        for condition in CONDITIONS:
            grouped = (
                selected_frame.group_by("val_participant_id")
                .agg(
                    pl.col(condition).first().alias("ground_truth"),
                    pl.col(f"{condition}_pd").alias("predictions"),
                )
                .sort("val_participant_id")
            )
            ground_truth = grouped["ground_truth"].to_numpy()
            prediction_lists = grouped["predictions"].to_list()
            counts = np.asarray([len(values) for values in prediction_lists])

            expected_single_correctness = np.asarray(
                [
                    np.mean(np.asarray(values) == label)
                    for values, label in zip(prediction_lists, ground_truth)
                ]
            )
            five_prompt_predictions = np.asarray(
                [np.floor(np.mean(values) + 0.5) for values in prediction_lists],
                dtype=int,
            )
            expected_single = fractional_balanced_accuracy(
                ground_truth, expected_single_correctness, labels
            )
            five_prompt = fractional_balanced_accuracy(
                ground_truth,
                five_prompt_predictions == ground_truth,
                labels,
            )

            complete_mask = counts == N_SAMPLES
            complete_ground_truth = ground_truth[complete_mask]
            complete_predictions = np.asarray(
                [
                    prediction_lists[index]
                    for index in np.flatnonzero(complete_mask)
                ],
                dtype=int,
            )
            assert complete_predictions.shape[1] == N_SAMPLES
            curve = {}
            for size in range(1, N_SAMPLES + 1):
                subset_scores = []
                for subset in itertools.combinations(range(N_SAMPLES), size):
                    subset_predictions = np.floor(
                        complete_predictions[:, subset].mean(axis=1) + 0.5
                    ).astype(int)
                    subset_scores.append(
                        fractional_balanced_accuracy(
                            complete_ground_truth,
                            subset_predictions == complete_ground_truth,
                            labels,
                        )
                    )
                curve[size] = float(np.mean(subset_scores))

            row = {
                "model": model,
                "task": task,
                "condition": condition,
                "run": run,
                "participants": len(ground_truth),
                "complete_case_participants": int(complete_mask.sum()),
                "expected_single_prompt_balacc": expected_single,
                "five_prompt_balacc": five_prompt,
                "five_minus_expected_single": five_prompt - expected_single,
                "complete_single_prompt_balacc": curve[1],
                "complete_five_prompt_balacc": curve[N_SAMPLES],
                "complete_five_minus_single": curve[N_SAMPLES] - curve[1],
            }
            for size, value in curve.items():
                row[f"complete_balacc_s{size}"] = value
            rows.append(row)
    return rows


def summarize_same_participant(rows):
    """Summarize the ten published runs for one model/task/condition cell."""
    fields = (
        "expected_single_prompt_balacc",
        "five_prompt_balacc",
        "five_minus_expected_single",
        "complete_single_prompt_balacc",
        "complete_five_prompt_balacc",
        "complete_five_minus_single",
    )
    result = {"runs": len(rows)}
    for field in fields:
        values = np.asarray([row[field] for row in rows], dtype=float)
        result[f"{field}_mean"] = float(values.mean())
        result[f"{field}_sd"] = float(values.std(ddof=1))
    result["runs_with_positive_primary_delta"] = int(
        sum(row["five_minus_expected_single"] > 0 for row in rows)
    )
    result["complete_case_participants_min"] = min(
        row["complete_case_participants"] for row in rows
    )
    result["complete_case_participants_max"] = max(
        row["complete_case_participants"] for row in rows
    )
    result["complete_case_curve"] = {
        f"s{size}": float(
            np.mean([row[f"complete_balacc_s{size}"] for row in rows])
        )
        for size in range(1, N_SAMPLES + 1)
    }
    return result


def main():
    output_dir = versioned_output_dir(OUTPUT_BASE)
    print(f"writing to {output_dir}")

    provenance = {
        "inputs": {},
        "packages": {
            "python": sys.version.split()[0],
            "polars": pl.__version__,
            "numpy": np.__version__,
            "sklearn": skm.__name__ and __import__("sklearn").__version__,
        },
        "gates": [],
        "notes": [
            "RAG arm provenance: public/compile_results.py scores all four RAG "
            "blocks with print_metrics_nosampling (single stored call per "
            "participant, no aggregation); prompt generator writes one "
            "deterministic prompt file per participant.",
            "Cohort asymmetry: RAG cells cover all 106 evaluated participants; "
            "each random combo covers the cohort minus that combo's members.",
            "tex inconsistency for the writing phase: the Results RAG paragraph "
            "quotes the Gemini binary-anxiety random baseline as 83.5 in the "
            "RAG-all sentence; tab:fewshot_mean(b) N=1 says 83.4.",
        ],
    }

    combo_rows = []
    sensitivity_rows = []
    summary = {}
    for model, prompts_dir in RAG_MODELS.items():
        frame = load_model(DOWNLOADS / prompts_dir)
        provenance["inputs"][model] = {
            "results_csv": str(DOWNLOADS / prompts_dir / "results.csv"),
            "sha256": sha256(DOWNLOADS / prompts_dir / "results.csv"),
        }
        assert frame.height == EXPECTED_ROWS, (model, frame.height)
        combos = frame["train_participant_ids"].unique().sort().to_list()
        assert len(combos) == EXPECTED_COMBOS, (model, len(combos))
        summary.setdefault("_combo_grid", combos)
        assert summary["_combo_grid"] == combos, f"{model}: combo grid differs"

        gt_counts = (
            frame.unique("val_participant_id")
            .select(pl.col("anxiety").replace_strict(MC_MAP), pl.col("depression").replace_strict(MC_MAP))
        )
        for condition in CONDITIONS:
            observed = tuple(
                int((gt_counts[condition] == k).sum()) for k in range(4)
            )
            assert observed == GT_CLASS_COUNTS[condition], (condition, observed)

        for task in TASKS:
            per_run, drawn_combos = reconstruct_aggregated(frame, task)
            gate_aggregated(model, task, per_run)
            provenance["gates"].append(
                f"{model} {task}: 8 published N=1 cells x (Acc,F1,Pre,Rec) "
                "reproduced to display rounding"
            )
            rows = single_prompt_rows(model, frame, task)
            combo_rows.extend(rows)
            task_sensitivity_rows = same_participant_rows(model, frame, task)
            sensitivity_rows.extend(task_sensitivity_rows)

            for condition in CONDITIONS:
                cond_rows = [r for r in rows if r["condition"] == condition]
                balaccs = [r["balacc"] for r in cond_rows]
                drawn_balaccs = [
                    r["balacc"] for r in cond_rows if r["combo"] in drawn_combos
                ]
                aggregated_reconstructed = float(
                    np.mean([row["balacc"] for row in per_run[condition]]) * 100
                )
                aggregated_published = PUBLISHED_AGGREGATED[(model, task)][condition][0]
                single_mean = float(np.mean(balaccs))
                cell = {
                    "single_prompt_balacc": distribution(balaccs),
                    "single_prompt_balacc_published_draws": distribution(drawn_balaccs),
                    "aggregated_published": aggregated_published,
                    "aggregated_reconstructed": aggregated_reconstructed,
                    "gap_b_aggregation_benefit": aggregated_published - single_mean,
                    "rag": {},
                    "combos_missing_gt_class": int(
                        sum(
                            r["n_gt_classes_present"] < len(TASK_LABELS[task])
                            for r in cond_rows
                        )
                    ),
                    "same_participant": summarize_same_participant(
                        [
                            row
                            for row in task_sensitivity_rows
                            if row["condition"] == condition
                        ]
                    ),
                }
                same_s5 = cell["same_participant"]["five_prompt_balacc_mean"]
                assert round(same_s5, 1) == aggregated_published, (
                    model,
                    task,
                    condition,
                    same_s5,
                    aggregated_published,
                )
                for variant in PUBLISHED_RAG:
                    rag_value = PUBLISHED_RAG[variant][(model, task)][condition]
                    cell["rag"][variant] = placement(balaccs, rag_value)
                    cell["rag"][variant]["gap_a_single_mean_minus_rag"] = (
                        single_mean - rag_value
                    )
                summary[f"{model}|{task}|{condition}"] = cell

    combo_frame = pl.DataFrame(combo_rows).sort(["model", "task", "condition", "combo"])
    combo_frame.write_csv(output_dir / "single_prompt_combo_metrics.csv")
    pl.DataFrame(sensitivity_rows).sort(
        ["model", "task", "condition", "run"]
    ).write_csv(output_dir / "same_participant_aggregation.csv")
    summary.pop("_combo_grid")
    (output_dir / "analysis_summary.json").write_text(
        json.dumps({"provenance": provenance, "cells": summary}, indent=2)
    )

    lines = [
        "# Single-prompt random vs nearest-neighbor (RAG) selection",
        "",
        "Single-prompt = one random N=1-per-class prompt, one stored call, no",
        "aggregation; distribution over all 200 stored combos. RAG values are the",
        "published single-deterministic-prompt appendix numbers. Aggregated =",
        "published fewshot_mean N=1 (5-prompt mean aggregation, mean of 10 runs).",
        "",
        "| Model | Task | Condition | single mean±sd [min–max] | agg | Gap B | RAG-pc (pct below) | RAG-all (pct below) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, cell in summary.items():
        model, task, condition = key.split("|")
        dist = cell["single_prompt_balacc"]
        pc = cell["rag"]["per_class"]
        ra = cell["rag"]["all"]
        lines.append(
            f"| {model} | {task} | {condition} "
            f"| {dist['mean']:.1f}±{dist['sd']:.1f} [{dist['min']:.1f}–{dist['max']:.1f}] "
            f"| {cell['aggregated_published']:.1f} "
            f"| {cell['gap_b_aggregation_benefit']:+.1f} "
            f"| {pc['value']:.1f} ({pc['pct_combos_below']:.0f}%) "
            f"| {ra['value']:.1f} ({ra['pct_combos_below']:.0f}%) |"
        )
    lines += [
        "",
        "Gap B = published aggregated minus single-prompt mean (descriptive ensemble contrast).",
        "(pct below) = percent of the 200 single random prompts scoring below the",
        "RAG value; 50% means the NN prompt performs like the median random draw.",
        "",
        "## Same-participant aggregation sensitivity",
        "",
        "Expected S=1 selects one available constituent output uniformly per",
        "participant; S=5 is mean-and-round. Complete-case values restrict to",
        "participants with all five outputs.",
        "",
        "| Model | Task | Condition | expected S=1 | S=5 | delta | complete S=1 | complete S=5 | delta |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, cell in summary.items():
        model, task, condition = key.split("|")
        same = cell["same_participant"]
        lines.append(
            f"| {model} | {task} | {condition} "
            f"| {same['expected_single_prompt_balacc_mean']:.1f} "
            f"| {same['five_prompt_balacc_mean']:.1f} "
            f"| {same['five_minus_expected_single_mean']:+.1f} "
            f"| {same['complete_single_prompt_balacc_mean']:.1f} "
            f"| {same['complete_five_prompt_balacc_mean']:.1f} "
            f"| {same['complete_five_minus_single_mean']:+.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
