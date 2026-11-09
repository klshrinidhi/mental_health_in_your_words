"""Test item-profile association beyond overall transcript severity.

This analysis asks whether per-item GAD-7/PHQ-9 predictions carry
participant-specific item information beyond overall transcript severity.

Under that mechanism, once a participant's overall predicted level and the
items' population base rates are accounted for, the *pattern* of which items
are elevated for a given participant should carry no information about the
ground-truth item pattern.

DESIGN
  Endpoint: the published item-wise endpoint (Fig. gad7_phq9_itemwise), i.e.
  for each of the 10 published runs (polars seeded sample of 5 of the 200 ICL
  combos, seed = 100*5 + run), each participant's item answer is the rounded
  mean of the item score over the run's 5 calls. The analysis matrix P
  (participants x items) averages these rounded per-run answers over the 10
  runs; G is the ground-truth item matrix from the questionnaires.

  Primary test (per model x questionnaire, 8 tests): residualize P and G by
  removing each participant's own mean (the global severity level) AND each
  severity-class x item cell mean (each item's response curve
  at class resolution -- base rates and how each item rises with severity),
  then correlate the residual matrices. Null distribution by permuting which
  participant's predicted profile is paired with which participant's
  ground-truth profile WITHIN ground-truth severity classes (Mantel-style
  row-pairing permutation, class-stratified). Both projections commute with
  within-class row permutation, so the test is exact under the null that
  predictions depend on the participant only through overall severity class
  (plus any participant-additive shift): under that null the residuals carry
  no participant-specific item information. Plain double-centering alone is
  NOT sufficient -- it removes only an additive severity effect, and an item
  whose prediction is constant (e.g. PHQ-9 Q9 predicted 0 for everyone)
  acquires a residual equal to minus the participant's mean severity, which
  correlates with the ground-truth residual of any similarly rare item purely
  through the global channel. The class-conditional projection absorbs those
  item-specific severity response curves. Two-sided p-values with the +1
  correction; Holm across the 8 primary tests.

  Reported alongside: the additive-only adjustment (plain double-centering,
  unrestricted permutation) for transparency, and a strictest variant with
  exact ground-truth total-score strata (cell means and permutations within
  groups of identical questionnaire totals; singleton groups contribute
  nothing), under which no function of the true total -- however nonlinear --
  can explain a positive result.

  Sensitivities: (a) same test on the unrounded per-run means (rounding not
  driving the result); (b) same test on the unrounded mean over ALL 200 ICL
  combos per participant (no run sampling at all, so independent of any
  sampler/seed provenance).

  Secondary (descriptive resolution): per-item residual correlations with the
  same permutation null, Holm within each model x questionnaire family; raw
  per-item Spearman; matched vs mismatched mean residual column correlations
  (does predicted item i track true item i specifically, or any item equally);
  per-run observed residual correlations (run stability); predicted-total vs
  ground-truth-total Spearman (the undisputed global channel, for context).

  Validation: inputs are SHA-256-chained to the paired-comparison analysis state,
  under which Table 1c/d (the paper's primary endpoint, same run machinery)
  reproduces digit-perfect with polars 1.19.0. The published item-wise figure
  values (GAD-7 model means 59.2/59.4/61.2/59.6, PHQ-9 71.7/70.3/71.4/69.8) do
  NOT reproduce exactly from the current results.csv under any polars sampler
  behavior (two distinct seeded-draw behaviors exist across polars 1.0-1.43;
  neither matches, nor do alternative seed formulas or aggregation chains) —
  the figure session evidently used a slightly earlier results.csv (rare
  unparseable-call reruns shift item scores by a few tenths of a point).
  Computed-now model means are asserted within 1.0 percentage point of the
  published values and the deviations are reported; conclusions do not depend
  on which snapshot underlies the figure (see the all-200-combo sensitivity).

Outputs are versioned and non-overwriting under
DATA_DIR/supplementary_analysis_outputs/item_profile_association_vNN, with
SHA-256 hashes of all inputs.
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import scipy
import scipy.stats as stats

assert pl.__version__ == "1.19.0", (
    f"polars must be 1.19.0 to reproduce the published seeded runs, got {pl.__version__}"
)

N_SAMPLES = 5
N_RUNS = 10

# (display name, prompts dir, results column prefix, prompt id) in manuscript order
MODELS = [
    ("GPT-4o", "prompts_1_11", "gpt-4o-2024-08-06", "prompt_1_11"),
    ("Gemini-1.5-pro", "prompts_1_12", "gemini_1.5_pro_001", "prompt_1_12"),
    ("Llama-3.2-90b", "prompts_1_13", "llama_3.2_90b_vision_instruct", "prompt_1_13"),
    ("MedLM-Large-1.5", "prompts_1_17", "medlm_large_1.5", "prompt_1_17"),
]

QUESTIONNAIRES = {
    "GAD-7": [f"GAD7_Q{i}" for i in range(1, 8)],
    "PHQ-9": [f"PHQ9_Q{i}" for i in range(1, 10)],
}

# Published model-wise mean item accuracies (manuscript, Item-wise paragraph)
PUBLISHED_MEAN_ACC = {
    ("GPT-4o", "GAD-7"): 59.2, ("Gemini-1.5-pro", "GAD-7"): 59.4,
    ("Llama-3.2-90b", "GAD-7"): 61.2, ("MedLM-Large-1.5", "GAD-7"): 59.6,
    ("GPT-4o", "PHQ-9"): 71.7, ("Gemini-1.5-pro", "PHQ-9"): 70.3,
    ("Llama-3.2-90b", "PHQ-9"): 71.4, ("MedLM-Large-1.5", "PHQ-9"): 69.8,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def holm(pairs):
    """pairs: list of (key, p). Returns dict key -> Holm-adjusted p."""
    m = len(pairs)
    order = sorted(range(m), key=lambda i: pairs[i][1])
    adj = {}
    running = 0.0
    for rank, i in enumerate(order):
        key, p = pairs[i]
        running = max(running, min(1.0, (m - rank) * p))
        adj[key] = running
    return adj


def double_center(x):
    return x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + x.mean()


def corr_flat(a, b):
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / den if den > 0 else float("nan")


def load_item_frames(data_dir, prompts_dir, col_prefix, prompt_id):
    """Replicates the notebook's compute_gad7_phq9_itemwise_mean_metrics data prep."""
    r = pl.read_csv(data_dir / prompts_dir / "results.csv")
    anx_col = f"{col_prefix}\nanxiety scores\n{prompt_id}"
    dep_col = f"{col_prefix}\ndepression scores\n{prompt_id}"
    df = (
        r.select("val_participant_id", "train_participant_ids", anx_col, dep_col)
        .rename({anx_col: "anxiety_pd", dep_col: "depression_pd"})
        .with_columns(pl.col("anxiety_pd").str.split_exact(",", 6).alias("gad7_pd"))
        .unnest("gad7_pd")
        .rename({f"field_{i}": f"GAD7_Q{i+1}_pd" for i in range(7)})
        .with_columns([pl.col(f"GAD7_Q{i+1}_pd").cast(pl.Int64) for i in range(7)])
        .with_columns(pl.col("depression_pd").str.split_exact(",", 8).alias("phq9_pd"))
        .unnest("phq9_pd")
        .rename({f"field_{i}": f"PHQ9_Q{i+1}_pd" for i in range(9)})
        .with_columns([pl.col(f"PHQ9_Q{i+1}_pd").cast(pl.Int64) for i in range(9)])
        .drop("anxiety_pd", "depression_pd")
    )
    q = pl.read_csv(data_dir / "questionnaire_data_clean.csv")
    item_cols = QUESTIONNAIRES["GAD-7"] + QUESTIONNAIRES["PHQ-9"]
    q = q.select(["participant_id", *item_cols, "GAD7_score", "PHQ9_score"])
    df = df.join(q, left_on="val_participant_id", right_on="participant_id", how="inner")
    return df


def per_run_matrices(df, items):
    """Per published run: rounded and unrounded per-participant item answers.

    Returns (participant_ids, per-run accuracy array [runs x items],
    rounded matrices [runs x n x K], unrounded matrices [runs x n x K],
    gt matrix [n x K]).
    """
    combos = df["train_participant_ids"].unique().sort()
    pid = None
    accs, rounded, unrounded = [], [], []
    gt = None
    for i_run in range(N_RUNS):
        sel = combos.sample(N_SAMPLES, seed=100 * N_SAMPLES + i_run)
        sel = list(sel.implode()[0])
        dfr = df.filter(pl.col("train_participant_ids").is_in(sel))
        agg = (
            dfr.group_by("val_participant_id")
            .agg(
                [pl.col(c).first() for c in items]
                + [pl.col(f"{c}_pd").mean().round().cast(pl.Int64).alias(f"{c}_rd") for c in items]
                + [pl.col(f"{c}_pd").mean().alias(f"{c}_mu") for c in items]
            )
            .sort("val_participant_id")
        )
        ids = agg["val_participant_id"].to_list()
        if pid is None:
            pid = ids
        assert ids == pid, "participant set differs between runs"
        g = agg.select(items).to_numpy().astype(float)
        if gt is None:
            gt = g
        assert np.array_equal(g, gt)
        rd = agg.select([f"{c}_rd" for c in items]).to_numpy().astype(float)
        mu = agg.select([f"{c}_mu" for c in items]).to_numpy().astype(float)
        accs.append([(rd[:, k] == gt[:, k]).mean() for k in range(len(items))])
        rounded.append(rd)
        unrounded.append(mu)
    return pid, np.array(accs), np.array(rounded), np.array(unrounded), gt


def project_conditional(X, strata):
    """Remove participant (row) means, then stratum x item cell means.

    With a single stratum this equals plain double-centering. Both steps
    commute with within-stratum row permutation, so permuting rows of the
    projected matrix within strata yields an exact test of the null that rows
    depend on the participant only through stratum plus an additive shift.
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    out = Xc.copy()
    for s in np.unique(strata):
        idx = strata == s
        out[idx] -= Xc[idx].mean(axis=0, keepdims=True)
    return out


def stratified_perms(rng, strata, m):
    """m row-permutations that shuffle only within strata."""
    n = len(strata)
    perms = np.tile(np.arange(n), (m, 1))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        if len(idx) > 1:
            local = rng.permuted(np.tile(np.arange(len(idx)), (m, 1)), axis=1)
            perms[:, idx] = idx[local]
    return perms


def scheme_tests(variants, G, strata, rng, n_perm, chunk, per_item=False):
    """Row-pairing permutation tests under one projection/stratification scheme.

    variants: dict name -> predicted matrix (raw); all share the projection
    defined by `strata` and one stream of within-stratum permutations.
    """
    n, K = G.shape
    strata = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    Gp = project_conditional(G, strata)
    Ps = {k: project_conditional(v, strata) for k, v in variants.items()}
    den_G = float((Gp**2).sum())
    r_obs = {k: corr_flat(P, Gp) for k, P in Ps.items()}
    dens = {k: math.sqrt(float((P**2).sum()) * den_G) for k, P in Ps.items()}
    cnts = {k: 0 for k in Ps}

    first_key = next(iter(Ps))
    P0 = Ps[first_key] if per_item else None
    if per_item:
        cn_p = np.sqrt((P0**2).sum(axis=0))
        cn_g = np.sqrt((Gp**2).sum(axis=0))
        valid = (cn_p > 0) & (cn_g > 0)
        r_items = np.full(K, np.nan)
        r_items[valid] = (P0 * Gp).sum(axis=0)[valid] / (cn_p * cn_g)[valid]
        cnt_items = np.zeros(K, dtype=np.int64)

    eps = 1e-12
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        perms = stratified_perms(rng, strata, m)
        for k, P in Ps.items():
            num = np.zeros(m)
            for i in range(K):
                d = P[:, i][perms] @ Gp[:, i]
                num += d
                if per_item and k == first_key and valid[i]:
                    r_i = d / (cn_p[i] * cn_g[i])
                    cnt_items[i] += int(np.count_nonzero(np.abs(r_i) >= abs(r_items[i]) - eps))
            cnts[k] += int(np.count_nonzero(np.abs(num / dens[k]) >= abs(r_obs[k]) - eps))
        done += m

    out = {k: {"r": r_obs[k], "p": (1 + cnts[k]) / (1 + n_perm)} for k in Ps}
    if per_item:
        p_items = np.full(K, np.nan)
        p_items[valid] = (1 + cnt_items[valid]) / (1 + n_perm)
        # matched vs mismatched residual column correlations
        Pn = np.where(cn_p > 0, cn_p, 1.0)
        Gn = np.where(cn_g > 0, cn_g, 1.0)
        C = (P0 / Pn).T @ (Gp / Gn)
        vv = np.outer(valid, valid)
        diag = np.eye(K, dtype=bool)
        out["per_item"] = {
            "r_items": r_items, "p_items": p_items, "valid": valid,
            "matched_mean_r": float(C[diag & vv].mean()),
            "mismatched_mean_r": float(C[~diag & vv].mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path.home() / "Downloads" / "mental_health"))
    ap.add_argument("--output-base", default=None,
                    help=("default: DATA_DIR/supplementary_analysis_outputs "
                          "(versioned item_profile_association_vNN inside)"))
    ap.add_argument("--n-perm", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--chunk", type=int, default=100_000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_base = (
        Path(args.output_base)
        if args.output_base
        else data_dir / "supplementary_analysis_outputs"
    )
    out_base.mkdir(parents=True, exist_ok=True)
    version = 1 + max(
        [int(p.name.rsplit("_v", 1)[1])
         for p in out_base.glob("item_profile_association_v*") if p.is_dir()],
        default=0,
    )
    out_dir = out_base / f"item_profile_association_v{version:02d}"
    out_dir.mkdir()

    inputs = {"questionnaire_data_clean.csv": sha256(data_dir / "questionnaire_data_clean.csv")}
    for _, pdir, _, _ in MODELS:
        inputs[f"{pdir}/results.csv"] = sha256(data_dir / pdir / "results.csv")

    rng = np.random.default_rng(args.seed)
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("Item-profile association beyond overall transcript severity")
    emit(f"endpoint: published item-wise runs ({N_RUNS} runs x rounded mean of {N_SAMPLES} calls)")
    emit(f"Monte Carlo permutations: {args.n_perm:,} (row-pairing); seed: {args.seed}; two-sided, +1-corrected")
    emit(f"software: Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__}, Polars {pl.__version__}")
    emit()

    # -------- load, validate, build matrices --------
    blocks = {}  # (model, questionnaire) -> dict
    pid_ref = None
    for name, pdir, prefix, pid_col in MODELS:
        df = load_item_frames(data_dir, pdir, prefix, pid_col)
        # ground-truth integrity: totals equal item sums
        chk = df.group_by("val_participant_id").agg(
            [pl.col(c).first() for c in
             QUESTIONNAIRES["GAD-7"] + QUESTIONNAIRES["PHQ-9"] + ["GAD7_score", "PHQ9_score"]]
        )
        g7 = chk.select(QUESTIONNAIRES["GAD-7"]).to_numpy().sum(axis=1)
        p9 = chk.select(QUESTIONNAIRES["PHQ-9"]).to_numpy().sum(axis=1)
        assert np.array_equal(g7, chk["GAD7_score"].to_numpy()), f"{name}: GAD7 totals != item sums"
        assert np.array_equal(p9, chk["PHQ9_score"].to_numpy()), f"{name}: PHQ9 totals != item sums"
        for qname, items in QUESTIONNAIRES.items():
            pid, accs, rounded, unrounded, gt = per_run_matrices(df, items)
            if pid_ref is None:
                pid_ref = pid
            assert pid == pid_ref, "participant sets differ across models"
            assert len(pid) == 106, f"expected 106 participants, got {len(pid)}"
            # provenance check against the published figure values (see docstring)
            acc_chain = np.round(accs * 100, 1)          # notebook rounds per-run accs to 0.1
            per_item_mean = acc_chain.mean(axis=0)       # figure bars: mean over runs
            model_mean = float(per_item_mean.mean())
            pub = PUBLISHED_MEAN_ACC[(name, qname)]
            assert abs(model_mean - pub) <= 1.0, (
                f"{name} {qname}: computed mean item accuracy {model_mean:.3f} "
                f"is implausibly far from published {pub}"
            )
            # aggregation-free matrix: unrounded mean over all 200 combos
            agg_all = (
                df.group_by("val_participant_id")
                .agg([pl.col(f"{c}_pd").mean().alias(f"{c}_a") for c in items])
                .sort("val_participant_id")
            )
            assert agg_all["val_participant_id"].to_list() == pid
            all200 = agg_all.select([f"{c}_a" for c in items]).to_numpy().astype(float)
            blocks[(name, qname)] = {
                "accs": accs, "per_item_mean_acc": per_item_mean,
                "model_mean_acc": model_mean, "published_mean_acc": pub,
                "rounded": rounded, "unrounded": unrounded, "all200": all200,
                "gt": gt, "items": items,
            }
    # canonical-state chain: same inputs as the paired-comparison analysis under which
    # Table 1c/d reproduced digit-perfect (same run machinery as used here)
    c03 = out_base / "paired_model_comparisons" / "analysis_summary.json"
    if c03.exists():
        c03_hashes = json.load(open(c03)).get("input_sha256", {})
        shared = [k for k in inputs if k in c03_hashes]
        mismatches = [k for k in shared if c03_hashes[k] != inputs[k]]
        assert shared and not mismatches, (
            f"SHA chain to paired comparisons failed: shared={shared}, mismatches={mismatches}"
        )
        emit(f"validation: input hashes match paired comparisons (Table 1c/d digit-perfect state); "
             f"{len(shared)}/{len(inputs)} files chained")
    else:
        emit("validation: paired-comparison summary not found; SHA chain skipped")
    emit("provenance: published item-wise figure values do not reproduce exactly from the")
    emit("current results.csv under any polars sampler behavior (figure session likely used")
    emit("a pre-rerun snapshot); computed-now model means and deviations:")
    for (name, qname), b in blocks.items():
        emit(f"  {name} {qname}: computed {b['model_mean_acc']:.2f} vs published {b['published_mean_acc']:.1f} "
             f"(diff {b['model_mean_acc'] - b['published_mean_acc']:+.2f} pp)")
    for qname in QUESTIONNAIRES:
        vals = [b["per_item_mean_acc"] for (m, q), b in blocks.items() if q == qname]
        allv = np.concatenate(vals)
        emit(f"  {qname}: per-item 10-run mean accuracies span {allv.min():.1f}-{allv.max():.1f} "
             f"across models (manuscript: {'50-77' if qname == 'GAD-7' else '52-97'})")
    emit()

    # -------- tests --------
    CLASS_COUNTS = {"GAD-7": [60, 24, 12, 10], "PHQ-9": [68, 27, 8, 3]}
    primary = []
    per_item_rows = []
    summary_blocks = {}
    for (name, qname), b in blocks.items():
        P_r = b["rounded"].mean(axis=0)     # mean over runs of rounded per-run answers
        P_u = b["unrounded"].mean(axis=0)
        G = b["gt"]
        tot = G.sum(axis=1).astype(int)
        cls = np.digitize(tot, [5, 10, 15])      # published severity classes (severe merged)
        counts = [int((cls == c).sum()) for c in range(4)]
        assert counts == CLASS_COUNTS[qname], f"{qname} class counts {counts} != published {CLASS_COUNTS[qname]}"
        res = {}
        # primary scheme: class-conditional projection, within-class permutation
        res["cls"] = scheme_tests(
            {"rounded": P_r, "unrounded": P_u, "all200": b["all200"]},
            G, cls, rng, args.n_perm, args.chunk, per_item=True)
        # transparency: additive-only adjustment (plain double-centering)
        res["add"] = scheme_tests({"rounded": P_r}, G, None, rng, args.n_perm, args.chunk)
        # strictest: exact-total strata (singletons contribute nothing)
        res["tot"] = scheme_tests({"rounded": P_r}, G, tot, rng, args.n_perm, args.chunk)
        n_singleton = int(sum(1 for t in np.unique(tot) if (tot == t).sum() == 1))
        res["tot_strata"] = {"n_strata": int(len(np.unique(tot))), "n_singleton": n_singleton}
        # per-run observed r under the primary projection (stability)
        res["r_runs"] = [corr_flat(project_conditional(b["rounded"][r], cls), project_conditional(G, cls))
                         for r in range(N_RUNS)]
        res["rho_total"] = float(stats.spearmanr(P_r.sum(axis=1), tot).statistic)
        res["raw_spearman"] = [
            stats.spearmanr(P_r[:, k], G[:, k]).statistic
            if np.std(G[:, k]) > 0 and np.std(P_r[:, k]) > 0 else float("nan")
            for k in range(len(b["items"]))]
        summary_blocks[(name, qname)] = res
        primary.append(((name, qname), res["cls"]["rounded"]["p"]))

    holm_primary = holm(primary)

    for (name, qname), res in summary_blocks.items():
        b = blocks[(name, qname)]
        pi = res["cls"]["per_item"]
        emit(f"PRIMARY: item-profile agreement beyond severity class, {name}, {qname}")
        emit("  unit: participant (n=106); statistic: correlation of predicted vs ground-truth")
        emit("  item matrices after removing participant means and severity-class x item cell")
        emit("  means; null: row-pairing permutation within ground-truth severity classes")
        emit(f"  r={res['cls']['rounded']['r']:+.4f}, p={res['cls']['rounded']['p']:.6f}, "
             f"Holm p (8 tests)={holm_primary[(name, qname)]:.6f}")
        emit(f"  sensitivity (unrounded per-run means): r={res['cls']['unrounded']['r']:+.4f}, p={res['cls']['unrounded']['p']:.6f}")
        emit(f"  sensitivity (all 200 combos, no run sampling): r={res['cls']['all200']['r']:+.4f}, p={res['cls']['all200']['p']:.6f}")
        emit(f"  additive-only adjustment (double-centering, unrestricted perms): "
             f"r={res['add']['rounded']['r']:+.4f}, p={res['add']['rounded']['p']:.6f}")
        emit(f"  exact-total strata ({res['tot_strata']['n_strata']} strata, {res['tot_strata']['n_singleton']} singleton): "
             f"r={res['tot']['rounded']['r']:+.4f}, p={res['tot']['rounded']['p']:.6f}")
        emit(f"  per-run observed r: min={min(res['r_runs']):+.4f}, max={max(res['r_runs']):+.4f}")
        emit(f"  matched-item mean residual r={pi['matched_mean_r']:+.4f} vs mismatched {pi['mismatched_mean_r']:+.4f}")
        emit(f"  predicted-total vs true-total Spearman rho={res['rho_total']:+.4f} (global channel, context)")
        item_pairs = [(k, pi["p_items"][k]) for k in range(len(b["items"])) if pi["valid"][k]]
        holm_items = holm(item_pairs)
        for k, item in enumerate(b["items"]):
            if not pi["valid"][k]:
                emit(f"    {item}: degenerate (no residual variance), skipped")
                continue
            hp = holm_items[k]
            rs = res["raw_spearman"][k]
            rs_txt = f"{rs:+.3f}" if not math.isnan(rs) else "n/a (constant prediction)"
            emit(f"    {item}: acc={b['per_item_mean_acc'][k]:.1f}%, raw Spearman rho={rs_txt}, "
                 f"residual r={pi['r_items'][k]:+.3f}, p={pi['p_items'][k]:.6f}, Holm p (within family)={hp:.6f}")
            per_item_rows.append({
                "model": name, "questionnaire": qname, "item": item,
                "mean_acc_pct": round(float(b["per_item_mean_acc"][k]), 1),
                "raw_spearman": None if math.isnan(rs) else round(float(rs), 4),
                "residual_r": round(float(pi["r_items"][k]), 4),
                "p_raw": pi["p_items"][k], "p_holm_within": hp,
            })
        emit()

    n_sig = sum(1 for k, p in holm_primary.items() if p < 0.05)
    emit(f"SUMMARY: {n_sig}/8 primary class-conditional profile tests significant after Holm across the 8 tests")
    emit()

    # -------- persist --------
    (out_dir / "analysis_results.txt").write_text("\n".join(lines) + "\n")
    pl.DataFrame(per_item_rows).write_csv(out_dir / "per_item_stats.csv")
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "n_perm": args.n_perm,
        "inputs_sha256": inputs,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "polars": pl.__version__},
        "blocks": {
            f"{name}|{qname}": {
                "primary_class_conditional": {
                    "r": res["cls"]["rounded"]["r"], "p_raw": res["cls"]["rounded"]["p"],
                    "p_holm_8": holm_primary[(name, qname)],
                },
                "sensitivity_unrounded": res["cls"]["unrounded"],
                "sensitivity_all200": res["cls"]["all200"],
                "additive_only": res["add"]["rounded"],
                "exact_total_strata": {**res["tot"]["rounded"], **res["tot_strata"]},
                "r_runs": res["r_runs"], "rho_total": res["rho_total"],
                "matched_mean_r": res["cls"]["per_item"]["matched_mean_r"],
                "mismatched_mean_r": res["cls"]["per_item"]["mismatched_mean_r"],
                "model_mean_acc": blocks[(name, qname)]["model_mean_acc"],
                "published_mean_acc": blocks[(name, qname)]["published_mean_acc"],
            }
            for (name, qname), res in summary_blocks.items()
        },
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\noutputs written to {out_dir}")


if __name__ == "__main__":
    main()
