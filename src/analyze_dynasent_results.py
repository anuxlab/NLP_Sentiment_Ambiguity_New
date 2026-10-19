#!/usr/bin/env python3
"""Reproducible statistical validation for DynaSent multi-seed prediction artifacts.

Primary contrast: human_soft_targets - majority_ce in mean per-example JS divergence,
separately by round. Bootstrap resamples test IDs while holding trained seeds fixed;
permutation p-values are example-level and conditional on these trained models. Since
these runs have already been inspected, outputs explicitly mark inferential results as
exploratory rather than confirmatory.
"""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
import numpy as np
import pandas as pd

PRIMARY_A = "human_soft_targets"
PRIMARY_B = "majority_ce"
CONDITIONS = ["majority_ce", "human_soft_targets", "label_smoothing", "human_soft_targets_null_augmented"]
LABELS = ["negative", "neutral", "positive", "mixed"]


def _read_round(round_dir: Path, expected_round: str):
    metrics_path = round_dir / "metrics.csv"
    pred_path = round_dir / "predictions.jsonl"
    if not metrics_path.exists() or not pred_path.exists():
        raise ValueError(f"{round_dir}: requires metrics.csv and predictions.jsonl")
    metrics = pd.read_csv(metrics_path)
    preds = pd.read_json(pred_path, lines=True)
    if metrics.empty or preds.empty:
        raise ValueError(f"{round_dir}: empty metrics or predictions")
    required_m = {"round", "backend", "seed", "condition", "js_divergence_mean_bits", "macro_f1", "accuracy", "soft_brier_score", "ece_10bin", "n_test"}
    required_p = {"round", "backend", "seed", "condition", "text_id", "gold_label", "human_distribution", "predicted_distribution", "predicted_label", "js_divergence_bits"}
    if not required_m.issubset(metrics.columns):
        raise ValueError(f"{metrics_path}: missing columns {sorted(required_m-set(metrics.columns))}")
    if not required_p.issubset(preds.columns):
        raise ValueError(f"{pred_path}: missing columns {sorted(required_p-set(preds.columns))}")
    if set(metrics["round"].astype(str)) != {expected_round} or set(preds["round"].astype(str)) != {expected_round}:
        raise ValueError(f"{round_dir}: round metadata does not match {expected_round}")
    if set(metrics["backend"].astype(str)) != set(preds["backend"].astype(str)):
        raise ValueError(f"{round_dir}: backend mismatch between metrics and predictions")
    return metrics, preds


def discover_round_dirs(root: Path):
    # Supports the repository runner layout (root/round1) and downloaded artifact layout.
    found = {}
    for name in ("round1", "round2"):
        p = root / name
        if (p / "metrics.csv").exists() and (p / "predictions.jsonl").exists():
            found[name] = p
    if set(found) != {"round1", "round2"}:
        raise ValueError(f"Expected round1/ and round2/ with metrics.csv + predictions.jsonl under {root}; found {sorted(found)}")
    return found


def _validate_matrix(metrics: pd.DataFrame, preds: pd.DataFrame, expected_seeds=None):
    backends = set(metrics.backend.astype(str)) | set(preds.backend.astype(str))
    if len(backends) != 1:
        raise ValueError(f"Mixed backends are prohibited within one analysis: {sorted(backends)}")
    if set(metrics.condition.astype(str)) != set(preds.condition.astype(str)):
        raise ValueError("Conditions in metrics and predictions differ")
    observed_conditions = set(metrics.condition.astype(str))
    if observed_conditions != set(CONDITIONS):
        raise ValueError(f"Expected all four research conditions {CONDITIONS}, found {sorted(observed_conditions)}")
    seeds = sorted(set(pd.to_numeric(metrics.seed).astype(int)))
    if expected_seeds is not None and seeds != sorted(expected_seeds):
        raise ValueError(f"Expected seeds {sorted(expected_seeds)}, found {seeds}")
    duplicate_metrics = metrics.duplicated(["round", "seed", "condition"])
    if duplicate_metrics.any():
        raise ValueError("Duplicate round/seed/condition metric rows")
    # Every observed seed must have all conditions represented in each round.
    for seed in seeds:
        got = set(metrics.loc[metrics.seed.astype(int) == seed, "condition"].astype(str))
        if got != observed_conditions:
            raise ValueError(f"Seed {seed} has incomplete condition matrix: {sorted(got)} vs {sorted(observed_conditions)}")
    keys = ["round", "seed", "condition", "text_id"]
    if preds.duplicated(keys).any():
        raise ValueError("Duplicate per-example prediction keys")
    # For each seed, conditions must evaluate exactly the same IDs and human targets.
    for (rnd, seed), group in preds.groupby(["round", "seed"]):
        sets = {cond: set(g.text_id.astype(str)) for cond, g in group.groupby("condition")}
        ref = sets.get(PRIMARY_B)
        if ref is None or any(ids != ref for ids in sets.values()):
            raise ValueError(f"Mismatched test IDs across conditions in {rnd}, seed {seed}")
        human_ref = group[group.condition.astype(str) == PRIMARY_B].set_index("text_id")["human_distribution"].map(_canonical_dist).to_dict()
        for cond, cg in group.groupby("condition"):
            for _, row in cg.iterrows():
                if _canonical_dist(row.human_distribution) != human_ref[str(row.text_id)]:
                    raise ValueError(f"Human distribution differs across conditions: {rnd}/{seed}/{row.text_id}")
    # The same test set and human targets must be used across all seeds within a round.
    for rnd, rg in preds.groupby("round"):
        reference = rg[(rg.seed.astype(int) == seeds[0]) & (rg.condition.astype(str) == PRIMARY_B)].set_index("text_id")["human_distribution"].map(_canonical_dist).to_dict()
        for seed in seeds[1:]:
            current = rg[(rg.seed.astype(int) == seed) & (rg.condition.astype(str) == PRIMARY_B)].set_index("text_id")["human_distribution"].map(_canonical_dist).to_dict()
            if current != reference:
                raise ValueError(f"Test IDs or human distributions differ across seeds in {rnd}")
    # Validate the stored per-example JS against the distributions, not just its aggregate mean.
    for _, row in preds.iterrows():
        computed = _js_divergence_bits(row.predicted_distribution, row.human_distribution)
        if abs(computed - float(row.js_divergence_bits)) > 2e-4:
            raise ValueError(f"Stored JS divergence inconsistent with distributions for {row['round']}/{row['seed']}/{row['condition']}/{row['text_id']}")
    # Compare row-level JS means with metric summary; tolerances allow float serialization only.
    means = preds.groupby(["round", "seed", "condition"]).js_divergence_bits.mean()
    for _, row in metrics.iterrows():
        key = (str(row["round"]), int(row["seed"]), str(row["condition"]))
        if key not in means.index:
            raise ValueError(f"Missing prediction group for metric key {key}")
        if abs(float(means.loc[key]) - float(row.js_divergence_mean_bits)) > 2e-4:
            raise ValueError(f"JS mean mismatch for {key}: predictions={means.loc[key]:.6f}, metrics={row.js_divergence_mean_bits:.6f}")
    return sorted(backends)[0], seeds, sorted(observed_conditions)


def _canonical_dist(value):
    if isinstance(value, str):
        value = json.loads(value)
    vals = tuple(round(float(x), 8) for x in value)
    if len(vals) != 4 or any(x < -1e-7 for x in vals) or not math.isclose(sum(vals), 1.0, abs_tol=1e-4):
        raise ValueError(f"Invalid four-class probability distribution: {value}")
    return vals


def _js_divergence_bits(p_value, q_value):
    p = np.asarray(_canonical_dist(p_value), dtype=float)
    q = np.asarray(_canonical_dist(q_value), dtype=float)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _entropy_bits(value):
    p = np.asarray(_canonical_dist(value), dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _holm(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p); order = np.argsort(p); adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def _bootstrap_mean_ci(values, n_boot, rng, alpha=0.05, chunk=256):
    values = np.asarray(values, dtype=float)
    if values.size == 0: raise ValueError("Cannot bootstrap empty values")
    estimates = np.empty(n_boot, dtype=float)
    n = len(values)
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot-start)
        ix = rng.integers(0, n, size=(k, n))
        estimates[start:start+k] = values[ix].mean(axis=1)
    return float(np.quantile(estimates, alpha/2)), float(np.quantile(estimates, 1-alpha/2))


def _paired_permutation_p(values, n_perm, rng, chunk=256):
    """Two-sided paired sign-flip test over per-example condition differences."""
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    extreme = 0; done = 0
    while done < n_perm:
        k = min(chunk, n_perm-done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(k, len(values)))
        stats = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(stats >= observed - 1e-15))
        done += k
    return (extreme + 1) / (n_perm + 1)


def _prediction_deltas(preds, round_name, treatment, baseline):
    df = preds[preds["round"].astype(str) == round_name]
    a = df[df.condition.astype(str) == treatment][["seed", "text_id", "js_divergence_bits"]].rename(columns={"js_divergence_bits":"js_treatment"})
    b = df[df.condition.astype(str) == baseline][["seed", "text_id", "js_divergence_bits"]].rename(columns={"js_divergence_bits":"js_baseline"})
    m = a.merge(b, on=["seed", "text_id"], how="inner", validate="one_to_one")
    if len(m) != len(a) or len(m) != len(b): raise ValueError(f"Unpaired predictions for {round_name}: {treatment} vs {baseline}")
    m["delta_treatment_minus_baseline"] = m.js_treatment - m.js_baseline
    # Average the matched seed-level deltas for each shared test item; bootstrap items, not 5*N rows.
    by_id = m.groupby("text_id", as_index=False).delta_treatment_minus_baseline.mean()
    return m, by_id


def _summarize_seed_metrics(metrics):
    cols = ["js_divergence_mean_bits", "js_divergence_median_bits", "macro_f1", "accuracy", "soft_brier_score", "ece_10bin", "train_time_s"]
    available = [c for c in cols if c in metrics.columns]
    rows=[]
    for (rnd, cond), g in metrics.groupby(["round", "condition"]):
        row={"round":rnd,"condition":cond,"n_seeds":int(g.seed.nunique())}
        for c in available:
            row[c+"_mean_across_seeds"] = float(g[c].mean())
            row[c+"_sd_across_seeds"] = float(g[c].std(ddof=1)) if len(g)>1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_analysis(input_root: Path, output_dir: Path, expected_seeds=None, n_boot=10000, n_perm=10000, random_seed=20261019):
    rounds = discover_round_dirs(input_root)
    all_m=[]; all_p=[]; backend=None; seed_union=None; condition_union=None
    for rnd, d in rounds.items():
        m,p = _read_round(d, rnd)
        be,seeds,conditions = _validate_matrix(m,p,expected_seeds=expected_seeds)
        if backend is not None and be != backend: raise ValueError("Backend differs between rounds")
        backend=be
        if seed_union is not None and seeds != seed_union: raise ValueError("Seed sets differ across rounds")
        if condition_union is not None and conditions != condition_union: raise ValueError("Condition sets differ across rounds")
        seed_union,condition_union=seeds,conditions
        all_m.append(m); all_p.append(p)
    metrics=pd.concat(all_m,ignore_index=True); preds=pd.concat(all_p,ignore_index=True)
    if len(rounds) != 2: raise ValueError("Both DynaSent rounds are required")
    # Human-vote-derived disagreement tier; never model confidence.
    preds["human_entropy_bits"] = preds.human_distribution.map(_entropy_bits)
    preds["disagreement_tier"] = np.where(preds.human_entropy_bits <= 1e-8, "unanimous", "non_unanimous")
    preds["soft_brier_score"] = preds.apply(lambda r: float(np.sum((np.asarray(_canonical_dist(r.predicted_distribution)) - np.asarray(_canonical_dist(r.human_distribution))) ** 2)), axis=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)

    # Seed-level summaries and matched per-seed contrasts.
    seed_summary = _summarize_seed_metrics(metrics)
    seed_summary.to_csv(output_dir/"metric_summary_by_condition.csv", index=False)
    primary=[]; per_seed=[]; secondary=[]
    for rnd in ("round1", "round2"):
        m_round=metrics[metrics["round"].astype(str)==rnd]
        for condition in [c for c in condition_union if c != PRIMARY_B]:
            base=m_round[m_round.condition.astype(str)==PRIMARY_B].set_index("seed")
            tr=m_round[m_round.condition.astype(str)==condition].set_index("seed")
            common=sorted(set(base.index.astype(int)) & set(tr.index.astype(int)))
            for seed in common:
                per_seed.append({"round":rnd,"treatment":condition,"baseline":PRIMARY_B,"seed":int(seed),
                    "js_delta_treatment_minus_baseline":float(tr.loc[seed,"js_divergence_mean_bits"]-base.loc[seed,"js_divergence_mean_bits"]),
                    "macro_f1_delta":float(tr.loc[seed,"macro_f1"]-base.loc[seed,"macro_f1"]),
                    "accuracy_delta":float(tr.loc[seed,"accuracy"]-base.loc[seed,"accuracy"]),
                    "soft_brier_delta":float(tr.loc[seed,"soft_brier_score"]-base.loc[seed,"soft_brier_score"]),
                    "ece_delta":float(tr.loc[seed,"ece_10bin"]-base.loc[seed,"ece_10bin"])})
            _, by_id = _prediction_deltas(preds, rnd, condition, PRIMARY_B)
            d=by_id.delta_treatment_minus_baseline.to_numpy(float)
            boot_rng=np.random.default_rng(random_seed + (1 if rnd=="round1" else 2) + sum(map(ord,condition)))
            lo,hi=_bootstrap_mean_ci(d,n_boot,boot_rng)
            pval=_paired_permutation_p(d,n_perm,np.random.default_rng(random_seed+100+sum(map(ord,rnd+condition))))
            row={"round":rnd,"treatment":condition,"baseline":PRIMARY_B,"n_test_examples":int(len(d)),
                 "n_seeds":int(metrics[metrics["round"].astype(str)==rnd].seed.nunique()),
                 "js_delta_treatment_minus_baseline":float(d.mean()),"bootstrap_95ci_low":lo,"bootstrap_95ci_high":hi,
                 "paired_example_signflip_p_two_sided":pval,
                 "inference_status":"exploratory; models/seeds fixed; test-example uncertainty only"}
            seed_d=[x["js_delta_treatment_minus_baseline"] for x in per_seed if x["round"]==rnd and x["treatment"]==condition]
            row["per_seed_delta_mean"]=float(np.mean(seed_d)) if seed_d else np.nan
            row["per_seed_delta_sd"]=float(np.std(seed_d,ddof=1)) if len(seed_d)>1 else 0.0
            if condition == PRIMARY_A: primary.append(row)
            else: secondary.append(row)
    # Holm only over the two frozen primary round-specific tests.
    if len(primary)!=2: raise ValueError("Expected exactly two primary round-specific comparisons")
    adj=_holm([x["paired_example_signflip_p_two_sided"] for x in primary])
    for row,padj in zip(primary,adj): row["holm_adjusted_p_two_primary_round_tests"]=padj
    pd.DataFrame(primary).to_csv(output_dir/"primary_js_comparisons.csv",index=False)
    pd.DataFrame(per_seed).to_csv(output_dir/"paired_per_seed_deltas.csv",index=False)
    pd.DataFrame(secondary).to_csv(output_dir/"secondary_condition_comparisons.csv",index=False)

    # Disagreement-tier descriptive metrics, including paired primary JS deltas and bootstrap CIs.
    tier_rows=[]
    for rnd in ("round1","round2"):
        pr=preds[preds["round"].astype(str)==rnd]
        for tier,tierdf in pr.groupby("disagreement_tier"):
            for cond,cdf in tierdf.groupby("condition"):
                tier_rows.append({"round":rnd,"tier":tier,"condition":cond,"n_unique_test_examples":int(cdf.text_id.nunique()),
                    "js_mean":float(cdf.js_divergence_bits.mean()),"js_sd_across_prediction_rows":float(cdf.js_divergence_bits.std(ddof=1)),
                    "soft_brier_mean":float(cdf.soft_brier_score.mean())})
        for tier,tierdf in pr.groupby("disagreement_tier"):
            if tierdf[tierdf.condition.astype(str)==PRIMARY_A].empty or tierdf[tierdf.condition.astype(str)==PRIMARY_B].empty: continue
            aa=tierdf[tierdf.condition.astype(str)==PRIMARY_A][["seed","text_id","js_divergence_bits"]].rename(columns={"js_divergence_bits":"a"})
            bb=tierdf[tierdf.condition.astype(str)==PRIMARY_B][["seed","text_id","js_divergence_bits"]].rename(columns={"js_divergence_bits":"b"})
            paired=aa.merge(bb,on=["seed","text_id"],validate="one_to_one"); byid=paired.assign(delta=paired.a-paired.b).groupby("text_id").delta.mean().to_numpy()
            lo,hi=_bootstrap_mean_ci(byid,n_boot,np.random.default_rng(random_seed+900+sum(map(ord,rnd+tier))))
            tier_rows.append({"round":rnd,"tier":tier,"condition":"PRIMARY_DELTA_human_soft_targets_minus_majority_ce","n_unique_test_examples":len(byid),"js_mean":float(byid.mean()),"js_sd_across_prediction_rows":np.nan,"soft_brier_mean":np.nan,"js_delta_bootstrap_95ci_low":lo,"js_delta_bootstrap_95ci_high":hi})
    pd.DataFrame(tier_rows).to_csv(output_dir/"disagreement_tier_analysis.csv",index=False)

    # Machine-readable manifest and human-readable report.
    manifest={"status":"passed","backend":backend,"rounds":["round1","round2"],"seeds":seed_union,"conditions":condition_union,
      "metric_records":int(len(metrics)),"prediction_rows":int(len(preds)),"primary_metric":"mean per-example JS divergence in bits; lower is better",
      "primary_contrast":"human_soft_targets - majority_ce; negative favors human soft targets",
      "bootstrap":{"replicates":n_boot,"unit":"test text_id resampled with replacement; five trained seeds held fixed","interval":"percentile 95%"},
      "permutation":{"replicates":n_perm,"method":"two-sided paired sign-flip over per-example differences averaged across fixed seeds"},
      "holm_family":"two primary round-specific JS tests only","interpretation":"Exploratory: the five-seed runs were already inspected; p-values are conditional on these fitted models and do not estimate between-training-run uncertainty.",
      "learning_rate_note":"Analysis reads recorded per-run metrics; legacy config args.lr may differ from effective tiny_bert_lr. Effective learning_rate in metric rows is the training value."}
    (output_dir/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    _write_report(output_dir,manifest,pd.DataFrame(primary),seed_summary,pd.DataFrame(secondary),pd.DataFrame(tier_rows))
    _write_plots(output_dir,seed_summary,pd.DataFrame(primary),pd.DataFrame(secondary))
    return manifest


def _write_report(out, manifest, primary, summary, secondary, tiers):
    lines=["# DynaSent Statistical Validation and Results Analysis","",f"**Analysis status:** {manifest['status']}","", "## Dataset and completeness", "",
      f"- Backend: `{manifest['backend']}` (analyzed separately; mixed backends rejected)",f"- Rounds: {', '.join(manifest['rounds'])}",f"- Seeds: {manifest['seeds']}",f"- Conditions: {', '.join(manifest['conditions'])}",
      f"- Metric records: {manifest['metric_records']}; prediction rows: {manifest['prediction_rows']}","- Test IDs, human distributions, backend labels, and per-group JS means were checked for consistency.","",
      "## Primary outcome", "", "Mean per-example Jensen–Shannon divergence (base 2, bits); lower is better. The contrast is human soft targets minus majority-label CE; negative values favor human soft targets.","",
      "| Round | Test examples | Mean JS difference | Paired example bootstrap 95% CI | Sign-flip p | Holm-adjusted p | Per-seed mean Δ (SD) |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for _,r in primary.iterrows():
        lines.append(f"| {r['round']} | {int(r['n_test_examples'])} | {r['js_delta_treatment_minus_baseline']:.6f} | [{r['bootstrap_95ci_low']:.6f}, {r['bootstrap_95ci_high']:.6f}] | {r['paired_example_signflip_p_two_sided']:.6g} | {r['holm_adjusted_p_two_primary_round_tests']:.6g} | {r['per_seed_delta_mean']:.6f} ({r['per_seed_delta_sd']:.6f}) |")
    lines += ["", "**Inference caveat:** these runs have already been inspected; all inferential outputs here are exploratory. Bootstrap intervals resample test examples while holding the five trained seeds fixed. They do not capture full training-run uncertainty. The sign-flip p-values are conditional on the observed fitted models and test examples; do not describe them as preregistered confirmatory evidence.","", "## Metrics by condition", "", "`metric_summary_by_condition.csv` reports means and sample SDs across seeds for JS divergence, macro-F1, accuracy, soft Brier, ECE, and runtime. Macro-F1/accuracy are defined on examples with valid majority labels by the runner. Calibration and Brier results are descriptive secondary outcomes.","", "## Secondary and exploratory comparisons", "", "See `secondary_condition_comparisons.csv` for label smoothing and null-augmented soft targets versus majority CE. Null-augmented training uses a different training-data composition and is exploratory, not a matched-data causal contrast.","", "## Disagreement-tier analysis", "", "See `disagreement_tier_analysis.csv` for unanimous versus non-unanimous groups derived from human vote distributions, never model confidence. Empty tiers are not fabricated.","", "## Reproducibility outputs", "", "- `primary_js_comparisons.csv`: primary round-specific contrasts and Holm correction.","- `paired_per_seed_deltas.csv`: paired seed-level effects for all alternative conditions.","- `metric_summary_by_condition.csv`: means and sample SDs across seeds.","- `secondary_condition_comparisons.csv`: secondary/exploratory contrasts.","- `disagreement_tier_analysis.csv`: descriptive subgroup results and primary delta intervals.","- `analysis_manifest.json`: settings, data counts, and inference caveats.","- `figures/`: summary figures.","", "## Interpretation guardrails", "", "Do not pool Round 1 and Round 2 into a single headline result. Do not claim universal superiority from the lowest mean alone. Five seeds are a small sample for training-run variability. Treat p-values as exploratory, and do not tune settings against these test results.",""]
    (out/"FINAL_RESULTS_REPORT.md").write_text("\n".join(lines))


def _write_plots(out, summary, primary, secondary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    figdir=out/"figures"; figdir.mkdir(exist_ok=True)
    # JS divergence by condition and round with across-seed SD error bars.
    s=summary.copy()
    mean_col="js_divergence_mean_bits_mean_across_seeds"; sd_col="js_divergence_mean_bits_sd_across_seeds"
    if mean_col in s.columns:
        conds=list(dict.fromkeys(s.condition.tolist())); rounds=["round1","round2"]
        fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
        for ax,rnd in zip(axes,rounds):
            g=s[s["round"]==rnd].set_index("condition").reindex(conds)
            ax.bar(range(len(conds)),g[mean_col],yerr=g[sd_col],capsize=3)
            ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds,rotation=25,ha="right")
            ax.set_title(rnd); ax.set_ylabel("Mean JS divergence (bits; lower is better)"); ax.grid(axis="y",alpha=.25)
        fig.tight_layout(); fig.savefig(figdir/"js_by_condition_round.png",dpi=180); plt.close(fig)
    if not primary.empty:
        fig,ax=plt.subplots(figsize=(6.5,4.2)); x=np.arange(len(primary)); y=primary.js_delta_treatment_minus_baseline.to_numpy()
        low=y-primary.bootstrap_95ci_low.to_numpy(); high=primary.bootstrap_95ci_high.to_numpy()-y
        ax.errorbar(x,y,yerr=np.vstack([low,high]),fmt="o",capsize=5); ax.axhline(0,linestyle="--",linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(primary["round"]); ax.set_ylabel("JS difference: soft targets − majority CE (bits)")
        ax.set_title("Primary contrast with paired test-example bootstrap 95% CI"); ax.grid(axis="y",alpha=.25)
        fig.tight_layout(); fig.savefig(figdir/"primary_js_effects.png",dpi=180); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir",required=True,type=Path,help="Directory containing round1/ and round2/ experiment outputs")
    ap.add_argument("--output-dir",required=True,type=Path,help="Analysis output directory (keep backend-specific)")
    ap.add_argument("--expected-seeds",nargs="*",type=int,default=None)
    ap.add_argument("--bootstrap-replicates",type=int,default=10000)
    ap.add_argument("--permutation-replicates",type=int,default=10000)
    ap.add_argument("--random-seed",type=int,default=20261019)
    args=ap.parse_args()
    if args.bootstrap_replicates < 100 or args.permutation_replicates < 100: ap.error("use at least 100 bootstrap/permutation replicates")
    manifest=run_analysis(args.input_dir,args.output_dir,args.expected_seeds,args.bootstrap_replicates,args.permutation_replicates,args.random_seed)
    print(json.dumps({"status":manifest["status"],"backend":manifest["backend"],"metric_records":manifest["metric_records"],"prediction_rows":manifest["prediction_rows"],"output_dir":str(args.output_dir)},indent=2))

if __name__=="__main__": main()
