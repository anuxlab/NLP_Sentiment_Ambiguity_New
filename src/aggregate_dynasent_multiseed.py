"""Aggregate repeated-seed DynaSent runs and produce paired and disagreement-tier summaries."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["js_divergence_mean_bits", "macro_f1", "accuracy", "soft_brier_score", "ece_10bin", "train_time_s"]

def read_json(path: Path):
    with path.open(encoding="utf-8") as f: return json.load(f)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results-root", default="results/dynasent_multiseed")
    p.add_argument("--rounds", nargs="+", default=["round1","round2"])
    p.add_argument("--expect-null-augmented", action="store_true", help="require the separately reported null-augmented condition")
    args=p.parse_args(); root=Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)

    def fail(message: str, *, round_name: str | None = None):
        # Preserve an audit artifact even when validation fails early. This makes
        # failed runs diagnosable and prevents downstream FileNotFoundError noise.
        audit = {
            "status": "failed",
            "all_complete": False,
            "error": message,
            "failed_round": round_name,
            "rounds_requested": list(args.rounds),
        }
        (root / "run_completeness.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        raise SystemExit(message)

    all_metrics=[]; all_preds=[]; provenance=[]
    for rnd in args.rounds:
        d=root/rnd
        metrics_path=d/"metrics.json"; preds_path=d/"predictions.jsonl"
        if not metrics_path.exists() or not preds_path.exists():
            fail(f"Expected both {metrics_path} and {preds_path}", round_name=rnd)
        config_path=d/"config_and_audit.json"
        if not config_path.exists():
            fail(f"Expected provenance file: {config_path}", round_name=rnd)
        config=read_json(config_path)
        configured_round=config.get("args",{}).get("round")
        configured_backend=config.get("args",{}).get("backend")
        if configured_round != rnd:
            fail(f"Round mismatch in {config_path}: {configured_round} != {rnd}", round_name=rnd)
        if configured_backend not in {"tiny-bert", "tfidf"}:
            fail(f"Missing/invalid backend in {config_path}: {configured_backend}", round_name=rnd)
        records=read_json(metrics_path)
        if not records:
            fail(f"No metric records in {metrics_path}", round_name=rnd)
        if {r.get("backend") for r in records} != {configured_backend}:
            fail(f"Backend mismatch between config and metrics for {rnd}", round_name=rnd)
        if {r.get("round") for r in records} != {rnd}:
            fail(f"Round mismatch between config and metrics for {rnd}", round_name=rnd)
        for r in records: all_metrics.append(r)
        provenance.append({"round":rnd,"backend":configured_backend,"seeds":config.get("args",{}).get("seeds"),"budget":config.get("args",{}).get("budget"),"epochs":config.get("args",{}).get("epochs"),"eval_size":config.get("args",{}).get("eval_size"),"tiny_bert_lr":config.get("args",{}).get("tiny_bert_lr")})
        with preds_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip(): all_preds.append(json.loads(line))
    if not all_metrics: raise ValueError("No metric records found")
    df=pd.DataFrame(all_metrics)
    if df.backend.nunique() != 1: raise ValueError(f"Mixed backends in one artifact: {sorted(df.backend.unique())}")
    if set(df["round"].unique()) != set(args.rounds): raise ValueError("Metrics do not cover exactly the requested rounds")
    summary=[]
    for (rnd,cond), g in df.groupby(["round","condition"], sort=True):
        row={"round":rnd,"condition":cond,"n_seeds":int(g.seed.nunique()),"n_runs":len(g)}
        for m in METRICS:
            vals=pd.to_numeric(g[m],errors="coerce").dropna()
            row[f"{m}_mean"]=float(vals.mean()) if len(vals) else None
            row[f"{m}_sd"]=float(vals.std(ddof=1)) if len(vals)>1 else (0.0 if len(vals)==1 else None)
        row["n_test_min"]=int(g.n_test.min()); row["n_test_max"]=int(g.n_test.max())
        row["n_null_gold_train_min"]=int(g.n_null_gold_train.min()); row["n_null_gold_train_max"]=int(g.n_null_gold_train.max())
        summary.append(row)
    summary_df=pd.DataFrame(summary); summary_df.to_csv(root/"multiseed_summary.csv",index=False)
    # Paired differences across matching round/seed, oriented so positive is better.
    paired=[]
    for rnd in sorted(df["round"].unique()):
        sub=df[df["round"]==rnd]
        for candidate in sorted(sub["condition"].unique()):
            if candidate=="majority_ce": continue
            for metric, higher in [("js_divergence_mean_bits",False),("macro_f1",True),("accuracy",True),("soft_brier_score",False),("ece_10bin",False)]:
                pivot=sub.pivot_table(index="seed",columns="condition",values=metric,aggfunc="first")
                if candidate not in pivot or "majority_ce" not in pivot: continue
                z=pivot[[candidate,"majority_ce"]].dropna()
                if z.empty: continue
                diff=(z[candidate]-z["majority_ce"])*(1 if higher else -1)
                paired.append({"round":rnd,"candidate":candidate,"baseline":"majority_ce","metric":metric,"direction":"higher_is_better" if higher else "lower_is_better","n_paired_seeds":len(diff),"mean_improvement":float(diff.mean()),"sd_improvement":float(diff.std(ddof=1)) if len(diff)>1 else 0.0,"seed_level_improvements":json.dumps([float(x) for x in diff.tolist()])})
    pd.DataFrame(paired).to_csv(root/"paired_vs_majority_ce.csv",index=False)
    # Human-disagreement tiers are defined from vote distributions, not model confidence.
    tiers=[]
    for r in all_preds:
        dist=np.asarray(r["human_distribution"],dtype=float); max_share=float(dist.max())
        entropy=float(-np.sum(dist[dist>0]*np.log2(dist[dist>0])))
        tier="unanimous" if entropy <= 1e-9 else "vote_disagreement"
        pred=np.asarray(r["predicted_distribution"],dtype=float)
        m=0.5*(dist+pred)
        def kl(a,b):
            mask=a>0
            return float(np.sum(a[mask]*np.log2(a[mask]/np.clip(b[mask],1e-12,None))))
        js=0.5*kl(dist,m)+0.5*kl(pred,m)
        tiers.append({"round":r["round"],"seed":r["seed"],"condition":r["condition"],"disagreement_tier":tier,"human_vote_entropy_bits":entropy,"max_human_vote_share":max_share,"js_divergence_bits":js,"correct":(float(r.get("predicted_label")==r.get("gold_label")) if r.get("gold_label") is not None else np.nan),"has_majority_gold":r.get("gold_label") is not None})
    tdf=pd.DataFrame(tiers)
    tdf.groupby(["round","condition","disagreement_tier"],dropna=False).agg(n_examples=("js_divergence_bits","size"),mean_js_divergence_bits=("js_divergence_bits","mean"),mean_max_human_vote_share=("max_human_vote_share","mean"),mean_human_vote_entropy_bits=("human_vote_entropy_bits","mean"),accuracy_on_gold_rows=("correct","mean"),n_with_majority_gold=("has_majority_gold","sum")).reset_index().to_csv(root/"disagreement_tier_summary.csv",index=False)
    metric_keys=set((str(r["round"]),int(r["seed"]),str(r["condition"])) for r in all_metrics)
    pred_keys=set((str(r["round"]),int(r["seed"]),str(r["condition"])) for r in all_preds)
    if not pred_keys.issubset(metric_keys): raise ValueError("Prediction keys absent from metrics")
    if not all_preds: raise ValueError("No predictions were loaded")
    # Protocol and run-completeness manifest.
    expected_rounds=set(args.rounds); expected_seeds=set()
    for r in all_metrics: expected_seeds.add(int(r["seed"]))
    expected_conditions={"majority_ce","human_soft_targets","label_smoothing"}
    if args.expect_null_augmented: expected_conditions.add("human_soft_targets_null_augmented")
    completeness=[]
    for rnd in sorted(expected_rounds):
        for seed in sorted(expected_seeds):
            got=set(df[(df["round"]==rnd)&(df["seed"]==seed)]["condition"])
            completeness.append({"round":rnd,"seed":seed,"expected_conditions":sorted(expected_conditions),"observed_conditions":sorted(got),"complete":expected_conditions.issubset(got)})
    (root/"run_completeness.json").write_text(json.dumps({"n_metric_rows":len(df),"n_prediction_rows":len(all_preds),"backend":str(df.backend.iloc[0]),"provenance":provenance,"rounds":sorted(expected_rounds),"seeds":sorted(expected_seeds),"checks":completeness,"all_complete":all(x["complete"] for x in completeness)},indent=2),encoding="utf-8")
    (root/"MULTISEED_SUMMARY.md").write_text(
        "# DynaSent multi-seed summary\n\n"+
        f"- Backend: {str(df.backend.iloc[0])}\n- Provenance validated for all rounds: yes\n- Metric records: {len(df)}\n- Prediction rows: {len(all_preds)}\n- Rounds: {', '.join(sorted(expected_rounds))}\n- Seeds: {', '.join(map(str,sorted(expected_seeds)))}\n- Complete round/seed/condition groups: {sum(x['complete'] for x in completeness)}/{len(completeness)}\n\n"+
        "Values are means and sample standard deviations across seeds. Paired improvements compare each condition with majority CE using matching seeds; they are descriptive and not a substitute for uncertainty analysis over examples. Disagreement tiers use human vote entropy, not model confidence: unanimous vote distributions (entropy 0) versus non-unanimous distributions (entropy > 0). Test-set results are not used to tune hyperparameters.\n",encoding="utf-8")
    if not all(x["complete"] for x in completeness): raise SystemExit("Incomplete run matrix; inspect run_completeness.json")
    print(f"Aggregated {len(df)} metric records and {len(all_preds)} predictions; all {len(completeness)} round/seed groups complete.")
if __name__=="__main__": main()
