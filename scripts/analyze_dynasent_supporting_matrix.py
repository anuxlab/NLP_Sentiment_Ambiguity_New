"""Aggregate model-comparison metrics and plot learning curves/ranking changes."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METRICS = ["macro_f1", "accuracy", "js_divergence_mean_bits", "soft_brier_score"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True)
    args = p.parse_args(); root = Path(args.results_root)
    frames = []
    for file in root.rglob("metrics.csv"):
        try: d = pd.read_csv(file)
        except Exception: continue
        if not {"backend", "budget", "seed", "condition"}.issubset(d.columns): continue
        d["source_file"] = str(file)
        frames.append(d)
    if not frames: raise SystemExit(f"No metrics.csv files found under {root}")
    df = pd.concat(frames, ignore_index=True)
    # Avoid duplicate outputs when a matrix is rerun into a nested output tree.
    df = df.drop_duplicates(subset=[c for c in ["round", "backend", "seed", "condition", "budget"] if c in df.columns], keep="last")
    df.to_csv(root / "all_metrics.csv", index=False)
    matched = df[df["condition"].eq("majority_ce")].copy()
    available = [m for m in METRICS if m in matched.columns]
    summary = matched.groupby(["round", "backend", "budget"], dropna=False)[available].agg(["mean", "std", "count"])
    summary.to_csv(root / "learning_curve_summary.csv")
    ranking_rows=[]
    for (round_name, budget), g in matched.groupby(["round", "budget"]):
        for metric in [m for m in ["macro_f1", "js_divergence_mean_bits"] if m in g.columns]:
            means = g.groupby("backend")[metric].mean().dropna()
            if means.empty: continue
            best = means.idxmax() if metric in ["macro_f1", "accuracy"] else means.idxmin()
            ranking_rows.append({"round":round_name,"budget":budget,"metric":metric,"best_backend_observed":best,"tfidf_mean":means.get("tfidf"),"tiny_bert_mean":means.get("tiny-bert"),"interpretation":"observed descriptive ranking; seeds are few and comparisons remain exploratory"})
    pd.DataFrame(ranking_rows).to_csv(root / "observed_rankings.csv", index=False)
    for metric in available:
        for round_name, g in matched.groupby("round"):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for backend, b in g.groupby("backend"):
                agg = b.groupby("budget")[metric].agg(["mean", "std", "count"]).sort_index()
                ax.plot(agg.index, agg["mean"], marker="o", label=backend)
                if agg["std"].notna().any():
                    err = agg["std"].fillna(0) / agg["count"].clip(lower=1).pow(0.5)
                    ax.fill_between(agg.index, agg["mean"]-err, agg["mean"]+err, alpha=0.15)
            ax.set_xlabel("Labeled training examples"); ax.set_ylabel(metric.replace("_", " ")); ax.set_title(f"{round_name}: {metric} by label budget (majority CE)")
            ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
            fig.savefig(root / f"{round_name}_{metric}_learning_curve.png", dpi=160); plt.close(fig)
    (root / "INTERPRETATION.md").write_text("# Supporting DynaSent model comparison\n\nOnly the `majority_ce` condition is used for model-family learning curves, so the TF-IDF and Tiny-BERT curves have the same target objective. Each round is analyzed separately. Error bands show standard error across available seeds, not test-example bootstrap uncertainty. Ranking changes are descriptive and are reported only where both backends have observations at the same budget. VADER is a zero-shot reference with no training budget and should be added as a horizontal baseline only when plotting a metric it shares. This supporting comparison does not replace the primary human-ambiguity/soft-target analysis.\n", encoding="utf-8")
    print(f"Wrote comparison tables and plots under {root}")
if __name__ == "__main__": main()
