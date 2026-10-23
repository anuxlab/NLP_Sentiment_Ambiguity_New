"""Evaluate a deterministic VADER-style lexicon baseline on DynaSent R1/R2.

This is a zero-training-data reference. DynaSent has four labels, so the mapping is
explicitly heuristic: simultaneous positive and negative lexical evidence -> mixed;
otherwise VADER compound thresholds map to positive/negative, with neutral in between.
Do not compare its runtime as though it were trained at a label budget.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from dynasent_data import LABELS, load_dynasent_round
from soft_targets import jensen_shannon_divergence
import torch


def map_vader(scores: dict[str, float], threshold: float = 0.05) -> int:
    # Explicit four-way mapping. Mixed is a heuristic extension, not a native VADER label.
    if scores["pos"] >= 0.15 and scores["neg"] >= 0.15:
        return LABELS.index("mixed")
    compound = scores["compound"]
    if compound >= threshold:
        return LABELS.index("positive")
    if compound <= -threshold:
        return LABELS.index("negative")
    return LABELS.index("neutral")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(ROOT / "data" / "dynasent-main.zip"))
    p.add_argument("--round", choices=["round1", "round2", "both"], default="both")
    p.add_argument("--out-dir", default=str(ROOT / "results" / "dynasent_vader"))
    p.add_argument("--threshold", type=float, default=0.05)
    args = p.parse_args()
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise SystemExit("Install dependencies with `pip install -r requirements.txt` (vaderSentiment).") from exc
    analyzer = SentimentIntensityAnalyzer()
    rounds = ["round1", "round2"] if args.round == "both" else [args.round]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    records = []
    for round_name in rounds:
        splits = load_dynasent_round(Path(args.data), round_name)
        pred_rows = []
        start = time.perf_counter()
        for row in splits.test:
            scores = analyzer.polarity_scores(row["text"])
            pred_id = map_vader(scores, args.threshold)
            pred = np.zeros(len(LABELS), dtype=float); pred[pred_id] = 1.0
            target = np.asarray(row["target_probs"], dtype=float)
            js = float(jensen_shannon_divergence(torch.tensor(pred[None, :], dtype=torch.float32), torch.tensor(target[None, :], dtype=torch.float32)).item())
            pred_rows.append({"round": round_name, "text_id": row.get("text_id"), "text": row["text"], "gold_label": row.get("gold_label"), "human_distribution": target.tolist(), "predicted_label": LABELS[pred_id], "predicted_distribution": pred.tolist(), "js_divergence_bits": js, "vader_compound": scores["compound"], "vader_pos": scores["pos"], "vader_neg": scores["neg"], "vader_neu": scores["neu"]})
        elapsed = time.perf_counter() - start
        valid = [r for r in pred_rows if r["gold_label"] in LABELS]
        y = [LABELS.index(r["gold_label"]) for r in valid]
        yp = [LABELS.index(r["predicted_label"]) for r in valid]
        record = {"round": round_name, "backend": "vader_lexicon", "condition": "zero_shot_rule_based", "budget": 0, "seed": 0, "n_test": len(pred_rows), "train_time_s": 0.0, "inference_time_s": elapsed, "js_divergence_mean_bits": float(np.mean([r["js_divergence_bits"] for r in pred_rows])), "accuracy": float(accuracy_score(y, yp)) if valid else None, "macro_f1": float(f1_score(y, yp, labels=list(range(len(LABELS))), average="macro", zero_division=0)) if valid else None, "n_with_majority_gold": len(valid), "mapping": "mixed if VADER pos>=0.15 and neg>=0.15; else compound >= threshold positive, <= -threshold negative, otherwise neutral", "compound_threshold": args.threshold}
        records.append(record)
        with (out / f"{round_name}_predictions.jsonl").open("w", encoding="utf-8") as f:
            for row in pred_rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(record, indent=2))
    (out / "metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (out / "README.md").write_text("# DynaSent VADER baseline\n\nZero-shot rule-based reference. VADER's three-way polarity is mapped heuristically to DynaSent's four classes; `mixed` is assigned when both positive and negative lexical evidence are sufficiently high. This mapping is a project-specific heuristic, not a canonical VADER output. The baseline has no labeled training budget and should be shown as a horizontal reference, not as a learned learning curve.\n", encoding="utf-8")

if __name__ == "__main__": main()
