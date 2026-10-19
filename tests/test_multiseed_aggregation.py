import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_multiseed_aggregator_checks_complete_matrix_and_writes_summaries(tmp_path):
    import csv
    root = tmp_path / "results"
    conditions = ["majority_ce", "human_soft_targets", "label_smoothing", "human_soft_targets_null_augmented"]
    for rnd in ["round1", "round2"]:
        d = root / rnd
        d.mkdir(parents=True)
        metrics = []
        predictions = []
        (d / "config_and_audit.json").write_text(json.dumps({"args": {
            "round": rnd, "backend": "tfidf", "seeds": [0, 1], "budget": 2,
            "epochs": 1, "eval_size": 0, "tiny_bert_lr": 0.00002
        }}))
        for seed in [0, 1]:
            for j, condition in enumerate(conditions):
                metrics.append({"round": rnd, "backend": "tfidf", "seed": seed, "condition": condition,
                    "js_divergence_mean_bits": .4 - .01*j, "macro_f1": .2 + .01*j,
                    "accuracy": .3 + .01*j, "soft_brier_score": .5 - .01*j,
                    "ece_10bin": .1, "train_time_s": 1.0, "n_test": 2,
                    "n_null_gold_train": 1 if "null_augmented" in condition else 0,
                    "n_non_null_gold_train": 1})
                for i in range(2):
                    predictions.append({"round": rnd, "seed": seed, "condition": condition,
                        "human_distribution": [.2,.2,.2,.4], "predicted_distribution": [.25,.25,.25,.25],
                        "gold_label": None, "predicted_label": "mixed"})
        (d / "metrics.json").write_text(json.dumps(metrics))
        (d / "predictions.jsonl").write_text("\n".join(json.dumps(x) for x in predictions) + "\n")
    subprocess.run([sys.executable, str(ROOT / "src" / "aggregate_dynasent_multiseed.py"),
                    "--results-root", str(root), "--rounds", "round1", "round2", "--expect-null-augmented"],
                   check=True, capture_output=True, text=True)
    audit = json.loads((root / "run_completeness.json").read_text())
    assert audit["all_complete"] is True
    assert audit["n_metric_rows"] == 16
    assert audit["n_prediction_rows"] == 32
    with (root / "multiseed_summary.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 8
    assert (root / "paired_vs_majority_ce.csv").exists()
    assert (root / "disagreement_tier_summary.csv").exists()


def test_multiseed_aggregator_flags_missing_condition(tmp_path):
    root = tmp_path / "results"
    d = root / "round1"
    d.mkdir(parents=True)
    (d / "config_and_audit.json").write_text(json.dumps({"args": {
        "round": "round1", "backend": "tfidf", "seeds": [0], "budget": 2,
        "epochs": 1, "eval_size": 0, "tiny_bert_lr": 0.00002
    }}))
    (d / "metrics.json").write_text(json.dumps([{"round":"round1","backend":"tfidf","seed":0,"condition":"majority_ce",
        "js_divergence_mean_bits":.4,"macro_f1":.2,"accuracy":.3,"soft_brier_score":.5,"ece_10bin":.1,
        "train_time_s":1,"n_test":2,"n_null_gold_train":0,"n_non_null_gold_train":2}]))
    (d / "predictions.jsonl").write_text(json.dumps({"round":"round1","seed":0,"condition":"majority_ce",
        "human_distribution":[.2,.2,.2,.4],"predicted_distribution":[.25,.25,.25,.25],"gold_label":None,"predicted_label":"mixed"})+"\n")
    result = subprocess.run([sys.executable, str(ROOT / "src" / "aggregate_dynasent_multiseed.py"),
        "--results-root", str(root), "--rounds", "round1", "--expect-null-augmented"], capture_output=True, text=True)
    assert result.returncode != 0
    audit = json.loads((root / "run_completeness.json").read_text())
    assert audit["all_complete"] is False


def test_multiseed_aggregator_writes_failure_audit_when_provenance_missing(tmp_path):
    root = tmp_path / "results"
    d = root / "round1"
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps([{"round": "round1", "backend": "tfidf",
        "seed": 0, "condition": "majority_ce", "js_divergence_mean_bits": .4,
        "macro_f1": .2, "accuracy": .3, "soft_brier_score": .5, "ece_10bin": .1,
        "train_time_s": 1, "n_test": 2, "n_null_gold_train": 0, "n_non_null_gold_train": 2}]))
    (d / "predictions.jsonl").write_text(json.dumps({"round": "round1", "seed": 0,
        "condition": "majority_ce", "human_distribution": [.2, .2, .2, .4],
        "predicted_distribution": [.25, .25, .25, .25], "gold_label": None,
        "predicted_label": "mixed"}) + "\n")
    result = subprocess.run([sys.executable, str(ROOT / "src" / "aggregate_dynasent_multiseed.py"),
        "--results-root", str(root), "--rounds", "round1"], capture_output=True, text=True)
    assert result.returncode != 0
    audit = json.loads((root / "run_completeness.json").read_text())
    assert audit["status"] == "failed"
    assert audit["all_complete"] is False
    assert "provenance file" in audit["error"]
