from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_dynasent_experiments import evaluate, sample_rows
from dynasent_data import LABELS


def _row(i, gold=0, probs=None):
    probs = probs or [1.0, 0.0, 0.0, 0.0]
    return {"text": f"example {i}", "text_id": str(i), "gold_label": LABELS[gold] if gold is not None else None,
            "gold_label_id": gold, "target_probs": probs, "num_votes": 5}


def test_evaluate_outputs_distributional_and_hard_label_metrics():
    rows = [_row(1), _row(2, gold=1, probs=[0, 1, 0, 0])]
    probs = np.array([[0.9, 0.05, 0.03, 0.02], [0.1, 0.8, 0.05, 0.05]], dtype=np.float32)
    m = evaluate(probs, rows)
    assert m["n_test"] == 2
    assert m["n_with_majority_gold"] == 2
    assert m["accuracy"] == 1.0
    assert m["js_divergence_mean_bits"] >= 0
    assert m["soft_brier_score"] >= 0
    assert 0 <= m["ece_10bin"] <= 1


def test_evaluate_keeps_null_gold_in_distributional_metrics():
    rows = [_row(1), _row(2, gold=None, probs=[0.2, 0.2, 0.2, 0.4])]
    probs = np.array([[0.9, 0.05, 0.03, 0.02], [0.2, 0.2, 0.2, 0.4]], dtype=np.float32)
    m = evaluate(probs, rows)
    assert m["n_test"] == 2
    assert m["n_with_majority_gold"] == 1
    assert m["accuracy"] == 1.0
    assert m["js_divergence_mean_bits"] >= 0


def test_sample_rows_is_reproducible_and_full_when_zero():
    rows = [_row(i) for i in range(10)]
    assert [r["text_id"] for r in sample_rows(rows, 4, 3)] == [r["text_id"] for r in sample_rows(rows, 4, 3)]
    assert len(sample_rows(rows, 0, 3)) == 10
