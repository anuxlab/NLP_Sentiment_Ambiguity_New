from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_dynasent_experiments import evaluate, sample_rows, sample_null_augmented_rows, declared_condition_names
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


def test_null_augmented_sampler_guarantees_null_rows_and_fixed_budget():
    rows = [
        _row(i, gold=None if i < 6 else i % 4,
             probs=[0.2, 0.2, 0.2, 0.4] if i < 6 else [1, 0, 0, 0])
        for i in range(12)
    ]
    sampled = sample_null_augmented_rows(rows, budget=8, seed=7, null_fraction=0.5)
    assert len(sampled) == 8
    assert sum(r["gold_label_id"] is None for r in sampled) == 4
    assert sum(r["gold_label_id"] is not None for r in sampled) == 4
    assert len({r["text_id"] for r in sampled}) == 8


def test_null_augmented_sampler_is_reproducible():
    rows = [_row(i, gold=None if i < 5 else i % 4,
                 probs=[0.25, 0.25, 0.25, 0.25] if i < 5 else [1, 0, 0, 0])
            for i in range(10)]
    a = sample_null_augmented_rows(rows, budget=6, seed=13)
    b = sample_null_augmented_rows(rows, budget=6, seed=13)
    assert [r["text_id"] for r in a] == [r["text_id"] for r in b]


def test_null_augmented_sampler_rejects_budget_one():
    rows = [_row(1, gold=None, probs=[0.2, 0.2, 0.2, 0.4]), _row(2)]
    import pytest
    with pytest.raises(ValueError, match="budget >= 2"):
        sample_null_augmented_rows(rows, budget=1, seed=0)


def test_declared_conditions_match_optional_null_augmented_run():
    matched = declared_condition_names(False)
    assert matched == ["majority_ce", "human_soft_targets", "label_smoothing"]
    augmented = declared_condition_names(True)
    assert augmented == matched + ["human_soft_targets_null_augmented"]
    assert len(augmented) == len(set(augmented))
