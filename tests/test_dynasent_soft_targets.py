import json
from pathlib import Path
import zipfile
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pytest
import torch

from dynasent_data import LABELS, load_dynasent_round, ternary_distribution
from soft_targets import (
    hard_label_cross_entropy,
    jensen_shannon_divergence,
    normalize_probabilities,
    soft_target_cross_entropy,
)


def _row(text, votes, gold=None, text_id=None):
    dist = {label: [] for label in LABELS}
    worker = 0
    for label, count in zip(LABELS, votes):
        for _ in range(count):
            dist[label].append(f"worker-{worker}")
            worker += 1
    return {
        "sentence": text,
        "text_id": text_id or text,
        "label_distribution": dist,
        "gold_label": gold,
    }


def _write_zip(path: Path):
    files = {
        "dynasent-v1.1-round01-yelp-train.jsonl": [
            _row("Overlap sentence", [3, 1, 1, 0], "negative", "tr-1"),
            _row("Unique train", [1, 1, 2, 1], None, "tr-2"),
            _row("Unique train", [1, 1, 2, 1], None, "tr-3"),
        ],
        "dynasent-v1.1-round01-yelp-dev.jsonl": [
            _row(" overlap   SENTENCE ", [2, 1, 1, 1], None, "dev-1"),
        ],
        "dynasent-v1.1-round01-yelp-test.jsonl": [
            _row("Test only", [0, 1, 2, 2], "mixed", "te-1"),
        ],
    }
    with zipfile.ZipFile(path, "w") as zf:
        for name, rows in files.items():
            zf.writestr(name, "\n".join(json.dumps(r) for r in rows) + "\n")


def test_loader_reads_repository_zip_with_nested_official_archive(tmp_path):
    official = tmp_path / "dynasent-v1.1.zip"
    _write_zip(official)
    repo_archive = tmp_path / "dynasent-main.zip"
    with zipfile.ZipFile(repo_archive, "w") as outer:
        outer.write(official, "dynasent-main/dynasent-v1.1.zip")
    splits = load_dynasent_round(repo_archive, "round1")
    assert len(splits.train) == 1
    assert splits.audit["train_overlap_rows_removed"] == 1


def test_loader_preserves_four_class_distribution_and_null_gold(tmp_path):
    archive = tmp_path / "synthetic.zip"
    _write_zip(archive)
    splits = load_dynasent_round(archive, "round1")
    assert LABELS == ("negative", "neutral", "positive", "mixed")
    assert len(splits.train) == 1
    assert splits.train[0]["text"] == "Unique train"
    assert splits.train[0]["gold_label"] is None
    assert sum(splits.train[0]["target_probs"]) == pytest.approx(1.0)
    assert splits.audit["train_overlap_rows_removed"] == 1
    assert splits.audit["train_duplicate_rows_removed"] == 1
    assert splits.audit["dev_rows_unchanged"] == 1
    assert splits.audit["test_rows_unchanged"] == 1
    assert splits.test[0]["gold_label_id"] == 3


def test_loader_can_disable_filtering_and_deduplication(tmp_path):
    archive = tmp_path / "synthetic.zip"
    _write_zip(archive)
    splits = load_dynasent_round(archive, "round1", filter_train_overlap=False, deduplicate_train=False)
    assert len(splits.train) == 3
    assert splits.audit["train_overlap_rows_removed"] == 0
    assert splits.audit["train_duplicate_rows_removed"] == 0


def test_loader_rejects_empty_vote_distribution(tmp_path):
    archive = tmp_path / "bad.zip"
    files = {
        "dynasent-v1.1-round01-yelp-train.jsonl": [{
            "sentence": "x", "text_id": "x", "label_distribution": {k: [] for k in LABELS}, "gold_label": None
        }],
        "dynasent-v1.1-round01-yelp-dev.jsonl": [_row("d", [1, 1, 1, 2])],
        "dynasent-v1.1-round01-yelp-test.jsonl": [_row("t", [1, 1, 1, 2])],
    }
    with zipfile.ZipFile(archive, "w") as zf:
        for name, rows in files.items():
            zf.writestr(name, "\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="no votes"):
        load_dynasent_round(archive, "round1")


def test_ternary_mapping_is_explicit_and_normalized():
    probs = [0.4, 0.2, 0.2, 0.2]
    mapped = ternary_distribution(probs)
    assert mapped == pytest.approx([0.5, 0.25, 0.25])
    assert ternary_distribution(probs, mixed_policy="drop") is None
    assert ternary_distribution([0, 0, 0, 1]) is None


def test_soft_target_cross_entropy_matches_manual_calculation():
    logits = torch.tensor([[2.0, 0.0, -1.0, 0.5]], dtype=torch.float64)
    target = torch.tensor([[0.6, 0.2, 0.1, 0.1]], dtype=torch.float64)
    expected = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    assert soft_target_cross_entropy(logits, target) == pytest.approx(expected.item())


def test_soft_target_cross_entropy_backpropagates():
    logits = torch.randn(3, 4, requires_grad=True)
    targets = torch.tensor([[1, 0, 0, 0], [0.2, 0.3, 0.4, 0.1], [0, 0, 0, 1.]], dtype=torch.float32)
    loss = soft_target_cross_entropy(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_soft_target_cross_entropy_rejects_bad_target_sums():
    with pytest.raises(ValueError, match="sum to 1"):
        soft_target_cross_entropy(torch.zeros(2, 4), torch.tensor([[1., 0, 0, 0], [0.2, 0.2, 0.2, 0.2]]))


def test_hard_ce_and_label_smoothing_baseline():
    logits = torch.tensor([[2., 0., -1., 0.], [0., 1., 2., -1.]])
    labels = torch.tensor([0, 2])
    plain = hard_label_cross_entropy(logits, labels)
    smooth = hard_label_cross_entropy(logits, labels, label_smoothing=0.1)
    assert plain.ndim == 0 and smooth.ndim == 0
    assert torch.isfinite(plain) and torch.isfinite(smooth)
    assert plain != smooth


def test_js_divergence_properties_and_gradient():
    p = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]], requires_grad=True)
    q = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0., 0., 0., 1.]])
    values = jensen_shannon_divergence(p, q, reduction="none")
    assert values[0].item() == pytest.approx(0.0, abs=1e-6)
    assert 0 <= values[1].item() <= 1.0 + 1e-6
    values.mean().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_probability_normalization_rejects_zero_mass():
    with pytest.raises(ValueError, match="positive mass"):
        normalize_probabilities(torch.zeros(2, 4))
