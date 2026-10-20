import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import _stratified_split


def test_split_uses_explicit_sentiment_label_not_metadata():
    rows = [
        ("positive a", 1, "low"), ("positive b", 1, "high"),
        ("negative a", 0, "high"), ("negative b", 0, "low"),
        ("positive c", 1, "low"), ("negative c", 0, "high"),
    ]
    tr, dev, te = _stratified_split(rows, seed=7, label_index=1, text_index=0)
    assert sorted(tr + dev + te) == list(range(len(rows)))
    assert not (set(tr) & set(dev) or set(tr) & set(te) or set(dev) & set(te))


def test_conflicting_duplicate_labels_stay_in_same_split_without_crashing():
    rows = [
        ("Same text!", 1, 0.8), (" same   text! ", 0, 0.9),
        ("other positive", 1, 0.8), ("other negative", 0, 0.7),
        ("positive extra", 1, 0.9), ("negative extra", 0, 0.9),
        ("positive extra 2", 1, 0.9), ("negative extra 2", 0, 0.9),
        ("positive extra 3", 1, 0.9), ("negative extra 3", 0, 0.9),
    ]
    parts = _stratified_split(rows, seed=42, label_index=1, text_index=0)
    split_by_idx = {idx: split_name for split_name, part in zip(("train", "dev", "test"), parts) for idx in part}
    assert split_by_idx[0] == split_by_idx[1]
    assert sorted(split_by_idx) == list(range(len(rows)))
