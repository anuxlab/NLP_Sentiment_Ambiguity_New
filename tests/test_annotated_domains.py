import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from annotated_domains import LABELS, _make_record, audit_records, deduplicate_records, split_records


def rec(text, labels, row):
    return _make_record(text, labels, "synthetic", row)


def test_vote_distribution_comes_only_from_observed_annotator_labels():
    row = rec("Good result", [2, 2, 0], 2)
    assert row["vote_counts"] == [1, 0, 2]
    assert np.allclose(row["target_probs"], [1/3, 0, 2/3])
    assert row["gold_label_id"] == 2
    assert row["num_votes"] == 3


def test_tied_majority_has_distribution_but_no_hard_gold():
    row = rec("Mixed response", [0, 1, 2], 3)
    assert row["target_probs"] == [1/3, 1/3, 1/3]
    assert row["gold_label_id"] is None


def test_duplicate_text_is_reported_not_merged_as_extra_votes():
    rows = [rec("Same text", [2, 2, 2], 2), rec(" same   TEXT ", [0, 0, 0], 3)]
    kept, report = deduplicate_records(rows)
    assert len(kept) == 1
    assert report["duplicate_rows_removed"] == 1
    assert report["duplicate_groups_with_different_vote_counts"] == 1


def test_split_is_disjoint_and_preserves_vote_distributions():
    rows = []
    i = 0
    for label in range(3):
        for j in range(20):
            labels = [label, label, (label + 1) % 3]
            rows.append(rec(f"text class {label} example {j}", labels, i))
            i += 1
    splits = split_records(rows, seed=17)
    assert set(splits) == {"train", "dev", "test", "_audit"}
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    sets = [{r["normalized_text"] for r in part} for part in (train, dev, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert all(np.isclose(sum(r["target_probs"]), 1.0) for part in (train, dev, test) for r in part)


def test_audit_marks_conflicting_duplicate_records_for_review():
    rows = [rec("dup", [2, 2, 2], 1), rec(" DUP ", [0, 0, 0], 2)]
    report = audit_records("synthetic", rows)
    assert report["status"] == "REVIEW"
    assert report["all_splits_disjoint"]


def test_mvsc_loader_reads_individual_annotator_columns(tmp_path):
    from annotated_domains import load_mvsc
    path = tmp_path / "mvsc.tsv"
    path.write_text(
        "tweet_id\ttweet_text\tuser_id\tkeyword\tTweet_Sentiment_A1\tTweet_Sentiment_A2\tTweet_Sentiment_A3\n"
        "1\tGreat movie\t1\tdeadpool\tPositive\tPositive\tNegative\n"
        "2\tNot my thing\t2\tdeadpool\tNegative\tNeutral\tNegative\n",
        encoding="utf-8",
    )
    rows = load_mvsc(path)
    assert len(rows) == 2
    assert rows[0]["vote_counts"] == [1, 0, 2]
    assert rows[0]["target_probs"] == [1/3, 0.0, 2/3]


def test_sentimp_loader_maps_numeric_labels_and_preserves_ties(tmp_path):
    from annotated_domains import load_sentimp_en
    path = tmp_path / "sentimp.csv"
    path.write_text(
        "full_text,fold,label_1,label_2,label_3,majority_vote,tie_break,gold_label\n"
        '"Positive text",0,1,1,0,1,,1\n'
        '"Ambiguous text",1,-1,0,1,TIE,0,0\n',
        encoding="utf-8",
    )
    rows = load_sentimp_en(path)
    assert rows[0]["vote_counts"] == [0, 1, 2]
    assert rows[0]["gold_label_id"] == 2
    assert rows[1]["vote_counts"] == [1, 1, 1]
    assert rows[1]["gold_label_id"] is None
