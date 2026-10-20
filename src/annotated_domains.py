"""Load and audit multi-annotator sentiment datasets for distribution-aware experiments.

The loaders preserve observed annotator labels and derive target probabilities only
from those labels. They never turn confidence/agreement metadata into probabilities.
"""
from __future__ import annotations

import csv
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from config import DATA

LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
MVSC_URL = "https://raw.githubusercontent.com/MIND-Lab/MVSC/main/MVSC%20Tweet%20Corpus.txt"
SENTIMP_URL = "https://huggingface.co/datasets/rbnuria/SentiMP-En/resolve/main/en.csv"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return path
    request = urllib.request.Request(url, headers={"User-Agent": "NLP-Sentiment-Research/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    path.write_bytes(payload)
    return path


def _canon(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lstrip("\ufeff").casefold())


def _read_table(path: Path) -> list[dict[str, str]]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    if not raw.strip():
        raise ValueError(f"Empty dataset file: {path}")
    sample = raw[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if "\t" in sample.splitlines()[0] else ","
    rows = list(csv.DictReader(raw.splitlines(), delimiter=delimiter))
    if not rows or not rows[0]:
        raise ValueError(f"Could not parse a header and rows from {path}")
    return rows


def _field(row: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    by_key = {_canon(k): v for k, v in row.items() if k is not None}
    for alias in aliases:
        if _canon(alias) in by_key:
            return by_key[_canon(alias)]
    return None


def _label_id(value: Any) -> int | None:
    if value is None:
        return None
    v = str(value).strip().casefold()
    mapping = {
        "negative": 0, "neg": 0, "-1": 0,
        "neutral": 1, "neu": 1, "0": 1,
        "positive": 2, "pos": 2, "1": 2,
    }
    return mapping.get(v)


def _make_record(text: str, annotations: list[int | None], source: str, row_index: int,
                 metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    text = text.strip()
    valid = [int(x) for x in annotations if x is not None]
    if not text or len(valid) < 2:
        return None
    counts = np.bincount(valid, minlength=3).astype(int)
    total = int(counts.sum())
    probs = (counts / total).tolist()
    max_count = int(counts.max())
    winners = np.flatnonzero(counts == max_count).tolist()
    gold = int(winners[0]) if len(winners) == 1 else None
    return {
        "text": text,
        "normalized_text": normalize_text(text),
        "annotator_labels": valid,
        "vote_counts": counts.tolist(),
        "target_probs": probs,
        "num_votes": total,
        "gold_label_id": gold,
        "gold_label": LABELS[gold] if gold is not None else None,
        "source": source,
        "source_row": row_index,
        "metadata": metadata or {},
    }


def load_mvsc(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load MVSC tweet corpus with three annotators; output uses negative/neutral/positive."""
    path = Path(path) if path else DATA / "mvsc_tweet_corpus.txt"
    _download(MVSC_URL, path)
    rows = _read_table(path)
    out = []
    for i, row in enumerate(rows, start=2):
        text = _field(row, ("tweet_text", "tweet text", "text"))
        if text is None:
            raise ValueError(f"MVSC schema lacks tweet_text/text; found {list(rows[0])}")
        ann = []
        for n in (1, 2, 3):
            value = _field(row, (f"Tweet_Sentiment_A{n}", f"Tweet Sentiment A{n}", f"sentiment_a{n}"))
            ann.append(_label_id(value))
        if not any(x is not None for x in ann):
            # Header/metadata lines or malformed records are not silently interpreted as labels.
            continue
        rec = _make_record(text, ann, "mvsc", i, {
            "tweet_id": _field(row, ("tweet_id",)),
            "keyword": _field(row, ("keyword",)),
            "irony": [_field(row, (f"Irony_A{n}",)) for n in (1, 2, 3)],
            "subjectivity": [_field(row, (f"Subjectivity_A{n}",)) for n in (1, 2, 3)],
        })
        if rec:
            out.append(rec)
    if not out:
        raise ValueError("MVSC loader found no rows with at least two valid annotator sentiment labels")
    return out


def load_sentimp_en(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load SentiMP-En; numeric labels -1/0/1 map to negative/neutral/positive."""
    path = Path(path) if path else DATA / "sentimp_en.csv"
    _download(SENTIMP_URL, path)
    rows = _read_table(path)
    out = []
    for i, row in enumerate(rows, start=2):
        text = _field(row, ("full_text", "text", "tweet_text"))
        if text is None:
            raise ValueError(f"SentiMP-En schema lacks full_text/text; found {list(rows[0])}")
        ann = [_label_id(_field(row, (f"label_{n}",))) for n in (1, 2, 3)]
        if not any(x is not None for x in ann):
            continue
        rec = _make_record(text, ann, "sentimp_en", i, {
            "fold": _field(row, ("fold",)),
            "provided_majority_vote": _field(row, ("majority_vote",)),
            "provided_gold_label": _field(row, ("gold_label",)),
            "tie_break": _field(row, ("tie_break",)),
        })
        if rec:
            out.append(rec)
    if not out:
        raise ValueError("SentiMP-En loader found no rows with at least two valid annotator labels")
    return out


def deduplicate_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep first source occurrence per normalized text; report rather than merge duplicates."""
    seen: dict[str, dict[str, Any]] = {}
    duplicate_groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        key = row["normalized_text"]
        if key in seen:
            duplicate_groups.setdefault(key, [seen[key]]).append(row)
        else:
            seen[key] = row
    kept = list(seen.values())
    conflicts = []
    for key, group in duplicate_groups.items():
        vote_vectors = {tuple(x["vote_counts"]) for x in group}
        if len(vote_vectors) > 1:
            conflicts.append({"normalized_text": key[:160], "source_rows": [x["source_row"] for x in group],
                              "vote_counts": [x["vote_counts"] for x in group]})
    return kept, {
        "input_rows": len(records), "unique_normalized_texts": len(kept),
        "duplicate_rows_removed": len(records) - len(kept),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_groups_with_different_vote_counts": len(conflicts),
        "conflicting_duplicate_examples": conflicts[:25],
        "policy": "keep first source occurrence; do not merge duplicate records or treat them as additional annotator votes",
    }


def split_records(records: list[dict[str, Any]], seed: int = 42,
                  fractions: tuple[float, float, float] = (.7, .1, .2)) -> dict[str, list[dict[str, Any]]]:
    """Deterministic, majority-label-stratified split with tied-majority rows in a separate stratum."""
    if not records:
        raise ValueError("No records to split")
    if not np.isclose(sum(fractions), 1.0) or any(x <= 0 for x in fractions):
        raise ValueError("fractions must be positive and sum to 1")
    unique, duplicate_report = deduplicate_records(records)
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in unique:
        key = str(row["gold_label_id"]) if row["gold_label_id"] is not None else "tie"
        strata.setdefault(key, []).append(row)
    rng = np.random.RandomState(seed)
    result = {"train": [], "dev": [], "test": []}
    for key in sorted(strata):
        group = list(strata[key])
        order = rng.permutation(len(group))
        group = [group[int(i)] for i in order]
        n = len(group)
        n_train = int(round(n * fractions[0]))
        n_dev = int(round(n * fractions[1]))
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_dev = min(n_dev, n - n_train - 1)
        result["train"].extend(group[:n_train])
        result["dev"].extend(group[n_train:n_train+n_dev])
        result["test"].extend(group[n_train+n_dev:])
    for rows in result.values():
        rng.shuffle(rows)
    # Preserve duplicate diagnostics as a non-row field in the returned dict.
    result["_audit"] = duplicate_report
    return result


def audit_records(dataset_name: str, records: list[dict[str, Any]], seed: int = 42) -> dict[str, Any]:
    splits = split_records(records, seed=seed)
    audit = {"dataset": dataset_name, "raw_records": len(records), "labels": list(LABELS),
             "split_fractions": {"train": .7, "dev": .1, "test": .2},
             "deduplication": splits.pop("_audit"), "splits": {}}
    text_sets = {}
    for name, rows in splits.items():
        counts = Counter(r["gold_label_id"] if r["gold_label_id"] is not None else "tie" for r in rows)
        text_sets[name] = {r["normalized_text"] for r in rows}
        audit["splits"][name] = {
            "rows": len(rows), "class_or_tie_counts": dict(counts),
            "rows_with_tied_majority": sum(r["gold_label_id"] is None for r in rows),
            "vote_count_distribution": dict(Counter(r["num_votes"] for r in rows)),
            "mean_vote_entropy_bits": float(np.mean([
                -sum(p * np.log2(p) for p in r["target_probs"] if p > 0) for r in rows
            ])) if rows else None,
        }
    overlap = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            overlap[f"{a}__{b}"] = len(text_sets[a] & text_sets[b])
    audit["normalized_text_overlap_across_splits"] = overlap
    audit["all_splits_disjoint"] = all(v == 0 for v in overlap.values())
    audit["status"] = "PASS" if audit["all_splits_disjoint"] and not audit["deduplication"]["duplicate_groups_with_different_vote_counts"] else "REVIEW"
    return audit


def main(output: str | Path) -> dict[str, Any]:
    reports = []
    for name, loader in (("mvsc", load_mvsc), ("sentimp_en", load_sentimp_en)):
        records = loader()
        reports.append(audit_records(name, records))
    report = {"status": "PASS" if all(x["status"] == "PASS" for x in reports) else "REVIEW",
              "audit_scope": "raw-source annotation and split-integrity audit; no model training",
              "datasets": reports,
              "caveat": "PASS verifies loader/split invariants, not external annotation validity or licensing. Review source terms before redistribution."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "datasets": [{"name": x["dataset"], "rows": x["raw_records"], "status": x["status"]} for x in reports]}, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DATA.parent / "results" / "annotated_domain_audit.json"))
    args = parser.parse_args()
    main(args.output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit multi-annotator sentiment datasets")
    parser.add_argument("--output", default=str(DATA.parent / "results" / "annotated_domain_audit.json"))
    args = parser.parse_args()
    main(args.output)
