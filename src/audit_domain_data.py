"""Forensic audit of the Financial PhraseBank and Twitter Airline data loaders.

Downloads the same public mirrors used by data.py, calls the production loaders,
and writes a machine-readable report. This is a data-integrity audit, not a model
experiment. Run from src/: python audit_domain_data.py --output ../results/domain_data_audit.json
"""
import argparse
import csv
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from config import RES
from data import load_airline, load_finance


def norm(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def split_report(splits):
    out = {}
    split_text_sets = {}
    for name, (texts, labels) in splits.items():
        labels = np.asarray(labels)
        counts = Counter(int(x) for x in labels.tolist())
        normalized = [norm(t) for t in texts]
        split_text_sets[name] = set(normalized)
        labels_by_text = {}
        for text, label in zip(normalized, labels.tolist()):
            labels_by_text.setdefault(text, set()).add(int(label))
        conflicting_groups = {text: sorted(vals) for text, vals in labels_by_text.items() if len(vals) > 1}
        out[name] = {
            "n_rows": len(texts),
            "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "class_proportions": {str(k): float(v / len(labels)) for k, v in sorted(counts.items())} if len(labels) else {},
            "empty_text_rows": int(sum(not x for x in normalized)),
            "duplicate_normalized_text_rows": int(len(normalized) - len(set(normalized))),
            "unique_normalized_texts": int(len(set(normalized))),
            "conflicting_label_duplicate_groups": int(len(conflicting_groups)),
            "conflicting_label_examples": [
                {"normalized_text_prefix": text[:120], "labels": vals}
                for text, vals in list(sorted(conflicting_groups.items()))[:25]
            ],
        }
    overlap = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a}__{b}"] = len(split_text_sets[a] & split_text_sets[b])
    return out, overlap


def audit_domain(name, loader, seed=42):
    splits, metadata = loader(seed=seed)
    split_stats, overlaps = split_report(splits)
    test_meta = np.asarray(metadata)
    if name == "finance":
        values, counts = np.unique(test_meta.astype(str), return_counts=True)
        metadata_stats = {
            "kind": "annotator_agreement_tier",
            "counts": {str(k): int(v) for k, v in zip(values, counts)},
            "order": ["50-65", "66-99", "100"],
            "semantics": "tiered majority-label agreement; not a per-example class-probability distribution",
        }
    else:
        numeric = test_meta.astype(float)
        metadata_stats = {
            "kind": "crowd_worker_confidence",
            "min": float(np.min(numeric)), "max": float(np.max(numeric)),
            "mean": float(np.mean(numeric)),
            "quantiles": {str(q): float(np.quantile(numeric, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
            "semantics": "confidence metadata; not a per-example class-probability distribution",
        }
    return {
        "domain": name,
        "seed": seed,
        "task": "binary sentiment; neutral rows dropped by current loader",
        "split_fractions_requested": {"train": .7, "dev": .1, "test": .2},
        "splits": split_stats,
        "normalized_text_overlap_across_splits": overlaps,
        "metadata": metadata_stats,
        "all_splits_disjoint_by_normalized_text": all(v == 0 for v in overlaps.values()),
    }

AIRLINE_SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    "MrColinHan/Twitter-US-Airline-Sentiment/master/"
    "Kaggle_Tweets.csv"
)


def audit_airline_raw_source(splits, seed=42):
    """Inspect duplicate airline source records without changing labels."""
    request = urllib.request.Request(
        AIRLINE_SOURCE_URL,
        headers={"User-Agent": "NLP-Sentiment-Data-Audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read().decode("utf-8-sig")

    reader = csv.DictReader(content.splitlines())
    if not reader.fieldnames:
        raise ValueError("Airline CSV has no header")

    columns = {name.strip().lower(): name for name in reader.fieldnames}
    if "text" not in columns or "airline_sentiment" not in columns:
        raise ValueError(
            f"Unexpected airline CSV columns: {reader.fieldnames}"
        )

    def field(row, name):
        original = columns.get(name)
        return row.get(original, "") if original else ""

    # Map the production loader's actual partitions by normalized text.
    split_by_text = {}
    for split_name, (texts, labels) in splits.items():
        for text, label in zip(texts, labels):
            split_by_text.setdefault(norm(text), set()).add(split_name)

    groups = {}
    row_count = 0

    for row_index, row in enumerate(reader, start=2):
        row_count += 1
        text = field(row, "text")
        normalized = norm(text)
        if not normalized:
            continue

        groups.setdefault(normalized, []).append({
            "source_row_number": row_index,
            "text": text,
            "sentiment": field(row, "airline_sentiment"),
            "confidence": field(row, "airline_sentiment_confidence"),
            "airline": field(row, "airline"),
            "tweet_id": field(row, "tweet_id"),
            "negative_reason": field(row, "negativereason"),
        })

    duplicate_groups = []
    conflicting_groups = []

    for normalized, records in sorted(groups.items()):
        if len(records) < 2:
            continue

        labels = sorted({record["sentiment"] for record in records})
        partitions = sorted(split_by_text.get(normalized, set()))

        group = {
            "normalized_text": normalized,
            "record_count": len(records),
            "distinct_original_labels": labels,
            "conflicting_labels": len(labels) > 1,
            "production_split_assignments": partitions or ["excluded_by_loader"],
            "records": records,
        }
        duplicate_groups.append(group)

        if len(labels) > 1:
            conflicting_groups.append(group)

    return {
        "source_url": AIRLINE_SOURCE_URL,
        "source_row_count": row_count,
        "seed": seed,
        "duplicate_group_count": len(duplicate_groups),
        "conflicting_label_group_count": len(conflicting_groups),
        "duplicate_groups": duplicate_groups,
        "conflicting_label_groups": conflicting_groups,
        "interpretation": (
            "Source rows are records, not independent annotator votes. "
            "Original labels are preserved; no labels are corrected."
        ),
    }

def main(output, seed=42):
    report = {
        "status": "PASS",
        "audit_scope": "live public-source download plus current production loader outputs",
        "caveat": "This audit does not create human vote distributions. Agreement tiers and worker confidence remain distinct metadata.",
        "domains": {},
    }
    for name, loader in (("finance", load_finance), ("airline", load_airline)):
        result = audit_domain(name, loader, seed=seed)
        report["domains"][name] = result
        if not result["all_splits_disjoint_by_normalized_text"]:
            report["status"] = "FAIL"
        elif any(stats["conflicting_label_duplicate_groups"] for stats in result["splits"].values()):
            if report["status"] != "FAIL":
                report["status"] = "REVIEW"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate the detailed airline source audit inside main(),
    # where the integer seed and output path are available.
    airline_splits, _ = load_airline(seed=seed)
    raw_audit = audit_airline_raw_source(airline_splits, seed=seed)

    raw_output = output.parent / "airline_raw_source_audit.json"
    raw_output.write_text(
        json.dumps(raw_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "counts": {d: {s: x["n_rows"] for s, x in v["splits"].items()}
                                 for d, v in report["domains"].items()}}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(RES / "domain_data_audit.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.output, seed=args.seed)

    airline_splits, _ = load_airline(seed=seed)
    raw_audit = audit_airline_raw_source(airline_splits, seed=seed)

    raw_output = output.parent / "airline_raw_source_audit.json"
    raw_output.write_text(
        json.dumps(raw_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
