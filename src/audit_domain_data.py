"""Forensic audit of the Financial PhraseBank and Twitter Airline data loaders.

Downloads the same public mirrors used by data.py, calls the production loaders,
and writes a machine-readable report. This is a data-integrity audit, not a model
experiment. Run from src/: python audit_domain_data.py --output ../results/domain_data_audit.json
"""
import argparse
import json
import re
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
        out[name] = {
            "n_rows": len(texts),
            "class_counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "class_proportions": {str(k): float(v / len(labels)) for k, v in sorted(counts.items())} if len(labels) else {},
            "empty_text_rows": int(sum(not x for x in normalized)),
            "duplicate_normalized_text_rows": int(len(normalized) - len(set(normalized))),
            "unique_normalized_texts": int(len(set(normalized))),
        }
    overlap = {}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a}__{b}"] = len(split_text_sets[a] & split_text_sets[b])
    return out, overlap


def audit_domain(name, loader):
    splits, metadata = loader(seed=42)
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
        "seed": 42,
        "task": "binary sentiment; neutral rows dropped by current loader",
        "split_fractions_requested": {"train": .7, "dev": .1, "test": .2},
        "splits": split_stats,
        "normalized_text_overlap_across_splits": overlaps,
        "metadata": metadata_stats,
        "all_splits_disjoint_by_normalized_text": all(v == 0 for v in overlaps.values()),
    }


def main(output):
    report = {
        "status": "PASS",
        "audit_scope": "live public-source download plus current production loader outputs",
        "caveat": "This audit does not create human vote distributions. Agreement tiers and worker confidence remain distinct metadata.",
        "domains": {},
    }
    for name, loader in (("finance", load_finance), ("airline", load_airline)):
        result = audit_domain(name, loader)
        report["domains"][name] = result
        if not result["all_splits_disjoint_by_normalized_text"]:
            report["status"] = "FAIL"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "counts": {d: {s: x["n_rows"] for s, x in v["splits"].items()}
                                 for d, v in report["domains"].items()}}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(RES / "domain_data_audit.json"))
    args = parser.parse_args()
    main(args.output)
