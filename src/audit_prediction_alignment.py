
#!/usr/bin/env python3
"""Audit test-example alignment in controlled DynaSent and SentiMP-En runs."""

from pathlib import Path
from collections import Counter
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DYNA_ROOT = ROOT / "results" / "dynasent_controlled"
SENTIMP_FILE = ROOT / "results" / "sentimp_en_controlled" / "predictions.jsonl"
OUT = ROOT / "results" / "cross_dataset_analysis"
CONDITIONS = {
    "majority_ce",
    "human_soft_targets",
    "label_smoothing",
}


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
    return rows


def check_distribution(value, path, row_number):
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{path}: row {row_number} has no valid target distribution"
        )
    if any(not isinstance(x, (int, float)) for x in value):
        raise ValueError(f"{path}: nonnumeric target probability")
    if any(x < -1e-8 for x in value):
        raise ValueError(f"{path}: negative target probability")
    if abs(sum(value) - 1.0) > 1e-5:
        raise ValueError(
            f"{path}: target probabilities do not sum to 1: {value}"
        )


def audit_group(rows, label, id_field, target_field, pred_field):
    print(f"\n--- {label} ---")
    if not rows:
        print("ERROR: no records")
        return {"label": label, "status": "EMPTY"}

    frame = pd.DataFrame(rows)
    required = {
        "seed", "condition", id_field, target_field, pred_field
    }
    missing = required - set(frame.columns)
    if missing:
        print(f"ERROR: missing fields: {sorted(missing)}")
        return {"label": label, "status": "MISSING_FIELDS"}

    frame = frame[frame["condition"].isin(CONDITIONS)].copy()
    if frame.empty:
        print("ERROR: no matched conditions found")
        return {"label": label, "status": "NO_MATCHED_CONDITIONS"}

    for i, row in enumerate(rows, start=1):
        check_distribution(row[target_field], label, i)
        check_distribution(row[pred_field], label, i)

    duplicate_key = ["seed", "condition", id_field]
    duplicate_count = int(frame.duplicated(duplicate_key).sum())
    if duplicate_count:
        print(
            f"WARNING: {duplicate_count} duplicate seed/condition/example keys"
        )

    # One signature per example and seed. The target distribution must not
    # change when the condition changes.
    frame["_target"] = frame[target_field].apply(
        lambda x: tuple(round(float(v), 8) for v in x)
    )
    frame["_pred"] = frame[pred_field].apply(
        lambda x: tuple(round(float(v), 8) for v in x)
    )

    target_variants = (
        frame.groupby(["seed", id_field])["_target"]
        .nunique()
    )
    target_mismatches = int((target_variants > 1).sum())

    # Report test-example counts by seed and condition.
    counts = (
        frame.groupby(["seed", "condition"])[id_field]
        .nunique()
        .unstack(fill_value=0)
    )
    print("Unique test examples by seed and condition:")
    print(counts.to_string())

    # Check whether each seed has the same set of test examples across
    # all conditions.
    condition_sets = {}
    for (seed, condition), group in frame.groupby(
        ["seed", "condition"]
    ):
        condition_sets[(seed, condition)] = set(group[id_field])

    alignment_errors = []
    for seed in sorted(frame["seed"].unique()):
        sets = [
            condition_sets.get((seed, condition), set())
            for condition in sorted(CONDITIONS)
        ]
        if any(not s for s in sets):
            alignment_errors.append(
                f"seed {seed}: missing at least one condition"
            )
        elif not all(s == sets[0] for s in sets[1:]):
            alignment_errors.append(
                f"seed {seed}: test example IDs differ across conditions"
            )

    # Check target distributions are invariant across conditions.
    print(f"Duplicate keys: {duplicate_count}")
    print(f"Target distribution mismatches: {target_mismatches}")
    print(f"Condition alignment errors: {len(alignment_errors)}")
    for error in alignment_errors[:10]:
        print("  -", error)

    status = (
        "PASS"
        if duplicate_count == 0
        and target_mismatches == 0
        and not alignment_errors
        else "REVIEW"
    )

    print("STATUS:", status)
    return {
        "label": label,
        "status": status,
        "rows": len(frame),
        "duplicate_keys": duplicate_count,
        "target_mismatches": target_mismatches,
        "alignment_errors": len(alignment_errors),
    }


def audit_dynasent():
    reports = []
    files = sorted(DYNA_ROOT.glob("round*/budget_*/predictions.jsonl"))
    if not files:
        raise FileNotFoundError(f"No DynaSent predictions found under {DYNA_ROOT}")

    for path in files:
        rows = read_jsonl(path)
        label = f"DynaSent {path.parent.parent.name} budget={path.parent.name}"
        reports.append(
            audit_group(
                rows,
                label,
                id_field="text_id",
                target_field="human_distribution",
                pred_field="predicted_distribution",
            )
        )
    return reports


def audit_sentimp():
    if not SENTIMP_FILE.exists():
        raise FileNotFoundError(f"Missing {SENTIMP_FILE}")

    rows = read_jsonl(SENTIMP_FILE)

    # Text is a candidate key only. This checks duplicates first; if
    # duplicate text occurs, text alone is not a safe unique example ID.
    text_counts = Counter(row.get("text") for row in rows)
    duplicate_text_values = sum(
        count > 1 for count in text_counts.values()
    )
    print("\nSentiMP-En duplicate text values:", duplicate_text_values)

    if duplicate_text_values:
        print(
            "WARNING: text alone is not a unique example ID. "
            "Do not bootstrap by text until duplicates are resolved."
        )

    # Use exact text provisionally, and include budget in the audit group.
    reports = []
    frame = pd.DataFrame(rows)
    if "budget_requested" not in frame.columns:
        raise ValueError("SentiMP-En predictions lack budget_requested")

    for budget, group in frame.groupby("budget_requested", sort=True):
        reports.append(
            audit_group(
                group.to_dict(orient="records"),
                f"SentiMP-En budget={budget}",
                id_field="text",
                target_field="target_probs",
                pred_field="predicted_probs",
            )
        )
    return reports


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = audit_dynasent() + audit_sentimp()
    report_path = OUT / "prediction_alignment_audit.json"
    report_path.write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    print(f"\nAudit report saved to {report_path.relative_to(ROOT)}")

    if any(item["status"] != "PASS" for item in reports):
        print(
            "\nSome groups require review. Do not run paired "
            "example-level bootstrap until alignment issues are resolved."
        )
        raise SystemExit(2)

    print("\nAll inspected groups passed the alignment audit.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
