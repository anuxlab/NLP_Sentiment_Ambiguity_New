#!/usr/bin/env python3
"""Audit archived experiment evidence without training or downloading data.

This checks record completeness and arithmetic consistency. It cannot prove
that the original training run was scientifically valid or reproducible.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.relative_to(ROOT)}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.relative_to(ROOT)}:{lineno}: expected JSON object")
        rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/evidence_audit.json")
    args = parser.parse_args()

    results_path = ROOT / "results/results.jsonl"
    family_path = ROOT / "results/families.jsonl"
    results = read_jsonl(results_path)
    families = read_jsonl(family_path)
    errors: list[str] = []
    warnings: list[str] = []

    if not results:
        errors.append("results/results.jsonl is missing or empty")
    if not families:
        warnings.append("results/families.jsonl is missing or empty")

    domains = Counter(str(r.get("domain", "sst2")) for r in results)
    model_counts = Counter((str(r.get("domain", "sst2")), str(r.get("stage")),
                            str(r.get("model"))) for r in results)

    # For each model record, validate numeric fields without asserting a target score.
    for i, row in enumerate(results):
        for key in ("test_acc", "test_f1"):
            if key in row:
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    errors.append(f"result row {i}: {key} is not numeric")
                    continue
                if not 0.0 <= value <= 1.0:
                    errors.append(f"result row {i}: {key} outside [0,1]: {value}")
        if "seed" in row and not isinstance(row["seed"], int):
            errors.append(f"result row {i}: seed is not an integer")

    expected = {"sst2": 6920, "finance": 1377, "airline": 8125}
    core_models = ("tiny_mlm", "tiny_scratch", "tfidf_lr")
    full_run_checks = []
    for domain, n in expected.items():
        for model in core_models:
            selected = [r for r in results if str(r.get("domain", "sst2")) == domain
                        and r.get("stage") == "curve" and r.get("model") == model
                        and r.get("n") == n]
            seeds = sorted(r.get("seed") for r in selected)
            passed = len(selected) == 3 and seeds == [0, 1, 2]
            full_run_checks.append({"domain": domain, "model": model, "budget": n,
                                    "records": len(selected), "seeds": seeds,
                                    "three_seed_complete": passed})
            if not passed:
                warnings.append(f"full-budget evidence incomplete: {domain}/{model}, n={n}, seeds={seeds}")

    artifact_paths = [results_path, family_path,
                      ROOT / "report/tables/tab_cross_domain.tex",
                      ROOT / "report/tables/tab_fin_air_f1.tex"]
    artifacts = []
    for path in artifact_paths:
        if path.exists():
            artifacts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                              "sha256": sha256(path)})
        else:
            warnings.append(f"optional evidence artifact missing: {path.relative_to(ROOT)}")

    report = {
        "audit_schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "saved records and generated assets only; no training or dataset download",
        "status": "FAIL" if errors else ("WARN" if warnings else "PASS"),
        "counts": {"result_records": len(results), "family_records": len(families),
                   "records_by_domain": dict(domains),
                   "model_stage_record_counts": {"|".join(k): v for k, v in model_counts.items()}},
        "full_budget_three_seed_checks": full_run_checks,
        "artifacts": artifacts,
        "errors": errors,
        "warnings": warnings,
        "interpretation_limit": (
            "A passing audit establishes record/schema/arithmetic consistency only. It does not "
            "validate the original training run, eliminate leakage, establish statistical significance, "
            "or replace an end-to-end reproduction."
        ),
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "result_records": len(results),
                      "family_records": len(families), "errors": errors,
                      "warnings": len(warnings), "output": str(output)}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
