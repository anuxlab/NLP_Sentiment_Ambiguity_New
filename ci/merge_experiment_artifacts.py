#!/usr/bin/env python3
"""Merge domain experiment artifacts, tolerating artifacts that contain global results.

Some workflow runs upload the complete accumulated results tree from every matrix
job. This merger therefore filters rows and prediction files to the requested
domain before combining them, while validating domain coverage and collisions.
It performs post-processing only; it never trains models.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

JSONL_FILES = ("results.jsonl", "families.jsonl")
DOMAINS = ("sst2", "finance", "airline")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object in {path}:{line_no}")
        rows.append(row)
    return rows


def row_domain(row: dict) -> str:
    # Legacy SST-2 rows predate the explicit domain field.
    return str(row.get("domain") or "sst2")


def find_domain_dir(root: Path, domain: str) -> Path:
    candidates = sorted(root.glob(f"*{domain}*"))
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "results.jsonl").exists():
            return candidate
        nested = candidate / "results"
        if (nested / "results.jsonl").exists():
            return nested
    raise FileNotFoundError(
        f"No downloaded artifact with results.jsonl for domain={domain} under {root}"
    )


def prediction_belongs_to_domain(filename: str, domain: str) -> bool:
    """Infer domain from prediction naming convention.

    SST-2 predictions use legacy unsuffixed names; Finance/Airline predictions
    carry their domain in the filename. Global artifacts can contain all three.
    """
    lower = filename.lower()
    is_finance = "finance" in lower
    is_airline = "airline" in lower
    if domain == "finance":
        return is_finance
    if domain == "airline":
        return is_airline
    return not is_finance and not is_airline


def copy_domain_predictions(pred_src: Path, out: Path, domain: str,
                            seen_pred_names: set[str]) -> int:
    if not pred_src.is_dir():
        raise FileNotFoundError(f"{domain}: prediction directory is missing: {pred_src}")
    files = sorted(p for p in pred_src.glob("*.npy")
                   if prediction_belongs_to_domain(p.name, domain))
    if not files:
        raise FileNotFoundError(
            f"{domain}: no prediction arrays matched the domain filename convention in {pred_src}"
        )
    for file in files:
        if file.name in seen_pred_names:
            raise ValueError(f"Prediction filename collision while merging: {file.name}")
        seen_pred_names.add(file.name)
        shutil.copy2(file, out / "preds" / file.name)
    return len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=Path("results"))
    ap.add_argument("--domains", nargs="+", choices=DOMAINS, required=True)
    args = ap.parse_args()
    if len(set(args.domains)) != len(args.domains):
        raise ValueError(f"Duplicate domains requested: {args.domains}")

    root, out = args.input_root, args.output_root
    out.mkdir(parents=True, exist_ok=True)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    merged: dict[str, list[dict]] = {name: [] for name in JSONL_FILES}
    seen_pred_names: set[str] = set()
    seen_rows: dict[str, set[str]] = {name: set() for name in JSONL_FILES}

    for domain in args.domains:
        src = find_domain_dir(root, domain)
        all_results = read_jsonl(src / "results.jsonl")
        if not all_results:
            raise ValueError(f"{domain}: results.jsonl is empty")
        domain_rows = [row for row in all_results if row_domain(row) == domain]
        if not domain_rows:
            found = sorted({row_domain(row) for row in all_results})
            raise ValueError(f"{domain}: no rows found for this domain; artifact contains {found}")

        all_families = read_jsonl(src / "families.jsonl")
        family_rows = [row for row in all_families if row_domain(row) == domain]
        if not family_rows:
            raise ValueError(f"{domain}: no family rows found for this domain")

        # De-duplicate exact JSON records within each domain artifact. This is
        # useful if artifacts contain accumulated results from repeated jobs.
        for filename, rows in (("results.jsonl", domain_rows),
                               ("families.jsonl", family_rows)):
            for row in rows:
                key = json.dumps(row, sort_keys=True, ensure_ascii=False)
                if key not in seen_rows[filename]:
                    seen_rows[filename].add(key)
                    merged[filename].append(row)

        pred_src = src / "preds"
        copied = copy_domain_predictions(pred_src, out, domain, seen_pred_names)

        # SST-2 job owns the efficiency report and shared MLM pretraining log.
        if domain == "sst2":
            for rel in (Path("efficiency.json"), Path("logs/pretrain.jsonl")):
                src_file = src / rel
                if src_file.exists():
                    target = out / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, target)

        print(
            f"[merge] {domain}: selected {len(domain_rows)}/{len(all_results)} result rows, "
            f"{len(family_rows)}/{len(all_families)} family rows, {copied} prediction arrays"
        )

    for name, rows in merged.items():
        if not rows:
            raise ValueError(f"Merged {name} is empty")
        target = out / name
        with target.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    domains_found = sorted({row_domain(r) for r in merged["results.jsonl"]})
    expected = sorted(args.domains)
    if domains_found != expected:
        raise ValueError(f"Merged domain coverage mismatch: expected {expected}, got {domains_found}")

    print(
        f"[merge] complete: {len(merged['results.jsonl'])} result rows, "
        f"{len(merged['families.jsonl'])} family rows; domains={domains_found}; "
        f"prediction_files={len(seen_pred_names)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
