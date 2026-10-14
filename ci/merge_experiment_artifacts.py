#!/usr/bin/env python3
"""Merge per-domain GitHub Actions artifacts into the canonical results tree.

This is post-processing only: it never trains models and never overwrites the
input artifacts. It rejects missing domain artifacts and colliding prediction
filenames rather than silently mixing evidence.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

JSONL_FILES = ("results.jsonl", "families.jsonl")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
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


def find_domain_dir(root: Path, domain: str) -> Path:
    candidates = list(root.glob(f"*{domain}*"))
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "results.jsonl").exists():
            return candidate
        # Also accept artifacts downloaded with an extra results/ prefix.
        if (candidate / "results" / "results.jsonl").exists():
            return candidate / "results"
    raise FileNotFoundError(f"No downloaded artifact with results.jsonl for domain={domain} under {root}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, default=Path("results"))
    ap.add_argument("--domains", nargs="+", required=True)
    args = ap.parse_args()
    root, out = args.input_root, args.output_root
    out.mkdir(parents=True, exist_ok=True)
    (out / "preds").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)

    merged: dict[str, list[dict]] = {name: [] for name in JSONL_FILES}
    seen_pred_names: set[str] = set()
    for domain in args.domains:
        src = find_domain_dir(root, domain)
        domain_rows = read_jsonl(src / "results.jsonl")
        if not domain_rows:
            raise ValueError(f"{domain}: results.jsonl is empty")
        wrong = sorted({str(r.get("domain", "sst2")) for r in domain_rows
                        if str(r.get("domain", "sst2")) != domain})
        if wrong:
            raise ValueError(f"{domain}: found records for unexpected domains {wrong}")
        merged["results.jsonl"].extend(domain_rows)

        family_rows = read_jsonl(src / "families.jsonl")
        wrong_families = sorted({str(r.get("domain", "sst2")) for r in family_rows
                                 if str(r.get("domain", "sst2")) != domain})
        if wrong_families:
            raise ValueError(f"{domain}: family records contain unexpected domains {wrong_families}")
        merged["families.jsonl"].extend(family_rows)

        pred_src = src / "preds"
        if not pred_src.is_dir() or not any(pred_src.glob("*.npy")):
            raise FileNotFoundError(f"{domain}: prediction arrays are missing from {pred_src}")
        for file in pred_src.glob("*.npy"):
            if file.name in seen_pred_names:
                raise ValueError(f"Prediction filename collision while merging: {file.name}")
            seen_pred_names.add(file.name)
            shutil.copy2(file, out / "preds" / file.name)

        # Only SST-2 produces the efficiency report and shared MLM pretraining log.
        if domain == "sst2":
            for rel in (Path("efficiency.json"), Path("logs/pretrain.jsonl")):
                src_file = src / rel
                if src_file.exists():
                    target = out / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, target)
        print(f"[merge] {domain}: {len(domain_rows)} result rows, {len(family_rows)} family rows, "
              f"{sum(1 for _ in pred_src.glob('*.npy'))} prediction arrays")

    for name, rows in merged.items():
        if name == "families.jsonl" and not rows:
            raise ValueError("Merged families.jsonl is empty")
        target = out / name
        with target.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    domains_found = sorted({str(r.get("domain", "sst2")) for r in merged["results.jsonl"]})
    if domains_found != sorted(args.domains):
        raise ValueError(f"Merged domain coverage mismatch: expected {sorted(args.domains)}, got {domains_found}")
    print(f"[merge] merged {len(merged['results.jsonl'])} results and {len(merged['families.jsonl'])} family rows; "
          f"domains={domains_found}; prediction_files={len(seen_pred_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
