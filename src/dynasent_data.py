"""DynaSent v1.1 loader for four-class human-distribution experiments.

The primary task preserves DynaSent's four recorded categories in this order:
negative, neutral, positive, mixed. This module is deliberately offline-first:
it reads an official archive or an extracted directory and never downloads data.

Leakage policy: for each round, normalized sentences present in dev or test are
excluded from the effective training view. Dev/test are never edited. Optional
training deduplication retains the first occurrence in source-file order.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

LABELS: tuple[str, ...] = ("negative", "neutral", "positive", "mixed")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

ROUND_FILES = {
    "round1": {
        "train": "dynasent-v1.1-round01-yelp-train.jsonl",
        "dev": "dynasent-v1.1-round01-yelp-dev.jsonl",
        "test": "dynasent-v1.1-round01-yelp-test.jsonl",
    },
    "round2": {
        "train": "dynasent-v1.1-round02-dynabench-train.jsonl",
        "dev": "dynasent-v1.1-round02-dynabench-dev.jsonl",
        "test": "dynasent-v1.1-round02-dynabench-test.jsonl",
    },
}


def normalize_sentence(text: str) -> str:
    """Whitespace-normalized, case-folded sentence key for overlap detection."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def _read_jsonl_lines(raw: str, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {filename}:{line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {filename}:{line_no}")
        rows.append(value)
    return rows


def _read_member_from_zip_bytes(raw_zip: bytes, member: str, *, source_label: str, depth: int = 0) -> str:
    """Read a JSONL member, also supporting the repository ZIP containing the data ZIP."""
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        matches = [name for name in archive.namelist() if name.endswith(member)]
        if matches:
            return archive.read(matches[0]).decode("utf-8")
        # The public GitHub repository archive wraps the official data archive in
        # dynasent-main/dynasent-v1.1.zip. Recurse only one level to avoid arbitrary nesting.
        if depth < 1:
            nested = [name for name in archive.namelist() if name.endswith("dynasent-v1.1.zip")]
            for name in nested:
                try:
                    return _read_member_from_zip_bytes(
                        archive.read(name), member, source_label=f"{source_label}!{name}", depth=depth + 1
                    )
                except FileNotFoundError:
                    continue
    raise FileNotFoundError(f"Could not find {member} in {source_label}")


def _read_member(source: Path, member: str) -> str:
    if source.is_dir():
        matches = list(source.rglob(member))
        if matches:
            return matches[0].read_text(encoding="utf-8")
        nested_archives = list(source.rglob("dynasent-v1.1.zip"))
        for nested in nested_archives:
            try:
                return _read_member_from_zip_bytes(nested.read_bytes(), member, source_label=str(nested))
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"Could not find {member} under {source}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if not zipfile.is_zipfile(source):
        raise ValueError(f"Expected a ZIP archive or extracted directory: {source}")
    return _read_member_from_zip_bytes(source.read_bytes(), member, source_label=str(source))


def _vote_counts(row: dict[str, Any]) -> list[int]:
    dist = row.get("label_distribution")
    if not isinstance(dist, dict):
        raise ValueError("label_distribution must be a mapping of category to worker-ID lists")
    counts: list[int] = []
    for label in LABELS:
        workers = dist.get(label, [])
        if not isinstance(workers, list):
            raise ValueError(f"label_distribution[{label!r}] must be a list")
        counts.append(len(workers))
    unknown = set(dist) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected label_distribution categories: {sorted(unknown)}")
    if sum(counts) == 0:
        raise ValueError("label_distribution contains no votes")
    return counts


def _prepare_row(row: dict[str, Any], *, source_file: str, row_index: int) -> dict[str, Any]:
    sentence = row.get("sentence")
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError(f"{source_file}[{row_index}]: sentence must be a non-empty string")
    counts = _vote_counts(row)
    total = sum(counts)
    gold = row.get("gold_label")
    if gold is not None and gold not in LABEL_TO_ID:
        raise ValueError(f"{source_file}[{row_index}]: unknown gold_label {gold!r}")
    probs = [count / total for count in counts]
    # Defensive normalization avoids float drift while preserving the vote ratio.
    probs = [p / sum(probs) for p in probs]
    return {
        "text": sentence.strip(),
        "text_id": row.get("text_id"),
        "gold_label": gold,
        "gold_label_id": LABEL_TO_ID.get(gold),
        "label_distribution": dict(zip(LABELS, probs)),
        "target_probs": probs,
        "vote_counts": dict(zip(LABELS, counts)),
        "num_votes": total,
        "normalized_text": normalize_sentence(sentence),
        "source_file": source_file,
        "source_row": row_index,
    }


def ternary_distribution(
    four_class_probs: Iterable[float], *, mixed_policy: Literal["renormalize_non_mixed", "drop"] = "renormalize_non_mixed"
) -> list[float] | None:
    """Sensitivity-analysis mapping from [neg, neutral, pos, mixed] to 3 classes.

    `renormalize_non_mixed` removes mixed vote mass then renormalizes negative,
    neutral and positive. Returns None if all vote mass is mixed. `drop` always
    returns None for rows with any mixed vote mass, as a stricter sensitivity.
    This function is not used for the four-class primary task.
    """
    values = list(four_class_probs)
    if len(values) != 4 or any(v < 0 for v in values):
        raise ValueError("Expected four non-negative probabilities in canonical label order")
    total = sum(values)
    if total <= 0:
        raise ValueError("Probability vector must have positive mass")
    values = [v / total for v in values]
    if mixed_policy == "drop" and values[3] > 0:
        return None
    non_mixed = values[:3]
    mass = sum(non_mixed)
    if mass <= 0:
        return None
    return [v / mass for v in non_mixed]


@dataclass(frozen=True)
class DynaSentSplits:
    """One DynaSent round with prepared rows and a training-only leakage filter."""
    round_name: str
    train: list[dict[str, Any]]
    dev: list[dict[str, Any]]
    test: list[dict[str, Any]]
    audit: dict[str, Any]


def load_dynasent_round(
    source: str | Path,
    round_name: Literal["round1", "round2"] = "round1",
    *,
    filter_train_overlap: bool = True,
    deduplicate_train: bool = True,
) -> DynaSentSplits:
    """Load one DynaSent round from ZIP or extracted directory, preserving four labels.

    Filtering applies only to the training view. It removes train examples whose
    normalized sentence appears in dev/test, then optionally removes duplicate
    normalized sentences within training. Dev/test remain exactly as supplied.
    """
    if round_name not in ROUND_FILES:
        raise ValueError(f"round_name must be one of {tuple(ROUND_FILES)}")
    source = Path(source)
    raw_splits: dict[str, list[dict[str, Any]]] = {}
    prepared: dict[str, list[dict[str, Any]]] = {}
    for split, filename in ROUND_FILES[round_name].items():
        raw_splits[split] = _read_jsonl_lines(_read_member(source, filename), filename)
        prepared[split] = [
            _prepare_row(row, source_file=filename, row_index=i)
            for i, row in enumerate(raw_splits[split])
        ]

    protected_texts = {
        row["normalized_text"]
        for split in ("dev", "test")
        for row in prepared[split]
    }
    original_train_n = len(prepared["train"])
    overlap_removed = 0
    dedup_removed = 0
    filtered_train: list[dict[str, Any]] = []
    seen_train: set[str] = set()
    for row in prepared["train"]:
        key = row["normalized_text"]
        if filter_train_overlap and key in protected_texts:
            overlap_removed += 1
            continue
        if deduplicate_train and key in seen_train:
            dedup_removed += 1
            continue
        seen_train.add(key)
        filtered_train.append(row)

    audit = {
        "round": round_name,
        "source": str(source),
        "label_order": list(LABELS),
        "split_rows_before_filtering": {k: len(v) for k, v in prepared.items()},
        "train_rows_before_filtering": original_train_n,
        "train_overlap_rows_removed": overlap_removed,
        "train_duplicate_rows_removed": dedup_removed,
        "train_rows_after_filtering": len(filtered_train),
        "dev_rows_unchanged": len(prepared["dev"]),
        "test_rows_unchanged": len(prepared["test"]),
        "train_filter_enabled": filter_train_overlap,
        "train_deduplication_enabled": deduplicate_train,
        "null_gold_label_rows": {
            split: sum(row["gold_label"] is None for row in rows)
            for split, rows in prepared.items()
        },
        "mixed_vote_rows": {
            split: sum(row["vote_counts"]["mixed"] > 0 for row in rows)
            for split, rows in prepared.items()
        },
    }
    return DynaSentSplits(round_name, filtered_train, prepared["dev"], prepared["test"], audit)
