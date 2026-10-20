"""Run controlled TF-IDF experiments on vote-distribution datasets.

All conditions share one PyTorch multinomial linear classifier, TF-IDF features,
initialization policy, AdamW optimizer, weight decay, full-batch update schedule,
and evaluation pipeline. For the first three conditions, the training examples and
feature vocabulary are identical; only target construction differs.

Input JSONL schema (one example per line):
  {"text": "...", "split": "train|dev|test",
   "votes": {"negative": 2, "neutral": 1, "positive": 0}}
Optional `gold_label` is accepted, but ties in observed vote counts are always
unlabeled for hard-label metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score


CONDITIONS = (
    "majority_ce",
    "human_soft_targets",
    "label_smoothing",
    "human_soft_targets_tie_inclusive",
)


def parse_rows(path, labels):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        text = str(raw.get("text", "")).strip()
        split = raw.get("split")
        if not text or split not in {"train", "dev", "test"}:
            raise ValueError(f"line {line_no}: text and split=train|dev|test required")

        votes = raw.get("votes")
        if votes is None and "target_probs" in raw:
            probs = np.asarray(raw["target_probs"], dtype=np.float64)
        elif isinstance(votes, dict):
            probs = np.asarray([float(votes.get(label, 0)) for label in labels], dtype=np.float64)
        else:
            raise ValueError(f"line {line_no}: provide votes mapping or target_probs")

        if (
            probs.shape != (len(labels),)
            or not np.isfinite(probs).all()
            or (probs < 0).any()
            or probs.sum() <= 0
        ):
            raise ValueError(f"line {line_no}: invalid vote distribution")

        probs = probs / probs.sum()
        winners = np.flatnonzero(np.isclose(probs, probs.max()))
        is_tie = len(winners) > 1
        gold = raw.get("gold_label")
        # Never allow a supplied arbitrary label to break an observed vote tie.
        if is_tie:
            gold = None
        elif gold not in labels:
            gold = labels[int(winners[0])]

        rows.append({
            "text": text,
            "split": split,
            "target_probs": probs,
            "gold_label": gold,
            "is_tie": is_tie,
        })

    for split in ("train", "dev", "test"):
        if not any(row["split"] == split for row in rows):
            raise ValueError(f"missing split: {split}")

    normalized = {
        split: {row["text"].casefold().strip() for row in rows if row["split"] == split}
        for split in ("train", "dev", "test")
    }
    if (
        normalized["train"] & normalized["dev"]
        or normalized["train"] & normalized["test"]
        or normalized["dev"] & normalized["test"]
    ):
        raise ValueError("normalized text overlap across splits")
    return rows


def jsd(p, q):
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    midpoint = (p + q) / 2
    return (
        0.5 * np.sum(p * np.log2(p / midpoint), axis=1)
        + 0.5 * np.sum(q * np.log2(q / midpoint), axis=1)
    )


def stable_seed(seed, budget, condition):
    """Deterministic condition-specific initialization, matched across conditions."""
    # Model initialization must be identical for the matched first three conditions.
    return int(seed + 100_003 * budget) % (2**31 - 1)


def make_targets(rows, labels, condition, smoothing):
    targets = np.stack([row["target_probs"] for row in rows]).astype(np.float32)
    if condition in ("majority_ce", "label_smoothing"):
        hard = np.asarray([
            labels.index(row["gold_label"]) if row["gold_label"] in labels else -1
            for row in rows
        ])
        if (hard < 0).any():
            raise ValueError(f"{condition} cannot train on tied rows")
        targets = np.eye(len(labels), dtype=np.float32)[hard]
        if condition == "label_smoothing":
            targets = (1.0 - smoothing) * targets + smoothing / len(labels)
    return targets


def fit_predict(X_train, target_rows, X_test, labels, condition, seed, budget, smoothing,
                epochs, learning_rate, weight_decay):
    """Train the shared linear model; condition only changes targets/training rows."""
    targets = make_targets(target_rows, labels, condition, smoothing)

    # Matched conditions share exactly the same initialization for a given seed/budget.
    torch.manual_seed(stable_seed(seed, budget, condition))
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Full-batch training gives every condition the same number of optimizer updates.
    x = torch.as_tensor(np.asarray(X_train, dtype=np.float32))
    y = torch.as_tensor(targets, dtype=torch.float32)
    xt = torch.as_tensor(np.asarray(X_test, dtype=np.float32))

    model = torch.nn.Linear(x.shape[1], len(labels))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = -(y * torch.log_softmax(model(x), dim=1)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return torch.softmax(model(xt), dim=1).cpu().numpy()


def row_fingerprint(rows):
    payload = "\n".join(row["text"].casefold().strip() for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--labels", nargs="+", default=["negative", "neutral", "positive"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[25, 50, 100, 200, 300])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-features", type=int, default=30000)
    args = parser.parse_args()

    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be in [0, 1)")
    if args.epochs < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("epochs and learning rate must be positive; weight decay cannot be negative")
    if any(budget < 1 for budget in args.budgets):
        parser.error("budgets must be positive integers")

    rows = parse_rows(args.input, args.labels)
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    non_tied = [row for row in train if row["gold_label"] in args.labels]
    tied_train = [row for row in train if row["gold_label"] not in args.labels]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    predictions = []
    audits = []

    for budget in args.budgets:
        if len(non_tied) < budget:
            raise ValueError(f"budget {budget} exceeds non-tied train rows ({len(non_tied)})")
        for seed in args.seeds:
            rng = np.random.RandomState(seed)
            chosen_indices = rng.choice(len(non_tied), budget, replace=False)
            matched = [non_tied[int(i)] for i in chosen_indices]

            extra_count = min(budget // 3, len(tied_train))
            if extra_count:
                tie_indices = rng.choice(len(tied_train), extra_count, replace=False)
                tie_inclusive = matched + [tied_train[int(i)] for i in tie_indices]
            else:
                tie_inclusive = matched.copy()

            # Fit one vectorizer on the matched subset and reuse it for every condition.
            # This keeps the vocabulary and feature representation fixed even in the
            # separately reported tie-inclusive condition.
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=args.max_features,
                sublinear_tf=True,
            )
            X_matched = vectorizer.fit_transform([row["text"] for row in matched]).toarray()
            X_tie = vectorizer.transform([row["text"] for row in tie_inclusive]).toarray()
            X_test = vectorizer.transform([row["text"] for row in test]).toarray()

            # Assertions make the intended matched design explicit.
            assert X_matched.shape[1] == X_tie.shape[1] == X_test.shape[1]
            assert len(matched) == budget

            condition_rows = {
                "majority_ce": matched,
                "human_soft_targets": matched,
                "label_smoothing": matched,
                "human_soft_targets_tie_inclusive": tie_inclusive,
            }
            condition_features = {
                "majority_ce": X_matched,
                "human_soft_targets": X_matched,
                "label_smoothing": X_matched,
                "human_soft_targets_tie_inclusive": X_tie,
            }

            # Same seed/budget initialization across all conditions: reset before
            # model creation with the same seed. Condition must not alter initialization.
            for condition in CONDITIONS:
                start = time.time()
                pred = fit_predict(
                    condition_features[condition],
                    condition_rows[condition],
                    X_test,
                    args.labels,
                    condition,
                    seed,
                    budget,
                    args.label_smoothing,
                    args.epochs,
                    args.learning_rate,
                    args.weight_decay,
                )
                elapsed = time.time() - start

                target_probs = np.stack([row["target_probs"] for row in test])
                per_row_js = jsd(pred, target_probs)
                brier = float(np.mean(np.sum((pred - target_probs) ** 2, axis=1)))
                hard_indices = [
                    i for i, row in enumerate(test) if row["gold_label"] in args.labels
                ]
                y_true = [args.labels.index(test[i]["gold_label"]) for i in hard_indices]
                y_hat = pred[hard_indices].argmax(axis=1)

                record = {
                    "dataset": Path(args.input).stem,
                    "seed": seed,
                    "budget_requested": budget,
                    "condition": condition,
                    "n_train": len(condition_rows[condition]),
                    "n_tied_train": sum(row["is_tie"] for row in condition_rows[condition]),
                    "n_test": len(test),
                    "n_tied_test": sum(row["is_tie"] for row in test),
                    "n_hard_test": len(hard_indices),
                    "js_divergence_bits": float(per_row_js.mean()),
                    "soft_brier_score": brier,
                    "accuracy": float(accuracy_score(y_true, y_hat)) if hard_indices else None,
                    "macro_f1": float(f1_score(y_true, y_hat, average="macro", zero_division=0))
                    if hard_indices else None,
                    "train_time_s": elapsed,
                    "model": "pytorch_linear_multinomial",
                    "optimizer": "AdamW",
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "epochs": args.epochs,
                    "label_smoothing": args.label_smoothing if condition == "label_smoothing" else None,
                    "vocab_size": len(vectorizer.vocabulary_),
                    "matched_train_fingerprint": row_fingerprint(matched),
                }
                metrics.append(record)
                for i, (row, probabilities) in enumerate(zip(test, pred)):
                    predictions.append({
                        "seed": seed,
                        "budget_requested": budget,
                        "condition": condition,
                        "text": row["text"],
                        "target_probs": row["target_probs"].tolist(),
                        "predicted_probs": probabilities.tolist(),
                        "js_divergence_bits": float(per_row_js[i]),
                    })
                print(json.dumps(record), flush=True)

            audits.append({
                "seed": seed,
                "budget": budget,
                "matched_train_fingerprint": row_fingerprint(matched),
                "n_matched": len(matched),
                "n_tie_inclusive": len(tie_inclusive),
                "n_tied_added": len(tie_inclusive) - len(matched),
                "vocab_size": len(vectorizer.vocabulary_),
                "matched_conditions_share_features": True,
                "tie_inclusive_uses_same_vocabulary": True,
            })

    (out_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in metrics), encoding="utf-8"
    )
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in predictions), encoding="utf-8"
    )
    audit = {
        "input": str(args.input),
        "labels": args.labels,
        "split_sizes": {split: sum(row["split"] == split for row in rows)
                        for split in ("train", "dev", "test")},
        "tied_rows_by_split": {split: sum(
            row["split"] == split and row["is_tie"] for row in rows
        ) for split in ("train", "dev", "test")},
        "normalized_split_overlap": "none",
        "conditions": list(CONDITIONS),
        "controlled_design": {
            "shared_model": "PyTorch Linear + softmax",
            "shared_optimizer": "AdamW",
            "shared_learning_rate": args.learning_rate,
            "shared_weight_decay": args.weight_decay,
            "shared_epochs": args.epochs,
            "training": "full-batch; same number of updates",
            "features": "one TF-IDF vectorizer fitted on matched non-tied subset per budget/seed",
            "matched_conditions": "same examples, feature vocabulary, model initialization, optimizer and schedule; target construction differs",
            "tie_inclusive": "same matched examples plus up to floor(budget/3) tied examples; reported separately because training composition differs",
            "limitation": "full-batch AdamW and fixed epochs are shared; this experiment does not tune hyperparameters per condition",
        },
        "run_audits": audits,
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
