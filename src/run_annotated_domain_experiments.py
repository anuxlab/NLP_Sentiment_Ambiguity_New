"""Distribution-aware TF-IDF experiments for MVSC and SentiMP-En.

Run the dataset audit first:
  python src/annotated_domains.py --output results/annotated_domain_audit.json
Then:
  python src/run_annotated_domain_experiments.py --dataset mvsc --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from annotated_domains import LABELS, audit_records, load_mvsc, load_sentimp_en, split_records
from config import RES


DATASETS = {
    "mvsc": (load_mvsc, [50, 100, 300, 700, 1500]),
    "sentimp_en": (load_sentimp_en, [25, 50, 100, 200, 300]),
}


def js_divergence(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    eps = 1e-12
    pred = np.clip(pred, eps, 1.0)
    target = np.clip(target, eps, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)
    target = target / target.sum(axis=1, keepdims=True)
    mid = 0.5 * (pred + target)
    return 0.5 * np.sum(pred * np.log2(pred / mid), axis=1) + 0.5 * np.sum(target * np.log2(target / mid), axis=1)


def sample_rows(rows, budget: int, seed: int, *, require_gold: bool) -> list[dict]:
    eligible = [r for r in rows if (r["gold_label_id"] is not None or not require_gold)]
    if require_gold:
        # Stratified sampling of hard-label examples only; paired across conditions.
        by_class = {c: [r for r in eligible if r["gold_label_id"] == c] for c in range(len(LABELS))}
        rng = np.random.RandomState(seed)
        selected = []
        remaining = min(budget, len(eligible))
        while remaining:
            active = [c for c, group in by_class.items() if group]
            if not active:
                break
            for c in active:
                if remaining <= 0:
                    break
                pick = int(rng.randint(len(by_class[c])))
                selected.append(by_class[c].pop(pick))
                remaining -= 1
        rng.shuffle(selected)
        return selected
    rng = np.random.RandomState(seed)
    if budget >= len(eligible):
        return list(eligible)
    idx = rng.choice(len(eligible), size=budget, replace=False)
    return [eligible[int(i)] for i in idx]


def fit_eval(train_rows, test_rows, condition, seed, args):
    start = time.time()
    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True, strip_accents="unicode")
    x_train = vectorizer.fit_transform([r["text"] for r in train_rows]).toarray().astype(np.float32)
    x_test = vectorizer.transform([r["text"] for r in test_rows]).toarray().astype(np.float32)
    targets = np.asarray([r["target_probs"] for r in train_rows], dtype=np.float32)
    gold = np.asarray([r["gold_label_id"] if r["gold_label_id"] is not None else -1 for r in train_rows], dtype=np.int64)
    if condition != "human_soft_targets" and (gold < 0).any():
        raise ValueError(f"{condition} requires non-tied majority labels")
    torch.manual_seed(seed)
    model = torch.nn.Linear(x_train.shape[1], len(LABELS))
    torch.nn.init.xavier_uniform_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    xt = torch.from_numpy(x_train)
    xv = torch.from_numpy(x_test)
    yt = torch.from_numpy(targets)
    yg = torch.from_numpy(gold)
    rng = np.random.RandomState(seed)
    for _ in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_rows))
        for start_idx in range(0, len(order), args.batch_size):
            ids = torch.from_numpy(order[start_idx:start_idx + args.batch_size])
            logits = model(xt[ids])
            if condition == "human_soft_targets":
                loss = -(yt[ids] * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            elif condition == "majority_ce":
                loss = torch.nn.functional.cross_entropy(logits, yg[ids])
            elif condition == "label_smoothing":
                loss = torch.nn.functional.cross_entropy(logits, yg[ids], label_smoothing=args.label_smoothing)
            else:
                raise ValueError(condition)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(xv), dim=-1).numpy()
    target = np.asarray([r["target_probs"] for r in test_rows], dtype=np.float32)
    gold_test = np.asarray([r["gold_label_id"] if r["gold_label_id"] is not None else -1 for r in test_rows])
    js = js_divergence(probs, target)
    valid = gold_test >= 0
    result = {
        "dataset": args.dataset, "condition": condition, "budget": len(train_rows), "seed": seed,
        "n_test": len(test_rows), "js_divergence_mean_bits": float(js.mean()),
        "soft_brier_score": float(np.mean(np.sum((probs - target) ** 2, axis=1))),
        "accuracy": float(accuracy_score(gold_test[valid], probs[valid].argmax(1))) if valid.any() else None,
        "macro_f1": float(f1_score(gold_test[valid], probs[valid].argmax(1), labels=list(range(len(LABELS))),
                                   average="macro", zero_division=0)) if valid.any() else None,
        "train_seconds": time.time() - start,
        "n_tied_majority_test": int((~valid).sum()),
    }
    return result


def main(args):
    loader, configured_budgets = DATASETS[args.dataset]
    records = loader()
    audit = audit_records(args.dataset, records, seed=args.split_seed)
    if audit["status"] == "FAIL" or not audit["all_splits_disjoint"]:
        raise SystemExit(f"Dataset failed split integrity audit: {audit}")
    if audit["status"] == "REVIEW" and not args.allow_review:
        raise SystemExit("Dataset audit is REVIEW. Inspect duplicate conflicts; pass --allow-review only after review.")
    splits = split_records(records, seed=args.split_seed)
    train, dev, test = splits["train"], splits["dev"], splits["test"]
    if args.budgets:
        budgets = args.budgets
    else:
        budgets = configured_budgets
    budgets = sorted(set(min(int(n), len([r for r in train if r["gold_label_id"] is not None])) for n in budgets))
    out = RES / "annotated_domain_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    records_out = []
    for budget in budgets:
        hard_train = sample_rows(train, budget, args.seeds[0], require_gold=True)
        if len(hard_train) < 2:
            continue
        for seed in args.seeds:
            # Same sampled hard-label rows for majority CE, smoothing and soft CE.
            matched = sample_rows(train, budget, seed, require_gold=True)
            for condition in ("majority_ce", "human_soft_targets", "label_smoothing"):
                # Hard-target conditions require non-tied rows; soft-target condition can also use ties,
                # but the matched comparison intentionally uses the same non-tied examples.
                result = fit_eval(matched, test, condition, seed, args)
                result["dev_size"] = len(dev)
                records_out.append(result)
                with out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result) + "\n")
                print(json.dumps(result), flush=True)
    audit_path = RES / f"{args.dataset}_annotated_domain_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records_out)} runs to {out}; audit: {audit_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--allow-review", action="store_true")
    main(parser.parse_args())
