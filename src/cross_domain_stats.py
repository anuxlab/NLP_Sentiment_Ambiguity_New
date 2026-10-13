"""Paired bootstrap comparisons for saved multi-domain predictions.

This script does not retrain models. It downloads/loads the evaluation labels via
src/data.py and pairs each saved prediction array with the same test examples.
Intervals resample test examples jointly across models, after averaging paired
metric differences over the available seeds. They are conditional on the saved
seeds; between-seed variation is reported separately and is not hidden inside the
sentence-bootstrap interval.

Run from src/ after the domain experiments have completed:
    python cross_domain_stats.py --bootstrap 2000
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from config import DOMAIN_TRAIN_SIZES, PREDS, RES
from data import load_airline, load_finance, load_sst2

DOMAINS = ("sst2", "finance", "airline")
MODELS = ("tiny_mlm", "tiny_scratch", "tfidf_lr", "fasttext_avg_lr")


def get_test_labels(domain):
    if domain == "sst2":
        return load_sst2()["test"][1]
    splits, _ = (load_finance() if domain == "finance" else load_airline())
    return splits["test"][1]


def prediction_path(domain, model, n, seed, root):
    family = model == "fasttext_avg_lr"
    prefix = "family_" if family else "curve_"
    if domain == "sst2":
        return root / f"{prefix}{model}__{n}_{seed}.npy"
    return root / f"{prefix}{model}__{domain}_{n}_{seed}.npy"


def paired_bootstrap_difference(preds_a, preds_b, y, metric="accuracy", n_boot=2000, seed=0):
    """Return mean A-B metric difference and sentence-bootstrap percentile CI.

    preds_* are lists of (N,2) probability arrays, one per seed. The same sampled
    sentence indices are used for A and B within each bootstrap replicate.
    """
    k = min(len(preds_a), len(preds_b))
    if k == 0:
        raise ValueError("No paired prediction runs")
    if any(p.shape != (len(y), 2) for p in list(preds_a[:k]) + list(preds_b[:k])):
        raise ValueError("Prediction shape does not match the test labels")

    def score(p, yy):
        pred = p.argmax(axis=1)
        if metric == "accuracy":
            return accuracy_score(yy, pred)
        if metric == "macro_f1":
            return f1_score(yy, pred, average="macro", zero_division=0)
        raise ValueError(metric)

    seed_diffs = np.array([score(a, y) - score(b, y) for a, b in zip(preds_a[:k], preds_b[:k])])
    point = float(seed_diffs.mean())
    rng = np.random.RandomState(seed)
    n = len(y)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        vals = [score(preds_a[j][idx], y[idx]) - score(preds_b[j][idx], y[idx]) for j in range(k)]
        boots[i] = np.mean(vals)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"metric": metric, "difference": point, "ci95_low": float(lo), "ci95_high": float(hi),
            "seed_differences": seed_diffs.tolist(), "seed_sd": float(seed_diffs.std(ddof=1)) if k > 1 else 0.0,
            "n_seeds": k, "n_test": n, "bootstrap_replicates": n_boot,
            "uncertainty_scope": "paired test-example bootstrap conditional on saved seeds; seed SD reported separately"}


def run(root=PREDS, n_boot=2000):
    root = Path(root)
    output = []
    for domain in DOMAINS:
        y = get_test_labels(domain)
        sizes = DOMAIN_TRAIN_SIZES[domain]
        for n in sizes:
            available = {}
            for model in MODELS:
                runs = {}
                for seed in (0, 1, 2):
                    p = prediction_path(domain, model, n, seed, root)
                    if p.exists():
                        runs[seed] = np.load(p)
                if runs:
                    available[model] = runs
            pairs = (("tiny_mlm", "tiny_scratch"), ("tiny_mlm", "tfidf_lr"),
                     ("tiny_scratch", "tfidf_lr"), ("tiny_mlm", "fasttext_avg_lr"))
            for a, b in pairs:
                if a not in available or b not in available:
                    continue
                common_seeds = sorted(set(available[a]) & set(available[b]))
                if not common_seeds:
                    continue
                preds_a = [available[a][seed] for seed in common_seeds]
                preds_b = [available[b][seed] for seed in common_seeds]
                for metric in ("accuracy", "macro_f1"):
                    seed_key = zlib.crc32(f"{domain}|{n}|{a}|{b}|{metric}".encode("utf-8"))
                    stat = paired_bootstrap_difference(preds_a, preds_b, y, metric, n_boot, seed_key)
                    output.append({"domain": domain, "n_labels": n, "model_a": a, "model_b": b, **stat})
    out_json = RES / "domain_pairwise_bootstrap.json"
    out_json.write_text(json.dumps(output, indent=2))
    lines = ["# Multi-domain paired bootstrap comparisons", "",
             "Positive differences favour model A. CIs resample test examples jointly and are conditional on the saved seeds; seed-level SD is reported separately.",
             "These are not substitutes for more independent training seeds.", "",
             "| Domain | Labels | Comparison (A - B) | Metric | Difference (pp) | 95% CI (pp) | Seed SD (pp) |", "|---|---:|---|---|---:|---:|---:|"]
    for r in output:
        lines.append(f"| {r['domain']} | {r['n_labels']} | {r['model_a']} − {r['model_b']} | {r['metric']} | "
                     f"{100*r['difference']:+.2f} | [{100*r['ci95_low']:+.2f}, {100*r['ci95_high']:+.2f}] | {100*r['seed_sd']:.2f} |")
    out_md = RES / "domain_pairwise_bootstrap.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_json} and {out_md} ({len(output)} comparisons)")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", default=str(PREDS))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    run(args.preds, args.bootstrap)
