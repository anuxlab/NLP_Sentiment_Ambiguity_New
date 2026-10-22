
#!/usr/bin/env python3
"""Two-level paired bootstrap for human soft targets vs majority CE.

Resamples seeds and test examples, preserving treatment pairing.
Uses stored target and predicted distributions; no retraining.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DYNA_ROOT = ROOT / "results" / "dynasent_controlled"
SENTIMP_FILE = ROOT / "results" / "sentimp_en_controlled_seed10" / "predictions.jsonl"
OUT = ROOT / "results" / "cross_dataset_analysis_seed10" / "two_level_bootstrap"

BASELINE = "majority_ce"
TREATMENT = "human_soft_targets"
N_BOOT = 10_000
RNG_SEED = 20261020
EPS = 1e-12


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return rows


def js_divergence_bits(target, prediction):
    """Compute base-2 Jensen-Shannon divergence for two distributions."""
    p = np.asarray(target, dtype=float)
    q = np.asarray(prediction, dtype=float)

    if p.ndim != 1 or q.ndim != 1 or len(p) != len(q):
        raise ValueError("Target and prediction vectors have incompatible shapes")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Non-finite probability values found")
    if (p < -1e-8).any() or (q < -1e-8).any():
        raise ValueError("Negative probabilities found")
    if not np.isclose(p.sum(), 1.0, atol=1e-5):
        raise ValueError(f"Target probabilities do not sum to 1: {p}")
    if not np.isclose(q.sum(), 1.0, atol=1e-5):
        raise ValueError(f"Predicted probabilities do not sum to 1: {q}")

    p = np.clip(p, 0, None)
    q = np.clip(q, 0, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def load_dyna():
    rows = []
    for path in sorted(DYNA_ROOT.glob("round*/budget_*/predictions.jsonl")):
        for r in read_jsonl(path):
            if r.get("condition") not in {BASELINE, TREATMENT}:
                continue
            rows.append({
                "dataset": f"DynaSent {r['round']}",
                "dataset_key": r["round"],
                "budget": int(path.parent.name.replace("budget_", "")),
                "seed": int(r["seed"]),
                "condition": r["condition"],
                "example_id": str(r["text_id"]),
                "target": r["human_distribution"],
                "prediction": r["predicted_distribution"],
            })
    return rows


def load_sentimp():
    rows = []
    for r in read_jsonl(SENTIMP_FILE):
        if r.get("condition") not in {BASELINE, TREATMENT}:
            continue
        rows.append({
            "dataset": "SentiMP-En",
            "dataset_key": "sentimp_en",
            "budget": int(r["budget_requested"]),
            "seed": int(r["seed"]),
            "condition": r["condition"],
            # The text is used as a provisional key. Duplicate texts within
            # a seed/condition are rejected below rather than silently merged.
            "example_id": str(r["text"]),
            "target": r["target_probs"],
            "prediction": r["predicted_probs"],
        })
    return rows


def prepare_group(group, dataset, budget):
    """Validate and align treatment/control predictions by seed and example."""
    group = group.copy()

    group["js"] = [
        js_divergence_bits(target, prediction)
        for target, prediction in zip(group["target"], group["prediction"])
    ]

    key = ["seed", "condition", "example_id"]
    duplicates = group.duplicated(key, keep=False)
    if duplicates.any():
        sample = group.loc[duplicates, key].head(10)
        raise ValueError(
            f"Duplicate seed/condition/example records in {dataset}, "
            f"budget {budget}:\n{sample.to_string(index=False)}"
        )

    wide = group.pivot(
        index=["seed", "example_id"],
        columns="condition",
        values="js",
    )

    if BASELINE not in wide or TREATMENT not in wide:
        raise ValueError(f"Missing treatment or baseline in {dataset}, {budget}")

    if wide[[BASELINE, TREATMENT]].isna().any().any():
        raise ValueError(
            f"Unmatched test examples between conditions in {dataset}, {budget}"
        )

    # Confirm the reference target is invariant across conditions and seeds.
    targets = group.pivot(
        index=["seed", "example_id"],
        columns="condition",
        values="target",
    )
    for _, row in targets.iterrows():
        if row[BASELINE] != row[TREATMENT]:
            raise ValueError(
                f"Target distribution mismatch in {dataset}, budget {budget}"
            )

    # The bootstrap operates on a seed x example matrix of paired deltas.
    wide["delta"] = wide[TREATMENT] - wide[BASELINE]
    delta = wide["delta"].unstack("example_id").sort_index(axis=0)

    # Check identical example sets across seeds.
    example_sets = [
        set(wide.xs(seed, level="seed").index)
        for seed in sorted(wide.index.get_level_values("seed").unique())
    ]
    if example_sets and not all(s == example_sets[0] for s in example_sets):
        raise ValueError(
            f"Test example sets differ across seeds in {dataset}, budget {budget}"
        )

    return delta.to_numpy(dtype=float), list(delta.index), list(delta.columns)


def bootstrap_matrix(delta, rng):
    """Resample seeds and examples independently and average paired deltas."""
    n_seeds, n_examples = delta.shape
    if n_seeds < 2 or n_examples < 2:
        raise ValueError("At least two seeds and two examples are required")

    estimates = np.empty(N_BOOT, dtype=float)

    for b in range(N_BOOT):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        example_idx = rng.integers(0, n_examples, size=n_examples)
        sampled = delta[np.ix_(seed_idx, example_idx)]
        estimates[b] = sampled.mean()

    return estimates


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_dyna() + load_sentimp()

    if not rows:
        raise ValueError("No controlled prediction records found")

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(RNG_SEED)
    results = []

    for (dataset, budget), group in df.groupby(
        ["dataset", "budget"], sort=True
    ):
        delta, seeds, examples = prepare_group(group, dataset, budget)

        boot = bootstrap_matrix(delta, rng)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        observed = float(delta.mean())

        results.append({
            "dataset": dataset,
            "budget": int(budget),
            "n_seeds": int(delta.shape[0]),
            "n_test_examples": int(delta.shape[1]),
            "baseline": BASELINE,
            "treatment": TREATMENT,
            "mean_js_delta_bits": observed,
            "bootstrap_ci_95_low": float(lo),
            "bootstrap_ci_95_high": float(hi),
            "bootstrap_replicates": N_BOOT,
            "bootstrap_unit": "seeds_and_test_examples",
            "random_seed": RNG_SEED,
        })

        print(
            f"{dataset:18s} budget={budget:3d} "
            f"seeds={delta.shape[0]:2d} examples={delta.shape[1]:4d} "
            f"delta={observed:+.6f} CI=[{lo:+.6f}, {hi:+.6f}]"
        )

    result_df = pd.DataFrame(results)
    output = OUT / "paired_two_level_bootstrap.csv"
    result_df.to_csv(output, index=False)
    print(f"\nSaved {output.relative_to(ROOT)}")
    print(
        "\nInterpretation: negative deltas favor human soft targets. "
        "Intervals are exploratory, conditional on the selected datasets, "
        "splits, and training protocol; they are not multiplicity-adjusted."
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
