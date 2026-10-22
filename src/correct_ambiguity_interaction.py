
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/ambiguity_paired_analysis"

INPUT = BASE / "paired_predictions.csv"
BOOTSTRAP_INPUT = BASE / "ambiguity_interaction_bootstrap.csv"
OUTPUT = BASE / "ambiguity_interaction_holm.csv"


def exact_sign_flip_test(seed_effects):
    """
    Exact two-sided sign-flip test on seed-level interaction effects.

    Tests the null under the assumption that the seed-level effects
    are exchangeable and symmetric around zero.
    """
    effects = np.asarray(seed_effects, dtype=float)
    observed = abs(effects.mean())
    n = len(effects)

    if n == 0 or not np.isfinite(effects).all():
        raise ValueError("Seed effects must be non-empty and finite.")

    # Enumerate every possible sign assignment.
    signs = np.asarray(list(product([-1.0, 1.0], repeat=n)))
    null_stats = np.abs((signs * effects).mean(axis=1))

    # Exact test: no Monte Carlo correction is needed.
    return float(np.mean(null_stats >= observed - 1e-12))


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, preserving original order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running_max = 0.0

    for rank, idx in enumerate(order):
        running_max = max(running_max, (n - rank) * p[idx])
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


df = pd.read_csv(INPUT)

required = {
    "dataset",
    "budget",
    "seed",
    "example_id",
    "ambiguity_group",
    "delta_js",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {INPUT}: {sorted(missing)}")

datasets = ["DynaSent Round 1", "DynaSent Round 2"]
results = []

for dataset in datasets:
    sub = df[df["dataset"] == dataset]

    for budget in sorted(sub["budget"].unique()):
        part = sub[sub["budget"] == budget]
        seeds = np.sort(part["seed"].unique())
        matrices = {}

        for group in ["H=0", "H>0"]:
            g = part[part["ambiguity_group"] == group]

            if g.duplicated(["seed", "example_id"]).any():
                raise ValueError(
                    f"Duplicate seed/example rows: {dataset}, {budget}, {group}"
                )

            pivot = g.pivot(
                index="seed",
                columns="example_id",
                values="delta_js",
            ).reindex(seeds)

            if pivot.empty or pivot.isna().any().any():
                raise ValueError(
                    f"Missing seed/example pairs: {dataset}, {budget}, {group}"
                )

            matrices[group] = pivot.to_numpy(dtype=float)

        a = matrices["H=0"]
        b = matrices["H>0"]

        if len(seeds) != 10:
            raise ValueError(
                f"Expected 10 seeds, found {len(seeds)} "
                f"for {dataset}, budget {budget}"
            )

        # First calculate each subgroup effect separately for every seed.
        seed_mean_h0 = a.mean(axis=1)
        seed_mean_hpositive = b.mean(axis=1)

        # Negative delta JS means soft targets improve over majority CE.
        # Negative interaction means the improvement is greater for H>0.
        seed_interactions = seed_mean_hpositive - seed_mean_h0

        interaction = float(seed_interactions.mean())
        p_value = exact_sign_flip_test(seed_interactions)

        results.append({
            "dataset": dataset,
            "budget": int(budget),
            "n_seeds": len(seeds),
            "n_H0_examples": a.shape[1],
            "n_Hpositive_examples": b.shape[1],
            "mean_delta_js_H0": float(seed_mean_h0.mean()),
            "mean_delta_js_Hpositive": float(seed_mean_hpositive.mean()),
            "interaction_Hpositive_minus_H0": interaction,
            "interaction_p_value": p_value,
            "seed_interaction_sd": float(seed_interactions.std(ddof=1)),
        })

result = pd.DataFrame(results)

if len(result) != 10:
    raise ValueError(
        f"Expected 10 round-budget tests, found {len(result)}. "
        "Check dataset and budget coverage before correction."
    )

result["holm_adjusted_p"] = holm_adjust(result["interaction_p_value"])
result["significant_holm_005"] = result["holm_adjusted_p"] < 0.05

# Preserve the previous two-level bootstrap intervals for comparison.
if BOOTSTRAP_INPUT.exists():
    boot = pd.read_csv(BOOTSTRAP_INPUT)
    interval_cols = [
        "dataset",
        "budget",
        "bootstrap_ci_95_low",
        "bootstrap_ci_95_high",
    ]
    if set(interval_cols).issubset(boot.columns):
        result = result.merge(
            boot[interval_cols],
            on=["dataset", "budget"],
            how="left",
            validate="one_to_one",
        )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print(
    result[
        [
            "dataset",
            "budget",
            "mean_delta_js_H0",
            "mean_delta_js_Hpositive",
            "interaction_Hpositive_minus_H0",
            "interaction_p_value",
            "holm_adjusted_p",
            "significant_holm_005",
        ]
    ].round(6).to_string(index=False)
)

print(f"\nSaved: {OUTPUT}")
print(
    "\nInterpretation: negative interaction means a more favorable "
    "JS change on non-unanimous examples."
)
print(
    "Inference is across training seeds, conditional on the observed "
    "test examples. The sign-flip test assumes symmetric seed-level "
    "effects under the null. Bootstrap intervals remain exploratory "
    "and are not multiplicity-adjusted."
)
