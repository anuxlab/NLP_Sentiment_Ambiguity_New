import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path(
    "results/ambiguity_paired_analysis/paired_predictions.csv"
)
OUT = Path("results/ambiguity_paired_analysis")
B = 10000
RNG_SEED = 20261022

df = pd.read_csv(INPUT)
rng = np.random.default_rng(RNG_SEED)
results = []

# Inferential comparison: DynaSent's two observed entropy groups.
# SentiMP-En's H=1 group has only two distinct examples and is
# therefore excluded from the interaction test.
datasets = ["DynaSent Round 1", "DynaSent Round 2"]

for dataset in datasets:
    sub = df[df["dataset"] == dataset]

    for budget in sorted(sub["budget"].unique()):
        part = sub[sub["budget"] == budget]
        seeds = np.sort(part["seed"].unique())

        matrices = {}
        for group in ["H=0", "H>0"]:
            g = part[part["ambiguity_group"] == group]
            pivot = g.pivot(
                index="seed",
                columns="example_id",
                values="delta_js",
            ).reindex(seeds)

            if pivot.isna().any().any():
                raise ValueError(
                    f"Missing seed/example pairs: {dataset}, {budget}, {group}"
                )

            matrices[group] = pivot.to_numpy(dtype=float)

        # Each matrix is seeds x examples. Resample seed indices jointly
        # across strata and resample examples separately within each stratum.
        a = matrices["H=0"]
        b = matrices["H>0"]

        interaction = b.mean() - a.mean()
        boot = np.empty(B)

        for k in range(B):
            seed_idx = rng.integers(0, len(seeds), size=len(seeds))
            a_idx = rng.integers(0, a.shape[1], size=a.shape[1])
            b_idx = rng.integers(0, b.shape[1], size=b.shape[1])

            mean_a = a[np.ix_(seed_idx, a_idx)].mean()
            mean_b = b[np.ix_(seed_idx, b_idx)].mean()
            boot[k] = mean_b - mean_a

        lo, hi = np.quantile(boot, [0.025, 0.975])

        results.append({
            "dataset": dataset,
            "budget": int(budget),
            "n_seeds": len(seeds),
            "n_H0_examples": a.shape[1],
            "n_Hpositive_examples": b.shape[1],
            "interaction_Hpositive_minus_H0": interaction,
            "bootstrap_ci_95_low": lo,
            "bootstrap_ci_95_high": hi,
            "bootstrap_replicates": B,
            "rng_seed": RNG_SEED,
        })

result = pd.DataFrame(results)
path = OUT / "ambiguity_interaction_bootstrap.csv"
result.to_csv(path, index=False)

print(result.round(6).to_string(index=False))
print(f"\nSaved: {path}")
print(
    "\nExploratory intervals: these are not multiplicity-adjusted, "
    "and do not establish a causal mechanism."
)
