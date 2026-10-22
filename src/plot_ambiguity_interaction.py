
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/ambiguity_paired_analysis/paired_predictions.csv"
OUT = ROOT / "results/ambiguity_paired_analysis"

df = pd.read_csv(INPUT)
datasets = ["DynaSent Round 1", "DynaSent Round 2"]
groups = ["H=0", "H>0"]
labels = {
    "H=0": "Unanimous (H=0)",
    "H>0": "Non-unanimous (H>0)",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
summary = []

for ax, dataset in zip(axes, datasets):
    sub = df[df["dataset"] == dataset]

    for group in groups:
        points = []

        for budget in sorted(sub["budget"].unique()):
            g = sub[
                (sub["budget"] == budget)
                & (sub["ambiguity_group"] == group)
            ]

            # Average over examples within each seed first.
            seed_means = g.groupby("seed")["delta_js"].mean()
            values = seed_means.to_numpy()
            n = len(values)

            mean = values.mean()
            se = values.std(ddof=1) / np.sqrt(n)
            half_width = t.ppf(0.975, n - 1) * se

            points.append((int(budget), mean, half_width))
            summary.append({
                "dataset": dataset,
                "budget": int(budget),
                "ambiguity_group": group,
                "n_seeds": n,
                "mean_delta_js": mean,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            })

        points.sort()
        x = np.array([p[0] for p in points])
        means = np.array([p[1] for p in points])
        errors = np.array([p[2] for p in points])

        ax.errorbar(
            x, means, yerr=errors,
            marker="o", capsize=4, linewidth=2,
            label=labels[group],
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(dataset)
    ax.set_xlabel("Label budget")
    ax.set_xticks([25, 50, 100, 200, 300])
    ax.grid(True, alpha=0.25)
    ax.legend()

axes[0].set_ylabel(
    "JS change: human soft targets − majority CE\n"
    "(negative values favor human soft targets)"
)

fig.suptitle("Effect of label ambiguity on sentiment prediction")
fig.tight_layout()

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "ambiguity_interaction.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "ambiguity_interaction.pdf", bbox_inches="tight")
plt.close(fig)

pd.DataFrame(summary).to_csv(
    OUT / "ambiguity_subgroup_effects.csv", index=False
)

print(f"Saved: {OUT / 'ambiguity_interaction.png'}")
print(f"Saved: {OUT / 'ambiguity_interaction.pdf'}")
print(f"Saved: {OUT / 'ambiguity_subgroup_effects.csv'}")
print(
    "Intervals are t-based 95% intervals across 10 seed-level means; "
    "they are not multiplicity-adjusted."
)
