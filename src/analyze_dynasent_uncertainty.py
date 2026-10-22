
#!/usr/bin/env python3
"""Paired uncertainty analysis and budget-wise plots for DynaSent."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "dynasent_controlled"
OUT = RESULTS / "aggregate_analysis"
FIGURES = OUT / "figures"

BASELINE = "majority_ce"
SEED_COLS = ["round", "budget", "seed"]

METRIC_MAP = {
    "js_divergence": "js_divergence_mean_bits",
    "soft_brier": "soft_brier_score",
    "macro_f1": "macro_f1",
}

CONDITIONS = [
    "human_soft_targets",
    "human_soft_targets_null_augmented",
    "label_smoothing",
]

LABELS = {
    "human_soft_targets": "Human soft targets",
    "human_soft_targets_null_augmented": "Null-augmented",
    "label_smoothing": "Label smoothing",
    "majority_ce": "Majority CE",
}

METRIC_LABELS = {
    "js_divergence": "Mean JS divergence (bits; lower is better)",
    "soft_brier": "Soft Brier score (lower is better)",
    "macro_f1": "Macro-F1 (higher is better)",
}

N_BOOTSTRAP = 20_000
RANDOM_SEED = 20261020


def load_data():
    files = sorted(RESULTS.glob("round*/budget_*/metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics.csv files found under {RESULTS}")

    frames = []
    required = {
        "round", "budget", "seed", "condition",
        *METRIC_MAP.values(),
    }

    for path in files:
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path}: missing columns {sorted(missing)}; "
                f"available columns: {list(frame.columns)}"
            )

        frame = frame.rename(
            columns={
                actual: standard
                for standard, actual in METRIC_MAP.items()
            }
        )
        frame["seed"] = pd.to_numeric(
            frame["seed"], errors="raise"
        ).astype(int)
        frame["budget"] = pd.to_numeric(
            frame["budget"], errors="raise"
        ).astype(int)

        for metric in METRIC_MAP:
            frame[metric] = pd.to_numeric(
                frame[metric], errors="coerce"
            )

        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    if df[list(METRIC_MAP)].isna().any().any():
        raise ValueError("Missing or non-numeric metric values found.")

    keys = ["round", "budget", "seed", "condition"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate round/budget/seed/condition records.")

    return df


def paired_uncertainty(df):
    """
    Bootstrap paired seed-level differences.

    Each bootstrap resample samples seeds with replacement, preserving
    the pairing between treatment and baseline within each seed.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    output = []

    for (round_name, budget), group in df.groupby(
        ["round", "budget"], sort=True
    ):
        baseline = group[group["condition"] == BASELINE].set_index("seed")

        for condition in CONDITIONS:
            treatment = group[group["condition"] == condition].set_index("seed")

            common = sorted(set(baseline.index) & set(treatment.index))
            if len(common) < 2:
                print(
                    f"WARNING: only {len(common)} paired seeds for "
                    f"{round_name}, budget {budget}, {condition}"
                )
                continue

            for metric in METRIC_MAP:
                # Delta = treatment minus majority_ce.
                # Negative favors treatment for JS/Brier.
                # Positive favors treatment for macro-F1.
                delta = (
                    treatment.loc[common, metric].to_numpy(dtype=float)
                    - baseline.loc[common, metric].to_numpy(dtype=float)
                )

                boot_indices = rng.integers(
                    0, len(delta), size=(N_BOOTSTRAP, len(delta))
                )
                boot_means = delta[boot_indices].mean(axis=1)

                ci_low, ci_high = np.quantile(
                    boot_means, [0.025, 0.975]
                )

                output.append({
                    "round": round_name,
                    "budget": budget,
                    "condition": condition,
                    "metric": metric,
                    "n_paired_seeds": len(delta),
                    "mean_paired_delta": float(delta.mean()),
                    "sd_paired_delta": float(delta.std(ddof=1)),
                    "bootstrap_ci_95_low": float(ci_low),
                    "bootstrap_ci_95_high": float(ci_high),
                    "bootstrap_replicates": N_BOOTSTRAP,
                    "bootstrap_unit": "seed",
                    "baseline": BASELINE,
                })

    return pd.DataFrame(output)


def make_plots(df):
    FIGURES.mkdir(parents=True, exist_ok=True)

    rounds = sorted(df["round"].unique())
    budgets = sorted(df["budget"].unique())

    for metric in METRIC_MAP:
        fig, axes = plt.subplots(
            1, len(rounds),
            figsize=(7 * len(rounds), 4.8),
            squeeze=False,
        )
        axes = axes[0]

        for ax, round_name in zip(axes, rounds):
            round_df = df[df["round"] == round_name]

            for condition in [BASELINE, *CONDITIONS]:
                means = []
                stds = []

                for budget in budgets:
                    values = round_df.loc[
                        (round_df["budget"] == budget)
                        & (round_df["condition"] == condition),
                        metric,
                    ].dropna()

                    means.append(values.mean() if len(values) else np.nan)
                    stds.append(
                        values.std(ddof=1)
                        if len(values) > 1 else np.nan
                    )

                ax.errorbar(
                    budgets,
                    means,
                    yerr=stds,
                    marker="o",
                    capsize=3,
                    linewidth=1.8,
                    label=LABELS[condition],
                )

            ax.set_title(round_name.replace("round", "Round "))
            ax.set_xlabel("Labeled training budget")
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_xticks(budgets)
            ax.grid(True, alpha=0.3)

        axes[-1].legend(
            loc="best",
            fontsize=8,
        )

        fig.suptitle(
            f"DynaSent controlled-budget results: {metric}",
            fontsize=13,
        )
        fig.tight_layout()

        png_path = FIGURES / f"{metric}_by_budget.png"
        pdf_path = FIGURES / f"{metric}_by_budget.pdf"

        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {png_path.relative_to(ROOT)}")
        print(f"Saved {pdf_path.relative_to(ROOT)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading experiments from: {RESULTS}")
    df = load_data()

    print(f"Loaded {len(df)} metric records.")
    print("Rounds:", sorted(df["round"].unique()))
    print("Budgets:", sorted(df["budget"].unique()))
    print("Seeds:", sorted(df["seed"].unique()))

    uncertainty = paired_uncertainty(df)
    uncertainty_path = OUT / "paired_uncertainty_summary.csv"
    uncertainty.to_csv(uncertainty_path, index=False)

    print(f"\nSaved paired uncertainty results:\n{uncertainty_path}")
    print("\nPaired uncertainty preview:")
    if not uncertainty.empty:
        print(
            uncertainty.round(6)
            .sort_values(["round", "metric", "budget", "condition"])
            .to_string(index=False)
        )

    make_plots(df)

    print("\nAnalysis complete.")
    print(
        "\nInterpretation: confidence intervals describe seed-level "
        "variation conditional on these experiment runs and evaluation "
        "sets. They do not capture all dataset or training-procedure "
        "uncertainty, and they are not multiplicity-adjusted."
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
