
#!/usr/bin/env python3
"""
Cross-dataset comparison of human soft targets vs majority CE.

Datasets:
  - DynaSent Round 1
  - DynaSent Round 2
  - SentiMP-En

Uses existing controlled runs only; does not train models or modify raw data.

Outputs:
  results/cross_dataset_analysis/
    dataset_condition_summary.csv
    paired_seed_differences.csv
    paired_bootstrap_summary.csv
    cross_dataset_js_by_budget.png
    cross_dataset_js_by_budget.pdf
    analysis_manifest.txt
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DYNA_ROOT = ROOT / "results" / "dynasent_controlled"
SENTIMP_FILE = ROOT / "results" / "sentimp_en_controlled" / "metrics.jsonl"
OUT = ROOT / "results" / "cross_dataset_analysis"

BASELINE = "majority_ce"
TREATMENTS = [
    "human_soft_targets",
    "label_smoothing",
]

METRICS = [
    "js_divergence",
    "soft_brier",
    "accuracy",
    "macro_f1",
]

METRIC_LABELS = {
    "js_divergence": "Mean JS divergence (bits; lower is better)",
    "soft_brier": "Soft Brier score (lower is better)",
    "accuracy": "Accuracy (higher is better)",
    "macro_f1": "Macro-F1 (higher is better)",
}

N_BOOTSTRAP = 20_000
RANDOM_SEED = 20261020


def load_dynasent():
    """Load the current 10-seed controlled DynaSent experiment."""
    files = sorted(DYNA_ROOT.glob("round*/budget_*/metrics.csv"))

    if not files:
        raise FileNotFoundError(
            f"No DynaSent controlled metrics.csv files under {DYNA_ROOT}"
        )

    frames = []

    required_actual = {
        "round", "budget", "seed", "condition",
        "js_divergence_mean_bits", "soft_brier_score",
        "accuracy", "macro_f1",
    }

    for path in files:
        frame = pd.read_csv(path)
        missing = required_actual - set(frame.columns)

        if missing:
            raise ValueError(
                f"{path}: missing columns {sorted(missing)}"
            )

        frame = frame.rename(columns={
            "js_divergence_mean_bits": "js_divergence",
            "soft_brier_score": "soft_brier",
        })

        round_name = path.parent.parent.name

        if not frame["round"].astype(str).eq(round_name).all():
            raise ValueError(f"Round metadata mismatch in {path}")

        budget_from_path = int(path.parent.name.replace("budget_", ""))

        if not pd.to_numeric(
            frame["budget"], errors="raise"
        ).eq(budget_from_path).all():
            raise ValueError(f"Budget metadata mismatch in {path}")

        frame["dataset_group"] = (
            "DynaSent Round 1"
            if round_name == "round1"
            else "DynaSent Round 2"
        )
        frame["dataset_key"] = round_name
        frame["budget"] = pd.to_numeric(
            frame["budget"], errors="raise"
        ).astype(int)
        frame["seed"] = pd.to_numeric(
            frame["seed"], errors="raise"
        ).astype(int)

        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    return df


def load_sentimp():
    """Load the matched controlled SentiMP-En experiment."""
    if not SENTIMP_FILE.exists():
        raise FileNotFoundError(
            f"SentiMP-En metrics file not found: {SENTIMP_FILE}"
        )

    df = pd.read_json(SENTIMP_FILE, lines=True)

    required = {
        "dataset", "seed", "budget_requested", "condition",
        "js_divergence_bits", "soft_brier_score",
        "accuracy", "macro_f1",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{SENTIMP_FILE}: missing columns {sorted(missing)}; "
            f"available columns: {list(df.columns)}"
        )

    df = df.rename(columns={
        "budget_requested": "budget",
        "js_divergence_bits": "js_divergence",
        "soft_brier_score": "soft_brier",
    })

    df["dataset_group"] = "SentiMP-En"
    df["dataset_key"] = "sentimp_en"
    df["budget"] = pd.to_numeric(
        df["budget"], errors="raise"
    ).astype(int)
    df["seed"] = pd.to_numeric(
        df["seed"], errors="raise"
    ).astype(int)

    return df


def validate_data(df):
    """Validate uniqueness, conditions, metrics, and paired coverage."""
    required_conditions = {BASELINE, *TREATMENTS}
    actual_conditions = set(df["condition"].unique())

    unexpected = actual_conditions - required_conditions
    if unexpected:
        raise ValueError(
            f"Unexpected conditions found: {sorted(unexpected)}"
        )

    missing_conditions = required_conditions - actual_conditions
    if missing_conditions:
        raise ValueError(
            f"Required conditions missing: {sorted(missing_conditions)}"
        )

    keys = ["dataset_key", "budget", "seed", "condition"]
    if df.duplicated(keys).any():
        duplicates = df.loc[df.duplicated(keys, keep=False), keys]
        raise ValueError(
            "Duplicate dataset/budget/seed/condition records:\n"
            + duplicates.head(20).to_string(index=False)
        )

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    if df[METRICS].isna().any().any():
        raise ValueError("Missing or nonnumeric metric values detected.")

    print("\nDataset coverage:")
    for dataset, group in df.groupby("dataset_group", sort=False):
        print(
            f"  {dataset}: {len(group)} rows; "
            f"budgets={sorted(group['budget'].unique())}; "
            f"seeds={sorted(group['seed'].unique())}"
        )

        for (budget, condition), subset in group.groupby(
            ["budget", "condition"]
        ):
            print(
                f"    budget={budget:>3}, "
                f"{condition}: {subset['seed'].nunique()} seeds"
            )

    # Explicitly check that each treatment can be paired with the baseline.
    for (dataset, budget), group in df.groupby(
        ["dataset_group", "budget"]
    ):
        baseline_seeds = set(
            group.loc[group["condition"] == BASELINE, "seed"]
        )

        for treatment in TREATMENTS:
            treatment_seeds = set(
                group.loc[group["condition"] == treatment, "seed"]
            )
            common = baseline_seeds & treatment_seeds

            if len(common) < 2:
                print(
                    f"WARNING: only {len(common)} paired seeds for "
                    f"{dataset}, budget {budget}, {treatment}"
                )

    return df


def make_condition_summary(df):
    """Mean and sample SD across seeds for each dataset/budget/condition."""
    rows = []

    for (dataset, budget, condition), group in df.groupby(
        ["dataset_group", "budget", "condition"], sort=False
    ):
        row = {
            "dataset": dataset,
            "budget": budget,
            "condition": condition,
            "n_seeds": group["seed"].nunique(),
        }

        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["dataset", "budget", "condition"]
    )


def make_paired_differences(df):
    """
    Treatment minus baseline, paired within dataset, budget, and seed.

    Negative deltas favor treatment for JS/Brier.
    Positive deltas favor treatment for accuracy/macro-F1.
    """
    rows = []

    for (dataset, budget), group in df.groupby(
        ["dataset_group", "budget"], sort=False
    ):
        baseline = group[group["condition"] == BASELINE].set_index("seed")

        for treatment in TREATMENTS:
            treated = (
                group[group["condition"] == treatment].set_index("seed")
            )

            common = sorted(set(baseline.index) & set(treated.index))

            for seed in common:
                row = {
                    "dataset": dataset,
                    "budget": budget,
                    "seed": int(seed),
                    "treatment": treatment,
                    "baseline": BASELINE,
                    "n_paired_seeds_group": len(common),
                }

                for metric in METRICS:
                    row[f"{metric}_delta"] = (
                        treated.loc[seed, metric]
                        - baseline.loc[seed, metric]
                    )

                rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["dataset", "budget", "treatment", "seed"]
    )


def make_bootstrap_summary(paired):
    """
    Percentile bootstrap CI of the mean paired seed difference.

    Seeds, not individual test examples, are resampled.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for (dataset, budget, treatment), group in paired.groupby(
        ["dataset", "budget", "treatment"], sort=False
    ):
        for metric in METRICS:
            delta = group[f"{metric}_delta"].to_numpy(dtype=float)

            if len(delta) < 2:
                continue

            indices = rng.integers(
                0, len(delta), size=(N_BOOTSTRAP, len(delta))
            )
            boot_means = delta[indices].mean(axis=1)
            ci_low, ci_high = np.quantile(
                boot_means, [0.025, 0.975]
            )

            rows.append({
                "dataset": dataset,
                "budget": budget,
                "treatment": treatment,
                "metric": metric,
                "n_paired_seeds": len(delta),
                "mean_paired_delta": delta.mean(),
                "sd_paired_delta": delta.std(ddof=1),
                "bootstrap_ci_95_low": ci_low,
                "bootstrap_ci_95_high": ci_high,
                "bootstrap_replicates": N_BOOTSTRAP,
                "bootstrap_unit": "seed",
                "baseline": BASELINE,
            })

    return pd.DataFrame(rows).sort_values(
        ["dataset", "metric", "budget", "treatment"]
    )


def make_js_plot(summary):
    """Plot mean JS by budget in separate panels for each dataset."""
    datasets = [
        "DynaSent Round 1",
        "DynaSent Round 2",
        "SentiMP-En",
    ]

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(17, 5.2), squeeze=False
    )
    axes = axes[0]

    conditions = [
        BASELINE,
        "human_soft_targets",
        "label_smoothing",
    ]

    labels = {
        BASELINE: "Majority CE",
        "human_soft_targets": "Human soft targets",
        "label_smoothing": "Label smoothing",
    }

    for ax, dataset in zip(axes, datasets):
        sub = summary[summary["dataset"] == dataset]

        if sub.empty:
            ax.set_visible(False)
            continue

        budgets = sorted(sub["budget"].unique())

        for condition in conditions:
            cond = sub[sub["condition"] == condition].set_index("budget")
            means = [
                cond.loc[b, "js_divergence_mean"]
                if b in cond.index else np.nan
                for b in budgets
            ]
            stds = [
                cond.loc[b, "js_divergence_std"]
                if b in cond.index else np.nan
                for b in budgets
            ]

            ax.errorbar(
                budgets,
                means,
                yerr=stds,
                marker="o",
                capsize=3,
                linewidth=1.8,
                label=labels[condition],
            )

        ax.set_title(dataset)
        ax.set_xlabel("Requested labeled training budget")
        ax.set_ylabel("Mean JS divergence (bits; lower is better)")
        ax.set_xticks(budgets)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Cross-dataset sentiment distribution prediction",
        fontsize=14,
    )
    fig.tight_layout()

    png = OUT / "cross_dataset_js_by_budget.png"
    pdf = OUT / "cross_dataset_js_by_budget.pdf"

    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {png.relative_to(ROOT)}")
    print(f"Saved {pdf.relative_to(ROOT)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    dynasent = load_dynasent()
    sentimp = load_sentimp()

    # Restrict to matched conditions. Null-augmented and tie-inclusive
    # runs are distinct interventions and intentionally excluded here.
    df = pd.concat([dynasent, sentimp], ignore_index=True)
    df = df[df["condition"].isin([BASELINE, *TREATMENTS])].copy()
    df = validate_data(df)

    summary = make_condition_summary(df)
    paired = make_paired_differences(df)
    bootstrap = make_bootstrap_summary(paired)

    summary_path = OUT / "dataset_condition_summary.csv"
    paired_path = OUT / "paired_seed_differences.csv"
    bootstrap_path = OUT / "paired_bootstrap_summary.csv"

    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)

    print(f"\nSaved {summary_path.relative_to(ROOT)}")
    print(f"Saved {paired_path.relative_to(ROOT)}")
    print(f"Saved {bootstrap_path.relative_to(ROOT)}")

    make_js_plot(summary)

    manifest = {
        "analysis": "cross-dataset controlled sentiment comparison",
        "inputs": [
            str(DYNA_ROOT.relative_to(ROOT)),
            str(SENTIMP_FILE.relative_to(ROOT)),
        ],
        "datasets": sorted(df["dataset_group"].unique().tolist()),
        "conditions": [BASELINE, *TREATMENTS],
        "metrics": METRICS,
        "bootstrap_replicates": N_BOOTSTRAP,
        "bootstrap_unit": "seed",
        "bootstrap_seed": RANDOM_SEED,
        "raw_metrics_modified": False,
        "interpretation": (
            "Exploratory cross-dataset comparison. Compare paired effects "
            "within each dataset; do not directly rank raw JS values across "
            "datasets with different label spaces and protocols."
        ),
    }

    manifest_path = OUT / "analysis_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, indent=2) + "\n")

    print(f"Saved {manifest_path.relative_to(ROOT)}")

    print("\nPrimary comparison: human soft targets vs majority CE")
    primary = bootstrap[
        (bootstrap["treatment"] == "human_soft_targets")
        & (bootstrap["metric"] == "js_divergence")
    ]

    if primary.empty:
        print("No paired primary comparisons available.")
    else:
        print(
            primary[
                [
                    "dataset", "budget", "n_paired_seeds",
                    "mean_paired_delta",
                    "bootstrap_ci_95_low",
                    "bootstrap_ci_95_high",
                ]
            ].round(6).to_string(index=False)
        )

    print(
        "\nDone. Confidence intervals resample seeds and are conditional "
        "on the existing fixed test sets. They do not capture test-set "
        "sampling uncertainty and are not multiplicity-adjusted."
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
