
#!/usr/bin/env python3
"""
Robustness audit for DynaSent controlled-budget experiments.

Input:
    results/dynasent_controlled/round*/budget_*/metrics.csv

Outputs:
    results/dynasent_controlled/aggregate_analysis/robustness/
        paired_js_tests.csv
        planned_primary_contrast.csv
        leave_one_seed_out.csv
        budget_trends.csv
        audit_report.txt

Inference is exploratory. Seeds are the replication units.
Rounds are analyzed separately. No test examples are treated as
independent replications.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "dynasent_controlled"
OUT = RESULTS / "aggregate_analysis" / "robustness"

BASELINE = "majority_ce"
TREATMENTS = [
    "human_soft_targets",
    "human_soft_targets_null_augmented",
    "label_smoothing",
]
EXPECTED_CONDITIONS = {BASELINE, *TREATMENTS}

METRIC_COLUMNS = {
    "js_divergence": "js_divergence_mean_bits",
    "soft_brier": "soft_brier_score",
    "macro_f1": "macro_f1",
    "accuracy": "accuracy",
    "ece": "ece_10bin",
}

SEED = 20261020
BOOTSTRAPS = 20_000


def holm_adjust(p_values):
    """Holm step-down adjustment; preserves NaNs."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)

    valid_indices = np.flatnonzero(np.isfinite(p))
    if len(valid_indices) == 0:
        return adjusted

    valid = p[valid_indices]
    order = np.argsort(valid)
    sorted_p = valid[order]
    m = len(sorted_p)

    sorted_adjusted = np.maximum.accumulate(
        [(m - i) * value for i, value in enumerate(sorted_p)]
    )
    sorted_adjusted = np.minimum(sorted_adjusted, 1.0)

    restored = np.empty(m)
    restored[order] = sorted_adjusted
    adjusted[valid_indices] = restored

    return adjusted


def load_data():
    files = sorted(RESULTS.glob("round*/budget_*/metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics.csv files found in {RESULTS}")

    required = {
        "round", "budget", "seed", "condition",
        *METRIC_COLUMNS.values(),
    }

    frames = []
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
                for standard, actual in METRIC_COLUMNS.items()
            }
        )

        frame["seed"] = pd.to_numeric(
            frame["seed"], errors="raise"
        ).astype(int)
        frame["budget"] = pd.to_numeric(
            frame["budget"], errors="raise"
        ).astype(int)

        for metric in METRIC_COLUMNS:
            frame[metric] = pd.to_numeric(
                frame[metric], errors="coerce"
            )

        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    if df[list(METRIC_COLUMNS)].isna().any().any():
        raise ValueError("Missing or nonnumeric metric values found.")

    keys = ["round", "budget", "seed", "condition"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate round/budget/seed/condition records.")

    unexpected = set(df["condition"].unique()) - EXPECTED_CONDITIONS
    if unexpected:
        raise ValueError(f"Unexpected conditions: {sorted(unexpected)}")

    return df


def check_completeness(df):
    """Check expected conditions and seed coverage per round and budget."""
    rows = []

    for (round_name, budget), group in df.groupby(["round", "budget"]):
        seed_sets = {
            condition: set(
                subset["seed"].astype(int)
            )
            for condition, subset in group.groupby("condition")
        }

        all_seeds = set(group["seed"].astype(int))

        for condition in sorted(EXPECTED_CONDITIONS):
            present = seed_sets.get(condition, set())

            rows.append({
                "round": round_name,
                "budget": budget,
                "condition": condition,
                "n_seeds": len(present),
                "missing_seeds": ",".join(
                    map(str, sorted(all_seeds - present))
                ),
                "condition_present": condition in seed_sets,
            })

    audit = pd.DataFrame(rows)

    if not audit["condition_present"].all():
        print("WARNING: at least one experiment group is missing a condition.")

    if (audit["n_seeds"] < 2).any():
        print("WARNING: at least one group has fewer than two seeds.")

    return audit


def paired_js_tests(df):
    """
    Paired two-sided Wilcoxon signed-rank tests on seed-level JS deltas.

    A paired t-test is also reported as a sensitivity check. Holm
    correction covers all treatment x budget x round comparisons.
    """
    rng = np.random.default_rng(SEED)
    rows = []

    for (round_name, budget), group in df.groupby(
        ["round", "budget"], sort=True
    ):
        baseline = (
            group[group["condition"] == BASELINE]
            .set_index("seed")
        )

        for treatment in TREATMENTS:
            treated = (
                group[group["condition"] == treatment]
                .set_index("seed")
            )

            common = sorted(set(baseline.index) & set(treated.index))
            if len(common) < 2:
                continue

            delta = (
                treated.loc[common, "js_divergence"].to_numpy()
                - baseline.loc[common, "js_divergence"].to_numpy()
            )

            if np.allclose(delta, 0):
                wilcoxon_p = 1.0
            else:
                try:
                    wilcoxon_p = stats.wilcoxon(
                        delta,
                        alternative="two-sided",
                        method="auto",
                    ).pvalue
                except ValueError:
                    wilcoxon_p = np.nan

            ttest_p = stats.ttest_1samp(delta, popmean=0).pvalue

            boot_idx = rng.integers(
                0, len(delta), size=(BOOTSTRAPS, len(delta))
            )
            boot_means = delta[boot_idx].mean(axis=1)
            ci_low, ci_high = np.quantile(
                boot_means, [0.025, 0.975]
            )

            rows.append({
                "round": round_name,
                "budget": budget,
                "treatment": treatment,
                "baseline": BASELINE,
                "metric": "js_divergence",
                "n_paired_seeds": len(delta),
                "mean_delta": delta.mean(),
                "median_delta": np.median(delta),
                "sd_delta": delta.std(ddof=1),
                "bootstrap_ci_95_low": ci_low,
                "bootstrap_ci_95_high": ci_high,
                "wilcoxon_p": wilcoxon_p,
                "paired_ttest_p": ttest_p,
                "bootstrap_replicates": BOOTSTRAPS,
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["wilcoxon_holm_p_all_30"] = holm_adjust(
            result["wilcoxon_p"].to_numpy()
        )
        result["ttest_holm_p_all_30"] = holm_adjust(
            result["paired_ttest_p"].to_numpy()
        )

    return result


def planned_primary_contrast(df):
    """
    Summarize human-soft-targets vs majority CE separately per round.

    This remains a budget-sweep analysis, not a replacement for the
    separately specified multiseed primary analysis.
    """
    rows = []

    for round_name, round_df in df.groupby("round", sort=True):
        for budget, group in round_df.groupby("budget", sort=True):
            baseline = group[group["condition"] == BASELINE].set_index("seed")
            treatment = (
                group[group["condition"] == "human_soft_targets"]
                .set_index("seed")
            )

            common = sorted(set(baseline.index) & set(treatment.index))
            if len(common) < 2:
                continue

            delta = (
                treatment.loc[common, "js_divergence"].to_numpy()
                - baseline.loc[common, "js_divergence"].to_numpy()
            )

            if np.allclose(delta, 0):
                p = 1.0
            else:
                p = stats.wilcoxon(
                    delta, alternative="two-sided", method="auto"
                ).pvalue

            rows.append({
                "round": round_name,
                "budget": budget,
                "contrast": "human_soft_targets - majority_ce",
                "n_paired_seeds": len(delta),
                "mean_js_delta": delta.mean(),
                "median_js_delta": np.median(delta),
                "wilcoxon_p_unadjusted": p,
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        # Five budget comparisons per round, adjusted separately.
        result["holm_p_within_round"] = np.nan

        for _, indices in result.groupby("round").groups.items():
            result.loc[indices, "holm_p_within_round"] = holm_adjust(
                result.loc[indices, "wilcoxon_p_unadjusted"].to_numpy()
            )

    return result


def leave_one_seed_out(df):
    """Check how much the mean JS difference changes when one seed is omitted."""
    rows = []

    for (round_name, budget), group in df.groupby(
        ["round", "budget"], sort=True
    ):
        baseline = group[group["condition"] == BASELINE].set_index("seed")

        for treatment in TREATMENTS:
            treated = group[group["condition"] == treatment].set_index("seed")
            common = sorted(set(baseline.index) & set(treated.index))

            if len(common) < 3:
                continue

            delta = (
                treated.loc[common, "js_divergence"].to_numpy()
                - baseline.loc[common, "js_divergence"].to_numpy()
            )
            full_mean = delta.mean()

            for i, omitted_seed in enumerate(common):
                reduced = np.delete(delta, i)

                rows.append({
                    "round": round_name,
                    "budget": budget,
                    "treatment": treatment,
                    "omitted_seed": omitted_seed,
                    "n_remaining_seeds": len(reduced),
                    "full_mean_delta": full_mean,
                    "leave_one_out_mean_delta": reduced.mean(),
                    "direction_changed": (
                        np.sign(reduced.mean()) != np.sign(full_mean)
                    ),
                })

    return pd.DataFrame(rows)


def budget_trends(df):
    """
    Estimate per-seed linear slopes of JS divergence against log2(budget).

    This is a descriptive trend analysis. Budgets share data-generation
    and experimental structure, so trend p-values are not calculated.
    """
    rows = []

    for (round_name, condition, seed), group in df.groupby(
        ["round", "condition", "seed"], sort=True
    ):
        group = group.sort_values("budget")

        if group["budget"].nunique() < 3:
            continue

        x = np.log2(group["budget"].to_numpy(dtype=float))
        y = group["js_divergence"].to_numpy(dtype=float)

        slope, intercept = np.polyfit(x, y, deg=1)

        rows.append({
            "round": round_name,
            "condition": condition,
            "seed": seed,
            "js_slope_per_log2_budget": slope,
            "intercept": intercept,
            "n_budgets": len(group),
        })

    slopes = pd.DataFrame(rows)
    if slopes.empty:
        return slopes, pd.DataFrame()

    summary = (
        slopes.groupby(["round", "condition"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_slope=("js_slope_per_log2_budget", "mean"),
            sd_slope=("js_slope_per_log2_budget", "std"),
            median_slope=("js_slope_per_log2_budget", "median"),
        )
    )

    return slopes, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} metric records.")
    print("Rounds:", sorted(df["round"].unique()))
    print("Budgets:", sorted(df["budget"].unique()))
    print("Seeds:", sorted(df["seed"].unique()))

    completeness = check_completeness(df)
    tests = paired_js_tests(df)
    planned = planned_primary_contrast(df)
    loo = leave_one_seed_out(df)
    slopes, trend_summary = budget_trends(df)

    outputs = {
        "paired_js_tests.csv": tests,
        "planned_primary_contrast.csv": planned,
        "leave_one_seed_out.csv": loo,
        "budget_trends.csv": slopes,
        "budget_trend_summary.csv": trend_summary,
        "completeness_audit.csv": completeness,
    }

    for filename, table in outputs.items():
        path = OUT / filename
        table.to_csv(path, index=False)
        print(f"Saved {path.relative_to(ROOT)}")

    report_path = OUT / "audit_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("DynaSent controlled-budget robustness audit\n")
        report.write("=" * 48 + "\n\n")
        report.write(f"Records: {len(df)}\n")
        report.write(f"Rounds: {sorted(df['round'].unique())}\n")
        report.write(f"Budgets: {sorted(df['budget'].unique())}\n")
        report.write(f"Seeds: {sorted(df['seed'].unique())}\n\n")

        report.write("Paired JS tests with Holm correction\n")
        report.write("-" * 36 + "\n")
        if not tests.empty:
            report.write(
                tests[
                    [
                        "round", "budget", "treatment",
                        "n_paired_seeds", "mean_delta",
                        "bootstrap_ci_95_low", "bootstrap_ci_95_high",
                        "wilcoxon_p", "wilcoxon_holm_p_all_30",
                    ]
                ].round(6).to_string(index=False)
            )
        else:
            report.write("No valid paired tests.\n")

        report.write("\n\nLimitations\n")
        report.write("-" * 12 + "\n")
        report.write(
            "These analyses use ten trained seeds and the existing fixed "
            "evaluation sets. Bootstrap intervals resample seeds, not test "
            "examples. The exploratory budget-sweep correction does not "
            "replace the frozen analysis plan or a separately conducted "
            "confirmatory analysis. Trend estimates are descriptive.\n"
        )

    print(f"Saved {report_path.relative_to(ROOT)}")
    print("\nPaired JS test preview:")
    if not tests.empty:
        print(
            tests[
                [
                    "round", "budget", "treatment", "mean_delta",
                    "wilcoxon_p", "wilcoxon_holm_p_all_30",
                ]
            ].round(6).to_string(index=False)
        )

    print("\nAudit complete. Review the CSV files before making claims.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
