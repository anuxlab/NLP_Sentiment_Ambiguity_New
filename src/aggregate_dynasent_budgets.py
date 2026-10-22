
#!/usr/bin/env python3
"""
Aggregate DynaSent controlled-budget experiment results.

Input:
    results/dynasent_controlled/round*/budget_*/metrics.csv

Outputs:
    results/dynasent_controlled/aggregate_analysis/
        budget_condition_summary.csv
        paired_seed_differences.csv

The analysis:
1. Summarizes each metric by round, budget, and condition.
2. Reports mean and standard deviation across seeds.
3. Calculates paired seed-level differences against majority_ce.
4. Keeps rounds and budgets separate.
5. Does not modify the original experiment artifacts.
"""

from pathlib import Path
import sys

import pandas as pd


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_ROOT = PROJECT_ROOT / "results" / "dynasent_controlled"

OUTPUT_DIR = RESULTS_ROOT / "aggregate_analysis"

SUMMARY_PATH = OUTPUT_DIR / "budget_condition_summary.csv"
PAIRED_PATH = OUTPUT_DIR / "paired_seed_differences.csv"

BASELINE_CONDITION = "majority_ce"

EXPECTED_CONDITIONS = {
    "majority_ce",
    "human_soft_targets",
    "label_smoothing",
    "human_soft_targets_null_augmented",
}

# Map the actual CSV column names to consistent analysis names.
METRIC_MAP = {
    "js_divergence": "js_divergence_mean_bits",
    "soft_brier": "soft_brier_score",
    "accuracy": "accuracy",
    "macro_f1": "macro_f1",
    "ece": "ece_10bin",
}

METRICS = list(METRIC_MAP.keys())

REQUIRED_COLUMNS = {
    "round",
    "seed",
    "condition",
    "budget",
    *METRIC_MAP.values(),
}


# ---------------------------------------------------------------------
# Load experiment results
# ---------------------------------------------------------------------

def load_results() -> pd.DataFrame:
    """Load and validate all controlled-budget metrics.csv files."""

    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Results directory not found: {RESULTS_ROOT}"
        )

    files = sorted(RESULTS_ROOT.glob("round*/budget_*/metrics.csv"))

    if not files:
        raise FileNotFoundError(
            f"No metrics.csv files found under {RESULTS_ROOT}"
        )

    frames = []

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results root: {RESULTS_ROOT}")
    print(f"Found {len(files)} metrics.csv files.\n")

    for file_path in files:
        current = pd.read_csv(file_path)

        missing = REQUIRED_COLUMNS - set(current.columns)

        if missing:
            raise ValueError(
                f"\nUnexpected metrics.csv schema in:\n"
                f"  {file_path.relative_to(PROJECT_ROOT)}\n"
                f"Missing columns: {sorted(missing)}\n"
                f"Available columns: {list(current.columns)}"
            )

        # Preserve the actual metric values; only rename columns.
        current = current.rename(
            columns={
                actual: expected
                for expected, actual in METRIC_MAP.items()
            }
        )

        # Check that the file path agrees with the metadata.
        path_round = file_path.parent.parent.name
        path_budget = file_path.parent.name

        if not current["round"].astype(str).eq(path_round).all():
            raise ValueError(
                f"Round metadata mismatch in {file_path}"
            )

        try:
            budget_from_path = int(path_budget.replace("budget_", ""))
        except ValueError as exc:
            raise ValueError(
                f"Could not parse budget from directory: {path_budget}"
            ) from exc

        if not pd.to_numeric(
            current["budget"], errors="coerce"
        ).eq(budget_from_path).all():
            raise ValueError(
                f"Budget metadata mismatch in {file_path}"
            )

        current["source_file"] = str(
            file_path.relative_to(PROJECT_ROOT)
        )

        frames.append(current)

        print(
            f"Loaded {file_path.relative_to(PROJECT_ROOT)}"
            f" ({len(current)} rows)"
        )

    df = pd.concat(frames, ignore_index=True)

    # Validate key fields.
    for column in ["round", "seed", "condition", "budget"]:
        if df[column].isna().any():
            raise ValueError(
                f"Missing values found in required column: {column}"
            )

    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["budget"] = pd.to_numeric(
        df["budget"], errors="raise"
    ).astype(int)

    for metric in METRICS:
        df[metric] = pd.to_numeric(
            df[metric], errors="coerce"
        )

    invalid_metrics = df[METRICS].isna().sum()
    invalid_metrics = invalid_metrics[invalid_metrics > 0]

    if not invalid_metrics.empty:
        raise ValueError(
            "Missing or non-numeric metric values detected:\n"
            f"{invalid_metrics.to_string()}"
        )

    # Each round/budget/seed/condition should have one record.
    key_columns = ["round", "budget", "seed", "condition"]

    duplicates = df.duplicated(key_columns, keep=False)

    if duplicates.any():
        examples = df.loc[
            duplicates, key_columns + ["source_file"]
        ].sort_values(key_columns)

        raise ValueError(
            "Duplicate round/budget/seed/condition records found:\n"
            f"{examples.head(20).to_string(index=False)}"
        )

    unknown_conditions = set(df["condition"].unique()) - EXPECTED_CONDITIONS

    if unknown_conditions:
        raise ValueError(
            f"Unexpected conditions found: {sorted(unknown_conditions)}\n"
            f"Expected conditions: {sorted(EXPECTED_CONDITIONS)}"
        )

    print(f"\nTotal records loaded: {len(df)}")
    print(f"Rounds: {sorted(df['round'].unique())}")
    print(f"Budgets: {sorted(df['budget'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Conditions: {sorted(df['condition'].unique())}")

    return df


# ---------------------------------------------------------------------
# Validate experimental groups
# ---------------------------------------------------------------------

def validate_groups(df: pd.DataFrame) -> None:
    """Report incomplete groups without silently inventing observations."""

    group_columns = ["round", "budget"]

    for (round_name, budget), group in df.groupby(group_columns):
        seed_sets = {
            condition: set(
                rows["seed"].astype(int).tolist()
            )
            for condition, rows in group.groupby("condition")
        }

        missing_conditions = EXPECTED_CONDITIONS - set(seed_sets)

        if missing_conditions:
            print(
                f"WARNING: {round_name}, budget {budget} is missing "
                f"conditions: {sorted(missing_conditions)}"
            )

        for condition, seeds in sorted(seed_sets.items()):
            missing_seeds = (
                set(group["seed"].unique().astype(int)) - seeds
            )

            if missing_seeds:
                print(
                    f"WARNING: {round_name}, budget {budget}, "
                    f"{condition} is missing seeds "
                    f"{sorted(missing_seeds)}"
                )

        # The paired analysis requires the same seed for each condition.
        baseline_seeds = seed_sets.get(BASELINE_CONDITION, set())

        for condition, seeds in sorted(seed_sets.items()):
            if condition == BASELINE_CONDITION:
                continue

            common_seeds = baseline_seeds & seeds

            if not common_seeds:
                print(
                    f"WARNING: No paired seeds for {round_name}, "
                    f"budget {budget}, {condition}"
                )


# ---------------------------------------------------------------------
# Descriptive summary
# ---------------------------------------------------------------------

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and standard deviation for each metric."""

    grouping = ["round", "budget", "condition"]

    summary = (
        df.groupby(grouping, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            **{
                f"{metric}_mean": (metric, "mean")
                for metric in METRICS
            },
            **{
                f"{metric}_std": (metric, "std")
                for metric in METRICS
            },
        )
        .sort_values(["round", "budget", "condition"])
        .reset_index(drop=True)
    )

    return summary


# ---------------------------------------------------------------------
# Paired seed-level differences
# ---------------------------------------------------------------------

def make_paired_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate treatment-minus-baseline differences within each
    round, budget, and seed.

    Negative differences are favorable for JS divergence, Brier score,
    and ECE. Positive differences are favorable for accuracy and F1.
    """

    grouping = ["round", "budget", "seed"]

    baseline = df.loc[
        df["condition"] == BASELINE_CONDITION,
        grouping + METRICS,
    ].copy()

    baseline = baseline.rename(
        columns={
            metric: f"{metric}_baseline"
            for metric in METRICS
        }
    )

    treatments = df.loc[
        df["condition"] != BASELINE_CONDITION
    ].copy()

    paired = treatments.merge(
        baseline,
        on=grouping,
        how="inner",
        validate="many_to_one",
    )

    for metric in METRICS:
        paired[f"{metric}_difference"] = (
            paired[metric] - paired[f"{metric}_baseline"]
        )

    difference_columns = [
        f"{metric}_difference" for metric in METRICS
    ]

    paired_summary = (
        paired.groupby(
            ["round", "budget", "condition"],
            as_index=False,
        )
        .agg(
            n_paired_seeds=("seed", "nunique"),
            **{
                f"{metric}_difference_mean": (
                    f"{metric}_difference", "mean"
                )
                for metric in METRICS
            },
            **{
                f"{metric}_difference_std": (
                    f"{metric}_difference", "std"
                )
                for metric in METRICS
            },
        )
        .sort_values(["round", "budget", "condition"])
        .reset_index(drop=True)
    )

    # Include the individual paired differences for transparency.
    paired_seed_rows = paired[
        ["round", "budget", "seed", "condition"]
        + difference_columns
    ].copy()

    paired_seed_rows = paired_seed_rows.rename(
        columns={
            f"{metric}_difference": f"{metric}_delta"
            for metric in METRICS
        }
    )

    paired_summary = paired_summary.rename(
        columns={
            f"{metric}_difference_mean": f"{metric}_delta_mean"
            for metric in METRICS
        }
        | {
            f"{metric}_difference_std": f"{metric}_delta_std"
            for metric in METRICS
        }
    )

    # Summary rows first, then individual seed rows, in one file.
    # A row_type column clearly distinguishes the two.
    paired_summary["row_type"] = "summary"
    paired_seed_rows["row_type"] = "seed"

    # Standardize columns so both row types can be concatenated.
    all_columns = list(
        dict.fromkeys(
            list(paired_summary.columns)
            + list(paired_seed_rows.columns)
        )
    )

    paired_output = pd.concat(
        [
            paired_summary.reindex(columns=all_columns),
            paired_seed_rows.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )

    return paired_output


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    try:
        df = load_results()
        validate_groups(df)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        summary = make_summary(df)
        paired = make_paired_differences(df)

        summary.to_csv(SUMMARY_PATH, index=False)
        paired.to_csv(PAIRED_PATH, index=False)

        print("\nAggregation completed successfully.")
        print(f"\nSummary saved to:\n  {SUMMARY_PATH}")
        print(f"\nPaired differences saved to:\n  {PAIRED_PATH}")

        print("\nSummary preview:")
        print(summary.head(12).to_string(index=False))

        print("\nPaired-difference preview:")
        print(paired.head(12).to_string(index=False))

        print(
            "\nInterpretation guide:\n"
            "  JS divergence, Brier score, ECE: lower is better.\n"
            "  Accuracy, macro-F1: higher is better.\n"
            "  Delta = treatment metric - majority_ce metric.\n"
            "  Keep rounds separate; do not pool their evaluation sets."
        )

        return 0

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
