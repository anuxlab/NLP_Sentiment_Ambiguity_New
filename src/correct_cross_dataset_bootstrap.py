
#!/usr/bin/env python3
"""Holm correction for cross-dataset two-level-bootstrap comparisons."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / "results" / "cross_dataset_analysis"
    / "two_level_bootstrap" / "paired_two_level_bootstrap.csv"
)
OUT = INPUT.parent
ALPHA = 0.05


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, preserving original order."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid_idx = np.flatnonzero(np.isfinite(p))

    if len(valid_idx) == 0:
        return adjusted

    pv = p[valid_idx]
    order = np.argsort(pv)
    sorted_p = pv[order]
    m = len(sorted_p)

    sorted_adj = np.maximum.accumulate(
        [(m - i) * value for i, value in enumerate(sorted_p)]
    )
    sorted_adj = np.minimum(sorted_adj, 1.0)

    restored = np.empty(m)
    restored[order] = sorted_adj
    adjusted[valid_idx] = restored
    return adjusted


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)
    required = {
        "dataset", "budget", "n_seeds", "n_test_examples",
        "mean_js_delta_bits", "bootstrap_ci_95_low",
        "bootstrap_ci_95_high",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # These are paired bootstrap draws summarized as percentile intervals.
    # A CI alone cannot provide an exact valid p-value, so we do not derive
    # p-values from whether the interval crosses zero.
    #
    # This script instead uses a paired sign-flip randomization test on the
    # seed-level mean per-example JS differences, if those values are
    # provided in a separate input file.
    paired_path = (
        ROOT / "results" / "cross_dataset_analysis"
        / "paired_seed_differences.csv"
    )

    if not paired_path.exists():
        raise FileNotFoundError(
            f"Need seed-level paired differences at {paired_path}. "
            "Bootstrap confidence intervals alone are insufficient "
            "to calculate Holm-adjusted p-values."
        )

    paired = pd.read_csv(paired_path)
    required_paired = {
        "dataset", "budget", "seed", "treatment", "js_divergence_delta"
    }
    missing = required_paired - set(paired.columns)
    if missing:
        raise ValueError(
            "Paired-difference file has incompatible columns. "
            f"Missing: {sorted(missing)}; found: {list(paired.columns)}"
        )

    paired = paired[
        paired["treatment"] == "human_soft_targets"
    ].copy()

    rows = []
    for (dataset, budget), group in paired.groupby(
        ["dataset", "budget"], sort=True
    ):
        delta = group["js_divergence_delta"].dropna().to_numpy(float)

        if len(delta) == 0:
            continue

        # Two-sided exact sign-flip test over paired seed-level deltas.
        # Appropriate as a small-sample sensitivity test under symmetry/
        # exchangeability assumptions, not a replacement for the bootstrap.
        observed = abs(delta.mean())
        n = len(delta)

        if n > 20:
            raise ValueError("Exact sign-flip implementation supports <=20 seeds")

        signs = np.array(
            np.meshgrid(*[[-1.0, 1.0]] * n)
        ).T.reshape(-1, n)

        permuted = (signs * delta).mean(axis=1)
        p_value = np.mean(np.abs(permuted) >= observed - 1e-12)

        rows.append({
            "dataset": dataset,
            "budget": int(budget),
            "n_seeds": n,
            "mean_seed_delta": delta.mean(),
            "sign_flip_p": p_value,
        })

    tests = pd.DataFrame(rows)
    if tests.empty:
        raise ValueError("No valid paired tests were produced")

    # Holm correction across all available dataset x budget comparisons.
    tests["holm_p_all_comparisons"] = holm_adjust(
        tests["sign_flip_p"].to_numpy()
    )
    tests["significant_holm_0_05"] = (
        tests["holm_p_all_comparisons"] < ALPHA
    )

    output = OUT / "holm_corrected_cross_dataset_tests.csv"
    tests.to_csv(output, index=False)

    print(f"Saved: {output.relative_to(ROOT)}")
    print(
        tests[
            [
                "dataset", "budget", "n_seeds",
                "mean_seed_delta", "sign_flip_p",
                "holm_p_all_comparisons",
                "significant_holm_0_05",
            ]
        ].round(6).to_string(index=False)
    )
    print(
        "\nInterpret cautiously: tests operate on seed-level paired "
        "differences. They do not directly test the two-level bootstrap "
        "distribution, and results are exploratory."
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
