import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOTS = {
    "DynaSent Round 1": Path("results/dynasent_controlled/round1"),
    "DynaSent Round 2": Path("results/dynasent_controlled/round2"),
    "SentiMP-En": Path("results/sentimp_en_controlled_seed10"),
}

OUT = Path("results/js_divergence_verification")
TOLERANCE = 1e-6
PROB_TOLERANCE = 1e-6
EPS = 1e-12


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON: {path}, line {line_number}"
                ) from exc


def get_probability_vector(row, primary, fallback, path):
    value = row.get(primary)
    if value is None:
        value = row.get(fallback)

    if value is None:
        raise ValueError(
            f"Missing {primary}/{fallback} in {path}"
        )

    p = np.asarray(value, dtype=float)

    if (
        p.ndim != 1
        or len(p) < 2
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.isclose(
            p.sum(), 1.0, atol=PROB_TOLERANCE, rtol=0
        )
    ):
        raise ValueError(
            f"Invalid probability vector for {primary} in {path}: {p}"
        )

    return p


def js_divergence_bits(p, q):
    """Compute Jensen-Shannon divergence in bits."""
    if p.shape != q.shape:
        raise ValueError(
            f"Probability-vector shape mismatch: {p.shape} vs {q.shape}"
        )

    midpoint = 0.5 * (p + q)

    def kl_bits(a, b):
        positive = a > 0
        return float(
            np.sum(
                a[positive]
                * np.log2(a[positive] / b[positive])
            )
        )

    return 0.5 * kl_bits(p, midpoint) + 0.5 * kl_bits(q, midpoint)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    errors = []

    for dataset, root in ROOTS.items():
        if not root.exists():
            print(f"ERROR: missing dataset directory: {root}")
            errors.append(
                {"dataset": dataset, "error": f"Missing directory: {root}"}
            )
            continue

        files = sorted(root.rglob("predictions.jsonl"))
        if not files:
            print(f"ERROR: no predictions.jsonl under {root}")
            errors.append(
                {"dataset": dataset, "error": "No prediction files"}
            )
            continue

        checked = 0
        missing_metrics = 0
        absolute_errors = []
        malformed = 0
        max_abs_error = 0.0

        for path in files:
            for line_number, row in enumerate(read_jsonl(path), start=1):
                saved_value = row.get("js_divergence_bits")
                if saved_value is None:
                    missing_metrics += 1
                    continue

                try:
                    saved_value = float(saved_value)
                    if not np.isfinite(saved_value):
                        raise ValueError("Saved JS value is non-finite")

                    target = get_probability_vector(
                        row,
                        "human_distribution",
                        "target_probs",
                        path,
                    )
                    prediction = get_probability_vector(
                        row,
                        "predicted_distribution",
                        "predicted_probs",
                        path,
                    )

                    if target.shape != prediction.shape:
                        raise ValueError(
                            "Target and prediction dimensions differ"
                        )

                    recomputed = js_divergence_bits(target, prediction)
                    abs_error = abs(saved_value - recomputed)

                    checked += 1
                    absolute_errors.append(abs_error)
                    max_abs_error = max(max_abs_error, abs_error)

                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    malformed += 1
                    if malformed <= 10:
                        errors.append(
                            {
                                "dataset": dataset,
                                "file": str(path),
                                "line": line_number,
                                "error": str(exc),
                            }
                        )

        if checked:
            errors_array = np.asarray(absolute_errors, dtype=float)
            over_tolerance = int(
                np.count_nonzero(errors_array > TOLERANCE)
            )
            summary = {
                "dataset": dataset,
                "rows_checked": checked,
                "missing_saved_js": missing_metrics,
                "malformed_rows": malformed,
                "tolerance": TOLERANCE,
                "max_absolute_error": max_abs_error,
                "mean_absolute_error": float(errors_array.mean()),
                "median_absolute_error": float(np.median(errors_array)),
                "rows_over_tolerance": over_tolerance,
                "status": (
                    "PASS"
                    if over_tolerance == 0
                    and malformed == 0
                    and missing_metrics == 0
                    else "REVIEW"
                ),
            }
        else:
            summary = {
                "dataset": dataset,
                "rows_checked": 0,
                "missing_saved_js": missing_metrics,
                "malformed_rows": malformed,
                "tolerance": TOLERANCE,
                "max_absolute_error": np.nan,
                "mean_absolute_error": np.nan,
                "median_absolute_error": np.nan,
                "rows_over_tolerance": 0,
                "status": "FAIL",
            }

        summaries.append(summary)

    report = pd.DataFrame(summaries)
    report_path = OUT / "js_divergence_verification.csv"
    report.to_csv(report_path, index=False)

    error_path = OUT / "verification_errors.json"
    with error_path.open("w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    print("\nJS DIVERGENCE VERIFICATION")
    if not report.empty:
        print(report.to_string(index=False))

    print(f"\nSummary: {report_path}")
    print(f"Diagnostics: {error_path}")

    failed = (
        report.empty
        or (report["status"] != "PASS").any()
        or bool(errors)
    )
    if failed:
        raise SystemExit(
            "Verification requires review. Inspect the CSV and diagnostics."
        )

    print("\nAll checked rows passed the configured tolerance.")


if __name__ == "__main__":
    main()
