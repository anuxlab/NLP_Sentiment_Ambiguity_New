
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOTS = {
    "DynaSent Round 1": Path("results/dynasent_controlled/round1"),
    "DynaSent Round 2": Path("results/dynasent_controlled/round2"),
    "SentiMP-En": Path("results/sentimp_en_controlled_seed10"),
}

OUT = Path("results/ambiguity_stratified_audit")
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = {"majority_ce", "human_soft_targets"}
EPS = 1e-12
PROB_TOL = 1e-6
ENTROPY_TOL = 1e-9


def read_jsonl(path):
    """Read JSONL records and report malformed lines with file context."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
    return records


def get_target(row, path):
    """Read and validate a human target probability distribution."""
    value = row.get("human_distribution", row.get("target_probs"))
    if value is None:
        raise ValueError(f"Missing human target distribution in {path}")

    p = np.asarray(value, dtype=float)

    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError(f"Invalid target distribution in {path}: {p}")

    if (p < 0).any() or not np.isclose(
        p.sum(), 1.0, atol=PROB_TOL, rtol=0
    ):
        raise ValueError(f"Target probabilities do not sum to 1 in {path}: {p}")

    return p


def normalized_entropy(p):
    """Shannon entropy normalized to [0, 1]."""
    n_classes = len(p)
    if n_classes <= 1:
        return 0.0

    return float(
        -(p * np.log(np.clip(p, EPS, 1.0))).sum()
        / np.log(n_classes)
    )


def normalize_text(value):
    """Normalize whitespace and case for text-based example matching."""
    return " ".join(str(value).casefold().split())


def get_example_id(row, dataset, path):
    if dataset.startswith("DynaSent"):
        example_id = row.get("text_id")
        if example_id is None or str(example_id).strip() == "":
            raise ValueError(f"Missing text_id in {path}")
        return str(example_id)

    text = row.get("text")
    if text is None or not normalize_text(text):
        raise ValueError(f"Missing text in {path}")

    return normalize_text(text)


def get_budget(row, path):
    """Read the recorded budget, falling back to a budget_N directory."""
    value = row.get("budget_requested")
    if value is not None:
        return int(value)

    parent = path.parent.name
    if parent.startswith("budget_"):
        return int(parent.removeprefix("budget_"))

    raise ValueError(f"Cannot determine budget for record in {path}")


def check_senti_text_collisions(records, dataset):
    """
    Fail if distinct original strings collapse to the same normalized text.

    Case and whitespace differences count as collisions by design.
    """
    if dataset != "SentiMP-En":
        return 0

    text_map = {}
    for row in records:
        text = row.get("text")
        if text is None or not normalize_text(text):
            raise ValueError("SentiMP-En record has missing or empty text")

        key = normalize_text(text)
        text_map.setdefault(key, set()).add(str(text))

    collisions = {
        key: originals
        for key, originals in text_map.items()
        if len(originals) > 1
    }

    if collisions:
        examples = list(collisions.items())[:5]
        raise ValueError(
            f"{dataset}: found {len(collisions)} normalized-text collisions. "
            f"Examples (normalized text -> original strings): {examples}"
        )

    return 0


def main():
    all_rows = []
    audit_rows = []
    threshold_rows = []

    for dataset, root in ROOTS.items():
        if not root.exists():
            print(f"SKIP missing directory: {root}")
            continue

        files = sorted(root.rglob("predictions.jsonl"))
        if not files:
            print(f"SKIP: no predictions.jsonl under {root}")
            continue

        dataset_records = []
        for path in files:
            records = read_jsonl(path)

            # SentiMP-En uses one combined file. DynaSent may use
            # multiple prediction files, usually organized by budget.
            for row in records:
                if row.get("condition") not in CONDITIONS:
                    continue

                target = get_target(row, path)
                prediction = row.get(
                    "predicted_distribution",
                    row.get("predicted_probs"),
                )

                if prediction is None:
                    raise ValueError(
                        f"Missing prediction probabilities in {path}"
                    )

                prediction = np.asarray(prediction, dtype=float)
                if (
                    prediction.ndim != 1
                    or len(prediction) != len(target)
                    or not np.isfinite(prediction).all()
                    or (prediction < 0).any()
                    or not np.isclose(
                        prediction.sum(), 1.0, atol=PROB_TOL, rtol=0
                    )
                ):
                    raise ValueError(
                        f"Invalid prediction probabilities in {path}"
                    )

                if "js_divergence_bits" not in row:
                    raise ValueError(
                        f"Missing js_divergence_bits in {path}"
                    )

                js_value = float(row["js_divergence_bits"])
                if not np.isfinite(js_value):
                    raise ValueError(
                        f"Non-finite JS divergence in {path}"
                    )

                if "seed" not in row:
                    raise ValueError(f"Missing seed in {path}")

                example_id = get_example_id(row, dataset, path)
                entropy = normalized_entropy(target)

                dataset_records.append(
                    {
                        "dataset": dataset,
                        "source_file": str(path),
                        "example_id": example_id,
                        "seed": int(row["seed"]),
                        "budget": get_budget(row, path),
                        "condition": row["condition"],
                        "target": json.dumps(
                            target.tolist(), separators=(",", ":")
                        ),
                        "entropy_normalized": entropy,
                        "js_divergence_bits": js_value,
                    }
                )

        if not dataset_records:
            print(f"SKIP: no eligible comparison rows in {root}")
            continue

        raw = pd.DataFrame(dataset_records)

        # Check collisions across all original SentiMP-En records,
        # including conditions outside the selected comparison.
        if dataset == "SentiMP-En":
            source_records = []
            for path in files:
                source_records.extend(read_jsonl(path))
            text_collisions = check_senti_text_collisions(
                source_records, dataset
            )
        else:
            text_collisions = 0

        # Every example must have the same human target distribution
        # regardless of seed, budget, or training condition.
        target_counts = raw.groupby("example_id")["target"].nunique()
        conflicts = int((target_counts > 1).sum())

        if conflicts:
            raise ValueError(
                f"{dataset}: {conflicts} examples have inconsistent "
                "target distributions across runs."
            )

        # Verify entropy consistency as an independent explicit check.
        entropy_counts = (
            raw.groupby("example_id")["entropy_normalized"]
            .agg(lambda values: float(values.max() - values.min()))
        )
        inconsistent_entropy = int(
            (entropy_counts > ENTROPY_TOL).sum()
        )
        if inconsistent_entropy:
            raise ValueError(
                f"{dataset}: {inconsistent_entropy} examples have "
                "inconsistent entropy values across runs."
            )

        # Require exactly one prediction from each condition for every
        # example/seed/budget combination.
        key = ["example_id", "seed", "budget"]
        expected_conditions = sorted(CONDITIONS)
        condition_counts = raw.groupby(key)["condition"].agg(list)

        invalid = condition_counts[
            condition_counts.apply(
                lambda values: sorted(values) != expected_conditions
            )
        ]

        if len(invalid):
            raise ValueError(
                f"{dataset}: {len(invalid)} example/seed/budget groups "
                "do not contain exactly one row per comparison condition. "
                "Check missing and duplicate predictions."
            )

        # Each example has one stable target distribution, so determine
        # entropy cutoffs from unique examples rather than repeated runs.
        unique_examples = (
            raw[["example_id", "entropy_normalized"]]
            .drop_duplicates()
            .sort_values("example_id")
        )

        q1 = float(unique_examples["entropy_normalized"].quantile(1 / 3))
        q2 = float(unique_examples["entropy_normalized"].quantile(2 / 3))

        def assign_stratum(entropy):
            if entropy <= q1:
                return "low"
            if entropy <= q2:
                return "medium"
            return "high"

        unique_examples["ambiguity_stratum"] = (
            unique_examples["entropy_normalized"].map(assign_stratum)
        )

        stratum_map = unique_examples.set_index("example_id")[
            "ambiguity_stratum"
        ]

        raw["ambiguity_stratum"] = raw["example_id"].map(stratum_map)

        # Report thresholds and distinct-example counts. Tied entropy
        # values can cause the strata to be unequal in size.
        stratum_sizes = (
            unique_examples["ambiguity_stratum"]
            .value_counts()
            .to_dict()
        )

        threshold_rows.append(
            {
                "dataset": dataset,
                "n_unique_examples": len(unique_examples),
                "entropy_cutoff_q1": q1,
                "entropy_cutoff_q2": q2,
                "low_unique_examples": int(stratum_sizes.get("low", 0)),
                "medium_unique_examples": int(
                    stratum_sizes.get("medium", 0)
                ),
                "high_unique_examples": int(stratum_sizes.get("high", 0)),
                "strata_method": "empirical_entropy_quantiles_with_ties",
            }
        )

        audit_rows.append(
            {
                "dataset": dataset,
                "prediction_rows": len(raw),
                "unique_example_ids": raw["example_id"].nunique(),
                "examples_with_conflicting_targets": conflicts,
                "examples_with_inconsistent_entropy": inconsistent_entropy,
                "normalized_text_collisions": text_collisions,
                "invalid_paired_groups": len(invalid),
                "budgets": ",".join(
                    map(str, sorted(raw["budget"].unique()))
                ),
                "seeds": ",".join(
                    map(str, sorted(raw["seed"].unique()))
                ),
            }
        )

        # Pivot only after checking uniqueness and complete pairing.
        wide = raw.pivot(
            index=[
                "example_id",
                "seed",
                "budget",
                "entropy_normalized",
                "ambiguity_stratum",
            ],
            columns="condition",
            values="js_divergence_bits",
        ).reset_index()

        wide["js_delta_soft_minus_ce"] = (
            wide["human_soft_targets"] - wide["majority_ce"]
        )
        wide["dataset"] = dataset
        all_rows.append(wide)

    if not all_rows:
        raise SystemExit("No eligible prediction rows found.")

    paired = pd.concat(all_rows, ignore_index=True)

    # Keep these files as reproducible diagnostic artifacts.
    paired.to_csv(
        OUT / "paired_example_ambiguity_differences.csv",
        index=False,
    )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT / "schema_and_alignment_audit.csv", index=False)

    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(OUT / "ambiguity_entropy_thresholds.csv", index=False)

    summary = (
        paired.groupby(["dataset", "budget", "ambiguity_stratum"])
        .agg(
            n_paired_examples=("js_delta_soft_minus_ce", "size"),
            mean_js_delta=("js_delta_soft_minus_ce", "mean"),
            sd_js_delta=("js_delta_soft_minus_ce", "std"),
            mean_entropy=("entropy_normalized", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT / "ambiguity_stratum_descriptives.csv",
        index=False,
    )

    print("\nDATA INTEGRITY AUDIT")
    print(audit.to_string(index=False))

    print("\nENTROPY CUT-OFFS AND UNIQUE-EXAMPLE STRATUM SIZES")
    print(thresholds.to_string(index=False))

    print("\nDESCRIPTIVE STRATIFIED RESULTS")
    print(summary.round(6).to_string(index=False))

    print(f"\nSaved audit files under: {OUT}")
    print(
        "Interpretation: descriptive results only; "
        "these outputs do not perform significance tests."
    )


if __name__ == "__main__":
    main()
