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


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_target(row):
    value = row.get("human_distribution", row.get("target_probs"))
    if value is None:
        raise ValueError("Missing human target distribution")
    p = np.asarray(value, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("Invalid target distribution")
    if (p < 0).any() or not np.isclose(p.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Target probabilities invalid: {p}")
    return p


def normalized_entropy(p):
    c = len(p)
    if c <= 1:
        return 0.0
    return float(-(p * np.log(np.clip(p, EPS, 1))).sum() / np.log(c))


all_rows = []
audit_rows = []

for dataset, root in ROOTS.items():
    if not root.exists():
        print(f"SKIP missing directory: {root}")
        continue

    files = list(root.rglob("predictions.jsonl"))
    if not files:
        print(f"SKIP: no predictions.jsonl under {root}")
        continue

    dataset_rows = []

    for path in files:
        records = read_jsonl(path)

        # SentiMP-En uses one combined predictions file; DynaSent uses
        # one file per round/budget.
        for r in records:
            if r.get("condition") not in CONDITIONS:
                continue

            p = get_target(r)
            entropy = normalized_entropy(p)

            if dataset.startswith("DynaSent"):
                example_id = r.get("text_id")
                if not example_id:
                    raise ValueError(f"Missing text_id in {path}")
            else:
                # Text is the only available example identifier in the
                # saved SentiMP-En predictions. Audit collisions below.
                text = " ".join(str(r.get("text", "")).casefold().split())
                if not text:
                    raise ValueError(f"Missing text in {path}")
                example_id = text

            pred = r.get("predicted_distribution", r.get("predicted_probs"))
            if pred is None:
                raise ValueError(f"Missing prediction probabilities in {path}")

            dataset_rows.append({
                "dataset": dataset,
                "source_file": str(path),
                "example_id": str(example_id),
                "seed": int(r["seed"]),
                "budget": int(r.get("budget_requested", path.parent.name.replace("budget_", "0"))),
                "condition": r["condition"],
                "target": json.dumps(p.tolist(), separators=(",", ":")),
                "entropy_normalized": entropy,
                "js_divergence_bits": float(r["js_divergence_bits"]),
            })

    if not dataset_rows:
        continue

    raw = pd.DataFrame(dataset_rows)

    # Same example should have the same human distribution across seeds
    # and conditions. Abort rather than silently averaging conflicts.
    target_counts = raw.groupby("example_id")["target"].nunique()
    conflicts = int((target_counts > 1).sum())

    # Check whether an example identifier maps to multiple distinct texts.
    if dataset == "SentiMP-En":
        text_map = {}
        for r in read_jsonl(root / "predictions.jsonl"):
            key = " ".join(str(r.get("text", "")).casefold().split())
            text_map.setdefault(key, set()).add(str(r.get("text", "")))
        text_collisions = sum(len(v) > 1 for v in text_map.values())
    else:
        text_collisions = 0

    audit_rows.append({
        "dataset": dataset,
        "prediction_rows": len(raw),
        "unique_example_ids": raw["example_id"].nunique(),
        "examples_with_conflicting_targets": conflicts,
        "normalized_text_collisions": text_collisions,
        "budgets": ",".join(map(str, sorted(raw["budget"].unique()))),
        "seeds": ",".join(map(str, sorted(raw["seed"].unique()))),
    })

    if conflicts:
        raise ValueError(
            f"{dataset}: {conflicts} examples have inconsistent target distributions"
        )

    # Each matched example/seed/budget needs exactly one row per condition.
    key = ["example_id", "seed", "budget"]
    counts = raw.groupby(key)["condition"].agg(list)
    invalid = counts[
        counts.apply(
            lambda xs: sorted(xs) != ["human_soft_targets", "majority_ce"]
        )
    ]
    if len(invalid):
        raise ValueError(
            f"{dataset}: {len(invalid)} example/seed/budget groups lack "
            "exactly one row per comparison condition. Inspect prediction "
            "coverage before computing paired ambiguity effects."
        )

    wide = raw.pivot(
        index=key + ["entropy_normalized"],
        columns="condition",
        values="js_divergence_bits",
    ).reset_index()

    wide["js_delta_soft_minus_ce"] = (
        wide["human_soft_targets"] - wide["majority_ce"]
    )

    # Three descriptive strata based on entropy quantiles within each dataset.
    # Quantile ties may result in fewer than three distinct groups.
    q1 = wide["entropy_normalized"].quantile(1 / 3)
    q2 = wide["entropy_normalized"].quantile(2 / 3)

    wide["ambiguity_stratum"] = np.select(
        [
            wide["entropy_normalized"] <= q1,
            wide["entropy_normalized"] <= q2,
        ],
        ["low", "medium"],
        default="high",
    )

    wide["dataset"] = dataset
    all_rows.append(wide)

if not all_rows:
    raise SystemExit("No eligible prediction rows found.")

paired = pd.concat(all_rows, ignore_index=True)
paired.to_csv(OUT / "paired_example_ambiguity_differences.csv", index=False)

audit = pd.DataFrame(audit_rows)
audit.to_csv(OUT / "schema_and_alignment_audit.csv", index=False)

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
summary.to_csv(OUT / "ambiguity_stratum_descriptives.csv", index=False)

print("\nAUDIT")
print(audit.to_string(index=False))

print("\nDESCRIPTIVE STRATIFIED RESULTS")
print(summary.round(6).to_string(index=False))

print(f"\nSaved new files under: {OUT}")
print("These are descriptive results, not significance tests.")
