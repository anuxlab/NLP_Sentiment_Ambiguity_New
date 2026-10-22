import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOTS = {
    "DynaSent Round 1": Path("results/dynasent_controlled/round1"),
    "DynaSent Round 2": Path("results/dynasent_controlled/round2"),
    "SentiMP-En": Path("results/sentimp_en_controlled_seed10"),
}

OUT = Path("results/ambiguity_paired_analysis")
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = {"majority_ce", "human_soft_targets"}
EPS = 1e-12


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def target_of(r):
    return r.get("human_distribution", r.get("target_probs"))


def entropy(p):
    p = np.asarray(p, dtype=float)
    return float(
        -(p * np.log(np.clip(p, EPS, 1))).sum() / np.log(len(p))
    )


def load_dataset(name, root):
    files = list(root.rglob("predictions.jsonl"))
    rows = []

    for path in files:
        for r in read_jsonl(path):
            if r.get("condition") not in CONDITIONS:
                continue

            target = target_of(r)
            pred = r.get("predicted_distribution", r.get("predicted_probs"))
            if target is None or pred is None:
                raise ValueError(f"Missing distribution in {path}")

            text_id = r.get("text_id")
            if text_id is None:
                text_id = " ".join(str(r.get("text", "")).casefold().split())

            rows.append({
                "dataset": name,
                "example_id": str(text_id),
                "seed": int(r["seed"]),
                "budget": int(r.get("budget_requested",
                                    path.parent.name.replace("budget_", "0"))),
                "condition": r["condition"],
                "target": json.dumps(target, separators=(",", ":")),
                "entropy": entropy(target),
                "js": float(r["js_divergence_bits"]),
            })

    if not rows:
        raise ValueError(f"No prediction rows found for {name}")

    df = pd.DataFrame(rows)

    if (df.groupby("example_id")["target"].nunique() > 1).any():
        raise ValueError(f"Conflicting human targets in {name}")

    # Require exactly one row per condition for every matched example,
    # seed, and budget.
    key = ["example_id", "seed", "budget"]
    pairs = df.pivot(
        index=["dataset"] + key + ["entropy"],
        columns="condition",
        values="js",
    ).reset_index()

    if pairs[["majority_ce", "human_soft_targets"]].isna().any().any():
        raise ValueError(f"Missing paired predictions in {name}")

    pairs["delta_js"] = pairs["human_soft_targets"] - pairs["majority_ce"]

    # Use the empirically observed entropy categories, not quantile bins.
    if name.startswith("DynaSent"):
        pairs["ambiguity_group"] = np.where(
            pairs["entropy"] < EPS, "H=0", "H>0"
        )
    else:
        pairs["ambiguity_group"] = np.select(
            [
                pairs["entropy"] < EPS,
                pairs["entropy"] >= 1 - EPS,
            ],
            ["H=0", "H=1"],
            default="0<H<1",
        )

    return pairs


all_pairs = []

for name, root in ROOTS.items():
    if not root.exists():
        print(f"SKIP missing: {root}")
        continue
    df = load_dataset(name, root)
    all_pairs.append(df)
    print(
        f"{name}: {df['example_id'].nunique()} examples, "
        f"{df['seed'].nunique()} seeds, "
        f"{df['budget'].nunique()} budgets"
    )

if not all_pairs:
    raise SystemExit("No datasets loaded")

paired = pd.concat(all_pairs, ignore_index=True)
paired.to_csv(OUT / "paired_predictions.csv", index=False)

summary = (
    paired.groupby(["dataset", "budget", "ambiguity_group"])
    .agg(
        n_example_seed_pairs=("delta_js", "size"),
        n_unique_examples=("example_id", "nunique"),
        n_seeds=("seed", "nunique"),
        mean_delta_js=("delta_js", "mean"),
        sd_delta_js=("delta_js", "std"),
    )
    .reset_index()
)

summary.to_csv(OUT / "descriptive_summary.csv", index=False)

print("\nDescriptive paired results; negative delta favors soft targets")
print(summary.round(6).to_string(index=False))
print(f"\nSaved results to {OUT}")
print("These summaries are descriptive, not significance tests.")
