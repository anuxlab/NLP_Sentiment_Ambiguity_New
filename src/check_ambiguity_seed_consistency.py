
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
src = root / "results/ambiguity_paired_analysis/paired_predictions.csv"
out = root / "results/ambiguity_paired_analysis/ambiguity_seed_consistency.csv"

df = pd.read_csv(src)
results = []

for (dataset, budget), part in df[
    df["dataset"].isin(["DynaSent Round 1", "DynaSent Round 2"])
].groupby(["dataset", "budget"]):

    seed_effects = {}
    for group in ["H=0", "H>0"]:
        g = part[part["ambiguity_group"] == group]
        seed_effects[group] = g.groupby("seed")["delta_js"].mean()

    paired = pd.concat(seed_effects, axis=1).dropna()
    paired["interaction"] = paired["H>0"] - paired["H=0"]
    x = paired["interaction"].to_numpy()

    results.append({
        "dataset": dataset,
        "budget": int(budget),
        "n_seeds": len(x),
        "negative_interaction_seeds": int((x < 0).sum()),
        "zero_interaction_seeds": int(np.isclose(x, 0).sum()),
        "positive_interaction_seeds": int((x > 0).sum()),
        "mean_interaction": x.mean(),
        "median_interaction": np.median(x),
        "min_interaction": x.min(),
        "max_interaction": x.max(),
    })

result = pd.DataFrame(results)
out.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out, index=False)

print(result.round(6).to_string(index=False))
print(f"\nSaved: {out}")
