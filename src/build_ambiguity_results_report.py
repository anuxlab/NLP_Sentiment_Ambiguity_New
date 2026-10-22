
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "ambiguity_paired_analysis"

FILES = {
    "effects": BASE / "ambiguity_subgroup_effects.csv",
    "tests": BASE / "ambiguity_interaction_holm.csv",
    "seeds": BASE / "ambiguity_seed_consistency.csv",
}

CSV_OUTPUT = BASE / "ambiguity_final_results.csv"
MD_OUTPUT = BASE / "ambiguity_results_report.md"

KEYS = ["dataset", "budget"]
EXPECTED_TESTS = 10


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def require_columns(df, required, filename):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing columns: {sorted(missing)}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def fmt(value, digits=6):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(df, columns, headers, formatters=None):
    """Render a DataFrame as a Markdown table."""
    formatters = formatters or {}

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if col in formatters:
                value = formatters[col](value)
            elif pd.isna(value):
                value = "NA"
            else:
                value = str(value)

            values.append(str(value).replace("|", r"\|"))

        lines.append("| " + " | ".join(values) + " |")

    return lines


# ---------------------------------------------------------------------
# Load and validate inputs
# ---------------------------------------------------------------------
for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name} input:\n{path}\n"
            "Run the corresponding analysis script first."
        )

effects = pd.read_csv(FILES["effects"])
tests = pd.read_csv(FILES["tests"])
seeds = pd.read_csv(FILES["seeds"])

print("Effects columns:", effects.columns.tolist())

require_columns(
    effects,
    KEYS + ["ambiguity_group", "n_seeds", "mean_delta_js",
            "ci95_low", "ci95_high"],
    FILES["effects"].name,
)

require_columns(
    tests,
    KEYS + [
        "interaction_Hpositive_minus_H0",
        "interaction_p_value",
        "holm_adjusted_p",
        "significant_holm_005",
        "mean_delta_js_H0",
        "mean_delta_js_Hpositive",
    ],
    FILES["tests"].name,
)

require_columns(
    seeds,
    KEYS + [
        "negative_interaction_seeds",
        "positive_interaction_seeds",
        "mean_interaction",
        "median_interaction",
    ],
    FILES["seeds"].name,
)

# ---------------------------------------------------------------------
# Validate keys and reshape subgroup effects
# ---------------------------------------------------------------------
if effects.duplicated(KEYS + ["ambiguity_group"]).any():
    raise ValueError(
        "Duplicate dataset/budget/ambiguity-group rows in subgroup effects."
    )

if tests.duplicated(KEYS).any():
    raise ValueError("Duplicate dataset/budget rows in interaction tests.")

if seeds.duplicated(KEYS).any():
    raise ValueError("Duplicate dataset/budget rows in seed consistency.")

groups_found = set(effects["ambiguity_group"].dropna().unique())
expected_groups = {"H=0", "H>0"}

if groups_found != expected_groups:
    raise ValueError(
        f"Expected ambiguity groups {expected_groups}; found {groups_found}."
    )

# One row per dataset and budget, with one column per ambiguity group.
wide = effects.pivot(
    index=KEYS,
    columns="ambiguity_group",
    values="mean_delta_js",
).reset_index()

wide.columns.name = None
wide = wide.rename(
    columns={
        "H=0": "mean_delta_js_H0_from_effects",
        "H>0": "mean_delta_js_Hpositive_from_effects",
    }
)

# Check subgroup effects agree with the values used for the interaction test.
check = wide.merge(
    tests[
        KEYS + ["mean_delta_js_H0", "mean_delta_js_Hpositive"]
    ],
    on=KEYS,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

if not (check["_merge"] == "both").all():
    raise ValueError(
        "Subgroup effects and interaction tests have different dataset/budget keys."
    )

for left, right in [
    ("mean_delta_js_H0_from_effects", "mean_delta_js_H0"),
    ("mean_delta_js_Hpositive_from_effects", "mean_delta_js_Hpositive"),
]:
    if not (
        (check[left] - check[right]).abs().fillna(0) < 1e-8
    ).all():
        raise ValueError(
            f"Subgroup effect mismatch between files: {left} vs {right}"
        )

# ---------------------------------------------------------------------
# Consolidate all outputs
# ---------------------------------------------------------------------
test_columns = KEYS + [
    "mean_delta_js_H0",
    "mean_delta_js_Hpositive",
    "interaction_Hpositive_minus_H0",
    "interaction_p_value",
    "holm_adjusted_p",
    "significant_holm_005",
]

seed_columns = KEYS + [
    "negative_interaction_seeds",
    "positive_interaction_seeds",
    "mean_interaction",
    "median_interaction",
]

report = tests[test_columns].merge(
    seeds[seed_columns],
    on=KEYS,
    how="left",
    validate="one_to_one",
)

report = report.sort_values(KEYS).reset_index(drop=True)

if len(report) != EXPECTED_TESTS:
    raise ValueError(
        f"Expected {EXPECTED_TESTS} interaction tests; found {len(report)}."
    )

if report["holm_adjusted_p"].isna().any():
    raise ValueError("Missing Holm-adjusted p-values.")

# Validate that the significance flag matches the corrected p-value.
expected_significance = report["holm_adjusted_p"] < 0.05

if not (
    report["significant_holm_005"].astype(bool).to_numpy()
    == expected_significance.to_numpy()
).all():
    raise ValueError(
        "Significance flags do not match Holm-adjusted p-values."
    )

BASE.mkdir(parents=True, exist_ok=True)
report.to_csv(CSV_OUTPUT, index=False)


# ---------------------------------------------------------------------
# Build Markdown report from the corrected 10-row table
# ---------------------------------------------------------------------
lines = [
    "# Ambiguity Analysis: Results",
    "",
    "## Research question",
    "",
    "Does the change in JS divergence from human soft-target training, "
    "relative to majority-label cross-entropy, differ between unanimous "
    "and non-unanimous examples?",
    "",
    "## Main findings",
    "",
    "Negative subgroup JS changes favor human soft-target training. "
    "A negative interaction means the change is more favorable for "
    "non-unanimous examples than for unanimous examples.",
    "",
    "The interaction tests use paired seed-level effects. Holm correction "
    "is applied across the 10 DynaSent round-budget comparisons.",
    "",
    "## Interaction results",
    "",
]

lines += markdown_table(
    report,
    columns=[
        "dataset",
        "budget",
        "mean_delta_js_H0",
        "mean_delta_js_Hpositive",
        "interaction_Hpositive_minus_H0",
        "interaction_p_value",
        "holm_adjusted_p",
        "negative_interaction_seeds",
    ],
    headers=[
        "Dataset",
        "Budget",
        "ΔJS H=0",
        "ΔJS H>0",
        "Interaction",
        "Raw p",
        "Holm p",
        "Negative seeds",
    ],
    formatters={
        "budget": lambda x: str(int(x)),
        "mean_delta_js_H0": lambda x: fmt(x),
        "mean_delta_js_Hpositive": lambda x: fmt(x),
        "interaction_Hpositive_minus_H0": lambda x: fmt(x),
        "interaction_p_value": lambda x: fmt(x, 6),
        "holm_adjusted_p": lambda x: fmt(x, 6),
        "negative_interaction_seeds": lambda x: f"{int(x)}/10",
    },
)

significant = report[report["significant_holm_005"].astype(bool)]

lines.extend([
    "",
    "## Multiplicity-adjusted findings",
    "",
])

if significant.empty:
    lines.append(
        "No interaction tests survived Holm correction at α = 0.05."
    )
else:
    for _, row in significant.iterrows():
        lines.append(
            f"- **{row['dataset']}, budget {int(row['budget'])}:** "
            f"interaction = "
            f"{row['interaction_Hpositive_minus_H0']:.6f}; "
            f"Holm-adjusted p = {row['holm_adjusted_p']:.6f}."
        )

lines.extend([
    "",
    "## Seed consistency",
    "",
    "The negative-seed count describes how many of the 10 training seeds "
    "had a negative interaction. It is a robustness diagnostic, not an "
    "independent confirmation of the interaction test.",
    "",
    "## Interpretation and limitations",
    "",
    "- Inference is conditional on the selected evaluation examples.",
    "- The exact seed-level sign-flip test assumes symmetric seed-level "
    "effects under the null.",
    "- Bootstrap intervals are exploratory and are not multiplicity-adjusted.",
    "- The analysis does not establish a causal mechanism or generalization "
    "to other datasets.",
    "- Entropy strata describe uncertainty in the target distribution; "
    "they should not automatically be interpreted as raw annotator disagreement.",
    "",
    "## Figure",
    "",
    "See `ambiguity_interaction.png` and `ambiguity_interaction.pdf` "
    "in this directory, if generated.",
    "",
])

MD_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Final validation and summary
# ---------------------------------------------------------------------
saved_report = pd.read_csv(CSV_OUTPUT)

if len(saved_report) != EXPECTED_TESTS:
    raise RuntimeError(
        f"CSV validation failed: expected 10 rows, found {len(saved_report)}."
    )

md_text = MD_OUTPUT.read_text(encoding="utf-8")
if "NA | NA" in md_text:
    raise RuntimeError(
        "Markdown report appears to contain missing subgroup effects."
    )

print("\nFinal interaction table:")
print(
    report[
        [
            "dataset",
            "budget",
            "mean_delta_js_H0",
            "mean_delta_js_Hpositive",
            "interaction_Hpositive_minus_H0",
            "holm_adjusted_p",
            "significant_holm_005",
            "negative_interaction_seeds",
        ]
    ].round(6).to_string(index=False)
)

print(f"\nSaved CSV: {CSV_OUTPUT}")
print(f"Saved Markdown report: {MD_OUTPUT}")
print(f"Validated interaction rows: {len(saved_report)}")
print(f"Holm-significant interactions: {len(significant)}")
