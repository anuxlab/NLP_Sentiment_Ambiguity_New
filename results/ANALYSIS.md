# Consolidated results and interpretation

## Scope
This package consolidates the canonical controlled TF-IDF soft-target study and the supporting report/presentation assets supplied in `Archive.zip`. The three datasets/settings are DynaSent Round 1, DynaSent Round 2, and SentiMP-En. The primary contrast is human-soft-target JS divergence minus majority-label CE JS divergence; negative values favour human soft targets.

## Main result
- Human soft targets have lower mean JS divergence in **15 of 15** dataset/round-by-budget comparisons.
- After paired seed-level exact sign-flip tests with Holm correction across all 15 comparisons, **1 comparison** survives the 0.05 threshold: DynaSent Round 2 at budget 300 (adjusted p=0.02930).
- The largest mean reduction in JS is 0.0186 bits at SentiMP-En, budget 300. This is an effect-size description, not a general superiority claim.
- Mean JS generally decreases as the label budget increases, although individual conditions and settings need not be strictly monotonic.

## Classification and interpretation
Human soft targets often have higher macro-F1 than majority CE, but classification improvements are not universal across all settings and budgets. Label smoothing is a useful comparator because it smooths targets without using the empirical per-example annotator distribution. JS divergence is the primary metric because it evaluates predictions against human vote proportions rather than only the majority label.

## Composition ablations
The null-augmented DynaSent condition and tie-inclusive SentiMP-En condition alter which examples enter training. They are data-composition ablations and must not be interpreted as pure loss-function comparisons. The canonical matched-condition summary excludes these ablations.

## Uncertainty and limitations
- Holm-adjusted p-values are from paired seed-level exact sign-flip tests; the bootstrap intervals in the primary results table are unadjusted. They answer different uncertainty questions and must not be conflated.
- DynaSent uses 10 seeds (0–9) per budget in the primary TF-IDF study; SentiMP-En uses 5 seeds in the supplied primary JS test table. The sample and seed counts differ by setting.
- Results have already been inspected and are exploratory, not a preregistered confirmatory study.
- The included Tiny-BERT pilot report is a separate model-family experiment (budget 100, seeds 0–4). Its results should not be merged numerically with the TF-IDF learning-curve results.
- The four-class project mapping includes negative, neutral, positive, and mixed; document the mapping when comparing to benchmark papers.

## Recommended presentation claim
“Human vote-distribution supervision consistently lowers mean JS divergence across the evaluated label budgets and datasets; corrected seed-level evidence is strongest for DynaSent Round 2 at 300 labels. Effects on majority-label classification metrics are mixed, so the result supports a promising direction rather than universal superiority.”

## Canonical files
- `tables/primary_js_results.csv`: paired primary contrast, Holm-adjusted tests, and bootstrap intervals.
- `tables/dataset_condition_summary.csv`: condition-level metrics by dataset and label budget.
- `tables/paired_uncertainty_summary.csv`: seed-paired secondary metric contrasts.
- `figures/js_learning_curves.png`: primary metric across label budgets.
- `figures/primary_js_effects.png`: primary contrast and unadjusted bootstrap intervals.
- `figures/macro_f1_learning_curves.png`: classification metric context.
