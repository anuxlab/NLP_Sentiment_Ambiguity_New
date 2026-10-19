# Frozen analysis plan for the next experiment cycle

**Status:** This protocol is frozen prospectively for the next experiment cycle. The current five-seed results have already been inspected; analyses of those results are exploratory, not preregistered or confirmatory.

## Research question
Under a scarce label budget, does training against human annotator vote distributions improve alignment between model output distributions and held-out human vote distributions, compared with majority-label cross-entropy? Separately, does adding eligible null-majority training examples help?

## Primary outcome and contrast
- Primary metric: mean per-example Jensen–Shannon divergence (base 2, in bits) between predicted class probabilities and held-out human vote distributions; lower is better.
- Primary contrast: `human_soft_targets` minus `majority_ce`, evaluated separately for DynaSent Round 1 and Round 2.
- Report both round-specific effect estimates and 95% paired test-example bootstrap confidence intervals. Adjust the two primary round-level tests with Holm's method.
- Bootstrap holds the five trained seeds fixed and resamples test examples with replacement. It quantifies test-example sampling uncertainty conditional on these models; it does not replace variability across independently trained seeds.

## Secondary and exploratory outcomes
- Secondary outcome: macro-F1 on examples with a valid majority gold label; accuracy reported alongside it.
- Secondary baseline: `label_smoothing` versus `majority_ce`, reported separately by round.
- Separate exploratory condition: `human_soft_targets_null_augmented` versus `majority_ce`. It uses a different training-data composition (50 null-majority + 50 non-null-majority examples at budget 100), so it is not interpreted as a matched-data causal comparison.
- Secondary subgroup analysis: unanimous vs non-unanimous human vote distributions, using entropy derived from human votes only. These are the only two observed tiers in the current dataset view (unanimous entropy 0; non-unanimous observed distribution 4:1, entropy 0.7219 bits). Do not invent empty moderate/high bins.
- Calibration metrics (soft Brier and ECE) are descriptive secondary metrics. ECE is bin-sensitive and will not be the primary outcome.

## Multiplicity and reporting
- Holm adjustment across the two primary JS-divergence round comparisons.
- Treat all other metrics, conditions, tiers, and interaction-like comparisons as secondary/exploratory; if inferential claims are made across a family of them, apply a declared multiplicity correction and label the family.
- Always show Round 1 and Round 2 separately; do not pool them into a single headline result.
- Report per-seed means/SDs alongside bootstrap intervals; do not conflate seed variability with test-example uncertainty.

## Data and reproducibility safeguards
- Keep train overlap/dedup filtering limited to the effective training view; preserve dev/test splits.
- Freeze label order as negative, neutral, positive, mixed.
- Never infer human disagreement from model confidence; derive tiers from annotator vote distributions.
- Do not tune hyperparameters, thresholds, or null fraction against the final test set.
- Preserve text IDs, per-example predictions, per-seed metrics, and training manifests.
- Before any confirmatory claim, verify that audit metadata lists all four conditions; the current artifact's `conditions` array omits the null-augmented condition even though metrics/manifests include it.

## Next experiment cycle
1. Fix condition metadata and add a test asserting that the declared conditions equal the observed metric/manifest conditions.
2. Re-run CI and the same five-seed experiment to confirm the fix is metadata-only and outputs remain consistent.
3. Analyze the frozen primary and secondary comparisons using the outputs above.
4. Only after that, extend to a small Transformer; preserve the same split, label budgets, metrics, seeds where feasible, and evaluation protocol.
