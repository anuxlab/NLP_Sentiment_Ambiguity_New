# Controlled Soft-Target Sentiment Learning Under Scarce Labels

**Document status:** Research consolidation draft
**Repository:** `NLP_Sentiment_Ambiguity_New`
**Research direction:** Human soft-target sentiment learning under scarce labels.

## 1. Research objective

Investigate whether training sentiment classifiers on empirical human label distributions improves agreement with observed human judgments when labeled training data is scarce, and whether the benefit varies by label budget and target-distribution uncertainty.

## 2. Research questions

- **RQ1 — Distributional agreement:** Does human soft-target training reduce Jensen–Shannon (JS) divergence relative to majority-label cross-entropy?
- **RQ2 — Label budget:** Does the relative effect vary at budgets 25, 50, 100, 200, and 300?
- **RQ3 — Target uncertainty:** Does the relative effect differ between unanimous and non-unanimous examples?

## 3. Experimental conditions

The DynaSent multiseed experiments include majority-label cross-entropy, human soft-target training, label smoothing, and null-augmented human soft targets. The primary comparisons evaluate the first three conditions within each reported experiment; the null-augmented condition is analyzed separately.

The documented DynaSent multiseed configuration uses five seeds, budgets of 100, and 10 epochs. Its saved verification report covers rounds 1 and 2. The ten-seed analysis uses five label budgets.

## 4. Primary metric and statistical analysis

The primary contrast is:

\[
\Delta JS = JS_{\mathrm{soft}}-JS_{\mathrm{majority}}
\]

Negative values favor human soft-target training in agreement with the observed human target distributions.

Paired seed-level sign-flip tests use Holm correction across 15 primary comparisons. Two-level bootstrap intervals are reported separately and are unadjusted.

The ambiguity analysis evaluates the interaction between the JS change on non-unanimous examples and the JS change on unanimous examples. Its Holm correction covers 10 DynaSent round-budget comparisons.

## 5. Findings supported by the current results

Human soft targets have lower mean JS than majority-label training in all 15 primary comparisons. After Holm correction, the primary paired tests identify statistically significant differences at DynaSent Round 2, budget 300, and SentiMP-En, budget 300.

The ambiguity-interaction analysis identifies significant interactions for DynaSent Round 1 at budgets 100, 200, and 300. The corresponding Round 2 interactions do not survive Holm correction.

These results are conditional on the evaluated datasets, examples, and experimental protocol. They do not establish universal superiority of soft-target training or imply that lower JS divergence necessarily yields higher classification accuracy.

## 6. Validation and reproducibility

- Automated test suite: 61 tests passed.
- Independent JS-divergence verification: 883,000 rows checked; no errors exceeded the \(10^{-6}\) tolerance.
- Ambiguity alignment audit: no conflicting targets, inconsistent entropy values, normalized-text collisions, or invalid paired groups were reported in the audited data.
- Git repository: latest analysis commits pushed to `origin/main`.

The numerical JS verification establishes consistency between saved divergences and the stored distributions; it is not a substitute for independent replication of the experimental design.

## 7. Canonical results

- Ambiguity interactions: `results/ambiguity_paired_analysis/`
- Ambiguity-stratified audit: `results/ambiguity_stratified_audit/`
- JS verification: `results/js_divergence_verification/`
- Relevant scripts: `src/analyze_ambiguity_paired.py`, `src/analyze_ambiguity_strata.py`, and `src/verify_js_divergence.py`

## 8. Limitations and pending work

The SentiMP-En evaluation set contains only 95 examples, including two tied cases. Entropy-stratified groups are uneven, and DynaSent has no examples in the high-entropy stratum under the current quantile thresholds. These limitations constrain fine-grained claims about ambiguity.

Before submission, verify the canonical artifact-generation commands, document the precise model and training configuration for each experiment, confirm the provenance and licensing of the datasets, and ensure the manuscript tables match the committed results.
