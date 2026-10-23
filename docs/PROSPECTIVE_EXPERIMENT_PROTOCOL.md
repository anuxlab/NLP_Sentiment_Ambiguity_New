# Prospective DynaSent experiment protocol

Version: 1.0  
Purpose: specify the next confirmatory cycle before inspecting its outcomes. Existing archived runs remain exploratory and must not be relabelled as prospective/confirmatory.

## Research question
Under scarce labeled sentiment data, does training on human annotator-vote distributions improve predictive distribution alignment versus majority-label cross-entropy and conventional label smoothing, and is the benefit larger on high-disagreement test examples?

## Dataset and scope
- Primary dataset: DynaSent Round 1 and Round 2, analyzed separately.
- Class order: `negative`, `neutral`, `positive`, `mixed`.
- Primary backend for this cycle: TF-IDF, to keep the objective comparison computationally controlled.
- Label budget: 100 eligible training examples per seed and round, unless a documented feasibility issue requires a protocol amendment before training.
- Seeds: use a new, fixed, published seed list distinct from the existing exploratory seeds 0–4 (recommended: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19).
- Keep dev/test split contents fixed. Apply overlap filtering and deduplication to training data only. Never select model settings using test results.

## Conditions
1. `majority_ce`: hard majority label, cross-entropy.
2. `human_soft_targets`: full human vote distribution as target probabilities.
3. `label_smoothing`: hard majority label with smoothing epsilon 0.1.
4. `human_soft_targets_null_augmented`: optional secondary data-composition ablation. Include only in a separate analysis; it is not a matched-data comparison because it includes examples without a majority gold label.

For the three primary conditions, use exactly the same sampled training example IDs for a given round and seed. Record those IDs or their stable hashes in the manifest. Use the same model settings and training budget wherever applicable. Record every deviation.

## Outcomes
### Primary outcome
Per-example Jensen–Shannon divergence between predicted class probabilities and the human annotator-vote distribution, averaged over the fixed test set. Lower is better.

### Secondary outcomes
- Majority-label macro-F1 and accuracy.
- Multiclass Brier score and expected calibration error, with binning details fixed in advance.
- JS divergence stratified by disagreement tier, using tiers defined from the human vote distribution alone and fixed before model evaluation.
- Difference-in-differences: (soft-target minus majority-CE JS in high disagreement) minus (soft-target minus majority-CE JS in low disagreement). Negative values support a larger soft-target benefit in the high-disagreement group.

Do not use majority-label accuracy as a substitute for distributional alignment. Report per-class support and explicitly state when a class has zero majority-label test support.

## Statistical analysis
- Unit of primary pairing: same test examples, with model predictions paired within each seed.
- Report mean effect sizes and uncertainty intervals; distinguish test-example uncertainty from seed/retraining uncertainty.
- Primary contrasts: `human_soft_targets` vs `majority_ce` on JS divergence separately for Round 1 and Round 2.
- Secondary contrasts: `label_smoothing` vs `majority_ce`; disagreement-tier effects and interaction; hard-label metrics and calibration.
- Control family-wise error across the two primary round-level contrasts with Holm correction at alpha = 0.05. Clearly label all secondary/subgroup analyses as secondary and adjust their p-values within a separately declared family.
- Publish all seed-level estimates, not only pooled means. Avoid claiming statistical generalization from test-example bootstrap alone when the number of independent training seeds is small.

## Integrity and reproducibility gates
Before interpreting results, automated checks must confirm:
1. Each round/seed has exactly the three primary conditions; the optional null-augmented condition is explicitly tracked separately.
2. Primary conditions share identical sampled training IDs within each round/seed.
3. Prediction IDs match the untouched test split and are identical across conditions.
4. Each probability vector has four finite nonnegative values summing to 1 within tolerance 1e-5.
5. Human vote distributions are finite, nonnegative, four-class vectors summing to 1 within tolerance 1e-5.
6. Metrics can be reproduced from saved predictions; no console-only values are accepted as final evidence.
7. Configuration, code revision, data hashes, seed list, environment, manifests, metrics, predictions, and audit outputs are archived together.

## Decision rule and claim discipline
The main claim is supported only if the soft-target JS effect is in the favorable direction in both rounds and the two primary contrasts remain significant after the predeclared Holm correction. If only one round supports the effect, report a round-dependent result. A larger effect in high-disagreement examples is evidence for the ambiguity-specific hypothesis only if the predeclared interaction contrast supports it; visual differences alone are insufficient.

## Required outputs
`metrics.csv`, `predictions.jsonl`, `training_manifest.jsonl`, `config_and_audit.json`, `run_completeness.json`, `paired_primary_tests.csv`, `ambiguity_interaction.csv`, `calibration_metrics.csv`, `REPORT.md`, and a SHA-256 manifest. Preserve prior exploratory outputs unchanged.
