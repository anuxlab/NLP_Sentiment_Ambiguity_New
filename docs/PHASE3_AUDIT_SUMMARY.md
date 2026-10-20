# Phase 3 statistical and reproducibility audit

**Source run:** GitHub Actions run `39378505404`  
**Backend:** Tiny-BERT  
**Status:** Local artifact audit passed; workflow/code changes require a fresh remote Actions run after commit.

## Data and provenance checks

- Both DynaSent rounds present.
- Seeds 0–4 and all four conditions present in each round.
- 40 metric records and 86,400 per-example prediction rows.
- Configuration backend/round provenance agrees with metric and prediction metadata.
- Effective Tiny-BERT learning rate is `tiny_bert_lr = 0.00002`; the legacy generic `lr` field is absent from the run's config.
- The effective learning rate matches all per-run metric rows.
- Recomputed per-example JS divergence, group mean/median JS, soft Brier score, majority-label count, accuracy, macro-F1, and ECE agree with the stored metrics within the configured numerical tolerance.
- Paired test IDs and human distributions are consistent across conditions and seeds.

## Primary results

| Round | Mean JS difference (soft targets − majority CE) | Paired test-example bootstrap 95% CI | Sign-flip p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| Round 1 | -0.004174 | [-0.004899, -0.003459] | 0.000100 | 0.000200 |
| Round 2 | -0.003991 | [-0.004374, -0.003604] | 0.000100 | 0.000200 |

Negative differences favor human-distribution soft targets on the evaluated test sets.

## Statistical interpretation

The five-seed runs and results have already been inspected, so all inferential results are exploratory. Bootstrap intervals resample test examples after averaging paired condition differences across the five fixed seeds. They do not estimate between-training-run uncertainty. The paired sign-flip procedure assumes sign exchangeability of item-level differences under the null and is conditional on the observed fitted models and test examples. Holm correction covers the two prespecified round-specific primary comparisons only.

The null-augmented soft-target condition uses a different training-data composition and remains exploratory. Do not pool the two rounds into one headline effect or claim universal superiority from these results.

## Code changes in this patch

1. Runner emits backend-specific learning-rate metadata and an explicit `audit.effective_learning_rate` record.
2. Verifier rejects missing/conflicting learning-rate provenance and checks metric learning rates.
3. Statistical analyzer checks configs, per-group counts, and recomputes metrics from predictions before writing outputs.
4. Manifest records the source Actions run/artifact, effective learning rate, and provenance checks; obsolete legacy learning-rate note removed.
5. Workflow runs artifact verification before statistical analysis.
6. Regression tests cover invalid learning-rate metadata and inconsistent metrics.

## Validation performed locally

- `pytest -q`: 45 passed.
- Python `py_compile`: passed for source and test files.
- Workflow YAML parsed successfully.
- Workflow shell blocks passed `bash -n` after substituting GitHub expression placeholders.
- Updated verifier passed against the supplied artifact: 40/40 groups, 86,400 prediction rows.
- Updated analyzer regenerated the report and figures from the supplied artifact with 10,000 bootstrap and 10,000 permutation replicates.

These local checks do not replace a fresh GitHub Actions run after the patch is pushed.
