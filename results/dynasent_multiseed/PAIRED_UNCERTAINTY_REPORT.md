# Paired uncertainty and disagreement analysis

Bootstrap: 2,000 paired test-example resamples per round/tier; the same resampled example indices are used across all five seeds and candidate/baseline pairs. Estimates are conditional on the five trained seeds.

## Primary contrast: human soft targets vs majority CE (JS divergence)

| Round | Mean difference (candidate − baseline) | 95% paired bootstrap CI | Holm-adjusted p (two-round primary family) |
|---|---:|---:|---:|
| round1 | -0.00297 | [-0.00363, -0.00226] | 0.0020 |
| round2 | -0.00532 | [-0.00628, -0.00434] | 0.0020 |

Negative JS differences favour the candidate. The intervals are conditional on fixed trained models and should not be presented as seed-level uncertainty. Bootstrap p-values are approximate and are supplementary to the intervals.

## Disagreement tiers

The current vote distributions contain two observed tiers only: unanimous and non-unanimous. The subgroup metrics are descriptive; tier sizes differ by round. See `disagreement_tier_metrics.csv` and `paired_bootstrap_all_tiers.csv`.

## Caveat

Current-run analyses are exploratory because the outputs were already inspected. Use `FROZEN_ANALYSIS_PLAN.md` prospectively for the next experiment cycle. The audit condition-list mismatch remains a reproducibility issue to fix.
