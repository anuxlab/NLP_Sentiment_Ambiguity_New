# Results summary — controlled TF-IDF soft-target study

**Evidence date:** 20 October 2026. This package uses the supplied canonical report and CSV outputs, superseding provisional values in the earlier draft package.

## Experimental coverage
- DynaSent Round 1 and Round 2: budgets 25, 50, 100, 200, 300; 10 seeds (0–9); four conditions per round/budget in the raw archive.
- SentiMP-En: five budgets; 10 seeds (0–9); four conditions including tie-inclusive soft targets.
- DynaSent: 400/400 expected metric combinations and 864,000 per-example prediction rows; zero validation errors in the supplied archive audit.
- SentiMP-En: 200 metric rows and 19,000 per-example prediction rows; train/dev/test split 337/48/95.

## Primary metric and contrast
Primary metric is mean Jensen–Shannon divergence (bits) against the empirical human vote distribution; lower is better. The contrast is human-soft-target minus majority-CE JS divergence; negative values favor soft-target training.

## Main inferential result
Across 15 dataset/round-by-budget comparisons, the mean contrast is negative in all 15. In seed-level exact sign-flip tests with Holm correction across all 15 comparisons, **only DynaSent Round 2 at budget 300 survives correction** (Holm-adjusted p = 0.029297). SentiMP-En budget 300 is directionally favorable but does not survive the seed-level Holm correction.

The two-level bootstrap resamples seeds and test examples (10,000 replicates; seed 20261020). Its intervals are unadjusted and address different uncertainty than seed-level sign-flip tests. Do not treat an unadjusted interval excluding zero as proof of Holm-adjusted significance.

## Matched-condition mean JS by budget (mean ± SD across seeds; lower is better)

### DynaSent Round 1

| Budget | Majority CE | Human soft targets | Label smoothing |
|---:|---:|---:|---:|
| 25 | 0.4073 ± 0.0063 | 0.4066 ± 0.0058 | 0.4073 ± 0.0063 |
| 50 | 0.3969 ± 0.0067 | 0.3955 ± 0.0059 | 0.3968 ± 0.0066 |
| 100 | 0.3872 ± 0.0094 | 0.3827 ± 0.0048 | 0.3867 ± 0.0089 |
| 200 | 0.3739 ± 0.0077 | 0.3694 ± 0.0050 | 0.3732 ± 0.0072 |
| 300 | 0.3683 ± 0.0072 | 0.3644 ± 0.0038 | 0.3683 ± 0.0065 |

### DynaSent Round 2

| Budget | Majority CE | Human soft targets | Label smoothing |
|---:|---:|---:|---:|
| 25 | 0.4416 ± 0.0175 | 0.4383 ± 0.0158 | 0.4415 ± 0.0174 |
| 50 | 0.4381 ± 0.0248 | 0.4342 ± 0.0206 | 0.4378 ± 0.0240 |
| 100 | 0.4432 ± 0.0192 | 0.4335 ± 0.0130 | 0.4417 ± 0.0177 |
| 200 | 0.4314 ± 0.0121 | 0.4226 ± 0.0107 | 0.4297 ± 0.0110 |
| 300 | 0.4265 ± 0.0107 | 0.4171 ± 0.0087 | 0.4246 ± 0.0097 |

### SentiMP-En

| Budget | Majority CE | Human soft targets | Label smoothing |
|---:|---:|---:|---:|
| 25 | 0.3224 ± 0.0117 | 0.3152 ± 0.0154 | 0.3154 ± 0.0096 |
| 50 | 0.2938 ± 0.0110 | 0.2919 ± 0.0120 | 0.2986 ± 0.0073 |
| 100 | 0.2668 ± 0.0113 | 0.2659 ± 0.0133 | 0.2837 ± 0.0100 |
| 200 | 0.2675 ± 0.0149 | 0.2601 ± 0.0126 | 0.2784 ± 0.0054 |
| 300 | 0.2524 ± 0.0046 | 0.2338 ± 0.0032 | 0.2703 ± 0.0018 |

## Interpretation guardrails
- Human soft targets have lower mean JS than majority CE at all observed budgets in all three settings, but corrected inferential evidence is limited.
- The null-augmented DynaSent condition changes the training composition (50% null-majority examples by default); treat it as a data-availability ablation, not a loss-only causal comparison.
- The tie-inclusive SentiMP-En condition also changes training composition and must be reported separately.
- JS and soft Brier use vote-distribution targets, including tied-majority examples; accuracy and macro-F1 exclude examples without a unique majority label.
- Fixed test sets mean seed-level intervals do not quantify test-population sampling uncertainty. Current results are exploratory because results have already been inspected.
