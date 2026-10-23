# DynaSent pilot results

- Round: `round1`
- Backend: `tiny-bert`
- Budget: `50`
- Evaluation rows: `3600`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 50 | 0 | 0.44 | 0.4092 | 0.5255 | 0.3325 | 0.13814647727317692 | 0.03681772388517859 |
| human_soft_targets | 0 | 50 | 0 | 0.43 | 0.4082 | 0.5219 | 0.3194444444444444 | 0.16603676777641915 | 0.027127001591854605 |
| label_smoothing | 0 | 50 | 0 | 0.41 | 0.4090 | 0.5247 | 0.3333333333333333 | 0.14218519458025475 | 0.0321968260821369 |
| majority_ce | 1 | 50 | 0 | 0.49 | 0.4110 | 0.5418 | 0.3333333333333333 | 0.1286434675830683 | 0.12481273202432529 |
| human_soft_targets | 1 | 50 | 0 | 0.42 | 0.4063 | 0.5279 | 0.33361111111111114 | 0.12992212956315236 | 0.09759682516256966 |
| label_smoothing | 1 | 50 | 0 | 0.57 | 0.4104 | 0.5399 | 0.3338888888888889 | 0.12952739519597384 | 0.12012653209269047 |
| majority_ce | 2 | 50 | 0 | 0.61 | 0.4133 | 0.5235 | 0.33111111111111113 | 0.1999159991504336 | 0.021826246827840817 |
| human_soft_targets | 2 | 50 | 0 | 0.71 | 0.4156 | 0.5265 | 0.3213888888888889 | 0.20499501814375964 | 0.021741164094871937 |
| label_smoothing | 2 | 50 | 0 | 0.47 | 0.4134 | 0.5236 | 0.3327777777777778 | 0.20144455006117878 | 0.024118672832846633 |
