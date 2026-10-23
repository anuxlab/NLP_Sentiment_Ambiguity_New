# DynaSent pilot results

- Round: `round2`
- Backend: `tiny-bert`
- Budget: `100`
- Evaluation rows: `720`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 100 | 0 | 0.88 | 0.4276 | 0.5503 | 0.3416666666666667 | 0.17307488758563933 | 0.007669533292452511 |
| human_soft_targets | 0 | 100 | 0 | 0.80 | 0.4268 | 0.5490 | 0.3472222222222222 | 0.18761398605867546 | 0.014908854911724735 |
| label_smoothing | 0 | 100 | 0 | 0.98 | 0.4279 | 0.5506 | 0.3402777777777778 | 0.17249813923737808 | 0.006658209984501179 |
| majority_ce | 1 | 100 | 0 | 0.87 | 0.4396 | 0.5663 | 0.2513888888888889 | 0.20422351231127187 | 0.045277305692434315 |
| human_soft_targets | 1 | 100 | 0 | 0.73 | 0.4361 | 0.5609 | 0.28055555555555556 | 0.21586948461253244 | 0.02401241639422046 |
| label_smoothing | 1 | 100 | 0 | 0.74 | 0.4398 | 0.5666 | 0.2569444444444444 | 0.2072796761231457 | 0.0399267603539758 |
| majority_ce | 2 | 100 | 0 | 0.80 | 0.4445 | 0.5763 | 0.3125 | 0.17667224589166158 | 0.03756774870885744 |
| human_soft_targets | 2 | 100 | 0 | 0.78 | 0.4360 | 0.5644 | 0.31666666666666665 | 0.16187391776180382 | 0.04927576494713626 |
| label_smoothing | 2 | 100 | 0 | 0.82 | 0.4436 | 0.5749 | 0.30972222222222223 | 0.17406049552163647 | 0.0392618423120843 |
