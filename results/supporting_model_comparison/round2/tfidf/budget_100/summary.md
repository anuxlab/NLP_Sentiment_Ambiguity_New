# DynaSent pilot results

- Round: `round2`
- Backend: `tfidf`
- Budget: `100`
- Evaluation rows: `720`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 100 | 0 | 1.40 | 0.4174 | 0.5441 | 0.35 | 0.1883893773225777 | 0.05556651308304736 |
| human_soft_targets | 0 | 100 | 0 | 0.03 | 0.4190 | 0.5454 | 0.3472222222222222 | 0.18726813391902042 | 0.05171575132343503 |
| label_smoothing | 0 | 100 | 0 | 0.04 | 0.4172 | 0.5427 | 0.3527777777777778 | 0.19165411501177926 | 0.04928305554721092 |
| majority_ce | 1 | 100 | 0 | 0.03 | 0.4522 | 0.6011 | 0.325 | 0.12760490880592762 | 0.10075624071889452 |
| human_soft_targets | 1 | 100 | 0 | 0.04 | 0.4438 | 0.5852 | 0.3277777777777778 | 0.1299048811081581 | 0.0846972803688712 |
| label_smoothing | 1 | 100 | 0 | 0.04 | 0.4508 | 0.5975 | 0.32083333333333336 | 0.1266513293117362 | 0.09462522711190913 |
| majority_ce | 2 | 100 | 0 | 0.03 | 0.4359 | 0.5673 | 0.3541666666666667 | 0.1999680306905371 | 0.038667116148604276 |
| human_soft_targets | 2 | 100 | 0 | 0.03 | 0.4254 | 0.5523 | 0.3472222222222222 | 0.18259053298946915 | 0.04586326703429223 |
| label_smoothing | 2 | 100 | 0 | 0.04 | 0.4355 | 0.5661 | 0.35138888888888886 | 0.19948050240021042 | 0.036192720797326834 |
