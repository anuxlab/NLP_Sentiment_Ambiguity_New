# DynaSent pilot results

- Round: `round2`
- Backend: `tiny-bert`
- Budget: `200`
- Evaluation rows: `720`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 200 | 0 | 1.62 | 0.4280 | 0.5532 | 0.3375 | 0.16114617630257394 | 0.01751478368209469 |
| human_soft_targets | 0 | 200 | 0 | 1.58 | 0.4248 | 0.5488 | 0.3347222222222222 | 0.16055788199538507 | 0.02031693218482867 |
| label_smoothing | 0 | 200 | 0 | 2.44 | 0.4284 | 0.5531 | 0.3347222222222222 | 0.160102617106229 | 0.02382803339925075 |
| majority_ce | 1 | 200 | 0 | 1.89 | 0.4332 | 0.5662 | 0.3375 | 0.13401918917419997 | 0.060736862983968534 |
| human_soft_targets | 1 | 200 | 0 | 2.88 | 0.4323 | 0.5655 | 0.33611111111111114 | 0.13369210739473203 | 0.06239719208743838 |
| label_smoothing | 1 | 200 | 0 | 1.95 | 0.4335 | 0.5658 | 0.3375 | 0.13401918917419997 | 0.05728894049922627 |
| majority_ce | 2 | 200 | 0 | 2.68 | 0.4341 | 0.5665 | 0.3125 | 0.1497159090909091 | 0.07119531826012662 |
| human_soft_targets | 2 | 200 | 0 | 2.83 | 0.4259 | 0.5556 | 0.3236111111111111 | 0.14432452648732402 | 0.0653294110049804 |
| label_smoothing | 2 | 200 | 0 | 1.82 | 0.4332 | 0.5640 | 0.3125 | 0.151798374303323 | 0.06336223723159894 |
