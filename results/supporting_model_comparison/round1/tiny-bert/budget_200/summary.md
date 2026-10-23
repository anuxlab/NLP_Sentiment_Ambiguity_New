# DynaSent pilot results

- Round: `round1`
- Backend: `tiny-bert`
- Budget: `200`
- Evaluation rows: `3600`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 200 | 0 | 1.49 | 0.3825 | 0.5230 | 0.3333333333333333 | 0.1253638683339356 | 0.15611617301901182 |
| human_soft_targets | 0 | 200 | 0 | 1.48 | 0.3784 | 0.5016 | 0.32944444444444443 | 0.1344878458519207 | 0.09864942479464743 |
| label_smoothing | 0 | 200 | 0 | 1.69 | 0.3814 | 0.5151 | 0.33361111111111114 | 0.12583056478405316 | 0.13685417138040068 |
| majority_ce | 1 | 200 | 0 | 1.80 | 0.3798 | 0.5235 | 0.3322222222222222 | 0.12515226280646663 | 0.18495804033345645 |
| human_soft_targets | 1 | 200 | 0 | 1.90 | 0.3746 | 0.4994 | 0.33361111111111114 | 0.128976484212987 | 0.1283856039494276 |
| label_smoothing | 1 | 200 | 0 | 2.32 | 0.3782 | 0.5131 | 0.3322222222222222 | 0.12515226280646663 | 0.16226847077409426 |
| majority_ce | 2 | 200 | 0 | 2.20 | 0.3836 | 0.5087 | 0.3347222222222222 | 0.12848004092163018 | 0.11780459112591214 |
| human_soft_targets | 2 | 200 | 0 | 2.54 | 0.3801 | 0.4949 | 0.34 | 0.1536118100162462 | 0.07395185046725802 |
| label_smoothing | 2 | 200 | 0 | 3.08 | 0.3841 | 0.5059 | 0.3347222222222222 | 0.13004775298023613 | 0.10573930794166195 |
