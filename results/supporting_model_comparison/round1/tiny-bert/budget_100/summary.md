# DynaSent pilot results

- Round: `round1`
- Backend: `tiny-bert`
- Budget: `100`
- Evaluation rows: `3600`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 100 | 0 | 1.12 | 0.3989 | 0.5280 | 0.3333333333333333 | 0.1253638683339356 | 0.11317495323717594 |
| human_soft_targets | 0 | 100 | 0 | 1.71 | 0.3946 | 0.5145 | 0.3338888888888889 | 0.13149524087024086 | 0.073192561533716 |
| label_smoothing | 0 | 100 | 0 | 1.01 | 0.3981 | 0.5245 | 0.3333333333333333 | 0.1253638683339356 | 0.10362263076835211 |
| majority_ce | 1 | 100 | 0 | 1.74 | 0.3990 | 0.5427 | 0.3327777777777778 | 0.12595575416969507 | 0.17740460229416688 |
| human_soft_targets | 1 | 100 | 0 | 1.57 | 0.3920 | 0.5169 | 0.3338888888888889 | 0.1277334663481397 | 0.12311065847675003 |
| label_smoothing | 1 | 100 | 0 | 1.36 | 0.3974 | 0.5360 | 0.3327777777777778 | 0.12595678464284712 | 0.16403579618367886 |
| majority_ce | 2 | 100 | 0 | 1.20 | 0.3979 | 0.5079 | 0.3375 | 0.14343874761922204 | 0.03362451834811106 |
| human_soft_targets | 2 | 100 | 0 | 0.88 | 0.3971 | 0.5058 | 0.3363888888888889 | 0.14514064471403218 | 0.030148213687870237 |
| label_smoothing | 2 | 100 | 0 | 0.90 | 0.3983 | 0.5080 | 0.33805555555555555 | 0.145016247373845 | 0.03005634004871052 |
