# DynaSent pilot results

- Round: `round2`
- Backend: `tiny-bert`
- Budget: `25`
- Evaluation rows: `720`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 25 | 0 | 0.25 | 0.4409 | 0.5695 | 0.3138888888888889 | 0.1865241182794499 | 0.023310227526558765 |
| human_soft_targets | 0 | 25 | 0 | 0.20 | 0.4407 | 0.5692 | 0.3125 | 0.18857791834387577 | 0.010848513286974702 |
| label_smoothing | 0 | 25 | 0 | 0.19 | 0.4410 | 0.5696 | 0.3055555555555556 | 0.18373311459903638 | 0.010171571580900086 |
| majority_ce | 1 | 25 | 0 | 0.20 | 0.4420 | 0.5725 | 0.33194444444444443 | 0.19733843537414966 | 0.024370417950881858 |
| human_soft_targets | 1 | 25 | 0 | 0.20 | 0.4434 | 0.5746 | 0.3277777777777778 | 0.20118942026182585 | 0.023601039333475984 |
| label_smoothing | 1 | 25 | 0 | 0.20 | 0.4421 | 0.5727 | 0.3333333333333333 | 0.1985601391616327 | 0.02323591945072016 |
| majority_ce | 2 | 25 | 0 | 0.20 | 0.4431 | 0.5725 | 0.3236111111111111 | 0.13394705708370305 | 0.005806019860837215 |
| human_soft_targets | 2 | 25 | 0 | 0.20 | 0.4403 | 0.5680 | 0.3277777777777778 | 0.1398336783988958 | 0.006050042601095284 |
| label_smoothing | 2 | 25 | 0 | 0.20 | 0.4429 | 0.5720 | 0.32083333333333336 | 0.13333988216573928 | 0.005996675044298188 |
