# DynaSent pilot results

- Round: `round2`
- Backend: `tfidf`
- Budget: `300`
- Evaluation rows: `720`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 300 | 0 | 1.31 | 0.4229 | 0.5548 | 0.3402777777777778 | 0.2055767542054901 | 0.08210878541900052 |
| human_soft_targets | 0 | 300 | 0 | 0.07 | 0.4214 | 0.5506 | 0.34305555555555556 | 0.21047423043433444 | 0.06446790053612655 |
| label_smoothing | 0 | 300 | 0 | 0.08 | 0.4222 | 0.5514 | 0.35 | 0.2209045512165376 | 0.06213026572432782 |
| majority_ce | 1 | 300 | 0 | 0.07 | 0.4286 | 0.5704 | 0.3416666666666667 | 0.18189985510198592 | 0.11059181582596567 |
| human_soft_targets | 1 | 300 | 0 | 0.07 | 0.4186 | 0.5493 | 0.3402777777777778 | 0.1910954215072424 | 0.08666886281636027 |
| label_smoothing | 1 | 300 | 0 | 0.07 | 0.4257 | 0.5619 | 0.3333333333333333 | 0.1833656867101683 | 0.10475681407584081 |
| majority_ce | 2 | 300 | 0 | 0.06 | 0.4188 | 0.5455 | 0.33194444444444443 | 0.21430141563351798 | 0.07311256656216251 |
| human_soft_targets | 2 | 300 | 0 | 0.07 | 0.4108 | 0.5321 | 0.3486111111111111 | 0.22030820661866873 | 0.05746746713088617 |
| label_smoothing | 2 | 300 | 0 | 0.07 | 0.4178 | 0.5423 | 0.33194444444444443 | 0.22047576279426184 | 0.05659181144502429 |
