# DynaSent pilot results

- Round: `round2`
- Backend: `tiny-bert`
- Budget: `300`
- Evaluation rows: `720`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 300 | 0 | 2.71 | 0.4309 | 0.5618 | 0.33194444444444443 | 0.15763071719025515 | 0.047840591271718365 |
| human_soft_targets | 0 | 300 | 0 | 2.46 | 0.4273 | 0.5553 | 0.3347222222222222 | 0.16946622374947568 | 0.03993783998820519 |
| label_smoothing | 0 | 300 | 0 | 2.53 | 0.4309 | 0.5602 | 0.33055555555555555 | 0.16155847408108182 | 0.038598818952838586 |
| majority_ce | 1 | 300 | 0 | 2.84 | 0.4371 | 0.5780 | 0.33611111111111114 | 0.1320695641659037 | 0.09370236164993709 |
| human_soft_targets | 1 | 300 | 0 | 2.92 | 0.4309 | 0.5661 | 0.3333333333333333 | 0.1312295567517449 | 0.0769229017612007 |
| label_smoothing | 1 | 300 | 0 | 2.26 | 0.4355 | 0.5730 | 0.33611111111111114 | 0.13217730130026692 | 0.07831798026131259 |
| majority_ce | 2 | 300 | 0 | 2.11 | 0.4319 | 0.5620 | 0.31805555555555554 | 0.1452635486265776 | 0.05419430078731643 |
| human_soft_targets | 2 | 300 | 0 | 2.59 | 0.4267 | 0.5546 | 0.33055555555555555 | 0.1417531936590751 | 0.05090137595931687 |
| label_smoothing | 2 | 300 | 0 | 2.27 | 0.4312 | 0.5594 | 0.3125 | 0.14773270863165872 | 0.0492739071034723 |
