# DynaSent pilot results

- Round: `round2`
- Backend: `tfidf`
- Budget: `50`
- Evaluation rows: `720`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 50 | 0 | 1.26 | 0.4268 | 0.5496 | 0.35555555555555557 | 0.18821589866047272 | 0.02511885128915308 |
| human_soft_targets | 0 | 50 | 0 | 0.02 | 0.4247 | 0.5461 | 0.3625 | 0.19929564364574304 | 0.03958799863855043 |
| label_smoothing | 0 | 50 | 0 | 0.02 | 0.4270 | 0.5496 | 0.35694444444444445 | 0.19030712876866723 | 0.023965132526225494 |
| majority_ce | 1 | 50 | 0 | 0.02 | 0.4308 | 0.5669 | 0.3458333333333333 | 0.14770877047727382 | 0.08399919362531767 |
| human_soft_targets | 1 | 50 | 0 | 0.02 | 0.4375 | 0.5768 | 0.34305555555555556 | 0.14377256079110268 | 0.0838472363849481 |
| label_smoothing | 1 | 50 | 0 | 0.02 | 0.4306 | 0.5657 | 0.3472222222222222 | 0.14973974515738903 | 0.07753970097336504 |
| majority_ce | 2 | 50 | 0 | 0.02 | 0.4123 | 0.5285 | 0.3680555555555556 | 0.2176413068673132 | 0.018915492544571554 |
| human_soft_targets | 2 | 50 | 0 | 0.02 | 0.4108 | 0.5248 | 0.3763888888888889 | 0.24068999367920296 | 0.028635319280955533 |
| label_smoothing | 2 | 50 | 0 | 0.02 | 0.4124 | 0.5284 | 0.3680555555555556 | 0.2181661662743206 | 0.018094848386115494 |
