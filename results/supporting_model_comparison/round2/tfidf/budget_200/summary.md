# DynaSent pilot results

- Round: `round2`
- Backend: `tfidf`
- Budget: `200`
- Evaluation rows: `720`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 200 | 0 | 1.30 | 0.4210 | 0.5461 | 0.35555555555555557 | 0.2182426501283833 | 0.03734481148421762 |
| human_soft_targets | 0 | 200 | 0 | 0.05 | 0.4210 | 0.5465 | 0.36944444444444446 | 0.2266377536146754 | 0.030924841885765397 |
| label_smoothing | 0 | 200 | 0 | 0.11 | 0.4210 | 0.5445 | 0.36527777777777776 | 0.2331303554427166 | 0.014336415173278922 |
| majority_ce | 1 | 200 | 0 | 0.05 | 0.4292 | 0.5752 | 0.33611111111111114 | 0.15075159065768107 | 0.1295751193745269 |
| human_soft_targets | 1 | 200 | 0 | 0.05 | 0.4227 | 0.5569 | 0.3388888888888889 | 0.16048117842035717 | 0.09377845612665016 |
| label_smoothing | 1 | 200 | 0 | 0.05 | 0.4271 | 0.5665 | 0.3375 | 0.15610444372007382 | 0.10600600507524277 |
| majority_ce | 2 | 200 | 0 | 0.05 | 0.4239 | 0.5560 | 0.3402777777777778 | 0.18556835398951718 | 0.08104705636699994 |
| human_soft_targets | 2 | 200 | 0 | 0.05 | 0.4136 | 0.5392 | 0.3333333333333333 | 0.17562987548588838 | 0.07793994636999235 |
| label_smoothing | 2 | 200 | 0 | 0.05 | 0.4232 | 0.5523 | 0.3388888888888889 | 0.18861377563598417 | 0.069144991123014 |
