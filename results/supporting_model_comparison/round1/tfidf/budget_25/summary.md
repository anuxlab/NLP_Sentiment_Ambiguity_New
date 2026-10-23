# DynaSent pilot results

- Round: `round1`
- Backend: `tfidf`
- Budget: `25`
- Evaluation rows: `3600`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 25 | 0 | 1.53 | 0.4110 | 0.5241 | 0.34555555555555556 | 0.22318295968140767 | 0.02202557408147387 |
| human_soft_targets | 0 | 25 | 0 | 0.08 | 0.4105 | 0.5227 | 0.3227777777777778 | 0.20829232127882125 | 0.007609726670715533 |
| label_smoothing | 0 | 25 | 0 | 0.07 | 0.4110 | 0.5241 | 0.3458333333333333 | 0.2232915888683722 | 0.024376410875055533 |
| majority_ce | 1 | 25 | 0 | 0.06 | 0.4176 | 0.5333 | 0.32 | 0.1544897950774928 | 0.025637356324328307 |
| human_soft_targets | 1 | 25 | 0 | 0.07 | 0.4145 | 0.5273 | 0.3258333333333333 | 0.19970585245283046 | 0.012602946029769049 |
| label_smoothing | 1 | 25 | 0 | 0.07 | 0.4176 | 0.5333 | 0.3194444444444444 | 0.15459935740263833 | 0.02684560754232936 |
| majority_ce | 2 | 25 | 0 | 0.07 | 0.4156 | 0.5299 | 0.33055555555555555 | 0.21939228686464476 | 0.01573668585883249 |
| human_soft_targets | 2 | 25 | 0 | 0.07 | 0.4128 | 0.5251 | 0.3566666666666667 | 0.2255976400788116 | 0.037272196362415945 |
| label_smoothing | 2 | 25 | 0 | 0.07 | 0.4157 | 0.5299 | 0.3288888888888889 | 0.22196054560306339 | 0.014323686204022837 |
