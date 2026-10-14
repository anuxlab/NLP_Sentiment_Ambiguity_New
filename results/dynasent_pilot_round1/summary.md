# DynaSent pilot results

- Round: `round1`
- Backend: `tfidf`
- Budget: `100`
- Evaluation rows: `500`
- Seeds: `[0]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 0.57 | 0.3819 | 0.5027 | 0.346 | 0.1568866419909121 | 0.08535235738754271 |
| human_soft_targets | 0 | 0.03 | 0.3805 | 0.4945 | 0.362 | 0.186890756302521 | 0.044677164793014514 |
| label_smoothing | 0 | 0.03 | 0.3823 | 0.5018 | 0.348 | 0.15931758530183726 | 0.07856691193580627 |
