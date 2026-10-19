# DynaSent pilot results

- Round: `round2`
- Backend: `tfidf`
- Budget: `100`
- Evaluation rows: `500`
- Seeds: `[0]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 0.53 | 0.4176 | 0.5447 | 0.342 | 0.1832547487538731 | 0.07099276089668276 |
| human_soft_targets | 0 | 0.03 | 0.4195 | 0.5466 | 0.336 | 0.17964926314076926 | 0.06232608395814897 |
| label_smoothing | 0 | 0.03 | 0.4174 | 0.5432 | 0.344 | 0.1861176565093986 | 0.06805308896303176 |
