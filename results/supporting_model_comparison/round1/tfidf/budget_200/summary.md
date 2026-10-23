# DynaSent pilot results

- Round: `round1`
- Backend: `tfidf`
- Budget: `200`
- Evaluation rows: `3600`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 200 | 0 | 1.53 | 0.3759 | 0.5149 | 0.365 | 0.18790930474711556 | 0.12403145670890808 |
| human_soft_targets | 0 | 200 | 0 | 0.11 | 0.3722 | 0.4906 | 0.3838888888888889 | 0.22069558739981 | 0.04722510625090864 |
| label_smoothing | 0 | 200 | 0 | 0.12 | 0.3751 | 0.5075 | 0.36527777777777776 | 0.1915403058418981 | 0.10753998312685223 |
| majority_ce | 1 | 200 | 0 | 0.11 | 0.3754 | 0.5175 | 0.36138888888888887 | 0.17950632020419582 | 0.1411589521666368 |
| human_soft_targets | 1 | 200 | 0 | 0.10 | 0.3704 | 0.4929 | 0.37722222222222224 | 0.20644585626166628 | 0.07351516376766894 |
| label_smoothing | 1 | 200 | 0 | 0.66 | 0.3746 | 0.5093 | 0.36666666666666664 | 0.1895831741872944 | 0.11785790781180064 |
| majority_ce | 2 | 200 | 0 | 0.15 | 0.3782 | 0.5069 | 0.37166666666666665 | 0.20614632102915678 | 0.08461567650238672 |
| human_soft_targets | 2 | 200 | 0 | 0.11 | 0.3763 | 0.4916 | 0.3861111111111111 | 0.24053194262419939 | 0.03097364135086536 |
| label_smoothing | 2 | 200 | 0 | 0.11 | 0.3786 | 0.5026 | 0.37305555555555553 | 0.21386811506297748 | 0.07017342620425754 |
