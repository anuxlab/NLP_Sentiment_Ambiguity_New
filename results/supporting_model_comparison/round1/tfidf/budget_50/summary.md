# DynaSent pilot results

- Round: `round1`
- Backend: `tfidf`
- Budget: `50`
- Evaluation rows: `3600`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 50 | 0 | 1.30 | 0.3972 | 0.5062 | 0.36527777777777776 | 0.22387144617801602 | 0.023272215806775607 |
| human_soft_targets | 0 | 50 | 0 | 0.09 | 0.3984 | 0.5080 | 0.37083333333333335 | 0.22537985768880087 | 0.036523460108372915 |
| label_smoothing | 0 | 50 | 0 | 0.18 | 0.3974 | 0.5063 | 0.36527777777777776 | 0.22416260776336508 | 0.023983260807063863 |
| majority_ce | 1 | 50 | 0 | 0.10 | 0.4041 | 0.5251 | 0.33805555555555555 | 0.14035692288447804 | 0.06897843212717109 |
| human_soft_targets | 1 | 50 | 0 | 0.10 | 0.3973 | 0.5113 | 0.3566666666666667 | 0.18006514579723704 | 0.038997134856051885 |
| label_smoothing | 1 | 50 | 0 | 0.07 | 0.4040 | 0.5246 | 0.33805555555555555 | 0.14038193964877818 | 0.06693621057603094 |
| majority_ce | 2 | 50 | 0 | 0.07 | 0.4032 | 0.5152 | 0.37277777777777776 | 0.21542686041792453 | 0.05309945755534699 |
| human_soft_targets | 2 | 50 | 0 | 0.07 | 0.4073 | 0.5200 | 0.37916666666666665 | 0.22648162072980843 | 0.057458624458975274 |
| label_smoothing | 2 | 50 | 0 | 0.07 | 0.4034 | 0.5151 | 0.37305555555555553 | 0.21572330593561603 | 0.050743086453941134 |
