# DynaSent pilot results

- Round: `round2`
- Backend: `tfidf`
- Budget: `25`
- Evaluation rows: `720`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 25 | 0 | 1.41 | 0.4254 | 0.5474 | 0.36666666666666664 | 0.2317447077497928 | 0.025866866774029208 |
| human_soft_targets | 0 | 25 | 0 | 0.02 | 0.4252 | 0.5467 | 0.3611111111111111 | 0.22352274812294748 | 0.015823988409505954 |
| label_smoothing | 0 | 25 | 0 | 0.02 | 0.4255 | 0.5475 | 0.36527777777777776 | 0.23138271413136358 | 0.0249756809324026 |
| majority_ce | 1 | 25 | 0 | 0.02 | 0.4433 | 0.5755 | 0.31805555555555554 | 0.15647773231186718 | 0.03849686914020115 |
| human_soft_targets | 1 | 25 | 0 | 0.02 | 0.4440 | 0.5765 | 0.31527777777777777 | 0.15659256140039796 | 0.03970078031222024 |
| label_smoothing | 1 | 25 | 0 | 0.02 | 0.4433 | 0.5755 | 0.31805555555555554 | 0.15653317003276296 | 0.038338626051942534 |
| majority_ce | 2 | 25 | 0 | 0.02 | 0.4302 | 0.5523 | 0.34444444444444444 | 0.22209587139411324 | 0.030520480622847873 |
| human_soft_targets | 2 | 25 | 0 | 0.02 | 0.4296 | 0.5517 | 0.34305555555555556 | 0.22734525422572172 | 0.02362565133306715 |
| label_smoothing | 2 | 25 | 0 | 0.02 | 0.4304 | 0.5526 | 0.3458333333333333 | 0.22269039527686318 | 0.032321844332748 |
