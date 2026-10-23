# DynaSent pilot results

- Round: `round1`
- Backend: `tfidf`
- Budget: `300`
- Evaluation rows: `3600`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 300 | 0 | 2.40 | 0.3683 | 0.5141 | 0.38222222222222224 | 0.21152818436165083 | 0.1354295268323686 |
| human_soft_targets | 0 | 300 | 0 | 0.14 | 0.3640 | 0.4865 | 0.4066666666666667 | 0.24456189930054428 | 0.05666482513977423 |
| label_smoothing | 0 | 300 | 0 | 0.12 | 0.3673 | 0.5039 | 0.3861111111111111 | 0.21866746048928343 | 0.1108735611372524 |
| majority_ce | 1 | 300 | 0 | 0.13 | 0.3748 | 0.5330 | 0.375 | 0.19722784340349592 | 0.17023339074518945 |
| human_soft_targets | 1 | 300 | 0 | 0.12 | 0.3666 | 0.4965 | 0.38805555555555554 | 0.21925202319330228 | 0.09570827005638016 |
| label_smoothing | 1 | 300 | 0 | 0.12 | 0.3732 | 0.5193 | 0.3775 | 0.2031646894865265 | 0.14242271858784886 |
| majority_ce | 2 | 300 | 0 | 0.12 | 0.3689 | 0.5070 | 0.4 | 0.2328746932748122 | 0.10330484352178045 |
| human_soft_targets | 2 | 300 | 0 | 0.12 | 0.3643 | 0.4818 | 0.4161111111111111 | 0.2606604649610965 | 0.04075449239048693 |
| label_smoothing | 2 | 300 | 0 | 0.12 | 0.3686 | 0.4986 | 0.40305555555555556 | 0.2384836087683515 | 0.08468983446558316 |
