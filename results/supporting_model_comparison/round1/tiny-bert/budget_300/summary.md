# DynaSent pilot results

- Round: `round1`
- Backend: `tiny-bert`
- Budget: `300`
- Evaluation rows: `3600`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 300 | 0 | 2.22 | 0.3761 | 0.5239 | 0.3327777777777778 | 0.12600373240423837 | 0.17761428140931657 |
| human_soft_targets | 0 | 300 | 0 | 2.15 | 0.3710 | 0.4981 | 0.33416666666666667 | 0.14389045512752477 | 0.11495190721419123 |
| label_smoothing | 0 | 300 | 0 | 2.15 | 0.3752 | 0.5139 | 0.33166666666666667 | 0.1264459117566462 | 0.15603790220287111 |
| majority_ce | 1 | 300 | 0 | 2.26 | 0.3788 | 0.5403 | 0.33194444444444443 | 0.1254073715642241 | 0.225474745730559 |
| human_soft_targets | 1 | 300 | 0 | 2.36 | 0.3702 | 0.5028 | 0.3338888888888889 | 0.13054362560652705 | 0.15100871151520148 |
| label_smoothing | 1 | 300 | 0 | 2.48 | 0.3766 | 0.5248 | 0.3325 | 0.1263806392197341 | 0.1963904155211316 |
| majority_ce | 2 | 300 | 0 | 3.15 | 0.3770 | 0.5189 | 0.3338888888888889 | 0.1272093832990298 | 0.1693311065187057 |
| human_soft_targets | 2 | 300 | 0 | 3.27 | 0.3713 | 0.4931 | 0.3411111111111111 | 0.14922888728069053 | 0.09481580891542965 |
| label_smoothing | 2 | 300 | 0 | 2.41 | 0.3768 | 0.5106 | 0.3363888888888889 | 0.13124961317707615 | 0.14952150937583714 |
