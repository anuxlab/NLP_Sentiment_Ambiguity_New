# DynaSent pilot results

- Round: `round1`
- Backend: `tiny-bert`
- Budget: `25`
- Evaluation rows: `3600`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 25 | 0 | 0.28 | 0.4208 | 0.5411 | 0.33055555555555555 | 0.12928096440507444 | 0.036226269627610844 |
| human_soft_targets | 0 | 25 | 0 | 0.22 | 0.4202 | 0.5386 | 0.32083333333333336 | 0.13502389634408407 | 0.031157915277613513 |
| label_smoothing | 0 | 25 | 0 | 0.21 | 0.4206 | 0.5405 | 0.33 | 0.12914096631799424 | 0.03547106507751679 |
| majority_ce | 1 | 25 | 0 | 0.23 | 0.4236 | 0.5507 | 0.3327777777777778 | 0.1293858090449527 | 0.08692436816791692 |
| human_soft_targets | 1 | 25 | 0 | 0.27 | 0.4203 | 0.5434 | 0.3333333333333333 | 0.13098378211628736 | 0.07210595688058269 |
| label_smoothing | 1 | 25 | 0 | 0.23 | 0.4234 | 0.5502 | 0.3327777777777778 | 0.1293858090449527 | 0.08596816414760217 |
| majority_ce | 2 | 25 | 0 | 0.38 | 0.4256 | 0.5411 | 0.2927777777777778 | 0.19203341176579572 | 0.015486825969484095 |
| human_soft_targets | 2 | 25 | 0 | 0.24 | 0.4254 | 0.5405 | 0.2825 | 0.19719555443011305 | 0.013820477161142573 |
| label_smoothing | 2 | 25 | 0 | 0.25 | 0.4256 | 0.5411 | 0.2916666666666667 | 0.19239349664177152 | 0.016343334590395314 |
