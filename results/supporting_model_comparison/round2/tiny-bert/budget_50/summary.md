# DynaSent pilot results

- Round: `round2`
- Backend: `tiny-bert`
- Budget: `50`
- Evaluation rows: `720`
- Seeds: `[0, 1, 2]`

> Pilot results are exploratory, not final research claims. The test subset is fixed by seed 20261014; use the full test split and multiple seeds for final evaluation.

## Metrics

| Condition | Seed | Train rows | Null-gold train rows | Train seconds | JS divergence ↓ | Soft Brier ↓ | Accuracy | Macro-F1 | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| majority_ce | 0 | 50 | 0 | 0.59 | 0.4381 | 0.5644 | 0.30972222222222223 | 0.19018848879541173 | 0.026219176045722416 |
| human_soft_targets | 0 | 50 | 0 | 0.36 | 0.4354 | 0.5601 | 0.3263888888888889 | 0.21250902354026127 | 0.04052480889691246 |
| label_smoothing | 0 | 50 | 0 | 0.36 | 0.4382 | 0.5645 | 0.3111111111111111 | 0.19027236880653448 | 0.024623960587713465 |
| majority_ce | 1 | 50 | 0 | 0.37 | 0.4362 | 0.5622 | 0.32083333333333336 | 0.19021321321321322 | 0.016697790142562646 |
| human_soft_targets | 1 | 50 | 0 | 0.37 | 0.4395 | 0.5671 | 0.29305555555555557 | 0.19496384331576866 | 0.008795094490051253 |
| label_smoothing | 1 | 50 | 0 | 0.37 | 0.4365 | 0.5624 | 0.32222222222222224 | 0.19345503200755587 | 0.017639647258652576 |
| majority_ce | 2 | 50 | 0 | 0.39 | 0.4358 | 0.5609 | 0.33055555555555555 | 0.1757333512914482 | 0.01730929530329176 |
| human_soft_targets | 2 | 50 | 0 | 0.53 | 0.4325 | 0.5555 | 0.33055555555555555 | 0.1857365063175047 | 0.023121212174495063 |
| label_smoothing | 2 | 50 | 0 | 0.40 | 0.4356 | 0.5604 | 0.32916666666666666 | 0.1766952115231573 | 0.02311184513900013 |
