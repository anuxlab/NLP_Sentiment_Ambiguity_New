# DynaSent initial CPU pilot comparison

**Exploratory only:** one seed, 100 training examples per objective, 10 epochs, 500 fixed test examples per round. These results are for pipeline validation, not a reliable claim of objective superiority.

| Round | Objective | Accuracy | Macro-F1 | Mean JS (bits) ↓ | Soft Brier ↓ | ECE ↓ |
|---|---|---:|---:|---:|---:|---:|
| round1 | majority_ce | 0.346 | 0.157 | 0.3819 | 0.5027 | 0.0854 |
| round1 | human_soft_targets | 0.362 | 0.187 | 0.3805 | 0.4945 | 0.0447 |
| round1 | label_smoothing | 0.348 | 0.159 | 0.3823 | 0.5018 | 0.0786 |
| round2 | majority_ce | 0.342 | 0.183 | 0.4176 | 0.5447 | 0.0710 |
| round2 | human_soft_targets | 0.336 | 0.180 | 0.4195 | 0.5466 | 0.0623 |
| round2 | label_smoothing | 0.344 | 0.186 | 0.4174 | 0.5432 | 0.0681 |

## Interpretation guardrails

- The backend is TF-IDF + a linear four-class classifier, not tiny BERT.
- The matched conditions use the same sampled 100 training examples with non-null majority labels.
- The evaluation set is a deterministic 500-row sample from each round test split; it is not the full test split.
- Round 1 and Round 2 are reported separately.
- One seed is insufficient to infer a stable improvement. Use multiple seeds, predeclared budgets, development-based selection, and the full held-out test set for final analysis.
