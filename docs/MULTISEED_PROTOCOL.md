# DynaSent multi-seed evaluation protocol

## Confirmatory configuration

- Dataset: official DynaSent archive in `data/dynasent-main.zip`.
- Rounds: Round 1 and Round 2 reported separately.
- Backend: TF-IDF + linear classifier (CPU-friendly baseline).
- Primary metric: mean Jensen–Shannon divergence between predicted distributions and annotator vote distributions (lower is better).
- Secondary metrics: macro-F1, accuracy, soft Brier score, 10-bin ECE, training time.
- Conditions: majority-label CE, human-distribution soft targets, generic label smoothing, and the separately labelled null-augmented soft-target condition.
- Default seeds: 0–4. Training budget: 100 examples per condition/seed. Epochs: 10. Evaluation size: 0 (full test split).
- Null-augmented condition: fixed total budget; default target 50% null-majority examples. This changes training composition and is not a matched-data comparison to the other three conditions.
- Disagreement tiers are computed from human vote distributions, not model confidence: unanimous (vote entropy = 0) versus non-unanimous (vote entropy > 0). The supplied test splits contain only these two observed vote-distribution types; do not force empty intermediate tiers.

## Analysis rules

1. Do not tune hyperparameters or the null fraction against the final test metrics.
2. Keep Round 1 and Round 2 separate in all primary tables.
3. Report per-seed values, means, and sample standard deviations. Include paired per-seed differences against majority CE.
4. Treat five seeds as a reproducibility check, not strong statistical proof. For inference, bootstrap examples with paired predictions and consider seed variability; disclose both sources of uncertainty.
5. JS divergence and soft Brier score use all test examples with valid annotator distributions, including rows with null majority labels. Accuracy and macro-F1 use only rows with a valid majority gold label.
6. Check `run_completeness.json` and the manifests before interpreting results. A successful upload is not equivalent to a complete or scientifically valid experiment.

## Workflow

Manually dispatch **DynaSent Multi-Seed Evaluation** from GitHub Actions. Defaults run both rounds, seeds `0 1 2 3 4`, full test splits, TF-IDF, budget 100, 10 epochs, and the null-augmented condition. The workflow is manual-only to prevent unplanned compute.
