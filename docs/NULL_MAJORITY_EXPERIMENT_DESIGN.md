# Null-majority soft-target experiment: frozen comparison plan

## Research question

Does access to valid human vote distributions on examples without a non-null majority label change sentiment-classification performance when the total training budget is held fixed?

## Primary controlled comparison: matched data

Use one sampled training set of `B` examples with non-null majority labels for all three objectives:

1. `majority_ce`: majority-label cross-entropy.
2. `human_soft_targets`: cross-entropy against the normalized four-class human vote distribution.
3. `label_smoothing`: majority-label cross-entropy with generic label smoothing.

The three conditions use the same train text IDs, test examples, seed, features/model architecture, epoch count, and optimization settings. This is the primary objective comparison.

## Secondary data-availability comparison: null-augmented soft targets

Enable `--include-null-soft-only` to add `human_soft_targets_null_augmented`. Its total training budget remains `B`, with a target share of null-majority examples set by `--null-soft-fraction` (default 0.5). Remaining slots are filled with non-null-majority examples. Selection is deterministic for a given seed, and the runner records the number of null/non-null rows plus `training_manifest.jsonl`.

This condition deliberately uses a different training set from the matched baseline. Therefore, differences cannot be attributed solely to the objective: they reflect both access to additional null-majority examples and the changed sample composition. Report it as a secondary data-availability experiment, not as a fourth matched objective. For a stronger follow-up, repeat the comparison over multiple fixed seeds and report results separately by round.

## Target and evaluation policy

- Preserve four-class order: `negative`, `neutral`, `positive`, `mixed`.
- Never invent hard labels for null-majority examples.
- Evaluate Jensen–Shannon divergence and distributional Brier score against human vote distributions for every test row with valid votes.
- Evaluate accuracy, macro-F1, and ECE only on rows with non-null majority gold labels.
- Keep Round 1 and Round 2 separate in the initial analysis.
- Use the development split for any tuning; freeze settings before final test analysis.
- Final evaluation should use the full held-out test split (`--eval-size 0`) and multiple predeclared seeds. Report uncertainty across seeds and avoid interpreting the single-seed pilot as conclusive.

## Reproduction commands

Matched pilot (default):

```bash
python src/run_dynasent_experiments.py \
  --data data/dynasent-main.zip --round round1 --backend tfidf \
  --budget 100 --eval-size 500 --epochs 10 --seeds 0 \
  --out-dir results/dynasent_matched_round1
```

Matched plus null-augmented condition:

```bash
python src/run_dynasent_experiments.py \
  --data data/dynasent-main.zip --round round1 --backend tfidf \
  --budget 100 --eval-size 500 --epochs 10 --seeds 0 \
  --include-null-soft-only --null-soft-fraction 0.5 \
  --out-dir results/dynasent_null_augmented_round1
```

Outputs include metrics, per-example predictions, configuration/audit metadata, a Markdown summary, and the sampled training manifest.
