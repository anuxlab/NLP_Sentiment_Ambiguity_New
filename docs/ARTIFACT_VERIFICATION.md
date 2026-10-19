# DynaSent artifact verification

The workflow writes each backend to its own root:

- Tiny-BERT: `results/dynasent_runs/tiny-bert/`
- TF-IDF: `results/dynasent_runs/tfidf/`

Never copy TF-IDF outputs into the Tiny-BERT directory or combine them in a single analysis table without an explicit backend column. Each run artifact is named with its backend and GitHub run ID.

The workflow runs `src/verify_dynasent_artifact.py` after aggregation and before upload. It checks:

- both `round1` and `round2`;
- configured seeds (default `0 1 2 3 4`);
- four conditions: `majority_ce`, `human_soft_targets`, `label_smoothing`, and `human_soft_targets_null_augmented`;
- config, `metrics.json`, and `metrics.csv` backend/round/seed/condition consistency;
- complete metric matrix (one row per round × seed × condition);
- prediction JSONL completeness against each metric row's `n_test`;
- duplicate prediction keys and invalid human/predicted distributions;
- recomputed prediction-level mean JS divergence against the aggregate metric.

A machine-readable `verification_report.json` is written even on failure. A non-zero exit fails the workflow, preventing a broken or mixed-backend artifact from being treated as verified.

## Local use

```bash
python src/verify_dynasent_artifact.py \
  --results-root results/dynasent_runs/tiny-bert \
  --backend tiny-bert \
  --seeds 0 1 2 3 4 \
  --rounds round1 round2
```

Use `--results-root results/dynasent_runs/tfidf --backend tfidf` for a TF-IDF run. The verifier intentionally refuses a backend mismatch.
