# DynaSent statistical validation

Run locally after downloading/unzipping an experiment artifact that contains `round1/` and `round2/`, each with `metrics.csv` and `predictions.jsonl`:

```bash
python src/analyze_dynasent_results.py \
  --input-dir /path/to/unzipped-artifact \
  --output-dir results/dynasent_analysis/tiny-bert \
  --expected-seeds 0 1 2 3 4 \
  --bootstrap-replicates 10000 \
  --permutation-replicates 10000
```

The script rejects mixed backends, incomplete round/seed/condition metric matrices, duplicate prediction keys, mismatched test IDs/human distributions, and metric-vs-prediction JS mean discrepancies. It produces primary round-specific JS contrasts, paired seed-level effects, across-seed metric summaries, exploratory secondary comparisons, human-vote-entropy disagreement tiers, plots, and `FINAL_RESULTS_REPORT.md`.

## Statistical interpretation

- Primary outcome: mean per-example Jensen–Shannon divergence in bits; lower is better.
- Primary contrast: `human_soft_targets - majority_ce`, separately for each round.
- Bootstrap resamples test `text_id`s with replacement while holding all five fitted seeds fixed. These intervals quantify test-example sampling uncertainty conditional on these models, not training-run uncertainty.
- Two-sided paired sign-flip p-values operate on per-example differences averaged across the fixed seeds. Holm adjustment is applied only to the two primary round-specific p-values.
- The existing five-seed run has already been inspected, so all inferential output is labelled exploratory, not confirmatory.
- Null-augmented soft targets are exploratory because the training-data composition differs.
- TF-IDF and Tiny-BERT must be analyzed in separate backend-specific output directories and must never be pooled.
- No missing tier is fabricated; disagreement is derived only from human vote distributions.

The workflow runs this analysis after aggregation and uploads the report, CSVs, manifest, and figures alongside the original predictions. It does not rerun training for analysis.
