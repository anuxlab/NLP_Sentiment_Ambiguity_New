# Supporting DynaSent comparison: TF-IDF, Tiny-BERT, and VADER

## Research priority

The primary contribution remains the ambiguity/distribution-aware study: compare majority-label cross-entropy, human-vote soft targets, and label smoothing; stratify by human vote disagreement; and evaluate human-distribution alignment with Jensen–Shannon divergence. The model-family comparison is a small supporting experiment, not a replacement for this core study.

## Common dataset and protocol

- Dataset: DynaSent v1.1, Round 1 and Round 2 analyzed separately.
- Training budgets: 25, 50, 100, 200, 300.
- Default seeds: 0, 1, 2.
- Evaluation: full test split, with the same data loader, label order, and evaluation implementation.
- For the model-family curves, use `majority_ce` only, so TF-IDF and Tiny-BERT share the same target objective.
- VADER is zero-shot and has no labeled training budget; it is a reference line/point, not a learned curve.

## Run

```bash
bash scripts/run_dynasent_supporting_matrix.sh
python src/run_dynasent_vader_baseline.py --round both
```

Or use GitHub Actions → **DynaSent Supporting Model Comparison**. The workflow uploads the raw metrics, per-example predictions, audit metadata, summaries, plots, and observed ranking table.

## Interpret cautiously

- Report a ranking change only at budgets where both model families have results.
- Error bands in the learning-curve plots are standard errors across the available training seeds, not test-example bootstrap intervals.
- The four-class VADER mapping is heuristic: if both VADER positive and negative proportions are at least 0.15, assign `mixed`; otherwise use compound thresholds ±0.05 for positive/negative and neutral between them. This is project-specific and should be documented as such.
- Do not infer human ambiguity from model confidence. Human ambiguity is defined from the annotator vote distribution (e.g. entropy or disagreement tiers).
- The existing Tiny-BERT result at budget 100 is historical exploratory evidence. A full Tiny-BERT learning curve must be generated with the same protocol before claiming a budget-dependent ranking change.
