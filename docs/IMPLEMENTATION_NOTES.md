# Next-step implementation: ambiguity-inclusive soft targets

## What is implemented

The legacy `run_annotated_softlabel_experiment.py` implementation has been retired because its majority-CE path used scikit-learn LogisticRegression while the other conditions used a PyTorch linear model. Use the controlled runner below to reproduce the SentiMP-En controlled comparison. For the ongoing multi-domain extension, use `src/run_annotated_domain_experiments.py`; DynaSent remains on `src/run_dynasent_experiments.py`.

- Added `src/run_annotated_softlabel_experiment_controlled.py`, a controlled TF-IDF runner for three matched conditions (`majority_ce`, `human_soft_targets`, `label_smoothing`) and a separately reported `human_soft_targets_tie_inclusive` condition.
- It derives the target distribution from per-class vote counts, excludes tied examples from hard-label metrics, retains tied test examples for JS divergence and Brier score, checks normalized text overlap across train/dev/test, records tie counts, and writes metrics, per-example predictions, and an audit JSON.
- The existing DynaSent runner already supports `--include-null-soft-only`; that condition is intentionally reported separately from the matched three-condition comparison because its training composition differs.

## Input format

JSONL, one example per line. Each record needs `text`, `split` (`train`, `dev`, or `test`), and `votes`, for example:

```json
{"text":"I liked it, but the ending was weak","split":"train","votes":{"negative":1,"neutral":0,"positive":2}}
```

Class order defaults to `negative neutral positive`; supply `--labels negative neutral positive mixed` for a four-class dataset. The input should contain the raw vote counts, not only a majority label. Duplicate or overlapping normalized text across splits fails fast.

## Run

```bash
python src/run_annotated_softlabel_experiment_controlled.py \
  --input data/sentimp_en_votes.jsonl \
  --out-dir results/sentimp_en_controlled \
  --budgets 25 50 100 200 300 --seeds 0 1 2 3 4
```

For DynaSent, use the existing runner and its separately labelled null-majority condition:

```bash
python src/run_dynasent_experiments.py --round round1 --backend tfidf \
  --budget 100 --eval-size 0 --seeds 0 1 2 3 4 \
  --include-null-soft-only --out-dir results/dynasent_round1_tie_inclusive
```

Repeat for `round2`. Use the same evaluation protocol and a fixed test set for all conditions. `--eval-size 0` evaluates the full test split.

## Important data blockers

The current conversation workspace has the SentiMP-En *audit JSON* and prior result JSONL, but not the raw 480-example vote-level dataset file. Therefore the new runner cannot honestly be reported as executed on SentiMP-En until that source file is supplied. Similarly, MVSC remains on hold pending review of the four duplicate-text records and the inconsistent `deadpool tonight!` vote-count group. Do not include MVSC in the primary results until those cases are resolved and documented.

## Interpretation safeguards

The tie-inclusive condition intentionally has a different training composition and may have more examples than the matched budget. Compare it to the matched soft-target condition as an ambiguity-inclusion ablation, not as a controlled loss-only comparison. Report both requested and actual training sizes, number of ties, JS divergence (primary), soft Brier score, and hard-label accuracy/macro-F1 on non-tied test rows. Keep the existing matched results unchanged for reproducibility.
