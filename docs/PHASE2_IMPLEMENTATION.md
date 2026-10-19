# Phase 2 implementation: DynaSent soft-target foundations

**Status:** foundational modules and unit tests implemented; no model training launched.

## Primary target definition

The primary DynaSent target is four-class, in this fixed order:

1. `negative`
2. `neutral`
3. `positive`
4. `mixed`

For each row, `label_distribution` is converted from worker-ID lists to vote proportions. The distribution is preserved even when `gold_label` is `null`; those examples remain usable for soft-target training and distributional evaluation, but are not eligible for hard-label cross-entropy or majority-label accuracy/F1.

## Files added

- `src/dynasent_data.py`: offline loader for the official DynaSent v1.1 ZIP or an extracted directory; vote normalization; canonical label IDs; overlap filtering; optional training deduplication; audit summary; explicit ternary sensitivity mapping.
- `src/soft_targets.py`: human-distribution soft-target cross-entropy, hard-label cross-entropy with optional label smoothing, probability normalization, and Jensen–Shannon divergence.
- `tests/test_dynasent_soft_targets.py`: synthetic-fixture tests for data parsing, four-class order, null hard labels, training-only overlap filtering, deduplication, target normalization, loss values/gradients, invalid targets, and JS divergence.

## Data leakage policy

For each round separately, normalized training sentences that occur in dev or test are excluded from the **effective training view**. Optional deduplication then retains the first training occurrence in source order. The loader never edits the source archive and never filters dev/test. Audit counts expose the number removed. Round 1 and Round 2 are loaded and reported separately by default; do not combine rounds without a documented protocol.

## Ternary sensitivity analysis

`ternary_distribution()` is deliberately separate from the primary loader. Its default `renormalize_non_mixed` option removes mixed vote mass and renormalizes negative/neutral/positive proportions. The stricter `drop` option excludes any row with nonzero mixed mass. A row with only mixed votes maps to `None` under either policy. Report the chosen policy and sensitivity results; do not silently apply it to primary experiments.

## Losses and metrics

- **Majority-label CE:** `hard_label_cross_entropy(logits, label_ids)` on rows with a non-null gold label.
- **Generic label smoothing:** `hard_label_cross_entropy(logits, label_ids, label_smoothing=epsilon)` using the same eligible rows and training conditions.
- **Human-distribution soft target:** `soft_target_cross_entropy(logits, target_probs)` using valid vote distributions, including rows without a majority label.
- **Distributional evaluation:** `jensen_shannon_divergence(predicted_probs, target_probs)`; default logarithm base 2 yields values bounded by 1 bit.

This module supplies the loss and metric primitives. It is **not yet wired into `run_experiments.py` or a model-specific training loop**; that integration is the next Phase 2 step. Keep model family, label budget, seed, optimizer, training steps, validation policy, and compute budget matched across methods. Hyperparameter selection must use dev data only.

## Real archive loader smoke audit

The loader was run against the user's uploaded GitHub repository archive, which contains the official `dynasent-v1.1.zip` as a nested archive. The loader supports this format directly. This is a loader/data-structure smoke audit, not a replacement for the earlier full dataset audit.

| Round | Train before filtering | Train rows excluded for dev/test sentence overlap | Duplicate train rows excluded | Effective train rows | Dev / test unchanged |
|---|---:|---:|---:|---:|---:|
| Round 1 (Yelp) | 94,459 | 19 | 180 | 94,260 | 3,600 / 3,600 |
| Round 2 (Dynabench) | 18,535 | 1 | 14 | 18,520 | 720 / 720 |

Null `gold_label` rows in the original training splits: 10,071 for Round 1 and 2,136 for Round 2. These are retained in the effective training view when their text passes the leakage/deduplication filters, because their vote distributions are valid soft targets. The row counts removed for overlap are counts of training rows, not counts of distinct normalized sentence keys; a normalized key can occur multiple times in train. The prior full audit's cross-split overlap summary counted distinct overlap findings under a different aggregation, so those totals should not be compared as if they were the same unit.

The loader keeps rounds separate. It does not combine train/dev/test across rounds or automatically use one round as another's validation set. See the machine-readable summary in `results/dynasent_loader_smoke_audit.json`.

## Local checks run

```bash
python -m py_compile src/*.py tests/*.py
PYTHONPATH=src pytest -q tests/test_dynasent_soft_targets.py
PYTHONPATH=src pytest -q tests
```

Latest result in the implementation workspace: **20 tests passed**, including the existing repository tests and 10 new tests. These tests use synthetic DynaSent-like fixtures for loader behavior; they do not replace rerunning the loader on the actual DynaSent archive and do not establish any model performance claim.

## Official research resources

- DynaSent repository and dataset: https://github.com/cgpotts/dynasent
- DynaSent v1.1 archive: https://github.com/cgpotts/dynasent/blob/main/dynasent-v1.1.zip
- Dataset card: https://huggingface.co/datasets/dynabench/dynasent
- DynaSent datasheet: https://github.com/cgpotts/dynasent/blob/main/dynasent_datasheet.md
- DynaSent paper: https://aclanthology.org/2021.acl-long.186/
- Soft labels / annotator disagreement: https://aclanthology.org/2021.naacl-main.204/ and https://aclanthology.org/2023.semeval-1.12/

## Next implementation tasks

1. Add a controlled training runner that selects one of the three objective functions without changing the existing experiment pipeline.
2. Add deterministic label-budget sampling and multi-seed config files.
3. Evaluate all methods against the same held-out human vote distributions using JS divergence, then report hard-label metrics on rows with valid gold labels.
4. Add a pilot command and result schema before enabling full runs.
5. Wire these tested components into a controlled training runner, then review licensing/provenance notes before publishing.
