# Financial PhraseBank and Twitter Airline: data-forensic audit

**Status: loader bug identified and patched; live source-level audit still required before new domain training.**

## Finding 1 — incorrect stratification key

The previous `_stratified_split` grouped rows by `row[-1]`. This is the sentiment label for simple `(text, label)` rows, but not for the domain loaders:

- Financial PhraseBank rows are `(text, binary_sentiment, agreement_tier)`. The old split stratified on agreement tier rather than sentiment.
- Twitter Airline rows are `(text, binary_sentiment, worker_confidence)`. The old split stratified on the continuous confidence value, potentially creating many tiny strata and materially distorting the requested 70/10/20 split.

The function now uses explicit `label_index=1`, groups normalized duplicate text into a single partition, and raises if the same normalized text has conflicting labels. Synthetic tests cover both metadata-bearing row shapes and text leakage.

## Dataset-specific semantics

### Financial PhraseBank

The upstream dataset contains 4,846 sentences in the >=50%-agreement configuration, 4,217 in >=66%, 3,453 in >=75%, and 2,264 in the all-agree subset. The public dataset card identifies labels as negative, neutral, positive and describes the agreement configurations as nested subsets. See the [dataset card](https://huggingface.co/datasets/takala/financial_phrasebank) and the original paper, Malo et al. (2014).

The current repository filters out neutral and maps positive/negative to a binary task. It derives three disjoint metadata tiers (`50-65`, `66-99`, `100`) by membership in the >=50, >=66, and all-agree sets. These tiers encode agreement strength, not the full annotator vote vector. They can support stratified subgroup analysis, but cannot support JS divergence against a human class-probability distribution.

### Twitter US Airline Sentiment

The current loader filters out neutral labels and rows without worker-confidence metadata, then maps positive/negative to binary sentiment. The retained confidence value is continuous metadata and is not a class distribution. It must not be converted into soft targets or treated as inter-annotator vote proportions.

## Historical result status

The archived multi-domain runs were produced using the previous split implementation. Their saved predictions, test sizes, and learning curves are **legacy exploratory evidence** and must not be treated as the final corrected-split results. The code correction changes test membership and can change the exact train/dev/test counts. Re-run the domain experiments from scratch after the live data audit passes.

## Validation limitation in this environment

The environment could not resolve GitHub hostnames while attempting to download the current raw source files. Therefore no claim is made that a live source-level audit has passed. The code includes `src/audit_domain_data.py` to download through the production loaders, report class distributions, duplicate text counts, metadata ranges/tiers, split overlaps, and verify normalized-text disjointness when run in a network-enabled environment.

## Required before final experiments

1. Run `python src/audit_domain_data.py --output results/domain_data_audit.json` in a network-enabled runner.
2. Confirm expected source schemas, label values, counts, duplicate/conflicting-label behavior, and metadata validity.
3. Record exact split counts and update full-budget constants from the actual corrected train split.
4. Re-run all finance/airline experiments with the corrected split and new run IDs; do not overwrite or merge legacy predictions.
5. Report Macro-F1, accuracy, and calibration only where supported by saved probabilities; reserve JS divergence against human vote distributions for DynaSent unless new distributional annotations are obtained.
