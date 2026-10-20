# Phase 3: Statistical results audit

The analyzer validates both round configuration files and checks that the effective backend-specific learning rate agrees with per-run metrics. For Tiny-BERT, the only active learning-rate field is `tiny_bert_lr`; the legacy generic `lr` field is rejected.

Before producing the report, the analyzer checks prediction/metric counts, recomputes per-example JS divergence, group JS mean/median, soft Brier score, and valid-majority-label accuracy, macro-F1, and ECE from prediction distributions. It rejects missing/mixed backends, incomplete conditions, mismatched test IDs or human distributions, and configuration/metric conflicts.

The analysis manifest records the effective learning rate and validation outcomes. Historical artifacts are not rewritten; rerun analysis on the preserved artifact to produce a corrected, versioned analysis output.

Statistical interpretation remains exploratory: test-example bootstrap and paired sign-flip tests condition on the five trained seeds and do not estimate between-training-run uncertainty. The sign-flip test assumes sign exchangeability of paired item differences under the null.
