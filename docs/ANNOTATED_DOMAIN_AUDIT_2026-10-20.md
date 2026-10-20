# Annotated-domain audit decision — 2026-10-20

Source artifact: GitHub Actions artifact `annotated-domain-audit-39401888732.zip`. The artifact reports a source/split audit only; it does not represent model training or external verification of annotation validity/licensing.

## Decision

- **SentiMP-En: eligible for the first pilot after the tie policy is applied.** Audit: 480 records, 480 unique normalized texts, three observed votes per row, and zero normalized-text overlap across train/dev/test. Nine rows have tied majorities (6 train, 1 dev, 2 test).
- **MVSC: hold from the core pilot pending manual source review.** Audit: 2,086 source rows, 2,082 unique normalized texts, 4 duplicate rows removed across 3 groups. One group (`deadpool tonight!`, source rows 486, 902, 1337) has vote counts `[0,0,3]`, `[0,0,2]`, `[0,0,3]`. Do not merge these rows as extra annotator votes. Inspect original source rows and document the chosen canonical record before clearing MVSC.
- Both datasets have zero normalized-text overlap across the generated splits. This audit did **not** find cross-split normalized-text leakage.

## Tie policy for the matched three-condition pilot

For a fair paired comparison, use the same non-tied training examples for majority-label cross-entropy, empirical soft-target cross-entropy, and label smoothing. This means the initial matched pilot intentionally excludes tied-majority rows from training across all three conditions; it does not claim to use every available annotation for soft-target training. Retain tied rows in test evaluation for JS divergence and soft Brier score using the observed vote proportions. Exclude tied test rows from accuracy and macro-F1 because no unique majority gold label exists. Report the tied counts and policy alongside results.

A separate all-vote soft-target experiment using tied training examples may be considered later, but should be reported as a different training-data condition rather than compared as if its training set were identical.

## Next steps

1. Run the SentiMP-En pilot first with five seeds.
2. Review its results and variance before scaling up.
3. Inspect MVSC duplicate source rows and rerun the audit after a documented policy is implemented.
4. Verify source terms/licensing before redistribution or publication.
