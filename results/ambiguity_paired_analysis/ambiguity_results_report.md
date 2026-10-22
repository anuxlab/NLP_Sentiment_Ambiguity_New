# Ambiguity Analysis: Results

## Research question

Does the change in JS divergence from human soft-target training, relative to majority-label cross-entropy, differ between unanimous and non-unanimous examples?

## Main findings

Negative subgroup JS changes favor human soft-target training. A negative interaction means the change is more favorable for non-unanimous examples than for unanimous examples.

The interaction tests use paired seed-level effects. Holm correction is applied across the 10 DynaSent round-budget comparisons.

## Interaction results

| Dataset | Budget | ΔJS H=0 | ΔJS H>0 | Interaction | Raw p | Holm p | Negative seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DynaSent Round 1 | 25 | -0.000756 | -0.000677 | 0.000079 | 0.812500 | 1.000000 | 5/10 |
| DynaSent Round 1 | 50 | -0.000330 | -0.001964 | -0.001634 | 0.046875 | 0.328125 | 9/10 |
| DynaSent Round 1 | 100 | -0.001260 | -0.006134 | -0.004874 | 0.003906 | 0.031250 | 9/10 |
| DynaSent Round 1 | 200 | 0.001781 | -0.007750 | -0.009531 | 0.001953 | 0.019531 | 10/10 |
| DynaSent Round 1 | 300 | 0.004315 | -0.008227 | -0.012542 | 0.001953 | 0.019531 | 10/10 |
| DynaSent Round 2 | 25 | -0.003865 | -0.002735 | 0.001129 | 0.332031 | 1.000000 | 5/10 |
| DynaSent Round 2 | 50 | -0.003424 | -0.004257 | -0.000834 | 0.558594 | 1.000000 | 6/10 |
| DynaSent Round 2 | 100 | -0.008094 | -0.010980 | -0.002886 | 0.087891 | 0.527344 | 8/10 |
| DynaSent Round 2 | 200 | -0.007558 | -0.009733 | -0.002175 | 0.302734 | 1.000000 | 6/10 |
| DynaSent Round 2 | 300 | -0.008175 | -0.010406 | -0.002231 | 0.175781 | 0.878906 | 7/10 |

## Multiplicity-adjusted findings

- **DynaSent Round 1, budget 100:** interaction = -0.004874; Holm-adjusted p = 0.031250.
- **DynaSent Round 1, budget 200:** interaction = -0.009531; Holm-adjusted p = 0.019531.
- **DynaSent Round 1, budget 300:** interaction = -0.012542; Holm-adjusted p = 0.019531.

## Seed consistency

The negative-seed count describes how many of the 10 training seeds had a negative interaction. It is a robustness diagnostic, not an independent confirmation of the interaction test.

## Interpretation and limitations

- Inference is conditional on the selected evaluation examples.
- The exact seed-level sign-flip test assumes symmetric seed-level effects under the null.
- Bootstrap intervals are exploratory and are not multiplicity-adjusted.
- The analysis does not establish a causal mechanism or generalization to other datasets.
- Entropy strata describe uncertainty in the target distribution; they should not automatically be interpreted as raw annotator disagreement.

## Figure

See `ambiguity_interaction.png` and `ambiguity_interaction.pdf` in this directory, if generated.
