# DynaSent multi-seed full-test evaluation — local execution report

## Execution and completeness

- Backend: TF-IDF linear classifier; no pretrained Transformer was used in this run.
- Rounds: DynaSent Round 1 and Round 2, reported separately.
- Seeds: 0, 1, 2, 3, 4.
- Label budget: 100 examples per condition and seed.
- Epochs: 10.
- Test split: full split (`--eval-size 0`); Round 1 has 3,600 examples and Round 2 has 720 examples in this loader configuration.
- Conditions: majority CE, human vote-distribution soft targets, label smoothing, and null-augmented human soft targets.
- Null-augmented condition: 50 null-majority and 50 non-null-majority training examples per seed. This changes the training-data composition, so it is a separate condition rather than a matched comparison.
- Outputs: 40 metric records, 86,400 per-example prediction rows, 10/10 complete round/seed groups.
- Local validation: 29 tests passed; Python compilation and shell syntax checks passed.

## Mean and sample standard deviation across seeds

| Round | Condition | JS ↓ mean (SD) | Macro-F1 ↑ mean (SD) | Accuracy mean (SD) |
|---|---|---:|---:|---:|
| Round 1 | Majority CE | 0.3910 (0.0092) | 0.1724 (0.0372) | 0.3534 (0.0188) |
| Round 1 | Human soft targets | **0.3880 (0.0063)** | **0.1890 (0.0375)** | **0.3603 (0.0185)** |
| Round 1 | Label smoothing | 0.3911 (0.0090) | 0.1754 (0.0375) | 0.3554 (0.0193) |
| Round 1 | Null-augmented soft targets | 0.3926 (0.0035) | 0.1631 (0.0196) | 0.3421 (0.0091) |
| Round 2 | Majority CE | 0.4382 (0.0133) | 0.1790 (0.0336) | **0.3414 (0.0144)** |
| Round 2 | Human soft targets | 0.4329 (0.0109) | 0.1706 (0.0323) | 0.3378 (0.0119) |
| Round 2 | Label smoothing | 0.4377 (0.0129) | 0.1809 (0.0341) | 0.3417 (0.0150) |
| Round 2 | Null-augmented soft targets | **0.4300 (0.0070)** | **0.2158 (0.0356)** | 0.3303 (0.0146) |

Lower JS divergence is better. Macro-F1 and accuracy are calculated only on test rows with a valid majority gold label; JS divergence and soft Brier use human vote distributions.

## Paired seed differences versus majority CE

Improvement is oriented so positive is better (JS divergence is sign-reversed; macro-F1/accuracy are not).

| Round | Candidate | JS improvement mean (SD) | Macro-F1 improvement mean (SD) | Accuracy improvement mean (SD) |
|---|---|---:|---:|---:|
| Round 1 | Human soft targets | +0.00297 (0.00403) | +0.01662 (0.01016) | +0.00689 (0.00541) |
| Round 1 | Null-augmented soft targets | -0.00156 (0.01221) | -0.00929 (0.03664) | -0.01139 (0.02151) |
| Round 2 | Human soft targets | +0.00532 (0.00469) | -0.00846 (0.01026) | -0.00361 (0.00400) |
| Round 2 | Null-augmented soft targets | +0.00821 (0.00845) | +0.03676 (0.05227) | -0.01111 (0.01629) |

These are descriptive paired differences over five seeds. They are not p-values and do not establish statistical significance.

## Human-vote disagreement analysis

The test split's observed vote distributions have only two types in this data loader: unanimous (entropy 0) and non-unanimous (entropy > 0; observed distributions are 4:1). The aggregator therefore reports those two observed tiers instead of forcing empty moderate/high bins. These tiers are derived from human votes, not model confidence. See `disagreement_tier_summary.csv` for all methods and both rounds.

## Interpretation

- Human soft targets show a small positive average paired improvement over majority CE in Round 1 on JS divergence, macro-F1, and accuracy.
- In Round 2, human soft targets improve JS divergence but have lower macro-F1 and accuracy than majority CE on average.
- The null-augmented condition is worse than majority CE in Round 1 on all three displayed metrics. In Round 2, it improves JS divergence and macro-F1 but lowers accuracy.
- This does **not** support a universal claim that adding null-majority examples improves sentiment classification. The outcome differs by DynaSent round and metric, and seed variability is substantial for macro-F1.

## Limitations and next step

This is a local CPU TF-IDF experiment, not yet a GitHub Actions reproduction. Five seeds are a useful reproducibility check but are not sufficient by themselves for strong inferential claims. No hyperparameters were tuned on the test set. Before a paper-level claim, verify exact code/data commit on GitHub, preserve per-example paired predictions, calculate paired bootstrap confidence intervals over test examples (alongside seed variability), and consider whether the null-majority subset is representative. Keep Round 1 and Round 2 results separate.
