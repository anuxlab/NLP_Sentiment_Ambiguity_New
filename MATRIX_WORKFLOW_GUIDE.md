# Matrix-parallelized research workflow

This repository snapshot includes an automated GitHub Actions pipeline for the three sentiment domains (`sst2`, `finance`, `airline`). It is a complete codebase, not a patch.

## What runs automatically

- **Push to `main`/`master`** when experiment source, tests, dependencies, CI scripts, `run_all.sh`, or the experiment workflow changes.
- **Weekly full reproduction** on Sundays at 02:00 UTC.
- **Manual dispatch** remains available for recovery/debugging; it is optional, not the only trigger.

The ordinary CI workflow remains separate from the expensive experiment workflow.

## Pipeline layout

1. `pretrain`: trains the shared MLM model once and uploads checkpoints/logs as an artifact.
2. `domain-experiments`: runs `sst2`, `finance`, and `airline` as independent matrix jobs (up to three concurrently). Each job downloads the shared checkpoint and uploads its own evidence.
3. `finalize`: merges the three artifacts, rebuilds report assets, recalculates cross-domain bootstrap statistics, runs tests, and creates an evidence audit.
4. `report-failure`: makes a failed pipeline visible as a failed job.

The helper `ci/merge_experiment_artifacts.py` is post-processing only. It does not retrain models and fails on missing domains, unexpected domain labels, empty results, or prediction filename collisions.

## Before relying on the first run

- Ensure GitHub Actions is enabled for the repository.
- Check the first run's `pretrain`, each matrix domain, and `finalize` logs. Artifact path layouts can differ if the existing scripts' output paths change.
- The first relevant push to `main`/`master` will launch a full experiment, so it may consume several hours of runner time. Use a feature branch for review if you do not want the first run to start immediately.
- Weekly scheduling uses GitHub's UTC schedule and may start late during high runner load.
- Artifacts are retained for 14 days during intermediate stages and 90 days for the final evidence bundle.

## Local validation performed for this package

- Python compilation for `ci/`, `src/`, and `tests/`
- YAML parsing of all workflow files
- `bash -n run_all.sh`
- `python -m pytest -q tests` — 10 tests passed

These checks validate repository syntax and tests; they do not substitute for a successful live GitHub Actions run or scientific review of experiment results.
