# GitHub Actions validation plan

## Principle

The pipeline distinguishes three evidence levels:

1. **Code correctness** — syntax, unit tests, schema and arithmetic checks.
2. **Archived-result integrity** — saved records and generated tables agree.
3. **Scientific reproduction** — fresh training from declared data, seeds, splits and tuning protocol.

Levels 1 and 2 must never be presented as proof of level 3.

## Workflows

- `NLP Research CI` (`ci.yml`): runs on pushes, pull requests and manual dispatch. Installs dependencies, compiles Python, validates shell syntax, runs tests and uploads an audit report.
- `Saved Evidence Audit` (`evidence-audit.yml`): manually runnable; weekly job is opt-in via repository variable `ENABLE_WEEKLY_AUDIT=true`. It checks saved records only and performs no training or data downloads.
- `DynaSent Soft-Target Pilot` (`dynasent-pilot.yml`): runs on relevant code pushes and manual dispatch. Weekly training is disabled by default; enable only with repository variable `ENABLE_WEEKLY_PILOT=true`.
- `Research Experiment (manual, CPU)` (`experiment.yml`): explicit manual dispatch only. The operator must type `RUN`, choose core-only or all domains, and choose clean or resume mode. Results are uploaded as a run artifact; the workflow does not commit results back to the branch.

## Recommended branch protection

Require `NLP Research CI / Static checks and artifact tests` to pass before merging. Keep experiment workflow manual because it is compute-intensive and network/data availability varies.

## How to use

1. Create a GitHub repository and push this directory to `main`.
2. Open **Actions** and enable workflows if GitHub prompts you. Weekly schedules stay disabled unless you explicitly add the opt-in repository variables.
3. Confirm the CI workflow passes on the initial push.
4. Open the `Saved Evidence Audit` run artifact and inspect `evidence_audit.json`.
5. Only after CI and artifact checks pass, run `Research Experiment (manual, CPU)` with `confirmation=RUN`. Start with `core-only` before `all-domains`.
6. Download the experiment artifact and independently inspect logs, result counts, predictions and generated tables.
7. Run the new multi-domain paired analysis on accessible data before making inferential claims. A CI green check is not evidence that a scientific claim is true.

## Limitations / follow-up work

- The current test suite checks archived records and does not retrain models.
- The full experiment workflow depends on upstream data hosts being reachable and can take hours. GitHub-hosted runner quotas and time limits apply.
- Do not store credentials or personal access tokens in source files.
- Pin action versions to full commit SHAs before using this in a high-assurance publication pipeline.
- Add a separate test that uses synthetic tiny predictions to exercise every branch of `cross_domain_stats.py`; then run the paired analysis on the real saved predictions.
