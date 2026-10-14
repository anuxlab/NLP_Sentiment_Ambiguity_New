# DynaSent pilot: deployment and interpretation

## What is runnable now?

The repository has an end-to-end offline-first pilot runner in `src/run_dynasent_experiments.py`, a two-round convenience script in `run_dynasent_pilot.sh`, and a manual GitHub Actions workflow in `.github/workflows/dynasent-pilot.yml`.

- `tfidf`: lightweight CPU baseline; needs the installed project requirements but no Transformers checkpoint or model download.
- `tiny-bert`: initializes the classifier encoder from `results/models/mlm_final` and adds a four-class head. It uses the local tokenizer/checkpoint only; it does not contact the Hugging Face Hub.
- Three matched conditions: `majority_ce`, `human_soft_targets`, and `label_smoothing`.
- Every matched condition receives the same sample of training rows with non-null majority labels. This avoids comparing different examples in the primary controlled contrast.
- `--include-null-soft-only` adds a fourth, explicitly unmatched exploratory condition that can use null-majority examples with valid vote distributions. Do not compare it as if it were matched to the hard-label conditions.

## Quick start

```bash
python -m pip install -r requirements.txt
pytest -q tests
./run_dynasent_pilot.sh
```

The default pilot uses budget 100, 500 test rows, one seed (0), and 10 epochs, on both DynaSent rounds. It is intentionally small. For the full test split, pass `EVAL_SIZE=0`. Increase seeds only after the pipeline has been checked and the experimental design is frozen.

A single run can be configured explicitly:

```bash
python src/run_dynasent_experiments.py \
  --data data/dynasent-main.zip --round round1 --backend tfidf \
  --budget 100 --eval-size 500 --seeds 0 --epochs 10 \
  --out-dir results/dynasent_pilot_round1
```

## Output files

- `metrics.csv` / `metrics.json`: one row per objective and seed.
- `predictions.jsonl`: each test sentence, human distribution, predicted distribution, predicted class, and per-example JS divergence.
- `config_and_audit.json`: arguments, split/filter audit, seed, label order, and matched-condition policy.
- `summary.md`: compact metric table with a pilot disclaimer.

Metrics include mean/median Jensen–Shannon divergence in bits (lower is better), distributional Brier score (lower is better), accuracy, macro-F1, and 10-bin ECE (lower is generally better). Accuracy and macro-F1 use the dataset's non-null `gold_label`; distributional metrics use all evaluated rows with valid human vote distributions.

## Current smoke pilot

The repository includes the initial CPU TF-IDF pilot outputs in `results/dynasent_pilot/` (Round 1) and `results/dynasent_pilot_round2/` (Round 2). These used budget 100, one seed, 10 epochs, and a fixed 500-example test sample. They are debugging/feasibility outputs only—not evidence of a statistically reliable advantage for any objective. The vocabulary is fitted only on the selected training sample, which is correct for leakage control but yields a small feature vocabulary at this budget.

## GitHub Actions

Open **Actions → DynaSent Soft-Target Pilot → Run workflow**. Select round (`round1`, `round2`, or `both`), backend, budget, test sample size, and epochs. The workflow runs tests and uploads the metrics/predictions as an artifact for 30 days. GitHub-hosted runners need internet access to install dependencies. The included dataset archive is used locally; the workflow does not download the data from Hugging Face.

## Before full experiments

1. Freeze the target policy (four-class primary; any ternary analysis is a separate sensitivity analysis).
2. Decide the final budgets and number of seeds before inspecting test results.
3. Run pilot checks, then use development data for model/epoch selection.
4. Run the full held-out test split only for the frozen settings and report uncertainty across seeds.
5. Keep Round 1 and Round 2 results separate initially; do not pool them without a pre-specified rationale.
6. Review DynaSent's data provenance/license notes before redistribution beyond this research workspace.

## Environment note

This execution environment had PyTorch and scikit-learn installed but no `transformers` or `tokenizers`, and outbound package installation was unavailable. Therefore the TF-IDF pilot was actually run here, while the tiny-BERT backend was syntax-checked but not trained in this environment. Install `requirements.txt` on the target machine or let GitHub Actions install it to run the tiny-BERT backend.


## Automatic GitHub Actions triggers and optional weekly runs

The workflows are designed for active development without unnecessary recurring training:

- **Push / pull request:** `NLP Research CI` runs tests and artifact checks automatically. Relevant code pushes also trigger the DynaSent pilot.
- **Weekly DynaSent pilot:** the schedule is present but **disabled by default**. The scheduled job is skipped unless repository variable `ENABLE_WEEKLY_PILOT` is exactly `true`.
- **Weekly saved-evidence audit:** also **disabled by default**. It runs only when repository variable `ENABLE_WEEKLY_AUDIT` is exactly `true`.
- **Manual dispatch:** both scheduled workflows remain manually runnable from **Actions**. The DynaSent workflow accepts round, backend, budget, evaluation size and epochs.
- **Full research experiment:** remains manual and requires typing `RUN`; it is not launched on a schedule.

To enable a weekly run later, open **Settings → Secrets and variables → Actions → Variables → New repository variable** and set the relevant variable to `true`. To disable it again, delete the variable or set it to any value other than `true`. No code edit is needed. Push-triggered CI and pilots continue regardless of these weekly switches.

The schedule definitions may wake GitHub Actions to evaluate the job condition, but when the variables are absent the compute job is skipped: no dependencies are installed and no model training or audit task runs. Generated outputs are uploaded as artifacts; the workflows do not commit results back to the repository.
