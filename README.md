# Does Self-Supervised Pre-training Help *Tiny* Transformers? 
### A data-efficiency, robustness and efficiency study for CPU-only sentiment classification

Research project in NLP built around a **1.2 M-parameter-encoder BERT** (2 layers, hidden 128) that is
pre-trained *and* fine-tuned entirely on a laptop CPU. No GPU, no internet-hosted model weights required.

## Research questions
* **RQ1 (data efficiency)** – With few labels, does MLM pre-training on ~1.2 M words of *unlabeled* in-domain text beat (a) the same network trained from random init and (b) strong TF-IDF baselines?
* **RQ2 (robustness)** – How do the models cope with typos, dropped words and an out-of-domain test set (IMDb documents)?
* **RQ3 (how much pre-training?)** – How does downstream accuracy / linear-probe accuracy evolve with pre-training steps?
* **RQ4 (efficiency)** – Size, CPU latency, throughput and the effect of INT8 dynamic quantization.

## Layout
```
src/config.py            paths, architecture, sizes, seeds
src/data.py              SST-2 + NLTK movie_reviews loaders (auto-download from GitHub), perturbations
src/tok.py               WordPiece tokenizer training
src/pretrain_mlm.py      MLM pre-training (CPU)
src/finetune.py          classifier (mean-pool head), training loop, prediction
src/baselines.py         TF-IDF + LogReg / NaiveBayes
src/families.py          non-transformer families: VADER lexicon, SVM, char n-grams, word2vec / fastText (same 1.2M words)
src/ambiguity.py         ambiguity-aware analysis (SST-5 polarity strength, consensus difficulty, linguistic markers)
src/exp_utils.py         evaluation sets & single-run helpers (+ frozen linear probe)
src/run_experiments.py   pilot / learning curves / checkpoint ablation (resumable)
src/efficiency.py        params, latency, throughput, INT8 quantization
src/make_assets.py       figures, LaTeX tables, bootstrap tests, results/summary.md
src/hf_benchmark.py      OPTIONAL: real HF checkpoints (bert-tiny/mini/small, DistilBERT, MiniLM)
run_all.sh               full pipeline
results/                 raw results (results.jsonl, all_runs.csv), figures, logs, summary.md
report/                  main.tex (+ figures/, tables/) -> upload folder to Overleaf
slides/                  presentation
```

## Setup (Python >= 3.9)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install tabulate                                     # only for make_assets.py markdown tables
```
CPU-only PyTorch is enough (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).
Disk: < 100 MB for data + models. RAM: < 2 GB.

## Run
```bash
bash run_all.sh                 # core + finance + airline (~2-3 h on 1 CPU core; varies by machine)
bash run_all.sh --core-only     # SST-2 core study only
```
or step by step:
```bash
cd src
python pretrain_mlm.py --steps 5000 --lr 1e-3 --dropout 0.0 --save_steps 0,100,300,1000,2500,5000
python run_experiments.py --stage pilot      # LR selection on dev
python run_experiments.py --stage curve      # learning curves + robustness + OOD
python run_experiments.py --stage ckpt       # pre-training-steps ablation
python efficiency.py
python make_assets.py                        # -> results/summary.md, report/figures, report/tables
```
All experiment stages are **resumable** when the matching checkpoints and logs are present. `--resume` reuses `results/models/mlm_final`; if result logs exist but that checkpoint is missing, the runner stops rather than mixing new weights with old results. Use a clean `bash run_all.sh` to regenerate a consistent run.
Data (SST-2 from GitHub, NLTK movie_reviews) is downloaded automatically on first use.

## Multi-domain (RQ8: cross-domain generalisation)
Two more domains, both with REAL (not proxy) ambiguity signals: Financial PhraseBank (finance news, 1,966
sentences, real annotator-agreement tiers) and Twitter Airline Sentiment (11,541 tweets, real per-tweet
confidence scores). Every run automatically reports accuracy on the OTHER two domains too (cross-domain
transfer), no extra training needed.
```bash
cd src
python run_experiments.py --stage all --domain finance    # ~15-25 min (rough estimate)
python run_experiments.py --stage all --domain airline    # ~15-25 min (rough estimate)
python backfill_cross_domain.py                            # one-off: adds cross-domain fields to old SST-2 rows
python make_assets.py                                       # figures, tables, Macro-F1 and cross-domain summaries
python cross_domain_stats.py --bootstrap 2000               # paired accuracy/Macro-F1 CIs; needs dataset access
```
`tiny_mlm` reuses the SAME movie-review-pretrained checkpoint on every domain (deliberately -- this tests
whether pretraining on one domain transfers to fine-tuning on another). Static embeddings (word2vec/fastText)
and TF-IDF are always trained fresh on each domain's own text. `--stage ckpt` is sst2-only (skipped for other
domains) since the pre-training-length ablation uses the movie-review checkpoint ladder specifically.

## Model families and ambiguity analysis
```bash
cd src
python run_experiments.py --stage families   # VADER, word SVM, char n-gram LR, word2vec/fastText + LR -> results/families.jsonl (~3 min)
python ambiguity.py                          # -> results/ambiguity.json (also run automatically by make_assets.py)
python make_assets.py                        # fig_families, fig_ambiguity, tab_families/tab_equiv/tab_amb_*.tex, summary.md
```
Ambiguity uses the fine-grained SST-5 test labels (downloaded automatically) as a *polarity-strength* proxy for ambiguity:
strong = very negative/very positive, weak = negative/positive (mild, close to neutral). It is not annotator disagreement.
Public checkpoints enter the ambiguity analysis when their predictions exist:
`python hf_benchmark.py --models tiny,distilbert --rerun` (re-runs earlier models and stores predictions).

## Public pre-trained checkpoints (BERT-Tiny/Mini/Small/Medium, DistilBERT, MiniLM)
Needs internet access to huggingface.co the first time (weights are cached afterwards).
```bash
cd src
python hf_benchmark.py --list                                  # model zoo with default learning rates
python hf_benchmark.py --models mini,small,minilm              # default sizes 1000,6920 x seeds 0,1,2
python hf_benchmark.py --models small --sizes 1000 --seeds 0   # cheap first check
python hf_benchmark.py --models mini,small --tune              # choose lr on dev (3 values) before the runs
python hf_benchmark.py --models /path/to/local/checkpoint      # any HF-format directory
python make_assets.py                                          # updates tab_hf*.tex, fig_hf, summary.md
```
Results go to `results/hf_results.jsonl`, `results/hf_efficiency.jsonl` and `results/preds/hf_*.npy` (stored predictions enable paired
bootstrap tests against TF-IDF). The script is **resumable**. Cost warning: DistilBERT-sized models need ~15-28 min per run on a laptop CPU.

The paired-bootstrap script resamples test examples jointly for both systems and averages paired differences over the available matched seeds. Its intervals are conditional on those seeds; the output also reports seed-to-seed SD. Do not describe these intervals as capturing retraining uncertainty.

## DynaSent human-distribution experiments (Phase 2)

DynaSent is now connected to an end-to-end pilot runner. The primary task preserves the four recorded categories in this order: `negative`, `neutral`, `positive`, `mixed`. The loader preserves human vote distributions even when `gold_label` is null. For the matched three-objective comparison, all methods use the same sampled training examples with non-null majority labels; an optional separate null-augmented soft-target condition can include a configured share of null-majority rows (default 50%) at the same total budget. It is explicitly not a matched-data comparison, and a training manifest records sampled IDs and null-gold status. Normalized train sentences found in dev/test are excluded from the effective training view, and duplicate training sentences are deduplicated; dev/test are not changed.

The uploaded DynaSent repository archive is included at `data/dynasent-main.zip` for reproducible pilot runs. Run the CPU-friendly TF-IDF baseline across both rounds:

```bash
./run_dynasent_pilot.sh
# Or configure it:
BUDGET=300 EVAL_SIZE=1000 EPOCHS=15 SEED=1 ./run_dynasent_pilot.sh
```

For the local tiny-BERT model (requires `transformers` and `tokenizers` from `requirements.txt`):

```bash
python src/run_dynasent_experiments.py \
  --data data/dynasent-main.zip --round round1 --backend tiny-bert \
  --checkpoint results/models/mlm_final --budget 100 --eval-size 500 --seeds 0 --epochs 3
```

The three matched objectives are majority-label cross-entropy, human-distribution soft-target cross-entropy, and generic label smoothing. Each run writes `metrics.json`, `metrics.csv`, `predictions.jsonl`, `config_and_audit.json`, `training_manifest.jsonl`, and `summary.md` under `results/dynasent_pilot_<round>/`. Reported metrics include held-out Jensen–Shannon divergence (bits), soft Brier score, accuracy, macro-F1, and ECE. Use `--eval-size 0` for the full test split and multiple seeds for final analysis. Pilot outputs are exploratory, not final research claims.

The null-majority comparison protocol, interpretation limits, and reproducible commands are documented in [`docs/NULL_MAJORITY_EXPERIMENT_DESIGN.md`](docs/NULL_MAJORITY_EXPERIMENT_DESIGN.md).

The DynaSent workflow is automated after this repository version is pushed to GitHub:

- **On relevant pushes to `main`/`master`:** run the fast checks, then the default CPU-friendly TF-IDF pilot for both rounds.
- **Every Sunday at 02:00 UTC:** the pilot is skipped unless repository variable `ENABLE_WEEKLY_PILOT=true` is set.
- **Manual dispatch remains optional:** select round, backend, budget, evaluation size, and epochs from the Actions page.
- **Outputs are uploaded as run-specific workflow artifacts** for 30 days. The workflow does not commit generated results back into the repository, so it cannot overwrite research files or create a commit loop.

No manual click is needed for the push-triggered or weekly runs once the workflow is present on GitHub and Actions are enabled. The first publication of this updated archive to your GitHub repository is still a one-time setup step. The default TF-IDF backend is quick and CPU-friendly; tiny-BERT uses the local MLM checkpoint and can take longer, so it remains an optional manually selected backend until its CI runtime is validated. See [`docs/PHASE2_IMPLEMENTATION.md`](docs/PHASE2_IMPLEMENTATION.md) and [`docs/DEPLOYMENT_AND_PILOT.md`](docs/DEPLOYMENT_AND_PILOT.md). Fast checks: `python -m py_compile src/*.py tests/*.py && pytest -q tests`.

## Build the report
Upload the whole `report/` folder to Overleaf (New Project -> Upload Project), compile `main.tex`
with pdfLaTeX + BibTeX (references are in `refs.bib`; Overleaf runs BibTeX automatically). Re-run `make_assets.py` to refresh figures/tables automatically after new experiments.

## Troubleshooting
* **`zsh: /usr/local/bin/pip: bad interpreter ... Python 2.7`** - a stale `pip` script points to a removed Python. Use `python -m pip install -r requirements.txt tabulate` (ideally inside a venv).
* **`ERROR:root:code for hash blake2b/blake2s was not found` + traceback** - printed by Python's `hashlib` at import time: your Python build lacks the
  blake2 hash functions (typically a partial pyenv build, e.g. missing OpenSSL headers when it was compiled). Nothing in this project uses blake2;
  every result in this repo was produced with these messages present, so they are safe to ignore. To remove them: `brew install openssl@3 xz`, then
  `pyenv install --force 3.13.0` (or use 3.12) and recreate the venv.
* **`Token indices sequence length is longer than ... (601 > 512)`** - harmless: whole reviews are tokenised before being cut into 62-token blocks; the long sequences are never fed to the model.
* **`Loading weights ...` / `Writing model shards ...` progress bars, qnnpack `reduce_range` warning, torch.ao deprecation notice** - informational.
* **`RuntimeError: ... NoQEngine` from `efficiency.py`** - the quantization backend must be selected explicitly; `efficiency.py` now picks fbgemm/x86/onednn/qnnpack
  (qnnpack worked on Apple Silicon) and skips the INT8 row if none is usable. INT8 speed depends on the backend: on Apple Silicon (qnnpack) it was *slower* than FP32 here.
* **Runs are "skipped"** - runs are resumable via `results/results.jsonl`. `bash run_all.sh` starts *clean*; use `--resume` to keep earlier results.
  Never mix a `results.jsonl` from one machine with checkpoints from another. `run_all.sh` does not delete `results/hf_results.jsonl`.
* **`make_assets.py` says results.jsonl not found** - run `bash run_all.sh` first (or pass `--results_dir <folder>`).
* **Numbers differ slightly on your machine** - different hardware/BLAS gives different random streams; expect changes of a fraction of a point.
  The report, slides and `results/` in this archive were all produced on one Apple-Silicon Mac.


## Fast integrity tests
```bash
python -m py_compile src/*.py tests/*.py
pytest -q tests
bash -n run_all.sh
```
These tests validate the saved three-seed records, cross-domain direction arithmetic, and generated accuracy/Macro-F1 tables. They do not retrain the models or replace the full experiment run.

## GitHub Actions CI/CD and evidence validation

The repository has four workflows under `.github/workflows/`:

- **NLP Research CI** runs on pushes and pull requests. It compiles Python, checks shell scripts, runs fast tests, and uploads a saved-evidence audit.
- **Saved Evidence Audit** runs weekly or manually. It checks archived result records and generated assets without training models or downloading datasets.
- **Research Experiment (manual, CPU)** remains an explicit, resource-intensive manual run for the legacy full pipeline.
- **DynaSent Soft-Target Pilot** runs automatically after relevant pushes to `main`/`master` and every Sunday at 02:00 UTC. It tests and runs the CPU-friendly TF-IDF comparison on both rounds, then uploads metrics and predictions as a workflow artifact. A manual override is still available but not required for routine runs.

Start with the fast checks locally or on GitHub:

```bash
python -m py_compile src/*.py tests/*.py ci/*.py
bash -n run_all.sh
pytest -q tests
python ci/audit_evidence.py --output results/evidence_audit.json
```

**Evidence policy:** passing CI and saved-record tests only establish code/artifact consistency. They do not prove the reported scientific claims. Fresh training, retained per-example predictions, fair hyperparameter tuning, independent seeds, and the paired multi-domain analysis remain separate validation requirements. See [`docs/CI_CD_PLAN.md`](docs/CI_CD_PLAN.md).
