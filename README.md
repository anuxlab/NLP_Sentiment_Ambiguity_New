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

## Distribution-aware multi-domain experiments (primary extension)

The primary multi-domain extension retains **DynaSent Round 1/Round 2** and conditionally adds **SentiMP-En**. **MVSC is on hold** until the conflicting duplicate group is reviewed against source records. Do not treat both new datasets as approved merely because the audit workflow completed. MVSC contains three independent annotator labels for sentiment on English movie-related tweets; SentiMP-En contains three annotator labels for English political tweets. The adapters preserve the individual labels and compute class vote distributions only from those observed labels.

Financial PhraseBank and Twitter US Airline Sentiment are **not part of the primary distribution-aware experiments**: their current loaders expose agreement tiers or worker confidence, not per-example class-vote distributions. The old `run_experiments.py --domain finance|airline` path remains available only for reproducing legacy exploratory hard-label results; do not mix those results with the human-distribution JS-divergence experiments.

### 1. Audit the new datasets first

```bash
python src/annotated_domains.py --output results/annotated_domain_audit.json
```

The audit reports label/vote counts, ties, duplicates, duplicate groups with different vote counts, split overlaps, and mean human-vote entropy. A `REVIEW` result requires manual inspection before training; it is not a clean pass.

### 2. Run the first matched soft-target pilot

SentiMP-En passed the current loader/split audit (480 records; no duplicate normalized texts; three observed votes per row). Start with it:

```bash
python src/run_annotated_domain_experiments.py --dataset sentimp_en --seeds 0 1 2 3 4
```

Do **not** run MVSC yet. Its audit reports a duplicate group for `deadpool tonight!` with differing vote counts. Inspect the original source rows and document the decision before rerunning its audit. The runner blocks any dataset with audit status `REVIEW` unless `--allow-review` is explicitly supplied after review.

The CPU-friendly TF-IDF pilot compares majority-label cross-entropy, empirical human-distribution soft-target cross-entropy, and label smoothing on the same non-tied training examples. The primary metric is JS divergence against observed annotator vote proportions; soft Brier score, accuracy, macro-F1, and training time are also recorded. Tied-majority rows are preserved in the dataset audit and test set, but excluded from the matched three-condition training subset because majority CE requires a single gold class. The runner prints this policy at startup; JS divergence and soft Brier still use observed vote distributions for every test row, while accuracy and macro-F1 omit tied test rows. Null-majority augmentation is not enabled for these datasets in this initial implementation.

Results append to `results/annotated_domain_results.jsonl`; per-dataset audit reports are saved under `results/`. This is a separate experiment path and does not change DynaSent's existing runner, results, or MLM checkpoint.

### Dataset sources and caveats

- MVSC: [MIND-Lab/MVSC](https://github.com/MIND-Lab/MVSC), EACL 2017. The repository notes that tweet text and identifiers are subject to Twitter's terms; respect the source terms and cite Nozza et al.
- SentiMP-En: [Hugging Face dataset card](https://huggingface.co/datasets/rbnuria/SentiMP-En), CC BY-SA 4.0 as listed by the publisher. It provides 480 English rows in the currently published dataset viewer, with three annotator labels and predefined fold metadata.
- PRemo is **not yet included in the core runner**. Its individual polarity labels and context/no-context passes should be audited separately before inclusion, because the task is software-development discussions and its `undefined` category requires a prespecified policy.
