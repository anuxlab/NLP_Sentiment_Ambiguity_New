#!/usr/bin/env bash
# Reproduce the study on CPU. Full run includes SST-2, Financial PhraseBank, and Twitter Airline.
#   bash run_all.sh                 # clean, all three domains
#   bash run_all.sh --resume        # resume all domains without deleting results
#   bash run_all.sh --core-only     # SST-2 study only (does not reproduce multi-domain results)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
RESUME=0
CORE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --resume) RESUME=1 ;;
    --core-only) CORE_ONLY=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
cd "$ROOT/src"
if [ "$RESUME" -eq 0 ]; then
  echo "[run_all] clean start: removing experiment results, checkpoints and predictions"
  rm -rf ../results/models ../results/preds ../results/results.jsonl ../results/efficiency.json \
         ../results/all_runs.csv ../results/summary.md ../results/families.jsonl ../results/ambiguity.json \
         ../results/ambiguity_airline.json ../results/ambiguity_finance.json ../results/domain_pairwise_bootstrap.json \
         ../results/domain_pairwise_bootstrap.md ../results/logs/*.jsonl ../results/logs/*.log
  mkdir -p ../results/models ../results/preds ../results/logs
fi
if [ "$RESUME" -eq 1 ]; then
  if [ -d ../results/models/mlm_final ]; then
    echo "[run_all] resume: reusing existing MLM checkpoint"
  elif [ -s ../results/results.jsonl ]; then
    echo "[run_all] ERROR: results exist but results/models/mlm_final is missing." >&2
    echo "Restore the matching checkpoint or run a clean study with: bash run_all.sh" >&2
    exit 2
  else
    python pretrain_mlm.py --steps 5000 --lr 1e-3 --dropout 0.0 --save_steps 0,100,300,1000,2500,5000
  fi
else
  python pretrain_mlm.py --steps 5000 --lr 1e-3 --dropout 0.0 --save_steps 0,100,300,1000,2500,5000
fi
python run_experiments.py --stage all --domain sst2
python efficiency.py
if [ "$CORE_ONLY" -eq 0 ]; then
  python run_experiments.py --stage all --domain finance
  python run_experiments.py --stage all --domain airline
fi
python make_assets.py
if [ "$CORE_ONLY" -eq 0 ]; then
  python cross_domain_stats.py --bootstrap 2000
fi
# Public HF checkpoints are optional because they require network access and can take substantial CPU time.
# python hf_benchmark.py --models mini,small,minilm
