#!/usr/bin/env bash
# Run small matched-budget support comparison. Primary soft-target study remains separate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$ROOT/data/dynasent-main.zip}"
OUT="${OUT:-$ROOT/results/supporting_model_comparison}"
BUDGETS=(25 50 100 200 300)
SEEDS=(0 1 2)
for round in round1 round2; do
  for backend in tfidf tiny-bert; do
    for budget in "${BUDGETS[@]}"; do
      python "$ROOT/src/run_dynasent_experiments.py" \
        --data "$DATA" --round "$round" --backend "$backend" \
        --checkpoint "$ROOT/results/models/mlm_final" --budget "$budget" \
        --eval-size 0 --seeds "${SEEDS[@]}" --epochs 10 --device cpu \
        --out-dir "$OUT/$round/$backend/budget_$budget"
    done
  done
done
python "$ROOT/scripts/analyze_dynasent_supporting_matrix.py" --results-root "$OUT"
