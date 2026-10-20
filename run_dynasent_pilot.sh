#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${DYNA_SENT_ARCHIVE:-$ROOT/data/dynasent-main.zip}"
BUDGET="${BUDGET:-100}"
EVAL_SIZE="${EVAL_SIZE:-500}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-0}"
for ROUND in round1 round2; do
  python "$ROOT/src/run_dynasent_experiments.py" \
    --data "$DATA" --round "$ROUND" --backend tfidf \
    --budget "$BUDGET" --eval-size "$EVAL_SIZE" --seeds "$SEED" \
    --epochs "$EPOCHS" --out-dir "$ROOT/results/dynasent_pilot_${ROUND}"
done
