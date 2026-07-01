#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PLANTUCD_DATA_PATH="${PLANTUCD_DATA_PATH:-data_splits/plantucd_rl_1127.json}"
export MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
export OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-120}"
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-3}"
export STRUCTURED_OUTPUT_ATTEMPTS="${STRUCTURED_OUTPUT_ATTEMPTS:-3}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

BATCH_SIZE="${BATCH_SIZE:-50}"
START="${START:-0}"
END="${END:-1126}"
OUT_DIR="plan_rag_extract_gpt4omini_plantucd_rl1127"
LOG_DIR="logs/rl1127"

mkdir -p "$LOG_DIR"

echo "Data path: $PLANTUCD_DATA_PATH"
echo "Output dir: $OUT_DIR"
echo "Index range: $START..$END, batch size: $BATCH_SIZE"
echo "Model: $MODEL_NAME"

for ((s=START; s<=END; s+=BATCH_SIZE)); do
  e=$((s + BATCH_SIZE - 1))
  if [ "$e" -gt "$END" ]; then
    e="$END"
  fi

  echo
  echo "===== Running batch $s..$e ====="
  python main.py \
    --exp plan_rag_extract \
    --model gpt4omini \
    --dataset plantucd_rl1127 \
    --start_index "$s" \
    --end_index "$e" \
    --gpus 0 2>&1 | tee "$LOG_DIR/batch_${s}_${e}.log"

  python scripts/check_progress.py --output-dir "$OUT_DIR" --expected 1127 || true
done

echo
echo "All requested batches finished."
python scripts/check_progress.py --output-dir "$OUT_DIR" --expected 1127 || true
