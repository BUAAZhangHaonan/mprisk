#!/usr/bin/env bash
# Driver: TME BiLSTM wave (cache_matrix_20260722) — parallel 2-GPU edition
set -euo pipefail
cd "$(dirname "$0")/../../"

ENC=bilstm
MODELS=(gemma3_12b gemma3_4b gemma4_12b glm4_6v_flash llava_onevision_qwen2_7b \
        llava_v1_5_7b minicpm_v_2_6 minicpm_v_4_5 internvl3_5_8b phi3_5_vision \
        phi4_multimodal qwen2_5_omni_7b qwen2_5_vl_7b qwen3_5_4b qwen3_5_9b qwen3_vl_8b)
SEEDS=(20260717 20260718 20260719)
GPUS=(0 1)  # GPU list, can be overridden via env
MAX_JOBS=${#GPUS[@]}

mkdir -p outputs/cache_matrix_20260722/_logs

# Build all (model, seed) cells
CELLS=()
for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    CELLS+=("$MODEL:$SEED")
  done
done

# Parallel runner: assign free GPU to next pending cell
LOGFILE=outputs/cache_matrix_20260722/_logs/driver_tme_${ENC}.log
echo "[$(date)] START encoder=$ENC cells=${#CELLS[@]} gpus=${GPUS[*]}" | tee -a "$LOGFILE"

gpu_busy=()  # index = gpu, value = pid using it
for g in "${GPUS[@]}"; do gpu_busy[$g]=""; done

for cell in "${CELLS[@]}"; do
  MODEL="${cell%:*}"
  SEED="${cell#*:}"

  OUT="outputs/cache_matrix_20260722/runs/tme_${ENC}/${MODEL}_seed${SEED}"
  if [ -f "$OUT/metrics.json" ]; then
    echo "[$(date)] SKIP $MODEL seed=$SEED (already done)" | tee -a "$LOGFILE"
    continue
  fi

  # Find a free GPU
  while true; do
    for g in "${GPUS[@]}"; do
      if [ -z "${gpu_busy[$g]}" ] || ! kill -0 "${gpu_busy[$g]}" 2>/dev/null; then
        gpu_busy[$g]=""
        free_gpu=$g
        break 2
      fi
    done
    sleep 5
  done

  echo "[$(date)] RUN model=$MODEL seed=$SEED encoder=$ENC gpu=$free_gpu" | tee -a "$LOGFILE"
  bash scripts/cache_matrix_20260722/run_tme_e2e.sh "$MODEL" "$SEED" "$free_gpu" "$ENC" \
    > "outputs/cache_matrix_20260722/_logs/tme_${ENC}_${MODEL}_seed${SEED}.log" 2>&1 &
  gpu_busy[$free_gpu]=$!
done

# Wait for all
wait
echo "[$(date)] DONE encoder=$ENC" | tee -a "$LOGFILE"
