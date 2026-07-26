#!/usr/bin/env bash
# Driver: T-LSTM wave (cache_matrix_20260722) — 2-stage pretrain+mn_head per cell.
# Parallel-2-GPU scheduler. Honors GPUS env var (comma-separated, default "0,1").
set -euo pipefail
cd "$(dirname "$0")/../../"

source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

METHOD=t_lstm
MODELS=(gemma3_12b gemma3_4b gemma4_12b glm4_6v_flash llava_onevision_qwen2_7b \
        llava_v1_5_7b minicpm_v_2_6 minicpm_v_4_5 internvl3_5_8b phi3_5_vision \
        phi4_multimodal qwen2_5_omni_7b qwen2_5_vl_7b qwen3_5_4b qwen3_5_9b qwen3_vl_8b)
SEEDS=(20260717 20260718 20260719)
IFS=',' read -ra GPUS <<< "${GPUS:-0,1}"
MAX_JOBS=${#GPUS[@]}

mkdir -p outputs/cache_matrix_20260722/_logs

CELLS=()
for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    CELLS+=("$MODEL:$SEED")
  done
done

LOGFILE=outputs/cache_matrix_20260722/_logs/driver_${METHOD}.log
echo "[$(date)] START method=$METHOD cells=${#CELLS[@]} gpus=${GPUS[*]}" | tee -a "$LOGFILE"

gpu_busy=()
for g in "${GPUS[@]}"; do gpu_busy[$g]=""; done

for cell in "${CELLS[@]}"; do
  MODEL="${cell%:*}"
  SEED="${cell#*:}"

  OUT="outputs/cache_matrix_20260722/runs/${METHOD}/${MODEL}_seed${SEED}"
  if [ -f "$OUT/mn_metrics.json" ]; then
    echo "[$(date)] SKIP $MODEL seed=$SEED (mn_head done)" | tee -a "$LOGFILE"
    continue
  fi

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

  echo "[$(date)] RUN model=$MODEL seed=$SEED method=$METHOD gpu=$free_gpu" | tee -a "$LOGFILE"
  (
    bash scripts/cache_matrix_20260722/run_t_lstm_pretrain.sh "$MODEL" "$SEED" "$free_gpu" \
      > "outputs/cache_matrix_20260722/_logs/t_lstm_pretrain_${MODEL}_seed${SEED}.driver.log" 2>&1
    if [ -f "$OUT/encoder.pt" ] && [ ! -f "$OUT/mn_metrics.json" ]; then
      bash scripts/cache_matrix_20260722/run_t_lstm_mn_head.sh "$MODEL" "$SEED" "$free_gpu" \
        > "outputs/cache_matrix_20260722/_logs/t_lstm_mn_head_${MODEL}_seed${SEED}.driver.log" 2>&1
    fi
  ) &
  gpu_busy[$free_gpu]=$!
done

wait
echo "[$(date)] DONE method=$METHOD" | tee -a "$LOGFILE"
