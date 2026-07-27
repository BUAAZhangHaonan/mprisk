#!/usr/bin/env bash
# Driver: SDR pipeline for 16 models (cache_matrix_20260722).
# Parallel-2-GPU scheduler (SDR is GPU-light, but we keep 2 for safety).
# Honors GPUS env var (comma-separated, default "0,1").
set -euo pipefail
cd "$(dirname "$0")/../../"

source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODELS=(gemma3_12b gemma3_4b gemma4_12b glm4_6v_flash llava_onevision_qwen2_7b \
        llava_v1_5_7b minicpm_v_2_6 minicpm_v_4_5 internvl3_5_8b phi3_5_vision \
        phi4_multimodal qwen2_5_omni_7b qwen2_5_vl_7b qwen3_5_4b qwen3_5_9b qwen3_vl_8b)
IFS=',' read -ra GPUS <<< "${GPUS:-0,1}"

mkdir -p outputs/cache_matrix_20260722/_logs

LOGFILE=outputs/cache_matrix_20260722/_logs/driver_sdr.log
echo "[$(date)] START driver_sdr models=${#MODELS[@]} gpus=${GPUS[*]}" | tee -a "$LOGFILE"

gpu_busy=()
for g in "${GPUS[@]}"; do gpu_busy[$g]=""; done

for MODEL in "${MODELS[@]}"; do
  OUT="outputs/cache_matrix_20260722/sdr/$MODEL"
  STATE_PATTERN="$OUT/outputs/states/$MODEL"
  # Skip if any state_patterns.jsonl already exists (search by glob below)
  if compgen -G "$OUT/outputs/states/$MODEL/*/*/tme_proxy_anchor_v1/state_patterns.jsonl" > /dev/null; then
    echo "[$(date)] SKIP $MODEL (state_patterns.jsonl exists)" | tee -a "$LOGFILE"
    continue
  fi
  if [ -f "$OUT/MISSING_DEPENDENCY" ]; then
    echo "[$(date)] SKIP $MODEL (MISSING_DEPENDENCY marker)" | tee -a "$LOGFILE"
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

  echo "[$(date)] RUN model=$MODEL driver=sdr gpu=$free_gpu" | tee -a "$LOGFILE"
  bash scripts/cache_matrix_20260722/run_sdr.sh "$MODEL" "$free_gpu" \
    > "outputs/cache_matrix_20260722/_logs/sdr_${MODEL}.driver.log" 2>&1 &
  gpu_busy[$free_gpu]=$!
done

wait
echo "[$(date)] DONE driver_sdr" | tee -a "$LOGFILE"
