#!/usr/bin/env bash
# Driver: TME GRU wave (cache_matrix_20260722)
set -euo pipefail
cd "$(dirname "$0")/../../"

ENCODER=gru
MODELS=(gemma3_12b gemma3_4b gemma4_12b glm4_6v_flash llava_onevision_qwen2_7b \
        llava_v1_5_7b minicpm_v_2_6 minicpm_v_4_5 internvl3_5_8b phi3_5_vision \
        phi4_multimodal qwen2_5_omni_7b qwen2_5_vl_7b qwen3_5_4b qwen3_5_9b qwen3_vl_8b)
SEEDS=(20260717 20260718 20260719)

mkdir -p outputs/cache_matrix_20260722/_logs

GPU=0
for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    OUT=outputs/cache_matrix_20260722/runs/tme_${ENCODER}/${MODEL}_seed${SEED}
    if [ -f "$OUT/metrics.json" ]; then
      echo "SKIP: $OUT/metrics.json exists"
      continue
    fi
    echo "[$(date)] RUN model=$MODEL seed=$SEED encoder=$ENCODER gpu=$GPU"
    bash scripts/cache_matrix_20260722/run_tme_e2e.sh "$MODEL" "$SEED" "$GPU" "$ENCODER"
    GPU=$((1 - GPU))
  done
done
