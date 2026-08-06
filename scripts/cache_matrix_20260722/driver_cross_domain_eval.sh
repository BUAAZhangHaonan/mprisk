#!/usr/bin/env bash
# Cross-domain C/A eval driver.
#
# For 15 models × 3 seeds × 3 encoders = 135 cells:
#   - Build Target relation_dataset_target.jsonl once per model (skip-if-exists)
#   - Run train_trajectory_encoder.py --load-existing --eval-dataset to score
#     Source-trained checkpoint against Target ch_sims_v2 cross-domain samples.
#   - Emit target_metrics.json next to the existing train_metrics.json.
#
# 2-GPU parallel scheduler (gpu_busy[] pattern from existing drivers).
# Skip-if-exists on target_metrics.json.
#
# Only C/A cross-domain. M/N cross-domain is BLOCKED on Target misread gen
# (separate task) and is intentionally skipped here.
#
# Honors GPUS env var (comma-separated, default "0,1").
set -euo pipefail
cd "$(dirname "$0")/../../"

source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODELS=(
  gemma3_12b gemma3_4b glm4_6v_flash llava_onevision_qwen2_7b
  minicpm_v_2_6 minicpm_v_4_5 internvl3_5_8b qwen2_5_vl_7b
  qwen3_5_4b qwen3_5_9b qwen3_vl_8b
  gemma4_12b qwen2_5_omni_7b
  phi3_5_vision llava_v1_5_7b
)
SEEDS=(20260717 20260718 20260719)
ENCODERS=(gru lstm bilstm)
IFS=',' read -ra GPUS <<< "${GPUS:-0,1}"

mkdir -p outputs/cache_matrix_20260722/_logs

LOGFILE=outputs/cache_matrix_20260722/_logs/driver_cross_domain_eval.log
echo "[$(date)] START cross_domain_eval cells=$(( ${#MODELS[@]} * ${#SEEDS[@]} * ${#ENCODERS[@]} )) gpus=${GPUS[*]}" | tee -a "$LOGFILE"

# Build all (model, seed, encoder) cells. Encoder is the outer loop so we
# cluster same-encoder cells together (cheaper GPU warmup).
CELLS=()
for ENC in "${ENCODERS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      CELLS+=("$ENC:$MODEL:$SEED")
    done
  done
done

gpu_busy=()
for g in "${GPUS[@]}"; do gpu_busy[$g]=""; done

for cell in "${CELLS[@]}"; do
    ENC="${cell%%:*}"
    rest="${cell#*:}"
    MODEL="${rest%:*}"
    SEED="${rest##*:}"

    OUT="outputs/cache_matrix_20260722/runs/ca_tme_${ENC}/${MODEL}_seed${SEED}"

    # Resolve protocol for this model (must match run_ca_tme.sh conventions).
    case "$MODEL" in
      qwen2_5_omni_7b|gemma4_12b_it|gemma4_12b|phi4_multimodal)
        PROTO=va; PROTO_UPPER=VA ;;
      *)
        PROTO=vt; PROTO_UPPER=VT ;;
    esac
    PROMPT_SET_KEY=${PROTO}_main_p8_seed20260717

    # Skip if target_metrics.json already exists with non-empty val_balanced_accuracy_ac.
    if [ -f "$OUT/target_metrics.json" ]; then
      acc=$(python3 -c "
import json
try:
    d = json.load(open('$OUT/target_metrics.json'))
    v = d.get('val_balanced_accuracy_ac')
    print('OK' if isinstance(v, (int, float)) else 'NO')
except Exception:
    print('NO')
" 2>/dev/null || echo "NO")
      if [ "$acc" = "OK" ]; then
        echo "[$(date)] SKIP $MODEL seed=$SEED enc=$ENC (target_metrics.json exists)" | tee -a "$LOGFILE"
        continue
      fi
    fi

    # Source checkpoint must exist.
    CKPT="$OUT/best_checkpoint.pt"
    if [ ! -f "$CKPT" ]; then
      echo "[$(date)] SKIP $MODEL seed=$SEED enc=$ENC (no Source checkpoint at $CKPT)" | tee -a "$LOGFILE"
      continue
    fi

    # Build Target relation_dataset_target.jsonl once per model if missing.
    TARGET_DATASET="outputs/cache_matrix_20260722/relation_data/${MODEL}/${PROTO_UPPER}/${PROMPT_SET_KEY}/relation_dataset_target.jsonl"
    if [ ! -f "$TARGET_DATASET" ]; then
      echo "[$(date)] BUILD target dataset for $MODEL" | tee -a "$LOGFILE"
      PYTHONPATH=src python3 -c "
import sys
sys.path.insert(0, 'scripts/cache_matrix_20260722')
from build_ca_datasets import _build_target_relation_dataset
_build_target_relation_dataset('$MODEL', '$PROTO', force=False)
" >> "outputs/cache_matrix_20260722/_logs/cd_build_${MODEL}.log" 2>&1 || {
        echo "[$(date)] FAIL target-build $MODEL" | tee -a "$LOGFILE"
        continue
      }
    fi
    if [ ! -s "$TARGET_DATASET" ]; then
      echo "[$(date)] SKIP $MODEL seed=$SEED enc=$ENC (empty target dataset)" | tee -a "$LOGFILE"
      continue
    fi

    # Find free GPU.
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

    echo "[$(date)] RUN model=$MODEL seed=$SEED enc=$ENC gpu=$free_gpu" | tee -a "$LOGFILE"
    (
      set -e
      mkdir -p "$OUT"
      PYTHONPATH=src CUDA_VISIBLE_DEVICES=$free_gpu python scripts/train_trajectory_encoder.py \
        --eval-dataset "$TARGET_DATASET" \
        --load-existing "$CKPT" \
        --output-dir "$OUT" \
        --device cuda:0 \
        > "outputs/cache_matrix_20260722/_logs/cd_eval_${ENC}_${MODEL}_seed${SEED}.log" 2>&1
    ) &
    gpu_busy[$free_gpu]=$!
done

wait
echo "[$(date)] DONE cross_domain_eval" | tee -a "$LOGFILE"
