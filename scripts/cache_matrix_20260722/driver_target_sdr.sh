#!/usr/bin/env bash
# Target SDR sweep driver.
#
# For each of the 15 C/A models (skip phi4_multimodal):
#   - Run run_core_sdr_pipeline.py with --target-cache-root + --target-output-dir
#   - Source SDR pass re-runs (idempotent; reuses calibration embeddings)
#   - Target SDR pass exports Target embeddings from Source checkpoint,
#     computes Target SDR scores, assigns Target state patterns.
#   - Output to outputs/cache_matrix_20260722/sdr_target/<model>/
#
# Skip-if-exists on Target state_patterns.jsonl.
# Serial on a single GPU (default GPU 0; override via GPU env var).
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
GPU=${GPU:-0}
TARGET_ROOT_PARENT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/target

mkdir -p outputs/cache_matrix_20260722/_logs
LOGFILE=outputs/cache_matrix_20260722/_logs/driver_target_sdr.log
echo "[$(date)] START target_sdr models=${#MODELS[@]} gpu=$GPU" | tee -a "$LOGFILE"

for MODEL in "${MODELS[@]}"; do
  # Resolve protocol for this model.
  case "$MODEL" in
    qwen2_5_omni_7b|gemma4_12b_it|gemma4_12b|phi4_multimodal)
      PROTO=va; PROTO_UPPER=VA ;;
    *)
      PROTO=vt; PROTO_UPPER=VT ;;
  esac
  PROMPT_SET_KEY=${PROTO}_main_p8_seed20260717

  # Source side mirrors run_sdr.sh conventions.
  if [ "$MODEL" = "internvl3_5_8b" ]; then
    SOURCE_CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/cache_manifests/internvl3_5_8b
  else
    SOURCE_CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/source/$MODEL
  fi
  SETUP_ROOT=outputs/cache_matrix_20260722/cache_manifests/$MODEL
  PROMPT_CACHE_MANIFEST=$SETUP_ROOT/prompt_cache_manifest.jsonl
  PROMPT_CONDITIONED_MANIFEST=$SETUP_ROOT/prompt_conditioned_cache/$MODEL/${PROTO,,}/$PROMPT_SET_KEY/manifest.jsonl
  TME_RUN=outputs/cache_matrix_20260722/runs/ca_tme_gru/${MODEL}_seed20260717
  ENCODER_CKPT=$TME_RUN/best_checkpoint.pt
  FILTERED_CACHE_ROOT=outputs/cache_matrix_20260722/cache_manifests_filtered/$MODEL
  FILTERED_MANIFEST=$FILTERED_CACHE_ROOT/unified_full_cache_manifest.json
  PROMPT_SET=configs/prompts/equiv_sets/${PROTO,,}_main_p8_seed20260717.yaml
  SPLIT=data/processed/manifests/splits/representation_v1/representation_split_assignment_v1_${PROTO,,}.jsonl
  MANIFEST=data/processed/manifests/protocol_manifests_merged/${PROTO,,}_merged_primary.jsonl

  # Resolve Source thresholds (mirror run_sdr.sh).
  THRESHOLDS_DIR=outputs/cache_matrix_20260722/thresholds/${MODEL}/thresholds.json
  THRESHOLDS_FILE=outputs/cache_matrix_20260722/thresholds/${MODEL}.json
  LEGACY_THRESHOLDS=outputs/state_analysis/${MODEL}/thresholds.json
  if [ -f "$THRESHOLDS_DIR" ]; then
    THRESHOLDS_ARG=$THRESHOLDS_DIR
  elif [ -f "$THRESHOLDS_FILE" ]; then
    THRESHOLDS_ARG=$THRESHOLDS_FILE
  elif [ -f "$LEGACY_THRESHOLDS" ]; then
    THRESHOLDS_ARG=$LEGACY_THRESHOLDS
  else
    echo "[$(date)] SKIP $MODEL (no thresholds)" | tee -a "$LOGFILE"
    continue
  fi

  # Target cache root + output dir.
  TARGET_CACHE_ROOT=${TARGET_ROOT_PARENT}/$MODEL
  if [ ! -d "$TARGET_CACHE_ROOT" ]; then
    echo "[$(date)] SKIP $MODEL (no target cache at $TARGET_CACHE_ROOT)" | tee -a "$LOGFILE"
    continue
  fi
  TARGET_OUT=outputs/cache_matrix_20260722/sdr_target/$MODEL
  TARGET_PATTERN_FILE=$TARGET_OUT/outputs/states/$MODEL/$PROTO_UPPER/$PROMPT_SET_KEY/tme_proxy_anchor_v1/state_patterns.jsonl

  # Skip-if-exists on Target state_patterns.jsonl.
  if [ -f "$TARGET_PATTERN_FILE" ]; then
    echo "[$(date)] SKIP $MODEL (Target state_patterns.jsonl exists)" | tee -a "$LOGFILE"
    continue
  fi

  # Required source artifacts.
  if [ ! -f "$ENCODER_CKPT" ]; then
    echo "[$(date)] SKIP $MODEL (no Source ckpt at $ENCODER_CKPT)" | tee -a "$LOGFILE"
    continue
  fi
  if [ ! -f "$FILTERED_MANIFEST" ]; then
    echo "[$(date)] SKIP $MODEL (no filtered cache manifest)" | tee -a "$LOGFILE"
    continue
  fi

  # Reuse Source calibration embeddings for Source pass identity binding.
  CALIB_EMBEDDING_MANIFEST=outputs/cache_matrix_20260722/thresholds/$MODEL/outputs/embeddings/$MODEL/$PROTO_UPPER/$PROMPT_SET_KEY/tme_proxy_anchor_v1/spherical_embedding_manifest.jsonl
  EMBEDDING_ARG=""
  if [ -f "$CALIB_EMBEDDING_MANIFEST" ]; then
    EMBEDDING_ARG="--embedding-manifest-path $CALIB_EMBEDDING_MANIFEST"
  fi

  echo "[$(date)] RUN target_sdr model=$MODEL proto=$PROTO gpu=$GPU" | tee -a "$LOGFILE"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU python scripts/run_core_sdr_pipeline.py \
    --model-key "$MODEL" \
    --protocol "${PROTO,,}" \
    --prompt-set-key "$PROMPT_SET_KEY" \
    --repr-key tme_proxy_anchor_v1 \
    --manifest-paths "$MANIFEST" \
    --full-cache-root "$FILTERED_CACHE_ROOT" \
    --cache-manifest-path "unified_full_cache_manifest.json" \
    --prompt-cache-manifest "$PROMPT_CACHE_MANIFEST" \
    --prompt-conditioned-cache-manifest "$PROMPT_CONDITIONED_MANIFEST" \
    --prompt-set "$PROMPT_SET" \
    --split-assignment "$SPLIT" \
    --output-root "outputs/cache_matrix_20260722/sdr/$MODEL" \
    --thresholds "$THRESHOLDS_ARG" \
    --checkpoint "$ENCODER_CKPT" \
    --target-cache-root "$TARGET_CACHE_ROOT" \
    --target-output-dir "$TARGET_OUT" \
    $EMBEDDING_ARG \
    > "outputs/cache_matrix_20260722/_logs/sdr_target_${MODEL}.log" 2>&1 \
    || {
      echo "[$(date)] FAIL target_sdr $MODEL" | tee -a "$LOGFILE"
      continue
    }
  echo "[$(date)] DONE target_sdr $MODEL -> $TARGET_PATTERN_FILE" | tee -a "$LOGFILE"
done

echo "[$(date)] DONE target_sdr all models" | tee -a "$LOGFILE"
