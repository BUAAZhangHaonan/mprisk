#!/usr/bin/env bash
# Target SDR (CH-SIMS v2 natural domain) for the sdr_strong extension run.
# Mirror of driver_target_sdr.sh loop body for ONE model, but checkpoint from
# runs/ca_tme_gru_sdr_strong/, thresholds/embeddings from thresholds_sdr_strong/,
# source pass into sdr_strong/<MODEL>/ and target pass into
# sdr_strong_target/<MODEL>/. Target cache root (read-only input) unchanged.
# InternVL SOURCE_CACHE_ROOT special case kept (as in the canonical driver).
#
# Args: MODEL GPU
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODEL=${1:?MODEL required}
GPU=${2:?GPU required}

TARGET_ROOT_PARENT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/target

# Resolve protocol for this model.
case "$MODEL" in
  qwen2_5_omni_7b|gemma4_12b_it|gemma4_12b|phi4_multimodal)
    PROTO=va; PROTO_UPPER=VA ;;
  *)
    PROTO=vt; PROTO_UPPER=VT ;;
esac
PROMPT_SET_KEY=${PROTO,,}_main_p8_seed20260717

# Source side mirrors run_sdr_strong.sh conventions.
if [ "$MODEL" = "internvl3_5_8b" ]; then
  SOURCE_CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/cache_manifests/internvl3_5_8b
else
  SOURCE_CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/source/$MODEL
fi
SETUP_ROOT=outputs/cache_matrix_20260722/cache_manifests/$MODEL
PROMPT_CACHE_MANIFEST=$SETUP_ROOT/prompt_cache_manifest.jsonl
PROMPT_CONDITIONED_MANIFEST=$SETUP_ROOT/prompt_conditioned_cache/$MODEL/${PROTO,,}/$PROMPT_SET_KEY/manifest.jsonl
TME_RUN=outputs/cache_matrix_20260722/runs/ca_tme_gru_sdr_strong/${MODEL}_seed20260717
ENCODER_CKPT=$TME_RUN/best_checkpoint.pt
FILTERED_CACHE_ROOT=outputs/cache_matrix_20260722/cache_manifests_filtered/$MODEL
FILTERED_MANIFEST=$FILTERED_CACHE_ROOT/unified_full_cache_manifest.json
PROMPT_SET=configs/prompts/equiv_sets/${PROTO,,}_main_p8_seed20260717.yaml
SPLIT=data/processed/manifests/splits/representation_v1/representation_split_assignment_v1_${PROTO,,}.jsonl
MANIFEST=data/processed/manifests/protocol_manifests_merged/${PROTO,,}_merged_primary.jsonl

# Thresholds: ONLY the sdr_strong calibration. No canonical fallback.
THRESHOLDS_ARG=outputs/cache_matrix_20260722/thresholds_sdr_strong/${MODEL}/thresholds.json

# Target cache root + output dir.
TARGET_CACHE_ROOT=${TARGET_ROOT_PARENT}/$MODEL
TARGET_OUT=outputs/cache_matrix_20260722/sdr_strong_target/$MODEL
TARGET_PATTERN_FILE=$TARGET_OUT/outputs/states/$MODEL/$PROTO_UPPER/$PROMPT_SET_KEY/tme_proxy_anchor_v1/state_patterns.jsonl

LOG=outputs/cache_matrix_20260722/_logs/sdr_strong_target_${MODEL}.log
mkdir -p "$(dirname "$LOG")"

# Skip-if-exists on Target state_patterns.jsonl.
if [ -f "$TARGET_PATTERN_FILE" ]; then
  echo "SKIP: $TARGET_PATTERN_FILE exists"
  exit 0
fi

# Required artifacts.
if [ ! -f "$THRESHOLDS_ARG" ]; then
  echo "[FATAL] no sdr_strong thresholds for $MODEL: $THRESHOLDS_ARG" | tee -a "$LOG"
  mkdir -p "$TARGET_OUT"
  echo "MISSING_STRONG_THRESHOLDS" > "$TARGET_OUT/MISSING_DEPENDENCY"
  exit 2
fi
if [ ! -f "$ENCODER_CKPT" ]; then
  echo "[FATAL] no sdr_strong Source ckpt at $ENCODER_CKPT" | tee -a "$LOG"
  exit 2
fi
if [ ! -d "$TARGET_CACHE_ROOT" ]; then
  echo "[FATAL] no target cache at $TARGET_CACHE_ROOT" | tee -a "$LOG"
  exit 2
fi
if [ ! -f "$FILTERED_MANIFEST" ]; then
  echo "[FATAL] no filtered cache manifest: $FILTERED_MANIFEST" | tee -a "$LOG"
  exit 2
fi

# Reuse sdr_strong calibration embeddings for Source pass identity binding.
CALIB_EMBEDDING_MANIFEST=outputs/cache_matrix_20260722/thresholds_sdr_strong/$MODEL/outputs/embeddings/$MODEL/$PROTO_UPPER/$PROMPT_SET_KEY/tme_proxy_anchor_v1/spherical_embedding_manifest.jsonl
EMBEDDING_ARG=""
if [ -f "$CALIB_EMBEDDING_MANIFEST" ]; then
  EMBEDDING_ARG="--embedding-manifest-path $CALIB_EMBEDDING_MANIFEST"
fi

echo "[TGT-SDR-STRONG] model=$MODEL proto=$PROTO gpu=$GPU"
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
  --output-root "outputs/cache_matrix_20260722/sdr_strong/$MODEL" \
  --thresholds "$THRESHOLDS_ARG" \
  --checkpoint "$ENCODER_CKPT" \
  --target-cache-root "$TARGET_CACHE_ROOT" \
  --target-output-dir "$TARGET_OUT" \
  $EMBEDDING_ARG \
  > "$LOG" 2>&1

echo "[TGT-SDR-STRONG] DONE -> $TARGET_PATTERN_FILE"
