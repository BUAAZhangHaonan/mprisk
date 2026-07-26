#!/bin/bash
# cache_matrix_20260722: TME e2e v3-B driver per (model, seed, encoder).
#   Cloned from experiments/canonical_rerun/C5_tme_e2e_mn.sh but pointed at
#   the cache_matrix_20260722 unified manifests.
#
#   Shared encoder (bilstm|lstm|gru) + 3-cond concat + MLP head [384->32->2].
#   AdamW(lr=5e-4, wd=1e-4) + plain CE + clip 1.0 + 100 epochs + best=test_mn_acc.
#   Training from scratch (no warm start).
#
# Args: MODEL SEED GPU ENCODER_TYPE
# Output: outputs/cache_matrix_20260722/runs/tme_<enc>/${MODEL}_seed${SEED}/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODEL=${1:?MODEL required}
SEED=${2:?SEED required}
GPU=${3:?GPU required}
ENCODER_TYPE=${4:-bilstm}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src

if [ "$MODEL" = "qwen2_5_omni_7b" ] || [ "$MODEL" = "gemma4_12b_it" ] || [ "$MODEL" = "gemma4_12b" ] || [ "$MODEL" = "phi4_multimodal" ]; then
  PROTO=va
else
  PROTO=vt
fi

SPLIT=outputs/cache_matrix_20260722/split_assignments/${PROTO,,}.jsonl
MANIFEST=data/processed/manifests/protocol_manifests_merged/${PROTO,,}_merged_primary.jsonl
PROMPT_SET=configs/prompts/equiv_sets/${PROTO,,}_main_p8_seed20260717.yaml
JUDGMENTS=outputs/misread/$MODEL/judgments.jsonl

if [ "$MODEL" = "internvl3_5_8b" ]; then
  CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/cache_manifests/internvl3_5_8b
else
  CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/source/$MODEL
fi

OUT=outputs/cache_matrix_20260722/runs/tme_${ENCODER_TYPE}/${MODEL}_seed${SEED}
if [ -f "$OUT/metrics.json" ]; then
  echo "SKIP: $OUT/metrics.json exists"
  exit 0
fi
mkdir -p "$OUT"

LOG=outputs/cache_matrix_20260722/_logs/tme_${ENCODER_TYPE}_${MODEL}_seed${SEED}.log
mkdir -p "$(dirname "$LOG")"

if [ ! -f "$JUDGMENTS" ]; then
  echo "[FATAL] misread judgments missing: $JUDGMENTS" >&2
  exit 2
fi
if [ ! -f "$SPLIT" ]; then
  echo "[FATAL] split missing: $SPLIT" >&2
  exit 2
fi
if [ ! -f "$MANIFEST" ]; then
  echo "[FATAL] manifest missing: $MANIFEST" >&2
  exit 2
fi
if [ ! -f "$CACHE_ROOT/manifest.jsonl" ]; then
  echo "[FATAL] cache manifest missing: $CACHE_ROOT/manifest.jsonl" >&2
  exit 2
fi

echo "[TME] MODEL=$MODEL SEED=$SEED GPU=$GPU encoder=$ENCODER_TYPE proto=$PROTO"
PYTHONPATH=src python scripts/train_tme_e2e.py \
  --task misread \
  --model-key "$MODEL" \
  --split-assignment "$SPLIT" \
  --misread-judgments "$JUDGMENTS" \
  --cache-roots "$CACHE_ROOT" \
  --prompt-set "$PROMPT_SET" \
  --main-manifest "$MANIFEST" \
  --encoder-type "$ENCODER_TYPE" \
  --max-epochs 100 \
  --batch-size 32 \
  --device cuda:0 \
  --seed "$SEED" \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --dropout 0.3 \
  --sequence-hidden-dim 256 \
  --embed-dim 128 \
  --head-hidden-dim 32 \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"

touch "${LOG}.done"
echo "[TME] DONE -> $OUT/metrics.json"
