#!/bin/bash
# cache_matrix_20260722: SP-MLP v2 Stage-2 (M/N head on frozen encoder).
#   Fresh head Linear(128, 2) trained on M/N labels using frozen encoder.
#   Adam(lr=1e-3, no WD) + plain CE + clip 1.0 + 100 epochs + best=test_mn_acc.
# Args: MODEL SEED GPU
# Depends on: pretrain (encoder.pt in sp_mlp output dir).
# Output: outputs/cache_matrix_20260722/runs/sp_mlp/${MODEL}_seed${SEED}/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODEL=${1:?MODEL required}
SEED=${2:?SEED required}
GPU=${3:?GPU required}
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

OUT=outputs/cache_matrix_20260722/runs/sp_mlp/${MODEL}_seed${SEED}
ENCODER_CKPT="$OUT/encoder.pt"
if [ -f "$OUT/mn_metrics.json" ]; then
  echo "SKIP mn_head: $OUT/mn_metrics.json exists"
  exit 0
fi
if [ ! -f "$ENCODER_CKPT" ]; then
  echo "[FATAL] pretrain encoder missing: $ENCODER_CKPT (run pretrain first)" >&2
  exit 2
fi
if [ ! -f "$JUDGMENTS" ]; then
  echo "[FATAL] misread judgments missing: $JUDGMENTS" >&2
  exit 2
fi
mkdir -p "$OUT"

LOG=outputs/cache_matrix_20260722/_logs/sp_mlp_mn_head_${MODEL}_seed${SEED}.log
mkdir -p "$(dirname "$LOG")"

echo "[SP-MLP mn_head] MODEL=$MODEL SEED=$SEED GPU=$GPU ckpt=$ENCODER_CKPT"
PYTHONPATH=src python scripts/train_sp_mlp.py \
  --stage mn_head \
  --model-key "$MODEL" \
  --split-assignment "$SPLIT" \
  --misread-judgments "$JUDGMENTS" \
  --cache-roots "$CACHE_ROOT" \
  --prompt-set "$PROMPT_SET" \
  --main-manifest "$MANIFEST" \
  --encoder-checkpoint "$ENCODER_CKPT" \
  --max-epochs 100 \
  --batch-size 256 \
  --lr 1e-3 \
  --embed-dim 128 \
  --device cuda:0 \
  --seed "$SEED" \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"

touch "${LOG}.done"
echo "[SP-MLP mn_head] DONE -> $OUT/mn_metrics.json"
