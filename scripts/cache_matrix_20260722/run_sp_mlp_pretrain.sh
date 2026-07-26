#!/bin/bash
# cache_matrix_20260722: SP-MLP v2 Stage-1 (C/A pretrain).
#   MLP(4096 -> 128) + temp_head(128 -> 2) trained on Conflict/Aligned.
#   Adam(lr=1e-3, no WD) + plain CE + clip 1.0 + 100 epochs + best=test_ac_acc.
# Args: MODEL SEED GPU
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

if [ "$MODEL" = "internvl3_5_8b" ]; then
  CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/cache_manifests/internvl3_5_8b
else
  CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/source/$MODEL
fi

OUT=outputs/cache_matrix_20260722/runs/sp_mlp/${MODEL}_seed${SEED}
if [ -f "$OUT/encoder.pt" ]; then
  echo "SKIP pretrain: $OUT/encoder.pt exists"
  exit 0
fi
mkdir -p "$OUT"

LOG=outputs/cache_matrix_20260722/_logs/sp_mlp_pretrain_${MODEL}_seed${SEED}.log
mkdir -p "$(dirname "$LOG")"

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

echo "[SP-MLP pretrain] MODEL=$MODEL SEED=$SEED GPU=$GPU proto=$PROTO"
PYTHONPATH=src python scripts/train_sp_mlp.py \
  --stage pretrain \
  --model-key "$MODEL" \
  --split-assignment "$SPLIT" \
  --cache-roots "$CACHE_ROOT" \
  --prompt-set "$PROMPT_SET" \
  --main-manifest "$MANIFEST" \
  --max-epochs 100 \
  --batch-size 256 \
  --lr 1e-3 \
  --embed-dim 128 \
  --device cuda:0 \
  --seed "$SEED" \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"

touch "${LOG}.done"
echo "[SP-MLP pretrain] DONE -> $OUT/pretrain_metrics.json"
