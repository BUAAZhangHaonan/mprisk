#!/usr/bin/env bash
# M/N TME-E2E end-to-end training (GRU encoder + CE loss, no C/A dependency).
# Usage: run_mn_tme_e2e.sh MODEL SEED GPU
# Output: outputs/cache_matrix_20260722/runs/mn_tme_e2e/<model>_seed<seed>/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODEL=${1:?MODEL required}
SEED=${2:?SEED required}
GPU=${3:?GPU required}
ENCODER_TYPE=gru
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src

case "$MODEL" in
  qwen2_5_omni_7b|gemma4_12b_it|gemma4_12b|phi4_multimodal)
    PROTO=va ;;
  *)
    PROTO=vt ;;
esac

SPLIT=outputs/cache_matrix_20260722/split_assignments/${PROTO,,}.jsonl
MANIFEST=data/processed/manifests/protocol_manifests_merged/${PROTO,,}_merged_primary.jsonl
PROMPT_SET=configs/prompts/equiv_sets/${PROTO,,}_main_p8_seed20260717.yaml
JUDGMENTS=outputs/misread/$MODEL/judgments.jsonl

if [ "$MODEL" = "internvl3_5_8b" ]; then
  CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/cache_manifests/internvl3_5_8b
else
  CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/source/$MODEL
fi

OUT=outputs/cache_matrix_20260722/runs/mn_tme_e2e/${MODEL}_seed${SEED}
if [ -f "$OUT/metrics.json" ]; then
  echo "SKIP: $OUT/metrics.json exists"
  exit 0
fi
mkdir -p "$OUT"

LOG=outputs/cache_matrix_20260722/_logs/mn_tme_e2e_${MODEL}_seed${SEED}.log
mkdir -p "$(dirname "$LOG")"

[ -f "$JUDGMENTS" ] || { echo "[FATAL] misread judgments missing: $JUDGMENTS" >&2; exit 2; }
[ -f "$SPLIT" ]     || { echo "[FATAL] split missing: $SPLIT" >&2; exit 2; }
[ -f "$MANIFEST" ]  || { echo "[FATAL] manifest missing: $MANIFEST" >&2; exit 2; }
[ -f "$CACHE_ROOT/manifest.jsonl" ] || { echo "[FATAL] cache manifest missing: $CACHE_ROOT/manifest.jsonl" >&2; exit 2; }

echo "[MN-TME-E2E] MODEL=$MODEL SEED=$SEED GPU=$GPU encoder=$ENCODER_TYPE proto=$PROTO"
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
echo "[MN-TME-E2E] DONE -> $OUT/metrics.json"
