#!/bin/bash
# canonical_rerun_v2: C5 TME e2e v3-B (single-stage M/N)
#   Shared GRU condition_encoder + 3-cond concat (384-d) + MLP head [384->32->2].
#   AdamW(lr=5e-4, wd=1e-4) + plain CE + clip 1.0 + 100 epochs + best=test_mn_acc.
#   Training from scratch (no warm start, 20260723).
# Args: MODEL_KEY SEED GPU
# No dependencies (training from scratch).
# Output:   outputs/canonical_rerun_v2/C5_tme_v3b_e2e_mn/${MODEL}_seed${SEED}/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODEL=${1:-qwen3_vl_8b}
SEED=${2:-20260717}
GPU=${3:-0}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src

if [ "$MODEL" = "qwen2_5_omni_7b" ] || [ "$MODEL" = "gemma4_12b_it" ]; then
  PROTO=va
else
  PROTO=vt
fi
SPLIT=data/processed/manifests/splits/representation_v1/representation_split_assignment_v1_${PROTO,,}.jsonl
MANIFEST=data/processed/manifests/protocol_manifests_merged/${PROTO,,}_merged_primary.jsonl
PROMPT_SET=configs/prompts/equiv_sets/${PROTO,,}_main_p8_seed20260717.yaml
JUDGMENTS=outputs/v2/misread/$MODEL/judgments.jsonl

MAIN_CACHE=outputs/prefill_cache/$MODEL/${PROTO,,}_main_p8_seed20260717
DELIV_CACHE=outputs/prefill_cache/$MODEL/${PROTO,,}_delivery_p8_seed20260717

# Training from scratch (20260723): all models from scratch, no warm start.
# The previous --tme-pa-checkpoint path leaked T1 PA features; removed for
# a clean single-stage MN baseline.
T1_ARG=""
echo "[C5] from scratch (no warm start)"

OUT=outputs/canonical_rerun_v2/C5_tme_v3b_e2e_mn/${MODEL}_seed${SEED}
mkdir -p "$OUT"
LOG=outputs/canonical_rerun_v2/_logs/C5_tme_v3b_${MODEL}_seed${SEED}.log
mkdir -p "$(dirname "$LOG")"

echo "[C5] MODEL=$MODEL SEED=$SEED GPU=$GPU method=tme_v3b stage=e2e task=MN"
PYTHONPATH=src python scripts/train_tme_e2e.py \
  --task misread \
  --model-key "$MODEL" \
  --split-assignment "$SPLIT" \
  --misread-judgments "$JUDGMENTS" \
  --cache-roots "$MAIN_CACHE" "$DELIV_CACHE" \
  --prompt-set "$PROMPT_SET" \
  --main-manifest "$MANIFEST" \
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
  $T1_ARG \
  2>&1 | tee "$LOG"
echo "[C5] DONE -> $OUT/metrics.json"
