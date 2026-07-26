#!/bin/bash
# canonical_rerun_v2: C2 SP-MLP v2 Stage-2 (M/N head on frozen encoder)
#   Fresh head Linear(128, 2) trained on M/N labels using frozen encoder.
#   Adam(lr=1e-3, no WD) + plain CE + clip 1.0 + 100 epochs + best=test_mn_acc.
# Args: MODEL_KEY SEED GPU
# Depends on: C1 (encoder.pt in C1_sp_mlp_v2_ca output dir).
# Output:   outputs/canonical_rerun/C2_sp_mlp_v2_mn/${MODEL}_seed${SEED}/
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
JUDGMENTS=outputs/misread/$MODEL/judgments.jsonl

MAIN_CACHE=outputs/prefill_cache/$MODEL/${PROTO,,}_main_p8_seed20260717
DELIV_CACHE=outputs/prefill_cache/$MODEL/${PROTO,,}_delivery_p8_seed20260717

C1_ENCODER=outputs/canonical_rerun/C1_sp_mlp_v2_ca/${MODEL}_seed${SEED}/encoder.pt
if [ ! -f "$C1_ENCODER" ]; then
  echo "[FATAL] C1 encoder missing: $C1_ENCODER (run C1 first)" >&2
  exit 2
fi

OUT=outputs/canonical_rerun/C2_sp_mlp_v2_mn/${MODEL}_seed${SEED}
mkdir -p "$OUT"
LOG=outputs/canonical_rerun/_logs/C2_sp_mlp_v2_${MODEL}_seed${SEED}.log
mkdir -p "$(dirname "$LOG")"

echo "[C2] MODEL=$MODEL SEED=$SEED GPU=$GPU method=sp_mlp_v2 stage=mn_head ckpt=$C1_ENCODER task=MN"
PYTHONPATH=src python scripts/train_sp_mlp.py \
  --stage mn_head \
  --model-key "$MODEL" \
  --split-assignment "$SPLIT" \
  --misread-judgments "$JUDGMENTS" \
  --cache-roots "$MAIN_CACHE" "$DELIV_CACHE" \
  --prompt-set "$PROMPT_SET" \
  --main-manifest "$MANIFEST" \
  --encoder-checkpoint "$C1_ENCODER" \
  --max-epochs 100 \
  --batch-size 256 \
  --lr 1e-3 \
  --embed-dim 128 \
  --device cuda:0 \
  --seed "$SEED" \
  --output-dir "$OUT" \
  2>&1 | tee "$LOG"
echo "[C2] DONE -> $OUT/mn_metrics.json"
