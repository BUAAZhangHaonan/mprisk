#!/bin/bash
# canonical_rerun_v2: C1 SP-MLP v2 Stage-1 (C/A pretrain)
#   MLP(4096 -> 128) + temp_head(128 -> 2) trained on Conflict/Aligned.
#   M12 LAST LAYER hidden state (4096-d) from prefill cache.
#   Adam(lr=1e-3, no WD) + plain CE + clip 1.0 + 100 epochs + best=test_ac_acc.
# Args: MODEL_KEY SEED GPU
#   MODEL_KEY in {qwen3_vl_8b, internvl3_5_8b, qwen2_5_omni_7b}
#   SEED in {20260717, 20260718, 20260719}
#   GPU in {0, 1}
# Output:   outputs/canonical_rerun_v2/C1_sp_mlp_v2_ca/${MODEL}_seed${SEED}/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source "${CONDA_PREFIX:-/opt/miniconda3}/../etc/profile.d/conda.sh" 2>/dev/null || true
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

MAIN_CACHE=outputs/prefill_cache/$MODEL/${PROTO,,}_main_p8_seed20260717
DELIV_CACHE=outputs/prefill_cache/$MODEL/${PROTO,,}_delivery_p8_seed20260717

OUT=outputs/canonical_rerun_v2/C1_sp_mlp_v2_ca/${MODEL}_seed${SEED}
mkdir -p "$OUT"
LOG=outputs/canonical_rerun_v2/_logs/C1_sp_mlp_v2_${MODEL}_seed${SEED}.log
mkdir -p "$(dirname "$LOG")"

echo "[C1] MODEL=$MODEL SEED=$SEED GPU=$GPU method=sp_mlp_v2 stage=pretrain task=CA"
PYTHONPATH=src python scripts/train_sp_mlp.py \
  --stage pretrain \
  --model-key "$MODEL" \
  --split-assignment "$SPLIT" \
  --cache-roots "$MAIN_CACHE" "$DELIV_CACHE" \
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
echo "[C1] DONE -> $OUT/pretrain_metrics.json"
