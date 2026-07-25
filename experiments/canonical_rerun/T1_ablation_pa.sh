#!/bin/bash
# Phase 3 PA ablation (unified): variant selected by $1.
# Args: VARIANT SEED GPU
#   VARIANT in {pa_only, pa_d, pa_s, pa_sd}
#   SEED in {20260717, 20260718, 20260719}
#   GPU in {0, 1}
# Selected yaml: configs/experiments/three_way_ablation/qwen3_vl_8b_tme_pa_ablation_${VARIANT}.yaml
# Output: outputs/canonical_rerun_v2/T1_ablation_${VARIANT}/${MODEL}_seed${SEED}/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

VARIANT=${1:?usage: $0 VARIANT SEED GPU (VARIANT in pa_only|pa_d|pa_s|pa_sd)}
SEED=${2:-20260717}
GPU=${3:-0}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src

MODEL=qwen3_vl_8b
PROTO=vt
RELATION_DATASET=outputs/v2/relation_data/$MODEL/${PROTO}/${PROTO}_main_p8_seed20260717/relation_dataset.jsonl

YAML=configs/experiments/three_way_ablation/qwen3_vl_8b_tme_pa_ablation_${VARIANT}.yaml
if [ ! -f "$YAML" ]; then
  echo "[FATAL] yaml missing: $YAML" >&2
  exit 3
fi

OUT=outputs/canonical_rerun_v2/T1_ablation_${VARIANT}/${MODEL}_seed${SEED}
mkdir -p "$OUT"

echo "[T1_ablation_${VARIANT}] SEED=$SEED GPU=$GPU yaml=$YAML"
PYTHONPATH=src python scripts/train_trajectory_encoder.py \
  --dataset "$RELATION_DATASET" \
  --config "$YAML" \
  --output-dir "$OUT" \
  --exclude-prefix ch_sims_v2: \
  --device cuda:0 \
  --encoder-type gru \
  --seed "$SEED"
echo "[T1_ablation_${VARIANT}] DONE -> $OUT/best_checkpoint.pt"
