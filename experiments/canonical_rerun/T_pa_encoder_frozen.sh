#!/bin/bash
# canonical_rerun_v2: Stage-1 PA encoder (GRU or LSTM, frozen C/A).
# Args: MODEL_KEY SEED GPU ENCODER_TYPE
#   MODEL_KEY in {qwen3_vl_8b, internvl3_5_8b, qwen2_5_omni_7b}
#   SEED in {20260717, 20260718, 20260719}
#   GPU in {0, 1}
#   ENCODER_TYPE in {gru, lstm}
# Selects yaml by ENCODER_TYPE:
#   gru  -> ${MODEL}_tme_pa_dstrong_bigdim_x2[_seed${SEED}].yaml
#   lstm -> ${MODEL}_tme_pa_dstrong_bigdim_x2_lstm.yaml
# Output: outputs/canonical_rerun/T${ENCODER_TYPE_NUM}_${ENCODER_TYPE}_ca_frozen/${MODEL}_seed${SEED}/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODEL=${1:-qwen3_vl_8b}
SEED=${2:-20260717}
GPU=${3:-0}
ENCODER_TYPE=${4:?usage: $0 MODEL SEED GPU ENCODER_TYPE (ENCODER_TYPE in gru|lstm)}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src

case "$ENCODER_TYPE" in
  gru) ENCODER_TYPE_NUM=1 ;;
  lstm) ENCODER_TYPE_NUM=5 ;;
  *) echo "[FATAL] ENCODER_TYPE must be gru or lstm, got: $ENCODER_TYPE" >&2; exit 2;;
esac

if [ "$MODEL" = "qwen2_5_omni_7b" ] || [ "$MODEL" = "gemma4_12b_it" ]; then
  PROTO=va
else
  PROTO=vt
fi
RELATION_DATASET=outputs/relation_data/$MODEL/${PROTO^^}/${PROTO,,}_main_p8_seed20260717/relation_dataset.jsonl

if [ "$ENCODER_TYPE" = "gru" ]; then
  YAML=configs/experiments/three_way_ablation/${MODEL}_tme_pa_dstrong_bigdim_x2.yaml
  if [ ! -f "$YAML" ]; then
    YAML=configs/experiments/three_way_ablation/${MODEL}_tme_pa_dstrong_bigdim_x2_seed${SEED}.yaml
  fi
else
  YAML=configs/experiments/three_way_ablation/${MODEL}_tme_pa_dstrong_bigdim_x2_lstm.yaml
fi
if [ ! -f "$YAML" ]; then
  echo "[FATAL] $ENCODER_TYPE yaml missing for $MODEL; expected at $YAML" >&2
  exit 3
fi

OUT=outputs/canonical_rerun/T${ENCODER_TYPE_NUM}_${ENCODER_TYPE}_ca_frozen/${MODEL}_seed${SEED}
mkdir -p "$OUT"

echo "[T_${ENCODER_TYPE}] MODEL=$MODEL SEED=$SEED GPU=$GPU yaml=$YAML"
PYTHONPATH=src python scripts/train_trajectory_encoder.py \
  --dataset "$RELATION_DATASET" \
  --config "$YAML" \
  --output-dir "$OUT" \
  --exclude-prefix ch_sims_v2: \
  --device cuda:0 \
  --encoder-type "$ENCODER_TYPE" \
  --seed "$SEED"
echo "[T_${ENCODER_TYPE}] DONE -> $OUT/best_checkpoint.pt"
