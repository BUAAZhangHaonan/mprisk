#!/usr/bin/env bash
# C/A TME training (GRU encoder + SDR hinge aux loss).
# Usage: run_ca_tme.sh MODEL SEED GPU [CONFIG_NAME]
#   CONFIG_NAME defaults to ${MODEL}_tme_sdr (yaml under configs/experiments/cache_matrix_20260722/)
# Output: outputs/cache_matrix_20260722/runs/ca_tme/${MODEL}_seed${SEED}/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODEL=${1:?MODEL required}
SEED=${2:?SEED required}
GPU=${3:?GPU required}
CONFIG_NAME=${4:-${MODEL}_tme_sdr}

# Resolve protocol (must match run_tme_e2e.sh conventions)
case "$MODEL" in
  qwen2_5_omni_7b|gemma4_12b_it|gemma4_12b|phi4_multimodal)
    PROTO=va; PROTO_UPPER=VA ;;
  *)
    PROTO=vt; PROTO_UPPER=VT ;;
esac

DATASET=outputs/cache_matrix_20260722/relation_data/${MODEL}/${PROTO_UPPER}/${PROTO}_main_p8_seed20260717/relation_dataset.jsonl
CONFIG=configs/experiments/cache_matrix_20260722/${CONFIG_NAME}.yaml
OUT=outputs/cache_matrix_20260722/runs/ca_tme/${MODEL}_seed${SEED}
LOG=outputs/cache_matrix_20260722/_logs/ca_tme_${MODEL}_seed${SEED}.log

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

if [ -f "$OUT/best_checkpoint.pt" ] || [ -f "$OUT/metrics.json" ]; then
  echo "SKIP: $OUT exists"
  exit 0
fi

[ -f "$DATASET" ] || { echo "[FATAL] dataset missing: $DATASET"; exit 2; }
[ -f "$CONFIG" ]   || { echo "[FATAL] config missing: $CONFIG";   exit 2; }

echo "[CA-TME] MODEL=$MODEL SEED=$SEED proto=$PROTO gpu=$GPU config=$CONFIG_NAME"
mkdir -p "$OUT"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU python scripts/train_trajectory_encoder.py \
  --dataset "$DATASET" \
  --config "$CONFIG" \
  --output-dir "$OUT" \
  --device cuda:0 \
  --encoder-type gru \
  --seed "$SEED" \
  > "$LOG" 2>&1

EXIT=$?
if [ $EXIT -eq 0 ]; then
  touch "${LOG}.done"
fi
exit $EXIT
