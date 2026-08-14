#!/usr/bin/env bash
# C/A TME training, sdr_strong variant (sdr_aux_weight=2.0 + state supervision).
# Mirror of run_ca_tme.sh (GRU branch) but reads <MODEL>_tme_sdr_strong.yaml and
# writes to runs/ca_tme_gru_sdr_strong/. Never touches canonical runs/ca_tme_gru/.
# Usage: run_ca_tme_sdr_strong.sh MODEL SEED GPU
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODEL=${1:?MODEL required}
SEED=${2:?SEED required}
GPU=${3:?GPU required}

# Resolve protocol (must match run_ca_tme.sh conventions)
case "$MODEL" in
  qwen2_5_omni_7b|gemma4_12b_it|gemma4_12b|phi4_multimodal)
    PROTO=va; PROTO_UPPER=VA ;;
  *)
    PROTO=vt; PROTO_UPPER=VT ;;
esac

DATASET=outputs/cache_matrix_20260722/relation_data/${MODEL}/${PROTO_UPPER}/${PROTO}_main_p8_seed20260717/relation_dataset.jsonl
CONFIG=configs/experiments/cache_matrix_20260722/${MODEL}_tme_sdr_strong.yaml
OUT=outputs/cache_matrix_20260722/runs/ca_tme_gru_sdr_strong/${MODEL}_seed${SEED}
LOG=outputs/cache_matrix_20260722/_logs/ca_tme_gru_sdr_strong_${MODEL}_seed${SEED}.log

mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"

# SKIP only when a real completion signature is present (same policy as run_ca_tme.sh).
if [ -f "$OUT/train_metrics.json" ]; then
  READ_RESULT=$(python3 - <<PY 2>/dev/null || echo "ERR:0:0"
import json
try:
    with open("$OUT/train_metrics.json") as fh:
        d = json.load(fh)
    sr = str(d.get("stop_reason", "") or "").lower()
    fe = int(d.get("final_epoch", 0) or 0)
    print(f"{sr}:{fe}")
except Exception:
    print(":0")
PY
)
  STOP_REASON="${READ_RESULT%%:*}"
  FINAL_EPOCH="${READ_RESULT##*:}"
  case "$STOP_REASON" in
    early_stopping|max_epochs_reached|completed|converged|ran_to_max)
      echo "SKIP: $OUT complete (stop_reason=$STOP_REASON final_epoch=$FINAL_EPOCH)"
      exit 0
      ;;
  esac
  if [ "${FINAL_EPOCH:-0}" -ge 50 ]; then
    echo "SKIP: $OUT complete (final_epoch=$FINAL_EPOCH, no stop_reason)"
    exit 0
  fi
  echo "[CA-TME-STRONG] $OUT stale (stop_reason=$STOP_REASON final_epoch=$FINAL_EPOCH); removing"
  rm -rf "$OUT"
fi

[ -f "$DATASET" ] || { echo "[FATAL] dataset missing: $DATASET"; exit 2; }
[ -f "$CONFIG" ]   || { echo "[FATAL] config missing: $CONFIG";   exit 2; }

echo "[CA-TME-STRONG] MODEL=$MODEL SEED=$SEED proto=$PROTO gpu=$GPU config=${MODEL}_tme_sdr_strong"
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
