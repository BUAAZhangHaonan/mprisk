#!/usr/bin/env bash
# F1 backfill driver (eval-only, no retraining).
#
# Phase 1 (Source val, 135 cells = 3 encoders x 15 models x 3 seeds):
#   - Re-evaluate each cell's Source-trained best_checkpoint.pt on the val
#     split of its own Source relation_dataset.jsonl via
#     train_trajectory_encoder.py --load-existing --eval-dataset --eval-split relation_val.
#   - Writes eval_f1.json (val_balanced_accuracy_ac + val_f1 macro-F1) in the
#     cell dir. train_metrics.json is NOT touched (training-time artifact).
#
# Phase 2 (Target C/A, 135 cells):
#   - Rerun the cross-domain eval exactly like driver_cross_domain_eval.sh
#     (--load-existing --eval-dataset <relation_dataset_target.jsonl>), which
#     now also emits val_f1. Overwrites target_metrics.json (our eval artifact).
#
# Skip-if-exists: eval_f1.json / target_metrics.json must contain a numeric
# val_f1. Serial on GPU 0 only (GPU 1 reserved for a parallel job).
#
# After all cells: rebuild _summary/MATRIX_TABLES.md (Table A/B F1 columns).
set -euo pipefail
cd "$(dirname "$0")/../../"

source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODELS=(
  gemma3_12b gemma3_4b glm4_6v_flash llava_onevision_qwen2_7b
  minicpm_v_2_6 minicpm_v_4_5 internvl3_5_8b qwen2_5_vl_7b
  qwen3_5_4b qwen3_5_9b qwen3_vl_8b
  gemma4_12b qwen2_5_omni_7b
  phi3_5_vision llava_v1_5_7b
)
SEEDS=(20260717 20260718 20260719)
ENCODERS=(gru lstm bilstm)
GPU=0

mkdir -p outputs/cache_matrix_20260722/_logs
LOGFILE=outputs/cache_matrix_20260722/_logs/driver_f1_backfill.log
echo "[$(date)] START f1_backfill cells=$(( ${#MODELS[@]} * ${#SEEDS[@]} * ${#ENCODERS[@]} * 2 )) gpu=$GPU" | tee -a "$LOGFILE"

has_val_f1() {
  python3 -c "
import json
try:
    d = json.load(open('$1'))
    v = d.get('val_f1')
    print('OK' if isinstance(v, (int, float)) else 'NO')
except Exception:
    print('NO')
" 2>/dev/null || echo "NO"
}

for ENC in "${ENCODERS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    case "$MODEL" in
      qwen2_5_omni_7b|gemma4_12b_it|gemma4_12b|phi4_multimodal)
        PROTO=va; PROTO_UPPER=VA ;;
      *)
        PROTO=vt; PROTO_UPPER=VT ;;
    esac
    PROMPT_SET_KEY=${PROTO}_main_p8_seed20260717
    SRC_DATASET="outputs/cache_matrix_20260722/relation_data/${MODEL}/${PROTO_UPPER}/${PROMPT_SET_KEY}/relation_dataset.jsonl"
    TGT_DATASET="outputs/cache_matrix_20260722/relation_data/${MODEL}/${PROTO_UPPER}/${PROMPT_SET_KEY}/relation_dataset_target.jsonl"

    for SEED in "${SEEDS[@]}"; do
      OUT="outputs/cache_matrix_20260722/runs/ca_tme_${ENC}/${MODEL}_seed${SEED}"
      CKPT="$OUT/best_checkpoint.pt"
      if [ ! -f "$CKPT" ]; then
        echo "[$(date)] SKIP-BOTH $MODEL seed=$SEED enc=$ENC (no checkpoint)" | tee -a "$LOGFILE"
        continue
      fi

      # ---- Phase 1: Source val F1 -> eval_f1.json ----
      if [ -f "$OUT/eval_f1.json" ] && [ "$(has_val_f1 "$OUT/eval_f1.json")" = "OK" ]; then
        echo "[$(date)] SKIP-SRC $MODEL seed=$SEED enc=$ENC (eval_f1.json has val_f1)" | tee -a "$LOGFILE"
      elif [ ! -f "$SRC_DATASET" ]; then
        echo "[$(date)] FAIL-SRC $MODEL seed=$SEED enc=$ENC (missing $SRC_DATASET)" | tee -a "$LOGFILE"
      else
        echo "[$(date)] RUN-SRC model=$MODEL seed=$SEED enc=$ENC" | tee -a "$LOGFILE"
        PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU python scripts/train_trajectory_encoder.py \
          --eval-dataset "$SRC_DATASET" \
          --eval-split relation_val \
          --load-existing "$CKPT" \
          --output-dir "$OUT" \
          --device cuda:0 \
          > "outputs/cache_matrix_20260722/_logs/f1_src_${ENC}_${MODEL}_seed${SEED}.log" 2>&1 \
          || echo "[$(date)] FAIL-SRC $MODEL seed=$SEED enc=$ENC (eval exited nonzero)" | tee -a "$LOGFILE"
      fi

      # ---- Phase 2: Target C/A eval with val_f1 -> target_metrics.json ----
      if [ -f "$OUT/target_metrics.json" ] && [ "$(has_val_f1 "$OUT/target_metrics.json")" = "OK" ]; then
        echo "[$(date)] SKIP-TGT $MODEL seed=$SEED enc=$ENC (target_metrics.json has val_f1)" | tee -a "$LOGFILE"
      elif [ ! -s "$TGT_DATASET" ]; then
        echo "[$(date)] FAIL-TGT $MODEL seed=$SEED enc=$ENC (missing/empty $TGT_DATASET)" | tee -a "$LOGFILE"
      else
        echo "[$(date)] RUN-TGT model=$MODEL seed=$SEED enc=$ENC" | tee -a "$LOGFILE"
        PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU python scripts/train_trajectory_encoder.py \
          --eval-dataset "$TGT_DATASET" \
          --load-existing "$CKPT" \
          --output-dir "$OUT" \
          --device cuda:0 \
          > "outputs/cache_matrix_20260722/_logs/f1_tgt_${ENC}_${MODEL}_seed${SEED}.log" 2>&1 \
          || echo "[$(date)] FAIL-TGT $MODEL seed=$SEED enc=$ENC (eval exited nonzero)" | tee -a "$LOGFILE"
      fi
    done
  done
done

echo "[$(date)] REBUILD matrix tables" | tee -a "$LOGFILE"
python3 scripts/cache_matrix_20260722/build_matrix_tables.py >> "$LOGFILE" 2>&1 \
  || echo "[$(date)] FAIL build_matrix_tables" | tee -a "$LOGFILE"
echo "[$(date)] DONE f1_backfill" | tee -a "$LOGFILE"
