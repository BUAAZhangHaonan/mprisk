#!/usr/bin/env bash
# Chain: sdr_strong extension experiment on the 7 weak-separation models.
# Per model (serial, single GPU): train -> calibrate -> Source SDR -> Target SDR.
# Final step: compare_sdr_strong.py -> _summary/SDR_STRONG_COMPARE.md.
# All outputs go to dedicated sdr_strong trees; canonical artifacts untouched.
#
# Usage: GPU=1 nohup bash scripts/cache_matrix_20260722/chain_sdr_strong.sh &
set -uo pipefail
cd "$(dirname "$0")/../../"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

GPU=${GPU:-1}
SEED=20260717
MODELS=(gemma4_12b phi3_5_vision qwen3_5_9b internvl3_5_8b qwen3_5_4b glm4_6v_flash minicpm_v_4_5)
LOGROOT=outputs/cache_matrix_20260722/_logs
LOGFILE=$LOGROOT/chain_sdr_strong.log
mkdir -p "$LOGROOT"

FAILED=()
echo "[$(date)] START chain_sdr_strong gpu=$GPU seed=$SEED models=${MODELS[*]}" | tee -a "$LOGFILE"
T0=$(date +%s)

for MODEL in "${MODELS[@]}"; do
  echo "[$(date)] PHASE=train model=$MODEL" | tee -a "$LOGFILE"
  bash scripts/cache_matrix_20260722/run_ca_tme_sdr_strong.sh "$MODEL" "$SEED" "$GPU" \
    > "$LOGROOT/ca_tme_gru_sdr_strong_${MODEL}_seed${SEED}.driver.log" 2>&1 \
    || { echo "[$(date)] FAIL train $MODEL" | tee -a "$LOGFILE"; FAILED+=("train:$MODEL"); continue; }

  echo "[$(date)] PHASE=calibrate model=$MODEL" | tee -a "$LOGFILE"
  bash scripts/cache_matrix_20260722/run_calibrate_sdr_strong.sh "$MODEL" "$GPU" \
    > "$LOGROOT/calibrate_sdr_strong_${MODEL}.driver.log" 2>&1 \
    || { echo "[$(date)] FAIL calibrate $MODEL" | tee -a "$LOGFILE"; FAILED+=("calibrate:$MODEL"); continue; }

  echo "[$(date)] PHASE=sdr model=$MODEL" | tee -a "$LOGFILE"
  bash scripts/cache_matrix_20260722/run_sdr_strong.sh "$MODEL" "$GPU" \
    > "$LOGROOT/sdr_strong_${MODEL}.driver.log" 2>&1 \
    || { echo "[$(date)] FAIL sdr $MODEL" | tee -a "$LOGFILE"; FAILED+=("sdr:$MODEL"); continue; }

  echo "[$(date)] PHASE=target_sdr model=$MODEL" | tee -a "$LOGFILE"
  bash scripts/cache_matrix_20260722/run_target_sdr_strong.sh "$MODEL" "$GPU" \
    > "$LOGROOT/sdr_strong_target_${MODEL}.driver.log" 2>&1 \
    || { echo "[$(date)] FAIL target_sdr $MODEL" | tee -a "$LOGFILE"; FAILED+=("target_sdr:$MODEL"); }
done

echo "[$(date)] PHASE=compare" | tee -a "$LOGFILE"
python scripts/cache_matrix_20260722/compare_sdr_strong.py >> "$LOGFILE" 2>&1 \
  || echo "[$(date)] FAIL compare" | tee -a "$LOGFILE"

ELAPSED=$(( $(date +%s) - T0 ))
echo "[$(date)] DONE chain_sdr_strong elapsed=${ELAPSED}s failed=${FAILED[*]:-none}" | tee -a "$LOGFILE"
