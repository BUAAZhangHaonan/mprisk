#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

LOGDIR=outputs/canonical_rerun/_logs/phase_c
mkdir -p "$LOGDIR"
DRIVER_LOG="$LOGDIR/driver_$(date +%Y%m%d_%H%M%S).log"
echo "[driver] start $(date)" | tee "$DRIVER_LOG"

MODELS=(qwen3_vl_8b qwen3_5_4b internvl3_5_8b qwen2_5_omni_7b)
SEEDS=(20260717 20260718 20260719)
EXPROOT=experiments/canonical_rerun_v2

run_one() {
  local sh=$1; local model=$2; local seed=$3; local gpu=$4
  local base=$(basename $sh .sh)
  local log="$LOGDIR/${base}_${model}_seed${seed}.log"
  if [ -f "$log.done" ]; then
    echo "[driver] SKIP $base $model seed=$seed (done)" | tee -a "$DRIVER_LOG"
    return 0
  fi
  echo "[driver] RUN $base $model seed=$seed GPU=$gpu $(date)" | tee -a "$DRIVER_LOG"
  bash $sh $model $seed $gpu > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$log.done"
    echo "[driver] DONE $base $model seed=$seed $(date)" | tee -a "$DRIVER_LOG"
  else
    echo "[driver] FAIL $base $model seed=$seed rc=$rc $(date)" | tee -a "$DRIVER_LOG"
    tail -20 "$log" | tee -a "$DRIVER_LOG"
  fi
}

# Wave 1: C5 (longest, from scratch 100 epoch) - all 4 models x 3 seeds
for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_one $EXPROOT/C5_tme_e2e_mn.sh $MODEL $SEED 1
  done
done

# Wave 2: T_pa_encoder_frozen (gru)
for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    base=T_pa_encoder_frozen
    log="$LOGDIR/${base}_${MODEL}_seed${SEED}_gru.log"
    if [ -f "$log.done" ]; then
      echo "[driver] SKIP $base $MODEL seed=$SEED (done)" | tee -a "$DRIVER_LOG"
      continue
    fi
    echo "[driver] RUN $base $MODEL seed=$SEED GPU=1 $(date)" | tee -a "$DRIVER_LOG"
    bash $EXPROOT/T_pa_encoder_frozen.sh $MODEL $SEED 1 gru > "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      touch "$log.done"
      echo "[driver] DONE $base $MODEL seed=$SEED $(date)" | tee -a "$DRIVER_LOG"
    else
      echo "[driver] FAIL $base $MODEL seed=$SEED rc=$rc $(date)" | tee -a "$DRIVER_LOG"
      tail -20 "$log" | tee -a "$DRIVER_LOG"
    fi
  done
done

# Wave 3: C1/C3 stage1 + C2/C4 stage2 (chained per model+seed)
for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_one $EXPROOT/C1_sp_mlp_pretrain_ca.sh $MODEL $SEED 1
    run_one $EXPROOT/C3_t_lstm_pretrain_ca.sh $MODEL $SEED 1
    run_one $EXPROOT/C2_sp_mlp_mn_head.sh $MODEL $SEED 1
    run_one $EXPROOT/C4_t_lstm_mn_head.sh $MODEL $SEED 1
  done
done

# Wave 4: PA ablation (Qwen3-VL-8B only); T1_ablation_pa.sh takes VARIANT SEED GPU.
for VARIANT in pa_only pa_d pa_s pa_sd; do
  for SEED in "${SEEDS[@]}"; do
    log="$LOGDIR/T1_ablation_pa_${VARIANT}_seed${SEED}.log"
    if [ -f "$log.done" ]; then
      echo "[driver] SKIP T1_ablation_pa variant=$VARIANT seed=$SEED (done)" | tee -a "$DRIVER_LOG"
      continue
    fi
    echo "[driver] RUN T1_ablation_pa variant=$VARIANT seed=$SEED GPU=1 $(date)" | tee -a "$DRIVER_LOG"
    bash $EXPROOT/T1_ablation_pa.sh $VARIANT $SEED 1 > "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
      touch "$log.done"
      echo "[driver] DONE T1_ablation_pa variant=$VARIANT seed=$SEED $(date)" | tee -a "$DRIVER_LOG"
    else
      echo "[driver] FAIL T1_ablation_pa variant=$VARIANT seed=$SEED rc=$rc $(date)" | tee -a "$DRIVER_LOG"
      tail -20 "$log" | tee -a "$DRIVER_LOG"
    fi
  done
done

echo "[driver] ALL DONE $(date)" | tee -a "$DRIVER_LOG"
