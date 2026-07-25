#!/bin/bash
# canonical_rerun v2.1 Qwen3.5-4B full sweep driver.
# Runs C1, C3 (Stage-1) -> C2, C4 (Stage-2) + C5 (TME v3-B from scratch)
# for 3 seeds, sequentially on a single GPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODEL=qwen3_5_4b
GPU=1
SEEDS=(20260717 20260718 20260719)
SCRIPT_DIR=experiments/canonical_rerun
LOG_DIR=outputs/canonical_rerun_v2/_logs/qwen3_5_4b
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
DRIVER_LOG=$LOG_DIR/driver_${STAMP}.log
echo "[driver] start $(date)" | tee "$DRIVER_LOG"

# Stage 1: C1 + C3 (PA pretrain). These are prerequisites for C2/C4.
for SEED in "${SEEDS[@]}"; do
  echo "[driver] === C1 SP-MLP stage1 seed=$SEED ===" | tee -a "$DRIVER_LOG"
  bash $SCRIPT_DIR/C1_sp_mlp_v2_pretrain_ca.sh "$MODEL" "$SEED" "$GPU" 2>&1 | tail -20 | tee -a "$DRIVER_LOG"
  echo "[driver] === C3 T-LSTM stage1 seed=$SEED ===" | tee -a "$DRIVER_LOG"
  bash $SCRIPT_DIR/C3_t_lstm_v2_pretrain_ca.sh "$MODEL" "$SEED" "$GPU" 2>&1 | tail -20 | tee -a "$DRIVER_LOG"
done

# Stage 2: C2, C4, C5 in parallel-by-seed (sequential per seed).
for SEED in "${SEEDS[@]}"; do
  echo "[driver] === C2 SP-MLP stage2 seed=$SEED ===" | tee -a "$DRIVER_LOG"
  bash $SCRIPT_DIR/C2_sp_mlp_v2_mn_head.sh "$MODEL" "$SEED" "$GPU" 2>&1 | tail -15 | tee -a "$DRIVER_LOG"
  echo "[driver] === C4 T-LSTM stage2 seed=$SEED ===" | tee -a "$DRIVER_LOG"
  bash $SCRIPT_DIR/C4_t_lstm_v2_mn_head.sh "$MODEL" "$SEED" "$GPU" 2>&1 | tail -15 | tee -a "$DRIVER_LOG"
  echo "[driver] === C5 TME v3-B e2e from scratch seed=$SEED ===" | tee -a "$DRIVER_LOG"
  bash $SCRIPT_DIR/C5_tme_v3b_e2e_mn.sh "$MODEL" "$SEED" "$GPU" 2>&1 | tail -15 | tee -a "$DRIVER_LOG"
done

echo "[driver] ALL DONE $(date)" | tee -a "$DRIVER_LOG"
