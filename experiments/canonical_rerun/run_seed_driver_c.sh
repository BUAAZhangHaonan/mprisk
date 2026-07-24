#!/bin/bash
# Driver: runs C1->C2, C3->C4, C5 for one seed on one GPU.
# Usage: run_seed_driver.sh MODEL SEED GPU
# Continues past any single failure, collects failing stage names, exits
# non-zero at the end if any failed. Each .sh has `set -euo pipefail` internally.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"

MODEL=${1:-qwen3_vl_8b}
SEED=${2:-20260717}
GPU=${3:-0}

EXPERIMENTS=experiments/canonical_rerun
FAILED=()

echo "=== [seed=$SEED gpu=$GPU] C1 (sp_mlp_v2 stage1) ==="
bash $EXPERIMENTS/C1_sp_mlp_v2_pretrain_ca.sh "$MODEL" "$SEED" "$GPU" || FAILED+=("C1")

echo "=== [seed=$SEED gpu=$GPU] C2 (sp_mlp_v2 stage2) ==="
bash $EXPERIMENTS/C2_sp_mlp_v2_mn_head.sh "$MODEL" "$SEED" "$GPU" || FAILED+=("C2")

echo "=== [seed=$SEED gpu=$GPU] C3 (t_lstm_v2 stage1) ==="
bash $EXPERIMENTS/C3_t_lstm_v2_pretrain_ca.sh "$MODEL" "$SEED" "$GPU" || FAILED+=("C3")

echo "=== [seed=$SEED gpu=$GPU] C4 (t_lstm_v2 stage2) ==="
bash $EXPERIMENTS/C4_t_lstm_v2_mn_head.sh "$MODEL" "$SEED" "$GPU" || FAILED+=("C4")

echo "=== [seed=$SEED gpu=$GPU] C5 (tme_v3b e2e) ==="
bash $EXPERIMENTS/C5_tme_v3b_e2e_mn.sh "$MODEL" "$SEED" "$GPU" || FAILED+=("C5")

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== [seed=$SEED gpu=$GPU] FAILED stages: ${FAILED[*]} ===" >&2
  exit 1
fi
echo "=== [seed=$SEED gpu=$GPU] ALL DONE ==="
