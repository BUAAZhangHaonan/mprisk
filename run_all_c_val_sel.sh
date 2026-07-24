#!/bin/bash
# Drive all 30 C1-C5 runs with val-selected + class-balanced CE.
# Per model+seed, runs C1 -> C2 (uses C1 encoder), C3 -> C4 (uses C3 encoder), C5 (uses T1 PA).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR}"
source "${CONDA_PREFIX:-/opt/miniconda3}/../etc/profile.d/conda.sh" 2>/dev/null || true
conda activate mprisk

GPU=0
MODELS=(qwen3_vl_8b qwen3_5_4b)
SEEDS=(20260717 20260718 20260719)
declare -A STATUS
TOTAL=0
DONE=0
FAIL=0
for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for STAGE in C1_sp_mlp_v2_pretrain_ca C2_sp_mlp_v2_mn_head C3_t_lstm_v2_pretrain_ca C4_t_lstm_v2_mn_head C5_tme_v3b_e2e_mn; do
      TOTAL=$((TOTAL+1))
    done
  done
done

for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for STAGE in C1_sp_mlp_v2_pretrain_ca C2_sp_mlp_v2_mn_head C3_t_lstm_v2_pretrain_ca C4_t_lstm_v2_mn_head C5_tme_v3b_e2e_mn; do
      KEY="${STAGE}|${MODEL}|${SEED}"
      SH="experiments/canonical_rerun_v2/${STAGE}.sh"
      echo ">>> [$((DONE+FAIL+1))/$TOTAL] $KEY"
      if bash "$SH" "$MODEL" "$SEED" "$GPU" > "outputs/canonical_rerun_v2/_logs/${STAGE}_${MODEL}_seed${SEED}.log" 2>&1; then
        STATUS[$KEY]="OK"
        DONE=$((DONE+1))
        echo "    OK (done=$DONE fail=$FAIL)"
      else
        STATUS[$KEY]="FAIL"
        FAIL=$((FAIL+1))
        echo "    FAIL (done=$DONE fail=$FAIL) -- see log"
      fi
    done
  done
done

echo ""
echo "=== SUMMARY ==="
echo "Total=$TOTAL Done=$DONE Fail=$FAIL"
for K in "${!STATUS[@]}"; do
  if [ "${STATUS[$K]}" = "FAIL" ]; then
    echo "FAIL: $K"
  fi
done
echo "=== END ==="
