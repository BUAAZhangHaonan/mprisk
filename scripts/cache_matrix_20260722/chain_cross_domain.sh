#!/usr/bin/env bash
# Chain: wait for C/A cross-domain eval driver to finish, then run Target SDR driver.
#
# Args: $1 = PID of the C/A cross-domain eval driver (driver_cross_domain_eval.sh)
# Env:  CUDA_VISIBLE_DEVICES (default 0) — passed to both drivers.
set -euo pipefail
cd "$(dirname "$0")/../../"

mkdir -p outputs/cache_matrix_20260722/_logs
CHAIN_LOG=outputs/cache_matrix_20260722/_logs/chain_cross_domain.log
DRIVER_PID=${1:?usage: chain_cross_domain.sh <cross_domain_driver_pid>}
GPU=${CUDA_VISIBLE_DEVICES:-0}

echo "[$(date)] CHAIN START driver_pid=$DRIVER_PID gpu=$GPU" | tee -a "$CHAIN_LOG"

# Phase 1: wait for C/A cross-domain eval driver.
PHASE1_LOG=outputs/cache_matrix_20260722/_logs/driver_cross_domain_eval.nohup.log
echo "[$(date)] PHASE 1 waiting on cross_domain_eval pid=$DRIVER_PID" | tee -a "$CHAIN_LOG"
# Wait for driver PID and all children to exit.
while kill -0 "$DRIVER_PID" 2>/dev/null; do
  sleep 30
done
# Drain any remaining child workers (the driver spawns eval PIDs that may
# outlive the parent's wait loop briefly).
wait 2>/dev/null || true
echo "[$(date)] PHASE 1 DONE cross_domain_eval pid=$DRIVER_PID exited" | tee -a "$CHAIN_LOG"

# Phase 2: Target SDR sweep.
echo "[$(date)] PHASE 2 START target_sdr gpu=$GPU" | tee -a "$CHAIN_LOG"
CUDA_VISIBLE_DEVICES=$GPU bash scripts/cache_matrix_20260722/driver_target_sdr.sh \
  >> "$CHAIN_LOG" 2>&1 || {
    echo "[$(date)] PHASE 2 FAIL target_sdr" | tee -a "$CHAIN_LOG"
    exit 1
  }
echo "[$(date)] PHASE 2 DONE target_sdr" | tee -a "$CHAIN_LOG"
echo "[$(date)] CHAIN DONE" | tee -a "$CHAIN_LOG"
