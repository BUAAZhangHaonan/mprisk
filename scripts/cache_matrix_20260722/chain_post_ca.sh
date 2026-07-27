#!/usr/bin/env bash
# Chain driver for cache_matrix_20260722 post-C/A phases.
#
# Sequence (after C/A GRU driver finishes):
#   1. Wait for driver_ca_tme_all (PID from _logs/driver_ca_tme_all.pid).
#   2. Calibrate thresholds (13 models, 2-GPU parallel, ~30 min).
#   3. SDR + state patterns (13 models, 2-GPU parallel, ~30 min).
#   4. M/N TME-E2E (39 cells, 2-GPU parallel, ~3-4 h).
#   5. M/N TME(frozen) (39 cells, 2-GPU parallel, ~2 h).
#   6. Aggregate + audit.
#
# Each step writes to outputs/cache_matrix_20260722/_logs/.
set -euo pipefail
cd "$(dirname "$0")/../../"

LOG_DIR=outputs/cache_matrix_20260722/_logs
mkdir -p "$LOG_DIR"
CHAIN_LOG="$LOG_DIR/chain_post_ca.log"

log() { echo "[$(date '+%F %T %z')] $*" | tee -a "$CHAIN_LOG"; }

wait_for_pid() {
  local name=$1
  local pid=$2
  if [ -z "$pid" ]; then
    log "$name: no pid, skipping wait"
    return
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    log "$name: pid=$pid not running, proceeding"
    return
  fi
  log "$name: waiting for pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  log "$name: pid=$pid done"
}

launch_and_wait() {
  local name=$1
  local script=$2
  log "Launching $name: $script"
  nohup bash "$script" > "$LOG_DIR/${name}.nohup.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG_DIR/${name}.pid"
  log "$name launched pid=$pid"
  wait_for_pid "$name" "$pid"
}

# Step 1: wait for C/A driver
CA_PID_FILE="$LOG_DIR/driver_ca_tme_all.pid"
if [ -f "$CA_PID_FILE" ]; then
  CA_PID=$(cat "$CA_PID_FILE")
  wait_for_pid "driver_ca_tme_all" "$CA_PID"
else
  log "No C/A pid file; assuming C/A already done"
fi

# Step 2: threshold calibration (consumes ca_tme_gru/<model>_seed20260717/best_checkpoint.pt)
launch_and_wait "driver_calibrate_post" "scripts/cache_matrix_20260722/driver_calibrate.sh"

# Step 3: SDR pipeline (consumes same ckpt + thresholds from step 2)
launch_and_wait "driver_sdr_post" "scripts/cache_matrix_20260722/driver_sdr.sh"

# Step 4: M/N TME-E2E (no C/A dependency; trains GRU encoder end-to-end with CE loss)
launch_and_wait "driver_mn_tme_e2e_post" "scripts/cache_matrix_20260722/driver_mn_tme_e2e.sh"

# Step 5: M/N TME(frozen) — loads C/A GRU encoder checkpoint, freezes encoder, trains head
launch_and_wait "driver_mn_tme_frozen_post" "scripts/cache_matrix_20260722/driver_mn_tme_frozen.sh"

# Step 6: aggregate + audit
log "Running aggregate_results.py"
PYTHONPATH=src /home/team/zhanghaonan/miniconda3/envs/mprisk/bin/python \
    scripts/cache_matrix_20260722/aggregate_results.py 2>&1 | tee -a "$LOG_DIR/aggregate_post.log"
log "Running audit.py"
PYTHONPATH=src /home/team/zhanghaonan/miniconda3/envs/mprisk/bin/python \
    scripts/cache_matrix_20260722/audit.py 2>&1 | tee -a "$LOG_DIR/audit_post.log"

log "CHAIN POST-C/A COMPLETE"
