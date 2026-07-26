#!/usr/bin/env bash
# Chain driver for cache_matrix_20260722 post-BiLSTM phases.
#
# Sequence:
#   1. Wait for BiLSTM TME driver to finish (pid poll).
#   2. Launch LSTM TME wave, wait.
#   3. Launch GRU TME wave, wait.
#   4. Launch SP-MLP (GPU 0) and T-LSTM (GPU 1) in parallel, wait.
#   5. Launch SDR pipeline for 16 models (parallel 2-GPU), wait.
#   6. Aggregate + audit.
#
# Each step writes to outputs/cache_matrix_20260722/_logs/.
set -euo pipefail
cd "$(dirname "$0")/../../"

LOG_DIR=outputs/cache_matrix_20260722/_logs
mkdir -p "$LOG_DIR"
CHAIN_LOG=$LOG_DIR/chain.log

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
  local extra_env=$3
  log "Launching $name: $script (${extra_env})"
  env $extra_env nohup bash "$script" > "$LOG_DIR/${name}.nohup.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG_DIR/${name}.pid"
  log "$name launched pid=$pid"
  wait_for_pid "$name" "$pid"
}

# Step 1: wait for BiLSTM driver
BLSTM_PID_FILE=$LOG_DIR/driver_tme_bilstm.pid
if [ -f "$BLSTM_PID_FILE" ]; then
  BLSTM_PID=$(cat "$BLSTM_PID_FILE")
  wait_for_pid "driver_tme_bilstm" "$BLSTM_PID"
else
  log "No BiLSTM pid file; assuming BiLSTM already done"
fi

# Step 2: LSTM wave
launch_and_wait "driver_tme_lstm" "scripts/cache_matrix_20260722/driver_tme_lstm.sh" ""

# Step 3: GRU wave
launch_and_wait "driver_tme_gru" "scripts/cache_matrix_20260722/driver_tme_gru.sh" ""

# Step 4: SP-MLP (GPU 0) and T-LSTM (GPU 1) in parallel
log "Launching SP-MLP (GPUS=0) and T-LSTM (GPUS=1) in parallel"
GPUS=0 nohup bash scripts/cache_matrix_20260722/driver_sp_mlp.sh \
  > "$LOG_DIR/driver_sp_mlp.nohup.log" 2>&1 &
SP_PID=$!
echo "$SP_PID" > "$LOG_DIR/driver_sp_mlp.pid"
GPUS=1 nohup bash scripts/cache_matrix_20260722/driver_t_lstm.sh \
  > "$LOG_DIR/driver_t_lstm.nohup.log" 2>&1 &
TL_PID=$!
echo "$TL_PID" > "$LOG_DIR/driver_t_lstm.pid"
log "SP-MLP pid=$SP_PID, T-LSTM pid=$TL_PID"
wait_for_pid "driver_sp_mlp" "$SP_PID"
wait_for_pid "driver_t_lstm" "$TL_PID"

# Step 5: SDR pipeline (parallel 2-GPU)
launch_and_wait "driver_sdr" "scripts/cache_matrix_20260722/driver_sdr.sh" ""

# Step 6: aggregate + audit
log "Running aggregate_results.py"
PYTHONPATH=src python scripts/cache_matrix_20260722/aggregate_results.py 2>&1 | tee -a "$CHAIN_LOG"
log "Running audit.py"
PYTHONPATH=src python scripts/cache_matrix_20260722/audit.py 2>&1 | tee -a "$CHAIN_LOG"

log "CHAIN COMPLETE"
