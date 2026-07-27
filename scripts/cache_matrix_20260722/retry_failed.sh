#!/usr/bin/env bash
# Retry chain: waits for main chain to finish, then re-runs every wave driver
# to pick up any cells that failed (skip-if-done makes this safe).
set -euo pipefail
cd "$(dirname "$0")/../../"

LOG_DIR=outputs/cache_matrix_20260722/_logs
mkdir -p "$LOG_DIR"
RETRY_LOG=$LOG_DIR/retry.log

log() { echo "[$(date '+%F %T %z')] $*" | tee -a "$RETRY_LOG"; }

# Wait for main chain to finish
MAIN_PID_FILE=$LOG_DIR/chain_post_bilstm.pid
if [ -f "$MAIN_PID_FILE" ]; then
  MAIN_PID=$(cat "$MAIN_PID_FILE")
  log "Waiting for main chain PID=$MAIN_PID"
  while kill -0 "$MAIN_PID" 2>/dev/null; do
    sleep 60
  done
  log "Main chain PID=$MAIN_PID done"
else
  log "No main chain pid file; proceeding"
fi

# Retry each wave. Skip-if-metrics.json makes cells that already passed safe.
log "Retry TME-BiLSTM"
bash scripts/cache_matrix_20260722/driver_tme_bilstm.sh > "$LOG_DIR/driver_tme_bilstm_retry.log" 2>&1
log "TME-BiLSTM retry done"

log "Retry TME-LSTM"
bash scripts/cache_matrix_20260722/driver_tme_lstm.sh > "$LOG_DIR/driver_tme_lstm_retry.log" 2>&1
log "TME-LSTM retry done"

log "Retry TME-GRU"
bash scripts/cache_matrix_20260722/driver_tme_gru.sh > "$LOG_DIR/driver_tme_gru_retry.log" 2>&1
log "TME-GRU retry done"

log "Retry SP-MLP (GPU 0)"
GPUS=0 bash scripts/cache_matrix_20260722/driver_sp_mlp.sh > "$LOG_DIR/driver_sp_mlp_retry.log" 2>&1
log "SP-MLP retry done"

log "Retry T-LSTM (GPU 1)"
GPUS=1 bash scripts/cache_matrix_20260722/driver_t_lstm.sh > "$LOG_DIR/driver_t_lstm_retry.log" 2>&1
log "T-LSTM retry done"

# Re-run calibration + SDR in case any cell state changed
log "Re-run calibration"
bash scripts/cache_matrix_20260722/driver_calibrate.sh > "$LOG_DIR/driver_calibrate_retry.log" 2>&1
log "Calibration retry done"

log "Re-run SDR"
bash scripts/cache_matrix_20260722/driver_sdr.sh > "$LOG_DIR/driver_sdr_retry.log" 2>&1
log "SDR retry done"

# Re-aggregate + audit
log "Re-aggregate"
PYTHONPATH=src /home/team/zhanghaonan/miniconda3/envs/mprisk/bin/python scripts/cache_matrix_20260722/aggregate_results.py 2>&1 | tee -a "$RETRY_LOG"
log "Re-audit"
PYTHONPATH=src /home/team/zhanghaonan/miniconda3/envs/mprisk/bin/python scripts/cache_matrix_20260722/audit.py 2>&1 | tee -a "$RETRY_LOG"

log "RETRY CHAIN COMPLETE"
