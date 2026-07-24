#!/usr/bin/env bash
# Batch regeneration of val-aligned best_test_preds.pt for canonical_rerun_v2
# T1 / T5 / PA-ablation runs. Idempotent: safe to re-run on already-fixed runs.
#
# Strategy: 2 GPUs in parallel via GNU-style xargs -P. Each run takes ~30s.
# Failures are logged but do not stop the batch.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${MPRISK_ROOT:-$SCRIPT_DIR/..}" && pwd)"
cd "$PROJECT_ROOT"

source "${CONDA_PREFIX:-/home/team/zhanghaonan/miniconda3}/../etc/profile.d/conda.sh" 2>/dev/null || true
conda activate mprisk 2>/dev/null || true

LOG_DIR="/tmp/regen_logs"
mkdir -p "$LOG_DIR"
SUMMARY_JSONL="$LOG_DIR/summary.jsonl"
FAILED_LOG="$LOG_DIR/failed.txt"
: > "$SUMMARY_JSONL"
: > "$FAILED_LOG"

TARGETS=(
    outputs/canonical_rerun_v2/T1_gru_ca_frozen
    outputs/canonical_rerun_v2/T5_lstm_ca_frozen
    outputs/canonical_rerun_v2/T1_ablation_pa_only
    outputs/canonical_rerun_v2/T1_ablation_pa_d
    outputs/canonical_rerun_v2/T1_ablation_pa_s
    outputs/canonical_rerun_v2/T1_ablation_pa_sd
)

# Collect run dirs into a file
RUN_LIST="$LOG_DIR/run_list.txt"
: > "$RUN_LIST"
for tree in "${TARGETS[@]}"; do
    if [ -d "$tree" ]; then
        find "$tree" -maxdepth 1 -mindepth 1 -type d >> "$RUN_LIST" 2>/dev/null
    fi
done

TOTAL=$(wc -l < "$RUN_LIST")
echo "[batch] total candidate run dirs: $TOTAL"
echo "[batch] start: $(date -Iseconds)"

# Worker function: takes run_dir and gpu index as args
worker() {
    local run="$1"
    local gpu="$2"
    local name
    name="$(basename "$(dirname "$run")")/$(basename "$run")"
    local safe_name
    safe_name="$(echo "$name" | tr '/ ' '__')"
    local log="$LOG_DIR/${safe_name}.log"
    local start=$SECONDS
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src python scripts/regenerate_val_test_preds.py \
        --run-dir "$run" --device cuda:0 > "$log" 2>&1
    local rc=$?
    local elapsed=$((SECONDS - start))
    if [ $rc -eq 0 ]; then
        local report
        report=$(grep '^REPORT_JSON ' "$log" | tail -1 | sed 's/^REPORT_JSON //')
        if [ -n "$report" ]; then
            echo "$report" >> "$SUMMARY_JSONL"
        else
            echo "{\"run_dir\": \"$run\", \"status\": \"no_report_json\", \"log\": \"$log\"}" >> "$SUMMARY_JSONL"
        fi
        echo "[ok ${elapsed}s gpu${gpu}] $name"
    else
        echo "{\"run_dir\": \"$run\", \"status\": \"failed\", \"rc\": $rc, \"log\": \"$log\", \"gpu\": $gpu}" >> "$SUMMARY_JSONL"
        echo "$run gpu${gpu} rc=$rc log=$log" >> "$FAILED_LOG"
        echo "[fail ${elapsed}s gpu${gpu}] $name (rc=$rc)"
    fi
}
export -f worker
export LOG_DIR SUMMARY_JSONL FAILED_LOG

# Build (run,gpu) pairs alternating GPUs
PAIRS_FILE="$LOG_DIR/pairs.txt"
: > "$PAIRS_FILE"
i=0
while IFS= read -r run; do
    [ -z "$run" ] && continue
    gpu=$((i % 2))
    # Print as tab-separated; xargs parses on whitespace
    printf '%s\t%s\n' "$run" "$gpu" >> "$PAIRS_FILE"
    i=$((i+1))
done < "$RUN_LIST"

# Launch 2 concurrent workers (one per GPU), feeding from PAIRS_FILE
# We use a per-GPU job queue: spawn one bash subshell per GPU that reads
# only its own lines. This guarantees at most 1 job per GPU at a time.
for gpu in 0 1; do
    awk -F'\t' -v g="$gpu" '$2 == g { print $1 }' "$PAIRS_FILE" | \
        while IFS= read -r run; do
            worker "$run" "$gpu"
        done > "$LOG_DIR/worker_gpu${gpu}.log" 2>&1 &
done

wait

echo "[batch] end: $(date -Iseconds)"
echo "[batch] summary jsonl: $SUMMARY_JSONL ($(wc -l < "$SUMMARY_JSONL") entries)"
echo "[batch] failed log: $FAILED_LOG ($(wc -l < "$FAILED_LOG") entries)"
