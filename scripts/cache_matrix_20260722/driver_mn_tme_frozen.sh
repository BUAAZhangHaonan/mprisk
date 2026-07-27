#!/usr/bin/env bash
# Driver: M/N TME(frozen) wave — load C/A GRU encoder checkpoint, freeze encoder,
# train only M/N head. 13 models x 3 seeds = 39 cells, 2-GPU parallel (~5 min/cell).
# Depends on C/A GRU driver having produced
#   outputs/cache_matrix_20260722/runs/ca_tme_gru/<model>_seed<seed>/best_checkpoint.pt
set -euo pipefail
cd "$(dirname "$0")/../../"

mkdir -p outputs/cache_matrix_20260722/_logs

MODELS=(gemma3_12b gemma3_4b glm4_6v_flash llava_onevision_qwen2_7b minicpm_v_2_6
        minicpm_v_4_5 internvl3_5_8b qwen2_5_vl_7b qwen3_5_4b qwen3_5_9b qwen3_vl_8b
        gemma4_12b qwen2_5_omni_7b)
SEEDS=(20260717 20260718 20260719)
GPUS=(${GPUS:-0 1})

CELLS=()
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CELLS+=("$MODEL:$SEED")
    done
done

LOGFILE=outputs/cache_matrix_20260722/_logs/driver_mn_tme_frozen.log
echo "[$(date)] START mn_tme_frozen cells=${#CELLS[@]} gpus=${GPUS[*]}" | tee -a "$LOGFILE"

gpu_busy=()
for g in "${GPUS[@]}"; do gpu_busy[$g]=""; done

for cell in "${CELLS[@]}"; do
    MODEL="${cell%:*}"
    SEED="${cell#*:}"

    OUT="outputs/cache_matrix_20260722/runs/mn_tme_frozen/${MODEL}_seed${SEED}"
    if [ -f "$OUT/metrics.json" ]; then
        echo "[$(date)] SKIP $MODEL seed=$SEED (metrics.json)" | tee -a "$LOGFILE"
        continue
    fi

    # Pre-flight: skip if C/A checkpoint missing (driver_ca_tme_all not yet done
    # for this cell). The chain waits for the whole C/A wave so this is just a
    # safety net.
    PA_CKPT="outputs/cache_matrix_20260722/runs/ca_tme_gru/${MODEL}_seed${SEED}/best_checkpoint.pt"
    if [ ! -f "$PA_CKPT" ]; then
        echo "[$(date)] SKIP $MODEL seed=$SEED (C/A ckpt missing: $PA_CKPT)" | tee -a "$LOGFILE"
        continue
    fi

    while true; do
        for g in "${GPUS[@]}"; do
            if [ -z "${gpu_busy[$g]}" ] || ! kill -0 "${gpu_busy[$g]}" 2>/dev/null; then
                gpu_busy[$g]=""
                free_gpu=$g
                break 2
            fi
        done
        sleep 5
    done

    echo "[$(date)] RUN model=$MODEL seed=$SEED gpu=$free_gpu" | tee -a "$LOGFILE"
    bash scripts/cache_matrix_20260722/run_mn_tme_frozen.sh "$MODEL" "$SEED" "$free_gpu" \
        > "outputs/cache_matrix_20260722/_logs/mn_tme_frozen_${MODEL}_seed${SEED}.log" 2>&1 &
    gpu_busy[$free_gpu]=$!
done

wait
echo "[$(date)] DONE mn_tme_frozen" | tee -a "$LOGFILE"
