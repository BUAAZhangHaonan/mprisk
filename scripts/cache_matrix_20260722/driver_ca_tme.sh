#!/usr/bin/env bash
# Driver: C/A TME wave (GRU encoder + SDR hinge aux loss) — 13 models x 3 seeds, 2-GPU parallel.
# Each cell takes ~10 min; full sweep ~3.5h.
set -euo pipefail
cd "$(dirname "$0")/../../"

mkdir -p outputs/cache_matrix_20260722/_logs

MODELS=(gemma3_12b gemma3_4b glm4_6v_flash llava_onevision_qwen2_7b minicpm_v_2_6         minicpm_v_4_5 internvl3_5_8b qwen2_5_vl_7b qwen3_5_4b qwen3_5_9b qwen3_vl_8b         gemma4_12b qwen2_5_omni_7b)
SEEDS=(20260717 20260718 20260719)
GPUS=(${GPUS:-0 1})

# Build all (model, seed) cells
CELLS=()
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CELLS+=("$MODEL:$SEED")
    done
done

LOGFILE=outputs/cache_matrix_20260722/_logs/driver_ca_tme.log
echo "[$(date)] START ca_tme cells=${#CELLS[@]} gpus=${GPUS[*]}" | tee -a "$LOGFILE"

gpu_busy=()
for g in "${GPUS[@]}"; do gpu_busy[$g]=""; done

for cell in "${CELLS[@]}"; do
    MODEL="${cell%:*}"
    SEED="${cell#*:}"

    OUT="outputs/cache_matrix_20260722/runs/ca_tme/${MODEL}_seed${SEED}"

    # Skip if train_metrics.json exists AND final_epoch >= 50
    if [ -f "$OUT/train_metrics.json" ]; then
        final_ep=$(python3 -c "
import json
try:
    d = json.load(open('$OUT/train_metrics.json'))
    print(d.get('final_epoch', 0))
except: print(0)
" 2>/dev/null || echo 0)
        if [ "${final_ep:-0}" -ge 50 ]; then
            echo "[$(date)] SKIP $MODEL seed=$SEED (final_epoch=$final_ep)" | tee -a "$LOGFILE"
            continue
        fi
    fi

    # Find free GPU
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
    bash scripts/cache_matrix_20260722/run_ca_tme.sh "$MODEL" "$SEED" "$free_gpu"         > "outputs/cache_matrix_20260722/_logs/ca_tme_${MODEL}_seed${SEED}.log" 2>&1 &
    gpu_busy[$free_gpu]=$!
done

wait
echo "[$(date)] DONE ca_tme" | tee -a "$LOGFILE"
