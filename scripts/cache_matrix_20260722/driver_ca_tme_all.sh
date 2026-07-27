#!/usr/bin/env bash
# Driver: C/A TME wave — 3 encoders (gru/lstm/bilstm) x 13 models x 3 seeds = 117 cells.
# 2-GPU parallel. Skip policy delegated to run_ca_tme.sh (stop_reason-based).
# Each cell ~10 min (early-stop). Full sweep ~6-10h on 2 GPUs.
set -uo pipefail
cd "$(dirname "$0")/../../"

mkdir -p outputs/cache_matrix_20260722/_logs

MODELS=(gemma3_12b gemma3_4b glm4_6v_flash llava_onevision_qwen2_7b minicpm_v_2_6
        minicpm_v_4_5 internvl3_5_8b qwen2_5_vl_7b qwen3_5_4b qwen3_5_9b qwen3_vl_8b
        gemma4_12b qwen2_5_omni_7b)
SEEDS=(20260717 20260718 20260719)
ENCODERS=(gru lstm bilstm)
GPUS=(${GPUS:-0 1})

# Cell order: gru first (10 already done -> instant skip), then lstm, then bilstm.
CELLS=()
for ENC in "${ENCODERS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            CELLS+=("$ENC:$MODEL:$SEED")
        done
    done
done

LOGFILE=outputs/cache_matrix_20260722/_logs/driver_ca_tme_all.log
echo "[$(date)] START ca_tme_all cells=${#CELLS[@]} gpus=${GPUS[*]}" | tee -a "$LOGFILE"

gpu_busy=()
for g in "${GPUS[@]}"; do gpu_busy[$g]=""; done

started=0
for cell in "${CELLS[@]}"; do
    ENC="${cell%%:*}"
    rest="${cell#*:}"
    MODEL="${rest%%:*}"
    SEED="${rest##*:}"

    # Find a free GPU slot
    free_gpu=""
    while [ -z "$free_gpu" ]; do
        for g in "${GPUS[@]}"; do
            if [ -z "${gpu_busy[$g]}" ] || ! kill -0 "${gpu_busy[$g]}" 2>/dev/null; then
                gpu_busy[$g]=""
                free_gpu=$g
                break
            fi
        done
        [ -z "$free_gpu" ] && sleep 5
    done

    echo "[$(date)] RUN enc=$ENC model=$MODEL seed=$SEED gpu=$free_gpu" | tee -a "$LOGFILE"
    (
        bash scripts/cache_matrix_20260722/run_ca_tme.sh "$MODEL" "$SEED" "$free_gpu" "$ENC"
    ) > "outputs/cache_matrix_20260722/_logs/ca_tme_${ENC}_${MODEL}_seed${SEED}.driver.log" 2>&1 &
    gpu_busy[$free_gpu]=$!
    started=$((started+1))
done

wait
echo "[$(date)] DONE ca_tme_all started=$started" | tee -a "$LOGFILE"
