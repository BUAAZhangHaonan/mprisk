#!/usr/bin/env bash
# Targeted launcher: TME + SDR pipeline for phi3_5_vision + llava_v1_5_7b.
#
# Why this script exists: cache_matrix_20260722 build_ca_datasets.py and
# build_ca_yamls.py have a hardcoded 13-model MODELS dict that excludes
# phi3_5_vision / phi4_multimodal / llava_v1_5_7b. Per user instruction
# (ignore 8-prompt design, run all remaining experiments with cache), we
# extend to the 2 extras (skip phi4_multimodal: 0 judgments).
#
# Phases:
#   0. Generate 6 YAMLs (2 models x gru/lstm/bilstm).
#   1. Build relation datasets for 2 models (in-process patch of MODELS dict).
#   2. ca_tme x 3 encoders x 3 seeds (2 models) = 18 cells.
#   3. mn_tme_e2e x 3 seeds (2 models) = 6 cells.
#   4. mn_tme_frozen x 3 seeds (2 models) = 6 cells (needs ca_tme_gru).
#   5. calibrate x 2 models.
#   6. sdr x 2 models.
#   7. aggregate + audit.
set -uo pipefail
cd "$(dirname "$0")/../../"

source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

EXTRAS=(phi3_5_vision llava_v1_5_7b)
SEEDS=(20260717 20260718 20260719)
ENCODERS=(gru lstm bilstm)
GPUS=(${GPUS:-0 1})

LOG_DIR=outputs/cache_matrix_20260722/_logs
mkdir -p "$LOG_DIR"
CHAIN_LOG="$LOG_DIR/chain_extras.log"
log() { echo "[$(date '+%F %T %z')] $*" | tee -a "$CHAIN_LOG"; }

log "=== START extras chain models=${EXTRAS[*]} gpus=${GPUS[*]} ==="

# -------------------------------------------------------------------
# Phase 0: generate YAMLs (template off qwen3_vl_8b variants)
# -------------------------------------------------------------------
log "Phase 0: generate 6 YAMLs"
YAML_DIR=configs/experiments/cache_matrix_20260722
VT_SHA=12eb3c725eaf80c302ffb412984d8dc0964f8fd81e20e5aa88cdc14e8461ea90

gen_yaml() {
  local model=$1 encoder=$2
  local arch key
  case "$encoder" in
    gru)    arch=layer_l2_gru_linear_relation_v1 ;;
    lstm)   arch=layer_l2_lstm_linear_relation_v1 ;;
    bilstm) arch=tme_bilstm_proxy_anchor_v1 ;;
  esac
  key="cache_matrix_${model}_tme_sdr_${encoder}_v1"
  # Strip "_gru" suffix to match run_ca_tme.sh naming convention (gru has no suffix)
  if [ "$encoder" = "gru" ]; then
    out="$YAML_DIR/${model}_tme_sdr.yaml"
  else
    out="$YAML_DIR/${model}_tme_sdr_${encoder}.yaml"
  fi

  python3 - "$model" "$encoder" "$arch" "$key" "$VT_SHA" "$out" <<'PY'
import sys, yaml
model, encoder, arch, key, vt_sha, out = sys.argv[1:7]
template = f"configs/experiments/cache_matrix_20260722/qwen3_vl_8b_tme_sdr{'' if encoder=='gru' else '_'+encoder}.yaml"
cfg = yaml.safe_load(open(template).read())
cfg["key"] = key
cfg["model_key"] = model
# protocol + prompt_set_artifact_sha256 stay (both vt); expected_prompt_ids stay
with open(out, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
print(f"[yaml] {out}")
PY
}

for M in "${EXTRAS[@]}"; do
  for ENC in "${ENCODERS[@]}"; do
    if [ "$ENC" = "gru" ]; then
      yml="$YAML_DIR/${M}_tme_sdr.yaml"
    else
      yml="$YAML_DIR/${M}_tme_sdr_${ENC}.yaml"
    fi
    if [ ! -f "$yml" ]; then
      gen_yaml "$M" "$ENC"
    fi
  done
done
log "Phase 0 done"

# -------------------------------------------------------------------
# Phase 1: build relation datasets (patch MODELS dict at runtime)
# -------------------------------------------------------------------
log "Phase 1: build relation datasets"
for M in "${EXTRAS[@]}"; do
  DST="outputs/cache_matrix_20260722/relation_data/$M/VT/vt_main_p8_seed20260717/relation_dataset.jsonl"
  if [ -f "$DST" ]; then
    log "  SKIP $M (relation_dataset.jsonl exists)"
    continue
  fi
  log "  BUILD $M"
  PYTHONPATH=src python - "$M" <<'PY'
import sys
sys.path.insert(0, "scripts/cache_matrix_20260722")
import importlib.util
spec = importlib.util.spec_from_file_location("build_ca_datasets", "scripts/cache_matrix_20260722/build_ca_datasets.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = sys.argv[1]
mod.MODELS[model] = "vt"   # patch dict
n = mod.build_one(model, "vt", force=False)
print(f"[build] {model}: {n} rows")
PY
done
log "Phase 1 done"

# -------------------------------------------------------------------
# Scheduler helpers: 2-GPU slot manager
# -------------------------------------------------------------------
declare -A gpu_busy
for g in "${GPUS[@]}"; do gpu_busy[$g]=""; done

grab_gpu() {
  while true; do
    for g in "${GPUS[@]}"; do
      if [ -z "${gpu_busy[$g]}" ] || ! kill -0 "${gpu_busy[$g]}" 2>/dev/null; then
        gpu_busy[$g]=""
        echo "$g"
        return 0
      fi
    done
    sleep 5
  done
}

# -------------------------------------------------------------------
# Phase 2: ca_tme x 3 enc x 3 seeds = 18 cells
# (run gru first because mn_tme_frozen depends on gru ckpt)
# -------------------------------------------------------------------
log "Phase 2: ca_tme (gru first, then lstm, then bilstm)"
for ENC in gru lstm bilstm; do
  for M in "${EXTRAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      G=$(grab_gpu)
      log "  RUN ca_tme enc=$ENC model=$M seed=$SEED gpu=$G"
      (
        bash scripts/cache_matrix_20260722/run_ca_tme.sh "$M" "$SEED" "$G" "$ENC"
      ) > "$LOG_DIR/ca_tme_${ENC}_${M}_seed${SEED}.extras.driver.log" 2>&1 &
      gpu_busy[$G]=$!
    done
  done
  # Wait for the current encoder batch before moving on (keeps gru ckpts ready
  # for phase 4 mn_tme_frozen as soon as gru wave finishes).
  log "  waiting on encoder=$ENC wave"
  for g in "${GPUS[@]}"; do
    [ -n "${gpu_busy[$g]}" ] && wait "${gpu_busy[$g]}" 2>/dev/null
    gpu_busy[$g]=""
  done
done
log "Phase 2 done"

# -------------------------------------------------------------------
# Phase 3 + 4: mn_tme_e2e and mn_tme_frozen in parallel
# (e2e has no dep; frozen depends on gru ckpt which is ready after phase 2)
# -------------------------------------------------------------------
log "Phase 3+4: mn_tme_e2e (no dep) + mn_tme_frozen (gru ckpt ready)"
PIDS=()
for M in "${EXTRAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    # e2e
    G=$(grab_gpu)
    log "  RUN mn_tme_e2e model=$M seed=$SEED gpu=$G"
    (
      bash scripts/cache_matrix_20260722/run_mn_tme_e2e.sh "$M" "$SEED" "$G"
    ) > "$LOG_DIR/mn_tme_e2e_${M}_seed${SEED}.extras.driver.log" 2>&1 &
    gpu_busy[$G]=$!

    # frozen
    G=$(grab_gpu)
    log "  RUN mn_tme_frozen model=$M seed=$SEED gpu=$G"
    (
      bash scripts/cache_matrix_20260722/run_mn_tme_frozen.sh "$M" "$SEED" "$G"
    ) > "$LOG_DIR/mn_tme_frozen_${M}_seed${SEED}.extras.driver.log" 2>&1 &
    gpu_busy[$G]=$!
  done
done

log "  waiting on mn_tme_e2e + mn_tme_frozen waves"
for g in "${GPUS[@]}"; do
  [ -n "${gpu_busy[$g]}" ] && wait "${gpu_busy[$g]}" 2>/dev/null
  gpu_busy[$g]=""
done
log "Phase 3+4 done"

# -------------------------------------------------------------------
# Phase 5: calibrate (uses ca_tme_gru seed20260717 best_checkpoint.pt)
# -------------------------------------------------------------------
log "Phase 5: calibrate thresholds"
for M in "${EXTRAS[@]}"; do
  G=$(grab_gpu)
  log "  RUN calibrate model=$M gpu=$G"
  (
    bash scripts/cache_matrix_20260722/run_calibrate.sh "$M" "$G"
  ) > "$LOG_DIR/calibrate_${M}.extras.driver.log" 2>&1 &
  gpu_busy[$G]=$!
done
log "  waiting on calibrate"
for g in "${GPUS[@]}"; do
  [ -n "${gpu_busy[$g]}" ] && wait "${gpu_busy[$g]}" 2>/dev/null
  gpu_busy[$g]=""
done
log "Phase 5 done"

# -------------------------------------------------------------------
# Phase 6: SDR pipeline
# -------------------------------------------------------------------
log "Phase 6: SDR + state patterns"
for M in "${EXTRAS[@]}"; do
  G=$(grab_gpu)
  log "  RUN sdr model=$M gpu=$G"
  (
    bash scripts/cache_matrix_20260722/run_sdr.sh "$M" "$G"
  ) > "$LOG_DIR/sdr_${M}.extras.driver.log" 2>&1 &
  gpu_busy[$G]=$!
done
log "  waiting on sdr"
for g in "${GPUS[@]}"; do
  [ -n "${gpu_busy[$g]}" ] && wait "${gpu_busy[$g]}" 2>/dev/null
  gpu_busy[$g]=""
done
log "Phase 6 done"

# -------------------------------------------------------------------
# Phase 7: re-aggregate + audit
# -------------------------------------------------------------------
log "Phase 7: aggregate + audit"
PYTHONPATH=src python scripts/cache_matrix_20260722/aggregate_results.py 2>&1 | tee -a "$LOG_DIR/aggregate_extras.log"
PYTHONPATH=src python scripts/cache_matrix_20260722/audit.py 2>&1 | tee -a "$LOG_DIR/audit_extras.log"

log "=== DONE extras chain ==="
