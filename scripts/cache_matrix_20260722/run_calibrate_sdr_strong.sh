#!/bin/bash
# cache_matrix_20260722: threshold calibration for the sdr_strong extension run.
# Mirror of run_calibrate.sh but checkpoint comes from
# runs/ca_tme_gru_sdr_strong/ and outputs go to thresholds_sdr_strong/<MODEL>/.
# Shared inputs (cache_manifests/, cache_manifests_filtered/) are reused
# read-only from the canonical sweep; InternVL SOURCE_CACHE_ROOT special case kept.
#
# Args: MODEL GPU
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODEL=${1:?MODEL required}
GPU=${2:?GPU required}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src

if [ "$MODEL" = "qwen2_5_omni_7b" ] || [ "$MODEL" = "gemma4_12b_it" ] || [ "$MODEL" = "gemma4_12b" ] || [ "$MODEL" = "phi4_multimodal" ]; then
  PROTO=va
else
  PROTO=vt
fi
PROMPT_SET_KEY=${PROTO,,}_main_p8_seed20260717
PROMPT_SET=configs/prompts/equiv_sets/${PROTO,,}_main_p8_seed20260717.yaml
SPLIT=data/processed/manifests/splits/representation_v1/representation_split_assignment_v1_${PROTO,,}.jsonl
MANIFEST=data/processed/manifests/protocol_manifests_merged/${PROTO,,}_merged_primary.jsonl

if [ "$MODEL" = "internvl3_5_8b" ]; then
  SOURCE_CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/cache_manifests/internvl3_5_8b
else
  SOURCE_CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/source/$MODEL
fi

SETUP_ROOT=outputs/cache_matrix_20260722/cache_manifests/$MODEL
PROMPT_CACHE_MANIFEST=$SETUP_ROOT/prompt_cache_manifest.jsonl
PROMPT_CONDITIONED_MANIFEST=$SETUP_ROOT/prompt_conditioned_cache/$MODEL/${PROTO,,}/$PROMPT_SET_KEY/manifest.jsonl

TME_RUN=outputs/cache_matrix_20260722/runs/ca_tme_gru_sdr_strong/${MODEL}_seed20260717
ENCODER_CKPT=$TME_RUN/best_checkpoint.pt

# layer_count is model-specific; read it from the source cache manifest's
# first row. Falls back to 36 if the manifest is missing or unreadable.
LAYER_COUNT=$(python3 -c "
import json, sys
try:
    with open('$SOURCE_CACHE_ROOT/manifest.jsonl') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            print(int(row.get('layer_count', 36)))
            sys.exit(0)
except Exception:
    pass
print(36)
" 2>/dev/null || echo 36)

# Filtered cache dir (shared with canonical sweep; rebuilt only if missing).
FILTERED_CACHE_ROOT=outputs/cache_matrix_20260722/cache_manifests_filtered/$MODEL
FILTERED_MANIFEST=$FILTERED_CACHE_ROOT/unified_full_cache_manifest.json

ENRICHED_CKPT=$TME_RUN/best_checkpoint.pt  # C/A GRU ckpt is already SDR-ready

OUT_ROOT=outputs/cache_matrix_20260722/thresholds_sdr_strong/$MODEL
mkdir -p "$OUT_ROOT"

LOG=outputs/cache_matrix_20260722/_logs/calibrate_sdr_strong_${MODEL}.log
mkdir -p "$(dirname "$LOG")"

# Idempotency: provenance sidecar with matching checkpoint sha = done.
TARGET=$OUT_ROOT/thresholds.json
if [ -f "$TARGET" ] && [ -f "$OUT_ROOT/calibration_provenance.json" ]; then
  echo "[calibrate-strong] SKIP: $TARGET already exists" | tee -a "$LOG"
  exit 0
fi

# Sanity checks.
if [ ! -f "$ENCODER_CKPT" ]; then
  echo "[FATAL] sdr_strong encoder missing for $MODEL: $ENCODER_CKPT" >&2
  echo "TME_STRONG_ENCODER_MISSING" > "$OUT_ROOT/MISSING_DEPENDENCY"
  exit 2
fi
if [ ! -f "$SOURCE_CACHE_ROOT/manifest.jsonl" ]; then
  echo "[FATAL] source cache manifest missing: $SOURCE_CACHE_ROOT/manifest.jsonl" >&2
  exit 2
fi
if [ ! -f "$SPLIT" ] || [ ! -f "$MANIFEST" ]; then
  echo "[FATAL] split or label manifest missing" >&2
  exit 2
fi

# Build prompt cache manifests if absent (shared; canonical sweep already built them).
if [ ! -f "$PROMPT_CACHE_MANIFEST" ] || [ ! -f "$PROMPT_CONDITIONED_MANIFEST" ]; then
  echo "[calibrate-strong] Building prompt cache manifests for $MODEL -> $SETUP_ROOT"
  PYTHONPATH=src python - <<EOF
from pathlib import Path
from mprisk.setup_helper import setup_cache_manifests
setup_cache_manifests(
    cache_root="$SOURCE_CACHE_ROOT",
    prompt_set_path="$PROMPT_SET",
    model_key="$MODEL",
    output_root=Path("$SETUP_ROOT"),
)
EOF
fi

# Filter source cache to canonical prompt (shared; rebuilt only if missing).
if [ ! -f "$FILTERED_MANIFEST" ]; then
  echo "[calibrate-strong] Filtering cache for $MODEL -> $FILTERED_CACHE_ROOT"
  PYTHONPATH=src python scripts/cache_matrix_20260722/filter_cache_manifest.py \
    --source-cache-root "$SOURCE_CACHE_ROOT" \
    --target-cache-root "$FILTERED_CACHE_ROOT" \
    --canonical-prompt pregen_risk_v1_p001 2>&1 | tee -a "$LOG"
fi

echo "[calibrate-strong] MODEL=$MODEL GPU=$GPU proto=$PROTO ckpt=$ENRICHED_CKPT"
PYTHONPATH=src python scripts/cache_matrix_20260722/calibrate_thresholds.py \
  --model "$MODEL" \
  --protocol "${PROTO,,}" \
  --prompt-set-key "$PROMPT_SET_KEY" \
  --prompt-set "$PROMPT_SET" \
  --manifest "$MANIFEST" \
  --full-cache-root "$FILTERED_CACHE_ROOT" \
  --prompt-cache-manifest "$PROMPT_CACHE_MANIFEST" \
  --prompt-conditioned-cache-manifest "$PROMPT_CONDITIONED_MANIFEST" \
  --split-assignment "$SPLIT" \
  --checkpoint "$ENRICHED_CKPT" \
  --output-dir "$OUT_ROOT" \
  --device cuda \
  2>&1 | tee -a "$LOG"

touch "${LOG}.done"
echo "[calibrate-strong] DONE -> $TARGET"
