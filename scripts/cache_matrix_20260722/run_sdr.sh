#!/bin/bash
# cache_matrix_20260722: SDR + state pattern pipeline for one model.
#
# Consumes a frozen TME BiLSTM encoder checkpoint and runs the core SDR
# pipeline (state_dataset -> state_bundles -> relation_dataset -> frozen
# repr export -> SDR scores -> state patterns).
#
# Per-model prompt-conditioned cache manifests are built on first run via
# mprisk.setup_helper.setup_cache_manifests.
#
# Thresholds: the SDR pipeline requires per-model calibrated Aligned
# thresholds. These come from a separate calibration step (not in this
# script). If outputs/cache_matrix_20260722/thresholds/<MODEL>/thresholds.json is
# present we use it; otherwise we write a MISSING_THRESHOLDS marker and
# exit 0 (the driver/audit will flag missing pipelines).
#
# Args: MODEL GPU
# Output: outputs/cache_matrix_20260722/sdr/<MODEL>/
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

# Per-model setup output (prompt_cache_manifest + prompt_conditioned_cache).
SETUP_ROOT=outputs/cache_matrix_20260722/cache_manifests/$MODEL
PROMPT_CACHE_MANIFEST=$SETUP_ROOT/prompt_cache_manifest.jsonl
PROMPT_CONDITIONED_MANIFEST=$SETUP_ROOT/prompt_conditioned_cache/$MODEL/${PROTO,,}/$PROMPT_SET_KEY/manifest.jsonl

# Frozen TME BiLSTM encoder checkpoint (use seed 20260717 as canonical).
TME_RUN=outputs/cache_matrix_20260722/runs/tme_bilstm/${MODEL}_seed20260717
ENCODER_CKPT=$TME_RUN/best_encoder.pt

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

# Filtered cache dir: source cache reduced to the canonical prompt id.
FILTERED_CACHE_ROOT=outputs/cache_matrix_20260722/cache_manifests_filtered/$MODEL
FILTERED_MANIFEST=$FILTERED_CACHE_ROOT/unified_full_cache_manifest.json

# Enriched checkpoint: train_tme_e2e.py's best_encoder.pt remapped to the
# SphericalTME_BiLSTM state_dict layout + the SDR-pipeline-required fields.
ENRICHED_CKPT=$TME_RUN/best_encoder.enriched.pt

OUT_ROOT=outputs/cache_matrix_20260722/sdr/$MODEL
STATE_PATTERN_FILE=$OUT_ROOT/outputs/states/$MODEL/${PROTO^^}/$PROMPT_SET_KEY/tme_proxy_anchor_v1/state_patterns.jsonl

if [ -f "$STATE_PATTERN_FILE" ]; then
  echo "SKIP: $STATE_PATTERN_FILE exists"
  exit 0
fi

mkdir -p "$OUT_ROOT"
LOG=outputs/cache_matrix_20260722/_logs/sdr_${MODEL}.log
mkdir -p "$(dirname "$LOG")"

# Sanity checks.
if [ ! -f "$ENCODER_CKPT" ]; then
  echo "[FATAL] TME BiLSTM encoder missing for $MODEL: $ENCODER_CKPT" >&2
  echo "TME_BILSTM_ENCODER_MISSING" > "$OUT_ROOT/MISSING_DEPENDENCY"
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

# Build prompt cache manifests if absent.
if [ ! -f "$PROMPT_CACHE_MANIFEST" ] || [ ! -f "$PROMPT_CONDITIONED_MANIFEST" ]; then
  echo "[SDR] Building prompt cache manifests for $MODEL -> $SETUP_ROOT"
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

# Thresholds. Prefer new per-model directory emitted by driver_calibrate;
# fall back to legacy flat layout; then fall back to historical state_analysis.
THRESHOLDS_DIR=outputs/cache_matrix_20260722/thresholds/${MODEL}/thresholds.json
THRESHOLDS_FILE=outputs/cache_matrix_20260722/thresholds/${MODEL}.json
LEGACY_THRESHOLDS=outputs/state_analysis/${MODEL}/thresholds.json
if [ -f "$THRESHOLDS_DIR" ]; then
  THRESHOLDS_ARG=$THRESHOLDS_DIR
elif [ -f "$THRESHOLDS_FILE" ]; then
  THRESHOLDS_ARG=$THRESHOLDS_FILE
elif [ -f "$LEGACY_THRESHOLDS" ]; then
  THRESHOLDS_ARG=$LEGACY_THRESHOLDS
else
  echo "[WARN] No thresholds for $MODEL; writing marker and skipping SDR" | tee -a "$LOG"
  echo "MISSING_THRESHOLDS" > "$OUT_ROOT/MISSING_DEPENDENCY"
  exit 0
fi

# Filter source cache to canonical prompt (idempotent).
if [ ! -f "$FILTERED_MANIFEST" ]; then
  echo "[SDR] Filtering cache for $MODEL -> $FILTERED_CACHE_ROOT"
  PYTHONPATH=src python scripts/cache_matrix_20260722/filter_cache_manifest.py \
    --source-cache-root "$SOURCE_CACHE_ROOT" \
    --target-cache-root "$FILTERED_CACHE_ROOT" \
    --canonical-prompt pregen_risk_v1_p001 2>&1 | tee "$LOG"
fi

# Enrich the train_tme_e2e.py checkpoint for the SDR pipeline (idempotent).
if [ ! -f "$ENRICHED_CKPT" ] || [ "$ENCODER_CKPT" -nt "$ENRICHED_CKPT" ]; then
  echo "[SDR] Enriching checkpoint for $MODEL -> $ENRICHED_CKPT"
  PYTHONPATH=src python scripts/cache_matrix_20260722/enrich_checkpoint.py \
    --in-ckpt "$ENCODER_CKPT" \
    --out-ckpt "$ENRICHED_CKPT" \
    --encoder-type bilstm \
    --model-key "$MODEL" \
    --protocol "${PROTO,,}" \
    --prompt-set "$PROMPT_SET" \
    --layer-count "$LAYER_COUNT" 2>&1 | tee "$LOG"
fi

# Reuse calibration's frozen embeddings so the SDR scoring identity matches
# the thresholds' identity binding (export_frozen_representations is non-
# deterministic across GPU runs due to floating-point reduction order).
CALIB_EMBEDDING_MANIFEST=outputs/cache_matrix_20260722/thresholds/$MODEL/outputs/embeddings/$MODEL/${PROTO^^}/$PROMPT_SET_KEY/tme_proxy_anchor_v1/spherical_embedding_manifest.jsonl
EMBEDDING_ARG=""
if [ -f "$CALIB_EMBEDDING_MANIFEST" ]; then
  EMBEDDING_ARG="--embedding-manifest-path $CALIB_EMBEDDING_MANIFEST"
  echo "[SDR] Reusing calibration embeddings: $CALIB_EMBEDDING_MANIFEST"
else
  echo "[SDR] No calibration embeddings found at $CALIB_EMBEDDING_MANIFEST; will re-export"
fi

echo "[SDR] MODEL=$MODEL GPU=$GPU proto=$PROTO ckpt=$ENRICHED_CKPT (enriched) thresholds=$THRESHOLDS_ARG"
PYTHONPATH=src python scripts/run_core_sdr_pipeline.py \
  --model-key "$MODEL" \
  --protocol "${PROTO,,}" \
  --prompt-set-key "$PROMPT_SET_KEY" \
  --repr-key tme_proxy_anchor_v1 \
  --manifest-paths "$MANIFEST" \
  --full-cache-root "$FILTERED_CACHE_ROOT" \
  --cache-manifest-path "unified_full_cache_manifest.json" \
  --prompt-cache-manifest "$PROMPT_CACHE_MANIFEST" \
  --prompt-conditioned-cache-manifest "$PROMPT_CONDITIONED_MANIFEST" \
  --prompt-set "$PROMPT_SET" \
  --split-assignment "$SPLIT" \
  --thresholds "$THRESHOLDS_ARG" \
  --checkpoint "$ENRICHED_CKPT" \
  --output-root "$OUT_ROOT" \
  $EMBEDDING_ARG \
  2>&1 | tee "$LOG"

touch "${LOG}.done"
echo "[SDR] DONE -> $STATE_PATTERN_FILE"
