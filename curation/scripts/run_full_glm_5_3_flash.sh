#!/bin/bash
# Full GLM-5.3-flash annotation chain: annotate -> aggregate -> adjudicate -> report.
#
# Required environment:
#   MPRISK_CURATION_DB    path to curation.sqlite
#   MPRISK_CURATION_DATA  curation data root (holds glm_5_3_flash_annotation.sqlite
#                         and media/frames96)
# Optional environment (passed through to the python script):
#   MPRISK_META_CSV       ground-truth meta.csv, enables the report truth comparison
#   GLM_MODEL, GLM_CONC, GLM_MAX_FRAMES, GLM_FRAME_WIDTH
#   GLM_PROXY             if non-empty, exported as http_proxy/https_proxy
#   PYTHON                python interpreter (default: python3)
set -e
cd "$(dirname "$0")/.."

: "${MPRISK_CURATION_DB:?must point to curation.sqlite}"
: "${MPRISK_CURATION_DATA:?must point to the curation data root}"

if [[ -n "${GLM_PROXY:-}" ]]; then
  export http_proxy="$GLM_PROXY" https_proxy="$GLM_PROXY"
fi

PYTHON="${PYTHON:-python3}"

echo "[chain] annotate start $(date)"
"$PYTHON" scripts/run_glm_5_3_flash_annotation.py --phase annotate --modalities V,T
echo "[chain] annotate done $(date)"
"$PYTHON" scripts/run_glm_5_3_flash_annotation.py --phase aggregate
echo "[chain] aggregate done $(date)"
"$PYTHON" scripts/run_glm_5_3_flash_annotation.py --phase adjudicate
echo "[chain] adjudicate done $(date)"
"$PYTHON" scripts/run_glm_5_3_flash_annotation.py --phase report
echo "[chain] ALL DONE $(date)"
