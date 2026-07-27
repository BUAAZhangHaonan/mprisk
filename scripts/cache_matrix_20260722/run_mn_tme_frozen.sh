#!/usr/bin/env bash
# M/N TME(frozen) training — load C/A TME-GRU encoder checkpoint, freeze encoder,
# train only the M/N head (CE loss). Falls back to nothing if checkpoint missing.
#
# Usage: run_mn_tme_frozen.sh MODEL SEED GPU
# Deps:  outputs/cache_matrix_20260722/runs/ca_tme/<model>_seed<seed>/best_checkpoint.pt
#        (arch layer_l2_gru_linear_relation_v1, written by C/A TME driver)
# Output: outputs/cache_matrix_20260722/runs/mn_tme_frozen/<model>_seed<seed>/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${MPRISK_ROOT:-$SCRIPT_DIR/../..}"
source /home/team/zhanghaonan/miniconda3/etc/profile.d/conda.sh
conda activate mprisk

MODEL=${1:?MODEL required}
SEED=${2:?SEED required}
GPU=${3:?GPU required}
ENCODER_TYPE=gru
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src

case "$MODEL" in
  qwen2_5_omni_7b|gemma4_12b_it|gemma4_12b|phi4_multimodal)
    PROTO=va ;;
  *)
    PROTO=vt ;;
esac

SPLIT=outputs/cache_matrix_20260722/split_assignments/${PROTO,,}.jsonl
MANIFEST=data/processed/manifests/protocol_manifests_merged/${PROTO,,}_merged_primary.jsonl
PROMPT_SET=configs/prompts/equiv_sets/${PROTO,,}_main_p8_seed20260717.yaml
JUDGMENTS=outputs/misread/$MODEL/judgments.jsonl

if [ "$MODEL" = "internvl3_5_8b" ]; then
  CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/cache_manifests/internvl3_5_8b
else
  CACHE_ROOT=/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/source/$MODEL
fi

# C/A TME-GRU best_checkpoint (GRU arch, condition_encoder.* keys).
PA_CKPT=outputs/cache_matrix_20260722/runs/ca_tme/${MODEL}_seed${SEED}/best_checkpoint.pt

OUT=outputs/cache_matrix_20260722/runs/mn_tme_frozen/${MODEL}_seed${SEED}
if [ -f "$OUT/metrics.json" ]; then
  echo "SKIP: $OUT/metrics.json exists"
  exit 0
fi
mkdir -p "$OUT"

LOG=outputs/cache_matrix_20260722/_logs/mn_tme_frozen_${MODEL}_seed${SEED}.log
mkdir -p "$(dirname "$LOG")"

[ -f "$JUDGMENTS" ] || { echo "[FATAL] misread judgments missing: $JUDGMENTS" >&2; exit 2; }
[ -f "$SPLIT" ]     || { echo "[FATAL] split missing: $SPLIT" >&2; exit 2; }
[ -f "$MANIFEST" ]  || { echo "[FATAL] manifest missing: $MANIFEST" >&2; exit 2; }
[ -f "$CACHE_ROOT/manifest.jsonl" ] || { echo "[FATAL] cache manifest missing: $CACHE_ROOT/manifest.jsonl" >&2; exit 2; }
[ -f "$PA_CKPT" ]   || { echo "[FATAL] C/A checkpoint missing: $PA_CKPT (wait for C/A TME driver to finish)" >&2; exit 3; }

echo "[MN-TME-FROZEN] MODEL=$MODEL SEED=$SEED GPU=$GPU encoder=$ENCODER_TYPE proto=$PROTO"
echo "[MN-TME-FROZEN] warm-start encoder from $PA_CKPT then freeze"

# Inline Python launcher: monkey-patch TME_E2E_v3B.__init__ to freeze encoder
# right after construction (so warm-start still loads condition_encoder.*,
# but trainable list excludes encoder params -> only head trains).
PYTHONPATH=src python - <<PYEOF 2>&1 | tee "$LOG"
import importlib
import sys

# Make sure train_tme_e2e is importable as a module (it is on sys.path[0]=scripts)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "train_tme_e2e", "scripts/train_tme_e2e.py"
)
tme_mod = importlib.util.module_from_spec(spec)
sys.modules["train_tme_e2e"] = tme_mod
spec.loader.exec_module(tme_mod)

# Patch __init__: after original init finishes, freeze encoder params.
orig_init = tme_mod.TME_E2E_v3B.__init__

def patched_init(self, **kwargs):
    orig_init(self, **kwargs)
    for p in self.encoder.parameters():
        p.requires_grad_(False)
    n_enc = sum(p.numel() for p in self.encoder.parameters())
    n_head = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print(
        f"[freeze] encoder frozen ({n_enc:,} params); "
        f"trainable head only ({n_head:,} params)",
        file=sys.stderr, flush=True,
    )

tme_mod.TME_E2E_v3B.__init__ = patched_init

# Rebuild argv and call main.
sys.argv = [
    "train_tme_e2e.py",
    "--task", "misread",
    "--model-key", "$MODEL",
    "--split-assignment", "$SPLIT",
    "--misread-judgments", "$JUDGMENTS",
    "--cache-roots", "$CACHE_ROOT",
    "--prompt-set", "$PROMPT_SET",
    "--main-manifest", "$MANIFEST",
    "--encoder-type", "$ENCODER_TYPE",
    "--tme-pa-checkpoint", "$PA_CKPT",
    "--max-epochs", "100",
    "--batch-size", "32",
    "--device", "cuda:0",
    "--seed", "$SEED",
    "--lr", "5e-4",
    "--weight-decay", "1e-4",
    "--dropout", "0.3",
    "--sequence-hidden-dim", "256",
    "--embed-dim", "128",
    "--head-hidden-dim", "32",
    "--output-dir", "$OUT",
]
tme_mod.main()
PYEOF

EXIT=${PIPESTATUS[0]}
if [ $EXIT -eq 0 ]; then
  touch "${LOG}.done"
fi
echo "[MN-TME-FROZEN] DONE -> $OUT/metrics.json"
exit $EXIT
