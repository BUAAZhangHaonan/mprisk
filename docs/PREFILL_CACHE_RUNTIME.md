# Three-model prefill cache runtime

The fixed runtime panel is `qwen3_vl_8b` and `internvl3_5_8b` for VT, plus
`qwen2_5_omni_7b` for VA. Model families and checkpoint paths resolve from
`configs/assets/model_assets.yaml`; batch commands must not substitute Qwen2.5-VL for
Qwen3-VL.

`configs/cache/prefill_main_p8_v1.yaml` freezes immutable prompt-subset selection seed
`20260717`; it is an identifier, not a run date. The config also freezes the VT/VA P=8
prompt sets, protocol manifests, smoke manifests, and output roots. Each smoke manifest
contains one Conflict and one Aligned sample. Two samples times M1/M2/M12 times P=8 gives
48 cache tasks per model. Misread rows are rejected.

## Condition views

- VT M1: video only; M2: transcript only; M12: video plus transcript.
- VA M1: silent video; M2: audio; M12: video with its embedded audio.

Qwen3-VL uses its processor chat template with native video or multiple image content and
an explicit model forward. InternVL3.5 follows the official dynamic frame slicing,
`num_patches_list`, frame-prefix, and image-context token contract, then calls
`language_model` directly. Qwen2.5-Omni remains Thinker-only. All wrappers take
`hidden_states[1:]` at the last non-padding conditioning token.

## GPU 1 smoke

Expose only physical GPU 1; the process-local device is `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
  /home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python \
  scripts/extract_prefill_batch.py \
  --manifest data/frozen/prefill_smoke_v1/qwen3_vl_8b.jsonl \
  --prompt-set configs/prompts/equiv_sets/vt_main_p8_seed20260717.yaml \
  --protocol vt --model-key qwen3_vl_8b --device cuda:0 \
  --output-root outputs/prefill_smoke/qwen3_vl_8b/v1 --fail-fast
```

Use the same call for `internvl3_5_8b`, adding `--video-num-segments 8`, and use the VA
prompt/smoke manifests for `qwen2_5_omni_7b`. The model path and family are always read
from the asset registry.

## Recovery contract

The batch signature hashes the asset registry, source manifest, prompt set, prompt values,
model runtime settings, and condition protocol. A mismatched rerun fails before extraction.
Identical reruns validate each sidecar request, safetensors checksum, and tensor shape.
Interrupted tasks return to pending; failed tasks require `--retry-failed`.

Every task records sample, model, protocol, prompt set, prompt, and condition identity.
Safetensors shards contain the float32 `[layer_count, hidden_dim]` trajectory. Sidecars,
the combined `manifest.jsonl`, per-prompt manifests, and `batch_state.sqlite3` retain shape,
t0, elapsed time, peak allocated GPU memory, and checksum.

## Source/target scheduling

The canonical complete-matrix config sets
`execution.allow_parallel_domain_extraction: true`. Source and target are independent
cache domains with separate output roots, ledgers, scoped runtime records, and locks.
The stage controller may therefore start a target lane as soon as that physical GPU lane
has no source supervisor ownership marker. It adopts an already-active canonical target
supervisor after a controller restart and never starts a duplicate session or process.
Before any parallel target launch or adoption, it writes
`EXTRACTION_LAUNCH_AUDIT.json` only after the complete matrix is launchable and every
source and target asset signature passes.

A source session or scoped source lock owns its GPU lane until both markers disappear.
In particular, target GPU 0 cannot start while the Phi-4 source queue owns GPU 0. An idle
GPU 1 may run target extraction at the same time. When source GPU 0 finishes and releases
its session and lock, target GPU 0 may start.

Parallel extraction does not relax completion. Source and target each require their
expected complete/accepted job counts, zero missing tasks, matching asset signatures,
and finalized supervisors. The controller writes `FINAL_CACHE_AUDIT.json` and reports
`complete` only after one full strict audit proves both domains complete. Bundle and
publication steps consume that final audit and remain fail-closed.
