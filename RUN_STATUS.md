# Run Status

## Three-model prefill cache

- Code node: Qwen3-VL-8B VT, InternVL3.5-8B VT, and Qwen2.5-Omni-7B VA.
- Prompt protocol: `prefill_main_p8_v1`, subset seed `20260717`, P=8.
- Smoke protocol: one Conflict and one Aligned sample per model, 48 tasks per model.
- GPU policy: physical GPU 1 only through `CUDA_VISIBLE_DEVICES=1`; never GPU 0.
- Runtime status: pending post-commit smoke validation.
- Full-batch status: not started; starts resumably only after the matching smoke passes.
- Misread: excluded and rejected by the batch planner.
