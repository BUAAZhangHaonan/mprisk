# Diagnostic Affect Description Generation

`scripts/generate_diagnostic_affect_descriptions.py` generates exactly one open-ended
Diagnostic Affect Description for each selected subject-model input. The task is model-independent:
the configuration names the subject model, protocol, condition, dataset, and split separately.

The canonical prompt is fixed by
`mprisk.diagnostic_affect.generation.CANONICAL_DIAGNOSTIC_AFFECT_PROMPT`:

> Based on the complete input, describe the person's overall emotional state in one concise
> sentence. Do not address the person, give advice, or explain your reasoning.

The active v2 implementation accepts `condition: M12`. For `protocol: VT`, it sends video plus
the manifest's `text_content`. For `protocol: VA`, it sends the configured vision and audio media
without adding transcript text. Sample type, annotations, archetype, trigger, and GT fields never
enter the subject-model request.

## Identity contract

The strict config schema is `mprisk_diagnostic_affect_description_config_v2`. It requires:

- a non-empty `run_id`, `subject_model_key`, and an exact `model_path` matching `asset_config`;
- `protocol`, `condition`, `dataset`, and `split`;
- one standard processed `manifest_path` and a new `output_root`;
- deterministic generation settings (`do_sample=false`, `num_beams=1`).

The subject model's family is resolved through the existing model asset registry and wrapper
registry. Diagnostic code does not select a Qwen, InternVL, or other family by name.

The output schema is `mprisk_diagnostic_affect_description_v2`. Every manifest row and provenance
record uses `schema_name` and carries the immutable `run_id`. Its semantic output field is
`DIAGNOSTIC_AFFECT_DESCRIPTION`. A SQLite ledger binds the config, asset registry, source manifest,
model identity, prompt, condition, dataset, split, and generation policy. Resume fails if that
identity changes. JSONL and JSON artifacts are written atomically and checksummed.

## Commands

```bash
PYTHONPATH=src python scripts/generate_diagnostic_affect_descriptions.py \
  --config configs/experiments/diagnostic_affect_description_v2.yaml

PYTHONPATH=src python scripts/generate_diagnostic_affect_descriptions.py \
  --config configs/experiments/diagnostic_affect_description_v2.yaml \
  --smoke --output-root outputs/diagnostic_affect/smoke
```

The smoke selector chooses one Conflict and one Aligned sample within the configured
dataset/split/protocol. Explicit sample IDs are accepted only with `--smoke`.

```bash
PYTHONPATH=src python scripts/verify_diagnostic_affect_descriptions.py \
  --manifest-path data/processed/manifests/protocol_manifests/va_aux.jsonl \
  --output-root outputs/diagnostic_affect/smoke \
  --subject-model-key qwen2_5_omni_7b \
  --run-id qwen2_5_omni_7b_ch_sims_v2_test_va_m12_v2 \
  --protocol VA --condition M12 \
  --dataset ch_sims_v2 --split test --smoke
```

## Legacy artifact

The frozen 162-row Qwen2.5-Omni artifact remains unchanged at
`outputs/diagnostics/qwen2_5_omni_7b_m12_v1/`. Its old config is retained only under
`configs/legacy/experiments/` to document that read-only run. The active generator does not read,
resume, migrate, or overwrite that directory.
