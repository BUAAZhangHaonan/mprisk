# GT Description Generation

This stage adds exactly one `GT_DESCRIPTION` field to each strict GT annotation input.
The task is named GT Description generation; DeepSeek is the current provider adapter.

## Annotation input

`scripts/build_gt_annotation_input_pilot.py` creates a new immutable artifact under
`data/frozen/generated_round1_v1/ground_truth_inputs/gt_annotation_input_v1/`.
It does not edit or read the old pilot as a fallback.

Each `mprisk_gt_annotation_input_v1` row contains:

- `sample_type`: `Conflict` or `Aligned`;
- `archetype` with one canonical meaning;
- `dialogue`, `scenario_context`, and nullable `surface_emotion`;
- `scenario_context_source`: `setting`, `trigger`, or `source_prompt`;
- protocol, media, and explicit source provenance.

The legacy archive codes `A` and `C` are mapped once at ingestion and retained only as
`source_provenance.source_class_code`. The legacy raw field `ltx2_prompt` is exposed as
`scenario_context_source: source_prompt`; it is never relabeled as a setting or trigger.

## Generator contract

The only active config schema is `mprisk_gt_description_generation_config_v3`.
It uses the task-level fields `provider_key`, `gt_generator_model`, `provider_settings`,
`conflict_prompt_path`, and `aligned_prompt_path`. The selected adapter strictly validates the
entire `provider_settings` mapping. An unknown provider or setting is an error; there is no
alternate-provider fallback. The annotation-input schema version, manifest SHA-256, expected row
count, provider, provider-settings SHA-256, model, prompts, and output directory are bound into the
ledger identity.

The provider request contains only `archetype`, `dialogue`, `scenario_context`, and
`surface_emotion`. It excludes sample type, protocol, media, source provenance, and future model
outputs. Conflict and Aligned choose different fixed prompts before the request is built.

The active DeepSeek adapter owns its endpoint, API-key environment variable, timeout, token,
temperature, and thinking settings. The generic task does not import or instantiate a vendor
client. Thinking is disabled and temperature is zero. The response must be exact JSON with only
`GT_DESCRIPTION`, containing one English declarative sentence. Invalid content is recorded as a
failure without repair. Only transport errors and configured retryable HTTP statuses are retried.

The final manifest uses the distinct `mprisk_gt_description_v1` row schema. It preserves every
annotation-input field except the input-only `schema_name`, records the generation `run_id`, keeps
`gt_input_schema_version`, and adds only `GT_DESCRIPTION`. Raw request/response evidence, attempts,
failures, provenance, and pending human review status are exported separately. Provenance records
both the output schema and the annotation-input schema version.

```bash
PYTHONPATH=src python scripts/build_gt_annotation_input_pilot.py
PYTHONPATH=src python scripts/run_gt_description_generation.py \
  --config configs/ground_truth/gt_description_generation_pilot.yaml
PYTHONPATH=src python scripts/verify_gt_description_generation.py \
  --config configs/ground_truth/gt_description_generation_pilot.yaml \
  --require-complete
```

Resume processes pending rows only. `--retry-failed` is explicit. The frozen old GT inputs,
configs, prompts, and outputs remain read-only legacy evidence and are never silently loaded by the
new schema. The superseded pilot config is a legacy record and is never accepted as v3.
