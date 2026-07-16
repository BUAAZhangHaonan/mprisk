# Misread Judgment

`scripts/run_misread_judgment.py` compares each `GT_DESCRIPTION` with the matching
`DIAGNOSTIC_AFFECT_DESCRIPTION`. The fixed prompt asks the judge for one strict JSON decision:
`MISREAD`, `NON_MISREAD`, or `UNCERTAIN`, with confidence and one short rationale sentence.

The v2 identity contract uses `mprisk_misread_judgment_config_v2`. It binds `run_id`,
`judge_model`, `subject_model_key`, `protocol`, `split`, both manifest checksums, the prompt,
temperature, confidence threshold, `mprisk_gt_description_v1` with
`gt_annotation_input_v1`, and the exact Diagnostic Affect Description schema and run ID. Every
manifest row is checked against these identities before a request is built. Legacy GT schemas,
handwritten rows without schema identity, and Diagnostic Affect Description rows from another run
fail closed. The service request is blinded and contains only the two canonical description fields.

The active config at `configs/judge/misread_judgment_v2.yaml` has `status: pending` and generic
future paths because Misread annotations and complete Diagnostic Affect Description manifests are
not available yet. A pending config fails before reading manifests or calling the provider. Change
it to `status: ready` only after replacing all identity fields and paths with one real frozen run.

All `UNCERTAIN` decisions and results below the configured confidence threshold enter the human
review queue. The public `misread_labels.jsonl` artifact is exported only after human decisions
exactly cover that queue. Each `mprisk_misread_label_v1` row contains the canonical
`misread_label` value `Misread` or `Non-misread`; `misread_binary_label` is explicitly defined as
1 for Misread and 0 for Non-misread. The accompanying `misread_labels_provenance.json` binds the
source decisions, human review, label schema, counts, and checksums.

The previous 162-row Qwen-specific v1 config and instructions are preserved under
`configs/legacy/judge/` and `docs/legacy/`. They are read-only provenance, not an active interface.
