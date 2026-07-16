# Misread Judgment

`scripts/run_misread_judgment.py` compares each `GT_DESCRIPTION` with the matching
`DIAGNOSTIC_AFFECT_DESCRIPTION`. The fixed prompt asks the judge for one strict JSON decision:
`MISREAD`, `NON_MISREAD`, or `UNCERTAIN`, with confidence and one short rationale sentence.

The v2 identity contract uses `mprisk_misread_judgment_config_v2`. It binds `run_id`,
`judge_model`, `subject_model_key`, `protocol`, `split`, both manifest checksums, the prompt,
temperature, and confidence threshold. The service request is blinded and contains only the two
canonical description fields. It never reads a legacy `text` field or assumes a fixed row count.

The active config at `configs/judge/misread_judgment_v2.yaml` has `status: pending` and generic
future paths because Misread annotations and complete Diagnostic Affect Description manifests are
not available yet. A pending config fails before reading manifests or calling the provider. Change
it to `status: ready` only after replacing all identity fields and paths with one real frozen run.

All `UNCERTAIN` decisions and results below the configured confidence threshold enter the human
review queue. Final binary labels are exported only after human decisions exactly cover that queue.

The previous 162-row Qwen-specific v1 config and instructions are preserved under
`configs/legacy/judge/` and `docs/legacy/`. They are read-only provenance, not an active interface.
