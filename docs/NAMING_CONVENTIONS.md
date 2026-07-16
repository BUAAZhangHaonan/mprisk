# Naming Conventions

This glossary is the canonical naming contract for active code, configs, artifacts, and paper
exports. A name may identify one concept only.

## Roles and outputs

| Concept | Canonical name |
|---|---|
| Multimodal model being measured | `subject_model_key` |
| Model generating reference descriptions | `gt_generator_model` |
| Model judging Misread | `judge_model` |
| Human/reference description | `GT_DESCRIPTION` / GT Description |
| Subject-model M12 description | `DIAGNOSTIC_AFFECT_DESCRIPTION` / Diagnostic Affect Description |
| Blinded comparison task | Misread Judgment |

GT Description generation and Misread judging are separate stages. DeepSeek is a provider/model,
not the task name.

GT Description generation selects adapters with `provider_key`. Vendor-specific settings and
credentials live only under `provider_settings` and `ground_truth/providers/`; task modules never
name a vendor API or credential. The active generation config is v3 because this provider boundary
changes its schema. Unknown providers and settings fail explicitly and never select an alternative.

## Protocol, modality, and condition

- A **protocol** is `VT` or `VA`.
- A **modality** is `V`, `T`, or `A`.
- A **condition** is `M1`, `M2`, or `M12`.
- Under VT, M1=V, M2=T, and M12=V+T.
- Under VA, M1=V, M2=A, and M12=V+A.

M1 and M2 must never be values of a modality or lean-direction field.

## Labels and state

- Representation learning uses `Conflict` / `Aligned` sample-type labels. `Ambiguous` may exist at
  data ingestion but is excluded from the locked TME experiments.
- Archive class codes `A` and `C` are provenance only: `source_class_code`.
- Misread evaluation uses `Misread` / `Non-misread` only after the representation is frozen.
- The three state indices are State Dispersion (`S`), Modality Split (`D`), and signed Joint Lean
  (`R`). Their categorical result is State Pattern.
- `joint_lean_direction` uses `V`, `T_or_A`, `No-lean`, or explicit `unclear`.
- `reference_dominant_modality` is a separate human/reference annotation with values `V`, `T`,
  `A`, `Balanced`, or `Unclear`. It must never be used as the signed Joint Lean output.
- TME means Trajectory Manifold Encoder. Its representation target is Conflict/Aligned; Proxy
  Anchor is the training objective, not part of the representation name.

The currently running cache/downstream identity retains its existing state API and figure keys
until that run finishes. This glossary does not authorize in-place migration of ledgers,
manifests, sidecars, checkpoints, prompt IDs, cache roots, or figure-map keys.

## Version suffixes

- Schemas end in `_vN`, starting at `_v1`.
- A schema version is immutable. A field or semantic change increments `N` and writes a new
  artifact directory.
- Config filenames end in `_vN.yaml`; generated artifact directories end in `/vN` or contain the
  same explicit version token.
- Generic modules and symbols describe the task. Model, protocol, condition, dataset, and split
  belong in configs and provenance, not module names.
- Legacy configs live under `configs/legacy/` and are read-only records. Active loaders reject
  their schemas; there are no import aliases, symlinks, or silent dual-schema readers.

Diagnostic Affect Description v2 uses `schema_name` consistently in configs, manifests,
signatures, and provenance. Every artifact also carries the same immutable `run_id`. Misread
Judgment v2 names the measured model as `subject_model_key` and the external judge as
`judge_model`; it reads only `GT_DESCRIPTION` and `DIAGNOSTIC_AFFECT_DESCRIPTION`.

The label-schema transition follows this rule exactly:

- `mprisk_stage1_relation_schema_v1` and `mprisk_sample_type_schema_v1` are immutable. The current
  state-dataset and state-bundle code is an isolated legacy consumer of v1-shaped fields, not a
  strict schema-ID binding.
- `mprisk_condition_affect_annotation_schema_v2` and `mprisk_sample_relation_schema_v2` define
  the canonical M1/M2/M12 and reference-dominance terminology for new annotation artifacts.
- No active loader selects v2 by filename discovery. A new pipeline needs a separate strict
  consumer that binds both v2 schema IDs and writes a new artifact identity.

## Deferred naming migrations

The current cache, representation, state, and figure pipelines are running under immutable
ledgers. Their `sdr` module/API names and existing AP/AUPRC result keys are intentionally deferred
until those runs finish. Renaming them now would change runtime identity and invalidate resumable
artifacts. This deferment does not define new paper terminology and does not authorize aliases.
