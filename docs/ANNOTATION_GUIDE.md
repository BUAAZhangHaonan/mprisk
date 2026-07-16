# Annotation Guide

This document records the versioned annotation surfaces. New annotation artifacts use
`mprisk_condition_affect_annotation_schema_v2` together with
`mprisk_sample_relation_schema_v2`; a producer must select both schemas explicitly.

## Screening

Screening decides whether a sample is `Conflict`, `Ambiguous`, or `Aligned`.

The annotation unit has three conditions:

- `M1`: first unimodal condition.
- `M2`: second unimodal condition.
- `M12`: joint multimodal condition.

Annotators label each condition with `positive`, `negative`, `neutral`, `uncertain`, or `invalid`.

## Stage 1 Labels

Stage 1 labels describe the relation between `M1`, `M2`, and `M12`.

Required fields:

```text
m1_label
m2_label
m12_label
m1_specific_affect
m2_specific_affect
m12_specific_affect
m1_is_clear
m2_is_clear
m12_is_clear
m1_confidence
m2_confidence
m12_confidence
sample_type
reference_dominant_modality
quality_flags
notes
```

`reference_dominant_modality` is a human/reference annotation with values `V`, `T`, `A`,
`Balanced`, or `Unclear`. VT rows use only V/T/Balanced/Unclear; VA rows use only
V/A/Balanced/Unclear. M1 and M2 are condition identifiers and are never modality values.

`joint_lean_direction` is not an annotation field. It is derived only after freezing TME and
computing signed Joint Lean (`R`) from state embeddings.

## Immutable legacy v1 contract

The current state-dataset and state-bundle implementation is a frozen legacy consumer of
v1-shaped fields. It does not validate the two v1 schema IDs and still accepts a compatibility
label alias, so this is isolated technical debt rather than a strict schema binding. The
`mprisk_stage1_relation_schema_v1` and `mprisk_sample_type_schema_v1` files retain their original
field names and meanings byte-for-byte. The v2 schemas do not replace them, and no loader
automatically upgrades between versions. A future v2 pipeline must use a new strict consumer,
select both v2 schemas, and write a new versioned artifact identity.

<!-- naming-contract: legacy-v1-start -->
The legacy v1 fields are:

```text
joint_label
joint_specific_affect
joint_is_clear
joint_confidence
dominant_modality
```
<!-- naming-contract: legacy-v1-end -->

These names document the frozen v1 interface only and must not appear in new annotation artifacts.

## Stage 2 Ratings

Stage 2 ratings evaluate natural-language responses under state-guided policies. The main dimensions are empathy, relevance, safety, and instruction adherence.

## Traceability

Every human annotation sheet must include annotator ID, sample ID, protocol, condition, timestamp or batch ID, and adjudication status.

Each sample should receive at least two annotations. Disagreement cases enter adjudication. The exported `annotator_agreement` field records the final agreement used by the manifest.
