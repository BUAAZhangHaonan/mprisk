# Annotation Guide

This document records the target annotation surfaces.

## Screening

Screening decides whether a sample is `Conflict`, `Ambiguous`, or `Aligned`.

The annotation unit has three views:

- `M1`: first modality only.
- `M2`: second modality only.
- `M12`: joint view.

Annotators label each view with `positive`, `negative`, `neutral`, `uncertain`, or `invalid`.

## Stage 1 Labels

Stage 1 labels describe the relation between `M1`, `M2`, and `M12`.

Required fields:

```text
m1_label
m2_label
joint_label
m1_specific_affect
m2_specific_affect
joint_specific_affect
m1_is_clear
m2_is_clear
joint_is_clear
m1_confidence
m2_confidence
joint_confidence
sample_type
dominant_modality
quality_flags
notes
```

## Stage 2 Ratings

Stage 2 ratings evaluate natural-language responses under state-guided policies. The main dimensions are empathy, relevance, safety, and instruction adherence.

## Traceability

Every human annotation sheet must include annotator ID, sample ID, protocol, condition, timestamp or batch ID, and adjudication status.

Each sample should receive at least two annotations. Disagreement cases enter adjudication. The exported `annotator_agreement` field records the final agreement used by the manifest.
