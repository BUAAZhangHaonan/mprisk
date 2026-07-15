# Data Protocol

## Sample Types

- `Conflict`: modalities give clear but different affective signals.
- `Ambiguous`: affective evidence is weak, mixed, or hard to label.
- `Aligned`: modalities support the same affective judgment.

Main experiments focus on `Conflict` and `Aligned`. `Ambiguous` is reserved for appendix or supplemental analysis.

The final data layer uses relation labels instead of a shared affective coordinate. M1, M2, and
M12 are conditions, not modality names. Each curated sample must carry:

```text
m1_label
m2_label
m12_label
m1_is_clear
m2_is_clear
m12_is_clear
sample_type
joint_lean_direction
```

Allowed coarse labels are `positive`, `negative`, `neutral`, `uncertain`, and `invalid`.

## Protocols

- `VT`: vision and text.
- `VA`: vision and audio.
- `IT`: image and text.

For each protocol, the pipeline builds three conditions:

- `M1`: first unimodal condition.
- `M2`: second unimodal condition.
- `M12`: joint multimodal condition.

`joint_lean_direction` is `V`, `T_or_A`, `No-lean`, or explicit `unclear`; it never uses M1 or M2.

## Main Datasets

`CH-SIMS v2` is the main dataset because it provides unimodal and multimodal annotations. `CMU-MOSI`, `CMU-MOSEI`, `DFEW`, and generated samples are supplemental or stress-test data.

## Curation Flow

All curated samples pass through three steps:

1. Initial screening from source labels or generation metadata.
2. LLM-assisted screening over `M1`, `M2`, and `M12`.
3. Human review and adjudication.

The LLM step is only a screening assistant. Human adjudication is the final label source.

## Dataset Screening Roles

- `CH-SIMS v2`: main source for natural Conflict and Aligned candidates. Initial screening uses unimodal and multimodal labels.
- `CMU-MOSI`: candidate pool. It needs LLM screening and human review before receiving relation labels.
- `CMU-MOSEI`: larger candidate and generalization pool. It also needs LLM screening and human review.
- `DFEW`: visual-anchor source. It enters the main conflict pool only when paired with natural text or another modality.
- Generated `sample_type.Conflict` and `sample_type.Aligned` data: supplemental and
  stress-test source. The source class-code mapping is `A=Conflict` and `C=Aligned`; A/C is
  retained only in ingestion provenance, and planned labels are candidate labels, not final labels.

## Normalized Sample Fields

Each processed sample manifest row should provide:

```text
sample_id
dataset_key
source_id
media_paths
available_modalities
sample_type
split
labels
annotations
protocol_conditions
```

## Split Rule

Splits must be deterministic and grouped by source media when needed. A sample, its prompt variants, and its three modality conditions must stay in the same split.

Final manifests are exported by `curation/scripts/export_final_manifests.py` into `data/processed/manifests/`.
