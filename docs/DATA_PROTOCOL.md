# Data Protocol

## Sample Types

- `Conflict`: modalities give clear but different affective signals.
- `Ambiguous`: affective evidence is weak, mixed, or hard to label.
- `Aligned`: modalities support the same affective judgment.

Main experiments focus on `Conflict` and `Aligned`. `Ambiguous` is reserved for appendix or supplemental analysis.

## Protocols

- `VT`: vision and text.
- `VA`: vision and audio.
- `IT`: image and text.

For each protocol, the pipeline builds three views:

- `M1`: first unimodal condition.
- `M2`: second unimodal condition.
- `M12`: joint multimodal condition.

## Main Datasets

`CH-SIMS v2` is the main dataset because it provides unimodal and multimodal annotations. `CMU-MOSI`, `CMU-MOSEI`, `DFEW`, and generated samples are supplemental or stress-test data.

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
protocol_views
```

## Split Rule

Splits must be deterministic and grouped by source media when needed. A sample, its prompt variants, and its three modality conditions must stay in the same split.
