# Curation Module

`curation/` is the independent data screening and human review module for `mprisk`.

Its only goal is to build final `Conflict`, `Aligned`, and `Ambiguous` manifests for the main experiment pipeline. It is not part of the core state-analysis code. It only exchanges JSONL and CSV files with the main project.

## What This Module Delivers

This module must deliver:

- dataset audit reports for local raw datasets
- candidate sample files from initial screening
- LLM-assisted screening outputs for `M1`, `M2`, and `M12`
- human annotation records
- adjudicated label files
- final manifests under `data/processed/manifests/`

The final main-experiment manifests are:

- `data/processed/manifests/unified_sample_manifest.jsonl`
- `data/processed/manifests/conflict_manifest.jsonl`
- `data/processed/manifests/aligned_manifest.jsonl`
- `data/processed/manifests/ambiguous_manifest.jsonl`

The curation summary files are:

- `curation/outputs/reports/DATASET_AUDIT.md`
- `curation/outputs/reports/dataset_audit.json`
- `curation/outputs/reports/CURATION_SUMMARY.md`

## Module Boundary

`curation/` may read:

- `configs/datasets/*.yaml`
- `configs/protocols/*.yaml`
- raw dataset paths from `configs/local_paths.yaml`

`curation/` must not import:

- `src/mprisk/evaluation`
- `src/mprisk/state`
- `src/mprisk/representation`

The main experiment pipeline must only consume curated JSONL/CSV outputs. It must not depend on the curation web app, SQLite database, or LLM screening client.

## Supported Datasets

The local dataset root is:

```text
/home/team/zhanghaonan/TAFFC/datasets
```

The expected local dataset folders are:

- `ch_sims_v2`
- `ch_sims`
- `cmu_mosi`
- `cmu_mosei`
- generated data pools, when available

Protocol roles:

- `VT` is the primary protocol.
- `VA` is a secondary protocol.
- `IT` is a derived protocol, not a native protocol of the current public datasets.

Dataset policy:

- `CH-SIMS v2` is the primary source for native `VT` and `VA`.
- `CH-SIMS` must be audited locally before treating it as native `VT` or `VA`.
- `CMU-MOSI` and `CMU-MOSEI` are candidate pools only. They must go through LLM screening and human review.
- `IT` always goes through derived image/frame construction, LLM screening, and human review.

## Three-Stage Workflow

### Stage 0: Local Dataset Audit

Run:

```bash
python3 curation/scripts/audit_local_datasets.py
```

Default input:

```text
/home/team/zhanghaonan/TAFFC/datasets
```

Default outputs:

```text
curation/outputs/reports/dataset_audit.json
curation/outputs/reports/DATASET_AUDIT.md
```

The audit report must list:

- local folder existence
- file counts and byte counts
- detected video, audio, text, and image files
- label-like files
- readable label columns
- native or derived protocol support
- suggested column mappings

Do not treat a dataset as native `VT` or `VA` just because it has video, audio, or text files. Native support requires usable labels for the isolated modalities and the joint view.

### Stage 1: Initial Screening

Initial screening scripts build candidate rows from dataset labels or generation metadata.

Main scripts:

- `curation/scripts/initial_screen_ch_sims_v2.py`
- `curation/scripts/initial_screen_mosi.py`
- `curation/scripts/initial_screen_mosei.py`
- `curation/scripts/initial_screen_dfew.py`
- `curation/scripts/initial_screen_generated.py`

Outputs:

```text
curation/outputs/candidates/*.jsonl
```

Goal:

- reduce scale
- keep likely `Conflict`, `Aligned`, and `Ambiguous` candidates
- preserve source metadata and media paths
- mark MOSI, MOSEI, and generated pools as needing LLM screening and human review

Candidate rows must contain at least:

```json
{
  "sample_id": "...",
  "source_dataset": "...",
  "source_id": "...",
  "protocol": "VT",
  "m1_modality": "vision",
  "m2_modality": "text",
  "m1_label": "...",
  "m2_label": "...",
  "joint_label": "...",
  "candidate_type": "Conflict",
  "needs_llm_screening": true,
  "source_is_generated": false,
  "media_paths": {
    "vision": "...",
    "audio": "...",
    "text": "..."
  }
}
```

These labels are candidate labels. They are not final truth labels.

### Stage 2: LLM-Assisted Screening

Run:

```bash
python3 curation/scripts/run_llm_screening.py \
  --input curation/outputs/candidates/example.jsonl \
  --output curation/outputs/llm_screening/example.jsonl \
  --provider mock
```

Use OpenRouter Gemini after setting `.env` or shell variables:

```bash
export OPENROUTER_API_KEY=...
export OPENROUTER_GEMINI_MODEL=google/gemini-3.1-pro-preview
python3 curation/scripts/run_llm_screening.py \
  --input curation/outputs/candidates/example.jsonl \
  --output curation/outputs/llm_screening/example.jsonl \
  --provider openrouter-gemini
```

The screening script runs three isolated views:

- `M1`
- `M2`
- `M12`

Important rule:

- `M1` prompts only receive `M1` media and metadata.
- `M2` prompts only receive `M2` media and metadata.
- `M12` prompts receive the paired media.
- Prompts must not include `candidate_type`, planned labels, raw labels, or joint suggestions.
- LLM outputs are suggestions only. They are not final labels.

LLM screening rows must contain:

```json
{
  "sample_id": "...",
  "protocol": "VT",
  "view_outputs": {
    "M1": {
      "label": "positive",
      "specific_affect": "smile",
      "is_clear": true,
      "confidence": 0.87,
      "evidence": "short phrase",
      "quality_flags": []
    },
    "M2": {
      "label": "negative",
      "specific_affect": "complaint",
      "is_clear": true,
      "confidence": 0.81,
      "evidence": "short phrase",
      "quality_flags": []
    },
    "M12": {
      "label": "negative",
      "specific_affect": "sarcasm",
      "is_clear": true,
      "confidence": 0.84,
      "evidence": "short phrase",
      "quality_flags": []
    }
  },
  "sample_type_suggestion": "Conflict",
  "dominant_modality_suggestion": "M2",
  "quality_flags": [],
  "needs_human_review": true
}
```

## Human Review App

Backend:

```bash
conda run -n mprisk uvicorn curation.backend.app:app --host 0.0.0.0 --port 8765
```

Frontend:

```bash
cd curation/frontend
VITE_API_BASE=http://localhost:8765 npm run dev -- --host 0.0.0.0 --port 8766
```

The annotation UI supports:

- sample queue
- `M1`, `M2`, and `M12` view tabs
- real image, video, and audio display through the backend `/media` route
- inline text display
- annotator id from URL parameter, localStorage, or manual input
- annotation save
- adjudication preview

Annotators must record:

```json
{
  "sample_id": "...",
  "annotator_id": "...",
  "m1_label": "positive",
  "m2_label": "negative",
  "joint_label": "negative",
  "m1_is_clear": true,
  "m2_is_clear": true,
  "joint_is_clear": true,
  "sample_type": "Conflict",
  "dominant_modality": "M2",
  "notes": "..."
}
```

The standard human-annotation workflow requires at least two independent annotations before a
sample enters a main experiment.

## Frozen Machine-Screened Delivery Exception

`delivery_20260714` is an explicit exception for the current experiment run. Its existing
machine-screened labels and generated-data design labels are accepted as the current inclusion
policy. All rows truthfully retain `annotation_count=0` and `annotator_agreement=0.0`, while
the delivered `use_in_main` values remain authoritative.

Do not pass these final manifests back through `export_final_manifests.py`. That exporter is
the strict path for future human-adjudicated batches and intentionally keeps the two-annotator
rule. Re-exporting this delivery would both change its inclusion labels and lose delivery-only
provenance fields.

The machine-verifiable exception, archive hash, label sources, counts, subtitle crop, variety-text
handling, real/generated source boundary, and pending future annotation statistics are recorded
in:

- `data/processed/manifests/delivery_20260714.provenance.json`

## Adjudication And Final Export

Adjudication outputs:

```text
curation/outputs/adjudicated/adjudicated_labels.jsonl
```

Final export:

```bash
python3 curation/scripts/export_final_manifests.py \
  --input curation/outputs/adjudicated/adjudicated_labels.jsonl \
  --output-dir data/processed/manifests
```

Final manifest rows must include:

- source dataset and source id
- protocol
- media paths
- `M1`, `M2`, and `M12` views
- final sample type
- dominant modality
- annotator agreement
- annotation count
- source_is_generated
- use_in_main

Main experiments only read rows with:

```text
sample_type in {Conflict, Aligned}
use_in_main = true
```

`use_in_main = true` requires:

- `sample_type in {Conflict, Aligned}`
- `annotation_count >= 2`
- `annotator_agreement >= 0.67`
- `m1_is_clear = true`
- `m2_is_clear = true`
- `joint_is_clear = true`
- no blocking quality flag
- `Conflict`: `m1_label != m2_label`
- `Aligned`: `m1_label = m2_label = joint_label`

Blocking quality flags include:

- `missing_vision`
- `missing_audio`
- `missing_text`
- `low_audio`
- `face_occluded`
- `corrupted_media`
- `generated_artifact_severe`
- `invalid_media`
- `modality_missing`

## Final Handoff Checklist

The curation module is ready for handoff only when:

1. `curation/scripts/audit_local_datasets.py` has been run on the local dataset root.
2. `DATASET_AUDIT.md` and `dataset_audit.json` exist under `curation/outputs/reports/`.
3. Candidate JSONL files exist under `curation/outputs/candidates/`.
4. LLM screening JSONL files exist under `curation/outputs/llm_screening/`.
5. The annotation UI displays media for `M1`, `M2`, and `M12`.
6. Human annotation rows include real `annotator_id` values.
7. Adjudicated labels exist under `curation/outputs/adjudicated/`.
8. Final manifests export without schema errors.
9. `Conflict` and `Aligned` manifests are non-empty.
10. A human has manually reviewed a small batch before main experiments start.

## Out Of Scope

This module does not:

- run the main paper experiments
- compute `S`, `D`, or `R`
- train trajectory encoders
- run large multimodal model hidden-state extraction
- decide final paper claims

It only builds curated sample manifests.
