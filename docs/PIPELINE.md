# Pipeline

The pipeline is designed around traceable paper artifacts.

## 1. Data Preparation

Inputs are raw datasets and annotation files. Outputs are normalized sample manifests, sample-type labels, protocol views, and deterministic splits.

Data preparation now starts in the independent `curation/` module:

```text
source data -> initial screening -> LLM screening -> human review -> adjudication -> final manifests
```

Main artifacts:

- `data/processed/manifests/unified_sample_manifest.jsonl`
- `data/processed/manifests/conflict_manifest.jsonl`
- `data/processed/manifests/aligned_manifest.jsonl`
- `data/processed/manifests/protocol_manifests/*.jsonl`

Curation intermediate artifacts:

- `curation/outputs/candidates/*.jsonl`
- `curation/outputs/llm_screening/*.jsonl`
- `curation/outputs/human/*.jsonl`
- `curation/outputs/adjudicated/*.jsonl`
- `curation/outputs/exports/*.jsonl`

## 2. Prompt Banks

Prompt banks define equivalent task formulations for each protocol. The main protocol uses `K = 5` templates selected from a larger candidate pool.

Main artifacts:

- `data/processed/prompt_banks/vt_primary_bank_v1.jsonl`
- `data/processed/prompt_banks/va_aux_bank_v1.jsonl`
- `data/processed/prompt_banks/it_aux_bank_v1.jsonl`

## 3. Pre-generation Cache

For each model, dataset, protocol, split, and condition, the extraction pipeline stores full-layer hidden trajectories at `t0`.

Main artifacts:

- `outputs/full_cache/manifests/unified_full_cache_manifest.json`
- `outputs/full_cache/manifests/extraction_ledger.csv`

## 4. State-Data Surface

The first implementation phase connects final labels to cache entries. It does not train encoders and does not compute `S`, `D`, or `R`.

The chain is:

```text
final manifests -> cache surface -> t0 trajectory bundle -> state dataset manifest
```

Run the exporter:

```bash
python scripts/build_state_dataset.py \
  --manifest data/processed/manifests/conflict_manifest.jsonl \
  --manifest data/processed/manifests/aligned_manifest.jsonl \
  --model-key qwen3_vl_8b \
  --protocol VT
```

Run the smoke check:

```bash
python scripts/verify_state_data_pipeline.py \
  --manifest data/processed/manifests/conflict_manifest.jsonl \
  --manifest data/processed/manifests/aligned_manifest.jsonl \
  --model-key qwen3_vl_8b \
  --protocol VT
```

Main artifacts:

- `outputs/state_data/{model_key}/{protocol}/state_dataset_manifest.jsonl`
- `outputs/state_data/{model_key}/{protocol}/state_dataset_summary.json`
- `outputs/state_data/{model_key}/{protocol}/missing_cache_rows.jsonl`
- `outputs/state_data/reports/STATE_DATA_SMOKE.md`

`state_dataset_manifest.jsonl` stores cache indexes and trajectory metadata only. It does not copy hidden-state tensors.

## 5. Representation

The main representation uses full-layer prefill trajectories and maps them into a manifold-aware embedding space.

Main artifacts:

- `outputs/representation/checkpoints/`
- `outputs/representation/embeddings/`
- `outputs/representation/diagnostics/`

## 6. State Analysis

`S`, `D`, and `R` are computed from the three conditions. They are then converted into four state patterns.

Main artifacts:

- `outputs/states/scores/`
- `outputs/states/assignments/`
- `outputs/states/summaries/`

## 7. Baselines and Evaluation

Baselines include simple behavior signals, uncertainty methods, classifier risk, and post-hoc full-response analysis.

Main artifacts:

- `outputs/baselines/`
- `outputs/evaluation/`

## 8. Paper Export

Figures and tables are generated from output artifacts.

Main artifacts:

- `paper/figures/generated/`
- `paper/tables/generated/`
- `outputs/paper_exports/`
