# Pipeline

The pipeline is designed around traceable paper artifacts.

## 1. Data Preparation

Inputs are raw datasets and annotation files. Outputs are normalized sample manifests, sample-type labels, protocol views, and deterministic splits.

Main artifacts:

- `data/processed/manifests/unified_sample_manifest.jsonl`
- `data/processed/manifests/conflict_manifest.jsonl`
- `data/processed/manifests/aligned_manifest.jsonl`
- `data/processed/manifests/protocol_manifests/*.jsonl`

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

## 4. Representation

The main representation uses full-layer prefill trajectories and maps them into a manifold-aware embedding space.

Main artifacts:

- `outputs/representation/checkpoints/`
- `outputs/representation/embeddings/`
- `outputs/representation/diagnostics/`

## 5. State Analysis

`S`, `D`, and `R` are computed from the three conditions. They are then converted into four state patterns.

Main artifacts:

- `outputs/states/scores/`
- `outputs/states/assignments/`
- `outputs/states/summaries/`

## 6. Baselines and Evaluation

Baselines include simple behavior signals, uncertainty methods, classifier risk, and post-hoc full-response analysis.

Main artifacts:

- `outputs/baselines/`
- `outputs/evaluation/`

## 7. Paper Export

Figures and tables are generated from output artifacts.

Main artifacts:

- `paper/figures/generated/`
- `paper/tables/generated/`
- `outputs/paper_exports/`
