# mprisk

`mprisk` is the paper engineering repository for multimodal pre-generation risk analysis.

The project studies misjudgment risk before generation in multimodal affective conflict settings. It keeps the paper, appendix, response letter, data manifests, hidden-state cache contracts, state measures, baselines, and figure exports in one traceable repository.

## Scope

- Analyze Conflict, Ambiguous, and Aligned multimodal samples.
- Compare `M1`, `M2`, and `M12` pre-generation states at `t0`.
- Represent state as a full-layer prefill trajectory, not a single hidden-state point.
- Compute `S`, `D`, and `R` state measures and assign four state patterns.
- Compare pre-generation analysis against behavior, uncertainty, classifier, and post-hoc baselines.
- Export paper-ready figures, tables, appendix material, and response-letter evidence.

## Repository Map

- `configs/`: model assets, datasets, protocols, prompts, experiment, and paper maps.
- `docs/`: project protocol, pipeline, model panel, figure map, and response-letter map.
- `data/`: data source notes, annotations, processed manifests, prompt banks, and mini smoke data.
- `outputs/`: generated caches, scores, baselines, evaluations, reports, and paper exports.
- `src/mprisk/`: Python package for data, models, cache, representation, state, baselines, evaluation, policy, and visualization.
- `scripts/`: command-line entry points for the paper pipeline.
- `tests/`: smoke tests and contract tests.
- `paper/`: LaTeX manuscript, appendix, figures, tables, legacy material, and response letter.

## Environment Split

Use the lightweight `mprisk` conda environment for core algorithms, cache reading, statistics, evaluation, and figure/table export.

Use existing model environments for large-model deployment and cache extraction:

- `mind-py311`: main model extraction environment.
- `mind-gemma4-py311`: separate Gemma 4 environment.
- `mind-molmo-py311`: separate Molmo environment.

The `mprisk` environment is intentionally not required to run every large model. It reads the cache and manifest outputs produced by the model environments.

## Large Files

Raw datasets, generated media, hidden-state shards, KV caches, checkpoints, and full experiment dumps are not committed by default. Their manifests, ledgers, checksums, summaries, and paper exports are committed when small enough to review.

## Core Principle

Every result must be traceable from paper figure or table back to a script, output summary, cache manifest, model asset, prompt bank, and sample manifest.
