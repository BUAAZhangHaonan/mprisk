# Curation

`curation/` is an independent data screening and human review module.

It builds Conflict, Ambiguous, and Aligned sample manifests through three stages:

1. Initial screening from dataset labels or generation metadata.
2. LLM-assisted screening over `M1`, `M2`, and `M12` views.
3. Human annotation and adjudication.

The module only exchanges JSONL and CSV files with the main `src/mprisk` experiment code.

## Boundaries

- It may read `configs/datasets/*.yaml` and `configs/protocols/*.yaml`.
- It must not import `src/mprisk/evaluation` or `src/mprisk/state`.
- It writes final manifests to `data/processed/manifests/`.
- It does not store API keys.
