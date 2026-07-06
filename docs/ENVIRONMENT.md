# Environment

The project uses a split environment design.

## Core Environment

`mprisk` is the paper and algorithm environment. It contains dependencies for:

- reading manifests and hidden-state cache metadata;
- loading small cache tensors with `safetensors`;
- trajectory representation experiments that do not require model deployment;
- `S/D/R` state computation;
- baselines that operate on cached scores or embeddings;
- statistics, evaluation, plots, tables, and paper exports;
- tests and linting.

The environment sets `PYTHONNOUSERSITE=1`. This prevents the core environment from accidentally importing user-site large-model packages such as Torch or Transformers.

Create or update it with:

```bash
conda env create -f environment.yml
conda env update -n mprisk -f environment.yml --prune
```

For an existing environment, make the isolation variable explicit:

```bash
conda env config vars set -n mprisk PYTHONNOUSERSITE=1
```

Run checks with:

```bash
conda run -n mprisk make verify
```

## Model Environments

Large-model loading and hidden-state extraction reuse existing environments:

```text
mind-py311
mind-gemma4-py311
mind-molmo-py311
```

Those environments produce cache shards, sidecars, manifests, and ledgers. The `mprisk` environment consumes those artifacts and does not need to satisfy every model runtime dependency.

## Boundary

Do not add model deployment dependencies to the base `mprisk` environment unless the dependency is also needed for cache reading or paper analysis. Keep model-specific dependencies in the model environments or in the optional `model` extra from `pyproject.toml`.
