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
- the curation backend and its schema checks.

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
```

Those environments produce cache shards, sidecars, manifests, and ledgers. The `mprisk` environment consumes those artifacts and does not need to satisfy every model runtime dependency.

The frozen model panel resolves checkpoints below `/home/team/lvshuyang/Models`. The
Qwen2.5-Omni-7B checkpoint is at
`/home/team/lvshuyang/Models/Qwen/Qwen2.5-Omni-7B`; a dedicated extraction environment is not
declared until its wrapper requirements are implemented and validated.

## OpenRouter Gemini Screening

LLM screening uses an OpenAI-compatible OpenRouter endpoint when `MPRISK_SCREENING_PROVIDER=openrouter`.

Required local variables:

```text
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_GEMINI_MODEL=google/gemini-3.1-pro-preview
```

Leave the API key blank in committed files. Fill it only in a local `.env`.

## Boundary

Do not add model deployment dependencies to the base `mprisk` environment unless the dependency is also needed for cache reading or paper analysis. Keep model-specific dependencies in the model environments or in the optional `model` extra from `pyproject.toml`.
