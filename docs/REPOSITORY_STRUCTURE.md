# Repository Structure and Artifact Boundaries

This document defines which repository surfaces are active code, immutable provenance, or generated
artifacts. The distinction prevents cleanup work from invalidating resumable experiments.

## Active runtime surfaces

- `src/mprisk/`: importable task, cache, model, representation, state, evaluation, and export code.
- `scripts/`: thin command-line entry points. Implemented commands delegate to `src/mprisk/`.
- `configs/`: versioned active configuration, except the exact `configs/legacy/` subtree.
- `tests/`: unit, contract, smoke, and provenance tests.

An active command must either perform its documented task or fail explicitly. A scaffold must not
return a successful status for an unimplemented experiment.

## Immutable compatibility surfaces

The following files are retained because existing manifests, ledgers, caches, or published exports
refer to their exact names:

- `configs/legacy/`;
- frozen v1 schemas named in `docs/NAMING_CONVENTIONS.md`;
- existing cache strategy, prompt-set, model, condition, and figure-map identity keys;
- committed provenance records and control manifests.

These surfaces are not templates for new code. New semantic contracts receive new versioned names;
existing artifacts are never silently rewritten in place.

## Generated and large artifacts

- `outputs/`: caches, ledgers, summaries, evaluation records, and export inputs.
- `paper/figures/generated/`: rendered figures derived from committed or checksummed inputs.
- large model weights and raw datasets: external to Git and resolved through asset/config manifests.

Cleanup must preserve generated artifacts unless a task explicitly requests regeneration. Cache
directories are append/resume targets controlled by their manifests and SQLite ledgers, not scratch
directories.

## Model wrapper boundary

A production model family is available only when all of the following hold:

1. its wrapper implements the shared prefill-extraction contract;
2. the family is registered explicitly in `mprisk.models.wrapper_registry`;
3. the model asset declares its family, protocol, path, and runtime environment;
4. contract tests cover input construction and `t0` trajectory extraction;
5. a real-model smoke run completes one Conflict and one Aligned sample for all configured prompts
   and all three conditions.

A source file that only declares a family name is not a usable wrapper and must not be registered.
Unsupported families fail explicitly instead of selecting a nearby implementation.

## Dataset boundary

Dataset rows are consumed through committed manifests. A loader must preserve sample identity,
protocol, label, media provenance, and group split. Placeholder loaders are not active dataset
implementations and must not be used by experiment configs.

## Cleanup rules

- Do not rename files referenced by active resumable ledgers or completed artifact manifests.
- Do not edit raw datasets or completed cache shards during source cleanup.
- Keep vendor, model, dataset, protocol, and run identity in configs and provenance where possible;
  generic task modules use task-level names.
- Remove generated interpreter caches from review scope; `.gitignore` excludes them.
- Commit source cleanup separately from wrapper ports, experiment launches, and generated results.
