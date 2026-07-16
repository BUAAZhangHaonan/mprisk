# Repository Bootstrap Design

## Goal

Build the initial `mprisk` paper engineering repository under `/home/team/zhanghaonan/TAFFC`.

## Approved Shape

The repository is organized as a paper-code-data system. It contains LaTeX manuscript sources, appendix and response-letter scaffolds, data and annotation manifests, model asset configuration, hidden-state cache contracts, trajectory representation modules, S/D/R state analysis modules, baselines, evaluation scripts, visualization exports, and tests.

## Commit Plan

1. Initialize repository metadata, README, Python packaging, environment, and this design note.
2. Add configuration, documentation, data, and output manifest scaffolds.
3. Add the `src/mprisk` package, script entry points, and tests.
4. Add paper, appendix, legacy, figure, table, and response-letter scaffolds.

## Completion Criteria

- Git repository exists on branch `master`.
- Remote `origin` points to `git@github.com:BUAAZhangHaonan/mprisk.git`.
- The repository has multiple traceable commits.
- The package imports and Python files compile.
- The test suite passes.
- Each commit is pushed to the remote.
