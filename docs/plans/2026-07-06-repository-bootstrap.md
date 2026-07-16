# Repository Bootstrap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the initial `mprisk` paper engineering repository for multimodal pre-generation misjudgment risk analysis.

**Architecture:** Keep paper sources, data manifests, cache contracts, model wrappers, trajectory representation, state measures, baselines, evaluation, and visualization in one traceable repository. Large artifacts stay out of git; manifests, ledgers, summaries, and paper exports stay versioned.

**Tech Stack:** Python 3.11, setuptools, pytest, YAML configs, LaTeX paper sources, safetensors cache contracts.

---

### Task 1: Initialize Project Metadata

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `pyproject.toml`
- Create: `environment.yml`
- Create: `Makefile`
- Create: `.gitignore`
- Create: `.gitattributes`
- Create: `.env.example`
- Create: `docs/plans/2026-07-06-repository-bootstrap-design.md`

**Steps:**
1. Add repository metadata and large-file rules.
2. Initialize git on branch `master`.
3. Commit with `chore: initialize repository metadata`.
4. Add SSH remote and push.

### Task 2: Add Paper Engineering Scaffolds

**Files:**
- Create configs, docs, data, and outputs scaffolds.

**Steps:**
1. Add model, dataset, protocol, prompt, experiment, and paper config stubs.
2. Add documentation for pipeline, data protocol, MIND porting, figures, appendix, and response letter.
3. Add data and output README/manifest scaffolds.
4. Commit with `chore: add project scaffolds`.
5. Push.

### Task 3: Add Python Package and Tests

**Files:**
- Create: `src/mprisk/**`
- Create: `scripts/*.py`
- Create: `tests/**`

**Steps:**
1. Add importable package modules with stable contracts and placeholders.
2. Add script entry points that expose the intended pipeline commands.
3. Add smoke tests for package metadata, state assignment, config loading, and cache schema.
4. Run compile and tests.
5. Commit with `chore: add package skeleton`.
6. Push.

### Task 4: Add Paper Sources

**Files:**
- Create: `paper/**`

**Steps:**
1. Add main manuscript skeleton with five fixed sections.
2. Add appendix and response-letter skeletons.
3. Add figure and table placeholders.
4. Commit with `chore: add paper skeleton`.
5. Push.

### Task 5: Final Verification

**Steps:**
1. Run `python -m compileall -q src scripts`.
2. Run `pytest -q`.
3. Run `git status --short --branch`.
4. Run `git log --oneline --decorate -n 8`.
5. Confirm all commits are pushed to `origin/master`.
