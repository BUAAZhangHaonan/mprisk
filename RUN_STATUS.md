# RUN STATUS

Live snapshot: `2026-07-26`. Refactor phase 4 complete; phase 5 (push to `origin/v2`) pending.

## Refactor status

The repository has been restructured through four phases. The viz exploration package has been inlined into the mainline `mprisk` package and all four monkey-patches have been replaced by first-class API.

| Phase | Scope | Status |
|---|---|---|
| Stage 0 | Baseline audit + dependency graph | Complete |
| Stage 1 (P1.1-P1.3) | Replace `src/mprisk/` with v2 master copy; promote 14 viz files to top-level `mprisk` modules and subpackages | Complete |
| Stage 2 (P2.1-P2.3) | Inline four monkey-patches as native API | Complete |
| Stage 3 (P3.1-P3.5) | Strip `_v2` suffixes from identifiers, schema strings, paths, and filenames; delete dead `spherical_norm.py` module | Complete |
| Stage 4 (P4-A-P4-G) | Documentation update + pyproject metadata + known-failure marking | Complete (this commit series) |
| Stage 5 | Fine-grained commit + push to `origin/v2` | Pending |

The four inlined monkey-patches, each replaced by a first-class API:

- `BOOTSTRAP_REPLICATES`: `compute_spherical_state` now accepts a `bootstrap_replicates` argument (mainline default `2000`, viz pipeline passes `200`).
- LSTM-TME bi-LSTM: `SphericalTME_BiLSTM` is a registered branch under `encoder_type="bilstm"` in `relation_models.py`, selectable via `TrainingConfig`.
- SDR-aware hinge loss: `SphericalSDRHingeLoss` is in `losses.py`, enabled via `TrainingConfig.sdr_aux_weight > 0` with four `sdr_*` fields.
- Shape check: `_require_consistent_entry_shape` takes a `strict` argument; viz pipeline passes `strict_shape=False`.

Removed dead entry points: `install_v2_pipeline_patches`, `install_v2_tme_factory`, `install_normalization`. Removed dead files: `lstm_tme.py`, `sdr_loss.py`, `spherical_norm.py`.

Branch is `v2`, 60+ commits ahead of `origin/v2`. The physical directory name `~/TAFFC/mprisk-v2/`, the git branch name `v2`, model-architecture version tokens (`t_lstm_v2`, `TLSTMEncoderV2`, etc.), and the CH-SIMS v2 dataset name are intentionally preserved.

## Test baseline

`pytest --collect-only` collects 328 tests. Full run: 288 passed, 40 failed.

| Bucket | Count | Notes |
|---|---:|---|
| Frozen-data-bound | 32 | Tests assert byte-level hashes of generated artifacts (annotation JSONL, archetype meaning sets, generated archive, delivery bundle, model panel image hashes, etc.). Frozen artifacts were produced under the old mainline code path; regenerating them under the inlined viz code path is out of scope for this refactor. These failures are the known baseline and are not fixed. |
| API drift / contract change | 7 | `test_proxy_anchor_training_pipeline` (7 cases) asserts the old TME training contract. The mainline `TrainingConfig` now rejects state-supervised TME configs that lack positive `D` and angular weights; the viz pipeline path bypasses the new validator. Fixing requires rewriting the tests against the new contract, deferred to phase 5. |
| Other / needs triage | 1 | Two tests in `test_t0_extraction` plus residual are a mix of API drift and frozen-data assumptions. |

Known baseline failures are not blocking the refactor. They will be addressed in a follow-up either by regenerating the frozen artifacts or by rewriting the affected tests against the new API. Neither path requires code changes under `src/`, `scripts/`, `configs/`, or `tests/`.

## Pending work

- Push `v2` branch to `origin/v2` (stage 5).
- After push: regenerate frozen artifacts under the inlined code path, or rewrite the 40 failing tests against the new API.
- Optional follow-up: reconsider whether `outputs/canonical_rerun/` and `outputs/state_analysis/` should be collapsed further; current state matches the post-stage-3 layout.
