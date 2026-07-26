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
- Includes a visualization exploration pipeline (`mprisk.pipeline.run_for_model()` entry point) with bi-LSTM TME and SDR-aware hinge loss for ablation studies.

## Repository Map

- `configs/`: model assets, datasets, protocols, prompts, experiment, and paper maps.
- `docs/`: project protocol, pipeline, model panel, figure map, and response-letter map.
- `data/`: data source notes, annotations, processed manifests, prompt banks, and mini smoke data.
- `outputs/`: generated caches, scores, baselines, evaluations, reports, and paper exports.
- `src/mprisk/`: Python package containing:
  - Top-level viz-exploration modules: `pipeline.py`, `setup_helper.py`, `plotting.py`, `misread.py`.
  - Mainline subpackages: `assets/`, `cache/`, `config/`, `data/`, `diagnostic_affect/`, `evaluation/`, `experiments/`, `ground_truth/`, `judge/`, `models/`, `policy/`, `prompts/`, `representation/`, `state/`, `utils/`, `viz/`.
  - Representation subpackage holds `relation_models.py` (TME encoders incl. `SphericalTME_BiLSTM`), `losses.py` (incl. `SphericalSDRHingeLoss`), `training.py` (incl. `TrainingConfig`), `baselines/`, `sdr_loss.py`.
  - State subpackage holds `s_measure.py`, `d_measure.py`, `r_measure.py`, `spherical.py`, `thresholds.py`, `patterns.py`, `aggregation.py`.
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

## Architecture Notes

The viz exploration pipeline lives alongside the mainline pipeline and shares the same data and state-measure contracts. Key configuration knobs:

- **bi-LSTM TME** is selected via `TrainingConfig(encoder_type="bilstm")` (registered as the `SphericalTME_BiLSTM` branch in `relation_models.py`). The default `encoder_type` is the mainline GRU-based TME.
- **SDR-aware hinge loss** is enabled via `TrainingConfig(sdr_aux_weight > 0)` together with `sdr_margin_d`, `sdr_warmup_epochs`, and `sdr_dominant_only`. The default `sdr_aux_weight=0` keeps the mainline Proxy-Anchor-only objective.
- **Bootstrap replicates** for spherical state variance: `BOOTSTRAP_REPLICATES` defaults to `2000` in the mainline S/D/R pipeline; the viz exploration pipeline passes `bootstrap_replicates=200` explicitly to `compute_spherical_state`.
- **Entry-shape check** is strict by default (`_require_consistent_entry_shape(..., strict=True)`). The viz exploration pipeline passes `strict_shape=False` so prompt-level trajectory slices of different lengths are accepted.

These knobs are the only sanctioned deviation points; everything else (cache manifests, split assignment, spherical normalization contracts, spherical S/D/R formulas, threshold calibration, state-pattern assignment) is shared between the two pipelines.
