# Project Overview

`mprisk` studies multimodal pre-generation misjudgment risk.

The paper focus is narrow: in affective multimodal conflict settings, the model may show a risk signal before it generates the first token. The repository is built to trace that signal from sample manifest to prefill cache, trajectory representation, `S/D/R` scores, state labels, baselines, figures, tables, appendix, and response letter.

## Main Paper Claim

The repository supports a paper about pre-generation risk analysis, not a clinical system and not a universal theory of multimodal models.

## Main Objects

- `sample`: one multimodal item with labels and metadata.
- `condition`: one of `M1`, `M2`, or `M12`.
- `t0`: the last conditioning token state before the first generated token.
- `trajectory`: full-layer prefill hidden states at `t0`.
- `state scores`: `S`, `D`, and `R`.
- `state pattern`: `Confusion`, `Consensus`, `Balanced`, or `Dominant`.

## Pipeline Variants

The repository ships two representation pipelines sharing the same data and state-measure contracts:

- **Mainline pipeline**: GRU-based TME encoder with Proxy-Anchor loss only. Used for all paper-reported results. Strict entry-shape check, `BOOTSTRAP_REPLICATES=2000`.
- **Viz exploration pipeline**: bi-LSTM TME encoder (`encoder_type="bilstm"`) plus SDR-aware hinge auxiliary loss (`sdr_aux_weight > 0`). Used for ablation and visualization studies. Relaxed entry-shape check, `bootstrap_replicates=200`. Entry point: `mprisk.pipeline.run_for_model()`.

The two variants are the same code path with two config knobs flipped. They do not fork any data, cache, or state-measure module.
