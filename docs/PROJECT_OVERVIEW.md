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
