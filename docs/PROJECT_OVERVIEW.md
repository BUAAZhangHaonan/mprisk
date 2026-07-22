# Project Overview

`mprisk` measures the internal state structures that multimodal models form before generating
their first token.

The paper focus is narrow: it asks how multimodal affective conflict appears in pre-generation
state structure, and how those structures are empirically associated with subsequent affective
Misreads. The repository traces that evidence chain from sample manifests to prefill caches,
trajectory representations, `S/D/R` indices, State Patterns, frozen-representation probes, and
paper exports.

## Main Paper Claim

State indices are complementary coordinates of a pre-generation configuration. They are not a
single Misread probability, and State Patterns are not an error taxonomy. Conflict/Aligned labels
supervise representation learning; Misread/Non-misread labels are held out for subsequent
behavioral analysis on frozen representations.

The repository does not implement a clinical system or claim a universal theory of multimodal
models.

## Main Objects

- `sample`: one multimodal item with labels and metadata.
- `condition`: one of `M1`, `M2`, or `M12`.
- `t0`: the last conditioning token state before the first generated token.
- `trajectory`: full-layer prefill hidden states at `t0`.
- `state indices`: State Dispersion (`S`), Modality Split (`D`), and signed Joint Lean (`R`).
- `State Pattern`: `Confusion`, `Consensus`, `Balanced`, or `Dominant`.
- `relation embedding`: the sample-level representation learned from the three ordered conditions.
- `Misread Judgment`: an independent behavioral label evaluated after representation learning.
