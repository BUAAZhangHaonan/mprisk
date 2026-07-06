# Figure Map

## Main Figures

| Figure | Role | Source |
|---|---|---|
| Fig. 1 Overview | Story figure for multimodal pre-generation misjudgment risk. | `paper/figures/generated/fig01_overview.pdf` |
| Fig. 2 Method | Three conditions, trajectory extraction, S/D/R, and states. | `paper/figures/generated/fig02_method.pdf` |
| Fig. 3 Four States | Confusion, Consensus, Balanced, Dominant. | `paper/figures/generated/fig03_sdr.pdf` |
| Fig. 4 Conflict vs Aligned | Main state-difference result. | `paper/figures/generated/fig04_conflict_vs_aligned.pdf` |
| Fig. 5 State vs Error | Main state-to-error result. | `paper/figures/generated/fig05_state_vs_error.pdf` |
| Fig. 6 Timing and Efficiency | `t0` analysis vs post-hoc full-response analysis. | `paper/figures/generated/fig06_t0_vs_posthoc_efficiency.pdf` |
| Fig. 7 Cases | Conflict case, Aligned case, and state-guided response case. | `paper/figures/generated/fig07_cases_and_policy.pdf` |

## Appendix Figures

Appendix figures hold template count, layer-wise analysis, conflict-level readout checks, model-panel geometry, state distributions, extra cases, and human-evaluation details.

## Export Rule

Generated paper figures must come from `outputs/` through `scripts/export_paper_figures.py`.

The first data-quality figures should consume final curation manifests, not raw candidate files. Screening and agreement summaries belong in the appendix unless they are directly used to explain the main Conflict vs Aligned sample construction.
