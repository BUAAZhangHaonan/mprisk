# Figure and Table Map

All generated figures consume declared CSV or JSON evidence from `outputs/`. Missing
inputs retain the final panel layout with explicit `Pending` cells and never receive
invented numbers. `pdfinfo` must open every exported PDF. The locked terms are `Conflict`,
`Aligned`, `Misread`, `Non-misread`, `V lean`, and `T/A lean`.

Ready CSV inputs require a sibling `.csv.provenance.json` using
`mprisk_figure_input_provenance_v1`. It records the generating argv, SHA-256 for every
source, thresholds, source/included counts, and the locked sample masks. Fig. 4 uses S
for all samples, D only where `S<=kappa`, and `abs(R)` only where `S<=kappa and D>tau`.
Fig. 5 uses every sample's hierarchical pattern. Fig. 6 includes only `S<=kappa`; its
R-direction emphasis is enabled only where `D>tau`. Export rejects rows that violate
these masks.

## Main Figures

| Figure | Role | Output |
|---|---|---|
| Fig. 1 | Affective Misread before the First Token | `paper/figures/generated/fig01_problem_protocol.pdf` |
| Fig. 2 | Overall Framework | `paper/figures/generated/fig02_representation_pipeline.pdf` |
| Fig. 3 | Spherical S/D/R and hierarchical states | `paper/figures/generated/fig03_spherical_sdr.pdf` |
| Fig. 4 | Conflict and Aligned S/D/R distributions | `paper/figures/generated/fig04_sdr_distributions.pdf` |
| Fig. 5 | Four-state stacked proportions | `paper/figures/generated/fig05_four_state_stacks.pdf` |
| Fig. 6 | Stable samples in D versus signed R | `paper/figures/generated/fig06_stable_d_signed_r.pdf` |
| Fig. 7 | Misread Pending upper panel and artifact-backed modality bias lower panel | `paper/figures/generated/fig07_misread_bias.pdf` |
| Fig. 8 | Held-out sample-level UMAP and Misread sensitivity panels | `paper/figures/generated/fig08_representation_comparison.pdf` |
| Fig. 9 | End-to-end Conflict case | `paper/figures/generated/fig09_conflict_case.pdf` |
| Fig. 10 | Four state-pattern cases | `paper/figures/generated/fig10_four_pattern_cases.pdf` |

The appendix map contains the required A1, A2, B1-B3, C1-C5, D1, D3, E1, and E2
layouts. D2 and E3 are explicitly excluded with reasons in the versioned map.

## Tables

The previous Table 1 mapping is removed. The remaining tables are renumbered I-III:

| Table | Key | Output |
|---|---|---|
| I | `tab01_cross_backbone_results` | `paper/tables/generated/tab01_cross_backbone_results.tex` |
| II | `tab02_conflict_misread_baselines` | `paper/tables/generated/tab02_conflict_misread_baselines.tex` |
| III | `tab03_downstream_quality` | `paper/tables/generated/tab03_downstream_quality.tex` |

## Commands

```bash
python scripts/build_figure_inputs.py --mode pending \
  --config configs/paper/figure_map.yaml \
  --run-records outputs/run_records/tme_node_v1.json
python scripts/snapshot_run_records.py \
  --output outputs/run_records/tme_node_v1.json \
  --cache-manifest unified_full_cache outputs/full_cache/manifests/unified_full_cache_manifest.json
python scripts/export_paper_figures.py \
  --config configs/paper/figure_map.yaml \
  --run-records outputs/run_records/tme_node_v1.json
python scripts/build_run_status.py \
  --config configs/paper/figure_map.yaml \
  --records outputs/run_records/tme_node_v1.json \
  --output RUN_STATUS.md
```

`RUN_STATUS.md` renders only supplied machine-readable records: actual argv/PID, GPU
snapshots, cache complete/failed/missing counts, experiment outcomes, PDF paths, and
Pending inputs. Missing runtime evidence remains explicitly unrecorded or Pending.

## Canonical Additive Exports

The canonical exporters never replace the registered root figures or either historical
`template_v2` / `template_v3_misread` export. They materialize checked snapshots and
figures in separate additive directories:

| Export | Checked inputs | Generated figures |
|---|---|---|
| State structure | `outputs/paper_exports/figures/state_structure/` | `paper/figures/generated/state_structure/` |
| Misread | `outputs/paper_exports/figures/misread/` | `paper/figures/generated/misread/` |
| Misread tables | `outputs/paper_exports/tables/misread/` | `paper/tables/generated/misread/` |

The committed Misread adapter root is self-contained under
`outputs/paper_exports/figures/misread/adapters/`. It binds verified labels, probe
metrics, and Conflict-supervision budget metrics by SHA-256. The exporter keeps the
probe-latency cell `Pending` because the registered probe queue did not record latency.
It does not substitute a numeric value.

```bash
PYTHONPATH=src python scripts/export_state_structure_figures.py
PYTHONPATH=src python scripts/export_misread_figures.py
```

To rebuild the canonical adapters from the original formal evidence, supply its three
immutable roots explicitly:

```bash
PYTHONPATH=src python scripts/build_misread_figure_adapters.py \
  --labels-root outputs/labels/delivery_20260716_single_flash_v1 \
  --queue-root outputs/downstream/delivery_20260716/seed20260717/misread_budget_probe_v1 \
  --conflict-root outputs/downstream/delivery_20260716/seed20260717/conflict_supervision_budget_v1
```
