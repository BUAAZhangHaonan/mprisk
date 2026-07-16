# TAFFC Figure 4–8 Matplotlib Templates

This package contains **five standalone Python plotting scripts** that reproduce the layout, color system, annotation structure, and data types of the five supplied TAFFC mock result figures.

> **Critical warning:** every value in `taffc_mock_data.npz` is synthetic. The generated figures are for layout planning only and must never appear in the manuscript, supplement, rebuttal, presentation of empirical findings, or any other scientific claim.

## Files

- `fig4_state_indices.py` — violin + box + jitter plots for `S`, `D`, and `|R|`, including eligible-sample composition, effect-size annotation, confidence interval, and significance legend.
- `fig5_state_patterns.py` — state-pattern composition by Aligned/Conflict and Misread composition within each Conflict state.
- `fig6_geometry.py` — stable-Conflict `D–R` geometry with `tau`, `±delta`, state regions, condition-state insets, trend line, and modality-direction colorbar.
- `fig7_misread_associations.py` — state-index/Misread-rate curves and signed model-specific `D–R` bias plots.
- `fig8_representation_quality.py` — frozen representation projections and Conflict-supervision sensitivity curves.
- `taffc_mock_data.npz` — the **single shared synthetic dataset** loaded by all five scripts.
- `taffc_mock_data_metadata.json` — seed, state counts, and synthetic-data warning.
- `outputs/` — PNGs produced by actually running all five scripts.
- `comparison_contact_sheet.png` — supplied raster mockups (left) versus the Matplotlib reproductions (right), for visual checking only.

## Requirements

```bash
python -m pip install -r requirements.txt
```

The scripts were executed with Python 3, NumPy, Matplotlib, and SciPy. No network access is required.

## Run

From this directory:

```bash
python fig4_state_indices.py
python fig5_state_patterns.py
python fig6_geometry.py
python fig7_misread_associations.py
python fig8_representation_quality.py
```

Each script writes a `1448 × 1086` PNG into `outputs/`.

## Replacing the synthetic data

For final paper figures, replace the arrays loaded from `taffc_mock_data.npz` with the actual experiment outputs while preserving:

1. the eligibility domains of `S`, `D`, and `R`;
2. Conflict-only conditioning for Misread analyses;
3. sample counts and denominators shown in each panel;
4. the distinction between condition-level spherical states and sample-level frozen representations;
5. the explicit statement of whether Accuracy or Balanced Accuracy is used;
6. the warning that `|R|` measures lean magnitude, while signed `R` identifies direction.

The supplied reference figures are generated raster mockups. The Python versions reproduce their publication-style visual grammar and panel structure, but are not intended as pixel-for-pixel copies of antialiased text and stochastic point placement.
