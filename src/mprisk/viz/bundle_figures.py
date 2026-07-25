"""Artifact-only vector PDF exports for the final ten-figure bundle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")

# Re-exports kept for backwards-compatible imports from external callers and tests.
from .figure_constants import (  # noqa: F401
    CONCEPTUAL_KEYS,
    FIGURE_SCHEMA,
    FORBIDDEN_PDF_TEXT,
    FULL_MODEL_LABELS,
    LOCKED_TERMS,
    MODEL_LABELS,
    MODEL_SPECS,
    STATUS_PENDING,
    STATUS_READY,
    UMAP_CONFIG,
)
from .figure_validators import (  # noqa: F401
    _as_bool,
    _is_sha256,
    _require_columns,
    _required_text,
    _sha256,
    _validate_pdf_open,
    _validate_pdf_text,
)
from .figure_pending_helpers import _add_pending_dr_framework, _pending_axis, _pending_card  # noqa: F401
from .figure_input_loaders import (  # noqa: F401
    _load_figure_input,
    _read_csv,
    _validate_fig04_masks,
    _validate_fig06_masks,
    _validate_provenance,
    _validate_state_provenance,
)
from .figure_conceptual import (  # noqa: F401
    _render_flow,
    _render_framework,
    _render_representation_details,
    _render_sdr_method,
)
from .figure_pending_layouts import (  # noqa: F401
    _render_appendix_layout,
    _render_cards,
    _render_model_facets,
    _render_two_by_three,
)
from .figure_artifact_renderers import (  # noqa: F401
    _render_artifact,
    _render_d_signed_r,
    _render_evidence_table,
    _render_four_state_stacks,
    _render_misread_bias,
    _render_representation_comparison,
    _render_sdr_distributions,
)


def export_bundle_figures(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    if config.get("schema") != FIGURE_SCHEMA:
        raise ValueError(f"figure config schema must be {FIGURE_SCHEMA}")
    figures = _export_group(config.get("figures"), expected_count=10)
    appendix = _export_group(config.get("appendix", {}), expected_count=14)
    excluded = config.get("optional_excluded")
    if not isinstance(excluded, Mapping) or set(excluded) != {
        "figD2_j_lens",
        "figE3_self_correction",
    }:
        raise ValueError("figure map must explicitly exclude optional D2 and E3")
    return {
        "schema": "mprisk_bundle_figure_export_v1",
        "config": str(config_file),
        "figures": figures,
        "appendix": appendix,
        "optional_excluded": dict(excluded),
    }


def _export_group(
    specs: object,
    *,
    expected_count: int | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(specs, Mapping):
        raise ValueError("figure group must be a mapping")
    if expected_count is not None and len(specs) != expected_count:
        raise ValueError(f"main figure map must contain exactly {expected_count} figures")
    exported: dict[str, dict[str, Any]] = {}
    for key, raw_spec in specs.items():
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"figure {key} specification must be a mapping")
        title = _required_text(raw_spec, "title")
        input_path = Path(_required_text(raw_spec, "input"))
        output_path = Path(_required_text(raw_spec, "output"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        status, rows, provenance = _load_figure_input(str(key), input_path)
        if str(key) in CONCEPTUAL_KEYS:
            if status != STATUS_READY:
                raise ValueError(f"conceptual figure input must be Ready: {input_path}")
            _render_locked_layout(key=str(key), title=title, output_path=output_path)
        elif status == STATUS_READY:
            _render_artifact(
                key=str(key),
                title=title,
                rows=rows,
                provenance=provenance,
                output_path=output_path,
            )
        else:
            _render_locked_layout(key=str(key), title=title, output_path=output_path)
        _validate_pdf_open(output_path)
        _validate_pdf_text(output_path)
        exported[str(key)] = {
            "status": status,
            "input": str(input_path),
            "output": str(output_path),
            "sha256": _sha256(output_path),
        }
    return exported


def _render_locked_layout(*, key: str, title: str, output_path: Path) -> None:
    if key == "fig01_problem_protocol":
        _render_flow(
            title,
            [
                "Complete multimodal input",
                r"Pre-generation state at $t_0$",
                "Diagnostic affect description",
                "Misread\nPending annotations",
            ],
            output_path,
        )
        return
    if key == "fig02_representation_pipeline":
        _render_framework(title, output_path)
        return
    if key == "fig03_spherical_sdr":
        _render_sdr_method(title, output_path)
        return
    if key == "figB1_representation_details":
        _render_representation_details(title, output_path)
        return
    if key in {"fig04_sdr_distributions", "fig05_four_state_stacks", "fig06_stable_d_signed_r"}:
        _render_model_facets(key, title, output_path)
        return
    if key in {"fig07_misread_bias", "fig08_representation_comparison"}:
        _render_two_by_three(key, title, output_path)
        return
    if key == "fig09_conflict_case":
        _render_cards(
            title,
            ("Conflict input + GT", "Baseline response", "State-guided response"),
            output_path,
        )
        return
    if key == "fig10_four_pattern_cases":
        _render_cards(title, ("Confusion", "Consensus", "Balanced", "Dominant"), output_path)
        return
    _render_appendix_layout(key, title, output_path)
