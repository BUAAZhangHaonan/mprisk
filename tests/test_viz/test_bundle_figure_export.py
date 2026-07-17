from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import pytest
import yaml

from mprisk.viz.bundle_figures import (
    UMAP_CONFIG,
    _render_misread_bias,
    _render_representation_comparison,
    export_bundle_figures,
)
from mprisk.viz.figure_inputs import write_pending_figure_inputs
from mprisk.viz.run_status import build_run_status

EXPECTED_KEYS = [
    "fig01_problem_protocol",
    "fig02_representation_pipeline",
    "fig03_spherical_sdr",
    "fig04_sdr_distributions",
    "fig05_four_state_stacks",
    "fig06_stable_d_signed_r",
    "fig07_misread_bias",
    "fig08_representation_comparison",
    "fig09_conflict_case",
    "fig10_four_pattern_cases",
]
JSON_MAIN_KEYS = {
    "fig01_problem_protocol",
    "fig02_representation_pipeline",
    "fig03_spherical_sdr",
    "fig09_conflict_case",
    "fig10_four_pattern_cases",
}

APPENDIX_KEYS = [
    "figA1_case_types",
    "figA2_misread_cases",
    "figB1_representation_details",
    "figB2_prompt_stability_latency",
    "figB3_delta_bootstrap_geometry",
    "figC1_ac_roc_pr",
    "figC2_conflict_retention",
    "figC3_seed_robustness",
    "figC4_threshold_sensitivity",
    "figC5_model_patterns",
    "figD1_misread_pr",
    "figD3_latency_breakdown",
    "figE1_human_quality",
    "figE2_pattern_cases",
]


def _config(tmp_path: Path) -> Path:
    config = {
        "schema": "mprisk_bundle_figure_map_v1",
        "figures": {
            key: {
                "title": key,
                "input": str(
                    tmp_path / "inputs" / f"{key}{'.json' if key in JSON_MAIN_KEYS else '.csv'}"
                ),
                "output": str(tmp_path / "generated" / f"{key}.pdf"),
            }
            for key in EXPECTED_KEYS
        },
        "appendix": {
            key: {
                "title": key,
                "input": str(tmp_path / "inputs" / f"{key}.json"),
                "output": str(tmp_path / "appendix" / f"{key}.pdf"),
            }
            for key in APPENDIX_KEYS
        },
        "optional_excluded": {
            "figD2_j_lens": {"reason": "No registered J-Lens artifact."},
            "figE3_self_correction": {"reason": "Outside the preregistered analysis."},
        },
    }
    path = tmp_path / "figures.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_figure_export_emits_openable_vector_pending_pdfs_without_fake_values(tmp_path) -> None:
    config = _config(tmp_path)
    write_pending_figure_inputs(config, generated_command=["pytest", "pending"])
    result = export_bundle_figures(config)
    assert list(result["figures"]) == EXPECTED_KEYS
    assert list(result["appendix"]) == APPENDIX_KEYS
    assert len(result["figures"]) + len(result["appendix"]) == 24
    assert all(
        result["figures"][key]["status"] == "Ready"
        for key in (
            "fig01_problem_protocol",
            "fig02_representation_pipeline",
            "fig03_spherical_sdr",
        )
    )
    assert result["appendix"]["figB1_representation_details"]["status"] == "Ready"
    for row in [*result["figures"].values(), *result["appendix"].values()]:
        pdf = Path(row["output"])
        assert pdf.read_bytes().startswith(b"%PDF-")
        completed = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
        assert completed.returncode == 0
        assert "Pending" not in completed.stderr
        raster_stem = tmp_path / "raster" / pdf.stem
        raster_stem.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["pdftoppm", "-f", "1", "-singlefile", "-png", str(pdf), str(raster_stem)],
            check=True,
            capture_output=True,
        )
        pixels = mpimg.imread(raster_stem.with_suffix(".png"))[..., :3]
        black_fraction = float((pixels.mean(axis=-1) < 0.05).mean())
        assert black_fraction < 0.05, f"unexpectedly dark first page: {pdf}"


def test_data_independent_inputs_are_materialized_as_ready_conceptual_artifacts(
    tmp_path: Path,
) -> None:
    config_path = _config(tmp_path)
    write_pending_figure_inputs(config_path, generated_command=["pytest", "pending"])
    config = yaml.safe_load(config_path.read_text())
    for key in (
        "fig01_problem_protocol",
        "fig02_representation_pipeline",
        "fig03_spherical_sdr",
    ):
        payload = json.loads(Path(config["figures"][key]["input"]).read_text())
        assert payload["schema"] == "mprisk_conceptual_figure_input_v1"
        assert payload["status"] == "Ready"
        assert payload["sample_masks"] == {"data_dependency": "none"}
    payload = json.loads(
        Path(config["appendix"]["figB1_representation_details"]["input"]).read_text()
    )
    assert payload["schema"] == "mprisk_conceptual_figure_input_v1"
    assert payload["status"] == "Ready"
    pending = json.loads(Path(config["figures"]["fig09_conflict_case"]["input"]).read_text())
    assert pending["status"] == "Pending"


def test_pending_outputs_keep_final_panel_layouts(tmp_path) -> None:
    config = _config(tmp_path)
    write_pending_figure_inputs(config, generated_command=["pytest", "pending"])
    result = export_bundle_figures(config)
    expected_text = {
        "fig01_problem_protocol": ("Pre-generation state at", "Diagnostic affect description"),
        "fig04_sdr_distributions": (
            "State Dispersion (S)",
            "Modality Split (D)",
            "Absolute Joint Lean (|R|)",
            "Sample class",
            "Aligned",
            "Conflict",
        ),
        "fig05_four_state_stacks": (
            "State Pattern proportion (%)",
            "Aligned",
            "Conflict",
            "Confusion",
            "Consensus",
            "Balanced",
            "Dominant",
        ),
        "fig06_stable_d_signed_r": (
            "Modality Split (D)",
            "signed Joint Lean (R)",
            "threshold position Pending",
            "V lean",
            "T/A lean",
        ),
        "fig07_misread_bias": (
            "Pending Misread annotations",
            "State-indicator quantile",
            "Misread rate (%)",
            "Modality Split (D)",
            "signed Joint Lean (R)",
        ),
        "fig08_representation_comparison": (
            "Single-Point",
            "Trajectory MLP",
            "TME",
            "UMAP-1",
            "UMAP-2",
            "Conflict samples retained (%)",
            "AUPRC",
        ),
        "fig09_conflict_case": (
            "Conflict input + GT",
            "Baseline response",
            "State-guided response",
        ),
        "fig10_four_pattern_cases": ("Confusion", "Consensus", "Balanced", "Dominant"),
        "figC5_model_patterns": ("16 models", "Pending"),
        "figB2_prompt_stability_latency": (
            "Equivalent prompts (P)",
            "Normalized value",
            "State stability",
            "Latency",
        ),
        "figB3_delta_bootstrap_geometry": (
            "Bootstrap resamples",
            "2000",
            "Modality Split (D)",
            "signed Joint Lean (R)",
        ),
        "figC1_ac_roc_pr": ("False-positive rate", "True-positive rate", "Recall", "Precision"),
        "figC2_conflict_retention": ("Conflict budget (%)", "A/C classification score"),
        "figC3_seed_robustness": ("Prompt-seed pair", "State Pattern agreement"),
        "figC4_threshold_sensitivity": (
            "Threshold multiplier",
            "State Pattern agreement (%)",
            "State Pattern proportion (%)",
        ),
        "figD1_misread_pr": ("Pending Misread annotations", "Recall", "Precision"),
        "figD3_latency_breakdown": ("Pipeline component", "Latency (s)"),
        "figE1_human_quality": ("Response method", "Mean human rating"),
    }
    for key, phrases in expected_text.items():
        group = result["figures"] if key in result["figures"] else result["appendix"]
        completed = subprocess.run(
            ["pdftotext", group[key]["output"], "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        for phrase in phrases:
            assert phrase in completed.stdout


def test_ready_fig7_renders_pending_misread_and_real_bias_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = ("qwen2_5_omni_7b", "qwen3_vl_8b", "internvl3_5_8b")
    rows = [
        {
            "panel": "bias",
            "model": model,
            "sample_id": f"{model}-c1",
            "sample_type": "Conflict",
            "S": "0.1",
            "D": "0.6",
            "R": str(0.35 - index * 0.3),
            "direction_emphasized": "true",
            "status": "Ready",
        }
        for index, model in enumerate(models)
    ]
    provenance = {
        "sample_masks": {
            "misread": "Pending Misread annotations",
            "bias": "representation_split=official_test and sample_type=Conflict and S<=kappa",
            "direction_emphasis": "D>tau",
        },
        "thresholds_by_model": {
            model: {"kappa": 0.5, "tau": 0.3} for model in models
        },
    }
    monkeypatch.setattr(
        "mprisk.viz.bundle_figures._validate_state_provenance",
        lambda _rows, _provenance: None,
    )
    output = tmp_path / "fig07.pdf"

    _render_misread_bias("Fig. 7", rows, provenance, output)

    assert output.read_bytes().startswith(b"%PDF-")
    extracted = subprocess.run(
        ["pdftotext", str(output), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert extracted.count("Pending Misread annotations") == 3
    assert "State Dispersion (S) vs Misread" in extracted
    assert "Modality Split (D) vs Misread" in extracted
    assert "State Pattern vs Misread" in extracted
    assert extracted.count("stable Conflict D-signed R") == 3


def test_fig4_uses_real_csv_and_run_status_reports_ready_vs_pending(tmp_path) -> None:
    config = _config(tmp_path)
    write_pending_figure_inputs(config, generated_command=["pytest", "pending"])
    input_path = tmp_path / "inputs/fig04_sdr_distributions.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "model", "sample_type", "S", "D", "R", "metric", "value"],
        )
        writer.writeheader()
        for model in ("qwen2_5_omni_7b", "qwen3_vl_8b", "internvl3_5_8b"):
            for sample_id, sample_type, s_value, d_value, r_value in (
                (f"{model}-a1", "Aligned", 0.1, 0.2, 0.2),
                (f"{model}-c1", "Conflict", 0.3, 0.8, 0.4),
            ):
                base = {
                    "sample_id": sample_id,
                    "model": model,
                    "sample_type": sample_type,
                    "S": s_value,
                    "D": d_value,
                    "R": r_value,
                }
                for metric, value in (
                    ("S", s_value),
                    ("D", d_value),
                    ("abs_R", abs(r_value)),
                ):
                    writer.writerow({**base, "metric": metric, "value": value})
    models = ("qwen2_5_omni_7b", "qwen3_vl_8b", "internvl3_5_8b")
    thresholds_by_model = {model: {"kappa": 0.5, "tau": 0.01} for model in models}
    split_identities = [
        {
            "model": model,
            "representation_split": "official_test",
            "split_assignment_sha256": "1" * 64,
        }
        for model in models
    ]
    calibration_identities = [
        {
            "model": model,
            "model_key": model,
            "protocol": "va" if model == "qwen2_5_omni_7b" else "vt",
            "prompt_set_key": "p8",
            "prompt_set_artifact_sha256": "2" * 64,
            "repr_key": "tme",
            "encoder_checkpoint_sha256": "3" * 64,
            "split_assignment_sha256": "1" * 64,
            "embedding_manifest_sha256": "4" * 64,
        }
        for model in models
    ]
    source_path = tmp_path / "fig04-source.json"
    source_path.write_text("{}", encoding="utf-8")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    input_path.with_suffix(input_path.suffix + ".provenance.json").write_text(
        json.dumps(
            {
                "schema": "mprisk_figure_input_provenance_v1",
                "figure_key": "fig04_sdr_distributions",
                "status": "Ready",
                "generated_command": ["pytest", "fixture"],
                "sources": [{"path": str(source_path), "sha256": source_sha256}],
                    "sample_masks": {
                        "S": "representation_split=official_test",
                        "D": "representation_split=official_test and S<=kappa",
                        "abs_R": (
                            "representation_split=official_test and S<=kappa and D>tau"
                        ),
                },
                "representation_split": "official_test",
                "source_representation_split_counts": {"official_test": 6},
                "official_test_sample_count": 6,
                "excluded_non_official_test_count": 0,
                "thresholds_by_model": thresholds_by_model,
                "split_identities": split_identities,
                "calibration_identities": calibration_identities,
                "source_sample_count": 6,
                "included_sample_count": 6,
            }
        ),
        encoding="utf-8",
    )

    result = export_bundle_figures(config)
    assert result["figures"]["fig04_sdr_distributions"]["status"] == "Ready"
    status_path = build_run_status(config, output_path=tmp_path / "RUN_STATUS.md")
    text = status_path.read_text(encoding="utf-8")
    assert "fig04_sdr_distributions | Ready" in text
    assert "fig07_misread_bias | Pending" in text
    assert "placeholder" not in text.casefold()


def test_fig8_rejects_duplicate_sample_rows_before_projection(tmp_path) -> None:
    rows = []
    for representation in ("Single-Point", "Trajectory MLP", "TME"):
        rows.extend(
            {
                "panel": "ac",
                "representation": representation,
                "model": "qwen3_vl_8b",
                "protocol": "VT",
                "seed": "20260717",
                "sample_id": "duplicate",
                "sample_type": "Aligned",
                "representation_split": "official_test",
                "feature": "[0.1, 0.2]",
                "status": "Ready",
            }
            for _ in range(2)
        )
    with pytest.raises(ValueError, match="duplicate sample rows"):
        _render_representation_comparison("Fig. 8", rows, {}, tmp_path / "fig08.pdf")


def test_versioned_map_has_final_ten_figures_and_three_tables() -> None:
    root = Path(__file__).resolve().parents[2]
    figure_map = yaml.safe_load((root / "configs/paper/figure_map.yaml").read_text())
    table_map = yaml.safe_load((root / "configs/paper/table_map.yaml").read_text())
    assert list(figure_map["figures"]) == EXPECTED_KEYS
    assert list(figure_map["appendix"]) == APPENDIX_KEYS
    assert set(figure_map["optional_excluded"]) == {"figD2_j_lens", "figE3_self_correction"}
    assert UMAP_CONFIG == {
        "random_state": 20260716,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "cosine",
    }
    assert "umap-learn==0.5.12" in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert list(table_map["tables"]) == [
        "tab01_cross_backbone_results",
        "tab02_conflict_misread_baselines",
        "tab03_downstream_quality",
    ]


def test_figure_export_rejects_forbidden_pdf_text(tmp_path) -> None:
    config_path = _config(tmp_path)
    write_pending_figure_inputs(config_path, generated_command=["pytest", "pending"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["figures"]["fig07_misread_bias"]["title"] = "Arbitration"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden text"):
        export_bundle_figures(config_path)


def test_figure_export_rejects_missing_conceptual_input(tmp_path) -> None:
    config_path = _config(tmp_path)
    with pytest.raises(ValueError, match="conceptual figure input is missing or empty"):
        export_bundle_figures(config_path)
