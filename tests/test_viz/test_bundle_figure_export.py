from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import pytest
import yaml

from mprisk.viz.bundle_figures import UMAP_CONFIG, export_bundle_figures
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
                "input": str(tmp_path / "inputs" / f"{key}.csv"),
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
    result = export_bundle_figures(_config(tmp_path))
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
        assert black_fraction < 0.01


def test_pending_outputs_keep_final_panel_layouts(tmp_path) -> None:
    result = export_bundle_figures(_config(tmp_path))
    expected_text = {
        "fig01_problem_protocol": ("Pre-generation state at", "Diagnostic affect description"),
        "fig04_sdr_distributions": ("Qwen2.5-Omni-7B", "Qwen3-VL-8B", "InternVL3.5-8B"),
        "fig07_misread_bias": ("Pending Misread annotations", "V lean", "T/A lean"),
        "fig08_representation_comparison": ("Single-Point", "Trajectory MLP", "TME", "UMAP"),
        "fig09_conflict_case": (
            "Conflict input + GT",
            "Baseline response",
            "State-guided response",
        ),
        "fig10_four_pattern_cases": ("Confusion", "Consensus", "Balanced", "Dominant"),
        "figC5_model_patterns": ("16 models", "Pending"),
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


def test_fig4_uses_real_csv_and_run_status_reports_ready_vs_pending(tmp_path) -> None:
    config = _config(tmp_path)
    input_path = tmp_path / "inputs/fig04_sdr_distributions.csv"
    input_path.parent.mkdir(parents=True)
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
                    "S": "all_samples",
                    "D": "S<=kappa",
                    "abs_R": "S<=kappa and D>tau",
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
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["figures"]["fig07_misread_bias"]["title"] = "Arbitration"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden text"):
        export_bundle_figures(config_path)
