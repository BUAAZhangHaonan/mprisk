from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import matplotlib.image as mpimg
import pytest
import yaml

from mprisk.viz.bundle_figures import export_bundle_figures
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
    "fig10_aligned_case",
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
            "figA01_calibration_audit": {
                "title": "Aligned calibration audit",
                "input": str(tmp_path / "inputs/calibration.json"),
                "output": str(tmp_path / "appendix/figA01_calibration_audit.pdf"),
            }
        },
    }
    path = tmp_path / "figures.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_figure_export_emits_openable_vector_pending_pdfs_without_fake_values(tmp_path) -> None:
    result = export_bundle_figures(_config(tmp_path))
    assert list(result["figures"]) == EXPECTED_KEYS
    assert all(row["status"] == "Pending" for row in result["figures"].values())
    for row in result["figures"].values():
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
        for sample_id, sample_type, s_value, d_value, r_value in (
            ("a1", "Aligned", 0.1, 0.2, 0.2),
            ("c1", "Conflict", 0.3, 0.8, 0.4),
        ):
            base = {
                "sample_id": sample_id,
                "model": "test",
                "sample_type": sample_type,
                "S": s_value,
                "D": d_value,
                "R": r_value,
            }
            for metric, value in (("S", s_value), ("D", d_value), ("abs_R", abs(r_value))):
                writer.writerow({**base, "metric": metric, "value": value})
    input_path.with_suffix(input_path.suffix + ".provenance.json").write_text(
        json.dumps(
            {
                "schema": "mprisk_figure_input_provenance_v1",
                "figure_key": "fig04_sdr_distributions",
                "status": "Ready",
                "generated_command": ["pytest", "fixture"],
                "sources": [{"path": "fixture", "sha256": "0" * 64}],
                "sample_masks": {
                    "S": "all_samples",
                    "D": "S<=kappa",
                    "abs_R": "S<=kappa and D>tau",
                },
                "thresholds": {"kappa": 0.5, "tau": 0.01},
                "source_sample_count": 2,
                "included_sample_count": 2,
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
    assert list(table_map["tables"]) == [
        "tab01_main_results",
        "tab02_baselines",
        "tab03_stage2",
    ]


def test_figure_export_rejects_forbidden_pdf_text(tmp_path) -> None:
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["figures"]["fig07_misread_bias"]["title"] = "Arbitration"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden text"):
        export_bundle_figures(config_path)
