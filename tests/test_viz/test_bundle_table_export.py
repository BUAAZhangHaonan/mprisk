from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mprisk.viz.bundle_tables import export_bundle_tables


def test_tables_export_pending_safe_latex_without_fake_metrics(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/paper/table_map.yaml").read_text())
    for spec in config["tables"].values():
        spec["input"] = str(tmp_path / Path(spec["input"]).name)
        spec["output"] = str(tmp_path / Path(spec["output"]).name)
    config_path = tmp_path / "tables.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = export_bundle_tables(config_path)

    assert len(result["tables"]) == 3
    assert {
        key: row["row_count"] for key, row in result["tables"].items()
    } == {
        "tab01_cross_backbone_results": 3,
        "tab02_conflict_misread_baselines": 7,
        "tab03_downstream_quality": 9,
    }
    for row in result["tables"].values():
        text = Path(row["output"]).read_text(encoding="utf-8")
        assert Path(row["input"]).is_file()
        assert Path(row["input"] + ".provenance.json").is_file()
        assert row["provenance"]["status"] == "Pending"
        assert len(row["provenance"]["input_sha256"]) == 64
        assert "\\begin{tabular}" in text
        assert "\\toprule" in text and "\\bottomrule" in text
        assert "Pending" in text
        assert "0.0" not in text and "0.5" not in text
    misread = Path(result["tables"]["tab02_conflict_misread_baselines"]["output"]).read_text()
    assert "AUPRC" in misread
    assert "Conflict/Aligned" not in misread
    assert "Description Disagreement" in misread
    assert "State-Indices Readout" in misread


def test_ready_table_requires_exact_schema_provenance_and_checksum(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/paper/table_map.yaml").read_text())
    for spec in config["tables"].values():
        spec["input"] = str(tmp_path / Path(spec["input"]).name)
        spec["output"] = str(tmp_path / Path(spec["output"]).name)
    config_path = tmp_path / "tables.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    key = "tab01_cross_backbone_results"
    spec = config["tables"][key]
    input_path = Path(spec["input"])
    columns = spec["columns"]
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "Model / Protocol": "Qwen3-VL-8B / VT",
                "Diagnostic Acc. (Aligned)": "0.81",
                "Diagnostic Acc. (Conflict)": "0.62",
                "Dominant Modality Signature": "V lean",
            }
        )
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    source_path = tmp_path / "table-source.json"
    source_path.write_text("{}", encoding="utf-8")
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    sidecar = input_path.with_suffix(input_path.suffix + ".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema": "mprisk_paper_table_input_v1",
                "table_key": key,
                "status": "Ready",
                "columns": columns,
                "input_sha256": digest,
                "row_count": 1,
                "generated_command": ["pytest", "fixture"],
                "sources": [{"path": str(source_path), "sha256": source_digest}],
            }
        ),
        encoding="utf-8",
    )

    result = export_bundle_tables(config_path)
    assert result["tables"][key]["status"] == "Ready"
    assert result["tables"][key]["input_sha256"] == digest

    input_path.write_text(input_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        export_bundle_tables(config_path)


def test_existing_table_csv_without_sidecar_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/paper/table_map.yaml").read_text())
    for spec in config["tables"].values():
        spec["input"] = str(tmp_path / Path(spec["input"]).name)
        spec["output"] = str(tmp_path / Path(spec["output"]).name)
    config_path = tmp_path / "tables.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    input_path = Path(config["tables"]["tab02_conflict_misread_baselines"]["input"])
    input_path.write_text("Method,Accuracy,Macro-F1,AUPRC,Latency\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance sidecar"):
        export_bundle_tables(config_path)
