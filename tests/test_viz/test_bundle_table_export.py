from __future__ import annotations

from pathlib import Path

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
    for row in result["tables"].values():
        text = Path(row["output"]).read_text(encoding="utf-8")
        assert "\\begin{tabular}" in text
        assert "\\toprule" in text and "\\bottomrule" in text
        assert "Pending" in text
        assert "0.0" not in text and "0.5" not in text
    misread = Path(result["tables"]["tab02_conflict_misread_baselines"]["output"]).read_text()
    assert "AUPRC" in misread
    assert "Conflict/Aligned" not in misread
