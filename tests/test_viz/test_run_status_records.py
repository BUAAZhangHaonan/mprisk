from __future__ import annotations

import json

import yaml

from mprisk.viz.run_status import build_run_status


def test_run_status_aggregates_only_machine_readable_runtime_records(tmp_path) -> None:
    config = {
        "schema": "mprisk_bundle_figure_map_v1",
        "figures": {
            "fig01_problem_protocol": {
                "input": str(tmp_path / "missing.json"),
                "output": str(tmp_path / "fig01.pdf"),
            }
        },
        "appendix": {},
    }
    config_path = tmp_path / "figures.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    records = {
        "schema": "mprisk_run_records_v1",
        "commands": [
            {
                "command_id": "figures",
                "argv": ["python", "scripts/export_paper_figures.py"],
                "status": "success",
                "pid": 1234,
                "gpu": {"physical_index": 1, "peak_memory_mib": 2048},
            }
        ],
        "caches": [
            {
                "cache_key": "qwen3_vl_8b_vt",
                "status": "incomplete",
                "complete": 10,
                "failed": 2,
                "missing": 4,
            }
        ],
        "experiments": [
            {
                "experiment_key": "qwen3_tme",
                "status": "failure",
                "command_id": "train-qwen3",
                "reason": "recorded failure",
            }
        ],
        "visual_qa": [
            {
                "qa_key": "pending_bundle_v1",
                "status": "pass",
                "pdf_count": 11,
                "rendered_png_count": 11,
                "embedded_font_pdf_count": 11,
                "forbidden_match_count": 0,
                "notes": "manual image inspection complete",
            }
        ],
    }
    records_path = tmp_path / "run_records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")

    output = build_run_status(
        config_path,
        records_path=records_path,
        output_path=tmp_path / "RUN_STATUS.md",
    )
    text = output.read_text(encoding="utf-8")
    assert "scripts/export_paper_figures.py" in text
    assert "1234" in text
    assert "GPU 1" in text
    assert "2048" in text
    assert "10" in text and "2" in text and "4" in text
    assert "qwen3_tme" in text and "failure" in text
    assert "pending_bundle_v1" in text and "manual image inspection complete" in text
    assert "fig01_problem_protocol" in text and "Pending" in text
