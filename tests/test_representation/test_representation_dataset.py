from __future__ import annotations

import json
import subprocess
import sys

from mprisk.representation.dataset import build_representation_dataset


def _prompted_state(sample_id: str, view_key: str, prompt_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "condition": view_key,
        "prompt_id": prompt_id,
        "shard_path": f"outputs/prompted/{sample_id}-{view_key}-{prompt_id}.safetensors",
        "index_in_shard": 0,
    }


def _bundle(sample_id: str, sample_type: str = "Conflict") -> dict[str, object]:
    prompt_ids = ["vt_primary_v1_t01", "vt_primary_v1_t02"]
    return {
        "sample_id": sample_id,
        "sample_type": sample_type,
        "model_key": "qwen3_vl_8b",
        "protocol": "vt",
        "prompt_set_key": "vt_primary_v1",
        "view_labels": {
            "M1": {"label": "positive", "specific_affect": "joy", "is_clear": True},
            "M2": {"label": "negative", "specific_affect": "anger", "is_clear": True},
            "M12": {"label": "neutral", "specific_affect": "calm", "is_clear": True},
        },
        "prompts": [{"prompt_id": prompt_id} for prompt_id in prompt_ids],
        "views": {
            view_key: {
                "prompts": {
                    prompt_id: {
                        "prompt_id": prompt_id,
                        "prompt_conditioned_state": _prompted_state(
                            sample_id, view_key, prompt_id
                        ),
                    }
                    for prompt_id in prompt_ids
                }
            }
            for view_key in ("M1", "M2", "M12")
        },
        "metadata": {"source_dataset": "ch_sims_v2", "split_group_id": "group-a"},
    }


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_build_representation_dataset_expands_bundles_by_view_and_prompt(tmp_path) -> None:
    bundle_manifest = tmp_path / "bundle_manifest.jsonl"
    output_dir = tmp_path / "representation"
    _write_jsonl(bundle_manifest, [_bundle("sample-ok")])

    result = build_representation_dataset(
        bundle_manifest_path=bundle_manifest,
        output_dir=output_dir,
    )

    rows = _read_jsonl(result.dataset_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert result.exported_rows == 6
    assert result.skipped_rows == 0
    assert [row["row_id"] for row in rows] == [
        "sample-ok:M1:vt_primary_v1_t01",
        "sample-ok:M1:vt_primary_v1_t02",
        "sample-ok:M2:vt_primary_v1_t01",
        "sample-ok:M2:vt_primary_v1_t02",
        "sample-ok:M12:vt_primary_v1_t01",
        "sample-ok:M12:vt_primary_v1_t02",
    ]
    assert rows[0] == {
        "row_id": "sample-ok:M1:vt_primary_v1_t01",
        "sample_id": "sample-ok",
        "sample_type": "Conflict",
        "model_key": "qwen3_vl_8b",
        "protocol": "vt",
        "view_key": "M1",
        "prompt_id": "vt_primary_v1_t01",
        "prompt_set_key": "vt_primary_v1",
        "label": "positive",
        "specific_affect": "joy",
        "is_clear": True,
        "prompt_conditioned_state": _prompted_state(
            "sample-ok", "M1", "vt_primary_v1_t01"
        ),
        "split_group_id": "group-a",
        "source_dataset": "ch_sims_v2",
    }
    assert summary == {
        "total_input_bundles": 1,
        "exported_rows": 6,
        "skipped_rows": 0,
        "label_counts": {"negative": 2, "neutral": 2, "positive": 2},
        "sample_type_counts": {"Conflict": 6},
    }


def test_build_representation_dataset_filters_rows_and_falls_back_metadata(tmp_path) -> None:
    bundle_manifest = tmp_path / "bundle_manifest.jsonl"
    output_dir = tmp_path / "representation"
    mixed = _bundle("sample-mixed", sample_type="Aligned")
    mixed["view_labels"]["M1"]["is_clear"] = False
    mixed["view_labels"]["M2"]["label"] = "mixed"
    mixed["metadata"] = {}
    ambiguous = _bundle("sample-ambiguous", sample_type="Ambiguous")
    _write_jsonl(bundle_manifest, [mixed, ambiguous])

    result = build_representation_dataset(
        bundle_manifest_path=bundle_manifest,
        output_dir=output_dir,
    )

    rows = _read_jsonl(result.dataset_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert [row["row_id"] for row in rows] == [
        "sample-mixed:M12:vt_primary_v1_t01",
        "sample-mixed:M12:vt_primary_v1_t02",
    ]
    assert rows[0]["split_group_id"] == "sample-mixed"
    assert rows[0]["source_dataset"] == ""
    assert summary["total_input_bundles"] == 2
    assert summary["exported_rows"] == 2
    assert summary["skipped_rows"] == 10
    assert summary["label_counts"] == {"neutral": 2}
    assert summary["sample_type_counts"] == {"Aligned": 2}


def test_build_representation_dataset_cli_writes_outputs(tmp_path) -> None:
    bundle_manifest = tmp_path / "bundle_manifest.jsonl"
    output_dir = tmp_path / "representation"
    _write_jsonl(bundle_manifest, [_bundle("sample-cli")])

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_representation_dataset.py",
            "--bundle-manifest",
            str(bundle_manifest),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
        check=True,
        capture_output=True,
        text=True,
    )

    rows = _read_jsonl(output_dir / "representation_dataset.jsonl")
    summary = json.loads(
        (output_dir / "representation_dataset_summary.json").read_text(encoding="utf-8")
    )

    assert "representation_dataset=" in completed.stdout
    assert len(rows) == 6
    assert summary["exported_rows"] == 6
