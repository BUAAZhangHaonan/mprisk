from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import scripts.export_ready_paper_figures as export


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _delivery_run(root: Path, model_key: str, method: str = export.STATE_METHOD) -> Path:
    method_root = root / model_key / method
    config = _write(root / "configs" / f"{model_key}_{method}.yaml", f"key: {method}\n")
    union = _write(root / "unions" / f"{model_key}.json", "{}\n")
    artifacts = {
        "best_checkpoint": _write(method_root / "training/best_checkpoint.pt", "checkpoint"),
        "official_frozen": _write(
            method_root / "official_test/frozen_tme_representations.jsonl", "{}\n"
        ),
        "official_sdr_scores": _write(
            method_root / "official_test/sdr_scores.jsonl", "{}\n"
        ),
        "official_patterns": _write(
            method_root / "official_test/state_patterns.jsonl", "{}\n"
        ),
        "geometry_metrics": _write(
            method_root / "official_test/geometry_metrics.json", "{}\n"
        ),
    }
    _write(method_root / "calibration/thresholds.json", "{}\n")
    marker = {
        "schema": export.DELIVERY_SCHEMA,
        "delivery": export.DELIVERY,
        "seed": export.SEED,
        "model_key": model_key,
        "method": method,
        "training_config": str(config),
        "training_config_sha256": _sha(config),
        "cache_union": str(union),
        "cache_union_sha256": _sha(union),
        "misread_labels_used": False,
    }
    sha_fields = {
        "best_checkpoint": "best_checkpoint_sha256",
        "official_frozen": "official_frozen_sha256",
        "official_sdr_scores": "official_sdr_sha256",
        "official_patterns": "official_patterns_sha256",
        "geometry_metrics": "geometry_metrics_sha256",
    }
    for field, path in artifacts.items():
        marker[field] = str(path)
        marker[sha_fields[field]] = _sha(path)
    _write(method_root / "RUN_COMPLETE.json", json.dumps(marker))
    return method_root


def _baseline_run(root: Path, repr_key: str, rows: list[dict[str, object]]) -> Path:
    source = root / "qwen3_vl_8b" / repr_key / "official_test/frozen_baseline_representations.jsonl"
    _write(source, "".join(json.dumps(row) + "\n" for row in rows))
    marker = {
        "schema": export.BASELINE_SCHEMA,
        "seed": export.SEED,
        "model_key": "qwen3_vl_8b",
        "repr_key": repr_key,
        "official_manifest": str(source),
        "official_manifest_sha256": _sha(source),
    }
    _write(source.parent.parent / "RUN_COMPLETE.json", json.dumps(marker))
    return source


def _representation_rows(field: str) -> list[dict[str, object]]:
    return [
        {
            "sample_id": "a1",
            "sample_type": "Aligned",
            "representation_split": "official_test",
            field: [1.0, 0.0],
        },
        {
            "sample_id": "c1",
            "sample_type": "Conflict",
            "representation_split": "official_test",
            field: [0.0, 1.0],
        },
    ]


def test_default_export_targets_delivery_dstrong() -> None:
    assert export.DEFAULT_DOWNSTREAM_ROOT == Path(
        "outputs/downstream/delivery_20260716/seed20260717/tme_ablation_v1"
    )
    assert export.STATE_METHOD == "tme_pa_dstrong_v2"
    assert export.TME_REPRESENTATION[0] == export.STATE_METHOD


def test_delivery_state_ready_requires_all_three_exact_dstrong_markers(tmp_path: Path) -> None:
    for model_key, _protocol, _label in export.MODELS:
        _delivery_run(tmp_path, model_key)

    ready = export._ready_state_artifacts(tmp_path)
    assert ready is not None
    assert set(ready) == {model_key for model_key, _protocol, _label in export.MODELS}
    assert all(item.method_root.name == export.STATE_METHOD for item in ready.values())

    marker_path = tmp_path / "internvl3_5_8b" / export.STATE_METHOD / "RUN_COMPLETE.json"
    marker = json.loads(marker_path.read_text())
    marker["official_sdr_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum drift"):
        export._ready_state_artifacts(tmp_path)


def test_old_tme_variants_never_satisfy_final_state_readiness(tmp_path: Path) -> None:
    for model_key, _protocol, _label in export.MODELS:
        _delivery_run(tmp_path, model_key, method="tme_pa_dtheta_v1")
    assert export._ready_state_artifacts(tmp_path) is None


def test_fig08_requires_real_baselines_and_exports_exact_ac_intersection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tme_source = _write(
        tmp_path / "qwen3_vl_8b" / export.STATE_METHOD
        / "official_test/frozen_tme_representations.jsonl",
        "".join(
            json.dumps(row) + "\n"
            for row in _representation_rows("sample_relation_feature")
        ),
    )
    state = {
        "qwen3_vl_8b": export.DeliveryStateArtifacts(
            model_key="qwen3_vl_8b",
            method_root=tme_source.parent.parent,
            scores=tmp_path / "unused-scores",
            patterns=tmp_path / "unused-patterns",
            thresholds=tmp_path / "unused-thresholds",
            frozen_tme=tme_source,
        )
    }
    assert export._ready_fig08_sources(tmp_path, state) is None

    _baseline_run(
        tmp_path,
        "single_point_binary_v1",
        _representation_rows("penultimate_feature"),
    )
    _baseline_run(
        tmp_path,
        "trajectory_mlp_binary_v1",
        _representation_rows("penultimate_feature"),
    )
    sources = export._ready_fig08_sources(tmp_path, state)
    assert sources is not None
    assert [item.label for item in sources] == ["Single-Point", "Trajectory MLP", "TME"]

    monkeypatch.setattr(export, "version", lambda _package: "test")
    output = export._build_fig08(sources, tmp_path / "figure-inputs", ["pytest"])
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["representation"] for row in rows} == {
        "Single-Point",
        "Trajectory MLP",
        "TME",
    }
    assert {row["sample_type"] for row in rows} == {"Aligned", "Conflict"}
    provenance = json.loads(Path(f"{output}.provenance.json").read_text())
    assert provenance["status"] == "Ready"
    assert provenance["sample_masks"]["misread"] == "Pending Misread annotations"
    assert provenance["misread_status"] == "Pending"


def test_fig08_rejects_nonidentical_official_test_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _representation_rows("penultimate_feature")
    sources = [
        export.FigureRepresentation(
            "Single-Point",
            "penultimate_feature",
            _write(tmp_path / "single.jsonl", "".join(json.dumps(row) + "\n" for row in rows)),
        ),
        export.FigureRepresentation(
            "Trajectory MLP",
            "penultimate_feature",
            _write(
                tmp_path / "trajectory.jsonl",
                json.dumps(dict(rows[0], sample_id="different"))
                + "\n"
                + json.dumps(rows[1])
                + "\n",
            ),
        ),
        export.FigureRepresentation(
            "TME",
            "sample_relation_feature",
            _write(
                tmp_path / "tme.jsonl",
                "".join(
                    json.dumps(row) + "\n"
                    for row in _representation_rows("sample_relation_feature")
                ),
            ),
        ),
    ]
    monkeypatch.setattr(export, "version", lambda _package: "test")
    with pytest.raises(RuntimeError, match="exact official_test set"):
        export._build_fig08(tuple(sources), tmp_path / "out", ["pytest"])
