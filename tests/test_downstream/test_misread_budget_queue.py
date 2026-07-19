from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import mprisk.experiments.misread_budget_queue as queue
from mprisk.experiments.misread_budget_queue import (
    FRACTIONS,
    METHODS,
    FractionProbeResult,
    MisreadBudgetQueueError,
    PendingFractionsError,
    QueueModel,
    QueuePlan,
    audit_fraction_complete,
    derive_ready_probe_config,
    finalize_misread_budget_queue,
    load_misread_budget_queue_config,
    run_misread_budget_queue,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_ids_sha(sample_ids: set[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(sample_ids), separators=(",", ":")).encode()
    ).hexdigest()


def _write(path: Path, content: str = "fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _json(path: Path, payload: dict) -> Path:
    return _write(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _jsonl(path: Path, rows: list[dict]) -> Path:
    return _write(
        path,
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
    )


def _plan(tmp_path: Path, models: tuple[QueueModel, ...] | None = None) -> QueuePlan:
    plan_path = _write(tmp_path / "queue.yaml", "schema: fixture\n")
    return QueuePlan(
        path=plan_path,
        delivery="delivery_20260716",
        seed=20260717,
        fractions=FRACTIONS,
        budget_root=(tmp_path / "budget").resolve(),
        formal_label_root=(tmp_path / "labels").resolve(),
        output_root=(tmp_path / "queue-output").resolve(),
        lock_path=(tmp_path / "queue.lock").resolve(),
        poll_seconds=30.0,
        models=models or (QueueModel("qwen3_vl_8b", "vt", "vt_main_p8_seed20260717"),),
        training={
            "seed": 20260717,
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "hidden_dim": 128,
            "dropout": 0.1,
            "device": "cpu",
        },
    )


def _fraction_fixture(plan: QueuePlan, model: QueueModel, fraction: float = 0.1) -> Path:
    fraction_root = plan.budget_root / model.model_key / f"fraction_{fraction:.2f}"
    full_relation = _write(plan.budget_root / model.model_key / "full_relation.jsonl")
    training_relation = _write(fraction_root / "training_relation/relation_dataset.jsonl")
    sample_ids = {"train-id", "val-id", "test-id"}
    sample_sha = _sample_ids_sha(sample_ids)
    retained_sha = "d" * 64
    method_entries: dict[str, dict] = {}
    dimensions = {"single_point": 2, "trajectory_mlp": 3, "tme": 4}
    for method in METHODS:
        method_root = fraction_root / method
        repr_key, feature_field = queue.METHOD_CONTRACTS[method]
        manifest = _jsonl(
            method_root / "conflict_probe/representations.jsonl",
            [
                {
                    "schema": "frozen-fixture-v1",
                    "sample_id": sample_id,
                    "model_key": model.model_key,
                    "protocol": model.protocol,
                    "prompt_set_key": model.prompt_set_key,
                    "repr_key": repr_key,
                    "sample_type": "Conflict",
                    "representation_split": split,
                    "split_assignment_key": "delivery-split-v1",
                    "split_assignment_sha256": "a" * 64,
                    feature_field: [float(index + 1) for index in range(dimensions[method])],
                }
                for sample_id, split in (
                    ("train-id", "relation_train"),
                    ("val-id", "relation_val"),
                    ("test-id", "official_test"),
                )
            ],
        )
        references = {
            "training_config": _write(plan.budget_root / f"{method}.yaml"),
            "full_relation_dataset": full_relation,
            "training_relation_dataset": training_relation,
            "best_checkpoint": _write(method_root / "training/best_checkpoint.pt"),
            "training_metrics": _write(method_root / "training/train_metrics.json", "{}\n"),
            "frozen_summary": _write(method_root / "frozen/summary.json", "{}\n"),
            "official_manifest": _write(method_root / "official/representations.jsonl"),
            "official_ac_metrics": _write(method_root / "official/metrics.json", "{}\n"),
            "conflict_probe_manifest": manifest,
        }
        marker = {
            "schema": queue.METHOD_COMPLETE_SCHEMA,
            "delivery": plan.delivery,
            "seed": plan.seed,
            "model_key": model.model_key,
            "protocol": model.protocol,
            "method": method,
            "repr_key": repr_key,
            "conflict_supervision_fraction": fraction,
            "retained_conflict_group_ids_sha256": retained_sha,
            "conflict_probe_sample_ids_sha256": sample_sha,
            "conflict_probe_sample_count": len(sample_ids),
            "probe_splits": list(queue.PROBE_SPLITS),
            "misread_labels_used_for_encoder_training": False,
        }
        for field, path in references.items():
            marker[field] = str(path)
            marker[f"{field}_sha256"] = _sha(path)
        method_marker = _json(method_root / "RUN_COMPLETE.json", marker)
        method_entries[method] = {"path": str(method_marker), "sha256": _sha(method_marker)}
    fraction_marker = {
        "schema": queue.FRACTION_COMPLETE_SCHEMA,
        "model_key": model.model_key,
        "fraction": fraction,
        "full_relation_dataset_sha256": _sha(full_relation),
        "training_relation_dataset_sha256": _sha(training_relation),
        "retained_conflict_group_ids_sha256": retained_sha,
        "full_conflict_probe_sample_ids_sha256": sample_sha,
        "full_conflict_probe_sample_count": len(sample_ids),
        "method_markers": method_entries,
        "misread_labels_used_for_encoder_training": False,
    }
    return _json(fraction_root / "FRACTION_COMPLETE.json", fraction_marker)


def test_registered_delivery_queue_config_has_30_second_poll() -> None:
    plan = load_misread_budget_queue_config(
        Path("configs/downstream/delivery_20260716_misread_budget_queue_v1.yaml")
    )
    assert plan.poll_seconds == 30.0
    assert {model.model_key for model in plan.models} == queue.MODEL_KEYS
    assert plan.training["device"] == "cpu"


def test_fraction_audit_requires_three_markers_and_derives_absolute_ready_config(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    model = plan.models[0]
    _fraction_fixture(plan, model)

    source = audit_fraction_complete(plan, model, 0.1)
    labels = queue.LabelSnapshot(
        model.model_key,
        _write(tmp_path / "labels/labels/qwen3_vl_8b.jsonl"),
        "1" * 64,
        _write(tmp_path / "labels/COMPLETE.json", "{}\n"),
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
        1,
    )
    payload = derive_ready_probe_config(plan, model, source, labels)

    assert [row["expected_feature_dim"] for row in payload["representations"]] == [
        2,
        3,
        4,
    ]
    assert payload["status"] == "ready"
    assert Path(payload["labels"]["root"]).is_absolute()
    assert all(Path(row["path"]).is_absolute() for row in payload["representations"])
    assert Path(payload["output_root"]).is_absolute()


def test_fraction_audit_rejects_partial_method_selection(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    model = plan.models[0]
    marker_path = _fraction_fixture(plan, model)
    marker = json.loads(marker_path.read_text())
    marker["method_markers"].pop("tme")
    _json(marker_path, marker)

    with pytest.raises(MisreadBudgetQueueError, match="exactly three"):
        audit_fraction_complete(plan, model, 0.1)


def test_fraction_audit_rejects_referenced_artifact_sha_drift(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    model = plan.models[0]
    marker_path = _fraction_fixture(plan, model)
    marker = json.loads(marker_path.read_text())
    method_path = Path(marker["method_markers"]["single_point"]["path"])
    method = json.loads(method_path.read_text())
    Path(method["best_checkpoint"]).write_text("tampered\n")

    with pytest.raises(MisreadBudgetQueueError, match="referenced artifact SHA mismatch"):
        audit_fraction_complete(plan, model, 0.1)


def test_once_fails_when_any_fraction_marker_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    monkeypatch.setattr(queue, "load_misread_budget_queue_config", lambda _: plan)
    monkeypatch.setattr(
        queue,
        "process_available_fractions",
        lambda _: ([], ["qwen3_vl_8b/fraction_0.25"]),
    )

    with pytest.raises(PendingFractionsError, match="fraction_0.25"):
        run_misread_budget_queue("unused.yaml", once=True)
    assert plan.lock_path.is_file()


def _result(plan: QueuePlan, model: QueueModel, fraction: float) -> FractionProbeResult:
    fraction_root = plan.output_root / model.model_key / f"fraction_{fraction:.2f}"
    source = _write(fraction_root / "source.json", "{}\n")
    config = _write(fraction_root / "probe_config.yaml")
    probe = _write(fraction_root / "probe/RUN_COMPLETE.json", "{}\n")
    _write(fraction_root / "FRACTION_PROBE_COMPLETE.json", "{}\n")
    rows = tuple(
        {
            "model_key": model.model_key,
            "protocol": model.protocol,
            "fraction": f"{fraction:.2f}",
            "representation": method,
            "eligible_sample_ids_sha256": "a" * 64,
            "official_test_sample_ids_sha256": "b" * 64,
            "official_test_sample_count": 3,
            "accuracy": 0.8,
            "balanced_accuracy": 0.75,
            "macro_f1": 0.7,
            "ap": 0.9,
        }
        for method in METHODS
    )
    return FractionProbeResult(
        model.model_key,
        model.protocol,
        fraction,
        source,
        _sha(source),
        config,
        _sha(config),
        probe,
        _sha(probe),
        "a" * 64,
        "b" * 64,
        3,
        rows,
    )


def test_finalize_enforces_cross_fraction_identity_and_writes_metric_contract(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    model = plan.models[0]
    results = [_result(plan, model, fraction) for fraction in FRACTIONS]

    marker_path = finalize_misread_budget_queue(plan, results)
    marker = json.loads(marker_path.read_text())
    with Path(marker["metrics_csv"]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 12
    assert set(("accuracy", "balanced_accuracy", "macro_f1", "ap")) <= set(rows[0])
    assert marker["misread_labels_used_for_encoder_training"] is False
    assert (
        _sha(marker_path) in (plan.output_root / "MISREAD_BUDGET_COMPLETE.json.sha256").read_text()
    )
    assert finalize_misread_budget_queue(plan, results) == marker_path

    drifted = [*results]
    drifted[-1] = replace(drifted[-1], official_test_sample_ids_sha256="c" * 64)
    with pytest.raises(MisreadBudgetQueueError, match="official-test sample IDs differ"):
        finalize_misread_budget_queue(plan, drifted)
