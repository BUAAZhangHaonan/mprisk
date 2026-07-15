from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from mprisk.representation.relation_models import TME_ARCHITECTURE_V1, TME_PROXY_ANCHOR_V1
from mprisk.representation.training import (
    TrainingConfig,
    export_frozen_representations,
    train_trajectory_encoder,
)


def _state(
    root: Path,
    *,
    sample_id: str,
    condition: str,
    prompt_id: str,
    vector: list[float],
    direct_2d: bool = False,
) -> dict[str, object]:
    relative = Path("cache") / f"{sample_id}-{condition}-{prompt_id}.safetensors"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if direct_2d:
        hidden = np.stack(
            [np.asarray(vector, dtype=np.float32), np.asarray(vector[::-1], dtype=np.float32)]
        )
    else:
        hidden = np.zeros((1, 2, 2, len(vector)), dtype=np.float32)
        hidden[0, 0, -1] = np.asarray(vector, dtype=np.float32)
        hidden[0, 1, -1] = np.asarray(vector[::-1], dtype=np.float32)
    save_file({"hidden_states": hidden}, path)
    return {
        "sample_id": sample_id,
        "model_key": "qwen3_vl_8b",
        "protocol": "VT",
        "condition": condition,
        "prompt_set_key": "vt_primary_v1",
        "prompt_id": prompt_id,
        "shard_path": str(relative),
        "index_in_shard": 0,
        "layer_count": 2,
        "hidden_dim": len(vector),
        "token_count": 2,
        "t0_token_index": -1,
        "cache_root": str(root),
        "metadata": {"tensor_key": "hidden_states"},
    }


def _dataset(tmp_path: Path, *, direct_2d: bool = False) -> Path:
    rows = []
    for index in range(8):
        sample_type = "Aligned" if index % 2 == 0 else "Conflict"
        split = "val" if index >= 6 else "train"
        base = [1.0, 0.1 + index * 0.02, 0.2]
        for prompt_index in range(2):
            prompt_id = f"p{prompt_index + 1}"
            conditions = {
                condition: _state(
                    tmp_path,
                    sample_id=f"s{index}",
                    condition=condition,
                    prompt_id=prompt_id,
                    vector=[value + condition_index * 0.03 for value in base],
                    direct_2d=direct_2d,
                )
                for condition_index, condition in enumerate(("M1", "M2", "M12"))
            }
            rows.append(
                {
                    "schema": "mprisk_relation_sample_v1",
                    "row_id": f"s{index}:{prompt_id}",
                    "sample_id": f"s{index}",
                    "sample_type": sample_type,
                    "label_id": int(sample_type == "Conflict"),
                    "model_key": "qwen3_vl_8b",
                    "protocol": "VT",
                    "prompt_set_key": "vt_primary_v1",
                    "prompt_id": prompt_id,
                    "split_group_id": f"g{index}",
                    "master_split": split,
                    "representation_split": (
                        "relation_val" if split == "val" else "relation_train"
                    ),
                    "calibration_split": "",
                    "split_assignment_key": "fixture_v1",
                    "split_assignment_sha256": "a" * 64,
                    "conditions": conditions,
                }
            )
    path = tmp_path / "relation_dataset.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _config(max_epochs: int = 3) -> TrainingConfig:
    return TrainingConfig(
        repr_key=TME_PROXY_ANCHOR_V1,
        model_key="qwen3_vl_8b",
        hidden_dim=6,
        condition_dim=4,
        relation_dim=3,
        dropout=0.0,
        max_epochs=max_epochs,
        batch_size=4,
        lr=0.01,
        weight_decay=0.0,
        proxy_alpha=8.0,
        proxy_margin=0.1,
        patience=2,
        min_delta=0.0,
        seed=7,
    )


def test_tme_training_selects_only_on_val_ac_and_exports_unit_z_r(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    result = train_trajectory_encoder(
        dataset_path=dataset,
        config=_config(),
        output_dir=tmp_path / "run",
    )

    checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
    assert checkpoint["repr_key"] == TME_PROXY_ANCHOR_V1
    assert checkpoint["architecture_version"] == TME_ARCHITECTURE_V1
    assert checkpoint["model_key"] == "qwen3_vl_8b"
    assert checkpoint["selection_metric"] == "val_balanced_accuracy_ac"
    assert checkpoint["proxy_state_dict"]["proxies"].shape == (2, 3)
    logs = [json.loads(line) for line in result.log_path.read_text().splitlines()]
    assert 1 <= len(logs) <= 3
    assert all(
        set(row) >= {"epoch", "train_loss", "val_loss", "val_balanced_accuracy_ac"}
        for row in logs
    )

    exported = export_frozen_representations(
        dataset_path=dataset,
        checkpoint_path=result.best_checkpoint_path,
        output_dir=tmp_path / "frozen",
    )
    rows = [json.loads(line) for line in exported.manifest_path.read_text().splitlines()]
    assert len(rows) == 16
    assert all(set(row["condition_z"]) == {"M1", "M2", "M12"} for row in rows)
    assert all(np.linalg.norm(row["relation_r"]) == pytest.approx(1.0) for row in rows)
    assert all(
        np.linalg.norm(vector) == pytest.approx(1.0)
        for row in rows
        for vector in row["condition_z"].values()
    )
    assert all("misread" not in json.dumps(row).casefold() for row in rows)
    assert result.metrics["train_group_count"] == 6
    assert result.metrics["val_group_count"] == 2
    assert result.metrics["split_assignment_sha256"] == "a" * 64


def test_training_resume_requires_matching_signature_and_continues_epochs(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    first = train_trajectory_encoder(
        dataset_path=dataset,
        config=_config(max_epochs=1),
        output_dir=tmp_path / "run",
    )
    resumed = train_trajectory_encoder(
        dataset_path=dataset,
        config=_config(max_epochs=3),
        output_dir=tmp_path / "run",
        resume_checkpoint=first.last_checkpoint_path,
    )
    logs = [json.loads(line) for line in resumed.log_path.read_text().splitlines()]
    assert logs[-1]["epoch"] > 1
    assert resumed.resumed_from == first.last_checkpoint_path

    bad = tmp_path / "bad.jsonl"
    bad.write_text(dataset.read_text() + dataset.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="resume signature mismatch"):
        train_trajectory_encoder(
            dataset_path=bad,
            config=_config(max_epochs=3),
            output_dir=tmp_path / "bad-run",
            resume_checkpoint=first.last_checkpoint_path,
        )


def test_training_rejects_misread_field_before_loading_cache(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    row = json.loads(dataset.read_text().splitlines()[0])
    row["misread"] = False
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Misread fields are forbidden"):
        train_trajectory_encoder(
            dataset_path=dataset,
            config=_config(),
            output_dir=tmp_path / "run",
        )


def test_relation_training_reads_direct_float32_layer_hidden_cache(tmp_path) -> None:
    dataset = _dataset(tmp_path, direct_2d=True)
    result = train_trajectory_encoder(
        dataset_path=dataset,
        config=_config(max_epochs=1),
        output_dir=tmp_path / "run-2d",
    )
    checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
    assert checkpoint["model_config"] == {"input_dim": 3, "layer_count": 2}


def test_training_never_loads_calibration_or_official_test_cache(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    for index, (master_split, representation_split, sample_type) in enumerate(
        (
            ("val", "aligned_calibration", "Aligned"),
            ("test", "official_test", "Conflict"),
        )
    ):
        row = json.loads(json.dumps(rows[index]))
        row["row_id"] = f"excluded-{index}:p1"
        row["sample_id"] = f"excluded-{index}"
        row["sample_type"] = sample_type
        row["label_id"] = int(sample_type == "Conflict")
        row["split_group_id"] = f"excluded-g{index}"
        row["master_split"] = master_split
        row["representation_split"] = representation_split
        row["calibration_split"] = (
            "aligned_calibration" if representation_split == "aligned_calibration" else ""
        )
        for state in row["conditions"].values():
            state["cache_root"] = str(tmp_path / "must-not-be-read")
            state["shard_path"] = "missing.safetensors"
        rows.append(row)
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = train_trajectory_encoder(
        dataset_path=dataset,
        config=_config(max_epochs=1),
        output_dir=tmp_path / "run-excluded",
    )

    assert result.metrics["train_rows"] == 12
    assert result.metrics["val_rows"] == 4
    assert result.metrics["excluded_rows"] == {
        "aligned_calibration": 1,
        "official_test": 1,
    }


def test_training_rejects_missing_registered_split_instead_of_rehashing(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    row = json.loads(dataset.read_text().splitlines()[0])
    del row["representation_split"]
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="representation_split"):
        train_trajectory_encoder(
            dataset_path=dataset,
            config=_config(max_epochs=1),
            output_dir=tmp_path / "run-missing-split",
        )
