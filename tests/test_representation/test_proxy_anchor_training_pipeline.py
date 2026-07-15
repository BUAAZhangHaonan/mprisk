from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from mprisk.representation import training as training_impl
from mprisk.representation.losses import ProxyAnchorLoss
from mprisk.representation.relation_models import TME_ARCHITECTURE_V1, TME_PROXY_ANCHOR_V1
from mprisk.representation.training import (
    TrainingConfig,
    _aggregate_sample_outputs,
    _load_trajectory_batch,
    _rows_to_sample_refs,
    _sample_level_predictions,
    _sample_prompt_augmentations,
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


def _dataset(
    tmp_path: Path, *, direct_2d: bool = False, prompt_count: int = 2
) -> Path:
    rows = []
    for index in range(8):
        sample_type = "Aligned" if index % 2 == 0 else "Conflict"
        split = "val" if index >= 6 else "train"
        base = [1.0, 0.1 + index * 0.02, 0.2]
        for prompt_index in range(prompt_count):
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
    assert checkpoint["selection_unit"] == "sample_id"
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
    bundles = [
        json.loads(line)
        for line in exported.bundle_manifest_path.read_text().splitlines()
    ]
    assert len(rows) == 16
    assert len(bundles) == 8
    assert all(set(row["condition_z"]) == {"M1", "M2", "M12"} for row in rows)
    assert all(np.linalg.norm(row["relation_r"]) == pytest.approx(1.0) for row in rows)
    assert all(
        np.linalg.norm(vector) == pytest.approx(1.0)
        for row in rows
        for vector in row["condition_z"].values()
    )
    assert all("misread" not in json.dumps(row).casefold() for row in rows)
    for bundle in bundles:
        relation_feature = np.asarray(bundle["sample_relation_feature"], dtype=float)
        prompt_relations = np.asarray(list(bundle["relations"].values()), dtype=float)
        expected = prompt_relations.mean(axis=0)
        expected /= np.linalg.norm(expected)
        assert relation_feature == pytest.approx(expected)
        assert np.linalg.norm(relation_feature) == pytest.approx(1.0)
        assert bundle["aggregation"] == "mean_over_synchronized_prompts_then_l2"
        assert (
            bundle["feature_definition"]
            == "unit_normalized_mean_prompt_ordered_relation_r"
        )
    assert result.metrics["train_group_count"] == 6
    assert result.metrics["val_group_count"] == 2
    assert result.metrics["val_rows"] == 4
    assert result.metrics["val_sample_count"] == 2
    assert result.metrics["selection_unit"] == "sample_id"
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


def test_relation_index_is_metadata_only_and_loads_one_bounded_batch(
    tmp_path, monkeypatch
) -> None:
    dataset = _dataset(tmp_path)
    rows = training_impl._read_relation_rows(
        dataset, expected_model_key="qwen3_vl_8b"
    )
    calls: list[tuple[str, str, str]] = []

    def fake_extract(entry):
        prompt_id = Path(entry.shard_path).stem.rsplit("-", maxsplit=1)[-1]
        calls.append((entry.sample_id, entry.condition, prompt_id))
        return np.zeros((entry.layer_count, entry.hidden_dim), dtype=np.float32)

    monkeypatch.setattr(training_impl, "extract_t0_trajectory", fake_extract)
    refs = _rows_to_sample_refs(rows)

    assert calls == []
    trajectories, labels = _load_trajectory_batch(refs[:2], device=torch.device("cpu"))
    assert trajectories.shape == (2, 3, 2, 3)
    assert labels.shape == (2,)
    assert len(calls) == 2 * 3
    assert set(calls) == {
        (ref.sample_id, condition, ref.prompt_id)
        for ref in refs[:2]
        for condition in ("M1", "M2", "M12")
    }


def test_relation_index_rejects_cross_prompt_condition_pairing_before_cache_io(
    tmp_path, monkeypatch
) -> None:
    dataset = _dataset(tmp_path)
    rows = training_impl._read_relation_rows(
        dataset, expected_model_key="qwen3_vl_8b"
    )
    rows[0]["conditions"]["M12"]["prompt_id"] = "wrong-prompt"
    monkeypatch.setattr(
        training_impl,
        "extract_t0_trajectory",
        lambda _entry: pytest.fail("cache must not be read while indexing metadata"),
    )

    with pytest.raises(ValueError, match="same sample/model/protocol/prompt"):
        _rows_to_sample_refs(rows)


def test_trajectory_loading_paths_never_convert_full_cache_to_python_lists() -> None:
    from mprisk.cache import prefill_extract

    assert ".tolist(" not in inspect.getsource(prefill_extract)
    assert ".tolist(" not in inspect.getsource(training_impl._load_trajectory_batch)
    assert ".tolist(" not in inspect.getsource(training_impl._stream_frozen_exports)


def test_validation_aggregates_eight_prompts_to_one_prediction_per_sample() -> None:
    class Ref:
        def __init__(self, sample_id: str, label_id: int) -> None:
            self.sample_id = sample_id
            self.label_id = label_id

    samples = [Ref("aligned", 0) for _ in range(8)] + [
        Ref("conflict", 1) for _ in range(8)
    ]
    baseline_outputs = torch.tensor([[3.0, 1.0]] * 8 + [[1.0, 4.0]] * 8)

    sample_ids, labels, aggregate = _aggregate_sample_outputs(
        samples, baseline_outputs, normalize=False
    )
    predicted = _sample_level_predictions(aggregate, objective=None)

    assert sample_ids == ["aligned", "conflict"]
    assert labels == [0, 1]
    assert aggregate.shape == (2, 2)
    assert predicted.cpu().numpy().tolist() == [0, 1]

    relation_outputs = torch.tensor([[2.0, 0.0]] * 8 + [[0.0, 3.0]] * 8)
    _, relation_labels, aggregate_r = _aggregate_sample_outputs(
        samples, relation_outputs, normalize=True
    )
    objective = ProxyAnchorLoss(embed_dim=2, num_classes=2)
    with torch.no_grad():
        objective.proxies.copy_(torch.eye(2))
    relation_predicted = _sample_level_predictions(aggregate_r, objective=objective)

    assert relation_labels == [0, 1]
    torch.testing.assert_close(torch.linalg.vector_norm(aggregate_r, dim=-1), torch.ones(2))
    assert relation_predicted.cpu().numpy().tolist() == [0, 1]


def test_training_selects_one_reproducible_prompt_per_sample_and_epoch(
    tmp_path, monkeypatch
) -> None:
    dataset = _dataset(tmp_path, prompt_count=8)
    rows = training_impl._read_relation_rows(
        dataset, expected_model_key="qwen3_vl_8b"
    )
    refs = _rows_to_sample_refs(
        [row for row in rows if row["representation_split"] == "relation_train"]
    )

    epoch_one = _sample_prompt_augmentations(refs, seed=7, epoch=1)
    epoch_one_repeated = _sample_prompt_augmentations(refs, seed=7, epoch=1)
    epoch_two = _sample_prompt_augmentations(refs, seed=7, epoch=2)
    resumed_epoch_two = _sample_prompt_augmentations(refs, seed=7, epoch=2)

    assert len(epoch_one) == 6
    assert len({sample.sample_id for sample in epoch_one}) == 6
    assert [(sample.sample_id, sample.prompt_id) for sample in epoch_one] == [
        (sample.sample_id, sample.prompt_id) for sample in epoch_one_repeated
    ]
    assert [(sample.sample_id, sample.prompt_id) for sample in epoch_two] == [
        (sample.sample_id, sample.prompt_id) for sample in resumed_epoch_two
    ]
    assert {
        sample.sample_id: sample.prompt_id for sample in epoch_one
    }.keys() == {sample.sample_id: sample.prompt_id for sample in epoch_two}.keys()
    assert all(
        first.prompt_id != second.prompt_id
        for first, second in zip(epoch_one, epoch_two, strict=True)
    )

    calls = []

    def fake_extract(entry):
        calls.append((entry.sample_id, entry.condition))
        return np.ones((entry.layer_count, entry.hidden_dim), dtype=np.float32)

    monkeypatch.setattr(training_impl, "extract_t0_trajectory", fake_extract)
    for batch in training_impl._batches(epoch_one, 4):
        _load_trajectory_batch(batch, device=torch.device("cpu"))
    assert len(calls) == 6 * 3


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
