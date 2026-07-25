"""Data IO and validation utilities for sample-level relation training.

Round 2 split (P2-R2-A): data loading / schema / split-validation helpers
extracted verbatim from ``training.py``. Behaviour is preserved exactly;
only the module location changed.

Note: ``extract_t0_trajectory`` is imported at module top-level so that
tests using ``monkeypatch.setattr(mprisk.representation.data,
"extract_t0_trajectory", fake)`` resolve to a writable attribute, just as
they previously did on ``training``. ``training.py`` still re-imports the
same name for its own use; Round 3 will consolidate the patch target.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mprisk.cache.prefill_extract import extract_t0_trajectory
from mprisk.cache.prompt_conditioned_cache import prompt_conditioned_entry_from_row
from mprisk.representation.config import (
    REGISTERED_SPLITS,
    TrainingConfig,
    _Sample,
)
from mprisk.representation.relation_dataset import CONDITIONS, _reject_forbidden_fields
from mprisk.representation.relation_models import (
    SINGLE_POINT_BINARY_V1,
    TME_ARCHITECTURE_LSTM_V1,
    TME_ARCHITECTURE_V1,
    TME_PROXY_ANCHOR_V1,
    TRAJECTORY_MLP_BINARY_V1,
)

__all__ = [
    "_validate_prompt_contract",
    "_validate_checkpoint_architecture",
    "_vector_values",
    "_read_relation_rows",
    "_rows_to_sample_refs",
    "_load_trajectory_batch",
    "_registered_group_split",
    "_validate_registered_splits",
    "_trajectory_shape",
    "_batches",
    "_baseline_feature_definition",
]


def _baseline_feature_definition(repr_key: str) -> str:
    if repr_key == SINGLE_POINT_BINARY_V1:
        return "mean_prompt_final_layer_m1_m2_m12_concat"
    if repr_key == TRAJECTORY_MLP_BINARY_V1:
        return "mean_prompt_first_linear_gelu_hidden"
    raise ValueError(f"unsupported baseline representation: {repr_key}")


def _validate_prompt_contract(samples: list[_Sample], *, config: TrainingConfig) -> None:
    grouped: dict[str, list[str]] = {}
    for sample in samples:
        if sample.prompt_set_key != config.prompt_set_key:
            raise ValueError(
                f"sample {sample.sample_id} prompt_set_key does not match training config"
            )
        grouped.setdefault(sample.sample_id, []).append(sample.prompt_id)
    expected_prompt_ids = set(config.expected_prompt_ids)
    for sample_id in sorted(grouped):
        prompt_ids = grouped[sample_id]
        unique_prompt_ids = set(prompt_ids)
        if len(unique_prompt_ids) != len(prompt_ids):
            raise ValueError(f"sample {sample_id} has duplicate prompt rows")
        if len(prompt_ids) != config.expected_prompt_count:
            raise ValueError(
                f"sample {sample_id} must have exactly {config.expected_prompt_count} prompts; "
                f"found {len(prompt_ids)}"
            )
        if unique_prompt_ids != expected_prompt_ids:
            raise ValueError(
                f"sample {sample_id} prompt IDs do not match the configured prompt set"
            )


def _validate_checkpoint_architecture(checkpoint: dict[str, Any]) -> None:
    repr_key = str(checkpoint.get("repr_key", ""))
    architecture_version = str(checkpoint.get("architecture_version", ""))
    if repr_key == TME_PROXY_ANCHOR_V1:
        # TME checkpoints can be either GRU (V1) or LSTM (LSTM_V1). Both
        # are valid for resuming as long as the architecture_version field
        # matches one of the two known TME architectures.
        if architecture_version not in (TME_ARCHITECTURE_V1, TME_ARCHITECTURE_LSTM_V1):
            raise ValueError(
                "checkpoint architecture_version does not match its representation"
            )
        return
    expected_architecture = repr_key
    if architecture_version != expected_architecture:
        raise ValueError("checkpoint architecture_version does not match its representation")
    if repr_key != SINGLE_POINT_BINARY_V1:
        return
    model_state = checkpoint.get("model_state_dict")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_state, dict) or not isinstance(model_config, dict):
        raise ValueError("Single-Point checkpoint architecture drift: metadata is incomplete")
    input_dim = model_config.get("input_dim")
    weight = model_state.get("classifier.weight")
    bias = model_state.get("classifier.bias")
    accepted_weight_shapes = {(2, 3 * input_dim), (2, input_dim)}
    if (
        not isinstance(input_dim, int)
        or input_dim <= 0
        or set(model_state) != {"classifier.weight", "classifier.bias"}
        or not isinstance(weight, torch.Tensor)
        or tuple(weight.shape) not in accepted_weight_shapes
        or not isinstance(bias, torch.Tensor)
        or tuple(bias.shape) != (2,)
    ):
        raise ValueError(
            "Single-Point checkpoint architecture drift: expected direct Linear(H or 3H, 2)"
        )


def _vector_values(vector: torch.Tensor) -> list[float]:
    return [float(value) for value in vector.detach().cpu().numpy()]


def _read_relation_rows(
    path: str | Path,
    *,
    expected_model_key: str,
    expected_protocol: str,
    expected_prompt_set_artifact_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: relation row must be an object")
            _reject_forbidden_fields(row)
            if row.get("schema") != "mprisk_relation_sample_v1":
                raise ValueError("relation row schema mismatch")
            if row.get("model_key") != expected_model_key:
                raise ValueError("relation dataset model_key does not match training backbone")
            if str(row.get("protocol", "")).lower() != expected_protocol.lower():
                raise ValueError("relation dataset protocol does not match training config")
            if row.get("prompt_set_artifact_sha256") != expected_prompt_set_artifact_sha256:
                raise ValueError(
                    "relation dataset prompt artifact SHA does not match training config"
                )
            if row.get("sample_type") not in {"Aligned", "Conflict"}:
                raise ValueError("relation training labels must be Conflict or Aligned")
            expected_label = int(row["sample_type"] == "Conflict")
            if row.get("label_id") != expected_label:
                raise ValueError("label_id must be derived from the sample-level A/C label")
            if set(row.get("conditions", {})) != set(CONDITIONS):
                raise ValueError("relation row requires exactly M1, M2, and M12")
            rows.append(row)
    if not rows:
        raise ValueError("relation dataset is empty")
    row_ids = [str(row.get("row_id")) for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("relation dataset row_id values must be unique")
    return rows


def _rows_to_sample_refs(rows: list[dict[str, Any]]) -> list[_Sample]:
    samples: list[_Sample] = []
    expected_shape: tuple[int, int] | None = None
    for row in rows:
        entries = tuple(
            prompt_conditioned_entry_from_row(row["conditions"][condition])
            for condition in CONDITIONS
        )
        expected_key = (
            str(row["sample_id"]),
            str(row["model_key"]),
            str(row["protocol"]).lower(),
            str(row["prompt_set_key"]),
            str(row["prompt_id"]),
        )
        for condition, entry in zip(CONDITIONS, entries, strict=True):
            actual_key = (
                entry.sample_id,
                entry.model_key,
                entry.protocol,
                entry.prompt_set_key,
                entry.prompt_id,
            )
            if actual_key != expected_key or entry.condition != condition:
                raise ValueError(
                    "M1, M2, and M12 cache entries must use the same "
                    "sample/model/protocol/prompt as the relation row"
                )
        shapes = {(entry.layer_count, entry.hidden_dim) for entry in entries}
        if len(shapes) != 1:
            raise ValueError("all three condition trajectories must have the same shape")
        shape = next(iter(shapes))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("all condition trajectories must have the same layer/hidden shape")
        samples.append(
            _Sample(
                row_id=str(row["row_id"]),
                sample_id=str(row["sample_id"]),
                sample_type=str(row["sample_type"]),
                label_id=int(row["label_id"]),
                split_group_id=str(row["split_group_id"]),
                master_split=str(row.get("master_split", "")),
                representation_split=str(row.get("representation_split", "")),
                calibration_split=str(row.get("calibration_split", "")),
                split_assignment_key=str(row.get("split_assignment_key", "")),
                split_assignment_sha256=str(row.get("split_assignment_sha256", "")),
                protocol=str(row["protocol"]),
                prompt_set_key=str(row["prompt_set_key"]),
                prompt_id=str(row["prompt_id"]),
                condition_entries=tuple(entry.to_hidden_state_entry() for entry in entries),
            )
        )
    return samples


def _load_trajectory_batch(
    batch: list[_Sample], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    arrays = [
        np.stack([extract_t0_trajectory(entry) for entry in sample.condition_entries])
        for sample in batch
    ]
    trajectories = torch.from_numpy(np.stack(arrays).astype(np.float32, copy=False)).to(device)
    labels = torch.tensor([sample.label_id for sample in batch], dtype=torch.long, device=device)
    return trajectories, labels


def _registered_group_split(samples: list[_Sample]) -> tuple[list[_Sample], list[_Sample]]:
    groups: dict[str, list[_Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.split_group_id, []).append(sample)
    for group_samples in groups.values():
        splits = {sample.representation_split for sample in group_samples}
        if len(splits) != 1:
            raise ValueError("split_group_id crosses registered representation splits")
    train = [sample for sample in samples if sample.representation_split == "relation_train"]
    val = [sample for sample in samples if sample.representation_split == "relation_val"]
    if not train or not val:
        raise ValueError("registered relation_train and relation_val splits are both required")
    if {sample.label_id for sample in train} != {0, 1} or {sample.label_id for sample in val} != {
        0,
        1,
    }:
        raise ValueError("train and val must both contain Aligned and Conflict samples")
    return train, val


def _validate_registered_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    expected_master = {
        "relation_train": "train",
        "relation_val": "val",
        "aligned_calibration": "val",
        "official_test": "test",
        # canonical_rerun_v2 (20260721): cross_domain_test rows carry
        # master_split == representation_split == "cross_domain_test"
        # (see relation_dataset.py build_relation_dataset).
        "cross_domain_test": "cross_domain_test",
    }
    group_splits: dict[str, set[str]] = {}
    keys: set[str] = set()
    checksums: set[str] = set()
    for row in rows:
        for field in (
            "split_group_id",
            "master_split",
            "representation_split",
            "split_assignment_key",
            "split_assignment_sha256",
        ):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"relation row requires non-empty {field}")
        split = row["representation_split"]
        if split not in REGISTERED_SPLITS:
            raise ValueError(f"unknown representation_split: {split}")
        if row["master_split"] != expected_master[split]:
            raise ValueError(f"{split} mismatches official master_split")
        expected_calibration = "aligned_calibration" if split == "aligned_calibration" else ""
        if str(row.get("calibration_split", "")) != expected_calibration:
            raise ValueError(f"{split} has invalid calibration_split")
        group_splits.setdefault(row["split_group_id"], set()).add(split)
        keys.add(row["split_assignment_key"])
        checksums.add(row["split_assignment_sha256"])
    leaked = [group for group, splits in group_splits.items() if len(splits) != 1]
    if leaked:
        raise ValueError(f"split groups cross registered assignments: {leaked[:3]}")
    if len(keys) != 1 or len(checksums) != 1 or len(next(iter(checksums), "")) != 64:
        raise ValueError("relation rows require one valid split assignment key/checksum")
    return {
        "split_assignment_key": next(iter(keys)),
        "split_assignment_sha256": next(iter(checksums)),
    }


def _trajectory_shape(samples: list[_Sample]) -> tuple[int, int]:
    entry = samples[0].condition_entries[0]
    return int(entry.layer_count), int(entry.hidden_dim)


def _batches(samples: list[_Sample], batch_size: int) -> list[list[_Sample]]:
    return [samples[index : index + batch_size] for index in range(0, len(samples), batch_size)]
