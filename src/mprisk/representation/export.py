"""Frozen representation export for sample-level relation representations.

This module isolates the export-to-disk entry points (TME checkpoint →
``frozen_representations.jsonl`` + ``spherical_embedding_manifest.jsonl`` and
Single-Point/Trajectory-MLP checkpoint → ``frozen_baseline_representations.jsonl``)
from the training loop in :mod:`mprisk.representation.training`.

The two public entry points (:func:`export_frozen_representations` and
:func:`export_frozen_baseline_representations`) plus their private helpers were
factored out without any change to function bodies. The helpers in
:mod:`mprisk.representation.data` (validation, dataset reading, trajectory
loading, batching, vector serialization) are imported lazily inside the
function bodies because :mod:`mprisk.representation.data` is produced by a
parallel refactor step (P2-R2-A) and may not yet exist on this branch. The
laziness keeps this module importable on its own; calls into it will only
succeed once :mod:`mprisk.representation.data` is available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mprisk.representation.config import (
    FrozenBaselineExportResult,
    FrozenRepresentationExportResult,
    TrainingConfig,
    _Sample,
    _validate_config,
)
from mprisk.representation._io_utils import _sha256
from mprisk.representation.relation_dataset import CONDITIONS
from mprisk.representation.relation_models import (
    SINGLE_POINT_BINARY_V1,
    TME_ARCHITECTURE_LSTM_V1,
    TME_ARCHITECTURE_V1,
    TME_PROXY_ANCHOR_V1,
    TRAJECTORY_MLP_BINARY_V1,
    build_representation_model,
    strict_l2_normalize,
)
from mprisk.utils.io import write_json

__all__ = [
    "export_frozen_representations",
    "export_frozen_baseline_representations",
    "_stream_baseline_exports",
    "_baseline_export_row",
    "_stream_frozen_exports",
    "_frozen_row",
    "_empty_frozen_bundle",
    "_append_frozen_row",
    "_finalize_frozen_bundle",
]


def export_frozen_representations(
    *,
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> FrozenRepresentationExportResult:
    from mprisk.representation.data import (
        _read_relation_rows,
        _rows_to_sample_refs,
        _validate_checkpoint_architecture,
        _validate_prompt_contract,
    )

    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    _validate_checkpoint_architecture(checkpoint)
    if checkpoint.get("checkpoint_role") == "unconstrained_diagnostic":
        raise ValueError("unconstrained diagnostic checkpoint cannot be exported")
    training_config = checkpoint.get("training_config")
    if (
        isinstance(training_config, dict)
        and training_config.get("enable_state_supervision") is True
        and (
            checkpoint.get("checkpoint_role") != "final_selected"
            or not checkpoint.get("checkpoint_feasibility", {}).get("feasible", False)
        )
    ):
        raise ValueError(
            "state-supervised TME export requires the final feasible checkpoint"
        )
    if checkpoint.get("repr_key") != TME_PROXY_ANCHOR_V1:
        raise ValueError(
            "condition z and relation r export requires a tme_proxy_anchor_v1 checkpoint"
        )
    config = TrainingConfig(**checkpoint["training_config"])
    _validate_config(config)
    rows = _read_relation_rows(
        dataset_path,
        expected_model_key=config.model_key,
        expected_protocol=config.protocol,
        expected_prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
    )
    samples = _rows_to_sample_refs(rows)
    _validate_prompt_contract(samples, config=config)
    model = build_representation_model(
        config.repr_key,
        input_dim=int(checkpoint["model_config"]["input_dim"]),
        layer_count=int(checkpoint["model_config"]["layer_count"]),
        hidden_dim=config.hidden_dim,
        condition_dim=config.condition_dim,
        relation_dim=config.relation_dim,
        dropout=config.dropout,
        encoder_type=getattr(config, "encoder_type", "gru"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "frozen_representations.jsonl"
    bundle_manifest_path = output_root / "spherical_embedding_manifest.jsonl"
    sample_count = _stream_frozen_exports(
        samples=sorted(samples, key=lambda sample: (sample.sample_id, sample.prompt_id)),
        model=model,
        config=config,
        manifest_path=manifest_path,
        bundle_manifest_path=bundle_manifest_path,
        encoder_checkpoint_sha256=_sha256(checkpoint_file),
    )
    summary_path = write_json(
        output_root / "frozen_representation_summary.json",
        {
            "schema": "mprisk_frozen_spherical_representation_summary_v1",
            "checkpoint": str(checkpoint_path),
            "dataset": str(dataset_path),
            "count": len(samples),
            "sample_count": sample_count,
            "bundle_manifest": str(bundle_manifest_path),
            "repr_key": config.repr_key,
            "model_key": config.model_key,
            "prompt_set_key": config.prompt_set_key,
            "prompt_set_artifact_sha256": config.prompt_set_artifact_sha256,
            "encoder_checkpoint_sha256": _sha256(checkpoint_file),
        },
    )
    return FrozenRepresentationExportResult(
        manifest_path, bundle_manifest_path, summary_path, len(samples)
    )


def export_frozen_baseline_representations(
    *,
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    representation_split: str = "official_test",
) -> FrozenBaselineExportResult:
    from mprisk.representation.data import (
        _baseline_feature_definition,
        _read_relation_rows,
        _rows_to_sample_refs,
        _validate_checkpoint_architecture,
        _validate_prompt_contract,
        _validate_registered_splits,
    )

    if representation_split not in {"relation_train", "relation_val", "aligned_calibration", "official_test"}:
        raise ValueError("baseline export requires a valid representation split")
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    _validate_checkpoint_architecture(checkpoint)
    repr_key = str(checkpoint.get("repr_key", ""))
    if repr_key not in {SINGLE_POINT_BINARY_V1, TRAJECTORY_MLP_BINARY_V1}:
        raise ValueError("baseline export requires a Single-Point or Trajectory MLP checkpoint")
    if checkpoint.get("proxy_state_dict") is not None:
        raise ValueError("baseline checkpoints must not contain Proxy Anchor state")
    config = TrainingConfig(**checkpoint["training_config"])
    _validate_config(config)
    rows = _read_relation_rows(
        dataset_path,
        expected_model_key=config.model_key,
        expected_protocol=config.protocol,
        expected_prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
    )
    _validate_registered_splits(rows)
    selected_rows = [row for row in rows if row["representation_split"] == representation_split]
    if not selected_rows:
        raise ValueError(f"relation dataset has no rows for {representation_split}")
    samples = sorted(
        _rows_to_sample_refs(selected_rows),
        key=lambda sample: (sample.sample_id, sample.prompt_id),
    )
    _validate_prompt_contract(samples, config=config)
    model = build_representation_model(
        repr_key,
        input_dim=int(checkpoint["model_config"]["input_dim"]),
        layer_count=int(checkpoint["model_config"]["layer_count"]),
        hidden_dim=config.hidden_dim,
        condition_dim=config.condition_dim,
        relation_dim=config.relation_dim,
        dropout=config.dropout,
        encoder_type=getattr(config, "encoder_type", "gru"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "frozen_baseline_representations.jsonl"
    checkpoint_sha256 = _sha256(checkpoint_file)
    sample_count, feature_dim = _stream_baseline_exports(
        samples=samples,
        model=model,
        batch_size=config.batch_size,
        model_key=config.model_key,
        repr_key=repr_key,
        checkpoint_sha256=checkpoint_sha256,
        prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
        manifest_path=manifest_path,
    )
    summary_path = write_json(
        output_root / "frozen_baseline_summary.json",
        {
            "schema": "mprisk_frozen_baseline_summary_v1",
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(Path(dataset_path)),
            "checkpoint": str(checkpoint_file),
            "encoder_checkpoint_sha256": checkpoint_sha256,
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "model_key": config.model_key,
            "prompt_set_key": config.prompt_set_key,
            "prompt_set_artifact_sha256": config.prompt_set_artifact_sha256,
            "repr_key": repr_key,
            "representation_split": representation_split,
            "aggregation": "mean_over_synchronized_prompts",
            "feature_definition": _baseline_feature_definition(repr_key),
            "feature_dim": feature_dim,
            "sample_count": sample_count,
        },
    )
    return FrozenBaselineExportResult(manifest_path, summary_path, sample_count)


def _stream_baseline_exports(
    *,
    samples: list[_Sample],
    model: nn.Module,
    batch_size: int,
    model_key: str,
    repr_key: str,
    checkpoint_sha256: str,
    prompt_set_artifact_sha256: str,
    manifest_path: Path,
) -> tuple[int, int]:
    from mprisk.representation.data import _batches, _load_trajectory_batch

    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    current_sample: _Sample | None = None
    feature_sum: torch.Tensor | None = None
    logits_sum: torch.Tensor | None = None
    prompt_count = 0
    prompt_counts: set[int] = set()
    sample_count = 0
    feature_dim = 0
    with temporary.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch in _batches(samples, batch_size):
            trajectories, _labels = _load_trajectory_batch(
                batch, device=next(model.parameters()).device
            )
            features = model.forward_features(trajectories)
            logits = model.classifier(features)
            for index, sample in enumerate(batch):
                if current_sample is not None and sample.sample_id != current_sample.sample_id:
                    row = _baseline_export_row(
                        current_sample,
                        feature_sum=feature_sum,
                        logits_sum=logits_sum,
                        prompt_count=prompt_count,
                        model_key=model_key,
                        repr_key=repr_key,
                        checkpoint_sha256=checkpoint_sha256,
                        prompt_set_artifact_sha256=prompt_set_artifact_sha256,
                    )
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    prompt_counts.add(prompt_count)
                    sample_count += 1
                    feature_dim = len(row["penultimate_feature"])
                    feature_sum = None
                    logits_sum = None
                    prompt_count = 0
                current_sample = sample
                feature_sum = (
                    features[index].clone()
                    if feature_sum is None
                    else feature_sum + features[index]
                )
                logits_sum = (
                    logits[index].clone() if logits_sum is None else logits_sum + logits[index]
                )
                prompt_count += 1
        if current_sample is not None:
            row = _baseline_export_row(
                current_sample,
                feature_sum=feature_sum,
                logits_sum=logits_sum,
                prompt_count=prompt_count,
                model_key=model_key,
                repr_key=repr_key,
                checkpoint_sha256=checkpoint_sha256,
                prompt_set_artifact_sha256=prompt_set_artifact_sha256,
            )
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            prompt_counts.add(prompt_count)
            sample_count += 1
            feature_dim = len(row["penultimate_feature"])
        handle.flush()
        os.fsync(handle.fileno())
    if len(prompt_counts) != 1:
        raise ValueError("held-out samples must have synchronized prompt counts")
    os.replace(temporary, manifest_path)
    return sample_count, feature_dim


def _baseline_export_row(
    sample: _Sample,
    *,
    feature_sum: torch.Tensor | None,
    logits_sum: torch.Tensor | None,
    prompt_count: int,
    model_key: str,
    repr_key: str,
    checkpoint_sha256: str,
    prompt_set_artifact_sha256: str,
) -> dict[str, Any]:
    from mprisk.representation.data import _baseline_feature_definition, _vector_values

    if feature_sum is None or logits_sum is None or prompt_count <= 0:
        raise ValueError("baseline sample aggregate is empty")
    mean_feature = feature_sum / prompt_count
    mean_logits = logits_sum / prompt_count
    prediction_id = int(mean_logits.argmax())
    return {
        "schema": "mprisk_frozen_baseline_representation_v1",
        "sample_id": sample.sample_id,
        "sample_type": sample.sample_type,
        "label_id": sample.label_id,
        "model_key": model_key,
        "protocol": sample.protocol,
        "prompt_set_key": sample.prompt_set_key,
        "master_split": sample.master_split,
        "representation_split": sample.representation_split,
        "split_group_id": sample.split_group_id,
        "split_assignment_key": sample.split_assignment_key,
        "split_assignment_sha256": sample.split_assignment_sha256,
        "repr_key": repr_key,
        "encoder_checkpoint_sha256": checkpoint_sha256,
        "prompt_set_artifact_sha256": prompt_set_artifact_sha256,
        "aggregation": "mean_over_synchronized_prompts",
        "feature_definition": _baseline_feature_definition(repr_key),
        "prompt_count": prompt_count,
        "penultimate_feature": _vector_values(mean_feature),
        "mean_logits": _vector_values(mean_logits),
        "prediction_id": prediction_id,
        "prediction_label": "Conflict" if prediction_id == 1 else "Aligned",
    }


def _stream_frozen_exports(
    *,
    samples: list[_Sample],
    model: nn.Module,
    config: TrainingConfig,
    manifest_path: Path,
    bundle_manifest_path: Path,
    encoder_checkpoint_sha256: str,
) -> int:
    from mprisk.representation.data import _batches, _load_trajectory_batch

    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    bundle_tmp = bundle_manifest_path.with_suffix(bundle_manifest_path.suffix + ".tmp")
    current_bundle: dict[str, Any] | None = None
    sample_count = 0
    with (
        manifest_tmp.open("w", encoding="utf-8") as manifest_handle,
        bundle_tmp.open("w", encoding="utf-8") as bundle_handle,
        torch.no_grad(),
    ):
        for batch in _batches(samples, config.batch_size):
            trajectories, _labels = _load_trajectory_batch(
                batch, device=next(model.parameters()).device
            )
            condition_z, relation_r = model(
                trajectories,
                sample_ids=[sample.sample_id for sample in batch],
            )
            for index, sample in enumerate(batch):
                row = _frozen_row(
                    sample,
                    model_key=config.model_key,
                    repr_key=config.repr_key,
                    condition_z=condition_z[index],
                    relation_r=relation_r[index],
                    encoder_checkpoint_sha256=encoder_checkpoint_sha256,
                    prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
                )
                manifest_handle.write(json.dumps(row, sort_keys=True) + "\n")
                if current_bundle is None or current_bundle["sample_id"] != sample.sample_id:
                    if current_bundle is not None:
                        _finalize_frozen_bundle(current_bundle)
                        bundle_handle.write(json.dumps(current_bundle, sort_keys=True) + "\n")
                    current_bundle = _empty_frozen_bundle(row)
                    sample_count += 1
                _append_frozen_row(current_bundle, row)
        if current_bundle is not None:
            _finalize_frozen_bundle(current_bundle)
            bundle_handle.write(json.dumps(current_bundle, sort_keys=True) + "\n")
        for handle in (manifest_handle, bundle_handle):
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(manifest_tmp, manifest_path)
    os.replace(bundle_tmp, bundle_manifest_path)
    return sample_count


def _frozen_row(
    sample: _Sample,
    *,
    model_key: str,
    repr_key: str,
    condition_z: torch.Tensor,
    relation_r: torch.Tensor,
    encoder_checkpoint_sha256: str,
    prompt_set_artifact_sha256: str,
) -> dict[str, Any]:
    from mprisk.representation.data import _vector_values

    return {
        "schema": "mprisk_frozen_spherical_representation_v1",
        "row_id": sample.row_id,
        "sample_id": sample.sample_id,
        "sample_type": sample.sample_type,
        "label_id": sample.label_id,
        "model_key": model_key,
        "protocol": sample.protocol,
        "prompt_set_key": sample.prompt_set_key,
        "calibration_split": sample.calibration_split,
        "master_split": sample.master_split,
        "representation_split": sample.representation_split,
        "split_group_id": sample.split_group_id,
        "split_assignment_key": sample.split_assignment_key,
        "split_assignment_sha256": sample.split_assignment_sha256,
        "prompt_id": sample.prompt_id,
        "repr_key": repr_key,
        "encoder_checkpoint_sha256": encoder_checkpoint_sha256,
        "prompt_set_artifact_sha256": prompt_set_artifact_sha256,
        "condition_z": {
            condition: _vector_values(condition_z[index])
            for index, condition in enumerate(CONDITIONS)
        },
        "relation_r": _vector_values(relation_r),
    }


def _empty_frozen_bundle(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "sample_id",
            "sample_type",
            "label_id",
            "model_key",
            "protocol",
            "prompt_set_key",
            "calibration_split",
            "master_split",
            "representation_split",
            "split_group_id",
            "split_assignment_key",
            "split_assignment_sha256",
            "repr_key",
            "encoder_checkpoint_sha256",
            "prompt_set_artifact_sha256",
        )
    } | {
        "embeddings": {condition: {} for condition in CONDITIONS},
        "relations": {},
    }


def _append_frozen_row(bundle: dict[str, Any], row: dict[str, Any]) -> None:
    prompt_id = str(row["prompt_id"])
    for condition in CONDITIONS:
        bundle["embeddings"][condition][prompt_id] = row["condition_z"][condition]
    bundle["relations"][prompt_id] = row["relation_r"]


def _finalize_frozen_bundle(bundle: dict[str, Any]) -> None:
    from mprisk.representation.data import _vector_values

    relations = bundle["relations"]
    if not relations:
        raise ValueError("frozen TME sample aggregate is empty")
    relation_rows = torch.tensor(list(relations.values()), dtype=torch.float32)
    mean_relation = relation_rows.mean(dim=0, keepdim=True)
    normalized = strict_l2_normalize(
        mean_relation,
        stage="tme_sample_relation_aggregate",
        sample_ids=[str(bundle["sample_id"])],
    )[0]
    bundle["sample_relation_feature"] = _vector_values(normalized)
    bundle["prompt_count"] = len(relations)
    bundle["aggregation"] = "mean_over_synchronized_prompts_then_l2"
    bundle["feature_definition"] = "unit_normalized_mean_prompt_ordered_relation_r"
