"""GT Description generation plan construction and config loading."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from mprisk.config.loader import load_yaml
from mprisk.data.generated_archive_freeze import _canonical_json, _sha256
from mprisk.ground_truth.annotation_inputs import GTAnnotationInput
from mprisk.ground_truth.providers.registry import (
    validate_provider_settings,
)
from mprisk.utils.io import read_jsonl as _read_jsonl

from mprisk.ground_truth._plan import (
    CONFIG_SCHEMA,
    GTDescriptionGenerationConfig,
    GTDescriptionGenerationTask,
)


def load_config(path: str | Path) -> GTDescriptionGenerationConfig:
    payload = load_yaml(path)
    if payload.get("schema_name") != CONFIG_SCHEMA:
        raise ValueError(
            f"Unsupported GT Description generation config: {payload.get('schema_name')!r}"
        )
    config = GTDescriptionGenerationConfig.model_validate(payload)
    validate_provider_settings(config.provider_key, config.provider_settings)
    return config


def _resolve_repo_path(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"GT Description path escapes repository: {resolved}")
    return resolved


def _validate_model_input(payload: dict[str, Any]) -> None:
    if set(payload) != {
        "archetype",
        "dialogue",
        "scenario_context",
        "surface_emotion",
    }:
        raise ValueError("GT generator model input contains forbidden fields")
    if set(payload["archetype"]) != {"id", "name", "canonical_meaning"}:
        raise ValueError("Archetype model input contains forbidden fields")
    for key in ("dialogue", "scenario_context"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if payload["surface_emotion"] is not None and not isinstance(
        payload["surface_emotion"], str
    ):
        raise TypeError("surface_emotion must be a string or null")


def prepare_tasks(
    repo_root: str | Path,
    config: GTDescriptionGenerationConfig,
) -> list[GTDescriptionGenerationTask]:
    root = Path(repo_root).resolve()
    manifest_path = _resolve_repo_path(root, config.input_manifest)
    if _sha256(manifest_path) != config.input_manifest_sha256:
        raise ValueError("GT annotation input manifest hash mismatch")
    rows = _read_jsonl(manifest_path)
    if len(rows) != config.expected_count:
        raise ValueError(
            f"GT annotation input count mismatch: expected {config.expected_count}, "
            f"got {len(rows)}"
        )
    prompts = {
        "Conflict": _resolve_repo_path(root, config.conflict_prompt_path).read_text(
            encoding="utf-8"
        ),
        "Aligned": _resolve_repo_path(root, config.aligned_prompt_path).read_text(
            encoding="utf-8"
        ),
    }
    ledger_signature = {
        "schema_name": config.schema_name,
        "run_id": config.run_id,
        "provider_key": config.provider_key,
        "gt_generator_model": config.gt_generator_model,
        "provider_settings_sha256": hashlib.sha256(
            _canonical_json(config.provider_settings).encode()
        ).hexdigest(),
        "gt_input_schema_version": config.gt_input_schema_version,
        "input_manifest_sha256": config.input_manifest_sha256,
        "expected_count": config.expected_count,
    }
    tasks: list[GTDescriptionGenerationTask] = []
    seen_ids: set[str] = set()
    for order, raw_row in enumerate(rows):
        annotation_input = GTAnnotationInput.model_validate(raw_row)
        row = annotation_input.model_dump(mode="json")
        sample_id = annotation_input.sample_id
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate GT annotation sample_id: {sample_id}")
        seen_ids.add(sample_id)
        sample_type = annotation_input.sample_type
        model_input = {
            "archetype": annotation_input.archetype.model_dump(mode="json"),
            "dialogue": annotation_input.dialogue,
            "scenario_context": annotation_input.scenario_context,
            "surface_emotion": annotation_input.surface_emotion,
        }
        _validate_model_input(model_input)
        prompt = prompts[sample_type]
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        input_hash = hashlib.sha256(
            _canonical_json(
                {
                    "gt_generator_model": config.gt_generator_model,
                    "prompt_hash": prompt_hash,
                    "model_input": model_input,
                    "ledger_signature": ledger_signature,
                }
            ).encode()
        ).hexdigest()
        tasks.append(
            GTDescriptionGenerationTask(
                order=order,
                sample_id=sample_id,
                sample_type=sample_type,
                source_archive=annotation_input.source_provenance.source_archive,
                input_hash=input_hash,
                prompt_hash=prompt_hash,
                system_prompt=prompt,
                model_input=model_input,
                annotation_input_row=row,
                ledger_signature=dict(ledger_signature),
            )
        )
    return tasks
