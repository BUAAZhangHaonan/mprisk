"""Build prompt-conditioned state bundle manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from mprisk.cache.prompt_conditioned_cache import (
    PromptConditionedManifest,
    load_prompt_conditioned_manifest,
)
from mprisk.cache.prompt_cache import load_prompt_cache_manifest
from mprisk.data.protocol_views import VIEW_KEYS, normalize_protocol
from mprisk.data.state_dataset import read_state_dataset_manifest
from mprisk.prompts.template_bank import EquivPromptSet, PromptTemplate, load_equiv_prompt_set
from mprisk.utils.io import write_json, write_jsonl


DEFAULT_PROMPT_SET_DIR = Path("configs/prompts/equiv_sets")
DEFAULT_OUTPUT_ROOT = Path("outputs/state_bundles")


@dataclass(frozen=True)
class StateBundleBuildResult:
    manifest_path: Path
    summary_path: Path
    missing_path: Path
    total_count: int
    complete_count: int
    missing_count: int
    prompt_count: int


def build_state_bundles(
    *,
    state_dataset_manifest_path: str | Path,
    prompt_cache_manifest_path: str | Path,
    prompt_conditioned_cache_manifest_path: str | Path,
    model_key: str,
    protocol: str,
    prompt_set_path: str | Path | None = None,
    prompt_set_key: str | None = None,
    prompt_set_dir: str | Path = DEFAULT_PROMPT_SET_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> StateBundleBuildResult:
    normalized_protocol = normalize_protocol(protocol)
    prompt_set = _load_prompt_set(
        prompt_set_path=prompt_set_path,
        prompt_set_key=prompt_set_key,
        prompt_set_dir=prompt_set_dir,
    )
    _require_prompt_set_matches(prompt_set, normalized_protocol)
    templates = prompt_set.enabled_templates()
    prompt_ids = [template.prompt_id for template in templates]
    prompt_cache = load_prompt_cache_manifest(prompt_cache_manifest_path)
    prompt_conditioned_cache = load_prompt_conditioned_manifest(
        prompt_conditioned_cache_manifest_path
    )
    state_rows = _read_validated_state_dataset_manifest(state_dataset_manifest_path)

    bundle_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for row in state_rows:
        row_protocol = normalize_protocol(str(row["protocol"]))
        if str(row.get("model_key")) != model_key or row_protocol != normalized_protocol:
            continue
        prompt_cache_rows = prompt_cache.rows_for_prompt_ids(
            model_key=model_key,
            prompt_set_key=prompt_set.key,
            prompt_ids=prompt_ids,
            protocol=normalized_protocol,
        )
        missing_prompt_ids = [
            prompt_id for prompt_id in prompt_ids if prompt_id not in prompt_cache_rows
        ]
        prompt_conditioned_rows, missing_prompted = _prompt_conditioned_rows_for_sample(
            prompt_conditioned_cache=prompt_conditioned_cache,
            row=row,
            model_key=model_key,
            protocol=normalized_protocol,
            prompt_set_key=prompt_set.key,
            prompt_ids=prompt_ids,
        )
        if missing_prompt_ids or missing_prompted:
            missing_rows.append(
                _missing_bundle_row(
                    row=row,
                    model_key=model_key,
                    protocol=normalized_protocol,
                    prompt_set_key=prompt_set.key,
                    missing_prompt_ids=missing_prompt_ids,
                    missing_prompt_conditioned_states=missing_prompted,
                )
            )
            continue
        bundle_rows.append(
            _bundle_row(
                row=row,
                model_key=model_key,
                protocol=normalized_protocol,
                prompt_set_key=prompt_set.key,
                templates=templates,
                prompt_cache_rows=prompt_cache_rows,
                prompt_conditioned_rows=prompt_conditioned_rows,
            )
        )

    output_dir = Path(output_root) / model_key / normalized_protocol / prompt_set.key
    manifest_path = write_jsonl(output_dir / "bundle_manifest.jsonl", bundle_rows)
    missing_path = write_jsonl(output_dir / "missing_prompt_cache_rows.jsonl", missing_rows)
    summary_path = write_json(
        output_dir / "bundle_summary.json",
        {
            "model_key": model_key,
            "protocol": normalized_protocol,
            "prompt_set_key": prompt_set.key,
            "total_samples": len(bundle_rows) + len(missing_rows),
            "complete_samples": len(bundle_rows),
            "missing_samples": len(missing_rows),
            "prompt_count": len(prompt_ids),
            "bundle_manifest": str(manifest_path),
            "missing_prompt_cache_rows": str(missing_path),
        },
    )
    return StateBundleBuildResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        missing_path=missing_path,
        total_count=len(bundle_rows) + len(missing_rows),
        complete_count=len(bundle_rows),
        missing_count=len(missing_rows),
        prompt_count=len(prompt_ids),
    )


def iter_state_bundles(bundle_manifest_path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(bundle_manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_state_bundle(sample_id: str, bundle_manifest_path: str | Path) -> dict[str, Any]:
    for bundle in iter_state_bundles(bundle_manifest_path):
        if bundle.get("sample_id") == sample_id:
            return bundle
    raise KeyError(f"State bundle not found for sample_id {sample_id!r}")


def _read_validated_state_dataset_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = read_state_dataset_manifest(path)
    for index, row in enumerate(rows, start=1):
        try:
            _validate_state_dataset_row(row)
        except ValueError as exc:
            raise ValueError(f"{path}:{index}: {exc}") from exc
    return rows


def _validate_state_dataset_row(row: dict[str, Any]) -> None:
    for field in ("sample_id", "sample_type", "model_key", "protocol", "trajectory_meta"):
        if field not in row:
            raise ValueError(f"state dataset row missing {field}")
    for view_key, entry_field in _ENTRY_FIELDS.items():
        entry = row.get(entry_field)
        if not isinstance(entry, dict):
            raise ValueError(f"state dataset row missing {entry_field}")
        if entry.get("condition") != view_key:
            raise ValueError(f"{entry_field}.condition must be {view_key}")


def _load_prompt_set(
    *,
    prompt_set_path: str | Path | None,
    prompt_set_key: str | None,
    prompt_set_dir: str | Path,
) -> EquivPromptSet:
    if prompt_set_path is not None:
        prompt_set = load_equiv_prompt_set(prompt_set_path)
        if prompt_set_key is not None and prompt_set.key != prompt_set_key:
            raise ValueError(
                f"prompt_set_key {prompt_set_key!r} does not match prompt set {prompt_set.key!r}"
            )
        return prompt_set
    if prompt_set_key is None:
        raise ValueError("Either prompt_set_path or prompt_set_key is required")
    return load_equiv_prompt_set(Path(prompt_set_dir) / f"{prompt_set_key}.yaml")


def _require_prompt_set_matches(prompt_set: EquivPromptSet, protocol: str) -> None:
    if normalize_protocol(prompt_set.protocol) != protocol:
        raise ValueError(
            f"Prompt set {prompt_set.key} protocol {prompt_set.protocol!r} "
            f"does not match {protocol!r}"
        )


def _bundle_row(
    *,
    row: dict[str, Any],
    model_key: str,
    protocol: str,
    prompt_set_key: str,
    templates: Iterable[PromptTemplate],
    prompt_cache_rows: dict[str, dict[str, Any]],
    prompt_conditioned_rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    prompts = [
        _prompt_ref(template=template, prompt_cache_row=prompt_cache_rows[template.prompt_id])
        for template in templates
    ]
    return {
        "sample_id": row["sample_id"],
        "sample_type": row["sample_type"],
        "model_key": model_key,
        "protocol": protocol,
        "prompt_set_key": prompt_set_key,
        "prompts": prompts,
        "views": {
            view_key: _view_bundle(
                row=row,
                view_key=view_key,
                prompts=prompts,
                prompt_cache_rows=prompt_cache_rows,
                prompt_conditioned_rows=prompt_conditioned_rows,
            )
            for view_key in VIEW_KEYS
        },
        "trajectory_meta": dict(row["trajectory_meta"]),
        "metadata": _metadata(row),
    }


def _prompt_ref(
    *,
    template: PromptTemplate,
    prompt_cache_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "prompt_id": template.prompt_id,
        "role": template.role,
        "template_text": template.template_text,
        "prompt_cache": dict(prompt_cache_row),
    }


def _view_bundle(
    *,
    row: dict[str, Any],
    view_key: str,
    prompts: Iterable[dict[str, Any]],
    prompt_cache_rows: dict[str, dict[str, Any]],
    prompt_conditioned_rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    state_cache = dict(row[_ENTRY_FIELDS[view_key]])
    return {
        "state_cache": state_cache,
        "trajectory_meta": dict(row["trajectory_meta"]),
        "prompts": {
            prompt["prompt_id"]: {
                "prompt_id": prompt["prompt_id"],
                "state_cache_condition": view_key,
                "prompt_cache": dict(prompt_cache_rows[prompt["prompt_id"]]),
                "prompt_conditioned_state": dict(
                    prompt_conditioned_rows[view_key][prompt["prompt_id"]]
                ),
            }
            for prompt in prompts
        },
    }


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field in ("source_dataset", "target_label", "dominant_modality"):
        if field in row:
            metadata[field] = row[field]
    return metadata


def _prompt_conditioned_rows_for_sample(
    *,
    prompt_conditioned_cache: PromptConditionedManifest,
    row: dict[str, Any],
    model_key: str,
    protocol: str,
    prompt_set_key: str,
    prompt_ids: list[str],
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    rows: dict[str, dict[str, dict[str, Any]]] = {view_key: {} for view_key in VIEW_KEYS}
    missing: list[str] = []
    for view_key in VIEW_KEYS:
        for prompt_id in prompt_ids:
            entry = prompt_conditioned_cache.lookup(
                sample_id=str(row["sample_id"]),
                model_key=model_key,
                protocol=protocol,
                condition=view_key,
                prompt_set_key=prompt_set_key,
                prompt_id=prompt_id,
            )
            if entry is None:
                missing.append(f"{view_key}:{prompt_id}")
            else:
                rows[view_key][prompt_id] = entry.to_manifest_row()
    return rows, missing


def _missing_bundle_row(
    *,
    row: dict[str, Any],
    model_key: str,
    protocol: str,
    prompt_set_key: str,
    missing_prompt_ids: list[str],
    missing_prompt_conditioned_states: list[str],
) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "sample_type": row["sample_type"],
        "model_key": model_key,
        "protocol": protocol,
        "prompt_set_key": prompt_set_key,
        "missing_prompt_ids": missing_prompt_ids,
        "missing_prompt_conditioned_states": missing_prompt_conditioned_states,
    }


_ENTRY_FIELDS = {
    "M1": "m1_entry",
    "M2": "m2_entry",
    "M12": "m12_entry",
}
