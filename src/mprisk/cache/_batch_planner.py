"""Manifest-driven batch plan construction for prefill extraction."""

from __future__ import annotations

import argparse
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mprisk.assets.registry import index_assets, load_model_assets
from mprisk.cache._batch_plan import (
    CONDITIONS,
    BatchPlan,
    BatchTask,
)
from mprisk.prompts.compiler import compile_prompt
from mprisk.prompts.template_bank import PromptTemplate, load_equiv_prompt_set
from mprisk.utils.io import (
    canonical_json as _canonical_json,
    hash_text as _hash_text,
    read_jsonl as _read_jsonl,
    sha256_file as _sha256,
)


def build_batch_plan(args: argparse.Namespace) -> BatchPlan:
    _resolve_runtime_asset(args)
    rows = _read_jsonl(args.manifest)
    _validate_rows(rows, args.protocol)
    prompt_set = load_equiv_prompt_set(args.prompt_set)
    if not prompt_set.active:
        raise ValueError(f"Prompt set is inactive: {prompt_set.key}")
    if prompt_set.protocol.lower() != args.protocol:
        raise ValueError("Prompt-set protocol does not match --protocol")
    templates = prompt_set.enabled_templates()
    if not templates:
        raise ValueError("Prompt set has no enabled templates")
    prompt_ids = tuple(template.prompt_id for template in templates)
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("Enabled prompt IDs must be unique")
    conditions = tuple(str(item).upper() for item in args.conditions)
    if set(conditions) != set(CONDITIONS) or len(conditions) != len(CONDITIONS):
        raise ValueError("Full extraction requires conditions M1, M2, and M12 exactly once")
    variables = _parse_variables(args.prompt_variable)
    required = set().union(*(_template_fields(template) for template in templates))
    allowed_external = required - {"sample_text"}
    extra = set(variables) - allowed_external
    if extra:
        raise ValueError(f"Unused or reserved prompt variables: {sorted(extra)}")
    unresolved = tuple(sorted(allowed_external - set(variables)))
    tasks = []
    for row in rows:
        for template in templates:
            values = {"sample_text": str(row.get("text_content", "")), **variables}
            prompt_text = None if unresolved else compile_prompt(template, values)
            for condition in conditions:
                identity = {
                    "sample_id": row["sample_id"],
                    "prompt_id": template.prompt_id,
                    "condition": condition,
                    "protocol": args.protocol,
                    "model_key": args.model_key,
                }
                task_id = _hash_text(_canonical_json(identity))
                tasks.append(
                    BatchTask(
                        task_id=task_id,
                        sample_id=str(row["sample_id"]),
                        prompt_set_key=prompt_set.key,
                        prompt_id=template.prompt_id,
                        prompt_text=prompt_text,
                        condition=condition,
                        row=row,
                    )
                )
    signature = {
        "schema": "mprisk_prefill_batch_signature_v2",
        "asset_config_sha256": _sha256(args.asset_config),
        "manifest_sha256": _sha256(args.manifest),
        "prompt_set_sha256": _sha256(args.prompt_set),
        "prompt_ids": prompt_ids,
        "prompt_variables": variables,
        "protocol": args.protocol,
        "conditions": conditions,
        "model_key": args.model_key,
        "family": args.family,
        "model_path": str(args.model_path.expanduser().resolve()),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "joint_audio_mode": args.joint_audio_mode,
        "video_fps": args.video_fps,
        "video_num_segments": args.video_num_segments,
        "internvl_max_num": args.internvl_max_num,
    }
    return BatchPlan(tasks, prompt_ids, unresolved, rows, signature)


def _validate_rows(rows: list[dict[str, Any]], protocol: str) -> None:
    if not rows:
        raise ValueError("Input manifest is empty")
    seen = set()
    required = {"sample_id", "protocol", "media_paths", "source_dataset", "split"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Manifest row is missing fields: {sorted(missing)}")
        if str(row["protocol"]).lower() != protocol:
            raise ValueError(f"Manifest contains non-{protocol} row: {row['sample_id']}")
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Manifest contains duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if not isinstance(row["media_paths"], dict):
            raise ValueError(f"Manifest row has invalid media_paths: {sample_id}")
        if str(row.get("sample_type", "")).lower() == "misread":
            raise ValueError(f"Prefill extraction must not process Misread rows: {sample_id}")


def _validate_media(rows: list[dict[str, Any]]) -> None:
    missing = sorted(
        {
            str(path)
            for row in rows
            for path in row["media_paths"].values()
            if not Path(str(path)).is_file()
        }
    )
    if missing:
        raise FileNotFoundError(f"Manifest references missing media files: {missing[:10]}")


def _resolve_runtime_asset(args: argparse.Namespace) -> None:
    assets = index_assets(load_model_assets(args.asset_config))
    asset = assets.get(args.model_key)
    if asset is None:
        raise KeyError(f"Model key is absent from asset config: {args.model_key}")
    if args.family is not None and args.family != asset.family:
        raise ValueError(
            f"Configured family for {args.model_key} is {asset.family!r}, not {args.family!r}"
        )
    args.family = asset.family
    if args.model_path is None:
        args.model_path = asset.local_model_path
    if args.attn_implementation is None:
        args.attn_implementation = "eager" if asset.family == "internvl" else "sdpa"


def _parse_variables(items: Sequence[str]) -> dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Prompt variable must be NAME=VALUE: {item!r}")
        key, value = item.split("=", 1)
        if not key or key in result:
            raise ValueError(f"Invalid or duplicate prompt variable: {key!r}")
        result[key] = value
    return result


def _template_fields(template: PromptTemplate) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(template.template_text) if field}
