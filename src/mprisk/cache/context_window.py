"""Checkpoint-bound context-window validation for prefill smoke caches."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FAMILY_CONTEXT_CEILING_PATHS: dict[str, tuple[str, ...]] = {
    "gemma3": ("text_config", "max_position_embeddings"),
    "glm4v": ("text_config", "max_position_embeddings"),
    "internvl": ("llm_config", "max_position_embeddings"),
    "llava_v15": ("text_config", "max_position_embeddings"),
    "llava_onevision": ("text_config", "max_position_embeddings"),
    "minicpm_v": ("max_position_embeddings",),
    "phi3_vision": ("max_position_embeddings",),
    "qwen2_5_vl": ("text_config", "max_position_embeddings"),
    "qwen_vl": ("text_config", "max_position_embeddings"),
    "qwen3_5": ("text_config", "max_position_embeddings"),
    "gemma4": ("text_config", "max_position_embeddings"),
    "phi4_multimodal": ("max_position_embeddings",),
    "qwen_omni": (
        "thinker_config",
        "text_config",
        "max_position_embeddings",
    ),
}

_AUTO_CONFIG_CONTEXT_SCRIPT = r"""
import json
import sys

from transformers import AutoConfig

model_path = sys.argv[1]
attribute_path = json.loads(sys.argv[2])
config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)
value = config
for attribute in attribute_path:
    if not hasattr(value, attribute):
        raise AttributeError(
            f"{type(value).__name__} has no context attribute {attribute!r}"
        )
    value = getattr(value, attribute)
if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise TypeError(f"Invalid max_position_embeddings: {value!r}")
print(json.dumps({"max_position_embeddings": value}, sort_keys=True))
"""


def context_ceiling_path(family: str) -> tuple[str, ...]:
    try:
        return FAMILY_CONTEXT_CEILING_PATHS[family]
    except KeyError as exc:
        raise ValueError(
            f"No context ceiling path is registered for family {family!r}"
        ) from exc


def load_context_ceiling(
    *,
    family: str,
    python: str | Path,
    model_path: str | Path,
    expected_model_config_sha256: str,
    environment: Mapping[str, str],
    cwd: str | Path,
) -> int:
    checkpoint = Path(model_path).expanduser().resolve()
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    actual_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_model_config_sha256:
        raise ValueError(
            f"Model config SHA mismatch for {checkpoint}: "
            f"{actual_sha256} != {expected_model_config_sha256}"
        )
    attribute_path = context_ceiling_path(family)
    completed = subprocess.run(
        [
            str(python),
            "-c",
            _AUTO_CONFIG_CONTEXT_SCRIPT,
            str(checkpoint),
            json.dumps(attribute_path),
        ],
        cwd=Path(cwd),
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Failed to read context ceiling for {family}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Context ceiling subprocess returned invalid JSON: {completed.stdout!r}"
        ) from exc
    value = payload.get("max_position_embeddings")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid context ceiling for {family}: {value!r}")
    return value


def validate_sidecar_token_count(
    *,
    model_key: str,
    sidecar_path: str | Path,
    manifest_token_count: int,
    sidecar_payload: Mapping[str, Any],
    context_ceiling: int,
) -> int:
    entry = sidecar_payload.get("entry")
    if not isinstance(entry, Mapping):
        raise ValueError(f"Sidecar has no entry object: {sidecar_path}")
    sidecar_token_count = entry.get("token_count")
    if (
        isinstance(sidecar_token_count, bool)
        or not isinstance(sidecar_token_count, int)
        or sidecar_token_count <= 0
    ):
        raise ValueError(
            f"Invalid sidecar token_count for {model_key}: {sidecar_token_count!r}"
        )
    if sidecar_token_count != manifest_token_count:
        raise ValueError(
            f"Manifest/sidecar token_count mismatch for {model_key}: "
            f"{manifest_token_count} != {sidecar_token_count}"
        )
    if sidecar_token_count > context_ceiling:
        raise ValueError(
            f"Context overflow for {model_key}: sidecar token_count "
            f"{sidecar_token_count} exceeds max_position_embeddings "
            f"{context_ceiling} ({sidecar_path})"
        )
    return sidecar_token_count


def audit_smoke_cache_context(
    *,
    cache_root: str | Path,
    model_key: str,
    context_ceiling: int,
) -> int:
    root = Path(cache_root).expanduser().resolve()
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    maximum = 0
    entries = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Manifest line {line_number} is not an object")
            entries.append(value)
    if not entries:
        raise ValueError(f"Smoke manifest is empty: {manifest_path}")
    for entry in entries:
        manifest_token_count = int(entry["token_count"])
        cache_entry_root = Path(str(entry["cache_root"])).expanduser().resolve()
        sidecar = cache_entry_root / str(entry["metadata"]["sidecar_path"])
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Sidecar is not an object: {sidecar}")
        maximum = max(
            maximum,
            validate_sidecar_token_count(
                model_key=model_key,
                sidecar_path=sidecar,
                manifest_token_count=manifest_token_count,
                sidecar_payload=payload,
                context_ceiling=context_ceiling,
            ),
        )
    return maximum
