"""Strict loading and validation for model asset metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from mprisk.config.loader import load_yaml

MODEL_ASSET_SCHEMA = "mprisk_model_assets_v2"
PANEL_GROUPS = frozenset({"vt", "va_vta"})
PROTOCOLS = frozenset({"vt", "va", "it"})
INPUT_MODALITIES = frozenset({"text", "image", "video", "audio"})
VIDEO_MODES = frozenset(
    {
        "multi_image_simulation",
        "native_video",
        "native_video_or_multi_image",
        "multi_image_or_video_frames",
        "extracted_frames",
    }
)
ASSET_STATUSES = frozenset({"available", "planned", "unavailable"})

_TOP_LEVEL_FIELDS = frozenset({"schema", "model_root", "models"})
_MODEL_FIELDS = frozenset(
    {
        "key",
        "display_name",
        "family",
        "hf_model_id",
        "local_path",
        "parameter_scale",
        "panel_group",
        "protocols",
        "input_modalities",
        "video_mode",
        "max_video_frames",
        "thinking",
        "policy",
        "status",
    }
)
_THINKING_FIELDS = frozenset({"supported", "enabled", "disable_argument"})
_POLICY_FIELDS = frozenset({"allow_thinking"})


@dataclass(frozen=True)
class ThinkingConfig:
    supported: bool
    enabled: bool
    disable_argument: str | None


@dataclass(frozen=True)
class PolicyConfig:
    allow_thinking: bool


@dataclass(frozen=True)
class ModelAsset:
    key: str
    display_name: str
    family: str
    hf_model_id: str
    local_path: str
    parameter_scale: str
    panel_group: str
    protocols: tuple[str, ...]
    input_modalities: tuple[str, ...]
    video_mode: str
    max_video_frames: int | None
    thinking: ThinkingConfig
    policy: PolicyConfig
    status: str

    @property
    def source(self) -> str:
        """Compatibility alias for callers that previously used ``source``."""
        return self.hf_model_id

    @property
    def local_model_path(self) -> Path:
        return Path(self.local_path)


def load_model_assets(
    path: str | Path,
    *,
    require_local_paths: bool = False,
) -> list[ModelAsset]:
    """Load a versioned model panel and reject incomplete or ambiguous metadata."""
    data = load_yaml(path)
    _require_exact_keys(data, _TOP_LEVEL_FIELDS, context="model asset registry")
    if data["schema"] != MODEL_ASSET_SCHEMA:
        raise ValueError(
            f"Unsupported model asset schema {data['schema']!r}; expected {MODEL_ASSET_SCHEMA!r}"
        )

    model_root = _required_string(data["model_root"], context="model_root")
    root = PurePosixPath(model_root)
    if not root.is_absolute():
        raise ValueError("model_root must be an absolute POSIX path")

    raw_models = data["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a non-empty list")

    assets = [
        _parse_model(item, index=index, model_root=root)
        for index, item in enumerate(raw_models)
    ]
    index_assets(assets)
    if require_local_paths:
        verify_local_model_paths(assets)
    return assets


def index_assets(assets: Iterable[ModelAsset]) -> dict[str, ModelAsset]:
    index: dict[str, ModelAsset] = {}
    for asset in assets:
        if asset.key in index:
            raise ValueError(f"Duplicate model asset key: {asset.key!r}")
        index[asset.key] = asset
    return index


def verify_local_model_paths(assets: Iterable[ModelAsset]) -> None:
    """Require every configured local checkpoint directory and config.json to exist."""
    failures: list[str] = []
    for asset in assets:
        model_path = asset.local_model_path
        if not model_path.is_dir():
            failures.append(f"{asset.key}: missing directory {model_path}")
        elif not (model_path / "config.json").is_file():
            failures.append(f"{asset.key}: missing config.json in {model_path}")
    if failures:
        raise FileNotFoundError("Invalid local model assets: " + "; ".join(failures))


def validate_model_reference_config(
    assets: Iterable[ModelAsset],
    config_path: str | Path,
) -> tuple[str, ...]:
    """Validate model references and protocol compatibility in one config file."""
    data = load_yaml(config_path)
    reference_fields = [field for field in ("primary_models", "models") if field in data]
    if len(reference_fields) != 1:
        raise ValueError(
            f"{config_path} must contain exactly one model reference field: "
            "primary_models or models"
        )
    field = reference_fields[0]
    model_keys = _string_list(data[field], context=f"{config_path}:{field}")
    if not model_keys:
        raise ValueError(f"{config_path}:{field} must not be empty")

    asset_index = index_assets(assets)
    unknown = [key for key in model_keys if key not in asset_index]
    if unknown:
        raise ValueError(f"{config_path} references unknown model keys: {', '.join(unknown)}")

    protocol_value = data.get("protocol")
    if protocol_value is None and str(data.get("key", "")).lower() in PROTOCOLS:
        protocol_value = data["key"]
    if protocol_value is not None:
        protocol = _required_string(protocol_value, context=f"{config_path}:protocol").lower()
        if protocol not in PROTOCOLS:
            raise ValueError(f"{config_path} has unsupported protocol {protocol!r}")
        incompatible = [key for key in model_keys if protocol not in asset_index[key].protocols]
        if incompatible:
            raise ValueError(
                f"{config_path} assigns protocol {protocol!r} to incompatible models: "
                + ", ".join(incompatible)
            )
    return tuple(model_keys)


def validate_model_panel_references(
    assets: Iterable[ModelAsset],
    config_paths: Iterable[str | Path],
) -> dict[Path, tuple[str, ...]]:
    asset_list = list(assets)
    return {
        Path(path): validate_model_reference_config(asset_list, path)
        for path in config_paths
    }


def assets_to_rows(assets: Iterable[ModelAsset]) -> list[dict[str, Any]]:
    return [
        {
            "model_key": asset.key,
            "display_name": asset.display_name,
            "family": asset.family,
            "hf_model_id": asset.hf_model_id,
            "local_path": asset.local_path,
            "parameter_scale": asset.parameter_scale,
            "panel_group": asset.panel_group,
            "protocols": ",".join(asset.protocols),
            "input_modalities": ",".join(asset.input_modalities),
            "video_mode": asset.video_mode,
            "max_video_frames": asset.max_video_frames,
            "thinking_supported": asset.thinking.supported,
            "thinking_enabled": asset.thinking.enabled,
            "allow_thinking": asset.policy.allow_thinking,
            "status": asset.status,
        }
        for asset in assets
    ]


def _parse_model(
    item: object,
    *,
    index: int,
    model_root: PurePosixPath,
) -> ModelAsset:
    context = f"models[{index}]"
    if not isinstance(item, Mapping):
        raise TypeError(f"{context} must be a mapping")
    _require_exact_keys(item, _MODEL_FIELDS, context=context)

    key = _required_string(item["key"], context=f"{context}.key")
    panel_group = _enum_string(
        item["panel_group"], PANEL_GROUPS, context=f"{context}.panel_group"
    )
    protocols = _string_list(item["protocols"], context=f"{context}.protocols")
    unsupported_protocols = sorted(set(protocols) - PROTOCOLS)
    if unsupported_protocols:
        raise ValueError(
            f"{context}.protocols contains unsupported values: {unsupported_protocols}"
        )
    expected_protocols = ("vt",) if panel_group == "vt" else ("va",)
    if protocols != expected_protocols:
        raise ValueError(
            f"{context}.protocols must be {expected_protocols!r} for panel_group {panel_group!r}"
        )

    modalities = _string_list(item["input_modalities"], context=f"{context}.input_modalities")
    unsupported_modalities = sorted(set(modalities) - INPUT_MODALITIES)
    if unsupported_modalities:
        raise ValueError(
            f"{context}.input_modalities contains unsupported values: {unsupported_modalities}"
        )
    if "text" not in modalities or "image" not in modalities:
        raise ValueError(f"{context}.input_modalities must contain text and image")
    if panel_group == "va_vta" and "audio" not in modalities:
        raise ValueError(f"{context}.input_modalities must contain audio for va_vta models")

    video_mode = _enum_string(item["video_mode"], VIDEO_MODES, context=f"{context}.video_mode")
    if video_mode in {"native_video", "native_video_or_multi_image"} and "video" not in modalities:
        raise ValueError(f"{context} declares native video but omits video input modality")
    if video_mode not in {"native_video", "native_video_or_multi_image"} and "video" in modalities:
        raise ValueError(f"{context} declares non-native video but includes video input modality")

    max_video_frames = item["max_video_frames"]
    if max_video_frames is not None:
        if isinstance(max_video_frames, bool) or not isinstance(max_video_frames, int):
            raise TypeError(f"{context}.max_video_frames must be an integer or null")
        if max_video_frames <= 0:
            raise ValueError(f"{context}.max_video_frames must be positive")
        if video_mode != "extracted_frames":
            raise ValueError(f"{context}.max_video_frames is only valid for extracted_frames")

    thinking = _parse_thinking(item["thinking"], context=f"{context}.thinking")
    policy = _parse_policy(item["policy"], context=f"{context}.policy")

    local_path = _required_string(item["local_path"], context=f"{context}.local_path")
    local = PurePosixPath(local_path)
    if not local.is_absolute():
        raise ValueError(f"{context}.local_path must be an absolute POSIX path")
    try:
        local.relative_to(model_root)
    except ValueError as error:
        raise ValueError(f"{context}.local_path must stay under model_root {model_root}") from error

    return ModelAsset(
        key=key,
        display_name=_required_string(item["display_name"], context=f"{context}.display_name"),
        family=_required_string(item["family"], context=f"{context}.family"),
        hf_model_id=_required_string(item["hf_model_id"], context=f"{context}.hf_model_id"),
        local_path=local_path,
        parameter_scale=_required_string(
            item["parameter_scale"], context=f"{context}.parameter_scale"
        ),
        panel_group=panel_group,
        protocols=protocols,
        input_modalities=modalities,
        video_mode=video_mode,
        max_video_frames=max_video_frames,
        thinking=thinking,
        policy=policy,
        status=_enum_string(item["status"], ASSET_STATUSES, context=f"{context}.status"),
    )


def _parse_thinking(value: object, *, context: str) -> ThinkingConfig:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    _require_exact_keys(value, _THINKING_FIELDS, context=context)
    supported = _required_bool(value["supported"], context=f"{context}.supported")
    enabled = _required_bool(value["enabled"], context=f"{context}.enabled")
    if enabled:
        raise ValueError(f"{context}.enabled must be false for this experiment panel")
    disable_argument = value["disable_argument"]
    if disable_argument is not None:
        disable_argument = _required_string(
            disable_argument, context=f"{context}.disable_argument"
        )
    if supported and disable_argument is None:
        raise ValueError(f"{context}.disable_argument is required when thinking is supported")
    if not supported and disable_argument is not None:
        raise ValueError(f"{context}.disable_argument must be null when thinking is unsupported")
    return ThinkingConfig(
        supported=supported,
        enabled=enabled,
        disable_argument=disable_argument,
    )


def _parse_policy(value: object, *, context: str) -> PolicyConfig:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    _require_exact_keys(value, _POLICY_FIELDS, context=context)
    allow_thinking = _required_bool(
        value["allow_thinking"], context=f"{context}.allow_thinking"
    )
    if allow_thinking:
        raise ValueError(f"{context}.allow_thinking must be false for this experiment panel")
    return PolicyConfig(allow_thinking=allow_thinking)


def _require_exact_keys(
    value: Mapping[object, object],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    actual = {str(key) for key in value}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{context} fields mismatch: missing={missing!r} extra={extra!r}")


def _required_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return value.strip()


def _required_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a boolean")
    return value


def _enum_string(value: object, allowed: frozenset[str], *, context: str) -> str:
    resolved = _required_string(value, context=context)
    if resolved not in allowed:
        raise ValueError(f"{context} must be one of {sorted(allowed)!r}; got {resolved!r}")
    return resolved


def _string_list(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    resolved = tuple(_required_string(item, context=f"{context}[]") for item in value)
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"{context} must not contain duplicates")
    return resolved
