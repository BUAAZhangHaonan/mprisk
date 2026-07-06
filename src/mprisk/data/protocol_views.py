"""Protocol view definitions for M1, M2, and M12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProtocolView:
    protocol: str
    m1: str
    m2: str
    m12: str


_CANONICAL_PROTOCOLS = {
    "VT": ProtocolView(protocol="VT", m1="vision", m2="text", m12="vision_text"),
    "VA": ProtocolView(protocol="VA", m1="vision", m2="audio", m12="vision_audio"),
    "IT": ProtocolView(protocol="IT", m1="image", m2="text", m12="image_text"),
}

PROTOCOLS = {
    "vt": ProtocolView(protocol="vt", m1="vision", m2="text", m12="vision_text"),
    "va": ProtocolView(protocol="va", m1="vision", m2="audio", m12="vision_audio"),
    "it": ProtocolView(protocol="it", m1="image", m2="text", m12="image_text"),
}
VIEW_KEYS = ("M1", "M2", "M12")


def normalize_protocol(protocol: str) -> str:
    normalized = str(protocol).strip().upper()
    if normalized not in _CANONICAL_PROTOCOLS:
        allowed = ", ".join(sorted(_CANONICAL_PROTOCOLS))
        raise ValueError(f"Unsupported protocol {protocol!r}; expected one of {allowed}")
    return normalized


def protocol_view(protocol: str) -> ProtocolView:
    return _CANONICAL_PROTOCOLS[normalize_protocol(protocol)]


def _view_media_paths(
    view_key: str,
    view: ProtocolView,
    media_paths: Mapping[str, str],
) -> dict[str, str]:
    if view_key in {"M1", "M2"}:
        modality = view.m1 if view_key == "M1" else view.m2
        return {
            key: media_paths[key]
            for key in (view_key, modality)
            if key in media_paths and media_paths[key]
        }

    keys = ("M12", view.m1, view.m2)
    return {key: media_paths[key] for key in keys if key in media_paths and media_paths[key]}


def expand_protocol_views(
    protocol: str,
    *,
    views: Mapping[str, Mapping[str, Any]] | None = None,
    media_paths: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    canonical_protocol = normalize_protocol(protocol)
    view = protocol_view(canonical_protocol)
    source_views = views or {key: {} for key in VIEW_KEYS}
    missing = [key for key in VIEW_KEYS if key not in source_views]
    if missing:
        raise ValueError(f"Missing required manifest view(s): {', '.join(missing)}")

    canonical_modalities = {"M1": view.m1, "M2": view.m2, "M12": view.m12}
    media = dict(media_paths or {})
    expanded: dict[str, dict[str, Any]] = {}
    for view_key in VIEW_KEYS:
        payload = source_views[view_key]
        if not isinstance(payload, Mapping):
            raise TypeError(f"views.{view_key} must be a mapping")
        expanded_payload = dict(payload)
        expanded_payload.setdefault("modality", canonical_modalities[view_key])
        expanded_payload["view"] = view_key
        expanded_payload["protocol"] = canonical_protocol
        resolved_media_paths = _view_media_paths(view_key, view, media)
        if resolved_media_paths:
            expanded_payload["media_paths"] = resolved_media_paths
        expanded[view_key] = expanded_payload
    return expanded
