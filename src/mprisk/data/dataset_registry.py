"""Dataset registry types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    role: str
    modalities: tuple[str, ...]


@dataclass(frozen=True)
class FinalManifestSpec:
    key: str
    filename: str
    sample_type: str | None = None


FINAL_MANIFESTS: dict[str, FinalManifestSpec] = {
    "unified": FinalManifestSpec(
        key="unified",
        filename="unified_sample_manifest.jsonl",
        sample_type=None,
    ),
    "conflict": FinalManifestSpec(
        key="conflict",
        filename="conflict_manifest.jsonl",
        sample_type="Conflict",
    ),
    "aligned": FinalManifestSpec(
        key="aligned",
        filename="aligned_manifest.jsonl",
        sample_type="Aligned",
    ),
}


def final_manifest_path(
    key: str,
    root: str | Path = "data/processed/manifests",
) -> Path:
    try:
        spec = FINAL_MANIFESTS[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(FINAL_MANIFESTS))
        raise ValueError(f"Unknown final manifest {key!r}; expected one of {allowed}") from exc
    return Path(root) / spec.filename
