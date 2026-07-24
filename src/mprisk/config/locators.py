"""Strict resolution of portable protocol locators.

Tracked protocol files describe resources with logical locators.  A separate,
untracked machine overlay maps each locator key to one absolute local path.
Resolution is deliberately explicit and fail-closed: there is no environment
lookup, directory search, legacy path handling, or implicit repository root.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

import yaml

RESOLVABLE_SCHEMES = frozenset({"repo", "model", "env", "artifact", "external"})
IDENTITY_SCHEMES = frozenset({"archive"})
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class LocatorError(ValueError):
    """Raised when a locator or machine overlay violates the protocol."""


@dataclass(frozen=True)
class ResolvedLocator:
    """Resolved resource plus the provenance needed for run records."""

    locator: str
    overlay_sha256: str
    resolved_path: Path

    def provenance(self) -> dict[str, str]:
        return {
            "locator": self.locator,
            "overlay_sha256": self.overlay_sha256,
            "resolved_path": str(self.resolved_path),
        }


ExpectedType = Literal["any", "file", "dir"]


class LocatorResolver:
    """Resolve locators using one explicitly supplied machine overlay."""

    def __init__(self, local_paths: str | Path) -> None:
        overlay_path = Path(local_paths)
        if not overlay_path.is_file():
            raise LocatorError(f"Local path overlay is not a file: {overlay_path}")
        raw = overlay_path.read_bytes()
        self.overlay_sha256 = hashlib.sha256(raw).hexdigest()
        loaded = yaml.safe_load(raw) or {}
        if not isinstance(loaded, dict):
            raise LocatorError("Local path overlay must be a YAML mapping")
        if loaded.get("schema_version") != "mprisk_local_paths_v1":
            raise LocatorError(
                "Local path overlay schema_version must be mprisk_local_paths_v1"
            )
        roots = loaded.get("roots")
        if not isinstance(roots, dict):
            raise LocatorError("Local path overlay must define a roots mapping")

        normalized: dict[str, dict[str, Path]] = {}
        unknown_schemes = set(roots) - RESOLVABLE_SCHEMES
        if unknown_schemes:
            raise LocatorError(
                "Unknown overlay scheme(s): " + ", ".join(sorted(unknown_schemes))
            )
        for scheme, entries in roots.items():
            if not isinstance(entries, dict):
                raise LocatorError(f"Overlay roots.{scheme} must be a mapping")
            normalized[scheme] = {}
            for key, value in entries.items():
                if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
                    raise LocatorError(f"Invalid overlay key for {scheme}: {key!r}")
                if not isinstance(value, str) or not value:
                    raise LocatorError(f"Overlay path for {scheme}://{key} must be text")
                root = Path(value)
                if not root.is_absolute():
                    raise LocatorError(
                        f"Overlay path for {scheme}://{key} must be absolute: {value}"
                    )
                normalized[scheme][key] = root.resolve(strict=False)
        self._roots = normalized

    def resolve(
        self,
        locator: str,
        *,
        expected_type: ExpectedType = "any",
    ) -> ResolvedLocator:
        scheme, key, suffix = _parse_locator(locator)
        if scheme in IDENTITY_SCHEMES:
            raise LocatorError(
                f"{scheme}:// is an immutable identity and cannot be resolved"
            )
        if scheme not in RESOLVABLE_SCHEMES:
            raise LocatorError(f"Unknown locator scheme: {scheme}")
        try:
            root = self._roots[scheme][key]
        except KeyError as exc:
            raise LocatorError(f"Unknown locator key: {scheme}://{key}") from exc

        path = root.joinpath(*suffix.parts) if suffix.parts else root
        resolved = path.resolve(strict=False)
        if suffix.parts and not resolved.is_relative_to(root):
            raise LocatorError(f"Locator escapes its configured root: {locator}")
        if not resolved.exists():
            raise LocatorError(f"Resolved target does not exist: {locator} -> {resolved}")
        if expected_type == "file" and not resolved.is_file():
            raise LocatorError(f"Resolved target is not a file: {locator} -> {resolved}")
        if expected_type == "dir" and not resolved.is_dir():
            raise LocatorError(
                f"Resolved target is not a directory: {locator} -> {resolved}"
            )
        if expected_type not in {"any", "file", "dir"}:
            raise LocatorError(f"Unknown expected_type: {expected_type}")
        return ResolvedLocator(
            locator=locator,
            overlay_sha256=self.overlay_sha256,
            resolved_path=resolved,
        )


def is_locator(value: object) -> bool:
    """Return whether *value* uses a recognized logical or identity scheme."""

    if not isinstance(value, str):
        return False
    try:
        scheme = urlsplit(value).scheme
    except ValueError:
        return False
    return scheme in RESOLVABLE_SCHEMES | IDENTITY_SCHEMES


def resolve_locator(
    locator: str,
    *,
    local_paths: str | Path,
    expected_type: ExpectedType = "any",
) -> ResolvedLocator:
    """Resolve one locator with an explicitly named overlay."""

    return LocatorResolver(local_paths).resolve(locator, expected_type=expected_type)


def _parse_locator(locator: str) -> tuple[str, str, PurePosixPath]:
    if not isinstance(locator, str) or not locator:
        raise LocatorError("Locator must be a non-empty string")
    if "\\" in locator:
        raise LocatorError(f"Locator must use POSIX separators: {locator}")
    parsed = urlsplit(locator)
    if not parsed.scheme:
        if Path(locator).is_absolute():
            raise LocatorError(f"Absolute protocol path is forbidden: {locator}")
        raise LocatorError(f"Value is not a logical locator: {locator}")
    if parsed.query or parsed.fragment or parsed.username or parsed.password or parsed.port:
        raise LocatorError(f"Locator cannot contain URI extras: {locator}")
    scheme = parsed.scheme
    if scheme not in RESOLVABLE_SCHEMES | IDENTITY_SCHEMES:
        raise LocatorError(f"Unknown locator scheme: {scheme}")
    key = parsed.hostname or ""
    if not key or not _KEY_RE.fullmatch(key):
        raise LocatorError(f"Invalid locator key: {locator}")
    decoded = unquote(parsed.path)
    if decoded != parsed.path:
        raise LocatorError(f"Percent-encoded locator paths are forbidden: {locator}")
    parts = tuple(part for part in PurePosixPath(decoded.lstrip("/")).parts if part)
    if any(part in {".", ".."} for part in parts):
        raise LocatorError(f"Locator traversal is forbidden: {locator}")
    return scheme, key, PurePosixPath(*parts)


def provenance_record(resolved: ResolvedLocator) -> dict[str, Any]:
    """Return the stable serializable provenance representation."""

    return resolved.provenance()
