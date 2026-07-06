from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mprisk.cache.hidden_state_cache import normalize_protocol
from mprisk.cache.prompt_conditioned_cache import (
    PromptConditionedStateEntry,
    prompt_conditioned_entry_from_row,
    write_prompt_conditioned_manifest,
)
from mprisk.utils.io import write_json, write_jsonl


DEFAULT_OUTPUT_ROOT = Path("outputs/prompt_conditioned_cache")


@dataclass(frozen=True)
class BuildPromptConditionedCacheResult:
    manifest_path: Path
    summary_path: Path
    missing_path: Path
    total_source_rows: int
    selected_source_rows: int
    exported_rows: int
    missing_rows: int


def build_prompt_conditioned_cache(
    *,
    mode: str = "A",
    source_manifest: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_key: str | None = None,
    protocol: str | None = None,
    prompt_set_key: str | None = None,
) -> BuildPromptConditionedCacheResult:
    if mode.upper() != "A":
        raise NotImplementedError("Mode B is not implemented yet; use mode A with --source-manifest")
    if source_manifest is None:
        raise ValueError("mode A requires --source-manifest")

    source_rows = _read_source_rows(source_manifest)
    entries: list[PromptConditionedStateEntry] = []
    missing: list[dict[str, Any]] = []
    selected_count = 0
    normalized_protocol = normalize_protocol(protocol) if protocol is not None else None

    for row_number, row in source_rows:
        if not _matches_filters(
            row,
            model_key=model_key,
            protocol=normalized_protocol,
            prompt_set_key=prompt_set_key,
        ):
            continue
        selected_count += 1
        try:
            entries.append(prompt_conditioned_entry_from_row(row))
        except (TypeError, ValueError) as exc:
            missing.append(_missing_row(row_number=row_number, row=row, reason=str(exc)))

    resolved_model_key = _resolve_single_value(
        "model_key",
        explicit=model_key,
        values=[entry.model_key for entry in entries],
    )
    resolved_protocol = _resolve_single_value(
        "protocol",
        explicit=normalized_protocol,
        values=[entry.protocol for entry in entries],
    )
    resolved_prompt_set_key = _resolve_single_value(
        "prompt_set_key",
        explicit=prompt_set_key,
        values=[entry.prompt_set_key for entry in entries],
    )

    output_dir = Path(output_root) / resolved_model_key / resolved_protocol / resolved_prompt_set_key
    manifest_path = write_prompt_conditioned_manifest(output_dir / "manifest.jsonl", entries)
    missing_path = write_jsonl(output_dir / "missing_rows.jsonl", missing)
    summary_path = write_json(
        output_dir / "summary.json",
        {
            "mode": "A",
            "source_manifest": str(source_manifest),
            "model_key": resolved_model_key,
            "protocol": resolved_protocol,
            "prompt_set_key": resolved_prompt_set_key,
            "total_source_rows": len(source_rows),
            "selected_source_rows": selected_count,
            "exported_rows": len(entries),
            "missing_rows": len(missing),
            "manifest_path": str(manifest_path),
            "missing_rows_path": str(missing_path),
            "summary_path": str(summary_path_for(output_dir)),
        },
    )
    return BuildPromptConditionedCacheResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        missing_path=missing_path,
        total_source_rows=len(source_rows),
        selected_source_rows=selected_count,
        exported_rows=len(entries),
        missing_rows=len(missing),
    )


def summary_path_for(output_dir: str | Path) -> Path:
    return Path(output_dir) / "summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build prompt-conditioned cache manifests from existing prompted cache metadata."
    )
    parser.add_argument("--mode", default="A", help="Build mode. Only mode A is implemented.")
    parser.add_argument("--source-manifest", help="Existing prompted cache JSONL metadata.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-key")
    parser.add_argument("--protocol")
    parser.add_argument("--prompt-set-key")
    args = parser.parse_args(argv)

    try:
        result = build_prompt_conditioned_cache(
            mode=args.mode,
            source_manifest=args.source_manifest,
            output_root=args.output_root,
            model_key=args.model_key,
            protocol=args.protocol,
            prompt_set_key=args.prompt_set_key,
        )
    except (NotImplementedError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(result.manifest_path)
    return 0


def _read_source_rows(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    source_path = Path(path)
    rows: list[tuple[int, dict[str, Any]]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source_path}:{line_number}: source row must be a JSON object")
            rows.append((line_number, row))
    return rows


def _matches_filters(
    row: dict[str, Any],
    *,
    model_key: str | None,
    protocol: str | None,
    prompt_set_key: str | None,
) -> bool:
    if model_key is not None and str(row.get("model_key")) != model_key:
        return False
    if protocol is not None and normalize_protocol(str(row.get("protocol"))) != protocol:
        return False
    if prompt_set_key is not None and str(row.get("prompt_set_key")) != prompt_set_key:
        return False
    return True


def _resolve_single_value(
    field: str,
    *,
    explicit: str | None,
    values: list[str],
) -> str:
    if explicit is not None:
        return explicit
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return unique_values[0]
    if not unique_values:
        raise ValueError(f"Cannot infer {field}; pass --{field.replace('_', '-')}")
    raise ValueError(
        f"Cannot infer {field}; selected source rows contain multiple values: "
        + ", ".join(unique_values)
    )


def _missing_row(*, row_number: int, row: dict[str, Any], reason: str) -> dict[str, Any]:
    missing = dict(row)
    missing["row_number"] = row_number
    missing["reason"] = reason
    return missing


if __name__ == "__main__":
    raise SystemExit(main())
