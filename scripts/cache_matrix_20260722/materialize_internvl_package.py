"""Materialize an InternVL cache package JSON into a JSONL manifest.

Reads a single JSON object of shape::

    {
        "schema": "mprisk_prefill_cache_union_v2",
        "version": "v2",
        "prefill_strategy": "full_prefill",
        "entries": [<entry object>, ...],
        "provenance": {...},
    }

and writes ``entries`` out one JSON object per line. The output is a plain
``manifest.jsonl`` that :func:`mprisk.setup_helper.setup_cache_manifests` and
:func:`scripts._trainer_lib.scan_cache` (which both read
``<root>/manifest.jsonl``) can consume directly.

The package JSON is ~175MB and parses in a few seconds with ``json.load``;
``ijson`` is used opportunistically when available so we can stream the
``entries`` array without holding the whole object in memory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path


def _have_ijson() -> bool:
    """Eagerly check whether ijson is importable.

    Done outside the generator because ``return None`` inside a generator
    function raises StopIteration rather than returning None — so a lazy
    ``import ijson`` inside the generator would silently yield zero items
    when ijson is missing.
    """
    try:
        import ijson  # noqa: F401  type: ignore[import-not-found]
        return True
    except ImportError:
        return False


def _iter_entries_streaming(package_path: Path) -> Iterable[dict]:
    """Yield entries from the package using ijson. Requires _have_ijson()."""
    import ijson  # type: ignore[import-not-found]

    with open(package_path, "rb") as fh:
        for entry in ijson.items(fh, "entries.item", use_float=True):
            yield entry


def _iter_entries_loaded(package_path: Path) -> Iterable[dict]:
    """Fallback: load the whole JSON, return entries list reference."""
    with open(package_path, "rb") as fh:
        data = json.load(fh)
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError(
            f"Expected top-level 'entries' list, got {type(entries).__name__}"
        )
    return entries


def _summarize(entries_seen: list[dict]) -> dict:
    """Compute summary stats over all entries seen.

    The list is already materialized here for the summary; for the streaming
    path we accumulate set memberships inline so we don't keep every entry.
    """
    sample_ids: set[str] = set()
    conditions: set[str] = set()
    prompts: set[str] = set()
    protocols: set[str] = set()
    for e in entries_seen:
        sample_ids.add(e.get("sample_id"))
        conditions.add(e.get("condition"))
        prompts.add(e.get("prompt_id"))
        protocols.add(e.get("protocol"))
    return {
        "total_entries": len(entries_seen),
        "unique_sample_ids": len(sample_ids),
        "unique_conditions": len(conditions),
        "unique_prompts": len(prompts),
        "unique_protocols": len(protocols),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--package-path",
        required=True,
        type=Path,
        help="Input manifest.package.json (single JSON object with 'entries').",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output manifest.jsonl path. Parent dirs are created.",
    )
    parser.add_argument(
        "--force-stream",
        action="store_true",
        help="Force ijson streaming even if json.load would work.",
    )
    args = parser.parse_args()

    package_path: Path = args.package_path
    output: Path = args.output

    if not package_path.is_file():
        print(f"[error] package not found: {package_path}", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)

    # Pick a source of entries. Prefer ijson streaming when available so we
    # don't hold the full structure in memory twice.
    use_streaming = (not args.force_stream and _have_ijson()) or (
        args.force_stream and _have_ijson()
    )
    if args.force_stream and not _have_ijson():
        print(
            "[error] --force-stream given but ijson is not installed",
            file=sys.stderr,
        )
        return 2
    if use_streaming:
        print(f"[info] streaming entries with ijson: {package_path}", file=sys.stderr)
        entries_iter = _iter_entries_streaming(package_path)
        is_streaming = True
    else:
        print(f"[info] loading JSON with json.load: {package_path}", file=sys.stderr)
        entries_iter = _iter_entries_loaded(package_path)
        is_streaming = False

    # For summary we need set memberships; we don't need to keep the entries
    # themselves. For the non-streaming json.load path we already have the
    # list materialized so we can pass it to _summarize cheaply.
    sample_ids: set[str] = set()
    conditions: set[str] = set()
    prompts: set[str] = set()
    protocols: set[str] = set()
    materialized_for_summary: list[dict] | None = [] if not is_streaming else None

    total = 0
    tmp_out = output.with_suffix(output.suffix + ".tmp")
    with open(tmp_out, "w", encoding="utf-8") as out_fh:
        for entry in entries_iter:
            if not isinstance(entry, dict):
                raise ValueError(f"entry {total} is not a dict: {type(entry)}")
            out_fh.write(json.dumps(entry, ensure_ascii=False))
            out_fh.write("\n")
            sample_ids.add(entry.get("sample_id"))
            conditions.add(entry.get("condition"))
            prompts.add(entry.get("prompt_id"))
            protocols.add(entry.get("protocol"))
            if materialized_for_summary is not None:
                materialized_for_summary.append(entry)
            total += 1
            if total % 5000 == 0:
                print(
                    f"[progress] wrote {total} entries -> {tmp_out}",
                    file=sys.stderr,
                )

    os.replace(tmp_out, output)

    summary = (
        _summarize(materialized_for_summary)
        if materialized_for_summary is not None
        else {
            "total_entries": total,
            "unique_sample_ids": len(sample_ids),
            "unique_conditions": len(conditions),
            "unique_prompts": len(prompts),
            "unique_protocols": len(protocols),
        }
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
