#!/usr/bin/env python
"""Filter a cache_matrix_20260722 source cache to a single canonical prompt_id.

cache_matrix_20260722 source caches store 8 prompt variants x 3 conditions per
sample (24 rows per sample). The state_dataset pipeline's
``FullCacheManifest.resolve_m_conditions`` picks the first entry per condition;
when those entries come from different prompts they have different
``t0_token_index`` values, triggering
``_require_consistent_entry_shape`` hard-fail in
``mprisk.data.state_dataset``.

This script reads a source cache directory (one containing ``manifest.jsonl``),
keeps only rows whose ``prompt_id`` matches the canonical prompt, and writes a
*new* cache directory containing:

  - ``manifest.jsonl``              (filtered JSONL)
  - ``unified_full_cache_manifest.json``   (wrapped form, ready for
                                           ``load_full_cache_manifest``)
  - ``shards`` -> source ``shards`` symlink (so safetensors shards still resolve)

Idempotent: re-running with the same output dir is a no-op if the canonical
prompt filter would produce the same content. The wrapped JSON is rebuilt
every time the filtered JSONL changes.

CLI:
  python filter_cache_manifest.py \\
      --source-cache-root /path/to/source/<MODEL> \\
      --target-cache-root /path/to/filtered/<MODEL> \\
      [--canonical-prompt pregen_risk_v1_p001]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

DEFAULT_CANONICAL_PROMPT = "pregen_risk_v1_p001"


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def filter_cache_to_canonical_prompt(
    *,
    source_cache_root: Path,
    target_cache_root: Path,
    canonical_prompt: str = DEFAULT_CANONICAL_PROMPT,
    overwrite: bool = False,
) -> tuple[Path, int, int]:
    """Materialise a filtered cache dir containing only the canonical prompt.

    Returns ``(target_cache_root, kept_rows, total_rows)``.
    """
    source_cache_root = source_cache_root.resolve()
    target_cache_root = target_cache_root.resolve()
    if not source_cache_root.is_dir():
        raise FileNotFoundError(f"source cache root not found: {source_cache_root}")
    source_jsonl = source_cache_root / "manifest.jsonl"
    if not source_jsonl.is_file():
        raise FileNotFoundError(f"source manifest.jsonl not found: {source_jsonl}")

    target_cache_root.mkdir(parents=True, exist_ok=True)
    target_jsonl = target_cache_root / "manifest.jsonl"
    target_wrapped = target_cache_root / "unified_full_cache_manifest.json"

    # Idempotency: if target JSONL exists and matches the canonical prompt
    # filter signature, skip rewrite. We detect this by checking that every
    # row in target has prompt_id == canonical AND that the target JSONL's
    # sha256 matches a fresh filter of the source.
    total = 0
    kept = 0
    filtered_rows: list[dict] = []
    for row in _iter_jsonl(source_jsonl):
        total += 1
        if row.get("prompt_id") == canonical_prompt:
            kept += 1
            filtered_rows.append(row)

    if kept == 0:
        raise ValueError(
            f"canonical prompt_id {canonical_prompt!r} not found in "
            f"{source_jsonl}; available prompts cannot be determined "
            f"(no rows matched)"
        )

    # Idempotency check: if existing target JSONL has identical sha256 to a
    # freshly serialized version of filtered_rows, we are done.
    new_jsonl_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        for row in filtered_rows
    )
    new_jsonl_sha = hashlib.sha256(new_jsonl_bytes).hexdigest()

    if target_jsonl.is_file() and _sha256_file(target_jsonl) == new_jsonl_sha:
        # Already up to date. Make sure wrapped JSON and shards link also exist.
        if not target_wrapped.is_file():
            _write_wrapped(target_wrapped, filtered_rows)
        _ensure_shards_link(source_cache_root, target_cache_root)
        return target_cache_root, kept, total

    if target_jsonl.is_file() and not overwrite:
        # Stale content; rewrite (filter_cache_manifest.py is idempotent on
        # canonical-prompt content, so an existing-but-different file means
        # either the source changed or canonical_prompt changed; either way
        # we overwrite).
        pass

    target_jsonl.write_bytes(new_jsonl_bytes)
    _write_wrapped(target_wrapped, filtered_rows)
    _ensure_shards_link(source_cache_root, target_cache_root)
    return target_cache_root, kept, total


def _write_wrapped(path: Path, rows: list[dict]) -> None:
    payload = {
        "schema": "cache_matrix_20260722_wrapped_jsonl_v1",
        "entries": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ensure_shards_link(source_cache_root: Path, target_cache_root: Path) -> None:
    """Symlink target/shards -> source/shards so shard paths resolve.

    Some cache rows reference absolute shard paths under
    ``source_cache_root/shards/``. Re-pointing them is risky, so we create a
    symlink at ``target_cache_root/shards`` that mirrors the source layout.

    Manifest rows carry their own absolute ``shard_path`` fields, so the
    symlink is only needed if any row uses a relative shard path. We always
    create the link to be safe; it is harmless if unused.
    """
    source_shards = source_cache_root / "shards"
    target_shards = target_cache_root / "shards"
    if target_shards.is_symlink() or target_shards.exists():
        return
    if source_shards.exists():
        target_shards.symlink_to(source_shards)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cache-root", required=True, type=Path)
    p.add_argument("--target-cache-root", required=True, type=Path)
    p.add_argument(
        "--canonical-prompt",
        default=DEFAULT_CANONICAL_PROMPT,
        help=f"prompt_id to keep (default: {DEFAULT_CANONICAL_PROMPT})",
    )
    args = p.parse_args(argv)

    target, kept, total = filter_cache_to_canonical_prompt(
        source_cache_root=args.source_cache_root,
        target_cache_root=args.target_cache_root,
        canonical_prompt=args.canonical_prompt,
    )
    print(
        f"[filter_cache_manifest] {args.source_cache_root} -> {target}: "
        f"kept {kept}/{total} rows (prompt_id={args.canonical_prompt})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
