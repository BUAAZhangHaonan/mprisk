#!/usr/bin/env python3
"""Batch rename yaml files removing _vN suffixes (preserving ch_sims_v2 dataset name).

Also updates all references in .py / .sh / .yaml / .toml / Makefile files.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

# Files / patterns to never rename (data-set names, architecture keys)
PRESERVE_PATTERNS = (
    "ch_sims_v2",        # dataset name
    "condition_affect_annotation_schema_v2",  # schema enum value (matches file)
    "sample_relation_schema_v2",               # schema enum value (matches file)
    "stage1_emotion_schema",   # already v1-free name
    "sample_type_schema",      # already v1-free name
)

# Regex for version suffixes to strip: _v1, _v2, _v3 followed by '.', '_', or end-of-string
VERSION_SUFFIX = re.compile(r"_v\d+(?=[_.]|$)")


def should_preserve(stem: str) -> bool:
    return any(p in stem for p in PRESERVE_PATTERNS)


def strip_version(name: str) -> str:
    """Remove _vN suffixes from a filename stem or path component."""
    # Repeatedly strip _vN until none left (e.g. foo_v2_bigdim_x2_v3 -> foo_bigdim_x2)
    prev = None
    while prev != name:
        prev = name
        name = VERSION_SUFFIX.sub("", name)
    return name


def find_yaml_files(root: Path) -> list[tuple[Path, Path]]:
    """Return list of (current_path, new_path) for files needing rename."""
    pairs = []
    for path in sorted(root.rglob("*.yaml")):
        if "__pycache__" in path.parts:
            continue
        stem = path.stem
        if should_preserve(stem):
            continue
        new_stem = strip_version(stem)
        if new_stem == stem:
            continue
        new_path = path.with_name(new_stem + path.suffix)
        pairs.append((path, new_path))
    return pairs


def update_references(root: Path, renames: list[tuple[str, str]]) -> dict:
    """Update all string references in source files. Returns count of files touched."""
    # Build a list of (old_string, new_string) replacements.
    # Old: filename stem (without .yaml) - replace even in middle of paths
    # Use longer strings first to avoid partial matches
    sorted_renames = sorted(renames, key=lambda pair: -len(pair[0]))

    extensions = {".py", ".sh", ".yaml", ".yml", ".toml", ".md"}
    files_touched = 0
    replacements_made = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in extensions:
            continue
        if "__pycache__" in path.parts or ".git" in path.parts:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        new = original
        for old, new_str in sorted_renames:
            if old in new:
                new = new.replace(old, new_str)
        if new != original:
            path.write_text(new, encoding="utf-8")
            files_touched += 1
            replacements_made += sum(
                original.count(old) for old, _ in sorted_renames if old in original
            )
    return {"files_touched": files_touched, "replacements": replacements_made}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0]) if args else Path(".")
    dry_run = "--dry-run" in sys.argv

    pairs = find_yaml_files(root)
    if not pairs:
        print("no yaml files need renaming")
        return 0

    print(f"Found {len(pairs)} yaml files to rename:")
    for old, new in pairs[:30]:
        print(f"  {old.relative_to(root)} -> {new.relative_to(root)}")
    if len(pairs) > 30:
        print(f"  ... and {len(pairs) - 30} more")

    # Detect collisions
    new_paths = [p for _, p in pairs]
    if len(set(new_paths)) != len(new_paths):
        seen = set()
        for _, p in pairs:
            if p in seen:
                print(f"COLLISION: multiple files rename to {p}")
            seen.add(p)
        return 1

    # Detect existing target files (would be overwritten)
    collisions = [(old, new) for old, new in pairs if new.exists()]
    if collisions:
        print("\nWARN: target file exists, will skip these renames:")
        for old, new in collisions:
            print(f"  {old.name} -> {new.name} (target exists)")

    if dry_run:
        print("\n(dry-run, no changes made)")
        return 0

    # Execute renames
    renamed = 0
    skip_existing = 0
    for old, new in pairs:
        if new.exists():
            skip_existing += 1
            continue
        old.rename(new)
        renamed += 1
    print(f"\nRenamed {renamed} files ({skip_existing} skipped due to existing targets)")

    # Update references
    rename_strs = [(old.stem, new.stem) for old, new in pairs]
    stats = update_references(root, rename_strs)
    print(f"Updated {stats['files_touched']} files, {stats['replacements']} replacements")
    return 0


if __name__ == "__main__":
    sys.exit(main())
