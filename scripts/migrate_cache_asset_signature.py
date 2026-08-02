#!/usr/bin/env python3
"""Verify or explicitly apply one fail-closed cache signature migration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mprisk.cache.signature_migration import migrate_asset_signature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report-path", required=True, type=Path)
    return parser


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = build_parser().parse_args()
    report = migrate_asset_signature(
        args.config,
        args.manifest,
        apply=args.apply,
    )
    _atomic_json(args.report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
