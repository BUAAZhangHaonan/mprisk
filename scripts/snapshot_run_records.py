from __future__ import annotations

import argparse
from pathlib import Path

from mprisk.viz.runtime_records import (
    snapshot_cache_manifest,
    snapshot_cache_summary,
    snapshot_gpu_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot actual GPU and cache runtime records.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-manifest",
        nargs=2,
        action="append",
        default=[],
        metavar=("KEY", "PATH"),
    )
    parser.add_argument(
        "--cache-summary",
        nargs=2,
        action="append",
        default=[],
        metavar=("KEY", "PATH"),
    )
    args = parser.parse_args()
    snapshot_gpu_records(args.output)
    for key, path in args.cache_manifest:
        snapshot_cache_manifest(args.output, cache_key=key, manifest_path=path)
    for key, path in args.cache_summary:
        snapshot_cache_summary(args.output, cache_key=key, summary_path=path)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
