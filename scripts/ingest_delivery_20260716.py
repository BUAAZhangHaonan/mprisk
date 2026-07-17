from __future__ import annotations

import argparse
import json
from pathlib import Path

from mprisk.data.delivery_20260716 import (
    DEFAULT_FFPROBE_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_ROOT,
    ingest_delivery_20260716,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and normalize the immutable delivery_20260716 manifests."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ffprobe-path", type=Path, default=DEFAULT_FFPROBE_PATH)
    parser.add_argument("--ffprobe-workers", type=int, default=16)
    args = parser.parse_args(argv)
    result = ingest_delivery_20260716(
        source_root=args.source_root,
        output_root=args.output_root,
        ffprobe_path=args.ffprobe_path,
        ffprobe_workers=args.ffprobe_workers,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(result.output_root),
                "provenance_path": str(result.provenance_path),
                "representation_split_path": str(result.representation_split_path),
                "total_rows": result.total_rows,
                "unique_split_groups": result.unique_split_groups,
                "state_valid_rows": result.state_valid_rows,
                "invalid_asset_rows": result.invalid_asset_rows,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
