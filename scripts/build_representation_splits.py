from __future__ import annotations

import argparse
from pathlib import Path

from mprisk.data.representation_splits import build_representation_split_assignment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the pre-registered group-level representation split artifact."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_representation_split_assignment(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(f"split_manifest={result.manifest_path}")
    print(f"split_summary={result.summary_path}")
    print(f"group_count={result.group_count}")
    print(f"sample_count={result.sample_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
