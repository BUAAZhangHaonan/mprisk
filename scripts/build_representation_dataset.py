from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.representation.relation_dataset import build_relation_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build sample-level A/C relation rows from state bundles."
    )
    parser.add_argument("--bundle-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = build_relation_dataset(
        bundle_manifest_path=Path(args.bundle_manifest),
        output_dir=Path(args.output_dir),
    )
    print(f"relation_dataset={result.dataset_path}")
    print(f"relation_dataset_summary={result.summary_path}")
    print(f"sample_count={result.sample_count}")
    print(f"row_count={result.row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
