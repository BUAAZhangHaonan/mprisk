from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.representation.dataset import build_representation_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build representation-training JSONL from state bundle manifests."
    )
    parser.add_argument(
        "--bundle-manifest",
        required=True,
        help="Input bundle_manifest.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for representation_dataset.jsonl and summary JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_representation_dataset(
        bundle_manifest_path=Path(args.bundle_manifest),
        output_dir=Path(args.output_dir),
    )
    print(f"representation_dataset={result.dataset_path}")
    print(f"representation_dataset_summary={result.summary_path}")
    print(f"total_input_bundles={result.total_input_bundles}")
    print(f"exported_rows={result.exported_rows}")
    print(f"skipped_rows={result.skipped_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
