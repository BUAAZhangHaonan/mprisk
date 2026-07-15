from __future__ import annotations

import argparse
from pathlib import Path

from mprisk.representation.training import export_frozen_baseline_representations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream frozen held-out sample representations from a trained baseline."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--representation-split",
        default="official_test",
        choices=("relation_val", "aligned_calibration", "official_test"),
    )
    args = parser.parse_args(argv)
    result = export_frozen_baseline_representations(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        representation_split=args.representation_split,
    )
    print(f"manifest={result.manifest_path}")
    print(f"summary={result.summary_path}")
    print(f"sample_count={result.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
