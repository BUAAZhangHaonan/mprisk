from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.representation.training import export_frozen_representations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export frozen condition z and ordered relation r from Proxy Anchor TME."
    )
    parser.add_argument("--dataset", required=True, help="Path to relation_dataset.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = export_frozen_representations(
        dataset_path=Path(args.dataset),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
    )
    print(f"frozen_manifest={result.manifest_path}")
    print(f"spherical_embedding_manifest={result.bundle_manifest_path}")
    print(f"summary={result.summary_path}")
    print(f"row_count={result.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
