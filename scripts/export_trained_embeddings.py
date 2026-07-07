from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.representation.export import export_trained_embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trained TME embeddings.")
    parser.add_argument("--bundle-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repr-key", default="tme_supcon_v1")
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_trained_embeddings(
        bundle_manifest_path=Path(args.bundle_manifest),
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        repr_key=args.repr_key,
        device=args.device,
    )
    print(f"embedding_manifest={result.manifest_path}")
    print(f"embedding_summary={result.summary_path}")
    print(f"total_samples={result.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
