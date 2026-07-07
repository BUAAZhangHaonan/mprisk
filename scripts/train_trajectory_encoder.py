from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.representation.training import load_training_config, train_trajectory_encoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the tme_supcon_v1 trajectory encoder.")
    parser.add_argument("--dataset", required=True, help="Path to representation_dataset.jsonl")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--output-dir", required=True, help="Directory for training artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_training_config(args.config)
    train_trajectory_encoder(
        dataset_path=args.dataset,
        config=config,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
