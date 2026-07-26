from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.representation.training import load_training_config, train_trajectory_encoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an A/C relation representation.")
    parser.add_argument("--dataset", required=True, help="Path to relation_dataset.jsonl")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--output-dir", required=True, help="Directory for training artifacts")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--encoder-type",
        choices=("gru", "lstm"),
        default=None,
        help="Override the TME sequence module. 'gru' (default) uses "
        "SphericalTMEV1 (1-layer GRU); 'lstm' uses SphericalTME_LSTM "
        "(2-layer uni-directional LSTM). When omitted, the encoder_type "
        "field in the YAML config is used (and falls back to 'gru' if "
        "the YAML does not specify it). Only meaningful when the config's "
        "repr_key is tme_proxy_anchor_v1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed field in the YAML config. Useful for "
        "canonical_rerun multi-seed runs (T1/T5).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_training_config(args.config)
    if args.encoder_type is not None:
        # CLI override wins. Replace the dataclass field via replace() so
        # the rest of the pipeline sees the corrected encoder_type and the
        # checkpoint's architecture_version metadata is consistent.
        config = dataclasses.replace(config, encoder_type=args.encoder_type)
    if args.seed is not None:
        config = dataclasses.replace(config, seed=int(args.seed))
    result = train_trajectory_encoder(
        dataset_path=args.dataset,
        config=config,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume_checkpoint,
        device=args.device,
    )
    print(f"best_checkpoint={result.best_checkpoint_path}")
    print(f"last_checkpoint={result.last_checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
