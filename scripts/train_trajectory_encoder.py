from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.representation.training import load_training_config, train_trajectory_encoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an A/C relation representation.")
    parser.add_argument("--dataset", default=None, help="Path to relation_dataset.jsonl (required unless --load-existing)")
    parser.add_argument("--config", default=None, help="Path to training YAML config (required unless --load-existing)")
    parser.add_argument("--output-dir", required=True, help="Directory for training artifacts (also receives target_metrics.json when --eval-dataset is set)")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--encoder-type",
        choices=("gru", "lstm", "bilstm"),
        default=None,
        help="Override the TME sequence module. 'gru' (default) uses "
        "SphericalTMEV1 (1-layer GRU); 'lstm' uses SphericalTME_LSTM "
        "(2-layer uni-directional LSTM); 'bilstm' uses SphericalTME_BiLSTM (2-layer bi-directional LSTM). When omitted, the encoder_type "
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
    parser.add_argument(
        "--eval-dataset",
        default=None,
        help="Optional path to a Target relation_dataset_target.jsonl. When set, "
        "after training (or after loading an existing checkpoint via "
        "--load-existing) the encoder is evaluated on this dataset and "
        "target_metrics.json is written next to train_metrics.json.",
    )
    parser.add_argument(
        "--load-existing",
        default=None,
        help="Path to an existing best_checkpoint.pt. When combined with "
        "--eval-dataset, skips training entirely and only runs Target eval. "
        "The training config is loaded from the checkpoint payload.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        choices=("relation_val", "relation_train", "official_test", "aligned_calibration"),
        help="Restrict --eval-dataset rows to this registered representation_split "
        "and write eval_f1.json instead of target_metrics.json. Used for "
        "eval-only Source val F1 backfill on best_checkpoint.pt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Cross-domain eval-only path: load an existing Source checkpoint and
    # evaluate it on a Target relation_dataset_target.jsonl. Skips training.
    if args.load_existing is not None:
        if args.eval_dataset is None:
            raise SystemExit("--load-existing requires --eval-dataset")
        from mprisk.representation.training import evaluate_target_dataset
        target_metrics_path = evaluate_target_dataset(
            checkpoint_path=args.load_existing,
            eval_dataset_path=args.eval_dataset,
            output_dir=args.output_dir,
            device=args.device,
            representation_split=args.eval_split,
        )
        print(f"eval_metrics_path={target_metrics_path}")
        return 0

    if args.dataset is None or args.config is None:
        raise SystemExit("--dataset and --config are required for the training path")

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

    # Optional cross-domain eval on a Target relation dataset. The encoder
    # is the just-trained best Source checkpoint.
    if args.eval_dataset is not None:
        from mprisk.representation.training import evaluate_target_dataset
        target_metrics_path = evaluate_target_dataset(
            checkpoint_path=result.best_checkpoint_path,
            eval_dataset_path=args.eval_dataset,
            output_dir=args.output_dir,
            device=args.device,
        )
        print(f"target_metrics_path={target_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
