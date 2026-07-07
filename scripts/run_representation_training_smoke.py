from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from scripts.assign_state_patterns import StatePatternResult, assign_state_patterns
    from scripts.compute_sdr_scores import SdrScoreResult, compute_sdr_scores
except ModuleNotFoundError:
    from assign_state_patterns import StatePatternResult, assign_state_patterns
    from compute_sdr_scores import SdrScoreResult, compute_sdr_scores

from mprisk.data.protocol_views import normalize_protocol
from mprisk.representation.dataset import (
    RepresentationDatasetBuildResult,
    build_representation_dataset,
)
from mprisk.representation.export import TrainedEmbeddingExportResult, export_trained_embeddings
from mprisk.representation.training import (
    REPR_KEY,
    TrainingResult,
    load_training_config,
    train_trajectory_encoder,
)
from mprisk.utils.io import ensure_parent


DEFAULT_THRESHOLDS = {"kappa": 0.5, "tau": 0.25, "delta": 0.2}


@dataclass(frozen=True)
class RepresentationTrainingSmokeResult:
    representation_dataset_path: Path
    checkpoint_path: Path
    embedding_manifest_path: Path
    sdr_scores_path: Path
    state_patterns_path: Path
    report_path: Path
    sample_count: int
    embedding_dim: int | None
    final_train_loss: float | None
    dataset_result: RepresentationDatasetBuildResult
    training_result: TrainingResult
    embedding_result: TrainedEmbeddingExportResult
    sdr_result: SdrScoreResult
    pattern_result: StatePatternResult


def run_representation_training_smoke(
    *,
    bundle_manifest_path: str | Path,
    config_path: str | Path,
    model_key: str,
    protocol: str,
    prompt_set_key: str,
    output_root: str | Path,
    device: str = "cpu",
    thresholds: dict[str, Any] | str | Path | None = None,
) -> RepresentationTrainingSmokeResult:
    normalized_protocol = normalize_protocol(protocol)
    outputs_root = Path(output_root) / "outputs"
    repr_key = REPR_KEY
    scoped_root = Path(model_key) / normalized_protocol / prompt_set_key

    representation_data_dir = outputs_root / "representation_data" / scoped_root
    train_dir = outputs_root / "representation_train" / scoped_root / repr_key
    embedding_dir = outputs_root / "representation" / scoped_root / repr_key
    state_dir = outputs_root / "states" / scoped_root / repr_key

    dataset_result = build_representation_dataset(
        bundle_manifest_path=bundle_manifest_path,
        output_dir=representation_data_dir,
    )
    training_result = train_trajectory_encoder(
        dataset_path=dataset_result.dataset_path,
        config=load_training_config(config_path),
        output_dir=train_dir,
    )
    embedding_result = export_trained_embeddings(
        bundle_manifest_path=bundle_manifest_path,
        checkpoint_path=training_result.checkpoint_path,
        output_dir=embedding_dir,
        repr_key=repr_key,
        device=device,
    )
    sdr_result = compute_sdr_scores(
        embedding_manifest_path=embedding_result.manifest_path,
        output_dir=state_dir,
    )
    pattern_result = assign_state_patterns(
        sdr_scores_path=sdr_result.scores_path,
        thresholds=thresholds or DEFAULT_THRESHOLDS,
        output_dir=state_dir,
    )
    final_train_loss = _final_train_loss(training_result.metrics)
    embedding_dim = embedding_result.summary.get("embedding_dim")
    report_path = _write_smoke_report(
        outputs_root=outputs_root,
        dataset_result=dataset_result,
        training_result=training_result,
        embedding_result=embedding_result,
        sdr_result=sdr_result,
        pattern_result=pattern_result,
        sample_count=embedding_result.count,
        embedding_dim=embedding_dim,
        final_train_loss=final_train_loss,
    )
    return RepresentationTrainingSmokeResult(
        representation_dataset_path=dataset_result.dataset_path,
        checkpoint_path=training_result.checkpoint_path,
        embedding_manifest_path=embedding_result.manifest_path,
        sdr_scores_path=sdr_result.scores_path,
        state_patterns_path=pattern_result.patterns_path,
        report_path=report_path,
        sample_count=embedding_result.count,
        embedding_dim=embedding_dim,
        final_train_loss=final_train_loss,
        dataset_result=dataset_result,
        training_result=training_result,
        embedding_result=embedding_result,
        sdr_result=sdr_result,
        pattern_result=pattern_result,
    )


def _final_train_loss(metrics: dict[str, Any]) -> float | None:
    value = metrics.get("final_train_loss")
    if value is None:
        return None
    return float(value)


def _write_smoke_report(
    *,
    outputs_root: Path,
    dataset_result: RepresentationDatasetBuildResult,
    training_result: TrainingResult,
    embedding_result: TrainedEmbeddingExportResult,
    sdr_result: SdrScoreResult,
    pattern_result: StatePatternResult,
    sample_count: int,
    embedding_dim: int | None,
    final_train_loss: float | None,
) -> Path:
    report_path = ensure_parent(
        outputs_root / "representation_train/reports/REPRESENTATION_TRAINING_SMOKE.md"
    )
    lines = [
        "# Representation Training Smoke Report",
        "",
        f"- Representation dataset: `{dataset_result.dataset_path}`",
        f"- Checkpoint: `{training_result.checkpoint_path}`",
        f"- Embedding manifest: `{embedding_result.manifest_path}`",
        f"- S/D/R scores: `{sdr_result.scores_path}`",
        f"- State patterns: `{pattern_result.patterns_path}`",
        f"- Sample count: {sample_count}",
        f"- Embedding dim: {embedding_dim}",
        f"- Final train loss: {final_train_loss}",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the trained representation smoke pipeline."
    )
    parser.add_argument("--bundle-manifest", required=True)
    parser.add_argument("--config", required=True, help="Training YAML config.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--prompt-set-key", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--thresholds",
        default=None,
        help="JSON string or path to JSON threshold config.",
    )
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_representation_training_smoke(
        bundle_manifest_path=Path(args.bundle_manifest),
        config_path=Path(args.config),
        model_key=args.model_key,
        protocol=args.protocol,
        prompt_set_key=args.prompt_set_key,
        output_root=Path(args.output_root),
        device=args.device,
        thresholds=_thresholds_from_args(args),
    )
    print(f"representation_dataset={result.representation_dataset_path}")
    print(f"checkpoint={result.checkpoint_path}")
    print(f"embedding_manifest={result.embedding_manifest_path}")
    print(f"sdr_scores={result.sdr_scores_path}")
    print(f"state_patterns={result.state_patterns_path}")
    print(f"smoke_report={result.report_path}")
    return 0


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, Any] | str | Path | None:
    overrides = {
        key: value
        for key, value in {
            "kappa": args.kappa,
            "tau": args.tau,
            "delta": args.delta,
        }.items()
        if value is not None
    }
    if args.thresholds is None:
        if overrides:
            return {**DEFAULT_THRESHOLDS, **overrides}
        return None
    if not overrides:
        return args.thresholds

    thresholds = _load_threshold_mapping(args.thresholds)
    thresholds.update(overrides)
    return thresholds


def _load_threshold_mapping(source: str) -> dict[str, Any]:
    path = Path(source)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(source)
    if not isinstance(payload, dict):
        raise ValueError("thresholds must be a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
