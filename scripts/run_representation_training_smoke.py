from __future__ import annotations

import argparse
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
from mprisk.representation.relation_dataset import (
    RelationDatasetBuildResult,
    build_relation_dataset,
)
from mprisk.representation.relation_models import TME_PROXY_ANCHOR_V1
from mprisk.representation.training import (
    FrozenRepresentationExportResult,
    TrainingResult,
    export_frozen_representations,
    load_training_config,
    train_trajectory_encoder,
)
from mprisk.utils.io import ensure_parent


@dataclass(frozen=True)
class RepresentationTrainingSmokeResult:
    relation_dataset_path: Path
    checkpoint_path: Path
    embedding_manifest_path: Path
    sdr_scores_path: Path
    state_patterns_path: Path
    report_path: Path
    sample_count: int
    condition_dim: int
    final_train_loss: float
    dataset_result: RelationDatasetBuildResult
    training_result: TrainingResult
    embedding_result: FrozenRepresentationExportResult
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
    thresholds: dict[str, Any] | str | Path,
    device: str = "cpu",
) -> RepresentationTrainingSmokeResult:
    normalized_protocol = normalize_protocol(protocol)
    config = load_training_config(config_path)
    if config.repr_key != TME_PROXY_ANCHOR_V1:
        raise ValueError("the spherical smoke pipeline requires tme_proxy_anchor_v1")
    if config.model_key != model_key:
        raise ValueError("config model_key does not match requested model_key")

    outputs_root = Path(output_root) / "outputs"
    scoped_root = Path(model_key) / normalized_protocol / prompt_set_key
    representation_data_dir = outputs_root / "representation_data" / scoped_root
    train_dir = outputs_root / "representation_train" / scoped_root / config.repr_key
    embedding_dir = outputs_root / "representation" / scoped_root / config.repr_key
    state_dir = outputs_root / "states" / scoped_root / config.repr_key

    dataset_result = build_relation_dataset(
        bundle_manifest_path=bundle_manifest_path,
        output_dir=representation_data_dir,
        prompt_set_key=config.prompt_set_key,
        prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
        expected_prompt_count=config.expected_prompt_count,
        expected_prompt_ids=config.expected_prompt_ids,
    )
    training_result = train_trajectory_encoder(
        dataset_path=dataset_result.dataset_path,
        config=config,
        output_dir=train_dir,
        device=device,
    )
    embedding_result = export_frozen_representations(
        dataset_path=dataset_result.dataset_path,
        checkpoint_path=training_result.checkpoint_path,
        output_dir=embedding_dir,
    )
    sdr_result = compute_sdr_scores(
        embedding_manifest_path=embedding_result.bundle_manifest_path,
        output_dir=state_dir,
    )
    pattern_result = assign_state_patterns(
        sdr_scores_path=sdr_result.scores_path,
        thresholds=thresholds,
        output_dir=state_dir,
    )
    final_train_loss = _final_train_loss(training_result.log_path)
    report_path = _write_smoke_report(
        outputs_root=outputs_root,
        dataset_result=dataset_result,
        training_result=training_result,
        embedding_result=embedding_result,
        sdr_result=sdr_result,
        pattern_result=pattern_result,
        condition_dim=config.condition_dim,
        final_train_loss=final_train_loss,
    )
    return RepresentationTrainingSmokeResult(
        relation_dataset_path=dataset_result.dataset_path,
        checkpoint_path=training_result.checkpoint_path,
        embedding_manifest_path=embedding_result.bundle_manifest_path,
        sdr_scores_path=sdr_result.scores_path,
        state_patterns_path=pattern_result.patterns_path,
        report_path=report_path,
        sample_count=dataset_result.sample_count,
        condition_dim=config.condition_dim,
        final_train_loss=final_train_loss,
        dataset_result=dataset_result,
        training_result=training_result,
        embedding_result=embedding_result,
        sdr_result=sdr_result,
        pattern_result=pattern_result,
    )


def _final_train_loss(log_path: Path) -> float:
    import json

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("training log is empty")
    return float(rows[-1]["train_loss"])


def _write_smoke_report(
    *,
    outputs_root: Path,
    dataset_result: RelationDatasetBuildResult,
    training_result: TrainingResult,
    embedding_result: FrozenRepresentationExportResult,
    sdr_result: SdrScoreResult,
    pattern_result: StatePatternResult,
    condition_dim: int,
    final_train_loss: float,
) -> Path:
    report_path = ensure_parent(
        outputs_root / "representation_train/reports/REPRESENTATION_TRAINING_SMOKE.md"
    )
    lines = [
        "# Representation Training Smoke Report",
        "",
        f"- Relation dataset: `{dataset_result.dataset_path}`",
        f"- Checkpoint: `{training_result.checkpoint_path}`",
        f"- Frozen embedding manifest: `{embedding_result.bundle_manifest_path}`",
        f"- S/D/R scores: `{sdr_result.scores_path}`",
        f"- State patterns: `{pattern_result.patterns_path}`",
        f"- Sample count: {dataset_result.sample_count}",
        f"- Condition dim: {condition_dim}",
        f"- Final train loss: {final_train_loss}",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Proxy Anchor TME smoke pipeline.")
    parser.add_argument("--bundle-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--prompt-set-key", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--device", default="cpu")
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
        thresholds=args.thresholds,
        device=args.device,
    )
    print(f"relation_dataset={result.relation_dataset_path}")
    print(f"checkpoint={result.checkpoint_path}")
    print(f"embedding_manifest={result.embedding_manifest_path}")
    print(f"sdr_scores={result.sdr_scores_path}")
    print(f"state_patterns={result.state_patterns_path}")
    print(f"smoke_report={result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
