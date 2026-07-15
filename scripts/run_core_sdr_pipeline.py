from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
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

from mprisk.data.manifests import read_jsonl
from mprisk.data.protocol_views import normalize_protocol
from mprisk.data.state_bundle import StateBundleBuildResult, build_state_bundles
from mprisk.data.state_dataset import StateDatasetBuildResult, build_state_dataset
from mprisk.representation.relation_dataset import (
    RelationDatasetBuildResult,
    build_relation_dataset,
)
from mprisk.representation.relation_models import TME_PROXY_ANCHOR_V1
from mprisk.representation.training import (
    FrozenRepresentationExportResult,
    export_frozen_representations,
)
from mprisk.utils.io import ensure_parent


@dataclass(frozen=True)
class CoreSdrPipelineResult:
    state_dataset_result: StateDatasetBuildResult
    bundle_result: StateBundleBuildResult
    relation_dataset_result: RelationDatasetBuildResult
    embedding_manifest_path: Path
    embedding_summary_path: Path
    embedding_count: int
    sdr_scores_path: Path
    state_patterns_path: Path
    state_summary_path: Path
    core_summary_path: Path


def run_core_sdr_pipeline(
    *,
    model_key: str,
    protocol: str,
    prompt_set_key: str,
    repr_key: str,
    manifest_paths: Iterable[str | Path],
    full_cache_root: str | Path,
    prompt_cache_manifest: str | Path,
    prompt_conditioned_cache_manifest: str | Path,
    prompt_set: str | Path,
    output_root: str | Path = ".",
    thresholds: dict[str, Any] | str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> CoreSdrPipelineResult:
    """Run the minimal core SDR pipeline without training or baseline stages."""
    normalized_protocol = normalize_protocol(protocol)
    output_base = Path(output_root) / "outputs"
    if thresholds is None:
        raise ValueError("calibrated Aligned thresholds are required")
    _validate_repr_request(repr_key=repr_key, checkpoint=checkpoint)

    state_dataset_result = build_state_dataset(
        manifest_paths=[Path(path) for path in manifest_paths],
        cache_root=full_cache_root,
        model_key=model_key,
        protocol=normalized_protocol,
        output_dir=output_base / "state_data" / model_key / normalized_protocol,
    )
    bundle_result = build_state_bundles(
        state_dataset_manifest_path=state_dataset_result.manifest_path,
        prompt_cache_manifest_path=prompt_cache_manifest,
        prompt_conditioned_cache_manifest_path=prompt_conditioned_cache_manifest,
        model_key=model_key,
        protocol=normalized_protocol,
        prompt_set_path=prompt_set,
        prompt_set_key=prompt_set_key,
        output_root=output_base / "state_bundles",
    )
    relation_dataset_result = build_relation_dataset(
        bundle_manifest_path=bundle_result.manifest_path,
        output_dir=(
            output_base / "representation_data" / model_key / normalized_protocol / prompt_set_key
        ),
    )
    embedding_result = _export_embeddings(
        relation_dataset_path=relation_dataset_result.dataset_path,
        repr_key=repr_key,
        output_base=output_base,
        checkpoint=checkpoint,
        model_key=model_key,
        protocol=normalized_protocol,
        prompt_set_key=prompt_set_key,
    )
    state_output_dir = (
        output_base / "states" / model_key / normalized_protocol / prompt_set_key / repr_key
    )
    sdr_result = compute_sdr_scores(
        embedding_manifest_path=embedding_result.bundle_manifest_path,
        output_dir=state_output_dir,
    )
    pattern_result = assign_state_patterns(
        sdr_scores_path=sdr_result.scores_path,
        thresholds=thresholds,
        output_dir=state_output_dir,
    )
    core_summary_path = _write_core_summary(
        output_dir=state_output_dir,
        model_key=model_key,
        protocol=normalized_protocol,
        prompt_set_key=prompt_set_key,
        repr_key=repr_key,
        state_dataset_result=state_dataset_result,
        bundle_result=bundle_result,
        relation_dataset_result=relation_dataset_result,
        embedding_result=embedding_result,
        sdr_result=sdr_result,
        pattern_result=pattern_result,
    )
    return CoreSdrPipelineResult(
        state_dataset_result=state_dataset_result,
        bundle_result=bundle_result,
        relation_dataset_result=relation_dataset_result,
        embedding_manifest_path=embedding_result.bundle_manifest_path,
        embedding_summary_path=embedding_result.summary_path,
        embedding_count=embedding_result.count,
        sdr_scores_path=sdr_result.scores_path,
        state_patterns_path=pattern_result.patterns_path,
        state_summary_path=pattern_result.summary_path,
        core_summary_path=core_summary_path,
    )


def _export_embeddings(
    *,
    relation_dataset_path: Path,
    model_key: str,
    protocol: str,
    prompt_set_key: str,
    repr_key: str,
    output_base: Path,
    checkpoint: str | Path | None,
) -> FrozenRepresentationExportResult:
    output_dir = output_base / "representation" / model_key / protocol / prompt_set_key / repr_key
    if repr_key == TME_PROXY_ANCHOR_V1:
        if checkpoint is None:
            raise ValueError(
                "repr_key 'tme_proxy_anchor_v1' requires --checkpoint; "
                "this pipeline does not train checkpoints automatically"
            )
        return export_frozen_representations(
            dataset_path=relation_dataset_path,
            checkpoint_path=checkpoint,
            output_dir=output_dir,
        )
    raise ValueError(
        f"Unsupported core repr_key {repr_key!r}; expected {TME_PROXY_ANCHOR_V1!r}"
    )


def _validate_repr_request(*, repr_key: str, checkpoint: str | Path | None) -> None:
    if repr_key != TME_PROXY_ANCHOR_V1:
        raise ValueError("raw_layernorm representations cannot stand in for the final TME pipeline")
    if checkpoint is None:
        raise ValueError(
            "repr_key 'tme_proxy_anchor_v1' requires --checkpoint; "
            "this pipeline does not train checkpoints automatically"
        )


def _write_core_summary(
    *,
    output_dir: Path,
    model_key: str,
    protocol: str,
    prompt_set_key: str,
    repr_key: str,
    state_dataset_result: StateDatasetBuildResult,
    bundle_result: StateBundleBuildResult,
    relation_dataset_result: RelationDatasetBuildResult,
    embedding_result: FrozenRepresentationExportResult,
    sdr_result: SdrScoreResult,
    pattern_result: StatePatternResult,
) -> Path:
    score_rows = read_jsonl(sdr_result.scores_path)
    pattern_rows = read_jsonl(pattern_result.patterns_path)
    sample_type_counts = Counter(str(row.get("sample_type", "")) for row in score_rows)
    pattern_counts = Counter(str(row.get("pattern", "")) for row in pattern_rows)
    means_by_type = _mean_sdr_by_sample_type(score_rows)
    missing_rows = state_dataset_result.missing_count + bundle_result.missing_count

    summary_path = ensure_parent(output_dir / "CORE_SDR_SUMMARY.md")
    lines = [
        "# Core SDR Summary",
        "",
        f"- Model key: {model_key}",
        f"- Protocol: {protocol}",
        f"- Prompt set key: {prompt_set_key}",
        f"- Repr key: {repr_key}",
        f"- Total samples: {len(score_rows)}",
        f"- Conflict samples: {sample_type_counts.get('Conflict', 0)}",
        f"- Aligned samples: {sample_type_counts.get('Aligned', 0)}",
        f"- Missing rows: {missing_rows}",
        "",
        "## State counts",
        *_mapping_lines(pattern_counts),
        "",
        "## Mean S/D/R by sample_type",
        *_mean_lines(means_by_type),
        "",
        "## Output paths",
        f"- S/D/R scores: `{sdr_result.scores_path}`",
        f"- State patterns: `{pattern_result.patterns_path}`",
        f"- State summary: `{pattern_result.summary_path}`",
        f"- Core SDR summary: `{summary_path}`",
        f"- Embedding manifest: `{embedding_result.bundle_manifest_path}`",
        f"- State dataset manifest: `{state_dataset_result.manifest_path}`",
        f"- Bundle manifest: `{bundle_result.manifest_path}`",
        f"- Relation dataset: `{relation_dataset_result.dataset_path}`",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def _mean_sdr_by_sample_type(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"S": [], "D": [], "R": []}
    )
    for row in rows:
        sample_type = str(row.get("sample_type", ""))
        buckets[sample_type]["S"].append(float(row["S_mean"]))
        buckets[sample_type]["D"].append(float(row["D"]))
        buckets[sample_type]["R"].append(float(row["R"]))
    return {
        sample_type: {
            metric: sum(values) / len(values)
            for metric, values in metrics.items()
            if values
        }
        for sample_type, metrics in sorted(buckets.items())
    }


def _mapping_lines(counts: Counter[str]) -> list[str]:
    if not counts:
        return ["- None"]
    return [f"- {key}: {value}" for key, value in sorted(counts.items())]


def _mean_lines(means_by_type: dict[str, dict[str, float]]) -> list[str]:
    if not means_by_type:
        return ["- None"]
    return [
        "- "
        f"{sample_type}: "
        f"S={metrics.get('S', 0.0):.6f}, "
        f"D={metrics.get('D', 0.0):.6f}, "
        f"R={metrics.get('R', 0.0):.6f}"
        for sample_type, metrics in means_by_type.items()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal core SDR pipeline.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--prompt-set-key", required=True)
    parser.add_argument("--repr-key", required=True)
    parser.add_argument("--manifest-paths", nargs="+", required=True)
    parser.add_argument("--full-cache-root", required=True)
    parser.add_argument("--prompt-cache-manifest", required=True)
    parser.add_argument("--prompt-conditioned-cache-manifest", required=True)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--output-root", default=".")
    parser.add_argument(
        "--thresholds",
        required=True,
        help="Independent Aligned calibration JSON path.",
    )
    parser.add_argument("--checkpoint", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_core_sdr_pipeline(
        model_key=args.model_key,
        protocol=args.protocol,
        prompt_set_key=args.prompt_set_key,
        repr_key=args.repr_key,
        manifest_paths=[Path(path) for path in args.manifest_paths],
        full_cache_root=Path(args.full_cache_root),
        prompt_cache_manifest=Path(args.prompt_cache_manifest),
        prompt_conditioned_cache_manifest=Path(args.prompt_conditioned_cache_manifest),
        prompt_set=Path(args.prompt_set),
        output_root=Path(args.output_root),
        thresholds=args.thresholds,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
    )
    print(f"sdr_scores={result.sdr_scores_path}")
    print(f"state_patterns={result.state_patterns_path}")
    print(f"state_summary={result.state_summary_path}")
    print(f"core_sdr_summary={result.core_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
