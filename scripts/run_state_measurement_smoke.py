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

from mprisk.cache.hidden_state_cache import HiddenStateEntry
from mprisk.cache.prefill_extract import extract_t0_trajectory
from mprisk.cache.prompt_conditioned_cache import prompt_conditioned_entry_from_row
from mprisk.data.manifests import read_jsonl
from mprisk.data.protocol_views import VIEW_KEYS, normalize_protocol
from mprisk.data.state_bundle import StateBundleBuildResult, build_state_bundles
from mprisk.representation.adapters import get_trajectory_encoder
from mprisk.utils.io import ensure_parent, write_json, write_jsonl


@dataclass(frozen=True)
class EmbeddingBuildResult:
    manifest_path: Path
    summary_path: Path
    count: int


@dataclass(frozen=True)
class StateMeasurementSmokeResult:
    bundle_result: StateBundleBuildResult
    embedding_manifest_path: Path
    embedding_summary_path: Path
    embedding_count: int
    sdr_path: Path
    sdr_count: int
    patterns_path: Path
    summary_path: Path
    pattern_count: int
    report_path: Path


def run_state_measurement_smoke(
    *,
    state_dataset_manifest_path: str | Path,
    prompt_cache_manifest_path: str | Path,
    prompt_conditioned_cache_manifest_path: str | Path,
    prompt_set_path: str | Path,
    model_key: str,
    protocol: str,
    prompt_set_key: str,
    repr_key: str,
    output_root: str | Path = "outputs",
    thresholds: dict[str, Any] | str | Path | None = None,
) -> StateMeasurementSmokeResult:
    output_root = Path(output_root)
    normalized_protocol = normalize_protocol(protocol)
    thresholds = thresholds or {"kappa": 0.5, "tau": 0.25, "delta": 0.2}
    bundle_result = build_state_bundles(
        state_dataset_manifest_path=state_dataset_manifest_path,
        prompt_cache_manifest_path=prompt_cache_manifest_path,
        prompt_conditioned_cache_manifest_path=prompt_conditioned_cache_manifest_path,
        prompt_set_path=prompt_set_path,
        prompt_set_key=prompt_set_key,
        model_key=model_key,
        protocol=normalized_protocol,
        output_root=output_root / "state_bundles",
    )
    embedding_result = build_embedding_manifest(
        bundle_manifest_path=bundle_result.manifest_path,
        repr_key=repr_key,
        output_dir=output_root
        / "representation"
        / model_key
        / normalized_protocol
        / prompt_set_key
        / repr_key,
    )
    state_output_dir = output_root / "states" / model_key / normalized_protocol / prompt_set_key / repr_key
    sdr_result = compute_sdr_scores(
        embedding_manifest_path=embedding_result.manifest_path,
        output_dir=state_output_dir,
    )
    pattern_result = assign_state_patterns(
        sdr_scores_path=sdr_result.scores_path,
        thresholds=thresholds,
        output_dir=state_output_dir,
    )
    report_path = _write_smoke_report(
        output_root=output_root,
        bundle_result=bundle_result,
        embedding_result=embedding_result,
        sdr_result=sdr_result,
        pattern_result=pattern_result,
    )
    return StateMeasurementSmokeResult(
        bundle_result=bundle_result,
        embedding_manifest_path=embedding_result.manifest_path,
        embedding_summary_path=embedding_result.summary_path,
        embedding_count=embedding_result.count,
        sdr_path=sdr_result.scores_path,
        sdr_count=sdr_result.count,
        patterns_path=pattern_result.patterns_path,
        summary_path=pattern_result.summary_path,
        pattern_count=pattern_result.count,
        report_path=report_path,
    )


def build_embedding_manifest(
    *,
    bundle_manifest_path: str | Path,
    repr_key: str,
    output_dir: str | Path,
) -> EmbeddingBuildResult:
    encoder = get_trajectory_encoder(repr_key)
    bundle_rows = read_jsonl(bundle_manifest_path)
    embedding_rows = [
        _embedding_row(bundle, repr_key=repr_key, encoder=encoder) for bundle in bundle_rows
    ]
    output_root = Path(output_dir)
    manifest_path = write_jsonl(output_root / "embedding_manifest.jsonl", embedding_rows)
    summary_path = write_json(
        output_root / "embedding_summary.json",
        {
            "bundle_manifest": str(bundle_manifest_path),
            "embedding_manifest": str(manifest_path),
            "repr_key": repr_key,
            "total_samples": len(embedding_rows),
        },
    )
    return EmbeddingBuildResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        count=len(embedding_rows),
    )


def _embedding_row(bundle: dict[str, Any], *, repr_key: str, encoder) -> dict[str, Any]:
    return {
        "sample_id": bundle["sample_id"],
        "sample_type": bundle["sample_type"],
        "model_key": bundle["model_key"],
        "protocol": bundle["protocol"],
        "prompt_set_key": bundle["prompt_set_key"],
        "repr_key": repr_key,
        "embeddings": {
            view_key: {
                prompt_id: encoder.encode(
                    extract_t0_trajectory(
                        _entry_from_prompt_conditioned_state(
                            bundle["views"][view_key]["prompts"][prompt_id][
                                "prompt_conditioned_state"
                            ]
                        )
                    )
                )
                for prompt_id in bundle["views"][view_key]["prompts"]
            }
            for view_key in VIEW_KEYS
        },
    }


def _entry_from_state_cache(row: dict[str, Any]) -> HiddenStateEntry:
    return HiddenStateEntry(
        sample_id=row["sample_id"],
        model_key=row["model_key"],
        protocol=row["protocol"],
        condition=row["condition"],
        dataset_key=row["dataset_key"],
        split=row["split"],
        shard_path=row["shard_path"],
        index_in_shard=row["index_in_shard"],
        layer_count=row["layer_count"],
        hidden_dim=row["hidden_dim"],
        token_count=row["token_count"],
        cache_root=row["cache_root"],
        checksum=row.get("checksum"),
        metadata=row.get("metadata") or {},
    )


def _entry_from_prompt_conditioned_state(row: dict[str, Any]) -> HiddenStateEntry:
    return prompt_conditioned_entry_from_row(row).to_hidden_state_entry()


def _write_smoke_report(
    *,
    output_root: Path,
    bundle_result: StateBundleBuildResult,
    embedding_result: EmbeddingBuildResult,
    sdr_result: SdrScoreResult,
    pattern_result: StatePatternResult,
) -> Path:
    report_path = ensure_parent(output_root / "states/reports/STATE_MEASUREMENT_SMOKE.md")
    lines = [
        "# State Measurement Smoke Report",
        "",
        f"- Bundle manifest: `{bundle_result.manifest_path}`",
        f"- Embedding manifest: `{embedding_result.manifest_path}`",
        f"- S/D/R scores: `{sdr_result.scores_path}`",
        f"- State patterns: `{pattern_result.patterns_path}`",
        f"- State summary: `{pattern_result.summary_path}`",
        f"- Complete bundles: {bundle_result.complete_count}",
        f"- Embedding rows: {embedding_result.count}",
        f"- S/D/R rows: {sdr_result.count}",
        f"- Pattern rows: {pattern_result.count}",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the state-measurement smoke pipeline.")
    parser.add_argument("--state-dataset-manifest", required=True)
    parser.add_argument("--prompt-cache-manifest", required=True)
    parser.add_argument("--prompt-conditioned-cache-manifest", required=True)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--prompt-set-key", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--repr-key", default="raw_layernorm_mean")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--thresholds",
        default='{"kappa": 0.5, "tau": 0.25, "delta": 0.2}',
        help="JSON string or path to JSON threshold config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_state_measurement_smoke(
        state_dataset_manifest_path=Path(args.state_dataset_manifest),
        prompt_cache_manifest_path=Path(args.prompt_cache_manifest),
        prompt_conditioned_cache_manifest_path=Path(args.prompt_conditioned_cache_manifest),
        prompt_set_path=Path(args.prompt_set),
        model_key=args.model_key,
        protocol=args.protocol,
        prompt_set_key=args.prompt_set_key,
        repr_key=args.repr_key,
        output_root=Path(args.output_root),
        thresholds=args.thresholds,
    )
    print(f"bundle_manifest={result.bundle_result.manifest_path}")
    print(f"embedding_manifest={result.embedding_manifest_path}")
    print(f"sdr_scores={result.sdr_path}")
    print(f"state_patterns={result.patterns_path}")
    print(f"state_summary={result.summary_path}")
    print(f"smoke_report={result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
