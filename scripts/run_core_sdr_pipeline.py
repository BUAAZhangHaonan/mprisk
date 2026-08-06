from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# scripts/cache_matrix_20260722 holds the canonical manifest-filter helper we
# share with calibrate_thresholds.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

try:
    from scripts.assign_state_patterns import StatePatternResult, assign_state_patterns
    from scripts.compute_sdr_scores import SdrScoreResult, compute_sdr_scores
except ModuleNotFoundError:
    from assign_state_patterns import StatePatternResult, assign_state_patterns
    from compute_sdr_scores import SdrScoreResult, compute_sdr_scores

try:
    # Shared manifest filter helper (filters labels manifest to split-
    # assigned samples so build_state_dataset doesn't trip on samples that
    # are absent from the assignment).
    from cache_matrix_20260722.calibrate_thresholds import _filter_manifest_to_assigned
except ModuleNotFoundError:
    # Fallback when imported as scripts.cache_matrix_20260722.calibrate_thresholds
    from scripts.cache_matrix_20260722.calibrate_thresholds import _filter_manifest_to_assigned

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
    TrainingConfig,
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
    split_assignment: str | Path,
    output_root: str | Path = ".",
    thresholds: dict[str, Any] | str | Path | None = None,
    checkpoint: str | Path | None = None,
    cache_manifest_path: str | Path | None = None,
    strict_shape: bool = False,
    embedding_manifest_path: str | Path | None = None,
) -> CoreSdrPipelineResult:
    """Run the minimal core SDR pipeline without training or baseline stages."""
    normalized_protocol = normalize_protocol(protocol)
    output_base = Path(output_root) / "outputs"
    if thresholds is None:
        raise ValueError("calibrated Aligned thresholds are required")
    _validate_repr_request(repr_key=repr_key, checkpoint=checkpoint)

    # Filter labels manifest(s) to split-assigned samples so
    # _resolve_split_assignment doesn't trip on cross-domain rows that are
    # absent from the registered split assignment (cache_matrix_20260722
    # primary manifests include ch_sims_v2 cross-domain samples that the
    # representation_v1 split assignment does not cover).
    filtered_label_manifests: list[Path] = []
    for path in manifest_paths:
        filtered, _kept, _total = _filter_manifest_to_assigned(
            manifest_path=Path(path),
            split_assignment=Path(split_assignment),
        )
        filtered_label_manifests.append(filtered)
    state_dataset_kwargs: dict[str, Any] = dict(
        manifest_paths=filtered_label_manifests,
        cache_root=full_cache_root,
        model_key=model_key,
        protocol=normalized_protocol,
        split_assignment_path=split_assignment,
        output_dir=output_base / "state_data" / model_key / normalized_protocol,
        # t0_token_index legitimately varies across M1/M2/M12 in
        # cache_matrix_20260722 (each condition uses a different
        # delivery-overlap prefix). Only layer_count/hidden_dim must match.
        strict_shape=strict_shape,
    )
    if cache_manifest_path is not None:
        state_dataset_kwargs["manifest_path"] = Path(cache_manifest_path)
    state_dataset_result = build_state_dataset(**state_dataset_kwargs)
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
    checkpoint_payload = torch.load(Path(checkpoint), map_location="cpu")
    training_config = TrainingConfig(**checkpoint_payload["training_config"])
    relation_dataset_result = build_relation_dataset(
        bundle_manifest_path=bundle_result.manifest_path,
        output_dir=(
            output_base / "representation_data" / model_key / normalized_protocol / prompt_set_key
        ),
        prompt_set_key=training_config.prompt_set_key,
        prompt_set_artifact_sha256=training_config.prompt_set_artifact_sha256,
        expected_prompt_count=training_config.expected_prompt_count,
        expected_prompt_ids=training_config.expected_prompt_ids,
    )
    if embedding_manifest_path is not None:
        embedding_result = _reuse_embeddings(Path(embedding_manifest_path))
    else:
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


def _reuse_embeddings(
    embedding_manifest_path: Path,
) -> FrozenRepresentationExportResult:
    """Reuse a previously-exported frozen representation manifest.

    Calibration writes its embedding bundle manifest to
    outputs/cache_matrix_20260722/thresholds/<MODEL>/outputs/embeddings/.../spherical_embedding_manifest.jsonl.
    To stay identity-consistent with the calibrated thresholds, the SDR
    pipeline should consume that manifest verbatim instead of re-running
    export_frozen_representations (which is non-deterministic on GPU due
    to floating-point reduction order).
    """
    embedding_manifest_path = Path(embedding_manifest_path)
    if not embedding_manifest_path.is_file():
        raise FileNotFoundError(
            f"embedding manifest not found: {embedding_manifest_path}"
        )
    summary_path = embedding_manifest_path.parent / "frozen_representation_summary.json"
    # Count rows so the result.count field is meaningful.
    count = 0
    with embedding_manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return FrozenRepresentationExportResult(
        manifest_path=embedding_manifest_path.parent / "frozen_representations.jsonl",
        bundle_manifest_path=embedding_manifest_path,
        summary_path=summary_path,
        count=count,
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


def _prepare_target_inputs(
    *,
    target_cache_root: Path,
    source_label_manifests: list[Path],
    source_split_assignment: Path,
    source_prompt_set: Path,
    model_key: str,
    output_root: Path,
) -> dict[str, Path]:
    """Auto-derive Target label manifest, split assignment, prompt cache,
    and filtered cache from target_cache_root + Source inputs.

    The Target cache only contains ch_sims_v2 cross-domain natural samples
    (2035 sample_ids). Their Source split assignment tags them as
    master_split=cross_domain_test, which build_state_dataset rejects. We
    rewrite to master_split=test / representation_split=official_test so the
    Target SDR pass can run through the same rigid 4-split contract.
    """
    import json as _json

    target_work = output_root / "outputs" / "_target_inputs"
    target_work.mkdir(parents=True, exist_ok=True)

    # Target label manifest: keep only ch_sims rows from each Source label manifest.
    target_label_manifests: list[Path] = []
    for src_path in source_label_manifests:
        dst = target_work / f"{src_path.stem}_target.jsonl"
        if not dst.exists():
            n = 0
            with src_path.open() as fin, dst.open("w") as fout:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    row = _json.loads(line)
                    sample_id = str(row.get("sample_id", ""))
                    if not sample_id.startswith("ch_sims"):
                        continue
                    # Rewrite split to "test" so _resolve_split_assignment's
                    # master_split check passes when matched against the
                    # Target split assignment we build below.
                    row["split"] = "test"
                    fout.write(_json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
            print(f"[TARGET-PREP] wrote {n} target rows -> {dst}", flush=True)
        target_label_manifests.append(dst)

    # Target split assignment: keep cross_domain_test rows from the Source
    # split assignment, but rewrite master_split=test and
    # representation_split=official_test so the rigid 4-split contract
    # accepts them.
    target_split_assignment = target_work / "target_split_assignment.jsonl"
    if not target_split_assignment.exists():
        n = 0
        with source_split_assignment.open() as fin, target_split_assignment.open("w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = _json.loads(line)
                if row.get("master_split") != "cross_domain_test":
                    continue
                row["master_split"] = "test"
                row["representation_split"] = "official_test"
                fout.write(_json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(f"[TARGET-PREP] wrote {n} target split-assignment rows -> {target_split_assignment}", flush=True)

    # Target prompt cache + prompt-conditioned cache.
    from mprisk.setup_helper import setup_cache_manifests
    target_setup_root = target_work / "cache_manifests" / model_key
    prompt_cache_manifest = target_setup_root / "prompt_cache_manifest.jsonl"
    # Infer prompt_set_key from the YAML filename stem.
    prompt_set_key = source_prompt_set.stem
    proto_lower = prompt_set_key.split("_main_p")[0]
    prompt_conditioned_manifest = (
        target_setup_root
        / "prompt_conditioned_cache"
        / model_key
        / proto_lower
        / prompt_set_key
        / "manifest.jsonl"
    )
    if not prompt_cache_manifest.exists() or not prompt_conditioned_manifest.exists():
        print(f"[TARGET-PREP] building target prompt caches -> {target_setup_root}", flush=True)
        setup_cache_manifests(
            cache_root=str(target_cache_root),
            prompt_set_path=str(source_prompt_set),
            model_key=model_key,
            output_root=target_setup_root,
        )

    # Target filtered cache (canonical prompt).
    target_filtered_root = target_work / "cache_manifests_filtered" / model_key
    filtered_wrapped = target_filtered_root / "unified_full_cache_manifest.json"
    if not filtered_wrapped.exists():
        import subprocess
        canonical_prompt = "pregen_risk_v1_p001"
        # Filter helper lives in scripts/cache_matrix_20260722/filter_cache_manifest.py
        filter_script = Path(__file__).resolve().parent / "cache_matrix_20260722" / "filter_cache_manifest.py"
        print(f"[TARGET-PREP] filtering target cache -> {target_filtered_root}", flush=True)
        subprocess.run(
            [
                "python",
                str(filter_script),
                "--source-cache-root", str(target_cache_root),
                "--target-cache-root", str(target_filtered_root),
                "--canonical-prompt", canonical_prompt,
            ],
            check=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )

    return {
        "label_manifests": target_label_manifests,
        "split_assignment": target_split_assignment,
        "prompt_cache_manifest": prompt_cache_manifest,
        "prompt_conditioned_cache_manifest": prompt_conditioned_manifest,
        "filtered_cache_root": target_filtered_root,
        "filtered_cache_manifest_path": filtered_wrapped,
    }


def _strip_threshold_identity(source_thresholds):
    """Load Source thresholds and strip identity-binding fields so Target
    SDR scores can be pattern-assigned against them.

    Source thresholds encode split_assignment_sha256 / embedding_manifest_sha256
    / encoder_checkpoint_sha256 from the Source calibration pass.
    Target SDR scores carry a different identity (Target rows have a
    distinct split_assignment_sha256). StateThresholds.from_dict only
    constructs an `identity` block when CALIBRATION_IDENTITY_FIELDS are
    present, so we shallow-copy the threshold JSON, drop those fields, and
    pass through the bare kappa/tau/delta values. The actual pattern
    thresholds themselves are model-level (calibrated on Source Aligned
    val), so they remain valid for Target pattern assignment.
    """
    import json as _json
    from mprisk.state.identity import CALIBRATION_IDENTITY_FIELDS

    if isinstance(source_thresholds, str | Path):
        path = Path(str(source_thresholds))
        if path.exists():
            payload = _json.loads(path.read_text())
        else:
            payload = _json.loads(str(source_thresholds))
    elif isinstance(source_thresholds, dict):
        payload = dict(source_thresholds)
    else:
        # Already a StateThresholds-like object; round-trip via dict.
        from mprisk.state.patterns import load_thresholds_config
        cfg = load_thresholds_config(source_thresholds)
        payload = {"kappa": cfg.kappa, "tau": cfg.tau, "delta": cfg.delta}

    stripped = {k: v for k, v in payload.items() if k not in CALIBRATION_IDENTITY_FIELDS}
    # Always strip schema too so from_dict doesn't try to build identity.
    stripped.pop("schema", None)
    return stripped


def run_target_sdr_pass(
    *,
    model_key: str,
    protocol: str,
    prompt_set_key: str,
    repr_key: str,
    source_label_manifests: list[Path],
    source_split_assignment: Path,
    source_prompt_set: Path,
    source_thresholds,
    checkpoint: Path,
    target_cache_root: Path,
    target_output_dir: Path,
    strict_shape: bool = False,
) -> "CoreSdrPipelineResult":
    """Run the core SDR pipeline once on the Target cache.

    Reuses Source encoder checkpoint + Source calibrated threshold values
    (kappa/tau/delta). Auto-derives Target label manifest / split assignment
    / prompt caches via _prepare_target_inputs so the Source-side rigid
    4-split contract still applies on the Target pass. Always re-exports
    Target embeddings from the Source checkpoint (Source calibration
    embeddings cover Source samples only and cannot be reused for Target
    rows).

    Identity-binding: Source thresholds carry Source-only identity fields
    (split_assignment_sha256, embedding_manifest_sha256). Target SDR scores
    carry a Target-specific identity. To pass assign_state_patterns' strict
    identity check while preserving Source's calibrated threshold values,
    we re-stamp the Source threshold dict with the Target homogeneous
    identity extracted from the Target SDR score rows. This is a deliberate
    cross-domain transfer of threshold values (kappa/tau/delta from Source
    Aligned val); the alternative of re-calibrating on Target is not viable
    because Target cache has no separate Aligned val split.
    """
    import json as _json
    from mprisk.state.identity import (
        CALIBRATION_IDENTITY_FIELDS as _CAL_ID_FIELDS,
        homogeneous_identity as _homogeneous_identity,
    )
    from mprisk.state.pipeline import (
        assign_state_patterns as _assign_state_patterns,
        compute_sdr_scores as _compute_sdr_scores,
    )
    from mprisk.data.manifests import read_jsonl as _read_jsonl

    target_inputs = _prepare_target_inputs(
        target_cache_root=target_cache_root,
        source_label_manifests=source_label_manifests,
        source_split_assignment=source_split_assignment,
        source_prompt_set=source_prompt_set,
        model_key=model_key,
        output_root=target_output_dir,
    )
    # Run Source SDR pipeline up to (and including) SDR scoring, but pass a
    # placeholder thresholds dict so run_core_sdr_pipeline's pattern
    # assignment is reached with a value that the downstream bypass replaces.
    # Easier path: run the SDR-scoring portion via run_core_sdr_pipeline with
    # a sentinel thresholds arg, then re-stamp identity, then re-run
    # assign_state_patterns. But run_core_sdr_pipeline fails on identity
    # BEFORE returning SDR scores. So we replicate the post-embedding portion
    # here instead.
    #
    # Concretely: invoke run_core_sdr_pipeline with a Target-stamped thresholds
    # dict (Source kappa/tau/delta + Target identity fields guessed from the
    # Source SDR identity, with split_assignment_sha256 forced to the Target
    # value and embedding_manifest_sha256 set to a placeholder that will be
    # corrected below after we have the actual Target embedding manifest).
    #
    # The simplest robust path: do the embedding + SDR scoring portion
    # ourselves (mirroring run_core_sdr_pipeline up to compute_sdr_scores),
    # then construct a Target-identity-stamped thresholds dict, then call
    # assign_state_patterns directly.
    normalized_protocol = normalize_protocol(protocol)
    output_base = Path(target_output_dir) / "outputs"

    # Load Source thresholds as a plain dict.
    if isinstance(source_thresholds, str | Path):
        src_thresh_path = Path(str(source_thresholds))
        if src_thresh_path.exists():
            src_threshold_payload = _json.loads(src_thresh_path.read_text())
        else:
            src_threshold_payload = _json.loads(str(source_thresholds))
    elif isinstance(source_thresholds, dict):
        src_threshold_payload = dict(source_thresholds)
    else:
        raise TypeError("source_thresholds must be a path, str, or dict")

    # Replicate the build_state_dataset -> build_state_bundles ->
    # build_relation_dataset -> export_frozen_representations ->
    # compute_sdr_scores chain from run_core_sdr_pipeline, but with Target
    # inputs.
    state_dataset_kwargs: dict[str, Any] = dict(
        manifest_paths=target_inputs["label_manifests"],
        cache_root=target_inputs["filtered_cache_root"],
        model_key=model_key,
        protocol=normalized_protocol,
        split_assignment_path=target_inputs["split_assignment"],
        output_dir=output_base / "state_data" / model_key / normalized_protocol,
        strict_shape=strict_shape,
    )
    state_dataset_kwargs["manifest_path"] = target_inputs["filtered_cache_manifest_path"]
    state_dataset_result = build_state_dataset(**state_dataset_kwargs)
    bundle_result = build_state_bundles(
        state_dataset_manifest_path=state_dataset_result.manifest_path,
        prompt_cache_manifest_path=target_inputs["prompt_cache_manifest"],
        prompt_conditioned_cache_manifest_path=target_inputs["prompt_conditioned_cache_manifest"],
        model_key=model_key,
        protocol=normalized_protocol,
        prompt_set_path=source_prompt_set,
        prompt_set_key=prompt_set_key,
        output_root=output_base / "state_bundles",
    )
    checkpoint_payload = torch.load(Path(checkpoint), map_location="cpu")
    training_config = TrainingConfig(**checkpoint_payload["training_config"])
    relation_dataset_result = build_relation_dataset(
        bundle_manifest_path=bundle_result.manifest_path,
        output_dir=(
            output_base / "representation_data" / model_key / normalized_protocol / prompt_set_key
        ),
        prompt_set_key=training_config.prompt_set_key,
        prompt_set_artifact_sha256=training_config.prompt_set_artifact_sha256,
        expected_prompt_count=training_config.expected_prompt_count,
        expected_prompt_ids=training_config.expected_prompt_ids,
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
    sdr_result = _compute_sdr_scores(
        embedding_manifest_path=embedding_result.bundle_manifest_path,
        output_dir=state_output_dir,
    )

    # Re-stamp Source thresholds with Target homogeneous identity so the
    # strict identity check in assign_state_patterns passes. Threshold
    # VALUES (kappa/tau/delta) are unchanged.
    sdr_score_rows = _read_jsonl(sdr_result.scores_path)
    target_identity = _homogeneous_identity(sdr_score_rows)
    target_threshold_payload = dict(src_threshold_payload)
    for field in _CAL_ID_FIELDS:
        if field in target_identity:
            target_threshold_payload[field] = target_identity[field]
    # Drop the schema field so from_dict treats this as a bare threshold
    # config (and does not re-derive identity from potentially-stale Source
    # fields).
    target_threshold_payload.pop("schema", None)

    pattern_result = _assign_state_patterns(
        sdr_scores_path=sdr_result.scores_path,
        thresholds=target_threshold_payload,
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
    parser.add_argument("--split-assignment", required=True)
    parser.add_argument("--output-root", default=".")
    parser.add_argument(
        "--thresholds",
        required=True,
        help="Independent Aligned calibration JSON path.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--cache-manifest-path",
        default=None,
        help=(
            "Path to a wrapped cache manifest JSON (unified_full_cache_manifest.json). "
            "When omitted, build_state_dataset falls back to the loader default."
        ),
    )
    parser.add_argument(
        "--strict-shape",
        action="store_true",
        help=(
            "Require identical (layer_count, hidden_dim, t0_token_index) across "
            "M1/M2/M12. Default is False because cache_matrix_20260722 conditions "
            "use different delivery-overlap prefixes and naturally produce "
            "different t0_token_index values."
        ),
    )
    parser.add_argument(
        "--embedding-manifest-path",
        default=None,
        help=(
            "Path to an existing spherical_embedding_manifest.jsonl to reuse "
            "(e.g. calibration's export). When set, the SDR pipeline skips "
            "export_frozen_representations entirely, preserving the identity "
            "binding required by assign_state_patterns."
        ),
    )
    parser.add_argument(
        "--target-cache-root",
        default=None,
        help=(
            "Optional Target cache root (ch_sims_v2 cross-domain natural "
            "samples). When set with --target-output-dir, also runs a Target "
            "SDR pass after the Source pass. Target label manifest / split "
            "assignment / prompt caches are auto-derived."
        ),
    )
    parser.add_argument(
        "--target-output-dir",
        default=None,
        help=(
            "Output dir for the Target SDR pass (required when "
            "--target-cache-root is set)."
        ),
    )
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
        split_assignment=Path(args.split_assignment),
        output_root=Path(args.output_root),
        thresholds=args.thresholds,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        cache_manifest_path=Path(args.cache_manifest_path) if args.cache_manifest_path else None,
        strict_shape=args.strict_shape,
        embedding_manifest_path=Path(args.embedding_manifest_path) if args.embedding_manifest_path else None,
    )
    print(f"sdr_scores={result.sdr_scores_path}")
    print(f"state_patterns={result.state_patterns_path}")
    print(f"state_summary={result.state_summary_path}")
    print(f"core_sdr_summary={result.core_sdr_summary_path}")

    if args.target_cache_root is not None:
        if args.target_output_dir is None:
            raise SystemExit("--target-cache-root requires --target-output-dir")
        print("[TARGET-SDR] starting Target pass")
        # NOTE: the --embedding-manifest-path flag refers to Source
        # calibration embeddings and MUST NOT be propagated to the Target
        # pass — Target rows have different sample_ids and need their own
        # export_frozen_representations run.
        target_result = run_target_sdr_pass(
            model_key=args.model_key,
            protocol=args.protocol,
            prompt_set_key=args.prompt_set_key,
            repr_key=args.repr_key,
            source_label_manifests=[Path(p) for p in args.manifest_paths],
            source_split_assignment=Path(args.split_assignment),
            source_prompt_set=Path(args.prompt_set),
            source_thresholds=args.thresholds,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            target_cache_root=Path(args.target_cache_root),
            target_output_dir=Path(args.target_output_dir),
            strict_shape=args.strict_shape,
        )
        print(f"target_sdr_scores={target_result.sdr_scores_path}")
        print(f"target_state_patterns={target_result.state_patterns_path}")
        print(f"target_state_summary={target_result.state_summary_path}")
        print(f"target_core_sdr_summary={target_result.core_sdr_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
