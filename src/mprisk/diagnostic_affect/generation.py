"""Resumable model-independent Diagnostic Affect Description generation."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mprisk.assets.registry import index_assets, load_model_assets
from mprisk.models.base_wrapper import GenerationRequest, GenerationResult
from mprisk.models.wrapper_registry import get_wrapper
from mprisk.utils.io import (
    canonical_json as _canonical_json,
    sha256_file as _sha256,
)

# Public re-exports -- keep historical import path stable.
from mprisk.diagnostic_affect._ledger import DiagnosticAffectDescriptionLedger
from mprisk.diagnostic_affect._materialize import (
    _materialize,
    _read_config,
    export_diagnostic_affect_descriptions,
)
from mprisk.diagnostic_affect._plan import (
    CANONICAL_DIAGNOSTIC_AFFECT_PROMPT,
    CONFIG_SCHEMA,
    OUTPUT_SCHEMA,
    PROVENANCE_SCHEMA,
    SIGNATURE_SCHEMA,
    DiagnosticAffectDescriptionPlan,
    DiagnosticAffectDescriptionTask,
    _request_payload,
    _result_payload,
)
from mprisk.diagnostic_affect._planner import (
    _model_weight_map_sha256,
    _required_media_paths,
    _required_string,
    _select_smoke_sample_ids,
    build_diagnostic_affect_description_plan,
)
from mprisk.diagnostic_affect._verifier import (
    validate_diagnostic_affect_description,
    verify_diagnostic_affect_descriptions,
)

__all__ = [
    "CANONICAL_DIAGNOSTIC_AFFECT_PROMPT",
    "CONFIG_SCHEMA",
    "DiagnosticAffectDescriptionLedger",
    "DiagnosticAffectDescriptionPlan",
    "DiagnosticAffectDescriptionTask",
    "GenerationRequest",
    "GenerationResult",
    "OUTPUT_SCHEMA",
    "PROVENANCE_SCHEMA",
    "SIGNATURE_SCHEMA",
    "_read_config",
    "build_diagnostic_affect_description_plan",
    "build_parser",
    "export_diagnostic_affect_descriptions",
    "generate_diagnostic_affect_descriptions",
    "main",
    "validate_diagnostic_affect_description",
    "verify_diagnostic_affect_descriptions",
]


def generate_diagnostic_affect_descriptions(
    plan: DiagnosticAffectDescriptionPlan,
    *,
    output_root: Path,
    subject_model_key: str,
    model_family: str,
    model_path: Path,
    device: str,
    dtype: str,
    attn_implementation: str,
    retry_failed: bool = False,
    wrapper_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one model process serially; resume only an identical immutable plan."""
    output_root = output_root.expanduser().resolve()
    ledger = DiagnosticAffectDescriptionLedger(output_root / "batch_state.sqlite3")
    ledger.prepare(plan.signature, retry_failed=retry_failed)
    ledger.add_tasks(plan.tasks)
    ledger.validate_completed(plan.tasks)
    if plan.signature.get("subject_model_key") != subject_model_key:
        raise ValueError("subject_model_key does not match the immutable plan")
    if plan.signature.get("model_family") != model_family:
        raise ValueError("model_family does not match the immutable plan")
    factory = wrapper_factory or get_wrapper(model_family)
    wrapper = factory(
        model_key=subject_model_key,
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    try:
        wrapper.load()
        for task, attempt in ledger.pending_tasks(plan.tasks):
            started = time.perf_counter()
            try:
                result = wrapper.generate_conditioned(task.request)
                provenance = {
                    "model_path": str(model_path.expanduser().resolve()),
                    "model_config_sha256": plan.signature["model_config_sha256"],
                    "model_weight_map_sha256": plan.signature["model_weight_map_sha256"],
                    "elapsed_seconds": time.perf_counter() - started,
                    "generation": dict(result.provenance),
                }
                ledger.complete(task.task_id, attempt, result, provenance)
            except Exception as error:
                ledger.fail(task.task_id, attempt, error)
                _materialize(ledger, output_root, plan.signature)
    finally:
        wrapper.close()
        _materialize(ledger, output_root, plan.signature)
        summary = ledger.summary()
        ledger.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate strict model-independent Diagnostic Affect Descriptions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/diagnostic_affect_description.yaml"),
    )
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _read_config(args.config)
    manifest_path = args.manifest_path or Path(config["manifest_path"])
    output_root = args.output_root or Path(config["output_root"])
    asset_config = Path(config["asset_config"])
    subject_model_key = str(config["subject_model_key"])
    assets = index_assets(load_model_assets(asset_config, require_local_paths=False))
    if subject_model_key not in assets:
        raise ValueError(f"Unknown subject_model_key: {subject_model_key!r}")
    asset = assets[subject_model_key]
    model_path = Path(config["model_path"]).expanduser().resolve()
    if model_path != asset.local_model_path.expanduser().resolve():
        raise ValueError("model_path does not match subject_model_key in asset_config")
    protocol = str(config["protocol"]).upper()
    condition = str(config["condition"]).upper()
    dataset = str(config["dataset"])
    split = str(config["split"])
    if protocol.lower() not in asset.protocols:
        raise ValueError(f"Subject model {subject_model_key!r} does not support {protocol!r}")
    max_new_tokens = int(config["max_new_tokens"])
    video_fps = float(config["video_fps"])
    device = args.device or str(config["device"])
    attn_implementation = str(config["attn_implementation"])
    if args.sample_id and not args.smoke:
        raise ValueError("--sample-id is only valid with --smoke")
    selected_sample_ids = args.sample_id
    if args.smoke and not selected_sample_ids:
        selected_sample_ids = _select_smoke_sample_ids(
            manifest_path,
            dataset=dataset,
            split=split,
            protocol=protocol,
        )
    plan = build_diagnostic_affect_description_plan(
        schema_name=str(config["schema_name"]),
        run_id=str(config["run_id"]),
        manifest_path=manifest_path,
        subject_model_key=subject_model_key,
        model_family=asset.family,
        model_path=model_path,
        protocol=protocol,
        condition=condition,
        dataset=dataset,
        split=split,
        max_new_tokens=max_new_tokens,
        video_fps=video_fps,
        asset_config_sha256=_sha256(asset_config),
        config_sha256=_sha256(args.config),
        selected_sample_ids=selected_sample_ids,
    )
    summary = generate_diagnostic_affect_descriptions(
        plan,
        output_root=output_root,
        subject_model_key=subject_model_key,
        model_family=asset.family,
        model_path=model_path,
        device=device,
        dtype=str(config["dtype"]),
        attn_implementation=attn_implementation,
        retry_failed=args.retry_failed,
    )
    if summary["failed"] or summary["pending"] or summary["running"]:
        raise RuntimeError(f"Generation did not complete cleanly: {summary}")
    verification = verify_diagnostic_affect_descriptions(
        manifest_path=manifest_path,
        output_root=output_root,
        subject_model_key=subject_model_key,
        run_id=str(config["run_id"]),
        protocol=protocol,
        condition=condition,
        dataset=dataset,
        split=split,
        strict_full=not args.smoke,
    )
    print(
        _canonical_json(
            {
                "summary": summary,
                "verification": verification,
                "output_root": str(output_root.resolve()),
            }
        )
    )
    return 0
