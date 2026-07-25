"""Manifest-wide, resumable prefill-cache extraction.

Public API entry point. Plan construction, the SQLite ledger, and the
per-task runner have been split into sibling modules under
``mprisk.cache._batch_*``; this module keeps the ``main``/``build_parser``
CLI surface and re-exports every previously-public symbol so existing
importers and monkey-patches continue to resolve against
``mprisk.cache.prefill_batch``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mprisk.cache._batch_ledger import BatchLedger
from mprisk.cache._batch_plan import (
    CONDITIONS,
    DEFAULT_ASSET_CONFIG,
    WrapperFactory,
    BatchPlan,
    BatchTask,
    RecoveredArtifact,
)
from mprisk.cache._batch_planner import (
    _parse_variables as _parse_variables,
    _resolve_runtime_asset as _resolve_runtime_asset,
    _template_fields as _template_fields,
    _validate_media as _validate_media,
    _validate_rows as _validate_rows,
    build_batch_plan,
)
from mprisk.cache._batch_runner import (
    _dry_run_payload as _dry_run_payload,
    _duration_summary as _duration_summary,
    _gpu_status as _gpu_status,
    _materialize_failures as _materialize_failures,
    _materialize_outputs as _materialize_outputs,
    _parse_condition_seconds as _parse_condition_seconds,
    _probe_durations as _probe_durations,
    _recover_entry as _recover_entry,
    _request_for_task as _request_for_task,
)
from mprisk.cache.prefill_writer import write_prefill_result
from mprisk.models.wrapper_registry import get_wrapper
from mprisk.utils.io import read_json_object as _read_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable manifest-wide prefill extraction.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prompt-set", required=True, type=Path)
    parser.add_argument("--prompt-variable", action="append", default=[])
    parser.add_argument("--protocol", default="va", choices=("vt", "va", "vta"))
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument(
        "--joint-audio-mode", default="embedded_video", choices=("embedded_video", "separate_file")
    )
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--video-num-segments", type=int, default=8)
    parser.add_argument("--internvl-max-num", type=int, default=1)
    parser.add_argument("--model-key", default="qwen2_5_omni_7b")
    parser.add_argument("--asset-config", default=DEFAULT_ASSET_CONFIG, type=Path)
    parser.add_argument("--family", choices=("qwen_omni", "qwen_vl", "qwen3_5", "internvl", "gemma_4"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16",))
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"))
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--materialize-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-media", action="store_true")
    parser.add_argument("--ffprobe-workers", type=int, default=16)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--trajectory-shape", nargs=2, type=int, metavar=("LAYERS", "HIDDEN"))
    parser.add_argument("--smoke-condition-seconds", action="append", default=[])
    parser.add_argument("--smoke-wall-seconds", type=float)
    parser.add_argument("--smoke-media-seconds", type=float)
    parser.add_argument("--smoke-artifact-bytes-per-task", type=float)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    wrapper_factory: WrapperFactory | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.materialize_every <= 0:
        raise ValueError("--materialize-every must be positive")
    plan = build_batch_plan(args)
    if args.dry_run:
        print(json.dumps(_dry_run_payload(args, plan), ensure_ascii=False, sort_keys=True))
        return 0
    if plan.unresolved_prompt_variables:
        raise ValueError(
            "Unresolved prompt variables: " + ", ".join(plan.unresolved_prompt_variables)
        )
    _validate_media(plan.rows)
    factory = wrapper_factory or get_wrapper(args.family)
    wrapper_kwargs = {
        "model_key": args.model_key,
        "model_path": args.model_path,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
    }
    if args.family == "internvl":
        wrapper_kwargs.update(
            video_num_segments=args.video_num_segments,
            internvl_max_num=args.internvl_max_num,
        )
    wrapper = factory(**wrapper_kwargs)
    output_root = args.output_root.expanduser().resolve()
    ledger = BatchLedger(output_root / "batch_state.sqlite3")
    ledger.prepare(plan, retry_failed=args.retry_failed)
    for task, recorded_entry in ledger.completed_tasks(plan):
        request = _request_for_task(args, task)
        prompt_root = output_root / "prompts" / task.prompt_id
        recovered_entry = _recover_entry(request, prompt_root)
        if recovered_entry is None:
            raise FileNotFoundError(f"Completed task has no cache artifact: {task.task_id}")
        if recovered_entry.entry != recorded_entry:
            raise ValueError(f"Completed task ledger entry mismatch: {task.task_id}")
    processed = 0
    try:
        wrapper.load()
        for task in ledger.pending_tasks(plan):
            request = _request_for_task(args, task)
            prompt_root = output_root / "prompts" / task.prompt_id
            try:
                recovered = _recover_entry(request, prompt_root)
                if recovered is None:
                    result = wrapper.extract_prefill(request)
                    artifact = write_prefill_result(
                        result,
                        output_root=prompt_root,
                        update_manifest=False,
                    )
                    entry = artifact.entry
                    provenance = dict(result.provenance)
                else:
                    entry = recovered.entry
                    provenance = recovered.provenance
                ledger.complete(task.task_id, entry, provenance)
            except Exception as exc:
                ledger.fail(task.task_id, exc)
                _materialize_failures(ledger, output_root)
                if args.fail_fast:
                    raise
            processed += 1
            if processed % args.materialize_every == 0:
                _materialize_outputs(ledger, output_root, plan.prompt_ids)
    finally:
        wrapper.close()
        _materialize_outputs(ledger, output_root, plan.prompt_ids)
        ledger.close()
    print(
        json.dumps(
            {"status": "ok", "summary": _read_json(output_root / "batch_summary.json")},
            ensure_ascii=False,
        )
    )
    return 0


__all__ = [
    "BatchLedger",
    "BatchPlan",
    "BatchTask",
    "CONDITIONS",
    "DEFAULT_ASSET_CONFIG",
    "RecoveredArtifact",
    "WrapperFactory",
    "build_batch_plan",
    "build_parser",
    "main",
]
