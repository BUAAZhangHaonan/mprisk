"""Manifest-wide, resumable prefill-cache extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import string
import subprocess
import traceback
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from mprisk.assets.registry import index_assets, load_model_assets
from mprisk.cache.llava_v15_frame_plan import (
    FRAME_PLAN_SCHEMA,
    index_frame_plan,
    load_frame_plan,
)
from mprisk.cache.prefill_strategy_registry import create_prompt_kv_extractor
from mprisk.cache.prefill_writer import (
    prefill_artifact_paths,
    write_full_cache_manifest,
    write_prefill_result,
)
from mprisk.models.base_wrapper import PrefillRequest, PrefillResult
from mprisk.models.gemma4 import build_va_request as build_gemma4_va_request
from mprisk.models.qwen_omni import build_condition_request
from mprisk.models.wrapper_registry import get_wrapper
from mprisk.prompts.compiler import compile_prompt
from mprisk.prompts.template_bank import PromptTemplate, load_equiv_prompt_set

WrapperFactory = Callable[..., Any]
PromptKvExtractorFactory = Callable[..., Any]
CONDITIONS = ("M1", "M2", "M12")
FULL_PREFILL_STRATEGY = "full_prefill"
QWEN_VL_PROMPT_KV_STRATEGY = "qwen_vl_prompt_kv"
PREFILL_STRATEGY_VERSIONS = {
    FULL_PREFILL_STRATEGY: "v1",
    QWEN_VL_PROMPT_KV_STRATEGY: "v1",
}
DEFAULT_ASSET_CONFIG = Path("configs/assets/model_assets.yaml")


@dataclass(frozen=True)
class BatchTask:
    task_id: str
    sample_id: str
    prompt_set_key: str
    prompt_id: str
    prompt_text: str | None
    condition: str
    row: dict[str, Any]
    runtime_contracts: dict[str, Any]


@dataclass(frozen=True)
class BatchPlan:
    tasks: list[BatchTask]
    prompt_ids: tuple[str, ...]
    unresolved_prompt_variables: tuple[str, ...]
    rows: list[dict[str, Any]]
    signature: dict[str, Any]


@dataclass(frozen=True)
class RecoveredArtifact:
    entry: dict[str, Any]
    provenance: dict[str, Any]


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
    parser.add_argument("--frame-plan", type=Path)
    parser.add_argument("--internvl-max-num", type=int, default=1)
    parser.add_argument("--model-key", default="qwen2_5_omni_7b")
    parser.add_argument("--asset-config", default=DEFAULT_ASSET_CONFIG, type=Path)
    parser.add_argument(
        "--family",
        choices=(
            "gemma3",
            "gemma4",
            "glm4v",
            "internvl",
            "llava_onevision",
            "llava_v15",
            "minicpm_v",
            "phi3_vision",
            "phi4_multimodal",
            "qwen2_5_vl",
            "qwen3_5",
            "qwen_omni",
            "qwen_vl",
        ),
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"))
    parser.add_argument(
        "--prefill-strategy",
        default=FULL_PREFILL_STRATEGY,
        choices=tuple(PREFILL_STRATEGY_VERSIONS),
    )
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
    prompt_kv_extractor_factory: PromptKvExtractorFactory | None = None,
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
    elif args.family in {
        "gemma3",
        "gemma4",
        "glm4v",
        "llava_onevision",
        "llava_v15",
        "minicpm_v",
        "phi3_vision",
        "phi4_multimodal",
        "qwen2_5_vl",
        "qwen3_5",
        "qwen_omni",
        "qwen_vl",
    }:
        wrapper_kwargs["video_num_segments"] = args.video_num_segments
    wrapper = factory(**wrapper_kwargs)
    output_root = args.output_root.expanduser().resolve()
    ledger = BatchLedger(output_root / "batch_state.sqlite3")
    ledger.prepare(plan, retry_failed=args.retry_failed)
    for task, recorded_entry in ledger.completed_tasks(plan):
        request = _request_for_task(args, task)
        prompt_root = output_root / "prompts" / task.prompt_id
        recovered_entry = _recover_entry(
            request,
            prompt_root,
            prefill_strategy=args.prefill_strategy,
            prefill_strategy_version=_prefill_strategy_version(args.prefill_strategy),
        )
        if recovered_entry is None:
            raise FileNotFoundError(f"Completed task has no cache artifact: {task.task_id}")
        if recovered_entry.entry != recorded_entry:
            raise ValueError(f"Completed task ledger entry mismatch: {task.task_id}")
    try:
        wrapper.load()
        if args.prefill_strategy == FULL_PREFILL_STRATEGY:
            _run_full_prefill_tasks(
                args=args,
                plan=plan,
                ledger=ledger,
                wrapper=wrapper,
                output_root=output_root,
            )
        else:
            factory = prompt_kv_extractor_factory or create_prompt_kv_extractor
            extractor = factory(args.prefill_strategy, wrapper)
            _run_prompt_kv_tasks(
                args=args,
                plan=plan,
                ledger=ledger,
                extractor=extractor,
                output_root=output_root,
            )
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


def build_batch_plan(args: argparse.Namespace) -> BatchPlan:
    _resolve_runtime_asset(args)
    _validate_prefill_strategy(args)
    rows = _read_jsonl(args.manifest)
    _validate_rows(rows, args.protocol)
    prompt_set = load_equiv_prompt_set(args.prompt_set)
    if not prompt_set.active:
        raise ValueError(f"Prompt set is inactive: {prompt_set.key}")
    if prompt_set.protocol.lower() != args.protocol:
        raise ValueError("Prompt-set protocol does not match --protocol")
    templates = prompt_set.enabled_templates()
    if not templates:
        raise ValueError("Prompt set has no enabled templates")
    prompt_ids = tuple(template.prompt_id for template in templates)
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("Enabled prompt IDs must be unique")
    conditions = tuple(str(item).upper() for item in args.conditions)
    if set(conditions) != set(CONDITIONS) or len(conditions) != len(CONDITIONS):
        raise ValueError("Full extraction requires conditions M1, M2, and M12 exactly once")
    variables = _parse_variables(args.prompt_variable)
    required = set().union(*(_template_fields(template) for template in templates))
    allowed_external = required - {"sample_text"}
    extra = set(variables) - allowed_external
    if extra:
        raise ValueError(f"Unused or reserved prompt variables: {sorted(extra)}")
    unresolved = tuple(sorted(allowed_external - set(variables)))
    frame_plan: dict[str, Any] | None = None
    frame_plan_by_sample: dict[str, dict[str, Any]] = {}
    if args.family == "llava_v15":
        if args.frame_plan is None:
            raise ValueError("LLaVA-v1.5 requires --frame-plan")
        frame_plan = load_frame_plan(args.frame_plan)
        _validate_llava_frame_plan(
            frame_plan,
            args=args,
            prompt_ids=prompt_ids,
        )
        frame_plan_by_sample = index_frame_plan(frame_plan)
        missing = sorted(
            str(row["sample_id"])
            for row in rows
            if str(row["sample_id"]) not in frame_plan_by_sample
        )
        if missing:
            raise ValueError(f"Frame plan is missing samples: {missing[:5]}")
    elif args.frame_plan is not None:
        raise ValueError("--frame-plan is only valid for LLaVA-v1.5")
    tasks = []
    for row in rows:
        for template in templates:
            values = {"sample_text": str(row.get("text_content", "")), **variables}
            prompt_text = None if unresolved else compile_prompt(template, values)
            for condition in conditions:
                runtime_contracts = {}
                if frame_plan is not None:
                    entry = frame_plan_by_sample[str(row["sample_id"])]
                    runtime_contracts = {
                        "context_budget_contract": entry["context_budget_contract"],
                        "frame_selection_contract": entry["frame_selection_contract"],
                    }
                identity = {
                    "sample_id": row["sample_id"],
                    "prompt_id": template.prompt_id,
                    "condition": condition,
                    "protocol": args.protocol,
                    "model_key": args.model_key,
                    "runtime_contracts": runtime_contracts,
                }
                task_id = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
                tasks.append(
                    BatchTask(
                        task_id=task_id,
                        sample_id=str(row["sample_id"]),
                        prompt_set_key=prompt_set.key,
                        prompt_id=template.prompt_id,
                        prompt_text=prompt_text,
                        condition=condition,
                        row=row,
                        runtime_contracts=runtime_contracts,
                    )
                )
    signature = {
        "schema": "mprisk_prefill_batch_signature_v3",
        "asset_config_sha256": _sha256(args.asset_config),
        "manifest_sha256": _sha256(args.manifest),
        "prompt_set_sha256": _sha256(args.prompt_set),
        "prompt_ids": prompt_ids,
        "prompt_variables": variables,
        "protocol": args.protocol,
        "conditions": conditions,
        "model_key": args.model_key,
        "family": args.family,
        "model_path": str(args.model_path.expanduser().resolve()),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "prefill_strategy": args.prefill_strategy,
        "prefill_strategy_version": _prefill_strategy_version(args.prefill_strategy),
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "joint_audio_mode": args.joint_audio_mode,
        "video_fps": args.video_fps,
        "video_num_segments": args.video_num_segments,
        "frame_plan_schema": None if frame_plan is None else FRAME_PLAN_SCHEMA,
        "frame_plan_path": None
        if args.frame_plan is None
        else str(args.frame_plan.expanduser().resolve()),
        "frame_plan_sha256": None
        if args.frame_plan is None
        else _sha256(args.frame_plan.expanduser().resolve()),
        "internvl_max_num": args.internvl_max_num,
    }
    return BatchPlan(tasks, prompt_ids, unresolved, rows, signature)


def _validate_llava_frame_plan(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    prompt_ids: tuple[str, ...],
) -> None:
    model_path = args.model_path.expanduser().resolve()
    if payload.get("model_key") != args.model_key:
        raise ValueError("Frame-plan model key does not match --model-key")
    if payload.get("model_path") != str(model_path):
        raise ValueError("Frame-plan model path does not match --model-path")
    if payload.get("model_config_sha256") != _sha256(model_path / "config.json"):
        raise ValueError("Frame-plan model config SHA is stale")
    prompt_path = args.prompt_set.expanduser().resolve()
    if payload.get("prompt_set_sha256") != _sha256(prompt_path):
        raise ValueError("Frame-plan prompt-set SHA is stale")
    if payload.get("prompt_ids") != list(prompt_ids):
        raise ValueError("Frame-plan prompt IDs do not match the batch plan")
    if payload.get("max_candidate_frames") != args.video_num_segments:
        raise ValueError("Frame-plan max candidate frames do not match the batch")


class BatchLedger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks (
              task_id TEXT PRIMARY KEY, sample_id TEXT NOT NULL, model_key TEXT NOT NULL,
              protocol TEXT NOT NULL, prompt_set_key TEXT NOT NULL, prompt_id TEXT NOT NULL,
              condition TEXT NOT NULL, sample_type TEXT NOT NULL, use_in_main INTEGER NOT NULL,
              annotation_count INTEGER NOT NULL, split TEXT NOT NULL, source_dataset TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
              attempts INTEGER NOT NULL DEFAULT 0, error_type TEXT, error_message TEXT,
              traceback TEXT, layer_count INTEGER, hidden_dim INTEGER, token_count INTEGER,
              t0_token_index INTEGER, elapsed_seconds REAL, peak_gpu_memory_bytes INTEGER,
              checksum TEXT, entry_json TEXT
            );
            """
        )

    def prepare(self, plan: BatchPlan, *, retry_failed: bool) -> None:
        signature = _canonical_json(plan.signature)
        with self.connection:
            row = self.connection.execute(
                "SELECT value FROM metadata WHERE key='signature'"
            ).fetchone()
            if row is not None and row["value"] != signature:
                raise ValueError("Existing batch ledger signature does not match this run")
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES('signature',?)", (signature,)
            )
            self.connection.executemany(
                """INSERT OR IGNORE INTO tasks(
                   task_id,sample_id,model_key,protocol,prompt_set_key,prompt_id,
                   condition,sample_type,use_in_main,
                   annotation_count,split,source_dataset,status)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,'pending')""",
                [
                    (
                        task.task_id,
                        task.sample_id,
                        str(plan.signature["model_key"]),
                        str(plan.signature["protocol"]),
                        task.prompt_set_key,
                        task.prompt_id,
                        task.condition,
                        str(task.row.get("sample_type", "")),
                        int(bool(task.row.get("use_in_main"))),
                        int(task.row.get("annotation_count", 0)),
                        str(task.row.get("split", "")),
                        str(task.row.get("source_dataset", "")),
                    )
                    for task in plan.tasks
                ],
            )
            count = self.connection.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
            if count != len(plan.tasks):
                raise ValueError("Existing batch ledger task set does not match this run")
            self.connection.execute("UPDATE tasks SET status='pending' WHERE status='running'")
            if retry_failed:
                self.connection.execute(
                    """UPDATE tasks SET status='pending', error_type=NULL, error_message=NULL,
                       traceback=NULL WHERE status='failed'"""
                )

    def pending_tasks(self, plan: BatchPlan) -> Iterator[BatchTask]:
        by_id = {task.task_id: task for task in plan.tasks}
        rows = self.connection.execute(
            "SELECT task_id FROM tasks WHERE status='pending' ORDER BY rowid"
        ).fetchall()
        for row in rows:
            task = by_id[row["task_id"]]
            with self.connection:
                changed = self.connection.execute(
                    """UPDATE tasks SET status='running', attempts=attempts+1
                       WHERE task_id=? AND status='pending'""",
                    (task.task_id,),
                ).rowcount
            if changed == 1:
                yield task

    def pending_task_groups(
        self,
        plan: BatchPlan,
    ) -> Iterator[tuple[tuple[BatchTask, ...], tuple[BatchTask, ...]]]:
        """Claim pending tasks by complete sample-condition prompt group."""
        statuses = {
            str(row["task_id"]): str(row["status"])
            for row in self.connection.execute("SELECT task_id,status FROM tasks").fetchall()
        }
        grouped: dict[tuple[str, str], list[BatchTask]] = {}
        for task in plan.tasks:
            grouped.setdefault((task.sample_id, task.condition), []).append(task)
        for all_group_tasks in grouped.values():
            pending = [task for task in all_group_tasks if statuses[task.task_id] == "pending"]
            if not pending:
                continue
            with self.connection:
                changed = 0
                for task in pending:
                    changed += self.connection.execute(
                        """UPDATE tasks SET status='running', attempts=attempts+1
                           WHERE task_id=? AND status='pending'""",
                        (task.task_id,),
                    ).rowcount
            if changed != len(pending):
                raise RuntimeError("Could not atomically claim a complete prompt-KV task group")
            yield tuple(all_group_tasks), tuple(pending)

    def completed_tasks(self, plan: BatchPlan) -> Iterator[tuple[BatchTask, dict[str, Any]]]:
        by_id = {task.task_id: task for task in plan.tasks}
        rows = self.connection.execute(
            """SELECT task_id,entry_json FROM tasks WHERE status='completed'
               ORDER BY rowid"""
        ).fetchall()
        for row in rows:
            if row["entry_json"] is None:
                raise ValueError(f"Completed task has no ledger entry: {row['task_id']}")
            yield by_id[row["task_id"]], json.loads(row["entry_json"])

    def complete(
        self,
        task_id: str,
        entry: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE tasks SET status='completed',entry_json=?,layer_count=?,hidden_dim=?,
                   token_count=?,t0_token_index=?,elapsed_seconds=?,peak_gpu_memory_bytes=?,checksum=?,
                   error_type=NULL,error_message=NULL,traceback=NULL WHERE task_id=?""",
                (
                    _canonical_json(entry),
                    int(entry["layer_count"]),
                    int(entry["hidden_dim"]),
                    int(entry["token_count"]),
                    int(entry["t0_token_index"]),
                    provenance.get("elapsed_seconds"),
                    provenance.get("peak_gpu_memory_bytes"),
                    str(entry["checksum"]),
                    task_id,
                ),
            )

    def fail(self, task_id: str, error: Exception) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE tasks SET status='failed', error_type=?, error_message=?, traceback=?
                   WHERE task_id=?""",
                (type(error).__name__, str(error), traceback.format_exc(), task_id),
            )

    def completed_entries(self, prompt_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT entry_json FROM tasks WHERE prompt_id=? AND status='completed'
               ORDER BY rowid""",
            (prompt_id,),
        ).fetchall()
        return [json.loads(row["entry_json"]) for row in rows]

    def completed_entries_all(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT entry_json FROM tasks WHERE status='completed' ORDER BY rowid"""
        ).fetchall()
        return [json.loads(row["entry_json"]) for row in rows]

    def failures(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """SELECT task_id,sample_id,model_key,protocol,prompt_set_key,prompt_id,
               condition,sample_type,use_in_main,
               annotation_count,split,source_dataset,attempts,error_type,error_message,traceback
               FROM tasks WHERE status='failed' ORDER BY rowid"""
            ).fetchall()
        ]

    def summary(self) -> dict[str, Any]:
        counts = {
            row["status"]: row["n"]
            for row in self.connection.execute(
                "SELECT status,COUNT(*) AS n FROM tasks GROUP BY status"
            ).fetchall()
        }
        return {
            "total": sum(counts.values()),
            **{key: counts.get(key, 0) for key in ("pending", "running", "completed", "failed")},
        }

    def close(self) -> None:
        self.connection.close()


def _request_for_task(args: argparse.Namespace, task: BatchTask) -> PrefillRequest:
    if task.prompt_text is None:
        raise ValueError(f"Task {task.task_id} has an unresolved prompt")
    media = task.row["media_paths"]
    if args.family == "gemma4":
        request = build_gemma4_va_request(
            sample_id=task.sample_id,
            model_key=args.model_key,
            dataset_key=str(task.row["source_dataset"]),
            split=str(task.row["split"]),
            media_paths={str(key): str(value) for key, value in media.items()},
            text_content=(
                ""
                if task.row.get("text_content") is None
                else str(task.row["text_content"])
            ),
            task_prompt=task.prompt_text,
            condition=task.condition,
            prompt_set_key=task.prompt_set_key,
            prompt_id=task.prompt_id,
        )
    else:
        request = build_condition_request(
            sample_id=task.sample_id,
            model_key=args.model_key,
            protocol=args.protocol,
            condition=task.condition,
            dataset_key=str(task.row["source_dataset"]),
            split=str(task.row["split"]),
            media_paths={str(key): str(value) for key, value in media.items()},
            transcript=None
            if task.row.get("text_content") is None
            else str(task.row["text_content"]),
            task_prompt=task.prompt_text,
            prompt_set_key=task.prompt_set_key,
            prompt_id=task.prompt_id,
            joint_audio_mode=args.joint_audio_mode,
            video_fps=args.video_fps,
        )
    return replace(request, runtime_contracts=task.runtime_contracts)


def _run_full_prefill_tasks(
    *,
    args: argparse.Namespace,
    plan: BatchPlan,
    ledger: BatchLedger,
    wrapper: Any,
    output_root: Path,
) -> int:
    processed = 0
    version = _prefill_strategy_version(args.prefill_strategy)
    for task in ledger.pending_tasks(plan):
        request = _request_for_task(args, task)
        prompt_root = output_root / "prompts" / task.prompt_id
        try:
            recovered = _recover_entry(
                request,
                prompt_root,
                prefill_strategy=args.prefill_strategy,
                prefill_strategy_version=version,
            )
            if recovered is None:
                result = _with_prefill_identity(
                    wrapper.extract_prefill(request),
                    prefill_strategy=args.prefill_strategy,
                    prefill_strategy_version=version,
                    prefix_identity=None,
                )
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
    return processed


def _run_prompt_kv_tasks(
    *,
    args: argparse.Namespace,
    plan: BatchPlan,
    ledger: BatchLedger,
    extractor: Any,
    output_root: Path,
) -> int:
    processed = 0
    version = _prefill_strategy_version(args.prefill_strategy)
    for all_tasks, pending_tasks in ledger.pending_task_groups(plan):
        missing: list[BatchTask] = []
        for task in pending_tasks:
            request = _request_for_task(args, task)
            prompt_root = output_root / "prompts" / task.prompt_id
            try:
                recovered = _recover_entry(
                    request,
                    prompt_root,
                    prefill_strategy=args.prefill_strategy,
                    prefill_strategy_version=version,
                )
                if recovered is None:
                    missing.append(task)
                else:
                    ledger.complete(task.task_id, recovered.entry, recovered.provenance)
            except Exception as exc:
                ledger.fail(task.task_id, exc)
                if args.fail_fast:
                    raise
        if missing:
            prompt_texts = []
            prompt_ids = []
            for task in all_tasks:
                if task.prompt_text is None:
                    raise ValueError(f"Task {task.task_id} has an unresolved prompt")
                prompt_texts.append(task.prompt_text)
                prompt_ids.append(task.prompt_id)
            first = all_tasks[0]
            try:
                results = extractor.extract_condition_batch(
                    sample_row=first.row,
                    build_request_fn=build_condition_request,
                    prompt_texts=prompt_texts,
                    condition=first.condition,
                    protocol=args.protocol,
                    prompt_set_key=first.prompt_set_key,
                    prompt_ids=prompt_ids,
                    common_kwargs={
                        "joint_audio_mode": args.joint_audio_mode,
                        "video_fps": args.video_fps,
                    },
                )
                by_prompt_id = {result.request.prompt_id: result for result in results}
                if set(by_prompt_id) != set(prompt_ids):
                    raise RuntimeError("Prompt-KV results do not match the complete prompt group")
                for task in missing:
                    result = by_prompt_id[task.prompt_id]
                    _validate_prefill_result_identity(
                        result,
                        prefill_strategy=args.prefill_strategy,
                        prefill_strategy_version=version,
                    )
                    prompt_root = output_root / "prompts" / task.prompt_id
                    artifact = write_prefill_result(
                        result,
                        output_root=prompt_root,
                        update_manifest=False,
                    )
                    ledger.complete(task.task_id, artifact.entry, dict(result.provenance))
            except Exception as exc:
                for task in missing:
                    ledger.fail(task.task_id, exc)
                _materialize_failures(ledger, output_root)
                if args.fail_fast:
                    raise
        processed += len(pending_tasks)
        if processed % args.materialize_every == 0:
            _materialize_outputs(ledger, output_root, plan.prompt_ids)
    return processed


def _recover_entry(
    request: PrefillRequest,
    prompt_root: Path,
    *,
    prefill_strategy: str,
    prefill_strategy_version: str,
) -> RecoveredArtifact | None:
    paths = prefill_artifact_paths(request, output_root=prompt_root)
    existing = (paths.shard_path.is_file(), paths.sidecar_path.is_file())
    if existing == (False, False):
        return None
    if existing != (True, True):
        raise RuntimeError(f"Incomplete cache artifact pair for {request.sample_id}")
    payload = _read_json(paths.sidecar_path)
    if payload.get("schema") != "mprisk_prefill_cache_sidecar_v1":
        raise ValueError(f"Unsupported sidecar schema: {paths.sidecar_path}")
    expected_request = {
        "sample_id": request.sample_id,
        "model_key": request.model_key,
        "protocol": request.protocol,
        "condition": request.condition,
        "prompt_set_key": request.prompt_set_key,
        "prompt_id": request.prompt_id,
        "dataset_key": request.dataset_key,
        "split": request.split,
        "messages": list(request.messages),
        "media_paths": dict(request.media_paths),
        "use_audio_in_video": request.use_audio_in_video,
        "runtime_contracts": dict(request.runtime_contracts),
    }
    if payload.get("request") != expected_request:
        raise ValueError(f"Existing sidecar request mismatch: {paths.sidecar_path}")
    entry = payload.get("entry")
    if not isinstance(entry, dict) or entry.get("checksum") != _sha256(paths.shard_path):
        raise ValueError(f"Existing cache checksum mismatch: {paths.shard_path}")
    tensors = load_file(paths.shard_path)
    hidden = tensors.get("hidden_states")
    if hidden is None or list(hidden.shape) != [entry.get("layer_count"), entry.get("hidden_dim")]:
        raise ValueError(f"Existing cache tensor shape mismatch: {paths.shard_path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Existing sidecar provenance is invalid: {paths.sidecar_path}")
    _validate_prefill_provenance(
        provenance,
        prefill_strategy=prefill_strategy,
        prefill_strategy_version=prefill_strategy_version,
    )
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Existing cache metadata is invalid: {paths.sidecar_path}")
    for key in ("prefill_strategy", "prefill_strategy_version", "prefix_identity"):
        if metadata.get(key) != provenance.get(key):
            raise ValueError(f"Existing cache {key} mismatch: {paths.sidecar_path}")
    return RecoveredArtifact(entry=entry, provenance=provenance)


def _with_prefill_identity(
    result: PrefillResult,
    *,
    prefill_strategy: str,
    prefill_strategy_version: str,
    prefix_identity: str | None,
) -> PrefillResult:
    provenance = dict(result.provenance)
    provenance.update(
        {
            "prefill_strategy": prefill_strategy,
            "prefill_strategy_version": prefill_strategy_version,
            "prefix_identity": prefix_identity,
        }
    )
    _validate_prefill_provenance(
        provenance,
        prefill_strategy=prefill_strategy,
        prefill_strategy_version=prefill_strategy_version,
    )
    return PrefillResult(
        request=result.request,
        trajectory=result.trajectory,
        token_count=result.token_count,
        t0_token_index=result.t0_token_index,
        provenance=provenance,
    )


def _validate_prefill_result_identity(
    result: PrefillResult,
    *,
    prefill_strategy: str,
    prefill_strategy_version: str,
) -> None:
    _validate_prefill_provenance(
        dict(result.provenance),
        prefill_strategy=prefill_strategy,
        prefill_strategy_version=prefill_strategy_version,
    )


def _validate_prefill_provenance(
    provenance: dict[str, Any],
    *,
    prefill_strategy: str,
    prefill_strategy_version: str,
) -> None:
    if provenance.get("prefill_strategy") != prefill_strategy:
        raise ValueError("Prefill result strategy does not match the configured strategy")
    if provenance.get("prefill_strategy_version") != prefill_strategy_version:
        raise ValueError("Prefill result strategy version does not match the configured version")
    prefix_identity = provenance.get("prefix_identity")
    if prefill_strategy == QWEN_VL_PROMPT_KV_STRATEGY:
        if (
            not isinstance(prefix_identity, str)
            or len(prefix_identity) != 64
            or any(character not in string.hexdigits for character in prefix_identity)
        ):
            raise ValueError("Prompt-KV prefill requires a SHA-256 prefix_identity")
    elif prefix_identity is not None:
        raise ValueError("Full-prefill provenance must not define a prefix_identity")


def _prefill_strategy_version(strategy: str) -> str:
    try:
        return PREFILL_STRATEGY_VERSIONS[strategy]
    except KeyError as exc:
        raise ValueError(f"Unsupported prefill strategy: {strategy!r}") from exc


def _validate_prefill_strategy(args: argparse.Namespace) -> None:
    if args.prefill_strategy == QWEN_VL_PROMPT_KV_STRATEGY:
        if args.family != "qwen_vl":
            raise ValueError(
                f"Prefill strategy {QWEN_VL_PROMPT_KV_STRATEGY!r} requires family "
                f"'qwen_vl', got {args.family!r}"
            )
        if args.protocol != "vt":
            raise ValueError(
                f"Prefill strategy {QWEN_VL_PROMPT_KV_STRATEGY!r} requires protocol 'vt', "
                f"got {args.protocol!r}"
            )


def _materialize_outputs(ledger: BatchLedger, root: Path, prompt_ids: Sequence[str]) -> None:
    for prompt_id in prompt_ids:
        manifest = root / "prompts" / prompt_id / "manifests" / "unified_full_cache_manifest.json"
        write_full_cache_manifest(ledger.completed_entries(prompt_id), manifest)
    entries = ledger.completed_entries_all()
    _atomic_text(root / "manifest.jsonl", "".join(_canonical_json(row) + "\n" for row in entries))
    _materialize_failures(ledger, root)
    _atomic_json(root / "batch_summary.json", ledger.summary())


def rematerialize_completed_batch(output_root: str | Path) -> dict[str, Any]:
    """Rebuild all derived batch manifests from a completed SQLite ledger.

    This operation never constructs a model wrapper or reads model weights. The
    ledger remains the source of truth, and every derived output is replaced
    atomically by :func:`_materialize_outputs`.
    """
    root = Path(output_root).expanduser().resolve()
    ledger_path = root / "batch_state.sqlite3"
    if not ledger_path.is_file():
        raise FileNotFoundError(f"Missing prefill batch ledger: {ledger_path}")

    ledger = BatchLedger(ledger_path)
    try:
        summary = ledger.summary()
        total = int(summary["total"])
        if total == 0:
            raise ValueError("Cannot rematerialize an empty prefill batch ledger")
        incomplete = {
            key: int(summary[key]) for key in ("pending", "running", "failed") if summary[key]
        }
        if incomplete or int(summary["completed"]) != total:
            raise ValueError(f"Prefill batch ledger is not complete: {summary}")

        missing_entries = int(
            ledger.connection.execute(
                "SELECT COUNT(*) FROM tasks WHERE status='completed' AND entry_json IS NULL"
            ).fetchone()[0]
        )
        if missing_entries:
            raise ValueError(
                f"Completed prefill batch tasks missing entry_json: {missing_entries}"
            )

        prompt_ids = tuple(
            str(row[0])
            for row in ledger.connection.execute(
                "SELECT prompt_id FROM tasks GROUP BY prompt_id ORDER BY MIN(rowid)"
            ).fetchall()
        )
        if not prompt_ids:
            raise ValueError("Completed prefill batch ledger contains no prompt IDs")

        _materialize_outputs(ledger, root, prompt_ids)
        return {
            "schema": "mprisk_prefill_batch_rematerialization_v1",
            "status": "complete",
            "output_root": str(root),
            "prompt_ids": list(prompt_ids),
            "manifest_rows": total,
            "summary": summary,
        }
    finally:
        ledger.close()


def _materialize_failures(ledger: BatchLedger, root: Path) -> None:
    lines = "".join(_canonical_json(row) + "\n" for row in ledger.failures())
    _atomic_text(root / "failures.jsonl", lines)


def _dry_run_payload(args: argparse.Namespace, plan: BatchPlan) -> dict[str, Any]:
    rows = plan.rows
    payload: dict[str, Any] = {
        "status": "dry_run",
        "ready": not plan.unresolved_prompt_variables,
        "unresolved_prompt_variables": plan.unresolved_prompt_variables,
        "sample_count": len(rows),
        "prompt_count": len(plan.prompt_ids),
        "prompt_ids": plan.prompt_ids,
        "condition_count": len(CONDITIONS),
        "conditions": CONDITIONS,
        "task_count": len(plan.tasks),
        "sample_type_counts": dict(Counter(str(row.get("sample_type")) for row in rows)),
        "use_in_main_counts": dict(
            Counter(str(bool(row.get("use_in_main"))).lower() for row in rows)
        ),
        "annotation_count_counts": dict(Counter(str(row.get("annotation_count")) for row in rows)),
        "split_counts": dict(Counter(str(row.get("split")) for row in rows)),
        "source_dataset_counts": dict(Counter(str(row.get("source_dataset")) for row in rows)),
        "writes_performed": 0,
    }
    durations = _probe_durations(rows, args.ffprobe_workers) if args.probe_media else None
    if durations is not None:
        payload["media_duration_seconds"] = _duration_summary(durations)
    smoke = _parse_condition_seconds(args.smoke_condition_seconds)
    if smoke:
        if set(smoke) != set(CONDITIONS):
            raise ValueError("Smoke timing requires M1, M2, and M12 values")
        triplet = sum(smoke.values())
        overhead = max(0.0, (args.smoke_wall_seconds or triplet) - triplet)
        payload["gpu_time_estimate"] = {
            "basis_condition_seconds": smoke,
            "model_load_overhead_seconds": overhead,
            "constant_sample_total_seconds": triplet * len(rows) * len(plan.prompt_ids) + overhead,
        }
        if durations is not None and args.smoke_media_seconds:
            payload["gpu_time_estimate"]["linear_duration_total_seconds"] = (
                triplet * sum(durations) / args.smoke_media_seconds * len(plan.prompt_ids)
                + overhead
            )
    if args.trajectory_shape:
        layers, hidden = args.trajectory_shape
        if layers <= 0 or hidden <= 0:
            raise ValueError("--trajectory-shape values must be positive")
        raw_per_task = layers * hidden * 4
        payload["storage_estimate"] = {
            "trajectory_bytes_per_task": raw_per_task,
            "trajectory_total_bytes": raw_per_task * len(plan.tasks),
        }
        if args.smoke_artifact_bytes_per_task is not None:
            payload["storage_estimate"]["smoke_artifact_total_bytes"] = (
                args.smoke_artifact_bytes_per_task * len(plan.tasks)
            )
    if args.gpu_index is not None:
        payload["gpu"] = _gpu_status(args.gpu_index)
    return payload


def _validate_rows(rows: list[dict[str, Any]], protocol: str) -> None:
    if not rows:
        raise ValueError("Input manifest is empty")
    seen = set()
    required = {"sample_id", "protocol", "media_paths", "source_dataset", "split"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Manifest row is missing fields: {sorted(missing)}")
        if str(row["protocol"]).lower() != protocol:
            raise ValueError(f"Manifest contains non-{protocol} row: {row['sample_id']}")
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Manifest contains duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if not isinstance(row["media_paths"], dict):
            raise ValueError(f"Manifest row has invalid media_paths: {sample_id}")
        if str(row.get("sample_type", "")).lower() == "misread":
            raise ValueError(f"Prefill extraction must not process Misread rows: {sample_id}")


def _validate_media(rows: list[dict[str, Any]]) -> None:
    missing = sorted(
        {
            str(path)
            for row in rows
            for path in row["media_paths"].values()
            if not Path(str(path)).is_file()
        }
    )
    if missing:
        raise FileNotFoundError(f"Manifest references missing media files: {missing[:10]}")


def _resolve_runtime_asset(args: argparse.Namespace) -> None:
    assets = index_assets(load_model_assets(args.asset_config))
    asset = assets.get(args.model_key)
    if asset is None:
        raise KeyError(f"Model key is absent from asset config: {args.model_key}")
    if args.family is not None and args.family != asset.family:
        raise ValueError(
            f"Configured family for {args.model_key} is {asset.family!r}, not {args.family!r}"
        )
    args.family = asset.family
    if args.model_path is None:
        args.model_path = asset.local_model_path
    if args.attn_implementation is None:
        eager_only_families = {"internvl", "phi3_vision"}
        args.attn_implementation = (
            "eager" if asset.family in eager_only_families else "sdpa"
        )


def _parse_variables(items: Sequence[str]) -> dict[str, str]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Prompt variable must be NAME=VALUE: {item!r}")
        key, value = item.split("=", 1)
        if not key or key in result:
            raise ValueError(f"Invalid or duplicate prompt variable: {key!r}")
        result[key] = value
    return result


def _template_fields(template: PromptTemplate) -> set[str]:
    return {field for _, field, _, _ in string.Formatter().parse(template.template_text) if field}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Manifest line {number} must be a JSON object")
            rows.append(value)
    return rows


def _probe_durations(rows: list[dict[str, Any]], workers: int) -> list[float]:
    if workers <= 0:
        raise ValueError("--ffprobe-workers must be positive")
    paths = sorted({str(path) for row in rows for path in row["media_paths"].values()})

    def probe(path: str) -> float:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        value = float(completed.stdout.strip())
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid media duration for {path}: {value}")
        return value

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(probe, paths))


def _duration_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total": float(sum(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(max(values)),
    }


def _gpu_status(index: int) -> dict[str, Any]:
    query = "index,name,memory.used,memory.total,utilization.gpu"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", str(index)],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    used, total, utilization = int(values[2]), int(values[3]), int(values[4])
    return {
        "index": int(values[0]),
        "name": values[1],
        "memory_used_mib": used,
        "memory_total_mib": total,
        "memory_fraction": used / total,
        "utilization_percent": utilization,
        "under_90_percent": used / total < 0.9 and utilization < 90,
    }


def _parse_condition_seconds(items: Sequence[str]) -> dict[str, float]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Smoke timing must be CONDITION=SECONDS: {item!r}")
        key, raw = item.split("=", 1)
        key = key.upper()
        if key in parsed or key not in CONDITIONS:
            raise ValueError(f"Invalid or duplicate smoke condition: {key}")
        value = float(raw)
        if value <= 0:
            raise ValueError("Smoke condition seconds must be positive")
        parsed[key] = value
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
