"""Strict Qwen3-VL full-prefill/Prompt-KV timing with independent generation.

The current Prompt-KV extractor returns hidden-state trajectories and does not
expose a continuation past_key_values contract.  This runner therefore never
labels a repeated model.generate call as KV generation.  It measures full
prefill, Prompt-KV prefill, and an independent deterministic generation call.
P=1 is the degenerate full-prefill baseline and has no speedup value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.cache.kv_prefill import QwenVlPromptKvPrefillExtractor
from mprisk.data.manifests import read_jsonl
from mprisk.models.qwen_omni import build_condition_request
from mprisk.models.qwen_vl import QwenVlWrapper

SCHEMA = "mprisk_qwen_vl_prefill_generation_timing_v1"
MODEL_KEY = "qwen3_vl_8b"
PROTOCOL = "vt"
CONDITIONS = ("M1", "M2", "M12")
DEFAULT_P_VALUES = (1, 2, 4, 8, 16, 32, 64)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_pool(path: Path, minimum: int) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(read_jsonl(path), start=1):
        if not bool(row.get("enabled", True)):
            continue
        prompt_id, text = row.get("prompt_id"), row.get("template_text")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"prompt row {index} has no prompt_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"prompt row {index} has no template_text")
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt_id: {prompt_id}")
        seen.add(prompt_id)
        prompts.append({"prompt_id": prompt_id, "template_text": text})
    if len(prompts) < minimum:
        raise ValueError(f"prompt pool needs {minimum} enabled rows, got {len(prompts)}")
    return prompts


def _read_samples(manifest: Path, sample_ids_file: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest)
    by_id = {
        str(row["sample_id"]): row for row in rows
        if str(row.get("protocol", "")).lower() == PROTOCOL
        and str(row.get("sample_type", "")) in {"Conflict", "Aligned"}
    }
    sample_ids = [
        line.strip() for line in sample_ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(sample_ids) != 5 or len(set(sample_ids)) != 5:
        raise ValueError("the formal timing cohort must contain exactly five unique sample IDs")
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"sample IDs are absent from manifest: {missing}")
    return [
        {
            "sample_id": str(by_id[sample_id]["sample_id"]),
            "sample_type": str(by_id[sample_id]["sample_type"]),
            "protocol": str(by_id[sample_id]["protocol"]),
            "split": str(by_id[sample_id]["split"]),
            "source_dataset": str(by_id[sample_id]["source_dataset"]),
            "media_paths": dict(by_id[sample_id]["media_paths"]),
            "text_content": by_id[sample_id].get("text_content"),
        }
        for sample_id in sample_ids
    ]


def _sync_cuda(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _requests(sample: dict[str, Any], prompts: Sequence[dict[str, str]], condition: str,
              prompt_set_key: str, video_fps: float) -> list[Any]:
    return [
        build_condition_request(
            sample_id=sample["sample_id"], model_key=MODEL_KEY, protocol=PROTOCOL,
            condition=condition, dataset_key=sample["source_dataset"], split=sample["split"],
            media_paths=sample["media_paths"], transcript=sample.get("text_content"),
            task_prompt=row["template_text"], prompt_set_key=prompt_set_key,
            prompt_id=row["prompt_id"], joint_audio_mode="embedded_video", video_fps=video_fps,
        )
        for row in prompts
    ]


def _generate_one(wrapper: Any, extractor: Any, request: Any, max_new_tokens: int, torch: Any) -> dict[str, Any]:
    model_inputs = extractor._build_model_inputs(request)
    input_ids = model_inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise RuntimeError("Qwen3-VL generation requires one input sequence")
    input_count = int(input_ids.shape[-1])
    with torch.inference_mode():
        generated = wrapper.model.generate(
            **model_inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens
        )
    if generated.ndim != 2 or int(generated.shape[0]) != 1:
        raise RuntimeError("Qwen3-VL generation returned an invalid shape")
    new_ids = generated[:, input_count:]
    if int(new_ids.shape[-1]) <= 0:
        raise RuntimeError("Qwen3-VL generation returned no new tokens")
    token_ids = [int(value) for value in new_ids[0].detach().cpu().tolist()]
    text = wrapper.processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    eos_id = getattr(getattr(wrapper.processor, "tokenizer", None), "eos_token_id", None)
    finish_reason = "eos" if eos_id is not None and token_ids[-1] == int(eos_id) else "max_new_tokens"
    return {
        "input_token_count": input_count,
        "output_token_count": len(token_ids),
        "output_text_sha256": _sha256_json({"text": text, "token_ids": token_ids}),
        "output_token_ids_sha256": _sha256_json(token_ids),
        "finish_reason": finish_reason,
    }


def _init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(
        """CREATE TABLE IF NOT EXISTS runs (
        p INTEGER NOT NULL, sample_id TEXT NOT NULL, repeat INTEGER NOT NULL,
        status TEXT NOT NULL, full_prefill_ms REAL, prompt_kv_prefill_ms REAL,
        generation_call_ms REAL, full_condition_ms_json TEXT NOT NULL,
        kv_condition_ms_json TEXT NOT NULL, generation_condition_ms_json TEXT NOT NULL,
        output_json TEXT NOT NULL, peak_gpu_bytes INTEGER, error TEXT, updated_at REAL NOT NULL,
        PRIMARY KEY(p,sample_id,repeat))"""
    )
    db.commit()
    return db


def _complete(db: sqlite3.Connection, p: int, sample_id: str, repeat: int) -> bool:
    row = db.execute(
        "SELECT status FROM runs WHERE p=? AND sample_id=? AND repeat=?",
        (p, sample_id, repeat),
    ).fetchone()
    return row is not None and row[0] == "complete"


def _failure(db: sqlite3.Connection, p: int, sample_id: str, repeat: int, error: str) -> None:
    db.execute(
        """INSERT INTO runs(p,sample_id,repeat,status,full_prefill_ms,prompt_kv_prefill_ms,
        generation_call_ms,full_condition_ms_json,kv_condition_ms_json,
        generation_condition_ms_json,output_json,peak_gpu_bytes,error,updated_at)
        VALUES(?,?,?,'failed',NULL,NULL,NULL,'{}','{}','{}','{}',NULL,?,?)
        ON CONFLICT(p,sample_id,repeat) DO UPDATE SET
        status='failed',error=excluded.error,updated_at=excluded.updated_at""",
        (p, sample_id, repeat, error, time.time()),
    )
    db.commit()


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean_ms": None, "std_ms": None, "median_ms": None, "p95_ms": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def _speedup(full_ms: float | None, kv_ms: float | None, p: int) -> float | None:
    return None if p == 1 or full_ms is None or kv_ms is None or kv_ms <= 0 else float(full_ms / kv_ms)


def _export(root: Path, metadata: dict[str, Any], db: sqlite3.Connection, p_values: Sequence[int]) -> None:
    summary: list[dict[str, Any]] = []
    for p in p_values:
        rows = db.execute(
            "SELECT full_prefill_ms,prompt_kv_prefill_ms,generation_call_ms FROM runs WHERE p=? AND status='complete'",
            (p,),
        ).fetchall()
        full, kv, generation = (_stats([float(row[i]) for row in rows]) for i in range(3))
        summary.append({
            "p": p, "completed_runs": len(rows),
            "expected_runs": metadata["selected_sample_count"] * metadata["measure_runs"],
            "full_prefill": full, "prompt_kv_prefill": kv, "generation_call": generation,
            "prefill_speedup_mean": _speedup(full["mean_ms"], kv["mean_ms"], p),
            "generation_cache_mode": "full_prompt_generate_only; KV continuation not implemented",
        })
    (root / "timing_summary.json").write_text(
        json.dumps({**metadata, "summary": summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fields = [
        "p", "completed_runs", "expected_runs",
        "full_prefill_mean_ms", "full_prefill_std_ms", "full_prefill_median_ms", "full_prefill_p95_ms",
        "prompt_kv_prefill_mean_ms", "prompt_kv_prefill_std_ms", "prompt_kv_prefill_median_ms", "prompt_kv_prefill_p95_ms",
        "generation_call_mean_ms", "generation_call_std_ms", "generation_call_median_ms", "generation_call_p95_ms",
        "prefill_speedup_mean", "generation_cache_mode",
    ]
    with (root / "timing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary:
            row = {"p": item["p"], "completed_runs": item["completed_runs"], "expected_runs": item["expected_runs"],
                   "prefill_speedup_mean": item["prefill_speedup_mean"],
                   "generation_cache_mode": item["generation_cache_mode"]}
            for name in ("full_prefill", "prompt_kv_prefill", "generation_call"):
                for key, value in item[name].items():
                    row[f"{name}_{key}"] = value
            writer.writerow(row)
    rows = db.execute(
        "SELECT p,sample_id,repeat,full_condition_ms_json,kv_condition_ms_json,generation_condition_ms_json,output_json FROM runs WHERE status='complete' ORDER BY p,sample_id,repeat"
    ).fetchall()
    (root / "timing_runs.jsonl").write_text(
        "".join(json.dumps({
            "p": row[0], "sample_id": row[1], "repeat": row[2],
            "full_condition_ms": json.loads(row[3]), "kv_condition_ms": json.loads(row[4]),
            "generation_condition_ms": json.loads(row[5]), "outputs": json.loads(row[6]),
        }, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> int:
    import torch
    if tuple(args.p_values) != tuple(sorted(set(args.p_values))):
        raise ValueError("--p-values must be strictly increasing and unique")
    if any(p <= 0 or p & (p - 1) for p in args.p_values):
        raise ValueError("all P values must be positive powers of two")
    prompts = _read_pool(args.prompt_pool.resolve(), max(args.p_values))
    samples = _read_samples(args.manifest.resolve(), args.sample_ids_file.resolve())
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    db = _init_db(root / "timing_ledger.sqlite3")
    metadata = {
        "schema": SCHEMA, "model_key": MODEL_KEY, "protocol": PROTOCOL,
        "generation_scope": "full_input_only",
        "p_values": list(args.p_values), "selected_sample_count": len(samples),
        "selected_sample_ids": [row["sample_id"] for row in samples],
        "sample_ids_sha256": _sha256_json([row["sample_id"] for row in samples]),
        "manifest": str(args.manifest.resolve()), "manifest_sha256": _sha256_file(args.manifest.resolve()),
        "prompt_pool": str(args.prompt_pool.resolve()), "prompt_pool_sha256": _sha256_file(args.prompt_pool.resolve()),
        "measure_runs": args.measure_runs, "warmup_runs": args.warmup_runs,
        "max_new_tokens": args.max_new_tokens, "video_fps": args.video_fps,
        "timing_contract": {
            "full_prefill": "wrapper.extract_prefill per prompt, CUDA synchronised",
            "prompt_kv_prefill": "QwenVlPromptKvPrefillExtractor prefix/suffix batch, CUDA synchronised",
            "generation_call": "model.generate per prompt, CUDA synchronised; independent prefill measurement",
            "kv_generation": "not reported: extractor has no continuation past_key_values contract",
        },
        "device": args.device, "dtype": args.dtype, "attn_implementation": args.attn_implementation,
    }
    (root / "provenance.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "sample_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples), encoding="utf-8"
    )
    torch.set_num_threads(args.cpu_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, torch.device(args.device))
    wrapper = QwenVlWrapper(
        model_key=MODEL_KEY, model_path=args.model_path, device=args.device,
        dtype=args.dtype, attn_implementation=args.attn_implementation,
    )
    wrapper.load()
    extractor = QwenVlPromptKvPrefillExtractor(wrapper, verbose=False, min_prefix_fraction=0.0)
    try:
        for p in args.p_values:
            prompt_rows = prompts[:p]
            prompt_set_key = f"kv_timing_p{p}_nested"
            for _ in range(args.warmup_runs):
                sample = samples[0]
                for condition in CONDITIONS:
                    requests = _requests(sample, prompt_rows, condition, prompt_set_key, args.video_fps)
                    for request in requests:
                        wrapper.extract_prefill(request)
                    extractor.extract_condition_batch(
                        sample_row=sample, build_request_fn=build_condition_request,
                        prompt_texts=[row["template_text"] for row in prompt_rows],
                        condition=condition, protocol=PROTOCOL, prompt_set_key=prompt_set_key,
                        prompt_ids=[row["prompt_id"] for row in prompt_rows],
                        common_kwargs={"joint_audio_mode": "embedded_video", "video_fps": args.video_fps},
                    )
                    for request in requests:
                        _generate_one(wrapper, extractor, request, args.max_new_tokens, torch)
                _sync_cuda(torch, args.device)
            for sample in samples:
                for repeat in range(args.measure_runs):
                    if _complete(db, p, sample["sample_id"], repeat):
                        continue
                    try:
                        full_condition: dict[str, float] = {}
                        kv_condition: dict[str, float] = {}
                        generation_condition: dict[str, float] = {}
                        output_meta: dict[str, list[dict[str, Any]]] = {}
                        _sync_cuda(torch, args.device)
                        started = time.perf_counter()
                        for condition in CONDITIONS:
                            condition_started = time.perf_counter()
                            for request in _requests(sample, prompt_rows, condition, prompt_set_key, args.video_fps):
                                wrapper.extract_prefill(request)
                            _sync_cuda(torch, args.device)
                            full_condition[condition] = (time.perf_counter() - condition_started) * 1000.0
                        full_ms = (time.perf_counter() - started) * 1000.0
                        _sync_cuda(torch, args.device)
                        started = time.perf_counter()
                        for condition in CONDITIONS:
                            condition_started = time.perf_counter()
                            extractor.extract_condition_batch(
                                sample_row=sample, build_request_fn=build_condition_request,
                                prompt_texts=[row["template_text"] for row in prompt_rows],
                                condition=condition, protocol=PROTOCOL, prompt_set_key=prompt_set_key,
                                prompt_ids=[row["prompt_id"] for row in prompt_rows],
                                common_kwargs={"joint_audio_mode": "embedded_video", "video_fps": args.video_fps},
                            )
                            _sync_cuda(torch, args.device)
                            kv_condition[condition] = (time.perf_counter() - condition_started) * 1000.0
                        kv_ms = (time.perf_counter() - started) * 1000.0
                        _sync_cuda(torch, args.device)
                        started = time.perf_counter()
                        for condition in CONDITIONS:
                            condition_started = time.perf_counter()
                            outputs = [
                                _generate_one(wrapper, extractor, request, args.max_new_tokens, torch)
                                for request in _requests(sample, prompt_rows, condition, prompt_set_key, args.video_fps)
                            ]
                            _sync_cuda(torch, args.device)
                            generation_condition[condition] = (time.perf_counter() - condition_started) * 1000.0
                            output_meta[condition] = outputs
                        generation_ms = (time.perf_counter() - started) * 1000.0
                        peak = int(torch.cuda.max_memory_allocated(torch.device(args.device))) if args.device.startswith("cuda") else None
                        db.execute(
                            """INSERT INTO runs(p,sample_id,repeat,status,full_prefill_ms,prompt_kv_prefill_ms,
                            generation_call_ms,full_condition_ms_json,kv_condition_ms_json,
                            generation_condition_ms_json,output_json,peak_gpu_bytes,error,updated_at)
                            VALUES(?,?,?,'complete',?,?,?,?,?,?,?,?,?,?)""",
                            (p, sample["sample_id"], repeat, full_ms, kv_ms, generation_ms,
                             json.dumps(full_condition, sort_keys=True), json.dumps(kv_condition, sort_keys=True),
                             json.dumps(generation_condition, sort_keys=True), json.dumps(output_meta, sort_keys=True),
                             peak, None, time.time()),
                        )
                        db.commit()
                    except Exception as exc:
                        _failure(db, p, sample["sample_id"], repeat, f"{type(exc).__name__}: {exc}")
                        raise
        _export(root, metadata, db, args.p_values)
    finally:
        db.close()
        wrapper.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-VL strict prefill and generation timing")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sample-ids-file", required=True, type=Path)
    parser.add_argument("--prompt-pool", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--p-values", nargs="+", type=int, default=list(DEFAULT_P_VALUES))
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-memory-fraction", type=float, default=0.9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup_runs < 0 or args.measure_runs <= 0 or args.max_new_tokens <= 0:
        raise ValueError("warmup-runs non-negative; measure-runs/max-new-tokens positive")
    if not 0.1 <= args.max_memory_fraction <= 0.9:
        raise ValueError("max-memory-fraction must be between 0.1 and 0.9")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
