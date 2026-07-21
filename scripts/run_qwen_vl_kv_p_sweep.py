"""Qwen3-VL prompt-prefix KV-cache P sweep.

The sweep keeps one immutable sample cohort and one ordered prefix of the
reviewed 128-prompt pool for each P.  It measures the model-native
``qwen_vl_prompt_kv`` extractor and records enough provenance to resume without
touching the source manifest or prompt bank.  The P=1 point is explicit: there
is no reusable prefix when only one prompt is requested, so it uses the
validated full-prefill path and is labelled as the degenerate KV baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.assets.registry import index_assets, load_model_assets
from mprisk.cache.kv_prefill import QwenVlPromptKvPrefillExtractor
from mprisk.data.manifests import read_jsonl
from mprisk.models.qwen_omni import build_condition_request
from mprisk.models.wrapper_registry import get_wrapper
from mprisk.state.spherical import compute_spherical_state

SCHEMA = "mprisk_qwen_vl_kv_p_sweep_v1"
MODEL_KEY = "qwen3_vl_8b"
PROTOCOL = "vt"
CONDITIONS = ("M1", "M2", "M12")
DEFAULT_P_VALUES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_POOL = Path("data/processed/prompt_banks/pregen_risk_v1_agent/pool128.jsonl")
DEFAULT_ASSET_CONFIG = Path("configs/assets/model_assets.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_pool(path: Path, *, minimum: int = 64) -> list[dict[str, str]]:
    rows = read_jsonl(path)
    prompts: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not bool(row.get("enabled", True)):
            continue
        prompt_id = row.get("prompt_id")
        text = row.get("template_text")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"prompt pool row {index} has no prompt_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"prompt pool row {index} has no template_text")
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt_id in pool: {prompt_id}")
        seen.add(prompt_id)
        prompts.append({"prompt_id": prompt_id, "template_text": text})
    if len(prompts) < minimum:
        raise ValueError(f"prompt pool needs at least {minimum} enabled rows, got {len(prompts)}")
    return prompts


def _read_sample_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("sample-id file must contain unique non-empty IDs")
    return ids


def _select_samples(
    manifest: Path,
    *,
    sample_count: int,
    sample_ids_file: Path | None,
) -> tuple[list[dict[str, Any]], str]:
    rows = read_jsonl(manifest)
    candidates = [
        row
        for row in rows
        if str(row.get("protocol", "")).lower() == PROTOCOL
        and str(row.get("sample_type", "")) in {"Conflict", "Aligned"}
    ]
    by_id = {str(row.get("sample_id")): row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("manifest contains duplicate sample_id values")
    if sample_ids_file is not None:
        requested = _read_sample_ids(sample_ids_file)
        missing = [sample_id for sample_id in requested if sample_id not in by_id]
        if missing:
            raise ValueError(f"sample IDs absent from manifest: {missing[:5]}")
        selected = [by_id[sample_id] for sample_id in requested]
    else:
        selected = sorted(candidates, key=lambda row: str(row["sample_id"]))
        if sample_count > 0:
            selected = selected[:sample_count]
    if not selected:
        raise ValueError("selected sample cohort is empty")
    cohort = [
        {
            "sample_id": str(row["sample_id"]),
            "sample_type": str(row["sample_type"]),
            "protocol": str(row.get("protocol", "")),
            "split": str(row.get("split", "")),
            "source_dataset": str(row.get("source_dataset", "")),
            "media_paths": dict(row.get("media_paths", {})),
            "text_content": row.get("text_content"),
        }
        for row in selected
    ]
    return cohort, _json_hash([row["sample_id"] for row in cohort])


def _init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            p INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            repeat INTEGER NOT NULL,
            status TEXT NOT NULL,
            elapsed_ms REAL,
            condition_elapsed_json TEXT,
            prompt_count INTEGER NOT NULL,
            cache_mode TEXT NOT NULL,
            peak_gpu_bytes INTEGER,
            error TEXT,
            updated_at REAL NOT NULL,
            PRIMARY KEY (p, sample_id, repeat)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS state_metrics (
            p INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (p, sample_id)
        )
        """
    )
    connection.commit()
    return connection


def _record_failure(
    connection: sqlite3.Connection,
    *,
    p: int,
    sample_id: str,
    repeat: int,
    prompt_count: int,
    cache_mode: str,
    error: str,
) -> None:
    connection.execute(
        """
        INSERT INTO runs(p, sample_id, repeat, status, elapsed_ms, condition_elapsed_json,
                         prompt_count, cache_mode, peak_gpu_bytes, error, updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(p,sample_id,repeat) DO UPDATE SET
          status=excluded.status, elapsed_ms=excluded.elapsed_ms,
          condition_elapsed_json=excluded.condition_elapsed_json,
          prompt_count=excluded.prompt_count, cache_mode=excluded.cache_mode,
          peak_gpu_bytes=excluded.peak_gpu_bytes, error=excluded.error,
          updated_at=excluded.updated_at
        """,
        (p, sample_id, repeat, "failed", None, None, prompt_count, cache_mode, None, error, time.time()),
    )
    connection.commit()


def _is_complete(connection: sqlite3.Connection, p: int, sample_id: str, repeat: int) -> bool:
    row = connection.execute(
        "SELECT status FROM runs WHERE p=? AND sample_id=? AND repeat=?", (p, sample_id, repeat)
    ).fetchone()
    if row is None or row[0] != "complete":
        return False
    if repeat == 0:
        metric = connection.execute(
            "SELECT 1 FROM state_metrics WHERE p=? AND sample_id=?", (p, sample_id)
        ).fetchone()
        return metric is not None
    return True


def _sync_cuda(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _extract(
    *,
    wrapper: Any,
    extractor: QwenVlPromptKvPrefillExtractor,
    sample: dict[str, Any],
    prompt_rows: Sequence[dict[str, str]],
    condition: str,
    prompt_set_key: str,
    video_fps: float,
    joint_audio_mode: str,
) -> tuple[list[Any], str]:
    prompt_texts = [row["template_text"] for row in prompt_rows]
    prompt_ids = [row["prompt_id"] for row in prompt_rows]
    kwargs = {
        "joint_audio_mode": joint_audio_mode,
        "video_fps": video_fps,
    }
    if len(prompt_rows) == 1:
        request = build_condition_request(
            sample_id=sample["sample_id"],
            model_key=wrapper.model_key,
            protocol=PROTOCOL,
            condition=condition,
            dataset_key=sample["source_dataset"],
            split=sample["split"],
            media_paths=sample["media_paths"],
            transcript=sample.get("text_content"),
            task_prompt=prompt_texts[0],
            prompt_set_key=prompt_set_key,
            prompt_id=prompt_ids[0],
            **kwargs,
        )
        return [wrapper.extract_prefill(request)], "degenerate_single_prompt_full_prefill"
    return (
        extractor.extract_condition_batch(
            sample_row=sample,
            build_request_fn=build_condition_request,
            prompt_texts=prompt_texts,
            condition=condition,
            protocol=PROTOCOL,
            prompt_set_key=prompt_set_key,
            prompt_ids=prompt_ids,
            common_kwargs=kwargs,
        ),
        "qwen_vl_prompt_kv",
    )


def _state_payload(
    *,
    sample: dict[str, Any],
    p: int,
    prompt_rows: Sequence[dict[str, str]],
    results_by_condition: dict[str, list[Any]],
) -> dict[str, Any]:
    embeddings: dict[str, dict[str, list[float]]] = {}
    for condition in CONDITIONS:
        results = results_by_condition[condition]
        if len(results) != p:
            raise RuntimeError(f"{condition} returned {len(results)} results for P={p}")
        vectors: dict[str, list[float]] = {}
        for prompt, result in zip(prompt_rows, results, strict=True):
            vector = result.trajectory[-1].astype(np.float64)
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-12:
                raise ValueError(
                    f"zero norm in raw last-layer trajectory for {sample['sample_id']} "
                    f"{condition}/{prompt['prompt_id']}"
                )
            vectors[prompt["prompt_id"]] = (vector / norm).tolist()
        embeddings[condition] = vectors
    state = compute_spherical_state(
        {
            "sample_id": sample["sample_id"],
            "sample_type": sample["sample_type"],
            "embeddings": embeddings,
        }
    )
    return {
        "schema": "mprisk_qwen_vl_kv_p_state_metrics_v1",
        "sample_id": sample["sample_id"],
        "sample_type": sample["sample_type"],
        "p": p,
        "prompt_ids": [row["prompt_id"] for row in prompt_rows],
        "repr_key": "raw_trajectory_last_layer_unit",
        "state": state,
    }


def _aggregate(connection: sqlite3.Connection, p: int) -> dict[str, Any]:
    rows = connection.execute(
        "SELECT elapsed_ms, condition_elapsed_json, peak_gpu_bytes FROM runs "
        "WHERE p=? AND status='complete' ORDER BY sample_id, repeat",
        (p,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"No completed timing rows for P={p}")
    elapsed = np.asarray([float(row[0]) for row in rows], dtype=np.float64)
    condition_values: dict[str, list[float]] = {condition: [] for condition in CONDITIONS}
    for row in rows:
        values = json.loads(row[1])
        for condition in CONDITIONS:
            condition_values[condition].append(float(values[condition]))
    metric_rows = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT payload_json FROM state_metrics WHERE p=? ORDER BY sample_id", (p,)
        )
    ]
    return {
        "p": p,
        "sample_count": len({
            str(row[0]) for row in connection.execute("SELECT sample_id FROM runs WHERE p=?", (p,))
        }),
        "timing": {
            "measured_runs": len(rows),
            "total_mean_ms": float(elapsed.mean()),
            "total_median_ms": float(np.median(elapsed)),
            "total_p95_ms": float(np.percentile(elapsed, 95)),
            "per_prompt_median_ms": float(np.median(elapsed) / p),
            "condition_mean_ms": {
                condition: float(np.mean(values)) for condition, values in condition_values.items()
            },
            "peak_gpu_bytes_max": max(
                int(row[2]) for row in rows if row[2] is not None
            ) if any(row[2] is not None for row in rows) else None,
        },
        "state_metrics": {
            "repr_key": "raw_trajectory_last_layer_unit",
            "sample_count": len(metric_rows),
            "S_mean_mean": float(np.mean([row["state"]["S_mean"] for row in metric_rows]))
            if metric_rows else None,
            "D_mean": float(np.mean([row["state"]["D"] for row in metric_rows]))
            if metric_rows else None,
            "R_mean": float(np.mean([row["state"]["R"] for row in metric_rows]))
            if metric_rows else None,
            "R_bootstrap_se_mean": float(
                np.mean([row["state"]["R_bootstrap_se"] for row in metric_rows])
            ) if metric_rows else None,
        },
    }


def _add_convergence(summary: list[dict[str, Any]], connection: sqlite3.Connection, reference_p: int) -> None:
    reference = {
        str(row[0]): json.loads(row[1])
        for row in connection.execute(
            "SELECT sample_id, payload_json FROM state_metrics WHERE p=?", (reference_p,)
        )
    }
    if not reference:
        raise RuntimeError(f"P={reference_p} has no state metrics for convergence reference")
    for item in summary:
        p = int(item["p"])
        current = {
            str(row[0]): json.loads(row[1])
            for row in connection.execute(
                "SELECT sample_id, payload_json FROM state_metrics WHERE p=?", (p,)
            )
        }
        shared = sorted(set(reference) & set(current))
        if not shared:
            raise RuntimeError(f"P={p} has no samples shared with P={reference_p}")
        errors = {
            key: float(
                np.mean(
                    [
                        abs(float(current[sample]["state"][key]) - float(reference[sample]["state"][key]))
                        for sample in shared
                    ]
                )
            )
            for key in ("S_mean", "D", "R")
        }
        item["stability"] = {
            "definition": "absolute state-index error against nested P=64 reference on shared samples",
            "reference_p": reference_p,
            "shared_sample_count": len(shared),
            "state_index_mae_vs_reference": errors,
            "state_index_mae_mean": float(np.mean(list(errors.values()))),
        }


def _write_exports(
    root: Path,
    *,
    metadata: dict[str, Any],
    summary: list[dict[str, Any]],
    connection: sqlite3.Connection,
) -> None:
    (root / "sweep_summary.json").write_text(
        json.dumps({**metadata, "summary": summary}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (root / "latency_stability.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "p", "sample_count", "measured_runs", "total_mean_ms", "total_median_ms",
            "total_p95_ms", "per_prompt_median_ms", "S_mean_mean", "D_mean", "R_mean",
            "R_bootstrap_se_mean", "state_index_mae_mean", "state_index_mae_S_mean",
            "state_index_mae_D", "state_index_mae_R", "reference_p",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary:
            timing = item["timing"]
            state = item["state_metrics"]
            stability = item.get("stability", {})
            errors = stability.get("state_index_mae_vs_reference", {})
            writer.writerow(
                {
                    "p": item["p"], "sample_count": item["sample_count"],
                    "measured_runs": timing["measured_runs"],
                    "total_mean_ms": timing["total_mean_ms"],
                    "total_median_ms": timing["total_median_ms"],
                    "total_p95_ms": timing["total_p95_ms"],
                    "per_prompt_median_ms": timing["per_prompt_median_ms"],
                    "S_mean_mean": state["S_mean_mean"], "D_mean": state["D_mean"],
                    "R_mean": state["R_mean"], "R_bootstrap_se_mean": state["R_bootstrap_se_mean"],
                    "state_index_mae_mean": stability.get("state_index_mae_mean"),
                    "state_index_mae_S_mean": errors.get("S_mean"),
                    "state_index_mae_D": errors.get("D"), "state_index_mae_R": errors.get("R"),
                    "reference_p": stability.get("reference_p"),
                }
            )
    metrics_path = root / "state_metrics.jsonl"
    rows = [
        json.loads(row[0])
        for row in connection.execute("SELECT payload_json FROM state_metrics ORDER BY p, sample_id")
    ]
    metrics_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if args.model_key != MODEL_KEY:
        raise ValueError(f"This runner is Qwen3-VL-only and requires {MODEL_KEY!r}")
    p_values = tuple(sorted(set(args.p_values)))
    if p_values != tuple(args.p_values):
        raise ValueError("--p-values must be strictly increasing")
    if not p_values or any(p <= 0 or p & (p - 1) for p in p_values):
        raise ValueError("--p-values must be positive powers of two")
    pool_path = args.prompt_pool.expanduser().resolve()
    prompt_pool = _read_pool(pool_path, minimum=max(p_values))
    manifest = args.manifest.expanduser().resolve()
    samples, cohort_sha = _select_samples(
        manifest, sample_count=args.sample_count, sample_ids_file=args.sample_ids_file
    )
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    db = _init_db(root / "sweep_ledger.sqlite3")
    metadata = {
        "schema": SCHEMA,
        "model_key": args.model_key,
        "protocol": PROTOCOL,
        "prefill_strategy": "qwen_vl_prompt_kv",
        "prefill_strategy_version": "v1",
        "p_values": list(p_values),
        "pool_path": str(pool_path),
        "pool_sha256": _sha256(pool_path),
        "sample_manifest": str(manifest),
        "sample_manifest_sha256": _sha256(manifest),
        "selected_sample_count": len(samples),
        "selected_sample_ids_sha256": cohort_sha,
        "sample_selection": "sorted sample_id prefix unless --sample-ids-file is supplied",
        "conditions": list(CONDITIONS),
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "stability_reference_p": max(p_values),
        "stability_definition": "absolute state-index error against nested P=64 reference on shared samples",
        "state_representation": "raw last transformer trajectory vector normalized to unit sphere; not a TME result",
        "device": args.device,
        "dtype": args.dtype,
        "cpu_threads": args.cpu_threads,
    }
    provenance_path = root / "provenance.json"
    immutable_keys = (
        "schema", "model_key", "protocol", "prefill_strategy", "prefill_strategy_version",
        "p_values", "pool_sha256", "sample_manifest_sha256", "selected_sample_count",
        "selected_sample_ids_sha256", "conditions", "stability_reference_p",
    )
    if provenance_path.is_file():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        mismatches = [key for key in immutable_keys if existing.get(key) != metadata.get(key)]
        if mismatches:
            raise ValueError(
                "existing sweep provenance does not match this run for: " + ", ".join(mismatches)
            )
    else:
        provenance_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (root / "sample_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples), encoding="utf-8"
    )
    try:
        import torch

        if args.cpu_threads is not None:
            torch.set_num_threads(args.cpu_threads)
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable for the requested KV sweep")
            torch.cuda.set_per_process_memory_fraction(args.max_memory_fraction, torch.device(args.device))
        assets = index_assets(load_model_assets(args.asset_config, require_local_paths=True))
        asset = assets.get(args.model_key)
        if asset is None:
            raise ValueError(f"model key not found in asset config: {args.model_key}")
        model_path = args.model_path or asset.local_model_path
        wrapper = get_wrapper("qwen_vl")(
            model_key=args.model_key,
            model_path=model_path,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        wrapper.load()
        # The reviewed prompt bank deliberately varies the first instruction
        # token.  The model-native cache remains exact as long as the media/chat
        # prefix is non-empty; the extractor's default 50% cost guard would
        # reject these valid short suffixes, so this sweep records the relaxed
        # contract explicitly instead of silently falling back.
        extractor = QwenVlPromptKvPrefillExtractor(
            wrapper, verbose=False, min_prefix_fraction=0.0
        )
        for p in p_values:
            prompts = prompt_pool[:p]
            prompt_set_key = f"kv_sweep_p{p}_nested"
            for _ in range(args.warmup_runs):
                sample = samples[0]
                for condition in CONDITIONS:
                    _extract(
                        wrapper=wrapper, extractor=extractor, sample=sample, prompt_rows=prompts,
                        condition=condition, prompt_set_key=prompt_set_key,
                        video_fps=args.video_fps, joint_audio_mode=args.joint_audio_mode,
                    )
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            for sample in samples:
                for repeat in range(args.measure_runs):
                    if _is_complete(db, p, sample["sample_id"], repeat):
                        continue
                    results_by_condition: dict[str, list[Any]] = {}
                    condition_elapsed: dict[str, float] = {}
                    cache_modes: set[str] = set()
                    try:
                        if args.device.startswith("cuda"):
                            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
                        _sync_cuda(args.device)
                        started = time.perf_counter()
                        for condition in CONDITIONS:
                            condition_started = time.perf_counter()
                            results, cache_mode = _extract(
                                wrapper=wrapper, extractor=extractor, sample=sample, prompt_rows=prompts,
                                condition=condition, prompt_set_key=prompt_set_key,
                                video_fps=args.video_fps, joint_audio_mode=args.joint_audio_mode,
                            )
                            _sync_cuda(args.device)
                            condition_elapsed[condition] = (time.perf_counter() - condition_started) * 1000.0
                            cache_modes.add(cache_mode)
                            results_by_condition[condition] = results
                        _sync_cuda(args.device)
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        peak = (
                            int(torch.cuda.max_memory_allocated(torch.device(args.device)))
                            if args.device.startswith("cuda") else None
                        )
                        mode = "+".join(sorted(cache_modes))
                        db.execute(
                            """
                            INSERT INTO runs(p,sample_id,repeat,status,elapsed_ms,condition_elapsed_json,
                                              prompt_count,cache_mode,peak_gpu_bytes,error,updated_at)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(p,sample_id,repeat) DO UPDATE SET
                              status=excluded.status, elapsed_ms=excluded.elapsed_ms,
                              condition_elapsed_json=excluded.condition_elapsed_json,
                              prompt_count=excluded.prompt_count, cache_mode=excluded.cache_mode,
                              peak_gpu_bytes=excluded.peak_gpu_bytes, error=excluded.error,
                              updated_at=excluded.updated_at
                            """,
                            (p, sample["sample_id"], repeat, "complete", elapsed_ms,
                             json.dumps(condition_elapsed, sort_keys=True), p, mode, peak, None, time.time()),
                        )
                        if repeat == 0:
                            payload = _state_payload(
                                sample=sample, p=p, prompt_rows=prompts,
                                results_by_condition=results_by_condition,
                            )
                            db.execute(
                                "INSERT INTO state_metrics(p,sample_id,payload_json,updated_at) VALUES(?,?,?,?) "
                                "ON CONFLICT(p,sample_id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at",
                                (p, sample["sample_id"], json.dumps(payload, sort_keys=True), time.time()),
                            )
                        db.commit()
                    except Exception as exc:
                        _record_failure(
                            db, p=p, sample_id=sample["sample_id"], repeat=repeat,
                            prompt_count=p, cache_mode="+".join(sorted(cache_modes)) or "unknown",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        raise
        summary = [_aggregate(db, p) for p in p_values]
        _add_convergence(summary, db, max(p_values))
        _write_exports(root, metadata=metadata, summary=summary, connection=db)
        print(json.dumps({**metadata, "summary": summary}, indent=2, sort_keys=True))
        return 0
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Qwen3-VL nested P KV-cache latency/stability sweep.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prompt-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--asset-config", type=Path, default=DEFAULT_ASSET_CONFIG)
    parser.add_argument("--model-key", default=MODEL_KEY)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--p-values", nargs="+", type=int, default=list(DEFAULT_P_VALUES))
    parser.add_argument("--sample-count", type=int, default=10, help="0 selects every eligible sample")
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-memory-fraction", type=float, default=0.90)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--joint-audio-mode", choices=("embedded_video", "separate_file"), default="embedded_video")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_count < 0 or args.warmup_runs < 0 or args.measure_runs <= 0:
        raise ValueError("sample-count/warmup-runs must be non-negative and measure-runs positive")
    if not 0.1 <= args.max_memory_fraction <= 0.9:
        raise ValueError("max-memory-fraction must be between 0.1 and 0.9")
    if args.cpu_threads <= 0:
        raise ValueError("cpu-threads must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
