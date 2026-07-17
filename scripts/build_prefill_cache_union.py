from __future__ import annotations

import argparse
import json
from pathlib import Path

from mprisk.cache.cache_union import (
    blocked_tasks_from_rows,
    build_cache_union,
    expected_tasks_from_plan,
    load_cache_source,
    write_cache_source,
    write_extractor_evidence,
)
from mprisk.cache.prefill_batch import (
    CONDITIONS,
    DEFAULT_ASSET_CONFIG,
    FULL_PREFILL_STRATEGY,
    build_batch_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an immutable, validated view over disjoint prefill-cache roots."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evidence = subparsers.add_parser(
        "record-evidence",
        help="Bind a completed source ledger to exact full-prefill code/config hashes.",
    )
    evidence.add_argument("--source-id", required=True)
    evidence.add_argument("--ledger", required=True, type=Path)
    evidence.add_argument("--cache-root", required=True, type=Path)
    evidence.add_argument("--code-root", required=True, type=Path)
    evidence.add_argument("--output", required=True, type=Path)

    source = subparsers.add_parser(
        "record-source",
        help="Create one immutable cache-source descriptor.",
    )
    source.add_argument("--source-id", required=True)
    source.add_argument("--ledger", required=True, type=Path)
    source.add_argument("--cache-root", required=True, type=Path)
    source.add_argument("--evidence", required=True, type=Path)
    source.add_argument("--output", required=True, type=Path)

    union = subparsers.add_parser("build", help="Validate sources and write the union index.")
    union.add_argument("--manifest", required=True, type=Path)
    union.add_argument("--blocked-manifest", type=Path)
    union.add_argument("--prompt-set", required=True, type=Path)
    union.add_argument("--prompt-variable", action="append", default=[])
    union.add_argument("--protocol", required=True, choices=("vt", "va", "vta"))
    union.add_argument("--conditions", nargs="+", default=CONDITIONS)
    union.add_argument(
        "--joint-audio-mode",
        default="embedded_video",
        choices=("embedded_video", "separate_file"),
    )
    union.add_argument("--video-fps", type=float, default=1.0)
    union.add_argument("--video-num-segments", type=int, default=8)
    union.add_argument("--internvl-max-num", type=int, default=1)
    union.add_argument("--model-key", required=True)
    union.add_argument("--asset-config", default=DEFAULT_ASSET_CONFIG, type=Path)
    union.add_argument("--family", choices=("qwen_omni", "qwen_vl", "internvl"))
    union.add_argument("--model-path", type=Path)
    union.add_argument("--device", default="cpu")
    union.add_argument("--dtype", default="bfloat16", choices=("bfloat16",))
    union.add_argument("--attn-implementation", choices=("sdpa", "eager"))
    union.add_argument("--min-pixels", type=int)
    union.add_argument("--max-pixels", type=int)
    union.add_argument("--source", action="append", required=True, type=Path)
    union.add_argument("--expected-resolved-tasks", required=True, type=int)
    union.add_argument("--expected-blocked-tasks", type=int, default=0)
    union.add_argument("--expected-raw-tasks", required=True, type=int)
    union.add_argument("--checksum-workers", type=int, default=8)
    union.add_argument("--output", required=True, type=Path)
    union.set_defaults(
        prefill_strategy=FULL_PREFILL_STRATEGY,
        output_root=Path("."),
        retry_failed=False,
        fail_fast=True,
        materialize_every=100,
        dry_run=False,
        probe_media=False,
        ffprobe_workers=1,
        gpu_index=None,
        trajectory_shape=None,
        smoke_condition_seconds=[],
        smoke_wall_seconds=None,
        smoke_media_seconds=None,
        smoke_artifact_bytes_per_task=None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "record-evidence":
        path = write_extractor_evidence(
            source_id=args.source_id,
            ledger_path=args.ledger,
            cache_root=args.cache_root,
            code_root=args.code_root,
            output_path=args.output,
        )
        print(json.dumps({"status": "complete", "evidence": str(path)}, sort_keys=True))
        return 0
    if args.command == "record-source":
        path = write_cache_source(
            source_id=args.source_id,
            cache_root=args.cache_root,
            ledger_path=args.ledger,
            evidence_path=args.evidence,
            output_path=args.output,
        )
        print(json.dumps({"status": "complete", "source": str(path)}, sort_keys=True))
        return 0

    plan = build_batch_plan(args)
    if plan.unresolved_prompt_variables:
        raise ValueError(
            "Unresolved prompt variables: " + ", ".join(plan.unresolved_prompt_variables)
        )
    expected = expected_tasks_from_plan(args, plan)
    blocked_rows = _read_jsonl(args.blocked_manifest) if args.blocked_manifest else []
    blocked = blocked_tasks_from_rows(
        blocked_rows,
        model_key=args.model_key,
        protocol=args.protocol,
        prompt_ids=plan.prompt_ids,
        conditions=args.conditions,
    )
    result = build_cache_union(
        expected_tasks=expected,
        expected_signature=plan.signature,
        sources=[load_cache_source(path) for path in args.source],
        output_path=args.output,
        blocked_tasks=blocked,
        expected_resolved_tasks=args.expected_resolved_tasks,
        expected_blocked_tasks=args.expected_blocked_tasks,
        expected_raw_tasks=args.expected_raw_tasks,
        checksum_workers=args.checksum_workers,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(result.output_path),
                "resolved_tasks": result.resolved_tasks,
                "blocked_tasks": result.blocked_tasks,
                "source_counts": result.source_counts,
                "extractor_semantic_fingerprint": result.extractor_semantic_fingerprint,
            },
            sort_keys=True,
        )
    )
    return 0


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
