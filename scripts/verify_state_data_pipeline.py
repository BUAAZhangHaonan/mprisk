from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mprisk.cache.hidden_state_cache import HiddenStateEntry
from mprisk.cache.prefill_extract import bundle_three_views
from mprisk.data.manifests import read_jsonl
from mprisk.data.state_dataset import StateDatasetBuildResult, build_state_dataset
from mprisk.utils.io import ensure_parent


@dataclass(frozen=True)
class StateDataSmokeResult:
    state_dataset: StateDatasetBuildResult
    report_path: Path
    trajectory_checked_rows: int
    trajectory_error_rows: int


def run_state_data_smoke(
    *,
    manifest_paths: list[str | Path],
    cache_root: str | Path = ".",
    model_key: str,
    protocol: str,
    split_assignment_path: str | Path,
    output_dir: str | Path | None = None,
    reports_dir: str | Path = "outputs/state_data/reports",
    cache_manifest_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    trajectory_check_limit: int = 5,
) -> StateDataSmokeResult:
    state_dataset = build_state_dataset(
        manifest_paths=manifest_paths,
        cache_root=cache_root,
        model_key=model_key,
        protocol=protocol,
        split_assignment_path=split_assignment_path,
        output_dir=output_dir,
        manifest_path=cache_manifest_path,
        ledger_path=ledger_path,
    )
    trajectory_checks = _check_trajectories(
        state_dataset.manifest_path,
        limit=trajectory_check_limit,
    )
    report_path = _write_smoke_report(
        state_dataset=state_dataset,
        trajectory_checks=trajectory_checks,
        reports_dir=reports_dir,
    )
    return StateDataSmokeResult(
        state_dataset=state_dataset,
        report_path=report_path,
        trajectory_checked_rows=len(trajectory_checks),
        trajectory_error_rows=sum(1 for item in trajectory_checks if item["status"] != "ok"),
    )


def _check_trajectories(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    checks: list[dict[str, Any]] = []
    for row in rows[: max(limit, 0)]:
        try:
            bundle = bundle_three_views(
                _entry_from_row(row["m1_entry"]),
                _entry_from_row(row["m2_entry"]),
                _entry_from_row(row["m12_entry"]),
            )
            checks.append(
                {
                    "sample_id": row["sample_id"],
                    "status": "ok",
                    "trajectory_meta": bundle.trajectory_meta,
                }
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(
                {
                    "sample_id": row.get("sample_id", "<unknown>"),
                    "status": "error",
                    "error": str(exc),
                }
            )
    return checks


def _entry_from_row(row: dict[str, Any]) -> HiddenStateEntry:
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


def _write_smoke_report(
    *,
    state_dataset: StateDatasetBuildResult,
    trajectory_checks: list[dict[str, Any]],
    reports_dir: str | Path,
) -> Path:
    report_path = ensure_parent(Path(reports_dir) / "STATE_DATA_SMOKE.md")
    error_count = sum(1 for item in trajectory_checks if item["status"] != "ok")
    lines = [
        "# State Data Smoke Report",
        "",
        f"- State manifest: `{state_dataset.manifest_path}`",
        f"- Summary: `{state_dataset.summary_path}`",
        f"- Missing cache rows: `{state_dataset.missing_path}`",
        f"- Resolved rows: {state_dataset.resolved_count}",
        f"- Missing cache rows: {state_dataset.missing_count}",
        f"- Trajectory rows checked: {len(trajectory_checks)}",
        f"- Trajectory check errors: {error_count}",
        "",
        "| sample_id | status | detail |",
        "| --- | --- | --- |",
    ]
    for item in trajectory_checks:
        detail = item.get("trajectory_meta") or item.get("error") or "-"
        lines.append(f"| {item['sample_id']} | {item['status']} | {detail} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the first-stage state-data pipeline.")
    parser.add_argument("--manifest", action="append", dest="manifests", required=True)
    parser.add_argument("--cache-root", default=".")
    parser.add_argument("--cache-manifest", default=None)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split-assignment", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reports-dir", default="outputs/state_data/reports")
    parser.add_argument("--trajectory-check-limit", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_state_data_smoke(
        manifest_paths=[Path(path) for path in args.manifests],
        cache_root=args.cache_root,
        cache_manifest_path=args.cache_manifest,
        ledger_path=args.ledger,
        model_key=args.model_key,
        protocol=args.protocol,
        split_assignment_path=args.split_assignment,
        output_dir=args.output_dir,
        reports_dir=args.reports_dir,
        trajectory_check_limit=args.trajectory_check_limit,
    )
    print(f"state_dataset_manifest={result.state_dataset.manifest_path}")
    print(f"state_dataset_summary={result.state_dataset.summary_path}")
    print(f"missing_cache_rows={result.state_dataset.missing_path}")
    print(f"smoke_report={result.report_path}")
    print(f"trajectory_checked_rows={result.trajectory_checked_rows}")
    print(f"trajectory_error_rows={result.trajectory_error_rows}")
    return 1 if result.trajectory_error_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
