from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mprisk.viz.figure_inputs import build_state_figure_inputs, write_pending_figure_inputs
from mprisk.viz.runtime_records import append_command_record, utc_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build provenance-backed paper figure inputs.")
    parser.add_argument("--mode", choices=("ready", "pending"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper/figure_map.yaml"))
    parser.add_argument("--sdr-scores", type=Path, action="append")
    parser.add_argument("--state-patterns", type=Path, action="append")
    parser.add_argument("--thresholds", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_exports/figures"))
    parser.add_argument("--run-records", type=Path, default=None)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    started_at = utc_now()
    try:
        if args.mode == "pending":
            written = write_pending_figure_inputs(args.config, generated_command=command)
        else:
            if not args.sdr_scores or not args.state_patterns or not args.thresholds:
                parser.error("ready mode requires --sdr-scores, --state-patterns, and --thresholds")
            result = build_state_figure_inputs(
                sdr_scores_path=args.sdr_scores,
                state_patterns_path=args.state_patterns,
                thresholds_path=args.thresholds,
                output_dir=args.output_dir,
                generated_command=command,
            )
            written = list(result.__dict__.values())
    except Exception as exc:
        if args.run_records:
            append_command_record(
                args.run_records,
                command_id="build_figure_inputs",
                argv=command,
                status="failure",
                pid=os.getpid(),
                started_at=started_at,
                reason=str(exc),
            )
        raise
    if args.run_records:
        append_command_record(
            args.run_records,
            command_id="build_figure_inputs",
            argv=command,
            status="success",
            pid=os.getpid(),
            started_at=started_at,
        )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
