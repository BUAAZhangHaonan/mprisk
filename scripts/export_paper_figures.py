from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from mprisk.viz.bundle_figures import export_bundle_figures
from mprisk.viz.runtime_records import append_command_record, utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the final artifact-backed figure bundle.")
    parser.add_argument("--config", type=Path, default=Path("configs/paper/figure_map.yaml"))
    parser.add_argument("--run-records", type=Path, default=None)
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    started_at = utc_now()
    try:
        result = export_bundle_figures(args.config)
    except Exception as exc:
        if args.run_records:
            append_command_record(
                args.run_records,
                command_id="export_paper_figures",
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
            command_id="export_paper_figures",
            argv=command,
            status="success",
            pid=os.getpid(),
            started_at=started_at,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
