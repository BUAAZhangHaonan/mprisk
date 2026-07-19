from __future__ import annotations

import argparse
import json

from mprisk.experiments.misread_budget_queue import run_misread_budget_queue


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the resumable Conflict-supervision Misread-probe queue."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process ready fractions once and fail if any FRACTION_COMPLETE is absent.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        help="Override the configured polling interval; the registered default is 30 seconds.",
    )
    args = parser.parse_args()
    marker = run_misread_budget_queue(
        args.config,
        once=args.once,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps({"complete_marker": str(marker)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
