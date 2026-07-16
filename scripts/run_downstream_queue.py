from __future__ import annotations

import argparse

from mprisk.experiments.downstream import run_queue


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the resumable three-seed representation/state queue."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    return run_queue(args.config, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
