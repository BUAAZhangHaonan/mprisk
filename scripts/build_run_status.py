from __future__ import annotations

import argparse
from pathlib import Path

from mprisk.viz.run_status import build_run_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Build artifact-backed RUN_STATUS.md.")
    parser.add_argument("--config", type=Path, default=Path("configs/paper/figure_map.yaml"))
    parser.add_argument("--output", type=Path, default=Path("RUN_STATUS.md"))
    parser.add_argument("--records", type=Path, default=None)
    args = parser.parse_args()
    print(build_run_status(args.config, records_path=args.records, output_path=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
