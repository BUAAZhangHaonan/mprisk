from __future__ import annotations

import argparse
import json
from pathlib import Path

from mprisk.viz.bundle_tables import export_bundle_tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export artifact-backed Tables I-III.")
    parser.add_argument("--config", type=Path, default=Path("configs/paper/table_map.yaml"))
    args = parser.parse_args(argv)
    print(json.dumps(export_bundle_tables(args.config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
