from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.viz.plot_kv_latency_stability_tables import export_tables


def main() -> int:
    parser = argparse.ArgumentParser(description="Export clean Qwen3-VL KV latency and stability figures.")
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--stability-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stability-tolerance", type=float, default=0.10)
    args = parser.parse_args()
    result = export_tables(
        args.timing_csv,
        args.stability_csv,
        args.output_dir,
        stability_tolerance=args.stability_tolerance,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
