from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.viz.plot_kv_p_sweep import export_curve


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export Qwen-VL KV-cache P latency/stability curves."
    )
    parser.add_argument("--input", type=Path, required=True, help="Sweep measurement JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export_curve(args.input, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

