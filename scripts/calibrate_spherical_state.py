from __future__ import annotations

import argparse
from pathlib import Path

from mprisk.data.manifests import read_jsonl
from mprisk.state.thresholds import calibrate_registered_aligned_thresholds
from mprisk.utils.io import write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate spherical kappa/tau on an independent Aligned split."
    )
    parser.add_argument("--sdr-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()
    payload = calibrate_registered_aligned_thresholds(
        read_jsonl(args.sdr_scores), quantile_level=args.quantile
    )
    write_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
