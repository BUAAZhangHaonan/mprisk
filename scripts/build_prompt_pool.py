from __future__ import annotations

import argparse
import json
from pathlib import Path

from mprisk.prompts.pool import build_prompt_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic prompt pool.")
    parser.add_argument("--raw-candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt-set-key", default="prompt_pool_v1")
    parser.add_argument("--protocol", default="vt")
    args = parser.parse_args()

    result = build_prompt_pool(
        args.raw_candidates,
        args.output_dir,
        prompt_set_key=args.prompt_set_key,
        protocol=args.protocol,
    )
    summary = {
        "raw_count": len(result.raw384),
        "rejected_count": len(result.rejections),
        "global_pool_size": len(result.pool128),
        "subset_sizes": {str(seed): len(rows) for seed, rows in result.subsets.items()},
        "provenance": str(args.output_dir / "provenance.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
