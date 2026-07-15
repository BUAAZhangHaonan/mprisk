from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.prompts.pool import build_prompt_pool, verify_prompt_pool_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic prompt pool.")
    parser.add_argument("--raw-candidates", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompt-set-key", default="prompt_pool_v1")
    parser.add_argument("--protocol", default="vt")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        print(json.dumps(verify_prompt_pool_artifacts(args.output_dir), indent=2, sort_keys=True))
        return

    if args.raw_candidates is None:
        parser.error("--raw-candidates is required unless --verify-only is set")

    result = build_prompt_pool(
        args.raw_candidates,
        args.output_dir,
        prompt_set_key=args.prompt_set_key,
        protocol=args.protocol,
    )
    summary = {
        "raw_count": len(result.raw384),
        "accepted_count": len(result.accepted),
        "rejected_count": len(result.rejections),
        "global_pool_size": len(result.pool128),
        "subset_sizes": {str(seed): len(rows) for seed, rows in result.subsets.items()},
        "provenance": str(args.output_dir / "provenance.json"),
        "artifact_verification": str(args.output_dir / "artifact_verification.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
