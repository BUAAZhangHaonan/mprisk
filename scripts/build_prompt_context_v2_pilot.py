from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.ground_truth.prompt_context_v2 import write_prompt_context_v2_pilot


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Freeze the deterministic prompt-context v2 pilot."
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/frozen/generated_round1_v1/ground_truth_inputs/"
            "prompt_context_v2_pilot.jsonl"
        ),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path(
            "data/frozen/generated_round1_v1/ground_truth_inputs/"
            "prompt_context_v2_pilot.provenance.json"
        ),
    )
    args = parser.parse_args()
    manifest, provenance = write_prompt_context_v2_pilot(
        args.repo_root,
        manifest_path=args.manifest,
        provenance_path=args.provenance,
    )
    print(json.dumps({"manifest": str(manifest), "provenance": str(provenance)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
