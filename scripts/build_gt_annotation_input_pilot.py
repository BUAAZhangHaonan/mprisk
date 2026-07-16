from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.ground_truth.annotation_inputs import write_gt_annotation_input_pilot


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Freeze the deterministic GT annotation-input pilot."
    )
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/frozen/generated_round1_v1/ground_truth_inputs/"
            "gt_annotation_input_v1/pilot.jsonl"
        ),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path(
            "data/frozen/generated_round1_v1/ground_truth_inputs/"
            "gt_annotation_input_v1/pilot.provenance.json"
        ),
    )
    args = parser.parse_args()
    manifest, provenance = write_gt_annotation_input_pilot(
        args.repo_root,
        manifest_path=args.manifest,
        provenance_path=args.provenance,
    )
    print(json.dumps({"manifest": str(manifest), "provenance": str(provenance)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
