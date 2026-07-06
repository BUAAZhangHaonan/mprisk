from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from curation.scripts.common import read_jsonl, write_jsonl


def candidate_from_metadata(row: dict[str, Any]) -> dict[str, Any]:
    planned_type = row.get("planned_sample_type", row.get("sample_type", "Ambiguous"))
    return {
        "sample_id": row["sample_id"],
        "source_dataset": row.get("source_dataset", "generated"),
        "source_id": row.get("source_id", row["sample_id"]),
        "protocol": row.get("protocol", "IT"),
        "m1_modality": row.get("m1_modality", "image"),
        "m2_modality": row.get("m2_modality", "text"),
        "m1_label": row.get("planned_m1_label", "uncertain"),
        "m2_label": row.get("planned_m2_label", "uncertain"),
        "joint_label": row.get("planned_joint_label", "uncertain"),
        "candidate_type": planned_type,
        "candidate_reason": "generated metadata planned labels",
        "media_paths": row.get("media_paths", {}),
        "generation_model": row.get("generation_model", ""),
        "generation_prompt": row.get("generation_prompt", ""),
        "generation_prompt_id": row.get("generation_prompt_id", ""),
        "needs_llm_screening": True,
        "source_is_generated": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_jsonl(Path(args.output), [candidate_from_metadata(row) for row in read_jsonl(args.input)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
