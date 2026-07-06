from __future__ import annotations

import argparse
from pathlib import Path

from curation.scripts.common import read_csv, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = []
    for row in read_csv(args.input):
        has_pair_text = bool(row.get("text") or row.get("text_path") or row.get("caption"))
        if not has_pair_text:
            continue
        source_id = row.get("source_id") or row.get("clip_id") or row.get("id") or ""
        candidates.append(
            {
                "sample_id": row.get("sample_id") or f"dfew:{source_id}",
                "source_dataset": "dfew",
                "source_id": source_id,
                "protocol": "IT",
                "m1_modality": "image",
                "m2_modality": "text",
                "candidate_type": "Ambiguous",
                "candidate_reason": "visual anchor with natural text",
                "visual_anchor_label": row.get("emotion", row.get("label", "")),
                "media_paths": {"vision": row.get("video_path", ""), "text": row.get("text", "")},
                "needs_llm_screening": True,
                "source_is_generated": False,
            }
        )
    write_jsonl(Path(args.output), candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
