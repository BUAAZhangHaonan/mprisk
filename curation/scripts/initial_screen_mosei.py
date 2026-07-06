from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from curation.scripts.common import is_clear, polarity_label, read_csv, write_jsonl


def build_candidate(row: dict[str, str], *, dataset: str, clear_abs_threshold: float) -> dict[str, Any]:
    source_id = row.get("source_id") or row.get("clip_id") or row.get("id") or row.get("video_id") or ""
    sample_id = row.get("sample_id") or f"{dataset}:{source_id}"
    sentiment = float(row.get("sentiment") or row.get("label") or row.get("overall") or 0.0)
    return {
        "sample_id": sample_id,
        "source_dataset": dataset,
        "source_id": source_id,
        "protocol": row.get("protocol", "VT"),
        "m1_modality": "vision",
        "m2_modality": "text",
        "joint_label": polarity_label(sentiment, clear_abs_threshold),
        "overall_raw_label": sentiment,
        "candidate_type": "Ambiguous",
        "candidate_reason": "overall sentiment candidate pool",
        "media_paths": {
            "vision": row.get("video_path", ""),
            "audio": row.get("audio_path", ""),
            "text": row.get("text", ""),
        },
        "needs_llm_screening": is_clear(sentiment, clear_abs_threshold),
        "source_is_generated": False,
    }


def main(default_dataset: str = "cmu_mosei") -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=default_dataset)
    parser.add_argument("--clear-abs-threshold", type=float, default=0.4)
    args = parser.parse_args()
    rows = [
        build_candidate(row, dataset=args.dataset, clear_abs_threshold=args.clear_abs_threshold)
        for row in read_csv(args.input)
    ]
    write_jsonl(Path(args.output), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
