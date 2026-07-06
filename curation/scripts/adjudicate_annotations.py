from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from curation.scripts.common import parse_bool, read_jsonl, write_jsonl


def majority(values: list[Any], default: Any = "") -> tuple[Any, float]:
    if not values:
        return default, 0.0
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    return value, count / len(values)


def adjudicate_sample(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot adjudicate empty annotation list")
    first = rows[0]
    sample_type, type_agreement = majority([row.get("sample_type", "Ambiguous") for row in rows], "Ambiguous")
    dominant_modality, _ = majority([row.get("dominant_modality", "unclear") for row in rows], "unclear")
    m1_label, m1_agreement = majority([row.get("m1_label", "uncertain") for row in rows], "uncertain")
    m2_label, m2_agreement = majority([row.get("m2_label", "uncertain") for row in rows], "uncertain")
    joint_label, joint_agreement = majority([row.get("joint_label", "uncertain") for row in rows], "uncertain")
    quality_flags = sorted({flag for row in rows for flag in row.get("quality_flags", [])})
    agreement = min(type_agreement, m1_agreement, m2_agreement, joint_agreement)
    return {
        **first,
        "m1_label": m1_label,
        "m2_label": m2_label,
        "joint_label": joint_label,
        "m1_is_clear": all(parse_bool(row.get("m1_is_clear")) for row in rows),
        "m2_is_clear": all(parse_bool(row.get("m2_is_clear")) for row in rows),
        "joint_is_clear": all(parse_bool(row.get("joint_is_clear")) for row in rows),
        "sample_type": sample_type,
        "dominant_modality": dominant_modality,
        "quality_flags": quality_flags,
        "annotator_agreement": round(agreement, 4),
        "annotation_count": len(rows),
    }


def adjudicate_all(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    return [adjudicate_sample(group) for group in grouped.values()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="curation/outputs/adjudicated/adjudicated_labels.jsonl")
    args = parser.parse_args()
    write_jsonl(Path(args.output), adjudicate_all(read_jsonl(args.input)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
