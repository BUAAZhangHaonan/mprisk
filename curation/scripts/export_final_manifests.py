from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from curation.scripts.common import parse_bool, read_jsonl, write_jsonl


BLOCKING_QUALITY_FLAGS = {
    "missing_vision",
    "missing_audio",
    "missing_text",
    "low_audio",
    "face_occluded",
    "corrupted_media",
    "generated_artifact_severe",
    "invalid_media",
    "modality_missing",
}


def _quality_flags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(flag) for flag in value]
    if isinstance(value, tuple | set):
        return [str(flag) for flag in value]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _annotation_count(value: Any) -> int:
    if value in {None, ""}:
        return 0
    return int(value)


def _use_in_main(
    *,
    sample_type: str,
    annotation_count: int,
    annotator_agreement: float,
    m1_label: str,
    m2_label: str,
    joint_label: str,
    m1_is_clear: bool,
    m2_is_clear: bool,
    joint_is_clear: bool,
    quality_flags: list[str],
) -> bool:
    if sample_type not in {"Conflict", "Aligned"}:
        return False
    if annotation_count < 2 or annotator_agreement < 0.67:
        return False
    if not (m1_is_clear and m2_is_clear and joint_is_clear):
        return False
    if BLOCKING_QUALITY_FLAGS.intersection(quality_flags):
        return False
    if sample_type == "Conflict":
        return m1_label != m2_label
    return m1_label == m2_label == joint_label


def build_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    source_dataset = row.get("source_dataset", "")
    source_id = row.get("source_id", row["sample_id"])
    protocol = row.get("protocol", "")
    m1_modality = row.get("m1_modality", "M1")
    m2_modality = row.get("m2_modality", "M2")
    media_paths = row.get("media_paths", {})
    sample_type = row.get("sample_type", "Ambiguous")
    m1_label = row.get("m1_label", "uncertain")
    m2_label = row.get("m2_label", "uncertain")
    joint_label = row.get("joint_label", "uncertain")
    m1_is_clear = parse_bool(row.get("m1_is_clear"))
    m2_is_clear = parse_bool(row.get("m2_is_clear"))
    joint_is_clear = parse_bool(row.get("joint_is_clear"))
    annotator_agreement = float(row.get("annotator_agreement", 0.0))
    annotation_count = _annotation_count(row.get("annotation_count"))
    quality_flags = _quality_flags(row.get("quality_flags", []))
    return {
        "sample_id": row["sample_id"],
        "source_dataset": source_dataset,
        "source_id": source_id,
        "protocol": protocol,
        "sample_type": sample_type,
        "split_group_id": row.get("split_group_id", f"{source_dataset}:{source_id}"),
        "media_paths": media_paths,
        "views": {
            "M1": {
                "modality": m1_modality,
                "label": m1_label,
                "specific_affect": row.get("m1_specific_affect", ""),
                "is_clear": m1_is_clear,
            },
            "M2": {
                "modality": m2_modality,
                "label": m2_label,
                "specific_affect": row.get("m2_specific_affect", ""),
                "is_clear": m2_is_clear,
            },
            "M12": {
                "modality": f"{m1_modality}+{m2_modality}",
                "label": joint_label,
                "specific_affect": row.get("joint_specific_affect", ""),
                "is_clear": joint_is_clear,
            },
        },
        "dominant_modality": row.get("dominant_modality", "unclear"),
        "annotator_agreement": annotator_agreement,
        "annotation_count": annotation_count,
        "quality_flags": quality_flags,
        "source_is_generated": parse_bool(row.get("source_is_generated")),
        "use_in_main": _use_in_main(
            sample_type=sample_type,
            annotation_count=annotation_count,
            annotator_agreement=annotator_agreement,
            m1_label=m1_label,
            m2_label=m2_label,
            joint_label=joint_label,
            m1_is_clear=m1_is_clear,
            m2_is_clear=m2_is_clear,
            joint_is_clear=joint_is_clear,
            quality_flags=quality_flags,
        ),
    }


def export_manifests(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    output_root = Path(output_dir)
    manifest_rows = [build_manifest_row(row) for row in rows]
    paths = {
        "unified": output_root / "unified_sample_manifest.jsonl",
        "Conflict": output_root / "conflict_manifest.jsonl",
        "Ambiguous": output_root / "ambiguous_manifest.jsonl",
        "Aligned": output_root / "aligned_manifest.jsonl",
    }
    write_jsonl(paths["unified"], manifest_rows)
    for sample_type in ("Conflict", "Ambiguous", "Aligned"):
        write_jsonl(paths[sample_type], [row for row in manifest_rows if row["sample_type"] == sample_type])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data/processed/manifests")
    args = parser.parse_args()
    export_manifests(read_jsonl(args.input), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
