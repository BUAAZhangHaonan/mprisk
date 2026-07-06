from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from curation.scripts.common import parse_bool, read_jsonl, write_jsonl


def build_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    source_dataset = row.get("source_dataset", "")
    source_id = row.get("source_id", row["sample_id"])
    protocol = row.get("protocol", "")
    m1_modality = row.get("m1_modality", "M1")
    m2_modality = row.get("m2_modality", "M2")
    media_paths = row.get("media_paths", {})
    return {
        "sample_id": row["sample_id"],
        "source_dataset": source_dataset,
        "source_id": source_id,
        "protocol": protocol,
        "sample_type": row.get("sample_type", "Ambiguous"),
        "split_group_id": row.get("split_group_id", f"{source_dataset}:{source_id}"),
        "media_paths": media_paths,
        "views": {
            "M1": {
                "modality": m1_modality,
                "label": row.get("m1_label", "uncertain"),
                "specific_affect": row.get("m1_specific_affect", ""),
                "is_clear": parse_bool(row.get("m1_is_clear")),
            },
            "M2": {
                "modality": m2_modality,
                "label": row.get("m2_label", "uncertain"),
                "specific_affect": row.get("m2_specific_affect", ""),
                "is_clear": parse_bool(row.get("m2_is_clear")),
            },
            "M12": {
                "modality": f"{m1_modality}+{m2_modality}",
                "label": row.get("joint_label", "uncertain"),
                "specific_affect": row.get("joint_specific_affect", ""),
                "is_clear": parse_bool(row.get("joint_is_clear")),
            },
        },
        "dominant_modality": row.get("dominant_modality", "unclear"),
        "annotator_agreement": float(row.get("annotator_agreement", 0.0)),
        "quality_flags": row.get("quality_flags", []),
        "source_is_generated": parse_bool(row.get("source_is_generated")),
        "use_in_main": row.get("sample_type") in {"Conflict", "Aligned"}
        and float(row.get("annotator_agreement", 0.0)) >= 0.5,
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
