from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from curation.scripts.common import is_clear, polarity_label, read_csv, write_jsonl


def classify_relation(
    m1_value: float,
    m2_value: float,
    joint_value: float,
    *,
    clear_abs_threshold: float = 0.4,
    conflict_gap_threshold: float = 0.8,
) -> str:
    m1_clear = is_clear(m1_value, clear_abs_threshold)
    m2_clear = is_clear(m2_value, clear_abs_threshold)
    joint_clear = is_clear(joint_value, clear_abs_threshold)
    if not (m1_clear and m2_clear and joint_clear):
        return "Ambiguous"

    m1_label = polarity_label(m1_value, clear_abs_threshold)
    m2_label = polarity_label(m2_value, clear_abs_threshold)
    joint_label = polarity_label(joint_value, clear_abs_threshold)
    if m1_label != m2_label and abs(m1_value - m2_value) >= conflict_gap_threshold:
        return "Conflict"
    if m1_label == m2_label == joint_label:
        return "Aligned"
    return "Ambiguous"


def make_candidate(
    *,
    sample_id: str,
    source_dataset: str,
    source_id: str,
    protocol: str,
    m1_modality: str,
    m2_modality: str,
    m1_raw: float,
    m2_raw: float,
    joint_raw: float,
    media_paths: dict[str, str] | None = None,
    clear_abs_threshold: float = 0.4,
    conflict_gap_threshold: float = 0.8,
) -> dict[str, Any]:
    candidate_type = classify_relation(
        m1_raw,
        m2_raw,
        joint_raw,
        clear_abs_threshold=clear_abs_threshold,
        conflict_gap_threshold=conflict_gap_threshold,
    )
    m1_label = polarity_label(m1_raw, clear_abs_threshold)
    m2_label = polarity_label(m2_raw, clear_abs_threshold)
    joint_label = polarity_label(joint_raw, clear_abs_threshold)
    return {
        "sample_id": sample_id,
        "source_dataset": source_dataset,
        "source_id": source_id,
        "protocol": protocol,
        "m1_modality": m1_modality,
        "m2_modality": m2_modality,
        "m1_raw_label": m1_raw,
        "m2_raw_label": m2_raw,
        "joint_raw_label": joint_raw,
        "m1_label": m1_label,
        "m2_label": m2_label,
        "joint_label": joint_label,
        "m1_is_clear": is_clear(m1_raw, clear_abs_threshold),
        "m2_is_clear": is_clear(m2_raw, clear_abs_threshold),
        "joint_is_clear": is_clear(joint_raw, clear_abs_threshold),
        "candidate_type": candidate_type,
        "candidate_reason": f"{protocol}: {m1_label}/{m2_label}/{joint_label}",
        "media_paths": media_paths or {},
        "needs_llm_screening": True,
        "source_is_generated": False,
    }


def _pick(row: dict[str, str], names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        if row.get(name) not in {None, ""}:
            return row[name]
    return default


def build_candidates(
    rows: list[dict[str, str]],
    *,
    protocol: str,
    clear_abs_threshold: float = 0.4,
    conflict_gap_threshold: float = 0.8,
) -> list[dict[str, Any]]:
    protocol_upper = protocol.upper()
    if protocol_upper == "VT":
        m2_modality = "text"
        m2_names = ("text", "text_label", "t", "T")
    elif protocol_upper == "VA":
        m2_modality = "audio"
        m2_names = ("audio", "audio_label", "a", "A")
    else:
        raise ValueError(f"Unsupported CH-SIMS v2 protocol: {protocol}")

    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_id = _pick(row, ("source_id", "clip_id", "video_id", "id"), f"row-{index}")
        sample_id = _pick(row, ("sample_id",), f"ch_sims_v2:{protocol_upper}:{source_id}")
        m1_raw = float(_pick(row, ("vision", "visual", "vision_label", "v", "V"), "0"))
        m2_raw = float(_pick(row, m2_names, "0"))
        joint_raw = float(_pick(row, ("joint", "multimodal", "label", "sentiment"), "0"))
        candidates.append(
            make_candidate(
                sample_id=sample_id,
                source_dataset="ch_sims_v2",
                source_id=source_id,
                protocol=protocol_upper,
                m1_modality="vision",
                m2_modality=m2_modality,
                m1_raw=m1_raw,
                m2_raw=m2_raw,
                joint_raw=joint_raw,
                media_paths={
                    "vision": _pick(row, ("video_path", "vision_path"), ""),
                    "audio": _pick(row, ("audio_path",), ""),
                    "text": _pick(row, ("text_path", "text"), ""),
                },
                clear_abs_threshold=clear_abs_threshold,
                conflict_gap_threshold=conflict_gap_threshold,
            )
        )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--protocol", choices=["VT", "VA"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clear-abs-threshold", type=float, default=0.4)
    parser.add_argument("--conflict-gap-threshold", type=float, default=0.8)
    args = parser.parse_args()
    rows = read_csv(args.input)
    candidates = build_candidates(
        rows,
        protocol=args.protocol,
        clear_abs_threshold=args.clear_abs_threshold,
        conflict_gap_threshold=args.conflict_gap_threshold,
    )
    write_jsonl(Path(args.output), candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
