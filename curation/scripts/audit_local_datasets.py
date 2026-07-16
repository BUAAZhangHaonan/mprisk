from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DATASET_ROOT = Path("/home/team/zhanghaonan/TAFFC/datasets")
DEFAULT_OUTPUT_DIR = Path("curation/outputs/reports")
DATASET_KEYS = ("ch_sims", "ch_sims_v2", "cmu_mosi", "cmu_mosei")

LABEL_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".xlsx",
    ".xls",
    ".mat",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
}
READABLE_LABEL_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
TEXT_EXTENSIONS = {".txt", ".srt", ".vtt", ".csv", ".tsv", ".json", ".jsonl"}

TEXT_HINTS = ("text", "transcript", "sentence", "utterance", "word")
VIDEO_HINTS = ("video", "visual", "vision", "face")
AUDIO_HINTS = ("audio", "acoustic", "wav", "speech")
IMAGE_HINTS = ("image", "img", "frame", "frames", "picture")
LABEL_HINTS = ("label", "sentiment", "emotion", "valence", "arousal", "annotation", "score")
MULTIMODAL_HINTS = ("multimodal", "multi_modal", "fusion", "joint")
ID_HINTS = ("id", "clip", "segment", "utterance")


def audit_datasets(dataset_root: str | Path = DEFAULT_DATASET_ROOT) -> dict[str, Any]:
    root = Path(dataset_root).expanduser()
    return {
        "dataset_root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [_audit_dataset(root, dataset_key) for dataset_key in DATASET_KEYS],
    }


def write_outputs(payload: dict[str, Any], output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "dataset_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (target_dir / "DATASET_AUDIT.md").write_text(_render_markdown(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit local multimodal dataset directories.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    payload = audit_datasets(args.dataset_root)
    write_outputs(payload, args.output_dir)
    return 0


def _audit_dataset(dataset_root: Path, dataset_key: str) -> dict[str, Any]:
    root_path = dataset_root / dataset_key
    exists = root_path.exists()
    notes: list[str] = []
    file_paths: list[Path] = []

    if exists:
        file_paths = sorted(path for path in root_path.rglob("*") if path.is_file())
    else:
        notes.append("Dataset directory is missing; pending audit.")

    total_bytes = _sum_file_sizes(file_paths, notes)
    label_paths = [path for path in file_paths if path.suffix.lower() in LABEL_EXTENSIONS]
    label_files = [_relative_path(path, root_path) for path in label_paths]
    label_columns_by_file = {
        _relative_path(path, root_path): _read_label_columns(path) for path in label_paths
    }

    hint_tokens = _collect_hint_tokens(root_path, file_paths, label_columns_by_file)
    column_tokens = _collect_column_tokens(label_columns_by_file)
    detected_modalities = _detect_modalities(file_paths, label_columns_by_file, hint_tokens)
    column_mapping_suggestions = _suggest_column_mappings(label_columns_by_file)
    protocol_support = _protocol_support(dataset_key, detected_modalities, hint_tokens, column_tokens)

    if exists and not label_files:
        notes.append("No recognizable label files found.")
    if dataset_key == "ch_sims" and not (
        protocol_support["VT_native"] and protocol_support["VA_native"]
    ):
        notes.append("CH-SIMS protocol support is pending audit until local label hints are confirmed.")

    return {
        "dataset_key": dataset_key,
        "root_path": str(root_path),
        "exists": exists,
        "file_count": len(file_paths),
        "total_bytes": total_bytes,
        "detected_modalities": detected_modalities,
        "label_files": label_files,
        "label_columns_by_file": label_columns_by_file,
        "protocol_support": protocol_support,
        "column_mapping_suggestions": column_mapping_suggestions,
        "notes": notes,
    }


def _sum_file_sizes(file_paths: list[Path], notes: list[str]) -> int:
    total = 0
    for path in file_paths:
        try:
            total += path.stat().st_size
        except OSError as exc:
            notes.append(f"Could not read file size for {path}: {exc}")
    return total


def _read_label_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            return [column.strip() for column in next(reader, []) if column.strip()]
    if suffix == ".jsonl":
        return _jsonl_columns(path)
    if suffix == ".json":
        return _json_columns(path)
    return []


def _jsonl_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                return list(value.keys())
            return []
    return []


def _json_columns(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for nested in value.values():
            if isinstance(nested, list) and nested and isinstance(nested[0], dict):
                return list(nested[0].keys())
        return list(value.keys())
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return list(value[0].keys())
    return []


def _detect_modalities(
    file_paths: list[Path],
    label_columns_by_file: dict[str, list[str]],
    hint_tokens: set[str],
) -> dict[str, bool]:
    suffixes = {path.suffix.lower() for path in file_paths}
    columns = " ".join(column.lower() for columns in label_columns_by_file.values() for column in columns)
    return {
        "video": bool(suffixes & VIDEO_EXTENSIONS) or _has_any(hint_tokens, VIDEO_HINTS) or "label_v" in columns,
        "audio": bool(suffixes & AUDIO_EXTENSIONS) or _has_any(hint_tokens, AUDIO_HINTS) or "label_a" in columns,
        "text": bool(suffixes & TEXT_EXTENSIONS)
        or _has_any(hint_tokens, TEXT_HINTS)
        or "text" in columns
        or "label_t" in columns,
        "image": bool(suffixes & IMAGE_EXTENSIONS) or _has_any(hint_tokens, IMAGE_HINTS),
    }


def _protocol_support(
    dataset_key: str,
    detected_modalities: dict[str, bool],
    hint_tokens: set[str],
    column_tokens: set[str],
) -> dict[str, bool]:
    if dataset_key == "ch_sims_v2":
        return {"VT_native": True, "VA_native": True, "IT_derived": False}

    if dataset_key == "ch_sims":
        has_text = _has_any(column_tokens, TEXT_HINTS)
        has_visual = _has_any(column_tokens, VIDEO_HINTS + IMAGE_HINTS)
        has_audio = _has_any(column_tokens, AUDIO_HINTS)
        has_multimodal_label = _has_any(column_tokens, MULTIMODAL_HINTS)
        return {
            "VT_native": has_text and has_visual and has_multimodal_label,
            "VA_native": has_visual and has_audio and has_multimodal_label,
            "IT_derived": False,
        }

    if dataset_key in {"cmu_mosi", "cmu_mosei"}:
        has_visual = detected_modalities["video"] or detected_modalities["image"]
        return {
            "VT_native": False,
            "VA_native": False,
            "IT_derived": has_visual and detected_modalities["text"],
        }

    return {"VT_native": False, "VA_native": False, "IT_derived": False}


def _suggest_column_mappings(
    label_columns_by_file: dict[str, list[str]],
) -> dict[str, dict[str, list[str]]]:
    suggestions: dict[str, dict[str, list[str]]] = {}
    for relative_path, columns in label_columns_by_file.items():
        suggestions[relative_path] = {
            "sample_id": _matching_columns(columns, ID_HINTS),
            "text": _matching_columns(columns, TEXT_HINTS),
            "video": _matching_columns(columns, VIDEO_HINTS + IMAGE_HINTS),
            "audio": _matching_columns(columns, AUDIO_HINTS),
            "label": _matching_columns(columns, LABEL_HINTS + MULTIMODAL_HINTS),
        }
    return suggestions


def _matching_columns(columns: list[str], hints: tuple[str, ...]) -> list[str]:
    return [column for column in columns if _contains_any(column, hints)]


def _collect_hint_tokens(
    root_path: Path,
    file_paths: list[Path],
    label_columns_by_file: dict[str, list[str]],
) -> set[str]:
    tokens: set[str] = set()
    if root_path.exists():
        tokens.update(_path_tokens(root_path.name))
    for path in file_paths:
        try:
            relative = path.relative_to(root_path)
        except ValueError:
            relative = path
        for part in relative.parts:
            tokens.update(_path_tokens(part))
    for columns in label_columns_by_file.values():
        for column in columns:
            tokens.update(_path_tokens(column))
    return tokens


def _collect_column_tokens(label_columns_by_file: dict[str, list[str]]) -> set[str]:
    tokens: set[str] = set()
    for columns in label_columns_by_file.values():
        for column in columns:
            tokens.update(_path_tokens(column))
    return tokens


def _path_tokens(value: str) -> set[str]:
    normalized = value.lower()
    for separator in (".", "-", "/", "\\"):
        normalized = normalized.replace(separator, "_")
    return {token for token in normalized.split("_") if token}


def _contains_any(value: str, hints: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(hint in lowered for hint in hints)


def _has_any(tokens: set[str], hints: tuple[str, ...]) -> bool:
    return any(any(hint in token for hint in hints) for token in tokens)


def _relative_path(path: Path, root_path: Path) -> str:
    try:
        return path.relative_to(root_path).as_posix()
    except ValueError:
        return path.as_posix()


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Dataset Audit",
        "",
        f"- Dataset root: `{payload['dataset_root']}`",
        f"- Generated at: `{payload['generated_at']}`",
        "",
        "| Dataset | Exists | Files | Bytes | Modalities | VT native | VA native | IT derived | Labels |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: |",
    ]
    for dataset in payload["datasets"]:
        modalities = ", ".join(
            modality for modality, detected in dataset["detected_modalities"].items() if detected
        )
        protocol = dataset["protocol_support"]
        lines.append(
            "| {dataset_key} | {exists} | {file_count} | {total_bytes} | {modalities} | "
            "{vt} | {va} | {it} | {label_count} |".format(
                dataset_key=dataset["dataset_key"],
                exists=_yes_no(dataset["exists"]),
                file_count=dataset["file_count"],
                total_bytes=dataset["total_bytes"],
                modalities=modalities or "-",
                vt=_yes_no(protocol["VT_native"]),
                va=_yes_no(protocol["VA_native"]),
                it=_yes_no(protocol["IT_derived"]),
                label_count=len(dataset["label_files"]),
            )
        )
    lines.append("")

    for dataset in payload["datasets"]:
        lines.extend(
            [
                f"## {dataset['dataset_key']}",
                "",
                f"- Root path: `{dataset['root_path']}`",
                f"- Label files: {', '.join(dataset['label_files']) or '-'}",
                f"- Notes: {'; '.join(dataset['notes']) or '-'}",
                "",
            ]
        )
        if dataset["label_columns_by_file"]:
            lines.append("Label columns:")
            lines.append("")
            for relative_path, columns in dataset["label_columns_by_file"].items():
                lines.append(f"- `{relative_path}`: {', '.join(columns) or '-'}")
            lines.append("")
    return "\n".join(lines)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
