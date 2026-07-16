from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

COARSE_LABELS = {"positive", "negative", "neutral", "uncertain", "invalid"}
SAMPLE_TYPES = {"Conflict", "Ambiguous", "Aligned"}
DOMINANT_MODALITIES = {"M1", "M2", "balanced", "unclear"}


def polarity_label(value: float, clear_abs_threshold: float = 0.4) -> str:
    if abs(value) < clear_abs_threshold:
        return "neutral"
    return "positive" if value > 0 else "negative"


def is_clear(value: float, clear_abs_threshold: float = 0.4) -> bool:
    return abs(value) >= clear_abs_threshold


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}

