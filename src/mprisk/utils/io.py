"""Input/output helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = ensure_parent(path)
    payload_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload_text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = ensure_parent(path)
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target
