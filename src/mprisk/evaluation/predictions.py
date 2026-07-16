"""Prediction row schema and label normalization for Stage 1 evaluation."""

from __future__ import annotations

import json
import string
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MAIN_LABELS = frozenset({"positive", "negative", "neutral"})
ABSTAIN_LABELS = frozenset({"uncertain", "invalid"})
NORMALIZED_LABELS = MAIN_LABELS | ABSTAIN_LABELS

REQUIRED_PREDICTION_FIELDS = frozenset(
    {
        "sample_id",
        "model_key",
        "protocol",
        "prediction",
        "target_label",
        "confidence_text",
        "raw_response",
        "source",
    }
)

_STRING_FIELDS = frozenset(
    {
        "sample_id",
        "model_key",
        "protocol",
        "prediction",
        "target_label",
        "raw_response",
        "source",
    }
)

_LABEL_SYNONYMS = {
    "positive": "positive",
    "pos": "positive",
    "+": "positive",
    "yes": "positive",
    "true": "positive",
    "negative": "negative",
    "neg": "negative",
    "-": "negative",
    "no": "negative",
    "false": "negative",
    "neutral": "neutral",
    "neut": "neutral",
    "neither": "neutral",
    "balanced": "neutral",
    "uncertain": "uncertain",
    "unclear": "uncertain",
    "unknown": "uncertain",
    "unsure": "uncertain",
    "not sure": "uncertain",
    "cannot determine": "uncertain",
    "cant determine": "uncertain",
    "can't determine": "uncertain",
    "i cannot determine": "uncertain",
    "abstain": "uncertain",
    "ambiguous": "uncertain",
    "invalid": "invalid",
    "malformed": "invalid",
    "error": "invalid",
    "other": "invalid",
    "n/a": "invalid",
    "na": "invalid",
    "none": "invalid",
    "": "invalid",
}


def normalize_label(value: object) -> str:
    """Normalize a coarse Stage 1 label, returning invalid for unknown labels."""

    if value is None:
        return "invalid"
    text = str(value).strip().lower()
    text = text.strip(string.whitespace + "\"'`“”‘’")
    text = text.strip(".,:;!?")
    text = " ".join(text.split())
    return _LABEL_SYNONYMS.get(text, "invalid")


def normalize_prediction_row(
    row: Mapping[str, Any],
    *,
    line_number: int | None = None,
) -> dict[str, Any]:
    """Validate and normalize a Stage 1 prediction row."""

    missing = REQUIRED_PREDICTION_FIELDS - row.keys()
    if missing:
        location = f" on line {line_number}" if line_number is not None else ""
        raise ValueError(f"Missing prediction field(s){location}: {', '.join(sorted(missing))}")

    for field in sorted(_STRING_FIELDS):
        if not isinstance(row[field], str):
            location = f" on line {line_number}" if line_number is not None else ""
            raise TypeError(f"Prediction field {field!r}{location} must be a string")

    if row["confidence_text"] is not None and not isinstance(row["confidence_text"], str):
        location = f" on line {line_number}" if line_number is not None else ""
        raise TypeError(f"Prediction field 'confidence_text'{location} must be a string or None")

    if "is_correct" in row and type(row["is_correct"]) is not bool:
        location = f" on line {line_number}" if line_number is not None else ""
        raise TypeError(f"Prediction field 'is_correct'{location} must be a bool")

    normalized = dict(row)
    normalized["normalized_prediction"] = normalize_label(row["prediction"])
    normalized["normalized_target_label"] = normalize_label(row["target_label"])
    return normalized


def read_prediction_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read Stage 1 prediction JSONL and return validated, normalized rows."""

    rows: list[dict[str, Any]] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise TypeError(f"Prediction row on line {line_number} must be a JSON object")
            rows.append(normalize_prediction_row(payload, line_number=line_number))
    return rows


def write_prediction_jsonl(rows: list[Mapping[str, Any]], path: str | Path) -> Path:
    """Write normalized prediction rows as JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return output_path
