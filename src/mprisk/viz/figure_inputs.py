"""Artifact-backed figure input builders with explicit masks and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mprisk.data.manifests import read_jsonl
from mprisk.state.spherical import DISTANCE_METRIC, SDR_SCHEMA, require_exact_sdr_rows
from mprisk.utils.io import write_json

PROVENANCE_SCHEMA = "mprisk_figure_input_provenance_v1"
PENDING_INPUT_SCHEMA = "mprisk_pending_figure_input_v1"
READY = "Ready"
PENDING = "Pending"

FIGURE_CSV_FIELDS = {
    "fig04_sdr_distributions": [
        "sample_id",
        "model",
        "sample_type",
        "S",
        "D",
        "R",
        "metric",
        "value",
    ],
    "fig05_four_state_stacks": [
        "model",
        "sample_type",
        "pattern",
        "count",
        "total",
        "proportion",
    ],
    "fig06_stable_d_signed_r": [
        "sample_id",
        "model",
        "sample_type",
        "S",
        "D",
        "R",
        "stable",
        "direction_emphasized",
        "lean",
    ],
    "fig07_misread_bias": ["panel", "category", "value", "status"],
    "fig08_representation_comparison": [
        "panel",
        "representation",
        "metric",
        "value",
        "status",
    ],
}


@dataclass(frozen=True)
class StateFigureInputResult:
    fig04_path: Path
    fig04_provenance_path: Path
    fig05_path: Path
    fig05_provenance_path: Path
    fig06_path: Path
    fig06_provenance_path: Path


def build_state_figure_inputs(
    *,
    sdr_scores_path: str | Path,
    state_patterns_path: str | Path,
    thresholds_path: str | Path,
    output_dir: str | Path,
    generated_command: list[str],
) -> StateFigureInputResult:
    """Build strict Fig. 4-6 CSV inputs from real state artifacts."""
    command = _validate_command(generated_command)
    scores_file = Path(sdr_scores_path)
    patterns_file = Path(state_patterns_path)
    thresholds_file = Path(thresholds_path)
    scores = read_jsonl(scores_file)
    patterns = read_jsonl(patterns_file)
    thresholds = json.loads(thresholds_file.read_text(encoding="utf-8"))
    require_exact_sdr_rows(scores)
    if (
        thresholds.get("schema") != "mprisk_spherical_calibration_v2"
        or thresholds.get("sdr_schema") != SDR_SCHEMA
        or thresholds.get("distance_metric") != DISTANCE_METRIC
    ):
        raise ValueError("figure thresholds must use exact spherical SDR v2 calibration")
    kappa = float(thresholds["kappa"])
    tau = float(thresholds["tau"])
    _validate_state_rows(scores, patterns)

    output_root = Path(output_dir)
    fig04_path = output_root / "fig04_sdr_distributions.csv"
    fig05_path = output_root / "fig05_four_state_stacks.csv"
    fig06_path = output_root / "fig06_stable_d_signed_r.csv"

    fig04_rows = _fig04_rows(scores, kappa=kappa, tau=tau)
    fig05_rows = _fig05_rows(patterns)
    fig06_rows = _fig06_rows(scores, kappa=kappa, tau=tau)
    _atomic_csv(fig04_path, FIGURE_CSV_FIELDS["fig04_sdr_distributions"], fig04_rows)
    _atomic_csv(fig05_path, FIGURE_CSV_FIELDS["fig05_four_state_stacks"], fig05_rows)
    _atomic_csv(fig06_path, FIGURE_CSV_FIELDS["fig06_stable_d_signed_r"], fig06_rows)

    fig04_provenance_path = _write_provenance(
        fig04_path,
        figure_key="fig04_sdr_distributions",
        generated_command=command,
        sources=[scores_file, thresholds_file],
        sample_masks={
            "S": "all_samples",
            "D": "S<=kappa",
            "abs_R": "S<=kappa and D>tau",
        },
        thresholds={"kappa": kappa, "tau": tau},
        source_sample_count=len(scores),
        included_sample_count=len(scores),
        sdr_contract={"sdr_schema": SDR_SCHEMA, "distance_metric": DISTANCE_METRIC},
    )
    fig05_provenance_path = _write_provenance(
        fig05_path,
        figure_key="fig05_four_state_stacks",
        generated_command=command,
        sources=[patterns_file],
        sample_masks={"patterns": "all_samples"},
        thresholds=None,
        source_sample_count=len(patterns),
        included_sample_count=len(patterns),
        sdr_contract={"sdr_schema": SDR_SCHEMA, "distance_metric": DISTANCE_METRIC},
    )
    fig06_provenance_path = _write_provenance(
        fig06_path,
        figure_key="fig06_stable_d_signed_r",
        generated_command=command,
        sources=[scores_file, thresholds_file],
        sample_masks={
            "points": "S<=kappa",
            "direction_emphasis": "S<=kappa and D>tau",
        },
        thresholds={"kappa": kappa, "tau": tau},
        source_sample_count=len(scores),
        included_sample_count=len(fig06_rows),
        sdr_contract={"sdr_schema": SDR_SCHEMA, "distance_metric": DISTANCE_METRIC},
    )
    return StateFigureInputResult(
        fig04_path,
        fig04_provenance_path,
        fig05_path,
        fig05_provenance_path,
        fig06_path,
        fig06_provenance_path,
    )


def write_pending_figure_inputs(
    config_path: str | Path,
    *,
    generated_command: list[str],
) -> list[Path]:
    """Materialize explicit Pending inputs without inventing observations."""
    command = _validate_command(generated_command)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    written: list[Path] = []
    for group_name in ("figures", "appendix"):
        for figure_key, spec in (config.get(group_name) or {}).items():
            input_path = Path(str(spec["input"]))
            if input_path.exists():
                continue
            input_path.parent.mkdir(parents=True, exist_ok=True)
            if input_path.suffix.casefold() == ".csv":
                fields = FIGURE_CSV_FIELDS.get(str(figure_key), ["status"])
                _atomic_csv(input_path, fields, [])
                provenance_path = _write_provenance(
                    input_path,
                    figure_key=str(figure_key),
                    generated_command=command,
                    sources=[],
                    sample_masks={"status": "no_source_artifact"},
                    thresholds=None,
                    source_sample_count=0,
                    included_sample_count=0,
                    status=PENDING,
                )
                written.extend((input_path, provenance_path))
            elif input_path.suffix.casefold() == ".json":
                write_json(
                    input_path,
                    {
                        "schema": PENDING_INPUT_SCHEMA,
                        "figure_key": str(figure_key),
                        "status": PENDING,
                        "generated_command": command,
                        "sources": [],
                        "sample_masks": {"status": "no_source_artifact"},
                        "rows": [],
                    },
                )
                written.append(input_path)
            else:
                raise ValueError(f"unsupported figure input extension: {input_path}")
    return written


def _fig04_rows(
    rows: list[dict[str, Any]], *, kappa: float, tau: float
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
        base = _state_values(row)
        exported.append({**base, "metric": "S", "value": base["S"]})
        if float(base["S"]) <= kappa:
            exported.append({**base, "metric": "D", "value": base["D"]})
            if float(base["D"]) > tau:
                exported.append({**base, "metric": "abs_R", "value": abs(float(base["R"]))})
    return exported


def _fig05_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (str(row["model_key"]), str(row["sample_type"]), str(row["pattern"]))
        for row in rows
    )
    totals = Counter((str(row["model_key"]), str(row["sample_type"])) for row in rows)
    return [
        {
            "model": model,
            "sample_type": sample_type,
            "pattern": pattern,
            "count": count,
            "total": totals[(model, sample_type)],
            "proportion": count / totals[(model, sample_type)],
        }
        for (model, sample_type, pattern), count in sorted(counts.items())
    ]


def _fig06_rows(
    rows: list[dict[str, Any]], *, kappa: float, tau: float
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
        base = _state_values(row)
        if float(base["S"]) > kappa:
            continue
        emphasized = float(base["D"]) > tau
        r_value = float(base["R"])
        exported.append(
            {
                **base,
                "stable": "true",
                "direction_emphasized": str(emphasized).lower(),
                "lean": "V" if r_value > 0 else "T/A" if r_value < 0 else "Balanced",
            }
        )
    return exported


def _state_values(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row["sample_id"]),
        "model": str(row["model_key"]),
        "sample_type": str(row["sample_type"]),
        "S": float(row["S_mean"]),
        "D": float(row["D"]),
        "R": float(row["R"]),
    }


def _validate_state_rows(
    scores: list[dict[str, Any]], patterns: list[dict[str, Any]]
) -> None:
    if not scores or not patterns:
        raise ValueError("state figure inputs require non-empty real artifacts")
    score_ids = [str(row.get("sample_id", "")) for row in scores]
    pattern_ids = [str(row.get("sample_id", "")) for row in patterns]
    if len(set(score_ids)) != len(score_ids) or len(set(pattern_ids)) != len(pattern_ids):
        raise ValueError("state figure sample IDs must be unique")
    if set(score_ids) != set(pattern_ids):
        raise ValueError("S/D/R and pattern inputs must have exact sample correspondence")
    if any(row.get("sample_type") not in {"Aligned", "Conflict"} for row in scores):
        raise ValueError("state figure rows require Aligned or Conflict sample_type")


def _write_provenance(
    csv_path: Path,
    *,
    figure_key: str,
    generated_command: list[str],
    sources: list[Path],
    sample_masks: dict[str, str],
    thresholds: dict[str, float] | None,
    source_sample_count: int,
    included_sample_count: int,
    status: str = READY,
    sdr_contract: dict[str, str] | None = None,
) -> Path:
    path = provenance_path(csv_path)
    payload: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "figure_key": figure_key,
        "status": status,
        "generated_command": generated_command,
        "sources": [
            {"path": str(source), "sha256": _sha256(source)} for source in sources
        ],
        "sample_masks": sample_masks,
        "source_sample_count": source_sample_count,
        "included_sample_count": included_sample_count,
    }
    if thresholds is not None:
        payload["thresholds"] = thresholds
    if sdr_contract is not None:
        payload.update(sdr_contract)
    write_json(path, payload)
    return path


def provenance_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_suffix(path.suffix + ".provenance.json")


def _atomic_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_command(command: list[str]) -> list[str]:
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("generated_command must be a non-empty argv list")
    return list(command)
