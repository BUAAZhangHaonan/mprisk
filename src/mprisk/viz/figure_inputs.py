"""Artifact-backed figure input builders with explicit masks and provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mprisk.data.manifests import read_jsonl
from mprisk.state.identity import CALIBRATION_IDENTITY_FIELDS, require_matching_identity
from mprisk.state.patterns import StateThresholds, assign_state
from mprisk.state.spherical import DISTANCE_METRIC, SDR_SCHEMA, require_exact_sdr_rows
from mprisk.utils.io import write_json

PROVENANCE_SCHEMA = "mprisk_figure_input_provenance_v1"
PENDING_INPUT_SCHEMA = "mprisk_pending_figure_input_v1"
CONCEPTUAL_INPUT_SCHEMA = "mprisk_conceptual_figure_input_v1"
DATA_INDEPENDENT_FIGURES = {
    "fig01_problem_protocol",
    "fig02_representation_pipeline",
    "fig03_spherical_sdr",
    "figB1_representation_details",
}
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
    "fig07_misread_bias": [
        "panel",
        "model",
        "sample_id",
        "sample_type",
        "S",
        "D",
        "R",
        "direction_emphasized",
        "status",
    ],
    "fig08_representation_comparison": [
        "panel",
        "representation",
        "model",
        "protocol",
        "seed",
        "sample_id",
        "sample_type",
        "representation_split",
        "feature",
        "status",
    ],
    "figB2_prompt_stability_latency": [
        "model",
        "prompt_count",
        "stability",
        "latency_ms",
        "status",
    ],
    "figC1_ac_roc_pr": ["model", "representation", "curve", "x", "y", "status"],
    "figC2_conflict_retention": ["model", "budget", "metric", "value", "status"],
    "figC3_seed_robustness": ["model", "seed_a", "seed_b", "correlation", "agreement", "status"],
    "figC4_threshold_sensitivity": [
        "model",
        "kappa",
        "tau",
        "delta",
        "pattern",
        "proportion",
        "status",
    ],
    "figC5_model_patterns": ["model", "pattern", "proportion", "status"],
    "figD1_misread_pr": ["representation", "recall", "precision", "status"],
    "figD3_latency_breakdown": ["model", "component", "latency_ms", "status"],
    "figE1_human_quality": ["model", "dimension", "score", "status"],
}


@dataclass(frozen=True)
class StateFigureInputResult:
    fig04_path: Path
    fig04_provenance_path: Path
    fig05_path: Path
    fig05_provenance_path: Path
    fig06_path: Path
    fig06_provenance_path: Path
    fig07_path: Path
    fig07_provenance_path: Path


def build_state_figure_inputs(
    *,
    sdr_scores_path: str | Path | Sequence[str | Path],
    state_patterns_path: str | Path | Sequence[str | Path],
    thresholds_path: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    generated_command: list[str],
) -> StateFigureInputResult:
    """Build strict Fig. 4-6 CSV inputs from real state artifacts."""
    command = _validate_command(generated_command)
    score_files = _path_list(sdr_scores_path)
    pattern_files = _path_list(state_patterns_path)
    threshold_files = _path_list(thresholds_path)
    if not (len(score_files) == len(pattern_files) == len(threshold_files)):
        raise ValueError("state figure inputs require paired score/pattern/calibration artifacts")

    scores: list[dict[str, Any]] = []
    patterns: list[dict[str, Any]] = []
    thresholds_by_model: dict[str, dict[str, float]] = {}
    split_counts: Counter[str] = Counter()
    split_identities: list[dict[str, str]] = []
    calibration_identities: list[dict[str, str]] = []
    source_sample_count = 0
    for scores_file, patterns_file, thresholds_file in zip(
        score_files, pattern_files, threshold_files, strict=True
    ):
        source_scores = read_jsonl(scores_file)
        source_patterns = read_jsonl(patterns_file)
        thresholds = json.loads(thresholds_file.read_text(encoding="utf-8"))
        require_exact_sdr_rows(source_scores)
        _validate_calibration(thresholds)
        require_matching_identity(source_scores, thresholds)
        _validate_state_rows(source_scores, source_patterns)
        source_sample_count += len(source_scores)
        split_counts.update(str(row["representation_split"]) for row in source_scores)
        official_scores = [
            row for row in source_scores if row["representation_split"] == "official_test"
        ]
        official_patterns = [
            row for row in source_patterns if row["representation_split"] == "official_test"
        ]
        _validate_state_rows(official_scores, official_patterns, require_official_test=True)
        _validate_pattern_assignments(official_scores, official_patterns, thresholds)
        model_keys = {str(row["model_key"]) for row in official_scores}
        if len(model_keys) != 1:
            raise ValueError("each state artifact triple must contain exactly one model")
        model_key = next(iter(model_keys))
        if model_key in thresholds_by_model:
            raise ValueError(f"duplicate state artifact triple for model {model_key}")
        thresholds_by_model[model_key] = {
            "kappa": float(thresholds["kappa"]),
            "tau": float(thresholds["tau"]),
        }
        split_identities.append(
            {
                "model": model_key,
                "representation_split": "official_test",
                "split_assignment_sha256": str(thresholds["split_assignment_sha256"]),
            }
        )
        calibration_identities.append(
            {
                "model": model_key,
                **{
                    field: str(thresholds[field])
                    for field in CALIBRATION_IDENTITY_FIELDS
                },
            }
        )
        scores.extend(official_scores)
        patterns.extend(official_patterns)

    output_root = Path(output_dir)
    fig04_path = output_root / "fig04_sdr_distributions.csv"
    fig05_path = output_root / "fig05_four_state_stacks.csv"
    fig06_path = output_root / "fig06_stable_d_signed_r.csv"
    fig07_path = output_root / "fig07_misread_bias.csv"

    fig04_rows = _fig04_rows(scores, thresholds_by_model=thresholds_by_model)
    fig05_rows = _fig05_rows(patterns)
    fig06_rows = _fig06_rows(scores, thresholds_by_model=thresholds_by_model)
    fig07_rows = _fig07_rows(scores, thresholds_by_model=thresholds_by_model)
    _atomic_csv(fig04_path, FIGURE_CSV_FIELDS["fig04_sdr_distributions"], fig04_rows)
    _atomic_csv(fig05_path, FIGURE_CSV_FIELDS["fig05_four_state_stacks"], fig05_rows)
    _atomic_csv(fig06_path, FIGURE_CSV_FIELDS["fig06_stable_d_signed_r"], fig06_rows)
    _atomic_csv(fig07_path, FIGURE_CSV_FIELDS["fig07_misread_bias"], fig07_rows)

    common_provenance = {
        "representation_split": "official_test",
        "source_representation_split_counts": dict(sorted(split_counts.items())),
        "official_test_sample_count": len(scores),
        "excluded_non_official_test_count": source_sample_count - len(scores),
        "split_identities": split_identities,
        "calibration_identities": calibration_identities,
        "thresholds_by_model": thresholds_by_model,
    }

    fig04_provenance_path = _write_provenance(
        fig04_path,
        figure_key="fig04_sdr_distributions",
        generated_command=command,
        sources=[*score_files, *threshold_files],
        sample_masks={
            "S": "representation_split=official_test",
            "D": "representation_split=official_test and S<=kappa",
            "abs_R": "representation_split=official_test and S<=kappa and D>tau",
        },
        thresholds=None,
        source_sample_count=source_sample_count,
        included_sample_count=len(scores),
        sdr_contract={"sdr_schema": SDR_SCHEMA, "distance_metric": DISTANCE_METRIC},
        extra=common_provenance,
    )
    fig05_provenance_path = _write_provenance(
        fig05_path,
        figure_key="fig05_four_state_stacks",
        generated_command=command,
        sources=[*score_files, *pattern_files, *threshold_files],
        sample_masks={"patterns": "representation_split=official_test"},
        thresholds=None,
        source_sample_count=source_sample_count,
        included_sample_count=len(patterns),
        sdr_contract={"sdr_schema": SDR_SCHEMA, "distance_metric": DISTANCE_METRIC},
        extra=common_provenance,
    )
    fig06_provenance_path = _write_provenance(
        fig06_path,
        figure_key="fig06_stable_d_signed_r",
        generated_command=command,
        sources=[*score_files, *threshold_files],
        sample_masks={
            "points": "S<=kappa",
            "direction_emphasis": "S<=kappa and D>tau",
        },
        thresholds=None,
        source_sample_count=source_sample_count,
        included_sample_count=len(fig06_rows),
        sdr_contract={"sdr_schema": SDR_SCHEMA, "distance_metric": DISTANCE_METRIC},
        extra=common_provenance,
    )
    fig07_provenance_path = _write_provenance(
        fig07_path,
        figure_key="fig07_misread_bias",
        generated_command=command,
        sources=[*score_files, *threshold_files],
        sample_masks={
            "misread": "Pending Misread annotations",
            "bias": "representation_split=official_test and sample_type=Conflict and S<=kappa",
            "direction_emphasis": "D>tau",
        },
        thresholds=None,
        source_sample_count=source_sample_count,
        included_sample_count=len(fig07_rows),
        sdr_contract={"sdr_schema": SDR_SCHEMA, "distance_metric": DISTANCE_METRIC},
        extra=common_provenance,
    )
    return StateFigureInputResult(
        fig04_path,
        fig04_provenance_path,
        fig05_path,
        fig05_provenance_path,
        fig06_path,
        fig06_provenance_path,
        fig07_path,
        fig07_provenance_path,
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
                data_independent = str(figure_key) in DATA_INDEPENDENT_FIGURES
                write_json(
                    input_path,
                    {
                        "schema": (
                            CONCEPTUAL_INPUT_SCHEMA
                            if data_independent
                            else PENDING_INPUT_SCHEMA
                        ),
                        "figure_key": str(figure_key),
                        "status": READY if data_independent else PENDING,
                        "generated_command": command,
                        "sources": [],
                        "sample_masks": {
                            "data_dependency": "none"
                            if data_independent
                            else "source_artifact_required"
                        },
                        "rows": [],
                    },
                )
                written.append(input_path)
            else:
                raise ValueError(f"unsupported figure input extension: {input_path}")
    return written


def _fig04_rows(
    rows: list[dict[str, Any]], *, thresholds_by_model: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
        base = _state_values(row)
        thresholds = thresholds_by_model[base["model"]]
        kappa = thresholds["kappa"]
        tau = thresholds["tau"]
        exported.append({**base, "metric": "S", "value": base["S"]})
        if float(base["S"]) <= kappa:
            exported.append({**base, "metric": "D", "value": base["D"]})
            if float(base["D"]) > tau:
                exported.append({**base, "metric": "abs_R", "value": abs(float(base["R"]))})
    return exported


def _fig05_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (str(row["model_key"]), str(row["sample_type"]), str(row["pattern"])) for row in rows
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
    rows: list[dict[str, Any]], *, thresholds_by_model: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
        base = _state_values(row)
        thresholds = thresholds_by_model[base["model"]]
        kappa = thresholds["kappa"]
        tau = thresholds["tau"]
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


def _fig07_rows(
    rows: list[dict[str, Any]], *, thresholds_by_model: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    exported: list[dict[str, Any]] = []
    for row in rows:
        base = _state_values(row)
        thresholds = thresholds_by_model[base["model"]]
        if base["sample_type"] != "Conflict" or float(base["S"]) > thresholds["kappa"]:
            continue
        exported.append(
            {
                "panel": "bias",
                **base,
                "direction_emphasized": str(float(base["D"]) > thresholds["tau"]).lower(),
                "status": READY,
            }
        )
    if not exported:
        raise ValueError("Fig. 7 bias panels require stable official-test Conflict samples")
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
    scores: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
    *,
    require_official_test: bool = False,
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
    score_metadata = {
        str(row["sample_id"]): (
            str(row.get("model_key", "")),
            str(row.get("sample_type", "")),
            str(row.get("representation_split", "")),
        )
        for row in scores
    }
    pattern_metadata = {
        str(row["sample_id"]): (
            str(row.get("model_key", "")),
            str(row.get("sample_type", "")),
            str(row.get("representation_split", "")),
        )
        for row in patterns
    }
    if score_metadata != pattern_metadata:
        raise ValueError("S/D/R and pattern metadata must match exactly")
    valid_splits = {"relation_train", "relation_val", "aligned_calibration", "official_test"}
    if any(metadata[2] not in valid_splits for metadata in score_metadata.values()):
        raise ValueError("state figure rows require a registered representation_split")
    if require_official_test:
        if any(metadata[2] != "official_test" for metadata in score_metadata.values()):
            raise ValueError(
                "paper state figures may include only representation_split=official_test"
            )
        if {metadata[1] for metadata in score_metadata.values()} != {"Aligned", "Conflict"}:
            raise ValueError(
                "official-test state figures require both Aligned and Conflict samples"
            )


def _validate_calibration(thresholds: dict[str, Any]) -> None:
    if (
        thresholds.get("schema") != "mprisk_spherical_calibration_v2"
        or thresholds.get("sdr_schema") != SDR_SCHEMA
        or thresholds.get("distance_metric") != DISTANCE_METRIC
    ):
        raise ValueError("figure thresholds must use exact spherical SDR v2 calibration")
    if (
        thresholds.get("sample_type") != "Aligned"
        or thresholds.get("calibration_split") != "aligned_calibration"
        or thresholds.get("selection_rule")
        != "representation_split=aligned_calibration then sample_type=Aligned"
    ):
        raise ValueError(
            "paper thresholds must come only from the registered Aligned calibration split"
        )


def _validate_pattern_assignments(
    scores: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> None:
    threshold_values = StateThresholds.from_dict(thresholds)
    patterns_by_id = {str(row["sample_id"]): row for row in patterns}
    for score in scores:
        sample_id = str(score["sample_id"])
        expected = assign_state(
            float(score["S_mean"]),
            float(score["D"]),
            float(score["R"]),
            threshold_values,
            delta_i=float(score["delta_i"]),
        ).value
        if patterns_by_id[sample_id].get("pattern") != expected:
            raise ValueError(
                f"state pattern does not match hierarchical S/D/R assignment: {sample_id}"
            )


def _path_list(value: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        paths = [Path(value)]
    else:
        paths = [Path(item) for item in value]
    if not paths:
        raise ValueError("state figure artifact lists must be non-empty")
    return paths


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
    extra: dict[str, Any] | None = None,
) -> Path:
    path = provenance_path(csv_path)
    payload: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "figure_key": figure_key,
        "status": status,
        "generated_command": generated_command,
        "sources": [{"path": str(source), "sha256": _sha256(source)} for source in sources],
        "sample_masks": sample_masks,
        "source_sample_count": source_sample_count,
        "included_sample_count": included_sample_count,
    }
    if thresholds is not None:
        payload["thresholds"] = thresholds
    if sdr_contract is not None:
        payload.update(sdr_contract)
    if extra is not None:
        collisions = set(payload) & set(extra)
        if collisions:
            raise ValueError(f"provenance fields collide: {', '.join(sorted(collisions))}")
        payload.update(extra)
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
