"""CSV/JSON figure input loading and provenance/mask validators."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mprisk.viz.figure_inputs import (
    CONCEPTUAL_INPUT_SCHEMA,
    PENDING_INPUT_SCHEMA,
    PROVENANCE_SCHEMA,
    provenance_path,
)

from .figure_constants import CONCEPTUAL_KEYS, MODEL_SPECS, STATUS_PENDING, STATUS_READY
from .figure_validators import _as_bool, _is_sha256, _sha256


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_figure_input(
    figure_key: str,
    input_path: Path,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not input_path.is_file() or input_path.stat().st_size == 0:
        if figure_key in CONCEPTUAL_KEYS:
            raise ValueError(f"conceptual figure input is missing or empty: {input_path}")
        return STATUS_PENDING, [], {}
    suffix = input_path.suffix.casefold()
    if suffix == ".csv":
        sidecar = provenance_path(input_path)
        if not sidecar.is_file():
            raise ValueError(f"Ready CSV figure input requires provenance sidecar: {sidecar}")
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        _validate_provenance(figure_key, provenance)
        status = str(provenance.get("status"))
        return status, _read_csv(input_path) if status == STATUS_READY else [], provenance
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON figure inputs must use a provenance envelope")
        if payload.get("schema") == PENDING_INPUT_SCHEMA:
            if payload.get("figure_key") != figure_key or payload.get("status") != STATUS_PENDING:
                raise ValueError("Pending JSON figure input identity/status mismatch")
            return STATUS_PENDING, [], payload
        if payload.get("schema") == CONCEPTUAL_INPUT_SCHEMA:
            if (
                figure_key not in CONCEPTUAL_KEYS
                or payload.get("figure_key") != figure_key
                or payload.get("status") != STATUS_READY
                or payload.get("sources") != []
                or payload.get("sample_masks") != {"data_dependency": "none"}
            ):
                raise ValueError("conceptual figure input identity/status mismatch")
            return STATUS_READY, [], payload
        _validate_provenance(figure_key, payload)
        rows = payload.get("rows")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Ready JSON figure input rows must be a list of objects")
        return str(payload["status"]), rows, payload
    raise ValueError(f"figure input must be CSV or JSON: {input_path}")


def _validate_provenance(figure_key: str, provenance: dict[str, Any]) -> None:
    if provenance.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError(f"figure provenance schema must be {PROVENANCE_SCHEMA}")
    if provenance.get("figure_key") != figure_key:
        raise ValueError("figure provenance key mismatch")
    if provenance.get("status") not in {STATUS_READY, STATUS_PENDING}:
        raise ValueError("figure provenance status must be Ready or Pending")
    if provenance.get("status") == STATUS_PENDING:
        return
    command = provenance.get("generated_command")
    sources = provenance.get("sources")
    if not isinstance(command, list) or not command:
        raise ValueError("Ready figure provenance requires generated_command argv")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Ready figure provenance requires source hashes")
    for source in sources:
        if (
            not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or not _is_sha256(source.get("sha256"))
        ):
            raise ValueError("figure provenance source path/sha256 is invalid")
        source_path = Path(source["path"])
        if not source_path.is_file() or _sha256(source_path) != source["sha256"]:
            raise ValueError(f"figure provenance source checksum mismatch: {source_path}")


def _validate_fig04_masks(rows: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    masks = provenance.get("sample_masks") or {}
    if masks != {
        "S": "representation_split=official_test",
        "D": "representation_split=official_test and S<=kappa",
        "abs_R": "representation_split=official_test and S<=kappa and D>tau",
    }:
        raise ValueError("Fig. 4 sample masks do not match the locked contract")
    thresholds_by_model = provenance.get("thresholds_by_model") or {}
    for row in rows:
        thresholds = thresholds_by_model.get(row["model"])
        if not isinstance(thresholds, dict):
            raise ValueError("Fig. 4 is missing per-model calibration thresholds")
        kappa = float(thresholds["kappa"])
        tau = float(thresholds["tau"])
        metric = row["metric"]
        if metric not in {"S", "D", "abs_R"}:
            raise ValueError("Fig. 4 metric must be S, D, or abs_R")
        if metric == "D" and float(row["S"]) > kappa:
            raise ValueError("Fig. 4 D row violates stable mask")
        if metric == "abs_R" and (float(row["S"]) > kappa or float(row["D"]) <= tau):
            raise ValueError("Fig. 4 abs_R row violates stable directional mask")
        expected = (
            float(row["S"])
            if metric == "S"
            else float(row["D"])
            if metric == "D"
            else abs(float(row["R"]))
        )
        if not abs(float(row["value"]) - expected) <= 1e-9:
            raise ValueError("Fig. 4 metric value does not match source S/D/R")


def _validate_fig06_masks(rows: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    masks = provenance.get("sample_masks") or {}
    if masks != {
        "points": "S<=kappa",
        "direction_emphasis": "S<=kappa and D>tau",
    }:
        raise ValueError("Fig. 6 sample masks do not match the locked contract")
    thresholds_by_model = provenance.get("thresholds_by_model") or {}
    for row in rows:
        thresholds = thresholds_by_model.get(row["model"])
        if not isinstance(thresholds, dict):
            raise ValueError("Fig. 6 is missing per-model calibration thresholds")
        kappa = float(thresholds["kappa"])
        tau = float(thresholds["tau"])
        if float(row["S"]) > kappa or not _as_bool(row["stable"]):
            raise ValueError("Fig. 6 stable mask violation")
        if _as_bool(row["direction_emphasized"]) != (float(row["D"]) > tau):
            raise ValueError("Fig. 6 direction emphasis mask violation")


def _validate_state_provenance(
    rows: list[dict[str, Any]], provenance: dict[str, Any]
) -> None:
    if provenance.get("representation_split") != "official_test":
        raise ValueError("paper state figures require representation_split=official_test")
    source_count = provenance.get("source_sample_count")
    official_count = provenance.get("official_test_sample_count")
    excluded_count = provenance.get("excluded_non_official_test_count")
    counts = (source_count, official_count, excluded_count)
    if not all(isinstance(value, int) and value >= 0 for value in counts):
        raise ValueError(
            "paper state provenance requires non-negative source/included/excluded counts"
        )
    if source_count != official_count + excluded_count:
        raise ValueError("paper state provenance source count does not reconcile")
    models = {str(row["model"]) for row in rows}
    if models != {key for key, _ in MODEL_SPECS}:
        raise ValueError("paper state figures require all three registered model facets")
    thresholds_by_model = provenance.get("thresholds_by_model")
    if not isinstance(thresholds_by_model, dict) or set(thresholds_by_model) != models:
        raise ValueError("paper state provenance requires per-model calibration thresholds")
    split_identities = provenance.get("split_identities")
    calibration_identities = provenance.get("calibration_identities")
    if not isinstance(split_identities, list) or {
        str(item.get("model")) for item in split_identities if isinstance(item, dict)
    } != models:
        raise ValueError("paper state provenance requires one split identity per model")
    if any(
        item.get("representation_split") != "official_test"
        or not _is_sha256(item.get("split_assignment_sha256"))
        for item in split_identities
    ):
        raise ValueError("paper state split identity is invalid")
    if not isinstance(calibration_identities, list) or {
        str(item.get("model")) for item in calibration_identities if isinstance(item, dict)
    } != models:
        raise ValueError("paper state provenance requires one calibration identity per model")
    for identity in calibration_identities:
        if identity.get("model_key") != identity.get("model"):
            raise ValueError("paper state calibration model identity mismatch")
        if any(not str(identity.get(field, "")) for field in (
            "protocol",
            "prompt_set_key",
            "repr_key",
            "prompt_set_artifact_sha256",
            "encoder_checkpoint_sha256",
            "split_assignment_sha256",
            "embedding_manifest_sha256",
        )):
            raise ValueError("paper state calibration identity is incomplete")
