"""Threshold calibration helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from mprisk.state.spherical import DISTANCE_METRIC, SDR_SCHEMA, require_exact_sdr_rows


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of empty values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def calibrate_aligned_thresholds(
    rows: list[dict[str, Any]],
    *,
    quantile_level: float = 0.95,
) -> dict[str, Any]:
    """Calibrate kappa and tau on an independent Aligned calibration split only."""
    if not 0.0 < quantile_level < 1.0:
        raise ValueError("quantile_level must be in (0, 1)")
    if not rows:
        raise ValueError("Aligned calibration requires at least one row")
    if any(row.get("sample_type") != "Aligned" for row in rows):
        raise ValueError("Aligned calibration must not contain Conflict samples")
    if any(row.get("calibration_split") != "aligned_calibration" for row in rows):
        raise ValueError("Aligned calibration rows require calibration_split=aligned_calibration")
    require_exact_sdr_rows(rows)
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Aligned calibration sample IDs must be non-empty and unique")
    kappa = quantile([float(row["S_mean"]) for row in rows], quantile_level)
    stable_rows = [row for row in rows if float(row["S_mean"]) <= kappa]
    tau = quantile([float(row["D"]) for row in stable_rows], quantile_level)
    signature = hashlib.sha256(
        json.dumps(sorted(sample_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "mprisk_spherical_calibration_v2",
        "sdr_schema": SDR_SCHEMA,
        "distance_metric": DISTANCE_METRIC,
        "sample_type": "Aligned",
        "calibration_split": "aligned_calibration",
        "quantile_level": quantile_level,
        "kappa": kappa,
        "tau": tau,
        "aligned_count": len(rows),
        "stable_aligned_count": len(stable_rows),
        "sample_ids_sha256": signature,
    }


def calibrate_registered_aligned_thresholds(
    rows: list[dict[str, Any]],
    *,
    quantile_level: float = 0.95,
) -> dict[str, Any]:
    """Select the registered calibration partition, then filter its Aligned rows."""
    if not rows:
        raise ValueError("registered calibration requires non-empty SDR rows")
    checksums = {str(row.get("split_assignment_sha256", "")) for row in rows}
    if len(checksums) != 1 or len(next(iter(checksums))) != 64:
        raise ValueError("SDR rows require one valid split assignment checksum")
    registered = [
        row for row in rows if row.get("representation_split") == "aligned_calibration"
    ]
    if not registered:
        raise ValueError("no rows use the registered aligned_calibration split")
    aligned = [row for row in registered if row.get("sample_type") == "Aligned"]
    if not aligned:
        raise ValueError("registered calibration split has no Aligned rows")
    payload = calibrate_aligned_thresholds(aligned, quantile_level=quantile_level)
    payload.update(
        {
            "input_count": len(rows),
            "registered_calibration_count": len(registered),
            "excluded_non_aligned_calibration_count": len(registered) - len(aligned),
            "split_assignment_sha256": next(iter(checksums)),
            "selection_rule": "representation_split=aligned_calibration then sample_type=Aligned",
        }
    )
    return payload
