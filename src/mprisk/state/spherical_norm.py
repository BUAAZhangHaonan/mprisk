"""Normalized SDR overrides.

Mainline mprisk defines:
    S = mean squared geodesic distance from per-prompt z to its condition center
    D = d(M1, M2) / sqrt(S_M1 + S_M2)              # unbounded, can be huge
    R = (d(M12, M2) - d(M12, M1)) / d(M1, M2)       # ~[-1, 1] but can exceed

^This module normalizes them to the user-requested ranges:
    S_norm = S / (pi^2)                              # in [0, 1]
    D_norm = d(M1, M2) / pi                          # in [0, 1]
    R     = same formula, but clip to [-1, 1]
    delta_i = min(1.96 * SE_R, 1.0)                  # cap to [0, 1]

Thresholds are then calibrated on normalized quantities so:
    kappa in [0, 1], tau in [0, 1], delta_i in [0, 1].
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from mprisk.state import spherical as _sph

_ORIG_COMPUTE = _sph.compute_spherical_state
EPSILON = _sph.EPSILON
CONDITIONS = ("M1", "M2", "M12")


def _unit(v):
    arr = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(arr)
    if n <= 1e-12:
        raise ValueError("zero-norm vector")
    return arr / n


def _geodesic(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _center(vectors):
    arr = np.stack([_unit(v) for v in vectors])
    m = arr.mean(axis=0)
    n = np.linalg.norm(m)
    if n <= 1e-12:
        raise ValueError("antipodal center")
    return m / n


def compute_spherical_state(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Same input/output contract as mprisk.state.spherical.compute_spherical_state,
    but with D and delta normalized to [0, 1].
    """
    embeddings = bundle["embeddings"]
    prompt_ids = sorted(set(embeddings["M1"].keys()))
    for cond in CONDITIONS[1:]:
        if set(embeddings[cond].keys()) != set(prompt_ids):
            raise ValueError(f"prompt ids disagree on {cond}")

    normalized = {
        cond: {pid: _unit(embeddings[cond][pid]).tolist() for pid in prompt_ids}
        for cond in CONDITIONS
    }
    centers = {
        cond: _center([normalized[cond][pid] for pid in prompt_ids])
        for cond in CONDITIONS
    }

    s_per_cond = {
        cond: sum(_geodesic(normalized[cond][pid], centers[cond]) ** 2 for pid in prompt_ids) / len(prompt_ids)
        for cond in CONDITIONS
    }
    s_mean = sum(s_per_cond.values()) / 3.0

    d_m1_m2 = _geodesic(centers["M1"], centers["M2"])
    d_m12_m1 = _geodesic(centers["M12"], centers["M1"])
    d_m12_m2 = _geodesic(centers["M12"], centers["M2"])

    # Normalized D: pure angular distance on unit sphere, in [0, 1].
    d_norm = d_m1_m2 / math.pi
    # Normalized S: divide by max possible value (pi^2), in [0, 1].
    s_norm = s_mean / (math.pi ** 2)

    # R: keep the signed ratio but clip to [-1, 1].
    r_signed = (d_m12_m2 - d_m12_m1) / (d_m1_m2 + EPSILON)
    r_clipped = float(np.clip(r_signed, -1.0, 1.0))

    # Per-prompt R for SE.
    prompt_r = []
    for pid in prompt_ids:
        d_m1_m2_p = _geodesic(normalized["M1"][pid], normalized["M2"][pid])
        d_m12_m1_p = _geodesic(normalized["M12"][pid], normalized["M1"][pid])
        d_m12_m2_p = _geodesic(normalized["M12"][pid], normalized["M2"][pid])
        prompt_r.append((d_m12_m2_p - d_m12_m1_p) / (d_m1_m2_p + EPSILON))
    prompt_r_arr = np.asarray(prompt_r, dtype=np.float64)
    if len(prompt_r) > 1:
        prompt_se = float(np.std(prompt_r_arr, ddof=1) / math.sqrt(len(prompt_r)))
    else:
        prompt_se = 0.0

    delta_i = min(1.96 * prompt_se, 1.0)

    return {
        "sdr_schema": "mprisk_spherical_sdr",
        "distance_metric": "geodesic_acos_v1",
        "sample_id": bundle.get("sample_id"),
        "sample_type": bundle.get("sample_type"),
        "calibration_split": bundle.get("calibration_split") or bundle.get("representation_split", ""),
        "representation_split": bundle.get("representation_split", ""),
        "master_split": bundle.get("master_split", ""),
        "prompt_ids": prompt_ids,
        # Raw (for traceability)
        "S_M1_raw": s_per_cond["M1"],
        "S_M2_raw": s_per_cond["M2"],
        "S_M12_raw": s_per_cond["M12"],
        "S_mean_raw": s_mean,
        "d_M1_M2_raw": d_m1_m2,
        "d_M12_M1_raw": d_m12_m1,
        "d_M12_M2_raw": d_m12_m2,
        # Normalized outputs
        "S_mean": float(s_norm),
        "D": float(d_norm),
        "R": float(r_clipped),
        "R_prompt_values": [float(x) for x in prompt_r],
        "R_prompt_se": prompt_se,
        "delta_i": float(delta_i),
        "delta_policy": "v2_capped_1.96_prompt_se",
        "lean": "V" if r_clipped > 0.0 else "T/A" if r_clipped < 0.0 else "Balanced",
    }


def install_normalization() -> None:
    """Monkey-patch mprisk.state.spherical.compute_spherical_state to the normalized version."""
    _sph.compute_spherical_state = compute_spherical_state
