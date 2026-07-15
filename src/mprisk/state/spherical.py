"""Strict unit-hypersphere S/D/R bundle measurements."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from statistics import stdev
from typing import Any

import numpy as np

CONDITIONS = ("M1", "M2", "M12")
UNIT_TOLERANCE = 1e-5
BOOTSTRAP_REPLICATES = 2000
DELTA_METHOD = "synchronous_prompt_bootstrap_1.96se_v1"


def spherical_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = _unit_vector(left)
    right_array = _unit_vector(right)
    return float(1.0 - np.clip(np.dot(left_array, right_array), -1.0, 1.0))


def spherical_center(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot compute a spherical center from no vectors")
    array = np.stack([_unit_vector(vector) for vector in vectors])
    mean = array.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12:
        raise ValueError("spherical center is undefined for antipodal embeddings")
    return (mean / norm).tolist()


def compute_spherical_state(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Compute prompt dispersion, modality split, signed arbitration, and prompt SE."""
    _reject_misread_fields(bundle)
    embeddings = bundle.get("embeddings")
    if not isinstance(embeddings, Mapping) or set(embeddings) != set(CONDITIONS):
        raise ValueError("spherical bundle embeddings must contain exactly M1, M2, and M12")
    prompt_sets: list[set[str]] = []
    normalized: dict[str, dict[str, list[float]]] = {}
    for condition in CONDITIONS:
        condition_embeddings = embeddings[condition]
        if not isinstance(condition_embeddings, Mapping) or not condition_embeddings:
            raise ValueError(f"{condition} embeddings must be a non-empty prompt mapping")
        prompt_sets.append(set(map(str, condition_embeddings)))
        normalized[condition] = {
            str(prompt_id): _unit_vector(vector).tolist()
            for prompt_id, vector in condition_embeddings.items()
        }
    if any(prompt_set != prompt_sets[0] for prompt_set in prompt_sets[1:]):
        raise ValueError("all conditions must have synchronized prompt IDs")
    prompt_ids = sorted(prompt_sets[0])

    centers = {
        condition: spherical_center([normalized[condition][prompt_id] for prompt_id in prompt_ids])
        for condition in CONDITIONS
    }
    s_by_condition = {
        condition: sum(
            spherical_distance(normalized[condition][prompt_id], centers[condition])
            for prompt_id in prompt_ids
        )
        / len(prompt_ids)
        for condition in CONDITIONS
    }
    s_mean = sum(s_by_condition.values()) / len(CONDITIONS)
    d_score = spherical_distance(centers["M1"], centers["M2"])
    d_v = spherical_distance(centers["M12"], centers["M1"])
    d_ta = spherical_distance(centers["M12"], centers["M2"])
    r_score = _signed_r(d_v, d_ta)
    prompt_r = [
        _signed_r(
            spherical_distance(normalized["M12"][prompt_id], normalized["M1"][prompt_id]),
            spherical_distance(normalized["M12"][prompt_id], normalized["M2"][prompt_id]),
        )
        for prompt_id in prompt_ids
    ]
    prompt_se = stdev(prompt_r) / math.sqrt(len(prompt_r)) if len(prompt_r) > 1 else 0.0
    bootstrap_se = _synchronous_prompt_bootstrap_se(
        normalized,
        prompt_ids,
        sample_id=str(bundle.get("sample_id", "")),
    )
    return {
        "sample_id": bundle.get("sample_id"),
        "sample_type": bundle.get("sample_type"),
        "calibration_split": bundle.get("calibration_split"),
        "prompt_ids": prompt_ids,
        "S_M1": s_by_condition["M1"],
        "S_M2": s_by_condition["M2"],
        "S_M12": s_by_condition["M12"],
        "S_mean": s_mean,
        "D": d_score,
        "R": r_score,
        "R_prompt_values": prompt_r,
        "R_prompt_se": prompt_se,
        "R_bootstrap_se": bootstrap_se,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "delta_method": DELTA_METHOD,
        "delta_i": 1.96 * bootstrap_se,
        "lean": "V" if r_score > 0.0 else "T/A" if r_score < 0.0 else "Balanced",
    }


def _signed_r(distance_to_v: float, distance_to_ta: float, eps: float = 1e-12) -> float:
    return (distance_to_ta - distance_to_v) / (distance_to_v + distance_to_ta + eps)


def _synchronous_prompt_bootstrap_se(
    embeddings: Mapping[str, Mapping[str, list[float]]],
    prompt_ids: list[str],
    *,
    sample_id: str,
) -> float:
    if len(prompt_ids) < 2:
        return 0.0
    signature = json.dumps(
        {"sample_id": sample_id, "prompt_ids": prompt_ids, "embeddings": embeddings},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    seed = int.from_bytes(hashlib.sha256(signature).digest()[:8], "big")
    random = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indexes = random.integers(0, len(prompt_ids), size=len(prompt_ids))
        sampled_ids = [prompt_ids[index] for index in indexes]
        centers = {
            condition: spherical_center(
                [embeddings[condition][prompt_id] for prompt_id in sampled_ids]
            )
            for condition in CONDITIONS
        }
        distance_to_v = spherical_distance(centers["M12"], centers["M1"])
        distance_to_ta = spherical_distance(centers["M12"], centers["M2"])
        estimates[replicate] = _signed_r(distance_to_v, distance_to_ta)
    return float(estimates.std(ddof=1))


def _unit_vector(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("spherical embeddings must be finite non-empty vectors")
    norm = float(np.linalg.norm(array))
    if not math.isclose(norm, 1.0, abs_tol=UNIT_TOLERANCE):
        raise ValueError("condition embeddings must lie on the unit hypersphere")
    return array


def _reject_misread_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if "misread" in normalized or normalized in {"binary_label", "final_decision"}:
                raise ValueError("Misread fields are forbidden in spherical state inputs")
            _reject_misread_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_misread_fields(child)
