from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

import mprisk.state.spherical as spherical


def _unit_rows(random: np.random.Generator, count: int, dimension: int) -> np.ndarray:
    rows = random.normal(size=(count, dimension))
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _bundle(*, prompt_count: int, dimension: int) -> dict[str, object]:
    random = np.random.default_rng(20260718)
    prompt_ids = [f"p{index:02d}" for index in range(prompt_count)]
    return {
        "sample_id": f"reference-p{prompt_count}-d{dimension}",
        "sample_type": "Conflict",
        "embeddings": {
            condition: dict(
                zip(
                    prompt_ids,
                    _unit_rows(random, prompt_count, dimension).tolist(),
                    strict=True,
                )
            )
            for condition in spherical.CONDITIONS
        },
    }


def _reference_bootstrap_se(bundle: dict[str, object], replicates: int) -> float:
    prompt_ids = sorted(bundle["embeddings"]["M1"])
    normalized = {
        condition: {
            prompt_id: spherical._unit_vector(bundle["embeddings"][condition][prompt_id]).tolist()
            for prompt_id in prompt_ids
        }
        for condition in spherical.CONDITIONS
    }
    signature = json.dumps(
        {
            "sample_id": bundle["sample_id"],
            "prompt_ids": prompt_ids,
            "embeddings": normalized,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    seed = int.from_bytes(hashlib.sha256(signature).digest()[:8], "big")
    random = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indexes = random.integers(0, len(prompt_ids), size=len(prompt_ids))
        sampled_ids = [prompt_ids[index] for index in indexes]
        centers = {
            condition: spherical.spherical_center(
                [normalized[condition][prompt_id] for prompt_id in sampled_ids]
            )
            for condition in spherical.CONDITIONS
        }
        estimates[replicate] = spherical._signed_r(
            spherical.spherical_distance(centers["M12"], centers["M1"]),
            spherical.spherical_distance(centers["M12"], centers["M2"]),
            spherical.spherical_distance(centers["M1"], centers["M2"]),
        )
    return float(estimates.std(ddof=1))


@pytest.mark.parametrize(("prompt_count", "dimension"), [(2, 3), (8, 64)])
def test_vectorized_bootstrap_matches_scalar_reference(
    monkeypatch: pytest.MonkeyPatch,
    prompt_count: int,
    dimension: int,
) -> None:
    replicates = 257
    bundle = _bundle(prompt_count=prompt_count, dimension=dimension)
    expected = _reference_bootstrap_se(bundle, replicates)
    monkeypatch.setattr(spherical, "BOOTSTRAP_REPLICATES", replicates)

    state = spherical.compute_spherical_state(bundle)

    assert state["R_bootstrap_se"] == pytest.approx(expected, rel=1e-13, abs=1e-13)


def test_bootstrap_draws_once_and_limits_vectorized_working_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replicates = 513
    prompt_count = 8
    bundle = _bundle(prompt_count=prompt_count, dimension=64)
    observed_draw_shapes: list[tuple[int, int]] = []
    observed_batch_shapes: list[tuple[int, int]] = []
    original_batch = spherical._bootstrap_r_batch

    class TrackingRng:
        def integers(
            self, low: int, high: int, size: tuple[int, int]
        ) -> np.ndarray:
            assert (low, high) == (0, prompt_count)
            observed_draw_shapes.append(size)
            return np.zeros(size, dtype=np.int64)

    def tracking_batch(
        prompt_vectors: dict[str, np.ndarray], indexes: np.ndarray
    ) -> np.ndarray:
        observed_batch_shapes.append(indexes.shape)
        return original_batch(prompt_vectors, indexes)

    monkeypatch.setattr(spherical, "BOOTSTRAP_REPLICATES", replicates)
    monkeypatch.setattr(spherical.np.random, "default_rng", lambda seed: TrackingRng())
    monkeypatch.setattr(spherical, "_bootstrap_r_batch", tracking_batch)

    state = spherical.compute_spherical_state(bundle)

    assert math.isfinite(state["R_bootstrap_se"])
    assert observed_draw_shapes == [(replicates, prompt_count)]
    assert observed_batch_shapes == [(256, prompt_count), (256, prompt_count), (1, prompt_count)]
    assert max(batch * prompt_count for batch, prompt_count in observed_batch_shapes) == 2048
