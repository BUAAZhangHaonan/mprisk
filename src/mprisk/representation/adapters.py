"""Representation adapter entry points."""

from __future__ import annotations

from mprisk.representation.trajectory_encoder import RawTrajectoryEncoder, TrajectoryEncoder


RAW_REPR_KEYS = frozenset({"raw_layernorm_mean", "raw_layernorm_flat"})


def get_trajectory_encoder(repr_key: str) -> TrajectoryEncoder:
    """Return the trajectory encoder registered for a representation key."""
    if repr_key in RAW_REPR_KEYS:
        return RawTrajectoryEncoder(repr_key=repr_key)
    raise ValueError(f"Unknown trajectory representation: {repr_key}")
