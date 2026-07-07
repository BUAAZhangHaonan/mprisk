"""Representation adapter entry points."""

from __future__ import annotations

from mprisk.representation.manifold_encoder import ManifoldEncoderAdapter
from mprisk.representation.trajectory_encoder import RawTrajectoryEncoder, TrajectoryEncoder

RAW_REPR_KEYS = frozenset({"raw_layernorm_mean", "raw_layernorm_flat"})
TME_SUPCON_V1_REPR_KEY = "tme_supcon_v1"
TRAINED_REPR_KEYS = frozenset({TME_SUPCON_V1_REPR_KEY})
SUPPORTED_REPR_KEYS = RAW_REPR_KEYS | TRAINED_REPR_KEYS


def get_trajectory_encoder(repr_key: str) -> TrajectoryEncoder:
    """Return the trajectory encoder registered for a representation key."""
    if repr_key in RAW_REPR_KEYS:
        return RawTrajectoryEncoder(repr_key=repr_key)
    if repr_key in TRAINED_REPR_KEYS:
        return ManifoldEncoderAdapter(repr_key=repr_key)
    raise ValueError(f"Unknown trajectory representation: {repr_key}")
