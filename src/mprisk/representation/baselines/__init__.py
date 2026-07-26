"""V2 representation-quality baselines (v2 rewrite, pure architectures).

Section E of the v2 paper compares three encoders on the Conflict-vs-Aligned
representation task and a downstream Misread-vs-Non-misread probe:

  * ``SinglePointMLP``   (SP-MLP): last-layer hidden state -> 2-layer MLP.
  * ``SinglePointLayerX``: same architecture but with a configurable layer index
    (used for the per-layer sweep reported in Table 3 / appendix).
  * ``TrajectoryMLP``    (T-MLP): full trajectory flattened -> 2-layer MLP.
  * ``TMEEncoder``       (TME):   wraps the trained ``SphericalTMEV2`` and
    exposes a plain ``.encode(trajectory)`` API so the same two-stage
    training harness works for it.

These architectures are intentionally pure:
  * no LayerNorm anywhere,
  * no unit-norm / sphere normalization on the hidden representation.

They share the trajectory-extraction utilities with the mainline but the
forward contract is plain supervised CE on 2 logits. The encoder
representation (penultimate layer) is what gets handed off to the
downstream M/N probe in ``train_baseline.py``.

The trajectory contract is ``[B, L, H]`` for SP-* / T-MLP, and ``[B, 3, L, H]``
(3 conditions M1, M2, M12 in fixed order) for TME — exactly what
``SphericalTMEV2.forward`` expects.

Encoder classes live in dedicated submodules:

  * :mod:`mprisk.representation.baselines.sp_mlp`    — ``SinglePointMLP`` / ``SinglePointLayerX``
  * :mod:`mprisk.representation.baselines.t_mlp`     — ``TrajectoryMLP``
  * :mod:`mprisk.representation.baselines.tme_wrapper` — ``TMEEncoder`` + its private
    GRU/LSTM compatibility wrappers

This module re-exports every encoder class so existing call sites that
``from mprisk.representation.baselines import ...`` continue to work unchanged. It
also hosts the factory + save/load helpers (``build_baseline``,
``save_encoder``, ``load_encoder``, ``encoder_out_dim``) because those
need to dispatch across all three encoder families.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .sp_mlp import SinglePointLayerX, SinglePointMLP
from .t_mlp import TrajectoryMLP
from .tme_wrapper import TMEEncoder, _infer_tme_dims_from_state


__all__ = [
    "SinglePointMLP",
    "SinglePointLayerX",
    "TrajectoryMLP",
    "TMEEncoder",
    "build_baseline",
    "save_encoder",
    "load_encoder",
    "encoder_out_dim",
]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_baseline(
    method: str,
    *,
    layer_count: int,
    hidden_dim: int,
    dropout: float = 0.1,
    layer_idx: int = -1,
    tme_checkpoint: str | Path | None = None,
    tme_kwargs: dict | None = None,
) -> nn.Module:
    """Factory used by ``train_baseline.py``.

    Parameters
    ----------
    method : {"sp_mlp", "t_mlp", "sp_layer_x", "tme"}
    layer_count, hidden_dim : trajectory shape from the cache
    layer_idx : only used by ``sp_layer_x``
    tme_checkpoint : required for ``tme``; ignored otherwise
    tme_kwargs : optional dict forwarded to ``TMEEncoder.__init__``
    """
    method = method.lower().strip()
    if method == "sp_mlp":
        return SinglePointMLP(
            hidden_dim=hidden_dim, n_classes=2, dropout=dropout, layer_count=layer_count
        )
    if method == "sp_layer_x":
        return SinglePointLayerX(
            hidden_dim=hidden_dim,
            layer_idx=layer_idx,
            n_classes=2,
            dropout=dropout,
            layer_count=layer_count,
        )
    if method == "t_mlp":
        return TrajectoryMLP(
            layer_count=layer_count, hidden_dim=hidden_dim, n_classes=2, dropout=dropout
        )
    if method == "tme":
        if tme_checkpoint is None:
            raise ValueError("method='tme' requires tme_checkpoint")
        # Let TMEEncoder auto-infer dims from checkpoint; only pass input_dim
        # (from trajectory shape) and dropout.
        kwargs = {
            "input_dim": hidden_dim,
            "dropout": 0.1,
        }
        if tme_kwargs:
            kwargs.update(tme_kwargs)
        return TMEEncoder(tme_checkpoint, **kwargs)
    raise ValueError(
        f"unknown baseline method: {method!r} "
        "(expected 'sp_mlp', 'sp_layer_x', 't_mlp', or 'tme')"
    )


# ---------------------------------------------------------------------------
# Encoder save / load helpers (Stage 1 -> Stage 2 contract)
# ---------------------------------------------------------------------------


def save_encoder(
    model: nn.Module,
    *,
    path: str | Path,
    method: str,
    layer_count: int,
    hidden_dim: int,
    layer_idx: int = -1,
    extra: dict | None = None,
) -> None:
    """Persist everything Stage 2 needs to reload a frozen encoder.

    Stage 2 needs: the architecture parameters, the trained weights, and
    the method tag (so it can build the same nn.Module then load the
    weights). We also stash the architecture_version so we can detect
    format mismatches later.
    """
    payload = {
        "method": method,
        "layer_count": int(layer_count),
        "hidden_dim": int(hidden_dim),
        "layer_idx": int(layer_idx),
        "architecture_version": getattr(model, "architecture_version", "unknown"),
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_encoder(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a frozen encoder from a Stage 1 checkpoint.

    Returns ``(model, meta)`` where ``meta`` is the full payload (without
    the state dict, which has been moved into ``model``).
    """
    payload = torch.load(path, map_location=map_location)
    method = payload["method"]
    model = build_baseline(
        method,
        layer_count=payload["layer_count"],
        hidden_dim=payload["hidden_dim"],
        layer_idx=payload.get("layer_idx", -1),
        tme_checkpoint=payload.get("extra", {}).get("tme_checkpoint"),
        tme_kwargs=payload.get("extra", {}).get("tme_kwargs"),
    )
    state = payload["model_state_dict"]
    # Filter to keys present in the rebuilt model (TME wrapper has different
    # layout than the trained SphericalTMEV2; for SP/T-MLP the state dicts
    # are identical so strict load works).
    own = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in own}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta = {k: v for k, v in payload.items() if k != "model_state_dict"}
    return model, meta


def encoder_out_dim(model: nn.Module) -> int:
    """Infer the dimensionality returned by ``model.encode(...)``."""
    if isinstance(model, TMEEncoder):
        return model.out_dim
    if isinstance(model, SinglePointMLP):
        return int(model.hidden_dim)
    if isinstance(model, SinglePointLayerX):
        return int(model.hidden_dim)
    if isinstance(model, TrajectoryMLP):
        return 1024
    raise TypeError(f"unknown encoder class: {type(model).__name__}")
