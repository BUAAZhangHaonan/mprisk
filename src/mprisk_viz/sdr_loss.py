"""V2 SDR-aware PA loss: keep Proxy Anchor but add a direct hinge on the
geometry so Conflict samples get larger d(M1, M2) and |R| than Aligned.

Motive: PA only organizes the 32-dim relation_r; the 64-dim condition_z
geometry (which produces S, D, R) is unconstrained. In our raw data,
d(M1, M2) is already larger for Conflict than Aligned (49.5 vs 47.3 deg),
but PA training reverses this (80 vs 123 deg). The aux loss pins the
geometry in the correct direction.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mprisk.representation import training as _training_mod
from mprisk.representation.training import (
    ProxyAnchorLoss,
    _load_trajectory_batch,
    _Sample,
)


# Defaults chosen from raw-layernorm baseline (Conflict d ~0.86 rad, Aligned ~0.83 rad)
DEFAULT_MARGIN_D = 0.30   # want mean_d(M1,M2)_C - mean_d(M1,M2)_A >= 0.30 rad (~17 deg)
DEFAULT_MARGIN_R = 0.30   # want mean_|R|_C - mean_|R|_A >= 0.30
DEFAULT_AUX_WEIGHT = 2.0  # weight relative to PA loss


def _safe_acos(cos, eps=1e-7):
    return cos.clamp(-1.0 + eps, 1.0 - eps).acos()


def make_sdr_aware_batch_loss(
    *,
    aux_weight: float = DEFAULT_AUX_WEIGHT,
    margin_D: float = DEFAULT_MARGIN_D,
    margin_R: float = DEFAULT_MARGIN_R,
    warmup_epochs: int = 5,  # let PA settle before applying aux
):
    """Build a replacement for training._batch_loss_and_outputs with SDR aux loss."""

    def batch_loss(
        model: nn.Module,
        objective: ProxyAnchorLoss | None,
        batch: Sequence["_Sample"],
        *,
        class_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(model.parameters()).device
        trajectories, labels = _load_trajectory_batch(batch, device=device)
        if objective is None:
            raise ValueError("v2 SDR-aware loss requires the Proxy Anchor objective")
        if class_weights is not None:
            raise ValueError("TME Proxy Anchor must not receive cross-entropy class weights")

        sample_ids = [sample.sample_id for sample in batch]
        condition_z, relation_r = model(trajectories, sample_ids=sample_ids)
        pa_loss = objective(relation_r, labels, sample_ids=sample_ids)

        # --- SDR aux loss ---------------------------------------------------
        # condition_z is [batch, 3, condition_dim] and already on unit sphere
        # (strict_l2_normalize applied at encoder output).
        z1 = condition_z[:, 0]
        z2 = condition_z[:, 1]
        z12 = condition_z[:, 2]

        cos_m1_m2 = (z1 * z2).sum(dim=-1)
        cos_m12_m1 = (z12 * z1).sum(dim=-1)
        cos_m12_m2 = (z12 * z2).sum(dim=-1)

        d_m1_m2 = _safe_acos(cos_m1_m2)
        d_m12_m1 = _safe_acos(cos_m12_m1)
        d_m12_m2 = _safe_acos(cos_m12_m2)
        # signed R per sample (1-prompt variant: no per-prompt averaging)
        r_signed = (d_m12_m2 - d_m12_m1) / (d_m1_m2 + 1e-7)

        # Mainline label_id: Aligned=0, Conflict=1
        c_mask = (labels == 1)
        a_mask = (labels == 0)

        aux_loss = pa_loss.new_zeros(())
        if int(c_mask.sum()) > 0 and int(a_mask.sum()) > 0:
            mean_d_C = d_m1_m2[c_mask].mean()
            mean_d_A = d_m1_m2[a_mask].mean()
            mean_absR_C = r_signed[c_mask].abs().mean()
            mean_absR_A = r_signed[a_mask].abs().mean()

            hinge_D = F.relu(margin_D - (mean_d_C - mean_d_A))
            hinge_R = F.relu(margin_R - (mean_absR_C - mean_absR_A))
            aux_loss = hinge_D + hinge_R

        # Warmup: ramp aux_weight from 0 over `warmup_epochs` epochs.
        epoch = _get_current_epoch(model)
        if epoch <= warmup_epochs:
            w = aux_weight * (epoch / max(warmup_epochs, 1))
        else:
            w = aux_weight
        total_loss = pa_loss + w * aux_loss
        return total_loss, relation_r

    return batch_loss


# Stash current epoch on the model object via a hook; train_trajectory_encoder
# doesn't expose epoch to _batch_loss_and_outputs directly, so we read from a
# WeakKeyDictionary keyed by the model object set by a wrapper around
# _train_epoch. WeakKeyDictionary avoids leaking models whose id() may be
# reused by the interpreter after garbage collection.
_EPOCH_STATE: "weakref.WeakKeyDictionary[nn.Module, int]" = weakref.WeakKeyDictionary()


def _get_current_epoch(model: nn.Module) -> int:
    return _EPOCH_STATE.get(model, 1)


def _set_current_epoch(model: nn.Module, epoch: int) -> None:
    _EPOCH_STATE[model] = int(epoch)


# Idempotency guard: install_sdr_aware_loss monkey-patches module globals.
# Calling it twice (e.g., on accidental re-import of pipeline.py) would
# re-wrap the *already-wrapped* _train_epoch and double-count aux loss.
_INSTALLED = False


def install_sdr_aware_loss(
    *,
    aux_weight: float = DEFAULT_AUX_WEIGHT,
    margin_D: float = DEFAULT_MARGIN_D,
    margin_R: float = DEFAULT_MARGIN_R,
    warmup_epochs: int = 5,
) -> None:
    """Monkey-patch _batch_loss_and_outputs to add SDR hinge.

    Idempotent: subsequent calls are a no-op (the first call wins). This
    guards against re-import side effects when ``mprisk_viz.pipeline`` is
    reloaded.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    batch_loss = make_sdr_aware_batch_loss(
        aux_weight=aux_weight,
        margin_D=margin_D,
        margin_R=margin_R,
        warmup_epochs=warmup_epochs,
    )
    _training_mod._batch_loss_and_outputs = batch_loss

    # Also wrap _train_epoch so we can track the current epoch.
    original_train_epoch = _training_mod._train_epoch

    def wrapped_train_epoch(model, *args, **kwargs):
        epoch = kwargs.get("epoch")
        if epoch is None:
            # _train_epoch signature is positional; fall back to args[4]
            epoch = args[4] if len(args) > 4 else 1
        _set_current_epoch(model, int(epoch))
        return original_train_epoch(model, *args, **kwargs)

    _training_mod._train_epoch = wrapped_train_epoch
    _INSTALLED = True
