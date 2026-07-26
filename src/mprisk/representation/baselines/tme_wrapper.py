"""TME wrapper baseline encoder (wraps a trained ``SphericalTME`` checkpoint).

Extracted from the original ``mprisk.representation.baselines`` module (P4-F refactor)
to keep each encoder family in its own file. Public surface stays
re-exported from ``mprisk.representation.baselines``.

This module contains:
  * :func:`_infer_tme_dims_from_state` — introspects a TME checkpoint to
    recover constructor args and the encoder variant.
  * :class:`TMEEncoder` — the public wrapper exposing a plain
    ``.encode(trajectory)`` API.
  * Internal lightweight TME re-implementations (``_GRUTMEWrapper``,
    ``_GRUConditionEncoder``, ``_LSTMTMEWrapper``, ``_LSTMConditionEncoder``,
    ``_OrderedLinearRelationV1Lite``) used to load ``canonical_rerun``
    checkpoints without dragging in the full mprisk training package.

Trajectory contract: ``[B, 3, L, H]`` (3 conditions M1, M2, M12 in fixed
order) — exactly what ``SphericalTMEV2.forward`` expects.
"""

from __future__ import annotations

import re

from pathlib import Path

import torch
from torch import nn


__all__ = ["TMEEncoder", "_infer_tme_dims_from_state"]


def _infer_tme_dims_from_state(state: dict) -> dict:
    """Inspect a SphericalTME(V1/V2/LSTM) state_dict to recover constructor args.

    Returns dict with sequence_hidden_dim, condition_dim, relation_dim,
    lstm_layers and encoder_type inferred from weight shapes. Falls back to
    defaults if a key is missing.

    ``encoder_type`` is one of:

    * ``"lstm"`` — keys like ``condition_encoder.lstm.weight_ih_l0`` are
      present (bi-LSTM ``SphericalTME_BiLSTM`` from
      ``mprisk.representation.relation_models``, originally the viz
      ``SphericalTMEV2``; bi-directional multi-layer LSTM + MLP, 4*H gate
      factor).
    * ``"gru"`` — keys like ``condition_encoder.sequence.weight_ih_l0``
      are present and NO ``condition_encoder.sequence.weight_ih_l1`` exists
      (``SphericalTMEV1`` / ``SequentialTrajectoryEncoderV1``, single-layer
      GRU, 3*H gate factor).
    * ``"lstm_multilayer"`` — keys like
      ``condition_encoder.sequence.weight_ih_l0`` AND
      ``condition_encoder.sequence.weight_ih_l1`` both exist
      (``SphericalTME_LSTM`` / ``SequentialTrajectoryEncoderLSTMV1``,
      multi-layer uni-directional LSTM, 4*H gate factor). The ``_l1`` key
      is what distinguishes this from the single-layer GRU layout since
      both share the ``condition_encoder.sequence.*`` prefix.
    """
    out = {"sequence_hidden_dim": 128, "condition_dim": 64,
           "relation_dim": 32, "lstm_layers": 2, "encoder_type": "lstm"}
    # Detect encoder type by key prefix + layer count.
    # Legacy bi-LSTM (SphericalTMEV2) uses ``condition_encoder.lstm.*``.
    # New uni-LSTM (SphericalTME_LSTM) and GRU (SphericalTMEV1) both use
    # ``condition_encoder.sequence.*``; we tell them apart by whether
    # ``weight_ih_l1`` exists (LSTM is multi-layer; GRU is single-layer).
    has_lstm_bi = any(k.startswith("condition_encoder.lstm.") for k in state.keys())
    has_seq_l1 = "condition_encoder.sequence.weight_ih_l1" in state.keys()
    has_seq_l0 = "condition_encoder.sequence.weight_ih_l0" in state.keys()
    if has_lstm_bi:
        out["encoder_type"] = "lstm"
        enc_prefix = "condition_encoder.lstm"
        gate_factor = 4  # LSTM: input/forget/cell/output
    elif has_seq_l0 and has_seq_l1:
        # Multi-layer uni-directional LSTM under the ``sequence`` prefix.
        out["encoder_type"] = "lstm_multilayer"
        enc_prefix = "condition_encoder.sequence"
        gate_factor = 4  # LSTM: input/forget/cell/output
    elif has_seq_l0:
        out["encoder_type"] = "gru"
        enc_prefix = "condition_encoder.sequence"
        gate_factor = 3  # GRU: reset/update/new
    else:
        # Nothing recognizable — leave defaults, assume legacy LSTM layout.
        enc_prefix = "condition_encoder.lstm"
        gate_factor = 4

    # hidden_size: weight_ih_l0 shape is (gate_factor*H, input_dim)
    w0 = state.get(f"{enc_prefix}.weight_ih_l0")
    if w0 is not None:
        out["sequence_hidden_dim"] = int(w0.shape[0] // gate_factor)
    # encoder layers: count distinct lN suffixes in weight_ih_lN keys
    layer_idxs = set()
    layer_re = re.compile(rf"{re.escape(enc_prefix)}\.weight_ih_l(\d+)$")
    for k in state.keys():
        m = layer_re.match(k)
        if m:
            layer_idxs.add(int(m.group(1)))
    if layer_idxs:
        out["lstm_layers"] = max(layer_idxs) + 1
    # condition_dim: last MLP/projection layer weight (out, in).
    # LSTM V2 uses ``condition_encoder.mlp.{idx}.weight`` (sequential).
    # GRU V1 / LSTM multilayer both use ``condition_encoder.projection.weight``
    # (single Linear, matching SphericalTMEV1 layout).
    mlp_outs = []
    for k, v in state.items():
        m = re.match(r"condition_encoder\.mlp\.(\d+)\.weight$", k)
        if m:
            mlp_outs.append((int(m.group(1)), int(v.shape[0])))
    if mlp_outs:
        mlp_outs.sort()
        out["condition_dim"] = mlp_outs[-1][1]
    else:
        proj = state.get("condition_encoder.projection.weight")
        if proj is not None:
            out["condition_dim"] = int(proj.shape[0])
    # relation_dim: relation.projection.weight (out, 3)
    rp = state.get("relation.projection.weight")
    if rp is not None:
        out["relation_dim"] = int(rp.shape[0])
    return out


class TMEEncoder(nn.Module):
    """Wrap a trained ``SphericalTMEV2`` (LSTM) or ``SphericalTMEV1`` (GRU)
    and expose ``encode(trajectory)``.

    trajectory contract: ``[B, 3, L, H]`` where dim 1 is the condition axis
    in the canonical order ``[M1, M2, M12]`` (matches what
    ``mprisk.representation.relation_models._validate_three_condition_trajectories``
    expects).

    ``encode`` returns the per-sample concatenated condition-z across the 3
    conditions: ``[B, 3 * condition_dim]``. The model is loaded in eval mode
    and never updated — Stage 2 treats it as a frozen feature extractor.

    The checkpoint must contain a top-level ``model_state_dict`` key whose
    weights match either:

      * SphericalTMEV2 (bi-LSTM): keys prefixed ``condition_encoder.lstm.*``
        + ``condition_encoder.mlp.*`` (this is what the v2 pipeline writes
        to ``outputs/v2/checkpoints/<model>/best_checkpoint.pt``), or
      * SphericalTMEV1 (GRU, single layer): keys prefixed
        ``condition_encoder.sequence.*`` + ``condition_encoder.projection.*``
        (this is what the canonical_rerun encoders under
        ``outputs/canonical_rerun/encoders/<run>/best_checkpoint.pt`` use).

    The encoder_type is auto-detected by :func:`_infer_tme_dims_from_state`.
    """

    architecture_version: str = "baseline_tme_wrapper_v2"

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        input_dim: int,
        sequence_hidden_dim: int | None = None,
        condition_dim: int | None = None,
        relation_dim: int | None = None,
        dropout: float = 0.1,
        lstm_layers: int | None = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        # Auto-infer dims (and encoder_type) from checkpoint if not specified
        inferred = _infer_tme_dims_from_state(state)
        encoder_type = inferred["encoder_type"]
        if sequence_hidden_dim is None:
            sequence_hidden_dim = inferred["sequence_hidden_dim"]
        if condition_dim is None:
            condition_dim = inferred["condition_dim"]
        if relation_dim is None:
            relation_dim = inferred["relation_dim"]
        if lstm_layers is None:
            lstm_layers = inferred["lstm_layers"]

        if encoder_type == "gru":
            # GRU V1 checkpoint: build a SphericalTMEV1-compatible wrapper.
            self.tme = _GRUTMEWrapper(
                input_dim=input_dim,
                sequence_hidden_dim=sequence_hidden_dim,
                condition_dim=condition_dim,
                relation_dim=relation_dim,
                dropout=dropout,
            )
        elif encoder_type == "lstm_multilayer":
            # Multi-layer uni-directional LSTM checkpoint produced by
            # ``mprisk.representation.relation_models.SphericalTME_LSTM``.
            # Same state-dict layout as GRU V1 (``condition_encoder.sequence.*``
            # + ``condition_encoder.projection.*``) except the sequence module
            # has multiple LSTM layers (gate factor 4 instead of 3).
            self.tme = _LSTMTMEWrapper(
                input_dim=input_dim,
                sequence_hidden_dim=sequence_hidden_dim,
                condition_dim=condition_dim,
                relation_dim=relation_dim,
                num_lstm_layers=int(lstm_layers) if lstm_layers else 2,
                dropout=dropout,
            )
        else:
            # Legacy bi-LSTM V2 checkpoint (originally produced by the viz
            # SphericalTMEV2). The class is now inlined into the mainline
            # relation_models module as SphericalTME_BiLSTM with an identical
            # state-dict layout, so existing checkpoints load unchanged.
            from mprisk.representation.relation_models import SphericalTME_BiLSTM  # noqa: WPS433

            self.tme = SphericalTME_BiLSTM(
                input_dim=input_dim,
                sequence_hidden_dim=sequence_hidden_dim,
                condition_dim=condition_dim,
                relation_dim=relation_dim,
                dropout=dropout,
                lstm_layers=lstm_layers,
            )
        # TME checkpoint may be saved as a top-level SphericalTMEV1/V2 state
        # dict (with keys like ``condition_encoder.lstm.weight_ih_l0`` or
        # ``condition_encoder.sequence.weight_ih_l0``) or nested under
        # ``tme.*`` depending on whether the wrapper was serialized as part
        # of a larger training container. Strip a leading ``tme.`` prefix
        # if present.
        if any(k.startswith("tme.") for k in state.keys()) and not any(
            k.startswith("condition_encoder.") for k in state.keys()
        ):
            state = {k[len("tme."):]: v for k, v in state.items() if k.startswith("tme.")}
        self.tme.load_state_dict(state, strict=strict)
        self.tme.eval()
        for p in self.tme.parameters():
            p.requires_grad_(False)
        self.encoder_type = encoder_type
        self.condition_dim = int(condition_dim)
        self.sequence_hidden_dim = int(sequence_hidden_dim)
        self.relation_dim = int(relation_dim)
        # Three conditions: M1, M2, M12 -> 3 * condition_dim output features
        # for the projected condition_z (default / CA path).
        self.out_dim = 3 * self.condition_dim
        # Rich path (MN probe only): pre-projection final hidden per
        # condition. Legacy bi-LSTM V2 is bi-directional -> 2 *
        # sequence_hidden_dim per condition; GRU V1 and the new multi-layer
        # uni-LSTM are both uni-directional -> sequence_hidden_dim per
        # condition.
        per_cond_rich = (
            2 * self.sequence_hidden_dim
            if encoder_type == "lstm"
            else self.sequence_hidden_dim
        )
        self.rich_dim = 3 * per_cond_rich

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """TME has no classifier head — return the encoded representation.

        This makes the wrapper usable anywhere an ``nn.Module`` returning
        logits is expected, but the actual "logits" are the frozen
        representation. Stage 2 ignores them and trains a fresh MLP head.
        """
        return self.encode(trajectory)

    @torch.no_grad()
    def encode(self, trajectory: torch.Tensor, *, rich: bool = False) -> torch.Tensor:
        """trajectory: [B, 3, L, H] -> [B, 3*condition_dim] (default) or
        ``[B, rich_dim]`` when ``rich=True``.

        When ``rich=True`` we bypass the encoder's projection (MLP for LSTM
        V2, single Linear for GRU V1) and return the raw sequence-model
        final hidden states per condition. Used by the MN probe so it sees
        a less compressed feature than the projected condition_z. CA eval
        continues to use the default path.
        """
        if trajectory.ndim != 4:
            raise ValueError(
                f"TMEEncoder.encode expects [B, 3, L, H], got shape {tuple(trajectory.shape)}"
            )
        if rich:
            pre_proj = self.tme.encode_pre_projection(trajectory)
            # pre_proj: [B, 3, per_cond_rich] -> flatten condition axis.
            return pre_proj.reshape(pre_proj.shape[0], -1)
        condition_z, _ = self.tme(trajectory)
        # condition_z: [B, 3, condition_dim] -> flatten condition axis.
        return condition_z.reshape(condition_z.shape[0], -1)

    @torch.no_grad()
    def export_embedding(self, trajectory: torch.Tensor) -> torch.Tensor:
        return self.encode(trajectory)


class _GRUTMEWrapper(nn.Module):
    """Lightweight re-implementation of ``SphericalTMEV1`` (GRU V1) for
    loading ``canonical_rerun`` encoder checkpoints.

    The original class lives in ``mprisk.representation.relation_models``
    but pulling it in requires the full mprisk package + its training
    machinery. We only need forward + state-dict load, so the relevant
    pieces (L2-normalize, GRU encoder, ordered linear relation head) are
    re-implemented here with matching key names so a v1 checkpoint's
    ``condition_encoder.sequence.*`` and ``condition_encoder.projection.*``
    tensors load with ``strict=True``.

    Construction parameters mirror :class:`SphericalTMEV1`.
    """

    architecture_version = "tme_gru_wrapper_v2_compat"

    def __init__(
        self,
        *,
        input_dim: int,
        sequence_hidden_dim: int,
        condition_dim: int,
        relation_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.sequence_hidden_dim = int(sequence_hidden_dim)
        self.condition_dim = int(condition_dim)
        self.relation_dim = int(relation_dim)
        # Match SphericalTMEV1's SequentialTrajectoryEncoderV1 layout:
        #   self.condition_encoder.sequence = nn.GRU(input, hidden, 1, batch_first)
        #   self.condition_encoder.projection = nn.Linear(hidden, embed)
        self.condition_encoder = _GRUConditionEncoder(
            input_dim=self.input_dim,
            sequence_hidden_dim=self.sequence_hidden_dim,
            embed_dim=self.condition_dim,
            dropout=dropout,
        )
        # Match OrderedLinearRelationV1: self.relation.projection = nn.Linear(3, relation_dim)
        self.relation = _OrderedLinearRelationV1Lite(relation_dim=self.relation_dim)

    def forward(
        self,
        trajectories: torch.Tensor,
        *,
        sample_ids=None,
        return_pre_mlp: bool = False,
    ):
        # Same contract as SphericalTMEV1.forward (3-condition validation
        # is enforced inside encode_pre_projection via shape checks).
        if return_pre_mlp:
            condition_z, pre_proj = self.condition_encoder(
                trajectories, return_pre_projection=True,
            )
        else:
            condition_z = self.condition_encoder(trajectories)
            pre_proj = None
        relation_r = self.relation(
            condition_z[:, 0], condition_z[:, 1], condition_z[:, 2],
        )
        if return_pre_mlp:
            return condition_z, relation_r, pre_proj
        return condition_z, relation_r

    @torch.no_grad()
    def encode_pre_projection(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Return raw GRU final-hidden per condition.

        Shape: ``[B, 3, sequence_hidden_dim]``. Used by the MN probe rich
        path. Mirrors what SphericalTMEV2 returns from
        ``forward(return_pre_mlp=True)`` minus the bi-directional factor.
        """
        _cz, pre_proj = self.condition_encoder(trajectories, return_pre_projection=True)
        return pre_proj


class _GRUConditionEncoder(nn.Module):
    """Match ``mprisk.representation.relation_models.SequentialTrajectoryEncoderV1``.

    State-dict keys: ``condition_encoder.sequence.*`` (GRU) +
    ``condition_encoder.projection.*`` (Linear). Forward applies
    layer-wise L2 normalization, runs the 1-layer GRU, then projects the
    final hidden through ``projection`` and L2-normalizes the result.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        sequence_hidden_dim: int,
        embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.sequence_hidden_dim = int(sequence_hidden_dim)
        self.embed_dim = int(embed_dim)
        self.sequence = nn.GRU(
            input_size=input_dim,
            hidden_size=sequence_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(sequence_hidden_dim, embed_dim)

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-12)
        return x / norm

    def forward(
        self,
        trajectories: torch.Tensor,
        *,
        return_pre_projection: bool = False,
    ):
        if trajectories.ndim != 4:
            raise ValueError(
                f"_GRUConditionEncoder expects [B, 3, L, H], got shape "
                f"{tuple(trajectories.shape)}"
            )
        if trajectories.shape[-1] != self.input_dim:
            raise ValueError(
                f"trajectories last dim {trajectories.shape[-1]} != input_dim "
                f"{self.input_dim}"
            )
        b, c, layer_count, h = trajectories.shape
        flat = trajectories.reshape(b * c, layer_count, h)
        # Layer-wise L2 normalize (matches strict_l2_normalize on the
        # ``tme_layer_input`` stage: per-token across feature dim).
        flat = self._l2_normalize(flat)
        _sequence, hidden = self.sequence(flat)  # hidden: [1, b*c, H]
        last = hidden[-1]  # [b*c, H]
        pre_proj = last.reshape(b, c, self.sequence_hidden_dim)
        projected = self.projection(self.dropout(last))  # [b*c, embed]
        projected = projected.reshape(b, c, self.embed_dim)
        projected = self._l2_normalize(projected)
        if return_pre_projection:
            return projected, pre_proj
        return projected


class _LSTMTMEWrapper(nn.Module):
    """Lightweight re-implementation of ``SphericalTME_LSTM`` for loading
    multi-layer LSTM ``canonical_rerun`` encoder checkpoints.

    Mirrors :class:`_GRUTMEWrapper` exactly except the sequence module is a
    multi-layer uni-directional LSTM instead of a single-layer GRU. The
    state-dict layout is identical to ``SphericalTMEV1`` (keys live under
    ``condition_encoder.sequence.*`` and ``condition_encoder.projection.*``)
    so v1-style checkpoints produced by training with ``encoder_type=lstm``
    load with ``strict=True``. The distinguishing factor vs GRU is the
    presence of ``weight_ih_l1`` (LSTM is multi-layer; GRU is single-layer).
    """

    architecture_version = "tme_lstm_wrapper_v2_compat"

    def __init__(
        self,
        *,
        input_dim: int,
        sequence_hidden_dim: int,
        condition_dim: int,
        relation_dim: int,
        num_lstm_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.sequence_hidden_dim = int(sequence_hidden_dim)
        self.condition_dim = int(condition_dim)
        self.relation_dim = int(relation_dim)
        self.num_lstm_layers = int(num_lstm_layers)
        # Match SphericalTME_LSTM's SequentialTrajectoryEncoderLSTMV1 layout:
        #   self.condition_encoder.sequence = nn.LSTM(input, hidden, N, batch_first)
        #   self.condition_encoder.projection = nn.Linear(hidden, embed)
        self.condition_encoder = _LSTMConditionEncoder(
            input_dim=self.input_dim,
            sequence_hidden_dim=self.sequence_hidden_dim,
            embed_dim=self.condition_dim,
            num_lstm_layers=self.num_lstm_layers,
            dropout=dropout,
        )
        # Match OrderedLinearRelationV1: self.relation.projection = nn.Linear(3, relation_dim)
        self.relation = _OrderedLinearRelationV1Lite(relation_dim=self.relation_dim)

    def forward(
        self,
        trajectories: torch.Tensor,
        *,
        sample_ids=None,
        return_pre_mlp: bool = False,
    ):
        # Same contract as SphericalTME_LSTM.forward (3-condition validation
        # is enforced inside the encoder via shape checks).
        if return_pre_mlp:
            condition_z, pre_proj = self.condition_encoder(
                trajectories, return_pre_projection=True,
            )
        else:
            condition_z = self.condition_encoder(trajectories)
            pre_proj = None
        relation_r = self.relation(
            condition_z[:, 0], condition_z[:, 1], condition_z[:, 2],
        )
        if return_pre_mlp:
            return condition_z, relation_r, pre_proj
        return condition_z, relation_r

    @torch.no_grad()
    def encode_pre_projection(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Return raw LSTM final-layer hidden per condition.

        Shape: ``[B, 3, sequence_hidden_dim]``. Used by the MN probe rich
        path. Uni-directional so per-condition width is ``hidden`` (not
        ``2*hidden`` like the bi-LSTM V2 path).
        """
        _cz, pre_proj = self.condition_encoder(trajectories, return_pre_projection=True)
        return pre_proj


class _LSTMConditionEncoder(nn.Module):
    """Match ``mprisk.representation.relation_models.SequentialTrajectoryEncoderLSTMV1``.

    State-dict keys: ``condition_encoder.sequence.*`` (LSTM, multi-layer) +
    ``condition_encoder.projection.*`` (Linear). Forward applies layer-wise
    L2 normalization, runs the multi-layer uni-directional LSTM, takes the
    top layer's final hidden state, projects through ``projection``, and
    L2-normalizes the result.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        sequence_hidden_dim: int,
        embed_dim: int,
        num_lstm_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.sequence_hidden_dim = int(sequence_hidden_dim)
        self.embed_dim = int(embed_dim)
        self.num_lstm_layers = int(num_lstm_layers)
        lstm_dropout = float(dropout) if self.num_lstm_layers > 1 else 0.0
        self.sequence = nn.LSTM(
            input_size=input_dim,
            hidden_size=sequence_hidden_dim,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(sequence_hidden_dim, embed_dim)

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-12)
        return x / norm

    def forward(
        self,
        trajectories: torch.Tensor,
        *,
        return_pre_projection: bool = False,
    ):
        if trajectories.ndim != 4:
            raise ValueError(
                f"_LSTMConditionEncoder expects [B, 3, L, H], got shape "
                f"{tuple(trajectories.shape)}"
            )
        if trajectories.shape[-1] != self.input_dim:
            raise ValueError(
                f"trajectories last dim {trajectories.shape[-1]} != input_dim "
                f"{self.input_dim}"
            )
        b, c, layer_count, h = trajectories.shape
        flat = trajectories.reshape(b * c, layer_count, h)
        # Layer-wise L2 normalize (matches strict_l2_normalize on the
        # ``tme_layer_input`` stage: per-token across feature dim).
        flat = self._l2_normalize(flat)
        # nn.LSTM returns (output, (h_n, c_n)). h_n shape: [num_layers, b*c, H].
        _sequence, (h_n, _c_n) = self.sequence(flat)
        last = h_n[-1]  # top layer's final hidden: [b*c, H]
        pre_proj = last.reshape(b, c, self.sequence_hidden_dim)
        projected = self.projection(self.dropout(last))  # [b*c, embed]
        projected = projected.reshape(b, c, self.embed_dim)
        projected = self._l2_normalize(projected)
        if return_pre_projection:
            return projected, pre_proj
        return projected


class _OrderedLinearRelationV1Lite(nn.Module):
    """Minimal copy of ``OrderedLinearRelationV1`` so v1 ckpts load.

    Computes ``u = [1 - z1·z2, 1 - z12·z1, 1 - z12·z2]`` stacked on the
    last dim (the cosine-distance features defined by
    ``mprisk.representation.relation_models.ordered_relation_features``),
    projects via ``self.projection = nn.Linear(3, relation_dim)``, then
    L2-normalizes the result. Key name ``relation.projection.*`` matches v1.
    """

    def __init__(self, *, relation_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, relation_dim)

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-12)
        return x / norm

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, z12: torch.Tensor) -> torch.Tensor:
        u = torch.stack(
            (
                1.0 - (z1 * z2).sum(dim=-1),
                1.0 - (z12 * z1).sum(dim=-1),
                1.0 - (z12 * z2).sum(dim=-1),
            ),
            dim=-1,
        )  # [..., 3]
        out = self.projection(u)
        return self._l2_normalize(out)
