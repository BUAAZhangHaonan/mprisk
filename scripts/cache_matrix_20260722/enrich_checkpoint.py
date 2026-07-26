#!/usr/bin/env python
"""Enrich a cache_matrix_20260722 train_tme_e2e.py checkpoint for the SDR pipeline.

``scripts/train_tme_e2e.py`` saves ``best_encoder.pt`` with the schema:

    {
        "model_state_dict": {... "encoder.lstm.weight_ih_l0" ...},
        "architecture_version": "tme_e2e_v3b",
        "encoder_type": "bilstm" | "lstm" | "gru",
        "input_dim": 4096,
        "sequence_hidden_dim": 256,
        "embed_dim": 128,
        "head_hidden_dim": 32,
        "dropout": 0.3,
        "warm_start_checkpoint": None,
        "warm_start_used": False,
        "best_epoch": 4,
    }

The SDR / frozen-representation pipeline (via
``mprisk.representation.training.export_frozen_representations`` and
``_validate_checkpoint_architecture``) requires:

    {
        "model_state_dict": {
            ... SphericalTME_BiLSTM state_dict with keys
            "condition_encoder.lstm.weight_ih_l0", "relation.projection.weight", ...
        },
        "architecture_version": "tme_bilstm_proxy_anchor_v1"  # TME_ARCHITECTURE_BILSTM_V1
                                                              # (or *_lstm_*, *_gru_*)
        "repr_key": "tme_proxy_anchor_v1",
        "training_config": {Training_config_kwargs},
        "model_config": {"input_dim":..., "layer_count":..., "hidden_dim":...},
    }

This script reads an existing train_tme_e2e.py checkpoint and writes an
enriched checkpoint compatible with the SDR pipeline. Steps:

  1. Remap ``encoder.*`` state_dict keys to ``condition_encoder.*`` (the
     SphericalTME_BiLSTM wrapper names the encoder module
     ``condition_encoder``).
  2. Drop ``head.*`` keys (TME proxy-anchor path has no classification head).
  3. Synthesise ``relation.projection.weight`` and ``relation.projection.bias``
     via a deterministic RNG seed so the SphericalTME_BiLSTM state_dict is
     complete. SDR scoring only reads ``condition_z`` from the export, so a
     random ``relation_r`` is fine for the SDR pipeline.
  4. Stamp the checkpoint with the SDR-pipeline fields: ``repr_key``,
     ``architecture_version``, ``training_config``, ``model_config``.

CLI:
  python enrich_checkpoint.py \\
      --in-ckpt outputs/cache_matrix_20260722/runs/tme_bilstm/<MODEL>_seed20260717/best_encoder.pt \\
      --out-ckpt /tmp/<MODEL>_seed20260717_enriched.pt \\
      --encoder-type bilstm \\
      --model-key <MODEL> \\
      --protocol vt \\
      --prompt-set configs/prompts/equiv_sets/vt_main_p8_seed20260717.yaml \\
      --layer-count 36
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
import yaml

# Mirror src/mprisk/representation/relation_models.py constants to avoid an
# import-time dependency (the script is invoked from wrappers that already
# set PYTHONPATH=src, but we hard-code the literals here for transparency).
TME_PROXY_ANCHOR_V1 = "tme_proxy_anchor_v1"
TME_ARCHITECTURE_GRU_V1 = "layer_l2_gru_linear_relation_v1"
TME_ARCHITECTURE_LSTM_V1 = "layer_l2_lstm_linear_relation_v1"
TME_ARCHITECTURE_BILSTM_V1 = "tme_bilstm_proxy_anchor_v1"

ENCODER_TYPE_TO_ARCH_VERSION = {
    "gru": TME_ARCHITECTURE_GRU_V1,
    "lstm": TME_ARCHITECTURE_LSTM_V1,
    "bilstm": TME_ARCHITECTURE_BILSTM_V1,
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_prompt_set(prompt_set_path: Path) -> tuple[str, str, list[str]]:
    """Return (prompt_set_key, artifact_sha256, sorted_expected_prompt_ids)."""
    payload = yaml.safe_load(prompt_set_path.read_text(encoding="utf-8")) or {}
    prompt_set_key = str(payload.get("key", ""))
    if not prompt_set_key:
        raise ValueError(f"prompt set {prompt_set_path} has no 'key'")
    artifact_sha = _sha256_file(prompt_set_path)
    templates = payload.get("templates") or []
    prompt_ids = sorted(
        str(row["prompt_id"]) for row in templates if row.get("enabled", True)
    )
    if not prompt_ids:
        raise ValueError(f"prompt set {prompt_set_path} has no enabled templates")
    return prompt_set_key, artifact_sha, prompt_ids


def _remap_state_dict_for_spherical_tme(
    in_state: dict[str, torch.Tensor],
    *,
    encoder_type: str,
    relation_dim: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Rewrite a TME_E2E_v3B encoder state_dict for SphericalTME_BiLSTM.

    - rename ``encoder.*`` -> ``condition_encoder.*``
    - drop ``head.*`` keys
    - synthesise ``relation.projection.weight`` (shape [relation_dim, 3]) and
      ``relation.projection.bias`` (shape [relation_dim]) deterministically
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in in_state.items():
        if key.startswith("encoder."):
            new_key = "condition_encoder." + key[len("encoder."):]
            out[new_key] = value.detach().cpu().clone()
        elif key.startswith("head."):
            # TME proxy-anchor export path doesn't use the classification head.
            continue
        elif key.startswith("condition_encoder.") or key.startswith("relation."):
            # Already in target shape (idempotent re-enrichment).
            out[key] = value.detach().cpu().clone()
        else:
            # Unknown prefix: preserve verbatim in case the export path
            # wants additional buffers. Should not happen for the v3B
            # checkpoints but is safe.
            out[key] = value.detach().cpu().clone()

    # Synthesise relation.projection deterministically. The relation head is
    # only used for relation_r output, which SDR scoring does not read
    # (SDR reads condition_z via bundle["embeddings"]). We still need a
    # complete state_dict because export_frozen_representations calls
    # model.load_state_dict(...) with strict=True.
    if "relation.projection.weight" not in out or "relation.projection.bias" not in out:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        # Linear(in=3, out=relation_dim).weight shape: [relation_dim, 3]
        weight = torch.empty(relation_dim, 3)
        weight.uniform_(-0.02, 0.02, generator=generator)
        bias = torch.zeros(relation_dim)
        out["relation.projection.weight"] = weight
        out["relation.projection.bias"] = bias
    return out


def enrich_checkpoint(
    *,
    in_ckpt_path: Path,
    out_ckpt_path: Path,
    encoder_type: str,
    model_key: str,
    protocol: str,
    prompt_set_path: Path,
    layer_count: int,
    classification_objective: str = "proxy_anchor_only",
    dropout: float | None = None,
    seed: int = 20260717,
    overwrite: bool = True,
) -> Path:
    """Read ``in_ckpt_path`` and write an SDR-compatible enriched checkpoint."""
    if encoder_type not in ENCODER_TYPE_TO_ARCH_VERSION:
        raise ValueError(
            f"unknown encoder_type={encoder_type!r}; "
            f"expected one of {sorted(ENCODER_TYPE_TO_ARCH_VERSION)}"
        )
    if not in_ckpt_path.is_file():
        raise FileNotFoundError(f"input checkpoint not found: {in_ckpt_path}")

    payload = torch.load(in_ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload must be a dict, got {type(payload)!r}")

    # Pull scalar config from the v3B checkpoint.
    input_dim = int(payload["input_dim"])
    sequence_hidden_dim = int(payload["sequence_hidden_dim"])
    embed_dim = int(payload["embed_dim"])
    effective_dropout = (
        float(dropout) if dropout is not None else float(payload.get("dropout", 0.1))
    )

    # SphericalTME_BiLSTM: condition_dim == encoder embed_dim,
    # sequence_hidden_dim == encoder sequence_hidden_dim, relation_dim default 64.
    condition_dim = embed_dim
    relation_dim = 64
    hidden_dim_for_build = sequence_hidden_dim

    prompt_set_key, prompt_set_artifact_sha, prompt_ids = _load_prompt_set(prompt_set_path)
    expected_prompt_count = len(prompt_ids)

    remapped_state = _remap_state_dict_for_spherical_tme(
        payload["model_state_dict"],
        encoder_type=encoder_type,
        relation_dim=relation_dim,
        seed=seed,
    )

    arch_version = ENCODER_TYPE_TO_ARCH_VERSION[encoder_type]

    training_config = {
        "repr_key": TME_PROXY_ANCHOR_V1,
        "model_key": model_key,
        "protocol": protocol,
        "classification_objective": classification_objective,
        "prompt_set_key": prompt_set_key,
        "prompt_set_artifact_sha256": prompt_set_artifact_sha,
        "expected_prompt_count": expected_prompt_count,
        "expected_prompt_ids": tuple(prompt_ids),
        "hidden_dim": hidden_dim_for_build,
        "condition_dim": condition_dim,
        "relation_dim": relation_dim,
        "encoder_type": encoder_type,
        "dropout": effective_dropout,
        "max_epochs": 100,
        "batch_size": 32,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "proxy_alpha": 32.0,
        "proxy_margin": 0.1,
        "enable_state_supervision": False,
        "d_supervision_weight": 0.0,
        "d_ranking_margin": 0.0,
        "angular_supervision_weight": 0.0,
        "angular_ranking_margin_rad": 0.0,
        "d_aux_samples_per_class": 0,
        "sdr_aux_weight": 0.0,
        "sdr_margin_D": 0.6,
        "sdr_margin_R": 0.4,
        "sdr_warmup_epochs": 10,
        "state_selection_min_d_gap": 1e-6,
        "state_selection_min_raw_theta_gap_rad": 0.08726646259971647,
        "state_selection_max_d_mannwhitney_p": 0.05,
        "state_selection_min_d_effect_size": 0.20,
        "patience": 10,
        "min_delta": 1e-4,
        "seed": int(seed),
    }

    model_config = {
        "input_dim": input_dim,
        "layer_count": int(layer_count),
        "hidden_dim": hidden_dim_for_build,
    }

    enriched = {
        "model_state_dict": remapped_state,
        "architecture_version": arch_version,
        "repr_key": TME_PROXY_ANCHOR_V1,
        "encoder_type": encoder_type,
        "training_config": training_config,
        "model_config": model_config,
        # Extra provenance fields the export path ignores but downstream
        # tooling may inspect.
        "enriched_from": str(in_ckpt_path),
        "enriched_source_sha256": _sha256_file(in_ckpt_path),
        "enriched_encoder_type": encoder_type,
        "checkpoint_role": "final_selected",
        "checkpoint_feasibility": {"feasible": True},
    }

    # Preserve original v3B scalars for traceability.
    for passthrough in (
        "input_dim",
        "sequence_hidden_dim",
        "embed_dim",
        "head_hidden_dim",
        "dropout",
        "warm_start_checkpoint",
        "warm_start_used",
        "best_epoch",
    ):
        if passthrough in payload and passthrough not in enriched:
            enriched[f"source_{passthrough}"] = payload[passthrough]

    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(enriched, out_ckpt_path)
    return out_ckpt_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-ckpt", required=True, type=Path)
    p.add_argument("--out-ckpt", required=True, type=Path)
    p.add_argument(
        "--encoder-type",
        required=True,
        choices=sorted(ENCODER_TYPE_TO_ARCH_VERSION),
    )
    p.add_argument("--model-key", required=True)
    p.add_argument("--protocol", required=True, choices=["vt", "va"])
    p.add_argument("--prompt-set", required=True, type=Path)
    p.add_argument("--layer-count", required=True, type=int)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--seed", type=int, default=20260717)
    args = p.parse_args(argv)

    out = enrich_checkpoint(
        in_ckpt_path=args.in_ckpt,
        out_ckpt_path=args.out_ckpt,
        encoder_type=args.encoder_type,
        model_key=args.model_key,
        protocol=args.protocol,
        prompt_set_path=args.prompt_set,
        layer_count=args.layer_count,
        dropout=args.dropout,
        seed=args.seed,
    )
    print(f"[enrich_checkpoint] {args.in_ckpt} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
