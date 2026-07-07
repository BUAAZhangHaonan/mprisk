"""Export learned trajectory embeddings from prompted cache bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mprisk.cache.prefill_extract import extract_t0_trajectory
from mprisk.cache.prompt_conditioned_cache import prompt_conditioned_entry_from_row
from mprisk.data.manifests import read_jsonl
from mprisk.representation.trajectory_model import MLPProjection
from mprisk.utils.io import write_json, write_jsonl


VIEW_KEYS = ("M1", "M2", "M12")


@dataclass(frozen=True)
class TrainedEmbeddingExportResult:
    manifest_path: Path
    summary_path: Path
    count: int
    summary: dict[str, Any]


def export_trained_embeddings(
    *,
    bundle_manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    repr_key: str = "tme_supcon_v1",
    device: str = "cpu",
) -> TrainedEmbeddingExportResult:
    """Project prompt-conditioned t0 trajectories with a trained TME checkpoint."""
    torch_device = torch.device(device)
    checkpoint = _load_checkpoint(checkpoint_path, device=torch_device)
    checkpoint_repr_key = checkpoint.get("repr_key")
    if checkpoint_repr_key is not None and str(checkpoint_repr_key) != repr_key:
        raise ValueError(
            f"checkpoint repr_key {checkpoint_repr_key!r} does not match requested {repr_key!r}"
        )

    model = _load_model(checkpoint, device=torch_device)
    bundle_rows = read_jsonl(bundle_manifest_path)
    embedding_rows = [
        _embedding_row(bundle, model=model, repr_key=repr_key, device=torch_device)
        for bundle in bundle_rows
    ]
    embedding_dim = _embedding_dim(embedding_rows)

    output_root = Path(output_dir)
    manifest_path = write_jsonl(output_root / "embedding_manifest.jsonl", embedding_rows)
    summary = {
        "bundle_manifest": str(bundle_manifest_path),
        "checkpoint": str(checkpoint_path),
        "embedding_manifest": str(manifest_path),
        "repr_key": repr_key,
        "device": str(torch_device),
        "total_samples": len(embedding_rows),
        "embedding_dim": embedding_dim,
    }
    summary_path = write_json(output_root / "embedding_summary.json", summary)
    return TrainedEmbeddingExportResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        count=len(embedding_rows),
        summary=summary,
    )


def _load_checkpoint(path: str | Path, *, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    for field in ("model_state_dict", "model_config"):
        if field not in checkpoint:
            raise ValueError(f"checkpoint missing required field {field}")
    return checkpoint


def _load_model(checkpoint: Mapping[str, Any], *, device: torch.device) -> MLPProjection:
    model_config = dict(checkpoint["model_config"])
    model = MLPProjection(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _embedding_row(
    bundle: dict[str, Any],
    *,
    model: MLPProjection,
    repr_key: str,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "sample_id": bundle["sample_id"],
        "sample_type": bundle["sample_type"],
        "model_key": bundle["model_key"],
        "protocol": bundle["protocol"],
        "prompt_set_key": bundle["prompt_set_key"],
        "repr_key": repr_key,
        "embeddings": {
            view_key: _view_embeddings(bundle, view_key=view_key, model=model, device=device)
            for view_key in VIEW_KEYS
        },
    }


def _view_embeddings(
    bundle: dict[str, Any],
    *,
    view_key: str,
    model: MLPProjection,
    device: torch.device,
) -> dict[str, list[float]]:
    view = bundle["views"][view_key]
    prompts = view["prompts"]
    if not isinstance(prompts, Mapping) or not prompts:
        raise ValueError(f"{view_key} must contain at least one prompt")
    return {
        str(prompt_id): _project_prompt(prompt_payload, model=model, device=device)
        for prompt_id, prompt_payload in prompts.items()
    }


def _project_prompt(
    prompt_payload: Mapping[str, Any],
    *,
    model: MLPProjection,
    device: torch.device,
) -> list[float]:
    state = prompt_payload["prompt_conditioned_state"]
    if isinstance(state, str):
        state = json.loads(state)
    if not isinstance(state, dict):
        raise ValueError("prompt_conditioned_state must be a mapping or JSON object string")

    entry = prompt_conditioned_entry_from_row(state).to_hidden_state_entry()
    trajectory = torch.tensor(
        extract_t0_trajectory(entry),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        embedding = model(trajectory).squeeze(0).detach().cpu().numpy()
    if embedding.ndim != 1 or embedding.size == 0:
        raise ValueError("trained embedding must be a non-empty vector")
    if not np.isfinite(embedding).all():
        raise ValueError("trained embedding must contain only finite values")
    return embedding.astype(float).tolist()


def _embedding_dim(rows: list[dict[str, Any]]) -> int | None:
    dims: set[int] = set()
    for row in rows:
        for view_embeddings in row["embeddings"].values():
            for embedding in view_embeddings.values():
                dims.add(len(embedding))
    if not dims:
        return None
    if len(dims) != 1:
        raise ValueError(f"trained embeddings have inconsistent dimensions: {sorted(dims)}")
    return dims.pop()
