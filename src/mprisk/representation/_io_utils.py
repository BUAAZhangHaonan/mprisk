"""IO / seed / device helpers extracted from training.py (P2-R1-B)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "_set_deterministic_seed",
    "_resolve_device",
    "_move_optimizer_state",
    "_atomic_torch_save",
    "_sha256",
]


def _set_deterministic_seed(seed: int) -> None:
    # m-A1-R5-1: delegate to mprisk.utils.seeds so the deterministic-algorithm
    # flags are defined in exactly one place (scripts/_seed.py imports the
    # same function).
    from mprisk.utils.seeds import set_deterministic_seed
    set_deterministic_seed(seed)


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")
    return resolved


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
