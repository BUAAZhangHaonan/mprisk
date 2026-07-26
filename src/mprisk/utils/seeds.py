"""Random seed helpers.

This module exposes two public helpers:

* seed_python(seed) – only seeds Python's random module. Kept for
  backward compatibility with older call sites that only need Python-level
  determinism.
* set_deterministic_seed(seed) – seeds Python, NumPy, PyTorch (CPU +
  CUDA) and enables deterministic cuDNN/torch algorithms. Use this from any
  training entry-point that needs reproducible behavior across the whole
  numerical stack.
"""

from __future__ import annotations

import os
import random


def seed_python(seed: int) -> None:
    random.seed(seed)


def set_deterministic_seed(seed: int) -> None:
    """Seed Python / NumPy / PyTorch and enable deterministic algorithms."""

    # Python stdlib.
    random.seed(seed)

    # NumPy (optional – not every entry point imports it, but training does).
    try:
        import numpy as _np
    except ImportError:  # pragma: no cover – numpy is a hard dep in training
        _np = None
    if _np is not None:
        _np.random.seed(seed)

    # PyTorch (optional – seeds module is import-safe even without torch).
    try:
        import torch as _torch
    except ImportError:  # pragma: no cover – torch is a hard dep in training
        _torch = None
    if _torch is not None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        _torch.use_deterministic_algorithms(True)
        _torch.backends.cudnn.benchmark = False


__all__ = ["seed_python", "set_deterministic_seed"]
