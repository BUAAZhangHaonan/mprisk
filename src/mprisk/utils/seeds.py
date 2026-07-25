"""Random seed helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_python(seed: int) -> None:
    random.seed(seed)


def set_deterministic_seed(seed: int) -> None:
    """Seed python/numpy/torch and force deterministic cuDNN / CUDA algos.

    Centralized here (m-A1-R5-1) so that ``scripts/_seed.py`` and
    ``representation/training.py`` cannot drift apart again. Order matters:
    set ``cudnn`` flags before ``use_deterministic_algorithms`` so the
    workspace config is in place when CUDA looks it up.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
