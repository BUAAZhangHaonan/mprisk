"""Shared deterministic seed helper for v2 training scripts.

m-A1-R5-1: implementation now lives in ``src/mprisk/utils/seeds.py``; this
file is a thin re-export so existing ``from _seed import set_deterministic_seed``
imports keep working while the codebase migrates to the canonical path.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mprisk.utils.seeds import set_deterministic_seed  # noqa: E402,F401

__all__ = ["set_deterministic_seed"]
