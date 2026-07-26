"""Device string helpers."""

from __future__ import annotations


def default_device(cuda_available: bool) -> str:
    return "cuda:0" if cuda_available else "cpu"
