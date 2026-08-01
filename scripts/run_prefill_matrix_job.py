"""Set a hard CUDA allocator limit before running one matrix extraction job."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence


_CUDA_ALLOCATOR_ENVIRONMENTS = (
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
)
_REQUIRED_ALLOCATOR_OPTION = ("expandable_segments", "True")


def _parse_allocator_config(name: str, value: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for raw_option in value.split(","):
        key, separator, option_value = raw_option.partition(":")
        key = key.strip()
        option_value = option_value.strip()
        if not separator or not key or not option_value:
            raise RuntimeError(f"Invalid {name} option: {raw_option!r}")
        if key in options:
            raise RuntimeError(f"Duplicate {name} option: {key}")
        options[key] = option_value
    return options


def _configure_cuda_allocator() -> dict[str, str]:
    configured = {
        name: os.environ[name]
        for name in _CUDA_ALLOCATOR_ENVIRONMENTS
        if name in os.environ
    }
    if not configured:
        name = "PYTORCH_CUDA_ALLOC_CONF"
        value = ":".join(_REQUIRED_ALLOCATOR_OPTION)
        os.environ[name] = value
        return {name: value}

    parsed = {
        name: _parse_allocator_config(name, value)
        for name, value in configured.items()
    }
    normalized = {tuple(sorted(value.items())) for value in parsed.values()}
    if len(parsed) == 2 and len(normalized) != 1:
        raise RuntimeError(
            "PYTORCH_ALLOC_CONF and PYTORCH_CUDA_ALLOC_CONF define conflicting options"
        )
    key, required_value = _REQUIRED_ALLOCATOR_OPTION
    for name, options in parsed.items():
        if options.get(key, "").lower() != required_value.lower():
            raise RuntimeError(
                f"{name} must explicitly set {key}:{required_value}; got {configured[name]!r}"
            )
    return configured


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-memory-fraction", required=True, type=float)
    parser.add_argument("extract_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not 0 < args.gpu_memory_fraction < 0.90:
        raise ValueError("GPU memory fraction must be positive and below 0.90")
    extract_args = list(args.extract_args)
    if extract_args and extract_args[0] == "--":
        extract_args.pop(0)
    if not extract_args:
        raise ValueError("Missing prefill extraction arguments")

    _configure_cuda_allocator()

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A matrix job requires exactly one visible CUDA device")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=0)

    from mprisk.cache.prefill_batch import main as extract_main

    return extract_main(extract_args)


if __name__ == "__main__":
    raise SystemExit(main())
