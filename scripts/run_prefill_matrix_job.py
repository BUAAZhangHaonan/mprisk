"""Set a hard CUDA allocator limit before running one matrix extraction job."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


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

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A matrix job requires exactly one visible CUDA device")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device=0)

    from mprisk.cache.prefill_batch import main as extract_main

    return extract_main(extract_args)


if __name__ == "__main__":
    raise SystemExit(main())
