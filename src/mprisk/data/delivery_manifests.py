"""Delivery manifest loaders (delivery_20260716).

Exposes:
    - DeliverySample dataclass (mirrors the one historically defined in
      scripts/generate_misread_descriptions.py).
    - load_delivery_filtered(protocol) -> list[DeliverySample]: reads the
      protocol-filtered manifest produced by
      ``scripts/build_delivery_filtered_manifests.py``.

The filtered manifests live under
``data/processed/manifests/delivery_20260716/{vt,va}_filtered.jsonl``
relative to the project root. The default output dir can be overridden
via the ``MPRISK_DELIVERY_DIR`` environment variable or by passing
``output_dir`` explicitly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# Default location of the filtered manifests, relative to the project
# root (the directory containing ``pyproject.toml``). Resolved lazily so
# the module works regardless of cwd.
_DEFAULT_REL_DIR = Path("data/processed/manifests/delivery_20260716")

Protocol = Literal["vt", "va"]


@dataclass(frozen=True)
class DeliverySample:
    sample_id: str
    source_id: str
    protocol: str           # "VT" or "VA"
    sample_type: str        # "Conflict" or "Aligned"
    media_paths: dict[str, str]
    text_content: str
    gt_emotion: str
    surface_emotion: str | None
    gt_describe: str        # the GT 4-segment description


def _default_output_dir() -> Path:
    """Resolve the default filtered-manifest directory.

    Priority:
        1. $MPRISK_DELIVERY_DIR (absolute or relative to cwd)
        2. <project_root>/data/processed/manifests/delivery_20260716

    The project root is discovered by walking up from this file until a
    ``pyproject.toml`` is found. If none is found, fall back to a
    path relative to cwd (matching the historical behavior where the pipeline
    scripts are run from the repo root).
    """
    env = os.environ.get("MPRISK_DELIVERY_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent / _DEFAULT_REL_DIR
    return _DEFAULT_REL_DIR


def load_delivery_filtered(
    protocol: Protocol,
    *,
    output_dir: str | Path | None = None,
) -> list[DeliverySample]:
    """Load protocol-filtered samples for ``protocol`` ("vt" or "va").

    Reads ``{output_dir}/{protocol}_filtered.jsonl``. When ``output_dir``
    is None, uses :func:`_default_output_dir`.

    Returns a list of :class:`DeliverySample`. Raises ``FileNotFoundError``
    if the filtered manifest is missing (run
    ``scripts/build_delivery_filtered_manifests.py`` first).
    """
    proto = protocol.lower()
    if proto not in ("vt", "va"):
        raise ValueError(f"protocol must be 'vt' or 'va', got {protocol!r}")
    base = Path(output_dir) if output_dir is not None else _default_output_dir()
    path = base / f"{proto}_filtered.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"filtered manifest not found: {path}. "
            f"Run scripts/build_delivery_filtered_manifests.py first."
        )

    samples: list[DeliverySample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append(
                DeliverySample(
                    sample_id=row["sample_id"],
                    source_id=str(row.get("source_id", "")),
                    protocol=str(row["protocol"]),
                    sample_type=str(row["sample_type"]),
                    media_paths=dict(row.get("media_paths", {})),
                    text_content=str(row.get("text_content", "")),
                    gt_emotion=str(row.get("gt_emotion", "")),
                    surface_emotion=row.get("surface_emotion"),
                    gt_describe=str(row.get("gt_describe", "")),
                )
            )
    return samples


def _main() -> int:
    for proto in ("vt", "va"):
        try:
            samples = load_delivery_filtered(proto)
        except FileNotFoundError as e:
            print(f"{proto}: MISSING ({e})")
            continue
        sids = [s.sample_id for s in samples]
        types = sorted({s.sample_type for s in samples})
        protos = sorted({s.protocol.upper() for s in samples})
        print(
            f"{proto}: n={len(samples)} unique={len(set(sids))} "
            f"types={types} protocols={protos}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
