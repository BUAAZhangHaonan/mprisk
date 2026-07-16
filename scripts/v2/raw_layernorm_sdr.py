"""Raw-layernorm SDR baseline (no TME encoder).

For each sample, condition, prompt:
  - Take raw last-layer hidden state at t0 (hidden_dim vector)
  - L2 normalize onto unit sphere
Compute paper SDR directly on these raw unit vectors.

Purpose: sanity check whether Conflict < Aligned on d(M1, M2) is:
  (a) intrinsic to the model's hidden states (then raw also shows C < A),
  (b) or introduced by TME training (then raw shows C > A or no signal).
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mprisk.cache.cache_manifest import load_full_cache_manifest
from mprisk.cache.prefill_extract import extract_t0_trajectory
from mprisk.cache.hidden_state_cache import normalize_protocol  # lowercase
from mprisk.data.manifests import read_final_manifest
from mprisk.data.representation_splits import load_representation_split_assignment


def _unit(v):
    arr = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(arr)
    if n <= 1e-12:
        raise ValueError("zero-norm")
    return arr / n


def _geodesic(a, b):
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _center(vectors):
    arr = np.stack(vectors)
    m = arr.mean(axis=0)
    n = np.linalg.norm(m)
    if n <= 1e-12:
        return m
    return m / n


def compute_raw_layernorm_sdr(
    *,
    cache_root: str | Path,
    unified_manifest: str | Path,
    main_manifest: str | Path,
    prompt_set_path: str | Path,
    split_assignment_path: str | Path,
    model_key: str,
    protocol: str,
    output_path: str | Path,
) -> Path:
    """Compute SDR directly on last-layer t0 hidden states (raw_layernorm).

    Outputs one row per sample with: sample_id, sample_type, representation_split,
    S, D, signed R, |R|, and the raw d(M1,M2) used for the user's paradox check.
    """
    import yaml

    protocol = normalize_protocol(protocol)
    with open(prompt_set_path, "r", encoding="utf-8") as f:
        prompt_set = yaml.safe_load(f)
    expected_prompt_ids = [
        t["prompt_id"] for t in prompt_set["templates"] if t.get("enabled", True)
    ]

    cache = load_full_cache_manifest(cache_root, manifest_path=unified_manifest)
    label_rows = read_final_manifest(main_manifest)
    label_by_id = {r.sample_id: r for r in label_rows if normalize_protocol(r.protocol) == protocol}
    _split_assignments = load_representation_split_assignment(split_assignment_path)
    # Invert: sample_id -> representation_split
    split_assignments = {}
    for assignment in _split_assignments.values():
        for sid in assignment.get("sample_ids", []):
            split_assignments[sid] = assignment

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_ids = list(label_by_id.keys())
    print(f"[raw-check] {model_key}/{protocol}: {len(sample_ids)} samples, "
          f"{len(expected_prompt_ids)} prompts", flush=True)

    # Group cache entries by (sample_id, condition) -> list of (prompt_id, entry)
    entries_by_sample: dict[str, dict[str, dict[str, object]]] = defaultdict(
        lambda: {"M1": {}, "M2": {}, "M12": {}}
    )
    for entry in cache.entries:
        if entry.model_key != model_key or entry.protocol != protocol:
            continue
        # The cache manifest doesn't carry prompt_id at the HiddenStateEntry level;
        # it's in metadata. Recover.
        prompt_id = (entry.metadata or {}).get("prompt_id")
        if prompt_id is None or prompt_id not in expected_prompt_ids:
            continue
        cond = entry.condition
        if cond not in ("M1", "M2", "M12"):
            continue
        entries_by_sample[entry.sample_id][cond][prompt_id] = entry

    rows = []
    processed = 0
    skipped_incomplete = 0
    for sample_id in sample_ids:
        info = entries_by_sample.get(sample_id)
        if info is None:
            continue
        if not all(len(info[c]) == len(expected_prompt_ids) for c in ("M1", "M2", "M12")):
            skipped_incomplete += 1
            continue

        try:
            # Extract raw t0 last-layer vector for every (condition, prompt)
            per_cond = {}
            for cond in ("M1", "M2", "M12"):
                vectors = []
                for pid in expected_prompt_ids:
                    entry = info[cond][pid]
                    traj = extract_t0_trajectory(entry)  # [layer_count, hidden_dim]
                    last_layer = traj[-1]                 # [hidden_dim]
                    vectors.append(_unit(last_layer.astype(np.float64)))
                per_cond[cond] = vectors

            centers = {c: _center(per_cond[c]) for c in ("M1", "M2", "M12")}
            s_per = {
                c: sum(_geodesic(per_cond[c][i], centers[c]) ** 2
                       for i in range(len(expected_prompt_ids))) / len(expected_prompt_ids)
                for c in ("M1", "M2", "M12")
            }
            s_mean = sum(s_per.values()) / 3.0

            d_m1_m2 = _geodesic(centers["M1"], centers["M2"])
            d_m12_m1 = _geodesic(centers["M12"], centers["M1"])
            d_m12_m2 = _geodesic(centers["M12"], centers["M2"])
            d_score = d_m1_m2 / (math.sqrt(s_per["M1"] + s_per["M2"]) + 1e-12)
            r_signed = (d_m12_m2 - d_m12_m1) / (d_m1_m2 + 1e-12)

            row = label_by_id[sample_id]
            sa = split_assignments.get(sample_id, {})
            rows.append({
                "sample_id": sample_id,
                "sample_type": getattr(row, "sample_type", ""),
                "representation_split": sa.get("representation_split", "") if isinstance(sa, dict) else "",
                "master_split": sa.get("master_split", "") if isinstance(sa, dict) else "",
                "S": float(s_mean),
                "S_M1": float(s_per["M1"]),
                "S_M2": float(s_per["M2"]),
                "S_M12": float(s_per["M12"]),
                "d_M1_M2": float(d_m1_m2),     # raw geodesic distance
                "d_M12_M1": float(d_m12_m1),
                "d_M12_M2": float(d_m12_m2),
                "D": float(d_score),
                "R": float(r_signed),
                "abs_R": abs(float(r_signed)),
                "n_prompts": len(expected_prompt_ids),
            })
            processed += 1
            if processed % 200 == 0:
                print(f"[raw-check] processed {processed}", flush=True)
        except Exception as exc:
            print(f"[raw-check] skip {sample_id}: {exc}", flush=True)

    print(f"[raw-check] done. processed={processed}, "
          f"skipped_incomplete={skipped_incomplete}", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return output_path


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--unified-manifest", required=True)
    parser.add_argument("--main-manifest", required=True)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--split-assignment", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    p = compute_raw_layernorm_sdr(
        cache_root=args.cache_root,
        unified_manifest=args.unified_manifest,
        main_manifest=args.main_manifest,
        prompt_set_path=args.prompt_set,
        split_assignment_path=args.split_assignment,
        model_key=args.model_key,
        protocol=args.protocol,
        output_path=args.output,
    )
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
