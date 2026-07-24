#!/usr/bin/env python3
"""SOTA TME (C5 v3-B e2e M/N) — recompute SDR scores from scratch.

Loads v3-B encoder (shared GRU per condition), runs each sample's
M1/M2/M12 trajectories through it, gets 128-d L2-normalized condition_z per
condition per prompt, then computes spherical SDR via the project's
compute_spherical_state.

Output: outputs/v2/state_data/qwen3_vl_8b/VT/sdr_scores_v3b_sota.jsonl
        outputs/v2/state_data/qwen3_vl_8b/VT/thresholds_v3b_sota.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

torch.set_num_threads(8)

PROJ_ROOT = Path(__file__).resolve().parents[3]
V2_SRC = PROJ_ROOT / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from mprisk.cache.prefill_extract import extract_t0_trajectory
from mprisk.cache.hidden_state_cache import normalize_protocol
from mprisk.data.manifests import read_final_manifest
from mprisk.representation.relation_models import SequentialTrajectoryEncoderV1
from mprisk.state.spherical import compute_spherical_state
from mprisk.state.thresholds import calibrate_registered_aligned_thresholds

CONDITIONS = ("M1", "M2", "M12")
COND_IDX = {"M1": 0, "M2": 1, "M12": 2}


def load_prompt_ids(prompt_set_path: Path) -> list[str]:
    with open(prompt_set_path, "r", encoding="utf-8") as f:
        ps = yaml.safe_load(f)
    return [t["prompt_id"] for t in ps["templates"] if t.get("enabled", True)]


def load_split_assignment(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split = row.get("representation_split", "")
            for sid in row.get("sample_ids", []):
                out[sid] = split
    return out


def load_sample_type_map(main_manifest: Path, protocol: str) -> dict[str, str]:
    rows = read_final_manifest(main_manifest)
    proto = normalize_protocol(protocol)
    return {
        r.sample_id: r.sample_type
        for r in rows
        if normalize_protocol(r.protocol) == proto
    }


def scan_cache(cache_root: Path, *, model_key: str, prompt_ids: set[str]):
    """Read manifest.jsonl and return sample_id -> {cond -> {prompt_id -> entry}}."""
    from mprisk.cache.cache_manifest import _can_materialize_entry, _entry_from_row

    out: dict[str, dict[str, dict[str, object]]] = {}
    manifest = cache_root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("model_key") != model_key:
                continue
            cond = row.get("condition")
            if cond not in CONDITIONS:
                continue
            pid = row.get("prompt_id") or (row.get("metadata") or {}).get("prompt_id")
            if pid is None or pid not in prompt_ids:
                continue
            if not _can_materialize_entry(row):
                continue
            entry = _entry_from_row(row, cache_root=cache_root)
            slot = out.setdefault(entry.sample_id, {c: {} for c in CONDITIONS})
            slot[cond].setdefault(pid, entry)
    return out


def encode_one_sample(
    encoder: nn.Module,
    info: dict[str, dict[str, object]],
    prompt_ids: list[str],
    *,
    device: torch.device,
) -> dict[str, dict[str, list[float]]] | None:
    """Return {cond: {prompt_id: condition_z_list}} or None if any missing.

    condition_z is a 128-d L2-normalized vector.
    """
    out: dict[str, dict[str, list[float]]] = {c: {} for c in CONDITIONS}
    # We run one prompt at a time (keeps memory small); batched per (cond, prompt)
    for pid in prompt_ids:
        per_cond_traj = []
        ok = True
        for cond in CONDITIONS:
            entry = info[cond].get(pid)
            if entry is None:
                ok = False
                break
            try:
                traj = extract_t0_trajectory(entry)  # [L, H]
            except Exception:
                ok = False
                break
            per_cond_traj.append(traj)
        if not ok:
            return None
        # Stack [3, L, H]
        arr = np.stack(per_cond_traj, axis=0).astype(np.float32)
        t = torch.from_numpy(arr).unsqueeze(0).to(device)  # [1, 3, L, H]
        with torch.no_grad():
            cond_z = encoder(t)  # [1, 3, 128]
        cond_z_np = cond_z.squeeze(0).cpu().float().numpy()  # [3, 128]
        for ci, cond in enumerate(CONDITIONS):
            out[cond][pid] = cond_z_np[ci].tolist()
    return out


def main():
    model_key = "qwen3_vl_8b"
    protocol = "VT"
    seed = 20260717
    prompt_set_key = f"vt_main_p8_seed{seed}"
    encoder_path = PROJ_ROOT / f"outputs/canonical_rerun_v2/C5_tme_v3b_e2e_mn/{model_key}_seed{seed}/best_encoder.pt"
    cache_root = PROJ_ROOT / f"outputs/prefill_cache/{model_key}/{prompt_set_key}"
    split_path = PROJ_ROOT / "data/processed/manifests/splits/representation_v1/representation_split_assignment_v1_vt.jsonl"
    prompt_set_path = PROJ_ROOT / f"configs/prompts/equiv_sets/{prompt_set_key}.yaml"
    main_manifest = PROJ_ROOT / "data/processed/manifests/unified_sample_manifest.jsonl"
    output_dir = PROJ_ROOT / f"outputs/v2/state_data/{model_key}/{protocol}"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "sdr_scores_v3b_sota.jsonl"
    thresholds_path = output_dir / "thresholds_v3b_sota.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    # Load encoder
    print(f"loading encoder: {encoder_path}", flush=True)
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    print(
        f"  arch={ckpt.get('architecture_version')} "
        f"input_dim={ckpt.get('input_dim')} "
        f"hidden={ckpt.get('sequence_hidden_dim')} "
        f"embed={ckpt.get('embed_dim')} "
        f"best_epoch={ckpt.get('best_epoch')}",
        flush=True,
    )
    encoder = SequentialTrajectoryEncoderV1(
        input_dim=int(ckpt["input_dim"]),
        sequence_hidden_dim=int(ckpt["sequence_hidden_dim"]),
        embed_dim=int(ckpt["embed_dim"]),
        dropout=0.0,  # turn off dropout for inference
    )
    # v3-B uses prefix "encoder." for the shared condition encoder
    sd_raw = ckpt["model_state_dict"]
    sd_remap = {}
    for k, v in sd_raw.items():
        if k.startswith("encoder."):
            # strip "encoder." prefix to match SequentialTrajectoryEncoderV1 top-level params
            sd_remap[k[len("encoder."):]] = v
    missing, unexpected = encoder.load_state_dict(sd_remap, strict=False)
    print(f"  missing={missing} unexpected={unexpected}", flush=True)
    encoder.eval().to(device)

    # Load prompts / split / sample types
    prompt_ids = load_prompt_ids(prompt_set_path)
    print(f"prompt_ids ({len(prompt_ids)}): {prompt_ids}", flush=True)
    split_of = load_split_assignment(split_path)
    sample_types = load_sample_type_map(main_manifest, protocol)
    print(f"samples in manifest: {len(sample_types)}; in split: {len(split_of)}", flush=True)

    # Scan cache
    print(f"scanning cache: {cache_root}", flush=True)
    cache_index = scan_cache(cache_root, model_key=model_key, prompt_ids=set(prompt_ids))
    print(f"  samples in cache: {len(cache_index)}", flush=True)

    # We need: official_test + aligned_calibration (Conflict + Aligned)
    keep_splits = {"official_test", "aligned_calibration"}
    keep_samples = [
        sid for sid, sp in split_of.items()
        if sp in keep_splits and sid in sample_types and sid in cache_index
    ]
    print(f"computing SDR for {len(keep_samples)} samples (splits={keep_splits})", flush=True)

    rows_written = 0
    skipped_incomplete = 0
    t0 = time.time()
    with open(scores_path, "w", encoding="utf-8") as fout:
        for i, sid in enumerate(keep_samples, 1):
            info = cache_index[sid]
            embeddings = encode_one_sample(encoder, info, prompt_ids, device=device)
            if embeddings is None:
                skipped_incomplete += 1
                continue
            sp = split_of[sid]
            stype = sample_types[sid]
            bundle = {
                "sample_id": sid,
                "sample_type": stype,
                "calibration_split": sp,
                "embeddings": embeddings,
            }
            try:
                sdr = compute_spherical_state(bundle)
            except Exception as exc:
                print(f"  skip {sid}: sdr failed: {exc}", flush=True)
                skipped_incomplete += 1
                continue
            # Add split fields
            sdr["representation_split"] = sp
            sdr["master_split"] = "test" if sp in ("official_test", "aligned_calibration") else "train"
            fout.write(json.dumps(sdr) + "\n")
            rows_written += 1
            if i % 100 == 0:
                rate = i / (time.time() - t0 + 1e-9)
                eta = (len(keep_samples) - i) / rate
                print(
                    f"  [{i}/{len(keep_samples)}] written={rows_written} skipped={skipped_incomplete} "
                    f"rate={rate:.1f}/s eta={eta:.0f}s",
                    flush=True,
                )

    print(f"DONE: written={rows_written} skipped={skipped_incomplete} -> {scores_path}", flush=True)

    # Calibrate thresholds
    with open(scores_path, "r", encoding="utf-8") as fin:
        rows = [json.loads(l) for l in fin]
    payload = calibrate_registered_aligned_thresholds(rows, quantile_level=0.95)
    # Rename keys to match older convention
    thresholds_out = {
        "schema": payload["schema"],
        "kappa": payload["kappa"],
        "tau": payload["tau"],
        "kappa_quantile": payload["quantile_level"],
        "tau_quantile": payload["quantile_level"],
        "delta_policy": "per_sample_synchronous_prompt_bootstrap_1.96se",
        "calibration_split": "aligned_calibration",
        "n_calibration_rows": payload["aligned_count"],
        "stable_aligned_count": payload["stable_aligned_count"],
    }
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_out, f, indent=2)
    print(f"WROTE thresholds: kappa={thresholds_out['kappa']:.6f} tau={thresholds_out['tau']:.4f} "
          f"n_cal={thresholds_out['n_calibration_rows']} -> {thresholds_path}", flush=True)


if __name__ == "__main__":
    main()
