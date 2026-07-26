"""Per-model Aligned thresholds calibration for cache_matrix_20260722.

Mirrors the building blocks of run_core_sdr_pipeline but emits only the
calibrated thresholds.json. Reuses:
  - build_state_dataset (state rows + aligned_calibration propagation)
  - build_state_bundles (per-prompt bundles)
  - build_relation_dataset (relation contract)
  - export_frozen_representations (uses TME BiLSTM checkpoint)
  - compute_sdr_scores (spherical S/D/R)
  - calibrate_registered_aligned_thresholds (kappa/tau quantiles on
    representation_split=aligned_calibration AND sample_type=Aligned)

Inputs:
  --model, --protocol (vt|va), --prompt-set-key, --prompt-set,
  --manifest, --full-cache-root, --prompt-cache-manifest,
  --prompt-conditioned-cache-manifest, --split-assignment,
  --checkpoint, --output-dir, --device (default cuda)

Output:
  <output-dir>/thresholds.json  (schema mprisk_spherical_calibration)

Idempotent: if <output-dir>/thresholds.json exists and was produced by a
matching encoder sha256, exit 0 with no work.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mprisk.data.manifests import read_jsonl
from mprisk.data.protocol_views import normalize_protocol
from mprisk.data.state_bundle import build_state_bundles
from mprisk.data.state_dataset import build_state_dataset
from mprisk.representation.relation_dataset import build_relation_dataset
from mprisk.representation.relation_models import TME_PROXY_ANCHOR_V1
from mprisk.representation.training import TrainingConfig, export_frozen_representations
from mprisk.state.pipeline import compute_sdr_scores
from mprisk.state.thresholds import calibrate_registered_aligned_thresholds
from mprisk.utils.io import ensure_parent, write_json


def _ensure_wrapped_manifest(cache_root: Path) -> Path:
    """Ensure cache_root has a unified_full_cache_manifest.json the loader can read.

    cache_matrix_20260722 source caches store rows as JSONL (one JSON object
    per line at cache_root/manifest.jsonl). load_full_cache_manifest expects a
    JSON OBJECT with an entries list (DEFAULT_MANIFEST_PATH). We materialise
    a wrapped JSON next to the JSONL and return its path so the caller can pass
    it via manifest_path=.

    Idempotent: if the wrapped file already exists and is newer than the JSONL,
    return it without rewriting.
    """
    import json as _json

    jsonl = cache_root / "manifest.jsonl"
    wrapped = cache_root / "unified_full_cache_manifest.json"
    if wrapped.exists() and jsonl.exists() and wrapped.stat().st_mtime >= jsonl.stat().st_mtime:
        return wrapped
    if not jsonl.exists():
        # Already in wrapped form or absent; let the downstream loader handle it.
        return cache_root / "unified_full_cache_manifest.json"

    rows: list[dict] = []
    with jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(_json.loads(line))
    with wrapped.open("w") as fh:
        _json.dump({"entries": rows, "schema": "cache_matrix_20260722_wrapped_jsonl_v1"}, fh)
    return wrapped


def _filter_manifest_to_assigned(
    manifest_path: Path, split_assignment: Path
) -> tuple[Path, int, int]:
    """Materialise a manifest subset that lines up with the split assignment.

    Two fixes are applied:
      1. Drop rows whose sample_id is not covered by any assignment group.
      2. Overwrite each row's split field with the assignment's
         master_split so state_dataset._resolve_split_assignment's
         master_split check passes (curation marks some cross-domain rows
         as 'train' but the assignment may say 'test' / 'cross_domain_test').

    The result is written next to the source manifest and reused on later runs.
    """
    import json

    # Map every sample_id -> assignment master_split.
    sample_to_master: dict[str, str] = {}
    with split_assignment.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            master = row.get("master_split", "")
            for sample_id in row.get("sample_ids", []):
                sample_to_master[str(sample_id)] = master

    filtered_rows: list[dict] = []
    total = 0
    with manifest_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if sample_id not in sample_to_master:
                continue
            master = sample_to_master[sample_id]
            # Skip cross-domain rows: state_dataset only accepts
            # train/val/test master_split, and cross-domain samples are not
            # in the Aligned calibration target set anyway.
            if master not in {"train", "val", "test"}:
                continue
            # Align the split field with the assignment's master_split so
            # _resolve_split_assignment's strict check passes.
            row["split"] = master
            filtered_rows.append(row)

    filtered_path = manifest_path.parent / f"{manifest_path.stem}_assigned_only.jsonl"
    with filtered_path.open("w") as fh:
        for row in filtered_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + chr(10))

    return filtered_path, len(filtered_rows), total


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _already_calibrated(output_dir: Path, checkpoint: Path) -> bool:
    target = output_dir / "thresholds.json"
    if not target.is_file():
        return False
    sidecar = output_dir / "calibration_provenance.json"
    if not sidecar.is_file():
        return False
    import json
    payload = json.loads(sidecar.read_text())
    return payload.get("checkpoint_sha256") == _sha256(checkpoint)


def calibrate(
    *,
    model_key: str,
    protocol: str,
    prompt_set_key: str,
    manifest_path: Path,
    full_cache_root: Path,
    prompt_cache_manifest: Path,
    prompt_conditioned_cache_manifest: Path,
    prompt_set: Path,
    split_assignment: Path,
    checkpoint: Path,
    output_dir: Path,
    device: str,
) -> Path:
    """Run the calibration pipeline and write thresholds.json."""
    normalized_protocol = normalize_protocol(protocol)
    output_base = output_dir / "outputs"

    if _already_calibrated(output_dir, checkpoint):
        print(f"[calibrate] SKIP {model_key}: thresholds.json already present", flush=True)
        return output_dir / "thresholds.json"

    filtered_manifest, kept, total = _filter_manifest_to_assigned(
        manifest_path=manifest_path,
        split_assignment=split_assignment,
    )
    print(
        f"[calibrate] {model_key}: manifest filtered {kept}/{total} -> {filtered_manifest.name}",
        flush=True,
    )
    wrapped_manifest = _ensure_wrapped_manifest(full_cache_root)
    print(
        f"[calibrate] {model_key}: wrapped cache manifest -> {wrapped_manifest.name}",
        flush=True,
    )
    print(f"[calibrate] {model_key}: state_dataset", flush=True)
    state_dataset_result = build_state_dataset(
        manifest_paths=[filtered_manifest],
        cache_root=full_cache_root,
        manifest_path=wrapped_manifest,
        model_key=model_key,
        protocol=normalized_protocol,
        split_assignment_path=split_assignment,
        output_dir=output_base / "state_data" / model_key / normalized_protocol,
    )

    print(f"[calibrate] {model_key}: state_bundles", flush=True)
    bundle_result = build_state_bundles(
        state_dataset_manifest_path=state_dataset_result.manifest_path,
        prompt_cache_manifest_path=prompt_cache_manifest,
        prompt_conditioned_cache_manifest_path=prompt_conditioned_cache_manifest,
        model_key=model_key,
        protocol=normalized_protocol,
        prompt_set_path=prompt_set,
        prompt_set_key=prompt_set_key,
        output_root=output_base / "state_bundles",
    )

    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    training_config = TrainingConfig(**checkpoint_payload["training_config"])

    print(f"[calibrate] {model_key}: relation_dataset", flush=True)
    relation_dataset_result = build_relation_dataset(
        bundle_manifest_path=bundle_result.manifest_path,
        output_dir=(
            output_base
            / "representation_data"
            / model_key
            / normalized_protocol
            / prompt_set_key
        ),
        prompt_set_key=training_config.prompt_set_key,
        prompt_set_artifact_sha256=training_config.prompt_set_artifact_sha256,
        expected_prompt_count=training_config.expected_prompt_count,
        expected_prompt_ids=training_config.expected_prompt_ids,
    )

    print(f"[calibrate] {model_key}: export_frozen device={device}", flush=True)
    embedding_result = export_frozen_representations(
        dataset_path=relation_dataset_result.dataset_path,
        checkpoint_path=checkpoint,
        output_dir=(
            output_base
            / "embeddings"
            / model_key
            / normalized_protocol
            / prompt_set_key
            / TME_PROXY_ANCHOR_V1
        ),
        device=device,
    )

    state_output_dir = (
        output_base
        / "states"
        / model_key
        / normalized_protocol
        / prompt_set_key
        / TME_PROXY_ANCHOR_V1
    )
    print(f"[calibrate] {model_key}: compute_sdr_scores", flush=True)
    sdr_result = compute_sdr_scores(
        embedding_manifest_path=embedding_result.bundle_manifest_path,
        output_dir=state_output_dir,
    )

    rows = read_jsonl(sdr_result.scores_path)
    print(
        f"[calibrate] {model_key}: rows={len(rows)} "         f"aligned_calibration={sum(1 for r in rows if r.get('representation_split') == 'aligned_calibration')}",
        flush=True,
    )
    calibration = calibrate_registered_aligned_thresholds(rows)

    target = output_dir / "thresholds.json"
    write_json(target, calibration)

    provenance = {
        "schema": "cache_matrix_20260722_calibration_provenance_v1",
        "model_key": model_key,
        "protocol": normalized_protocol,
        "prompt_set_key": prompt_set_key,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "sdr_scores_path": str(sdr_result.scores_path),
        "thresholds_path": str(target),
        "kappa": calibration["kappa"],
        "tau": calibration["tau"],
        "aligned_count": calibration["aligned_count"],
    }
    write_json(ensure_parent(output_dir / "calibration_provenance.json"), provenance)

    print(
        f"[calibrate] DONE {model_key}: kappa={calibration['kappa']:.6f} "         f"tau={calibration['tau']:.6f} n={calibration['aligned_count']} -> {target}",
        flush=True,
    )
    return target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--protocol", required=True, choices=["vt", "va"])
    p.add_argument("--prompt-set-key", required=True)
    p.add_argument("--prompt-set", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--full-cache-root", required=True, type=Path)
    p.add_argument("--prompt-cache-manifest", required=True, type=Path)
    p.add_argument("--prompt-conditioned-cache-manifest", required=True, type=Path)
    p.add_argument("--split-assignment", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    calibrate(
        model_key=args.model,
        protocol=args.protocol,
        prompt_set_key=args.prompt_set_key,
        manifest_path=args.manifest,
        full_cache_root=args.full_cache_root,
        prompt_cache_manifest=args.prompt_cache_manifest,
        prompt_conditioned_cache_manifest=args.prompt_conditioned_cache_manifest,
        prompt_set=args.prompt_set,
        split_assignment=args.split_assignment,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
