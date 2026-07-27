"""Build per-model C/A relation_dataset.jsonl for cache_matrix_20260722.

Mirrors the first three stages of calibrate_thresholds.py (manifest filter,
state_dataset, state_bundles, relation_dataset) but skips the checkpoint /
export / SDR / thresholds stages — those require a TME checkpoint and are not
needed for the C/A TME GRU+SDR training input.

Inputs (per model):
  - source cache manifest   : /home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/{cache_manifests/internvl3_5_8b | source/<MODEL>}/manifest.jsonl
  - protocol label manifest : data/processed/manifests/protocol_manifests_merged/{proto}_merged_primary.jsonl
  - split assignment        : outputs/cache_matrix_20260722/split_assignments/{proto}.jsonl
  - prompt set              : configs/prompts/equiv_sets/{proto}_main_p8_seed20260717.yaml

Output:
  outputs/cache_matrix_20260722/relation_data/<MODEL>/<PROTO>/<proto>_main_p8_seed20260717/relation_dataset.jsonl

Steps (per model):
  1. setup_cache_manifests       : build prompt_cache + prompt_conditioned caches
                                    (idempotent; mirrors run_sdr.sh / run_calibrate.sh)
  2. filter_cache_manifest       : canonical-prompt filtered cache + manifest
                                    (idempotent; uses filter_cache_manifest.py)
  3. _filter_manifest_to_assigned: drop cross-domain / unassigned rows
  4. build_state_dataset         : state rows (one per sample × condition × layer)
  5. build_state_bundles         : sample-level bundles (one row per sample × prompt)
  6. build_relation_dataset      : relation rows (8 prompts × N samples)

Usage:
  python scripts/cache_matrix_20260722/build_ca_datasets.py [--model MODEL]

Omit --model to build all 13 valid models.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from mprisk.data.protocol_views import normalize_protocol
from mprisk.data.state_bundle import build_state_bundles
from mprisk.data.state_dataset import build_state_dataset
from mprisk.representation.relation_dataset import build_relation_dataset
from mprisk.setup_helper import setup_cache_manifests

# 13 valid models (drop phi3_5_vision / phi4_multimodal / llava_v1_5_7b)
MODELS = {
    # VT (11)
    "gemma3_12b":               "vt",
    "gemma3_4b":                "vt",
    "glm4_6v_flash":            "vt",
    "llava_onevision_qwen2_7b": "vt",
    "minicpm_v_2_6":            "vt",
    "minicpm_v_4_5":            "vt",
    "internvl3_5_8b":           "vt",
    "qwen2_5_vl_7b":            "vt",
    "qwen3_5_4b":               "vt",
    "qwen3_5_9b":               "vt",
    "qwen3_vl_8b":              "vt",
    # VA (2)
    "gemma4_12b":               "va",
    "qwen2_5_omni_7b":          "va",
}

CANONICAL_PROMPT = "pregen_risk_v1_p001"

# Per-model source cache root. InternVL has its own materialized package
# outside the source/ tree.
INTERNVL_CACHE = Path(
    "/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/cache_manifests/internvl3_5_8b"
)
SOURCE_BASE = Path(
    "/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722/source"
)

# Output paths
OUTPUT_BASE = REPO / "outputs/cache_matrix_20260722"
SETUP_BASE = OUTPUT_BASE / "cache_manifests"
FILTERED_BASE = OUTPUT_BASE / "cache_manifests_filtered"
RELATION_BASE = OUTPUT_BASE / "relation_data"
WORK_BASE = OUTPUT_BASE / "_ca_dataset_build"
# IMPORTANT: must use the legacy split assignment (seed20260716 v2), not the
# cache_matrix_20260722 freshly-built one (seed20260717 v1 has different
# split_group_id format that the label manifest does not match).
SPLIT_BASE = REPO / "data/processed/manifests/splits/representation_v1"
MANIFEST_BASE = REPO / "data/processed/manifests/protocol_manifests_merged"
PROMPT_SET_BASE = REPO / "configs/prompts/equiv_sets"


def source_cache_root(model: str) -> Path:
    if model == "internvl3_5_8b":
        return INTERNVL_CACHE
    return SOURCE_BASE / model


def filter_manifest_to_assigned(manifest_path: Path, split_assignment: Path) -> Path:
    """Drop rows whose sample_id is not in the split assignment, and rewrite
    the `split` field to match the assignment's master_split.

    Cross-domain rows (master_split != train/val/test) are skipped.
    Returns path to the filtered manifest next to the source.
    """
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
            if master not in {"train", "val", "test"}:
                continue
            row["split"] = master
            filtered_rows.append(row)

    filtered_path = manifest_path.parent / f"{manifest_path.stem}_assigned_only.jsonl"
    with filtered_path.open("w") as fh:
        for row in filtered_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return filtered_path


def build_one(model: str, proto: str, *, force: bool = False) -> int:
    """Build relation_dataset for one (model, proto). Returns row count.

    proto is lowercase ("vt"/"va"). normalize_protocol() returns UPPERCASE
    ("VT"/"VA") — used only for paths following the UPPERCASE convention
    (relation_data/<MODEL>/VT/...). All setup_cache / state_dataset /
    state_bundles paths use the lowercase proto from the YAML.
    """
    proto_norm = normalize_protocol(proto)  # uppercase
    proto_lower = proto.lower()             # lowercase (matches YAML)
    proto_upper = proto_norm                # uppercase
    prompt_set_key = f"{proto_lower}_main_p8_seed20260717"

    dst_dir = RELATION_BASE / model / proto_upper / prompt_set_key
    dst = dst_dir / "relation_dataset.jsonl"
    if dst.exists() and not force:
        n = sum(1 for _ in dst.open())
        print(f"[SKIP] {model}: already exists ({n} rows) at {dst}")
        return n

    src_root = source_cache_root(model)
    src_manifest = src_root / "manifest.jsonl"
    if not src_manifest.exists():
        print(f"[FAIL] {model}: source manifest missing {src_manifest}", flush=True)
        return 0

    # Paths
    setup_root = SETUP_BASE / model
    prompt_cache_manifest = setup_root / "prompt_cache_manifest.jsonl"
    # Path matches setup_cache_manifests convention:
    # <setup_root>/prompt_conditioned_cache/<model>/<proto_lower>/<prompt_set_key>/manifest.jsonl
    prompt_conditioned_manifest = (
        setup_root
        / "prompt_conditioned_cache"
        / model
        / proto_lower
        / prompt_set_key
        / "manifest.jsonl"
    )
    prompt_set_path = PROMPT_SET_BASE / f"{proto_lower}_main_p8_seed20260717.yaml"
    split_assignment = SPLIT_BASE / f"representation_split_assignment_v1_{proto_lower}.jsonl"
    label_manifest = MANIFEST_BASE / f"{proto_lower}_merged_primary.jsonl"

    filtered_cache_root = FILTERED_BASE / model
    filtered_cache_manifest = filtered_cache_root / "manifest.jsonl"
    filtered_wrapped = filtered_cache_root / "unified_full_cache_manifest.json"

    work_root = WORK_BASE / model
    work_root.mkdir(parents=True, exist_ok=True)

    # Step 1: setup_cache_manifests (prompt-conditioned cache + prompt cache manifest)
    if not prompt_cache_manifest.exists() or not prompt_conditioned_manifest.exists():
        print(f"[BUILD] {model}: setup_cache_manifests -> {setup_root}", flush=True)
        setup_cache_manifests(
            cache_root=str(src_root),
            prompt_set_path=str(prompt_set_path),
            model_key=model,
            output_root=setup_root,
        )

    # Step 2: filter_cache_manifest.py (canonical prompt filter, idempotent)
    if not filtered_wrapped.exists():
        print(f"[BUILD] {model}: filter_cache_manifest -> {filtered_cache_root}", flush=True)
        subprocess.run(
            [
                "python",
                str(REPO / "scripts/cache_matrix_20260722/filter_cache_manifest.py"),
                "--source-cache-root", str(src_root),
                "--target-cache-root", str(filtered_cache_root),
                "--canonical-prompt", CANONICAL_PROMPT,
            ],
            check=True,
            cwd=str(REPO),
        )

    # Step 3: filter to assigned samples
    print(f"[BUILD] {model}: filter manifest to assigned", flush=True)
    assigned_manifest = filter_manifest_to_assigned(label_manifest, split_assignment)

    # Step 4: state_dataset
    print(f"[BUILD] {model}: state_dataset", flush=True)
    state_dataset_result = build_state_dataset(
        manifest_paths=[assigned_manifest],
        cache_root=filtered_cache_root,
        manifest_path=Path("unified_full_cache_manifest.json"),
        model_key=model,
        protocol=proto_norm,
        split_assignment_path=split_assignment,
        output_dir=work_root / "state_data" / model / proto_norm,
        strict_shape=False,
    )

    # Step 5: state_bundles
    print(f"[BUILD] {model}: state_bundles", flush=True)
    bundle_result = build_state_bundles(
        state_dataset_manifest_path=state_dataset_result.manifest_path,
        prompt_cache_manifest_path=prompt_cache_manifest,
        prompt_conditioned_cache_manifest_path=prompt_conditioned_manifest,
        model_key=model,
        protocol=proto_norm,
        prompt_set_path=prompt_set_path,
        prompt_set_key=prompt_set_key,
        output_root=work_root / "state_bundles",
    )

    # Step 6: relation_dataset
    print(f"[BUILD] {model}: relation_dataset -> {dst_dir}", flush=True)
    # Use the prompt_set SHA + IDs that match what qwen3_vl_8b_tme_sdr.yaml expects
    import hashlib
    prompt_set_sha = hashlib.sha256(prompt_set_path.read_bytes()).hexdigest()
    expected_prompt_ids = [
        "pregen_risk_v1_p001", "pregen_risk_v1_p008", "pregen_risk_v1_p012",
        "pregen_risk_v1_p018", "pregen_risk_v1_p022", "pregen_risk_v1_p054",
        "pregen_risk_v1_p056", "pregen_risk_v1_p067",
    ]

    relation_result = build_relation_dataset(
        bundle_manifest_path=bundle_result.manifest_path,
        output_dir=dst_dir,
        prompt_set_key=prompt_set_key,
        prompt_set_artifact_sha256=prompt_set_sha,
        expected_prompt_count=8,
        expected_prompt_ids=expected_prompt_ids,
    )

    n = relation_result.row_count
    print(f"[OK] {model}: {n} rows -> {dst}", flush=True)
    return n


def main(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Specific model; default all 13")
    p.add_argument("--force", action="store_true", help="Rebuild even if dst exists")
    args = p.parse_args(argv)

    models = [args.model] if args.model else list(MODELS.keys())
    total = 0
    failed: list[str] = []
    for m in models:
        proto = MODELS.get(m)
        if proto is None:
            print(f"[SKIP] unknown model {m}", flush=True)
            continue
        try:
            n = build_one(m, proto, force=args.force)
            total += n
        except Exception as e:
            print(f"[FAIL] {m}: {type(e).__name__}: {e}", flush=True)
            failed.append(m)
    print(f"\nTotal: {total} rows across {len(models)} models", flush=True)
    if failed:
        print(f"Failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
