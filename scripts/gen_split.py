"""Generate protocol-independent VT and VA representation split assignments.

Reads vt_merged_primary.jsonl and va_merged_primary.jsonl, re-splits the gen
domain stratified by sample_type (80/10/10), preserves ch_sims_v2 master_split,
and applies the v1 calibration rule (50% of Aligned val groups go to
aligned_calibration via sha256(seed:group) rank).
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20260716
GEN_TRAIN_FRAC = 0.8
GEN_VAL_FRAC = 0.1
GEN_TEST_FRAC = 0.1
ASSIGNMENT_SCHEMA = "mprisk_representation_split_assignment_v1"
CONFIG_KEY_VT = "representation_split_seed20260716_vt_v2"
CONFIG_KEY_VA = "representation_split_seed20260716_va_v2"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/processed/manifests/splits/representation_v1"


def sha_rank(seed, group):
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def stratified_split(gen_rows):
    """Split gen rows by sample_type stratified 80/10/10 using sha256(seed:group)."""
    by_type = defaultdict(list)
    for r in gen_rows:
        by_type[r["sample_type"]].append(r)
    train, val, test = [], [], []
    for stype, rows in by_type.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (sha_rank(SEED, r["split_group_id"]), r["sample_id"]),
        )
        n = len(rows_sorted)
        n_test = max(1, round(n * GEN_TEST_FRAC))
        n_val = max(1, round(n * GEN_VAL_FRAC))
        test.extend(rows_sorted[:n_test])
        val.extend(rows_sorted[n_test : n_test + n_val])
        train.extend(rows_sorted[n_test + n_val :])
    return train, val, test


def build_split(merged_path, protocol_label, config_key, out_filename, *, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    with open(merged_path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    gen_rows = [r for r in rows if r["sample_id"].startswith("gen:")]
    ch_rows = [r for r in rows if r["sample_id"].startswith("ch_sims_v2:")]
    print(f"\n=== {protocol_label} ({merged_path}) ===")
    print(f"  total={len(rows)} gen={len(gen_rows)} ch_sims={len(ch_rows)}")
    print(
        f"  gen sample_type dist: {dict(Counter(r['sample_type'] for r in gen_rows))}"
    )
    print(
        f"  ch  master_split dist: {dict(Counter(r['split'] for r in ch_rows))}"
    )

    g_train, g_val, g_test = stratified_split(gen_rows)
    print(
        f"  gen re-split: train={len(g_train)} val={len(g_val)} test={len(g_test)}"
    )

    gen_master = {}
    for r in g_train:
        gen_master[r["split_group_id"]] = "train"
    for r in g_val:
        gen_master[r["split_group_id"]] = "val"
    for r in g_test:
        gen_master[r["split_group_id"]] = "test"

    # ch_sims_v2 is a separate natural-domain corpus; never mix it into gen
    # train/val/test splits. All ch_sims_v2 groups become cross_domain_test.
    ch_master = {r["split_group_id"]: "cross_domain_test" for r in ch_rows}

    groups = defaultdict(list)
    for r in rows:
        groups[r["split_group_id"]].append(r)

    group_master = {}
    for gid, grows in groups.items():
        if gid in gen_master:
            group_master[gid] = gen_master[gid]
        else:
            group_master[gid] = ch_master.get(gid, "cross_domain_test")

    val_aligned_groups = sorted(
        [
            gid
            for gid, grows in groups.items()
            if group_master[gid] == "val"
            and all(r["sample_type"] == "Aligned" for r in grows)
        ],
        key=lambda gid: sha_rank(SEED, gid),
    )
    n_calib = (
        math.floor(len(val_aligned_groups) * 0.5) if val_aligned_groups else 0
    )
    calib_set = set(val_aligned_groups[:n_calib])

    assignment = {}
    for gid in groups:
        ms = group_master[gid]
        if ms == "cross_domain_test":
            assignment[gid] = "cross_domain_test"
        elif ms == "train":
            assignment[gid] = "relation_train"
        elif ms == "test":
            assignment[gid] = "official_test"
        elif gid in calib_set:
            assignment[gid] = "aligned_calibration"
        else:
            assignment[gid] = "relation_val"

    cnt = Counter(assignment.values())
    sample_cnt = Counter()
    for gid, split in assignment.items():
        sample_cnt[split] += len(groups[gid])
    print(f"  group counts by rep_split: {dict(cnt)}")
    print(f"  sample counts by rep_split: {dict(sample_cnt)}")
    label_cnt = defaultdict(Counter)
    for gid, split in assignment.items():
        for r in groups[gid]:
            label_cnt[split][r["sample_type"]] += 1
    for s in sorted(label_cnt):
        print(f"  {s} labels: {dict(label_cnt[s])}")

    # Check train/val must each contain both Aligned and Conflict (skip if not possible)
    for split in ("relation_train", "relation_val"):
        if not {"Aligned", "Conflict"} <= set(label_cnt[split].keys()):
            print(f"  WARNING: {split} missing both labels: {dict(label_cnt[split])}")

    manifest_rows = []
    for gid in sorted(groups):
        grows = groups[gid]
        ms = group_master[gid]
        manifest_rows.append(
            {
                "schema": ASSIGNMENT_SCHEMA,
                "config_key": config_key,
                "split_group_id": gid,
                "master_split": ms,
                "representation_split": assignment[gid],
                "sample_ids": sorted(str(r["sample_id"]) for r in grows),
                "sample_count": len(grows),
                "protocols": sorted({str(r.get("protocol", "")) for r in grows}),
                "source_datasets": sorted(
                    {str(r.get("source_dataset", "")) for r in grows}
                ),
            }
        )

    out_path = out_dir / out_filename
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as h:
        for r in manifest_rows:
            h.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        h.flush()
        import os

        os.fsync(h.fileno())
    import os

    os.replace(tmp, out_path)
    sha = hashlib.sha256()
    with out_path.open("rb") as h:
        for chunk in iter(lambda: h.read(1024 * 1024), b""):
            sha.update(chunk)
    print(f"  wrote {out_path} ({len(manifest_rows)} rows, sha256={sha.hexdigest()[:16]})")

    # Build summary
    summary = {
        "schema": "mprisk_representation_split_summary_v1",
        "config_key": config_key,
        "config_path": "protocol-merged-v2 (in-script)",
        "source_merged_manifest": merged_path,
        "seed": SEED,
        "ranking_rule": "sha256(seed:split_group_id)",
        "scope": "all_valid_conflict_aligned_per_protocol",
        "gen_resplit_fractions": {
            "train": GEN_TRAIN_FRAC,
            "val": GEN_VAL_FRAC,
            "test": GEN_TEST_FRAC,
        },
        "gen_resplit_stratified_by": "sample_type",
        "ch_sims_master_split": "all ch_sims_v2 groups -> cross_domain_test",
        "calibration_fraction": 0.5,
        "calibration_rounding": "floor",
        "group_counts": dict(sorted(cnt.items())),
        "sample_counts": dict(sorted(sample_cnt.items())),
        "label_counts": {s: dict(sorted(label_cnt[s].items())) for s in sorted(label_cnt)},
        "group_count": len(groups),
        "sample_count": len(rows),
        "manifest_path": str(out_path),
        "manifest_sha256": sha.hexdigest(),
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as h:
        json.dump(summary, h, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"  wrote {summary_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: auto-detected from this file's location).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for split JSONLs (default: <root>/data/processed/manifests/splits/representation_v1).",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.output_dir if args.output_dir else root / "data/processed/manifests/splits/representation_v1"
    build_split(
        str(root / "data/processed/manifests/protocol_manifests_merged/vt_merged_primary.jsonl"),
        "VT",
        CONFIG_KEY_VT,
        "representation_split_assignment_v1_vt.jsonl",
        out_dir=out_dir,
    )
    build_split(
        str(root / "data/processed/manifests/protocol_manifests_merged/va_merged_primary.jsonl"),
        "VA",
        CONFIG_KEY_VA,
        "representation_split_assignment_v1_va.jsonl",
        out_dir=out_dir,
    )
