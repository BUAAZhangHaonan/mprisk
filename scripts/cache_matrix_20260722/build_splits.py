"""Build split-assignment JSONL files for cache_matrix_20260722.

Reads sample-level manifests from cache_matrix_20260722/manifests/ and emits
grouped split-assignment files matching the schema used by
`scripts/_trainer_lib.load_split_assignment`:

    load_split_assignment consumes only `representation_split` and `sample_ids`
    from each row. The downstream trainer (train_tme_e2e) only uses three
    representation_split values: relation_train / relation_val / official_test.
    We still emit aligned_calibration and cross_domain_test rows for downstream
    consumers (PA / TME calibration, cross-domain eval) but they are ignored by
    the MN trainer.

Layout for each protocol (VT, VA):
    - Read source_* + target_* manifests.
    - Stratified split by (sample_type x source_dataset), 70/15/15 train/val/test.
    - master_split mapping:
        * source-domain train -> relation_train (full) + aligned_calibration
          (random subset of train, 50% by default)
        * source-domain val   -> relation_val
        * source-domain test  -> official_test
        * target-domain (ch_sims_v2) -> cross_domain_test (all of them, kept
          out of train/val/test so the in-domain MN trainer does not see them)

Seed: 20260717 (matches prompt-set selection seed for cache_matrix_20260722).

Outputs:
    outputs/cache_matrix_20260722/split_assignments/vt.jsonl
    outputs/cache_matrix_20260722/split_assignments/va.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# sklearn is available in the `mprisk` env; fall back to manual stratified
# shuffle if not.
try:
    from sklearn.model_selection import train_test_split  # type: ignore
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


SCHEMA = "mprisk_representation_split_assignment_v1"
CONFIG_KEY_PROTO = "representation_split_seed20260717_{proto}_v1"
SPLIT_GROUP_SEP = "::"

# Seed pinned for cache_matrix_20260722 — matches the prompt-set selection
# seed in vt_main_p8_seed20260717.yaml / va_main_p8_seed20260717.yaml.
SEED = 20260717

# Fractions for the master split (apply only to source-domain samples).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
# Fraction of train rows reserved as aligned_calibration (subset, not extra).
ALIGNED_CALIB_FRAC_OF_TRAIN = 0.50


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _stratify_key(row: dict) -> str:
    """Combine sample_type and source_dataset for stratification.

    Using both gives finer-grained balancing when multiple source datasets are
    present; when source_dataset is constant this collapses to sample_type.
    """
    st = row.get("sample_type", "") or "UNKNOWN"
    sd = row.get("source_dataset", "") or "UNKNOWN"
    return f"{st}{SPLIT_GROUP_SEP}{sd}"


def _stratified_train_val_test(
    rows: list[dict],
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[dict]]:
    """Return {"train": [...], "val": [...], "test": [...]}.

    Stratifies by `_stratify_key`. Uses sklearn.train_test_split when
    available (two stacked calls), otherwise falls back to a manual per-stratum
    shuffle.
    """
    if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
        raise ValueError(
            f"fractions must sum to 1.0; got {train_frac}+{val_frac}+{test_frac}"
        )

    # Group rows by stratum.
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[_stratify_key(r)].append(r)

    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for key in sorted(buckets.keys()):
        items = list(buckets[key])
        # Stable ordering by sample_id so seed is deterministic regardless of
        # filesystem order.
        items.sort(key=lambda r: str(r.get("sample_id", "")))
        n = len(items)

        if n < 3:
            # Too small to split reliably — put everything in train.
            out["train"].extend(items)
            continue

        if _HAVE_SKLEARN:
            # First cut: train vs (val+test).
            val_test_frac = val_frac + test_frac
            train, val_test = train_test_split(
                items,
                test_size=val_test_frac,
                random_state=seed,
                shuffle=True,
            )
            # Second cut: val vs test. Keep relative ratio.
            if val_test:
                rel_val = val_frac / val_test_frac if val_test_frac > 0 else 0.0
                # test_size=1-rel_val gives val_size=rel_val
                val, test = train_test_split(
                    val_test,
                    test_size=(1.0 - rel_val),
                    random_state=seed,
                    shuffle=True,
                )
            else:
                val, test = [], []
        else:
            rng = random.Random(seed + hash(key) % (2**31))
            shuffled = list(items)
            rng.shuffle(shuffled)
            n_train = int(round(n * train_frac))
            n_val = int(round(n * val_frac))
            # remainder -> test
            n_test = n - n_train - n_val
            train = shuffled[:n_train]
            val = shuffled[n_train:n_train + n_val]
            test = shuffled[n_train + n_val:n_train + n_val + n_test]

        out["train"].extend(train)
        out["val"].extend(val)
        out["test"].extend(test)

    return out


def _split_group_id(row: dict) -> str:
    """Reconstruct a deterministic per-row group id (one row per sample here).

    Hash of sample_id keeps each row unique while staying stable across runs.
    """
    sid = str(row.get("sample_id", ""))
    h = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:12]
    return f"{sid}:{h}"


def _make_row(
    *,
    sample: dict,
    master_split: str,
    representation_split: str,
    protocol: str,
    config_key: str,
) -> dict:
    sid = str(sample.get("sample_id", ""))
    sd = sample.get("source_dataset", "")
    return {
        "schema": SCHEMA,
        "config_key": config_key,
        "master_split": master_split,
        "representation_split": representation_split,
        "protocols": [protocol.upper()],
        "sample_count": 1,
        "sample_ids": [sid],
        "source_datasets": [sd] if sd else [],
        "split_group_id": _split_group_id(sample),
    }


def _stable_subset(samples: list[dict], frac: float, seed: int) -> list[dict]:
    """Deterministic random subset of `samples` keyed by sample_id."""
    if frac >= 1.0:
        return list(samples)
    if frac <= 0.0:
        return []
    rng = random.Random(seed)
    ordered = sorted(samples, key=lambda r: str(r.get("sample_id", "")))
    n = len(ordered)
    k = int(round(n * frac))
    idx = set(rng.sample(range(n), k))
    return [ordered[i] for i in sorted(idx)]


def _build_for_protocol(
    *,
    protocol: str,
    source_manifest: Path,
    target_manifest: Path,
    out_path: Path,
    seed: int = SEED,
) -> dict:
    proto_upper = protocol.upper()
    config_key = CONFIG_KEY_PROTO.format(proto=protocol.lower())

    source_rows = _load_jsonl(source_manifest)
    target_rows = _load_jsonl(target_manifest)

    # Sanity: duplicate sample_ids inside each manifest.
    for label, rows in (("source", source_rows), ("target", target_rows)):
        sids = [r.get("sample_id") for r in rows]
        dup = {s for s in sids if sids.count(s) > 1}
        if dup:
            raise RuntimeError(
                f"[{proto_upper}] duplicate sample_ids in {label} manifest: "
                f"{sorted(dup)[:5]} (total {len(dup)})"
            )

    # Cross-manifest duplicate check (should not overlap).
    src_ids = {r.get("sample_id") for r in source_rows}
    tgt_ids = {r.get("sample_id") for r in target_rows}
    overlap = src_ids & tgt_ids
    if overlap:
        raise RuntimeError(
            f"[{proto_upper}] sample_ids overlap source vs target: "
            f"{sorted(overlap)[:5]} (total {len(overlap)})"
        )

    # Master split applies only to source domain. Target domain goes to
    # cross_domain_test entirely (used by PA / cross-domain eval, NOT by the
    # in-domain MN trainer).
    master_split = _stratified_train_val_test(
        source_rows,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        seed=seed,
    )

    # Aligned calibration = subset of train (no overlap with relation_train
    # would be wrong — calibration samples are a *separate* view of train, but
    # since each sample can only be in one row's `sample_ids`, we mark some
    # train samples as aligned_calibration instead of relation_train).
    # Decision: aligned_calibration takes ALIGNED_CALIB_FRAC_OF_TRAIN of train,
    # the rest stay as relation_train. This is consistent with the historical
    # v1 split semantics where the same sample could appear under multiple
    # representation_split rows. To stay safe and unambiguous we partition:
    train_for_relation = []
    train_for_calib: list[dict] = []
    if master_split["train"]:
        calib_subset = _stable_subset(
            master_split["train"], ALIGNED_CALIB_FRAC_OF_TRAIN, seed=seed
        )
        calib_ids = {r.get("sample_id") for r in calib_subset}
        for r in master_split["train"]:
            if r.get("sample_id") in calib_ids:
                train_for_calib.append(r)
            else:
                train_for_relation.append(r)

    out_rows: list[dict] = []

    def emit(samples: Iterable[dict], *, master: str, repr_split: str) -> None:
        for s in samples:
            out_rows.append(
                _make_row(
                    sample=s,
                    master_split=master,
                    representation_split=repr_split,
                    protocol=proto_upper,
                    config_key=config_key,
                )
            )

    # Source domain.
    emit(train_for_relation, master="train", repr_split="relation_train")
    emit(train_for_calib, master="train", repr_split="aligned_calibration")
    emit(master_split["val"], master="val", repr_split="relation_val")
    emit(master_split["test"], master="test", repr_split="official_test")
    # Target domain.
    emit(target_rows, master="test", repr_split="cross_domain_test")

    # Final sanity: every sample_id appears exactly once across emitted rows.
    seen: dict[str, str] = {}
    for row in out_rows:
        for sid in row["sample_ids"]:
            if sid in seen:
                raise RuntimeError(
                    f"[{proto_upper}] sample_id {sid} appears in both "
                    f"{seen[sid]} and {row['representation_split']}"
                )
            seen[sid] = row["representation_split"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary stats.
    n_src = len(source_rows)
    n_tgt = len(target_rows)
    by_repr: dict[str, int] = defaultdict(int)
    by_master: dict[str, int] = defaultdict(int)
    by_repr_stype: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in out_rows:
        by_repr[row["representation_split"]] += row["sample_count"]
        by_master[row["master_split"]] += row["sample_count"]
    # sample_type breakdown per representation_split
    sid_to_row: dict[str, dict] = {}
    for r in source_rows + target_rows:
        sid_to_row[r.get("sample_id", "")] = r
    for row in out_rows:
        for sid in row["sample_ids"]:
            stype = sid_to_row.get(sid, {}).get("sample_type", "?")
            by_repr_stype[row["representation_split"]][stype] += 1

    summary = {
        "protocol": proto_upper,
        "n_source": n_src,
        "n_target": n_tgt,
        "n_total": n_src + n_tgt,
        "by_master_split": dict(by_master),
        "by_representation_split": dict(by_repr),
        "by_representation_split_x_sample_type": {
            k: dict(v) for k, v in by_repr_stype.items()
        },
        "have_sklearn": _HAVE_SKLEARN,
        "seed": seed,
        "out_path": str(out_path),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722"),
        help="Root of cache_matrix_20260722 (containing manifests/)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/team/zhanghaonan/TAFFC/mprisk-v2"),
        help="mprisk-v2 repo root (where outputs/cache_matrix_20260722 lives)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["vt", "va"],
        choices=["vt", "va"],
    )
    args = parser.parse_args(argv)

    manifests = args.cache_root / "manifests"
    out_dir = args.repo_root / "outputs" / "cache_matrix_20260722" / "split_assignments"

    summaries: list[dict] = []
    for proto in args.protocols:
        src = manifests / f"source_{proto}.jsonl"
        tgt = manifests / f"target_{proto}.jsonl"
        for p in (src, tgt):
            if not p.exists():
                print(f"[error] manifest not found: {p}", file=sys.stderr)
                return 1
        out_path = out_dir / f"{proto}.jsonl"
        s = _build_for_protocol(
            protocol=proto,
            source_manifest=src,
            target_manifest=tgt,
            out_path=out_path,
            seed=args.seed,
        )
        summaries.append(s)

    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
