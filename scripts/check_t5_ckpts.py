#!/usr/bin/env python3
"""Check T5 best_checkpoint.pt epoch alignment; cp unconstrained -> best if mismatched.

M-A1-R5-5: --root is now configurable (defaults to the historical path so
            existing call sites keep working).
M-A1-R5-6: track mismatches and fixed separately; print dry-run hint
            unconditionally when --apply is not set.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

DEFAULT_ROOT = "outputs/canonical_rerun_v2/T5_lstm_ca_frozen"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=(
            "Run-root containing per-run subdirs with best_checkpoint.pt, "
            f"unconstrained_best_checkpoint.pt and train_metrics.json. "
            f"Defaults to {DEFAULT_ROOT!r}."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually copy unconstrained_best_checkpoint.pt over best_checkpoint.pt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(args.root)
    apply = args.apply
    if not root.is_dir():
        print(f"T5 root not found: {root}")
        return 1

    mismatches = 0
    fixed = 0
    skipped = 0

    for run in sorted(root.iterdir()):
        if not run.is_dir():
            continue
        bc = run / "best_checkpoint.pt"
        uc = run / "unconstrained_best_checkpoint.pt"
        mt = run / "train_metrics.json"
        if not (bc.is_file() and uc.is_file() and mt.is_file()):
            print(f"SKIP {run.name}: missing files")
            skipped += 1
            continue

        bc_ckpt = torch.load(bc, map_location="cpu")
        uc_ckpt = torch.load(uc, map_location="cpu")
        m = json.loads(mt.read_text())

        bc_epoch = bc_ckpt.get("epoch")
        bc_best = bc_ckpt.get("best_epoch")
        uc_epoch = uc_ckpt.get("epoch")
        metric_best = m.get("best_epoch")
        mismatch = bc_epoch != metric_best or bc_best != metric_best

        if mismatch:
            mismatches += 1
            print(
                f"MISMATCH {run.name}: "
                f"bc_epoch={bc_epoch} bc_best={bc_best} "
                f"uc_epoch={uc_epoch} metric_best={metric_best} -> "
                f"cp unconstrained -> best"
            )
            if apply:
                shutil.copy2(uc, bc)
                # Verify
                new_bc = torch.load(bc, map_location="cpu")
                print(
                    f"   FIXED: new bc_epoch={new_bc.get('epoch')} "
                    f"bc_best={new_bc.get('best_epoch')}"
                )
                fixed += 1
        else:
            print(
                f"OK       {run.name}: bc_epoch={bc_epoch} "
                f"bc_best={bc_best} metric_best={metric_best}"
            )

    print(
        f"\nSummary: {mismatches} mismatches, {fixed} fixed, {skipped} skipped"
    )
    if not apply:
        print("(dry-run; rerun with --apply to execute the copies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
