#!/usr/bin/env python3
"""Check T5 best_checkpoint.pt epoch alignment; cp unconstrained -> best if mismatched."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path("outputs/canonical_rerun_v2/T5_lstm_ca_frozen")


def main() -> int:
    if not ROOT.is_dir():
        print(f"T5 root not found: {ROOT}")
        return 1

    apply = "--apply" in sys.argv
    fixed = 0
    skipped = 0

    for run in sorted(ROOT.iterdir()):
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

    print(f"\nSummary: {fixed} fixed, {skipped} skipped")
    if not apply and fixed > 0:
        print("(dry-run; rerun with --apply to execute the copies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
