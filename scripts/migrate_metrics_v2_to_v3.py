#!/usr/bin/env python3
"""Migrate metrics files from test-keyed (v3-deprecated) to val-keyed selection.

For each run directory under <root>:
  - Read metrics.json or train_metrics.json
  - If selection_metric == "test_balanced_accuracy_ac" (old test-keyed):
      - Find best_epoch (already val-selected in the current code)
      - Read train_log.jsonl (or train_metrics.jsonl), find the best_epoch row
      - Write back with selection_metric -> "val_balanced_accuracy_ac" and
        test_at_best_val_{balanced_accuracy_ac,ac_f1,ac_ap} populated from that row
      - Mark best_test_preds.pt as migrated (per-sample preds are still test-keyed;
        downstream must rerun inference to get val-aligned preds if needed)
  - Otherwise: skip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


METRICS_CANDIDATES = ("train_metrics.json", "metrics.json")
LOG_CANDIDATES = ("train_log.jsonl", "train_metrics.jsonl")


def _find_run_metrics(run_dir: Path) -> Path | None:
    for name in METRICS_CANDIDATES:
        p = run_dir / name
        if p.is_file():
            return p
    return None


def _find_run_log(run_dir: Path) -> Path | None:
    for name in LOG_CANDIDATES:
        p = run_dir / name
        if p.is_file():
            return p
    return None


def _find_best_epoch_row(log_path: Path, best_epoch: int) -> dict | None:
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("epoch") == best_epoch:
                return row
    return None


def migrate_one(run_dir: Path, dry_run: bool = False) -> dict:
    metrics_path = _find_run_metrics(run_dir)
    if metrics_path is None:
        return {"status": "no metrics file"}

    try:
        metrics = json.loads(metrics_path.read_text())
    except json.JSONDecodeError as exc:
        return {"status": "metrics parse error", "error": str(exc)}

    selection = metrics.get("selection_metric")
    if selection is None:
        return {"status": "no selection_metric field"}
    if selection != "test_balanced_accuracy_ac":
        return {"status": "already val-keyed", "selection_metric": selection}

    best_epoch = metrics.get("best_epoch")
    if best_epoch is None:
        return {"status": "no best_epoch"}

    log_path = _find_run_log(run_dir)
    if log_path is None:
        return {"status": "no log file"}

    best_row = _find_best_epoch_row(log_path, int(best_epoch))
    if best_row is None:
        return {"status": "best_epoch row not in log", "best_epoch": best_epoch}

    new_test_acc = best_row.get("test_balanced_accuracy_ac")
    new_test_f1 = best_row.get("test_ac_f1")
    new_test_ap = best_row.get("test_ac_ap")
    if new_test_acc is None:
        return {"status": "best_epoch row missing test_balanced_accuracy_ac"}

    old_test_acc = metrics.get("best_test_balanced_accuracy_ac")
    delta = None
    if isinstance(old_test_acc, (int, float)) and isinstance(new_test_acc, (int, float)):
        delta = float(new_test_acc) - float(old_test_acc)

    metrics["selection_metric"] = "val_balanced_accuracy_ac"
    metrics["test_at_best_val_balanced_accuracy_ac"] = float(new_test_acc)
    metrics["test_at_best_val_ac_f1"] = float(new_test_f1) if new_test_f1 is not None else None
    metrics["test_at_best_val_ac_ap"] = float(new_test_ap) if new_test_ap is not None else None
    metrics["migrated_from_test_keyed_v3"] = True

    btp = run_dir / "best_test_preds.pt"
    btp_note = "absent"
    if btp.is_file():
        btp_note = "tagged"
        if not dry_run:
            try:
                import torch

                payload = torch.load(btp, map_location="cpu")
                payload["selection_metric"] = "val_balanced_accuracy_ac"
                payload["test_at_best_val_balanced_accuracy_ac"] = float(new_test_acc)
                payload["test_at_best_val_ac_f1"] = float(new_test_f1) if new_test_f1 is not None else None
                payload["test_at_best_val_ac_ap"] = float(new_test_ap) if new_test_ap is not None else None
                payload["migrated_from_test_keyed_v3"] = True
                payload["migrated_note"] = (
                    "Per-sample preds still correspond to the OLD test-selected epoch. "
                    "To get val-aligned preds, delete this file and rerun inference "
                    "using best_checkpoint.pt."
                )
                torch.save(payload, btp)
            except Exception as exc:
                btp_note = f"tag-failed: {exc}"

    if not dry_run:
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )

    return {
        "status": "migrated" if not dry_run else "would-migrate",
        # M-A1-R5-4: previously this called .relative_to(parents[-1]) which
        # produced a path relative to the filesystem root and was meaningless.
        # Use the absolute run_dir so downstream tooling can actually find it.
        "run_dir": str(run_dir),
        "best_epoch": int(best_epoch),
        "old_best_test_acc": old_test_acc,
        "new_test_at_best_val_acc": float(new_test_acc),
        "delta_acc": delta,
        "best_test_preds": btp_note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="outputs/canonical_rerun_v2",
        help="Root directory containing run subdirectories.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"root not found: {root}", file=sys.stderr)
        return 1

    migrated = 0
    skipped = 0
    failed = 0
    abs_deltas: list[float] = []
    for mp in sorted(root.rglob("train_metrics.json")):
        result = migrate_one(mp.parent, dry_run=args.dry_run)
        status = result.get("status", "")
        if status in ("migrated", "would-migrate"):
            migrated += 1
            delta = result.get("delta_acc")
            if isinstance(delta, float):
                abs_deltas.append(abs(delta))
            print(
                f"[{status}] {mp.parent}: "
                f"epoch={result.get('best_epoch')} "
                f"old={result.get('old_best_test_acc')} "
                f"new={result.get('new_test_at_best_val_acc')} "
                f"delta={delta:+.4f} "
                f"btp={result.get('best_test_preds')}"
            )
        elif status == "already val-keyed":
            skipped += 1
        else:
            failed += 1
            print(f"[skip] {mp.parent}: {status}", file=sys.stderr)

    # Also scan for metrics.json (non-train runs)
    for mp in sorted(root.rglob("metrics.json")):
        if (mp.parent / "train_metrics.json").exists():
            continue
        result = migrate_one(mp.parent, dry_run=args.dry_run)
        status = result.get("status", "")
        if status in ("migrated", "would-migrate"):
            migrated += 1
            delta = result.get("delta_acc")
            if isinstance(delta, float):
                abs_deltas.append(abs(delta))
            print(
                f"[{status}] {mp.parent}: "
                f"epoch={result.get('best_epoch')} "
                f"old={result.get('old_best_test_acc')} "
                f"new={result.get('new_test_at_best_val_acc')} "
                f"delta={delta:+.4f}"
            )
        elif status == "already val-keyed":
            skipped += 1
        else:
            failed += 1

    print()
    print(f"Summary: {migrated} migrated, {skipped} already-val-keyed, {failed} skipped")
    if abs_deltas:
        avg_delta = sum(abs_deltas) / len(abs_deltas)
        max_delta = max(abs_deltas)
        print(f"|delta| avg={avg_delta:.4f} max={max_delta:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
