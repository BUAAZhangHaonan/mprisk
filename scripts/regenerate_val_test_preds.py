#!/usr/bin/env python3
"""Regenerate val-aligned ``best_test_preds.pt`` from a val-selected checkpoint.

Background
----------
``canonical_rerun_v2`` T1 / T5 / PA-ablation runs originally wrote
``best_test_preds.pt`` keyed on ``test_balanced_accuracy_ac`` (per-sample
preds came from the epoch with the best *test* score). The migration script
``migrate_metrics_v2_to_v3.py`` patched the metrics JSON and the .pt
metadata to declare ``selection_metric = val_balanced_accuracy_ac`` but
left the per-sample preds pointing at the OLD test-selected epoch.

This script performs the actual regeneration: it loads the val-selected
``best_checkpoint.pt`` (already written keyed on val score), runs one
``_evaluate(return_preds=True)`` pass over ``official_test`` rows, and
writes a clean ``best_test_preds.pt`` whose preds/probs/labels truly
correspond to ``best_epoch``.

The schema written matches what ``train_trajectory_encoder`` would have
produced had it always keyed on val:

    {
        "epoch": <best_epoch>,
        "sample_ids": [...],
        "labels":     [...],
        "preds":      [...],
        "probs":      [...],
        "selection_metric": "val_balanced_accuracy_ac",
        "test_at_best_val_balanced_accuracy_ac": <float>,
        "test_at_best_val_ac_f1":                <float>,
        "test_at_best_val_ac_ap":                <float>,
    }

No ``migrated_from_test_keyed_v3`` field is written — the file is now
genuinely val-aligned.

Usage
-----
    python scripts/regenerate_val_test_preds.py --run-dir <RUN_DIR> [--dataset <path>] [--device cuda:0]

If ``--dataset`` is omitted, the dataset path is inferred from the
training config embedded in the checkpoint:

    outputs/v2/relation_data/<model_key>/<PROTO>/<proto>_main_p8_seed20260717/relation_dataset.jsonl

Correctness check
-----------------
After the eval pass, the script compares the computed
``balanced_accuracy_ac`` against ``test_at_best_val_balanced_accuracy_ac``
in ``train_metrics.json`` (if present). A mismatch > 1e-6 is reported but
the file is still written so the user can inspect it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from mprisk.representation.training import (  # noqa: E402
    TrainingConfig,
    _evaluate,
    _read_relation_rows,
    _resolve_device,
    _rows_to_sample_refs,
    _set_deterministic_seed,
    _validate_checkpoint_architecture,
    _validate_config,
)
from mprisk.representation.losses import (  # noqa: E402
    ModalitySplitRankingLoss,
    ProxyAnchorLoss,
)
from mprisk.representation.relation_models import (  # noqa: E402
    TME_PROXY_ANCHOR_V1,
    build_representation_model,
)


PROTO_DIR = {"vt": "VT", "va": "VA"}


def _infer_dataset_path(project_root: Path, config: TrainingConfig) -> Path:
    proto = config.protocol.lower()
    proto_dir = PROTO_DIR.get(proto)
    if proto_dir is None:
        raise ValueError(f"unsupported protocol {proto!r}")
    # All canonical_rerun_v2 runs used the seed20260717 prompt set
    # (prompt_set_key like '<proto>_main_p8_seed20260717'), so we hard-code
    # that stem here. prompt_set_artifact_sha256 is still validated by
    # _read_relation_rows.
    return (
        project_root
        / "outputs"
        / "v2"
        / "relation_data"
        / config.model_key
        / proto_dir
        / f"{proto}_main_p8_seed20260717"
        / "relation_dataset.jsonl"
    )


def _build_model_and_objective(
    checkpoint: dict[str, Any],
    config: TrainingConfig,
    *,
    device: torch.device,
):
    model = build_representation_model(
        config.repr_key,
        input_dim=int(checkpoint["model_config"]["input_dim"]),
        layer_count=int(checkpoint["model_config"]["layer_count"]),
        hidden_dim=config.hidden_dim,
        condition_dim=config.condition_dim,
        relation_dim=config.relation_dim,
        dropout=config.dropout,
        encoder_type=getattr(config, "encoder_type", "gru"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    objective: ProxyAnchorLoss | None = None
    d_objective: ModalitySplitRankingLoss | None = None
    if config.repr_key == TME_PROXY_ANCHOR_V1:
        objective = ProxyAnchorLoss(
            embed_dim=config.relation_dim,
            num_classes=2,
            alpha=config.proxy_alpha,
            margin=config.proxy_margin,
        ).to(device)
        proxy_state = checkpoint.get("proxy_state_dict")
        if proxy_state is not None:
            objective.load_state_dict(proxy_state)
        objective.eval()
        if config.enable_state_supervision:
            d_objective = ModalitySplitRankingLoss(
                d_margin=config.d_ranking_margin,
                angular_margin_rad=config.angular_ranking_margin_rad,
            ).to(device)
            d_objective.eval()
    return model, objective, d_objective


def _load_test_samples(
    dataset_path: Path,
    config: TrainingConfig,
    *,
    exclude_prefix: str | None,
):
    rows = _read_relation_rows(
        dataset_path,
        expected_model_key=config.model_key,
        expected_protocol=config.protocol,
        expected_prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
    )
    test_rows = [
        row
        for row in rows
        if row["representation_split"] == "official_test"
        and not (exclude_prefix and row["sample_id"].startswith(exclude_prefix))
    ]
    if not test_rows:
        raise RuntimeError(
            "no official_test rows found after exclude_prefix filter; "
            "cannot produce val-aligned test preds"
        )
    return _rows_to_sample_refs(test_rows)


def regenerate_run(
    run_dir: Path,
    *,
    dataset_override: Path | None,
    device_str: str,
    project_root: Path,
) -> dict[str, Any]:
    """Regenerate ``best_test_preds.pt`` for one run dir. Returns a report dict."""
    best_ckpt_path = run_dir / "best_checkpoint.pt"
    if not best_ckpt_path.is_file():
        raise FileNotFoundError(f"best_checkpoint.pt missing in {run_dir}")

    metrics_path = run_dir / "train_metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    _validate_checkpoint_architecture(checkpoint)
    if checkpoint.get("checkpoint_role") == "unconstrained_diagnostic":
        raise ValueError(
            f"{best_ckpt_path} is an unconstrained diagnostic checkpoint; "
            "cannot regenerate test preds from it"
        )

    # canonical_rerun_v2 training bug (fixed in commit 059f20e but existing
    # T5_lstm_ca_frozen checkpoints on disk still carry it): for state-
    # supervised PA runs, best_checkpoint.pt was sometimes overwritten by a
    # later "improved=True" save whose val_score was actually LOWER than the
    # already-recorded best. The symptom is ckpt.epoch != ckpt.best_epoch
    # (and, in some T5 runs, ckpt.best_epoch itself is stale relative to
    # train_metrics.json's best_epoch). The authoritative val-best epoch
    # lives in train_metrics.json["best_epoch"]; when we detect any
    # mismatch, fall back to unconstrained_best_checkpoint.pt (which IS at
    # the genuine val-best epoch, role=unconstrained_diagnostic).
    ckpt_epoch = int(checkpoint.get("epoch", -1))
    ckpt_best_epoch = int(checkpoint.get("best_epoch", -1))
    metrics_best_epoch = int(metrics.get("best_epoch", ckpt_best_epoch))
    used_fallback = False
    fallback_reason = ""
    unconstrained_path = run_dir / "unconstrained_best_checkpoint.pt"
    needs_fallback = ckpt_epoch != ckpt_best_epoch or ckpt_best_epoch != metrics_best_epoch
    if needs_fallback:
        if not unconstrained_path.is_file():
            raise ValueError(
                f"{best_ckpt_path} has epoch={ckpt_epoch}, "
                f"best_epoch={ckpt_best_epoch}, metrics.best_epoch="
                f"{metrics_best_epoch} — canonical_rerun_v2 training bug "
                f"detected, but unconstrained_best_checkpoint.pt is "
                f"missing; cannot recover the genuine val-best weights."
            )
        unconstrained = torch.load(
            unconstrained_path, map_location="cpu", weights_only=False
        )
        # Sanity: the unconstrained checkpoint should be at metrics.best_epoch.
        unc_epoch = int(unconstrained.get("epoch", -1))
        unc_best = int(unconstrained.get("best_epoch", -1))
        if unc_epoch != metrics_best_epoch or unc_best != metrics_best_epoch:
            raise ValueError(
                f"unconstrained_best_checkpoint.pt epoch={unc_epoch} "
                f"best_epoch={unc_best} does not match train_metrics.json "
                f"best_epoch={metrics_best_epoch}; refusing to use it."
            )
        # Swap in the unconstrained checkpoint as the source of truth.
        checkpoint = unconstrained
        used_fallback = True
        fallback_reason = (
            "best_checkpoint.pt had epoch != best_epoch or stale "
            f"best_epoch (ckpt epoch={ckpt_epoch}, best_epoch="
            f"{ckpt_best_epoch}, metrics.best_epoch={metrics_best_epoch}); "
            "used unconstrained_best_checkpoint.pt which is at the genuine "
            "val-best epoch."
        )

    config = TrainingConfig(**checkpoint["training_config"])
    _validate_config(config)

    # The original canonical_rerun_v2 drivers always passed
    # --exclude-prefix ch_sims_v2:; read it back from the metrics file if
    # present, else default to that value to stay symmetric with training.
    exclude_prefix = metrics.get("exclude_prefix", "ch_sims_v2:")
    # Some older metrics files might not carry the field; fall back to the
    # documented canonical_rerun_v2 default.
    if not exclude_prefix:
        exclude_prefix = "ch_sims_v2:"

    dataset_path = dataset_override or _infer_dataset_path(project_root, config)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"inferred dataset path does not exist: {dataset_path}")

    # Determinism: same scheme as _set_deterministic_seed (called at the
    # top of train_trajectory_encoder). We re-seed before inference so the
    # eval pass is reproducible.
    _set_deterministic_seed(config.seed)
    torch_device = _resolve_device(device_str)

    test_samples = _load_test_samples(
        dataset_path, config, exclude_prefix=exclude_prefix
    )

    model, objective, d_objective = _build_model_and_objective(
        checkpoint, config, device=torch_device
    )

    # No grad needed; _evaluate already wraps in torch.no_grad, but we add
    # one here for safety since we are not training.
    with torch.no_grad():
        (
            _test_loss,
            test_score,
            _test_state_separation,
            test_f1,
            test_ap,
            sample_ids_out,
            labels_out,
            preds_out,
            probs_out,
        ) = _evaluate(
            model,
            objective,
            d_objective,
            test_samples,
            config=config,
            class_weights=None,
            return_preds=True,
        )

    best_epoch = int(metrics.get("best_epoch", checkpoint.get("best_epoch", checkpoint.get("epoch", -1))))

    # Correctness check against train_metrics.json.
    expected_acc = metrics.get("test_at_best_val_balanced_accuracy_ac")
    expected_f1 = metrics.get("test_at_best_val_ac_f1")
    expected_ap = metrics.get("test_at_best_val_ac_ap")
    acc_delta: float | None = None
    f1_delta: float | None = None
    ap_delta: float | None = None
    if isinstance(expected_acc, (int, float)):
        acc_delta = float(test_score) - float(expected_acc)
    if isinstance(expected_f1, (int, float)):
        f1_delta = float(test_f1) - float(expected_f1)
    if isinstance(expected_ap, (int, float)):
        ap_delta = float(test_ap) - float(expected_ap)

    # Read OLD values from the existing best_test_preds.pt for the report.
    btp_path = run_dir / "best_test_preds.pt"
    old_payload: dict[str, Any] = {}
    if btp_path.is_file():
        try:
            old_payload = torch.load(btp_path, map_location="cpu", weights_only=False)
        except Exception:
            old_payload = {}
    old_acc = old_payload.get("test_balanced_accuracy_ac")
    old_epoch = old_payload.get("epoch")
    # After migration, the .pt also carries test_at_best_val_*; prefer that
    # over the test-selected number for the "old" column in the report so
    # the delta column answers the right question (new vs target).
    old_target_acc = old_payload.get("test_at_best_val_balanced_accuracy_ac", expected_acc)

    # _evaluate returns sample_ids/labels/preds as Python lists, but
    # conflict_scores (probs_out) is a numpy array. Coerce everything to
    # plain Python scalars so torch.save produces a clean, portable file.
    def _to_int_list(values):
        if values is None:
            return []
        return [int(v) for v in values]

    def _to_float_list(values):
        if values is None:
            return []
        return [float(v) for v in values]

    def _to_str_list(values):
        if values is None:
            return []
        return [str(v) for v in values]

    # Write the new file (no migrated_from_test_keyed_v3 field — genuine).
    new_payload = {
        "epoch": int(best_epoch),
        "sample_ids": _to_str_list(sample_ids_out),
        "labels": _to_int_list(labels_out),
        "preds": _to_int_list(preds_out),
        "probs": _to_float_list(probs_out),
        "selection_metric": "val_balanced_accuracy_ac",
        "test_at_best_val_balanced_accuracy_ac": float(test_score),
        "test_at_best_val_ac_f1": float(test_f1),
        "test_at_best_val_ac_ap": float(test_ap),
    }
    temporary = btp_path.with_suffix(btp_path.suffix + ".tmp")
    torch.save(new_payload, temporary)
    temporary.replace(btp_path)

    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "old_epoch": old_epoch,
        "old_test_balanced_accuracy_ac": old_acc,
        "old_target_test_at_best_val_acc": old_target_acc,
        "new_test_at_best_val_balanced_accuracy_ac": float(test_score),
        "new_test_at_best_val_ac_f1": float(test_f1),
        "new_test_at_best_val_ac_ap": float(test_ap),
        "expected_test_at_best_val_balanced_accuracy_ac": expected_acc,
        "acc_delta_vs_expected": acc_delta,
        "f1_delta_vs_expected": f1_delta,
        "ap_delta_vs_expected": ap_delta,
        "sample_count": len(sample_ids_out) if sample_ids_out is not None else 0,
        "dataset": str(dataset_path),
        "device": str(torch_device),
        "used_unconstrained_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "ckpt_epoch_seen": ckpt_epoch,
        "ckpt_best_epoch_seen": ckpt_best_epoch,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing best_checkpoint.pt and train_metrics.json",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override relation_dataset.jsonl path. Inferred from the checkpoint if absent.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="torch device (the original runs used cuda:0 with CUDA_VISIBLE_DEVICES set)",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root for dataset inference (defaults to two levels above this script)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[1]
    )
    dataset_override = Path(args.dataset).resolve() if args.dataset else None

    t0 = time.time()
    try:
        report = regenerate_run(
            run_dir,
            dataset_override=dataset_override,
            device_str=args.device,
            project_root=project_root,
        )
    except Exception as exc:
        print(f"[FAIL] {run_dir}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    elapsed = time.time() - t0
    print(
        f"[OK] {run_dir.name}: "
        f"epoch={report['best_epoch']} "
        f"new_acc={report['new_test_at_best_val_balanced_accuracy_ac']:.6f} "
        f"expected={report['expected_test_at_best_val_balanced_accuracy_ac']} "
        f"delta={report['acc_delta_vs_expected']} "
        f"n={report['sample_count']} "
        f"({elapsed:.1f}s)"
    )
    # Emit the full JSON report on stdout for the batch driver to collect.
    print("REPORT_JSON " + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
