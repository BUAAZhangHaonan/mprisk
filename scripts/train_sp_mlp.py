#!/usr/bin/env python3
"""canonical_rerun SP-MLP trainer (two-stage).

Stage 1 (C/A pretrain):
    MLP encoder (4096 -> 128, ~524k params) + temp head (128 -> 2)
    trained on Conflict/Aligned labels using M12 LAST LAYER hidden state
    ([B, 4096]) drawn from the prefill cache.
    Plain Adam(lr=1e-3) + plain CE + clip 1.0 + 100 epochs.
    best.pt = val_balanced_acc highest epoch (val-selected).

Stage 2 (M/N head on frozen encoder):
    Fresh 2-layer head (128 -> 2) trained on M/N labels using the frozen
    encoder embedding.
    Plain Adam(lr=1e-3) + plain CE + clip 1.0 + 100 epochs.
    best.pt = val_balanced_acc highest epoch (val-selected).

Input feature: M12 LAST LAYER hidden state from the prefill cache
(``m12_trajectory[-1]``, shape ``[4096]``). This differs from
``sp_ca_baseline.py`` which reads ``penultimate_feature`` from a frozen
export dir.

Output (per stage):
    stage1: outputs/.../<run>/<MODEL>_seed<SEED>/{encoder.pt, pretrain_metrics.json}
    stage2: outputs/.../<run>/<MODEL>_seed<SEED>/{mn_head.pt, mn_metrics.json, embeddings.npz}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn as nn

HERE = Path(__file__).resolve().parent
PROJ_ROOT = HERE.parent.parent
SRC = PROJ_ROOT / "src"
for _p in (str(SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from mprisk.cache.prefill_extract import extract_t0_trajectory  # noqa: E402
from _trainer_lib import (  # noqa: E402  # P5-B/P7-C shared helpers
    _load_prompt_ids, _scan_cache, _load_split_assignment,
    _load_sample_type_map, _load_misread_labels, _domain_of,
    _train_classifier,  # P7-C: identical loop, was duplicated
)


# ---------------------------------------------------------------------------
# Data loading: M12 last layer [4096]
# ---------------------------------------------------------------------------


def _build_m12_last_layer(info, prompt_id):
    """Return [H] = m12 trajectory's last layer hidden state."""
    entry = info["M12"].get(prompt_id)
    if entry is None:
        return None
    try:
        traj = extract_t0_trajectory(entry)  # [L, H]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] t0 extraction failed: {exc}", file=sys.stderr)
        return None
    return traj[-1].astype(np.float32)  # [H]


# ---------------------------------------------------------------------------
# Stage-1: build C/A rows from M12 last layer
# ---------------------------------------------------------------------------


def _build_ca_rows_m12last(
    *,
    cache_index,
    prompt_ids,
    sample_types,
    split_of,
    domain_gen_only: bool = True,
):
    """Build C/A rows for Stage 1 encoder pretraining.

    Each row is one (sid, pid) pair with feature [H] (M12 last layer)
    and label {Conflict:1, Aligned:0}.
    """
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for sid, stype in sample_types.items():
        if stype not in ("Conflict", "Aligned"):
            continue
        split = split_of.get(sid, "")
        if split not in ("relation_train", "relation_val", "official_test"):
            continue
        if domain_gen_only and _domain_of(sid) != "gen":
            continue
        info = cache_index.get(sid)
        if info is None:
            continue
        label = 1 if stype == "Conflict" else 0
        for pid in prompt_ids:
            key = (sid, pid)
            if key in seen:
                continue
            arr = _build_m12_last_layer(info, pid)
            if arr is None:
                continue
            seen.add(key)
            rows.append({
                "sample_id": sid,
                "prompt_id": pid,
                "feat": arr,
                "label": label,
                "split": split,
            })
    return rows


def _stack_ca(rows, split_name, device):
    sub = [r for r in rows if r["split"] == split_name]
    if not sub:
        return torch.empty(0, 0, device=device, dtype=torch.float32), torch.empty(0, dtype=torch.long), []
    X = torch.from_numpy(np.stack([r["feat"] for r in sub], axis=0)).to(device)
    y = torch.tensor([r["label"] for r in sub], dtype=torch.long)
    sids = [r["sample_id"] for r in sub]
    return X, y, sids


# ---------------------------------------------------------------------------
# Stage-2: build M/N rows from M12 last layer (encoder frozen)
# ---------------------------------------------------------------------------


def _build_mn_rows_m12last(
    *,
    cache_index,
    prompt_ids,
    misread_labels,
    sample_types,
    split_of,
):
    """Build M/N rows for Stage 2 head training (Conflict only).

    Each row is one (sid, pid) pair with feature [H] (M12 last layer)
    and label {MISREAD:1, NON_MISREAD:0}.
    """
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for sid, label_str in misread_labels.items():
        if label_str not in ("MISREAD", "NON_MISREAD"):
            continue
        if _domain_of(sid) != "gen":
            continue
        stype = sample_types.get(sid)
        if stype != "Conflict":
            continue
        split = split_of.get(sid, "")
        if split not in ("relation_train", "relation_val", "official_test"):
            continue
        info = cache_index.get(sid)
        if info is None:
            continue
        label = 1 if label_str == "MISREAD" else 0
        for pid in prompt_ids:
            key = (sid, pid)
            if key in seen:
                continue
            arr = _build_m12_last_layer(info, pid)
            if arr is None:
                continue
            seen.add(key)
            rows.append({
                "sample_id": sid,
                "prompt_id": pid,
                "feat": arr,
                "label": label,
                "split": split,
            })
    return rows


def _stack_mn(rows, split_name, device):
    sub = [r for r in rows if r["split"] == split_name]
    if not sub:
        return torch.empty(0, 0, device=device, dtype=torch.float32), torch.empty(0, dtype=torch.long), []
    X = torch.from_numpy(np.stack([r["feat"] for r in sub], axis=0)).to(device)
    y = torch.tensor([r["label"] for r in sub], dtype=torch.long)
    sids = [r["sample_id"] for r in sub]
    return X, y, sids


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SPMLPEncoderV2(nn.Module):
    """M12 last layer (4096) -> Linear(4096, 128) -> GELU. ~524k params."""

    def __init__(self, in_dim: int = 4096, embed_dim: int = 128) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.fc = nn.Linear(self.in_dim, self.embed_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


class SPMLPStage1Wrapper(nn.Module):
    """Encoder + temp head (Linear 128 -> 2) for Stage-1 C/A pretraining."""

    def __init__(self, encoder: SPMLPEncoderV2) -> None:
        super().__init__()
        self.encoder = encoder
        self.temp_head = nn.Linear(encoder.embed_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temp_head(self.encoder(x))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Stage 1 runner
# ---------------------------------------------------------------------------


def run_stage1(args, *, cache_index, sample_types, split_of, device, out_dir):
    """Train MLP(4096, 128) encoder on C/A labels. Save encoder.pt."""
    t0 = time.time()
    prompt_ids = _load_prompt_ids(Path(args.prompt_set))
    rows = _build_ca_rows_m12last(
        cache_index=cache_index,
        prompt_ids=prompt_ids,
        sample_types=sample_types,
        split_of=split_of,
        domain_gen_only=True,
    )
    n_tr = sum(1 for r in rows if r["split"] == "relation_train")
    n_va = sum(1 for r in rows if r["split"] == "relation_val")
    n_te = sum(1 for r in rows if r["split"] == "official_test")
    print(f"[stage1] rows: train={n_tr} val={n_va} test={n_te}", file=sys.stderr, flush=True)
    if n_tr == 0 or n_te == 0:
        raise RuntimeError(f"empty stage1 split: train={n_tr} test={n_te}")

    splits = {
        "train": _stack_ca(rows, "relation_train", device),
        "val": _stack_ca(rows, "relation_val", device),
        "test": _stack_ca(rows, "official_test", device),
    }
    X_tr, y_tr, _ = splits["train"]
    X_va, y_va, _ = splits["val"]
    X_te, y_te, _ = splits["test"]
    print(
        f"[stage1] tensor shapes: train={tuple(X_tr.shape)} val={tuple(X_va.shape)} test={tuple(X_te.shape)}",
        file=sys.stderr, flush=True,
    )

    in_dim = int(X_tr.shape[-1])
    encoder = SPMLPEncoderV2(in_dim=in_dim, embed_dim=args.embed_dim)
    model = SPMLPStage1Wrapper(encoder)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stage1] SP-MLP v2 trainable params = {n_params:,}", file=sys.stderr, flush=True)

    best_metrics, best_state, history = _train_classifier(
        model, X_tr, y_tr, X_va, y_va, X_te, y_te,
        device=device,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        select_metric="val_balanced_acc",
    )

    # Save frozen encoder weights from best_state.
    enc_state = {
        k[len("encoder."):]: v for k, v in best_state.items() if k.startswith("encoder.")
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": enc_state,
            "architecture_version": "sp_mlp_v2_encoder",
            "in_dim": int(in_dim),
            "embed_dim": int(args.embed_dim),
            "stage": "ca_pretrain",
            "best_epoch": int(best_metrics["best_epoch"]),
        },
        out_dir / "encoder.pt",
    )
    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": history}, f, indent=2, sort_keys=True)

    metrics = {
        "experiment": "sp_mlp_v2_stage1_ca",
        "model_key": args.model_key,
        "method": "sp_mlp_v2",
        "stage": "ca_pretrain",
        "lr": float(args.lr),
        "weight_decay": 0.0,
        "max_epochs": int(args.max_epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "n_trainable_params": int(n_params),
        "best_epoch": int(best_metrics["best_epoch"]),
        "best_test_ac_acc": float(best_metrics["test_balanced_acc"]),
        "best_test_ac_f1": float(best_metrics["test_macro_f1"]),
        "best_test_ac_ap": float(best_metrics["test_ap"]),
        "best_metrics": best_metrics,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(out_dir / "pretrain_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(
        f"[stage1] DONE best_test_ac_acc={metrics['best_test_ac_acc']:.4f} "
        f"best_epoch={metrics['best_epoch']} elapsed={metrics['elapsed_seconds']}s",
        file=sys.stderr, flush=True,
    )
    return metrics


# ---------------------------------------------------------------------------
# Stage 2 runner (M/N head on frozen encoder)
# ---------------------------------------------------------------------------


def run_stage2(args, *, cache_index, sample_types, split_of, misread_labels, device, out_dir):
    """Train MLP(128, 2) head on M/N labels using frozen encoder."""
    t0 = time.time()
    enc_ckpt_path = Path(args.encoder_checkpoint)
    if not enc_ckpt_path.exists():
        raise FileNotFoundError(f"encoder ckpt missing: {enc_ckpt_path}")
    enc_ckpt = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
    enc_state = enc_ckpt.get("encoder_state_dict", enc_ckpt)
    in_dim = int(enc_ckpt.get("in_dim"))
    embed_dim = int(enc_ckpt.get("embed_dim"))
    encoder = SPMLPEncoderV2(in_dim=in_dim, embed_dim=embed_dim)
    encoder.load_state_dict(enc_state, strict=True)
    encoder.to(device).eval()
    for p_ in encoder.parameters():
        p_.requires_grad_(False)
    print(
        f"[stage2] loaded encoder: in_dim={in_dim} embed_dim={embed_dim} "
        f"best_epoch={enc_ckpt.get('best_epoch')}",
        file=sys.stderr, flush=True,
    )

    prompt_ids = _load_prompt_ids(Path(args.prompt_set))
    rows = _build_mn_rows_m12last(
        cache_index=cache_index,
        prompt_ids=prompt_ids,
        misread_labels=misread_labels,
        sample_types=sample_types,
        split_of=split_of,
    )
    n_tr = sum(1 for r in rows if r["split"] == "relation_train")
    n_va = sum(1 for r in rows if r["split"] == "relation_val")
    n_te = sum(1 for r in rows if r["split"] == "official_test")
    print(f"[stage2] rows: train={n_tr} val={n_va} test={n_te}", file=sys.stderr, flush=True)
    if n_tr == 0 or n_te == 0:
        raise RuntimeError(f"empty stage2 split: train={n_tr} test={n_te}")

    splits = {
        "train": _stack_mn(rows, "relation_train", device),
        "val": _stack_mn(rows, "relation_val", device),
        "test": _stack_mn(rows, "official_test", device),
    }
    X_tr_raw, y_tr, sid_tr = splits["train"]
    X_va_raw, y_va, sid_va = splits["val"]
    X_te_raw, y_te, sid_te = splits["test"]

    # Encode with frozen encoder.
    with torch.no_grad():
        Z_tr = encoder(X_tr_raw).detach()
        Z_va = encoder(X_va_raw).detach()
        Z_te = encoder(X_te_raw).detach()
    print(
        f"[stage2] encoded shapes: train={tuple(Z_tr.shape)} val={tuple(Z_va.shape)} test={tuple(Z_te.shape)}",
        file=sys.stderr, flush=True,
    )

    head = nn.Linear(embed_dim, 2)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[stage2] head trainable params = {n_params:,}", file=sys.stderr, flush=True)

    best_metrics, best_state, history = _train_classifier(
        head, Z_tr, y_tr, Z_va, y_va, Z_te, y_te,
        device=device,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        select_metric="val_balanced_acc",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "mn_head.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": history}, f, indent=2, sort_keys=True)

    # Save embeddings for downstream analysis.
    np.savez(
        out_dir / "embeddings.npz",
        train_ids=np.array(sid_tr, dtype=object),
        val_ids=np.array(sid_va, dtype=object),
        test_ids=np.array(sid_te, dtype=object),
        train_Z=Z_tr.cpu().numpy(),
        val_Z=Z_va.cpu().numpy(),
        test_Z=Z_te.cpu().numpy(),
        train_y=y_tr.cpu().numpy(),
        val_y=y_va.cpu().numpy(),
        test_y=y_te.cpu().numpy(),
    )

    pc = best_metrics["per_class_acc"]
    per_class_mn = {
        "MISREAD": pc.get("class_1", 0.0),
        "NON_MISREAD": pc.get("class_0", 0.0),
    }
    metrics = {
        "experiment": "sp_mlp_v2_stage2_mn",
        "model_key": args.model_key,
        "method": "sp_mlp_v2",
        "stage": "mn_head_frozen_encoder",
        "encoder_checkpoint": str(enc_ckpt_path),
        "lr": float(args.lr),
        "weight_decay": 0.0,
        "max_epochs": int(args.max_epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "n_trainable_params": int(n_params),
        "best_epoch": int(best_metrics["best_epoch"]),
        "best_test_mn_acc": float(best_metrics["test_balanced_acc"]),
        "best_test_mn_f1": float(best_metrics["test_macro_f1"]),
        "best_test_mn_ap": float(best_metrics["test_ap"]),
        "per_class_acc": per_class_mn,
        "best_metrics": best_metrics,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(out_dir / "mn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(
        f"[stage2] DONE best_test_mn_acc={metrics['best_test_mn_acc']:.4f} "
        f"best_epoch={metrics['best_epoch']} elapsed={metrics['elapsed_seconds']}s",
        file=sys.stderr, flush=True,
    )
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():
    p = argparse.ArgumentParser(
        description="SP-MLP v2 (two-stage): MLP(4096, 128) C/A pretrain + MLP(128, 2) M/N head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", required=True, choices=("pretrain", "mn_head"))
    p.add_argument("--model-key", required=True)
    p.add_argument("--split-assignment", required=True)
    p.add_argument("--misread-judgments", help="required for stage=mn_head")
    p.add_argument("--cache-roots", nargs="+", required=True)
    p.add_argument("--prompt-set", required=True)
    p.add_argument("--main-manifest", required=True)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260717)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--encoder-checkpoint", help="required for stage=mn_head")
    p.add_argument("--output-dir", required=True)
    return p


def main():
    args = _build_parser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_ids = _load_prompt_ids(Path(args.prompt_set))
    cache_index = _scan_cache(
        [Path(p) for p in args.cache_roots],
        model_key=args.model_key,
        prompt_ids=prompt_ids,
    )
    print(f"[setup] cache covers {len(cache_index)} samples", file=sys.stderr, flush=True)
    proto = "va" if args.model_key in ("qwen2_5_omni_7b", "gemma4_12b", "gemma4_12b_it", "phi4_multimodal") else "vt"
    sample_types = _load_sample_type_map(Path(args.main_manifest), protocol=proto)
    split_of = _load_split_assignment(Path(args.split_assignment))

    if args.stage == "pretrain":
        run_stage1(
            args,
            cache_index=cache_index,
            sample_types=sample_types,
            split_of=split_of,
            device=device,
            out_dir=out_dir,
        )
    elif args.stage == "mn_head":
        if not args.misread_judgments:
            print("--misread-judgments required for stage=mn_head", file=sys.stderr)
            return 2
        if not args.encoder_checkpoint:
            print("--encoder-checkpoint required for stage=mn_head", file=sys.stderr)
            return 2
        misread_labels = _load_misread_labels(Path(args.misread_judgments))
        run_stage2(
            args,
            cache_index=cache_index,
            sample_types=sample_types,
            split_of=split_of,
            misread_labels=misread_labels,
            device=device,
            out_dir=out_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
