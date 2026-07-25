#!/usr/bin/env python3
"""canonical_rerun_v2 T-LSTM v2 trainer (two-stage).

Stage 1 (C/A pretrain):
    LSTM encoder (input=4096, hidden=128, num_layers=2, ~4.3M params) +
    temp head (128 -> 2) trained on Conflict/Aligned labels using the M12
    trajectory [B, 36, 4096] drawn from the prefill cache.
    Layer-L2 normalize before LSTM (matches TME encoder behavior).
    Plain Adam(lr=1e-3) + plain CE + clip 1.0 + 100 epochs.
    best.pt = val_balanced_acc highest epoch (val-selected).

Stage 2 (M/N head on frozen encoder):
    Fresh head Linear(128, 2) trained on M/N labels using the frozen
    LSTM encoder's last-layer final hidden state.
    Plain Adam(lr=1e-3) + plain CE + clip 1.0 + 100 epochs.
    best.pt = val_balanced_acc highest epoch (val-selected).

Output (per stage):
    stage1: outputs/.../<run>/<MODEL>_seed<SEED>/{encoder.pt, pretrain_metrics.json}
    stage2: outputs/.../<run>/<MODEL>_seed<SEED>/{mn_head.pt, mn_metrics.json, embeddings.npz}
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
PROJ_ROOT = HERE.parent.parent
V2_SRC = PROJ_ROOT / "src"
for _p in (str(V2_SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from _seed import set_deterministic_seed  # noqa: E402
from mprisk.cache.hidden_state_cache import normalize_protocol  # noqa: E402
from mprisk.cache.prefill_extract import extract_t0_trajectory  # noqa: E402
from mprisk.data.manifests import read_final_manifest  # noqa: E402
from mprisk.representation.relation_models import strict_l2_normalize  # noqa: E402
from _trainer_lib import (  # noqa: E402  # P5-B shared helpers
    CONDITIONS, COND_IDX,
    _load_prompt_ids, _scan_cache, _load_split_assignment,
    _load_sample_type_map, _load_misread_labels, _domain_of,
    _balanced_per_class_acc, _eval_loss,
)


# ---------------------------------------------------------------------------
# Data loading: M12 trajectory [L, H]
# ---------------------------------------------------------------------------


def _build_m12_traj(info, prompt_id):
    """Return [L, H] = M12 trajectory for one (sid, pid)."""
    entry = info["M12"].get(prompt_id)
    if entry is None:
        return None
    try:
        traj = extract_t0_trajectory(entry)  # [L, H]
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] t0 extraction failed: {exc}", file=sys.stderr)
        return None
    return traj.astype(np.float32)


# ---------------------------------------------------------------------------
# Row builders (shared between stage1 and stage2)
# ---------------------------------------------------------------------------


def _build_ca_rows_m12traj(
    *,
    cache_index,
    prompt_ids,
    sample_types,
    split_of,
    domain_gen_only: bool = True,
):
    """C/A rows from M12 trajectory. label {Conflict:1, Aligned:0}."""
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
            arr = _build_m12_traj(info, pid)
            if arr is None:
                continue
            seen.add(key)
            rows.append({
                "sample_id": sid,
                "prompt_id": pid,
                "traj": arr,
                "label": label,
                "split": split,
            })
    return rows


def _build_mn_rows_m12traj(
    *,
    cache_index,
    prompt_ids,
    misread_labels,
    sample_types,
    split_of,
):
    """M/N rows from M12 trajectory. Conflict-only. label {MISREAD:1, NON_MISREAD:0}."""
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
            arr = _build_m12_traj(info, pid)
            if arr is None:
                continue
            seen.add(key)
            rows.append({
                "sample_id": sid,
                "prompt_id": pid,
                "traj": arr,
                "label": label,
                "split": split,
            })
    return rows


def _stack_rows(rows, split_name, device):
    sub = [r for r in rows if r["split"] == split_name]
    if not sub:
        return torch.empty(0, 0, 0, device=device, dtype=torch.float32), torch.empty(0, dtype=torch.long), []
    X = torch.from_numpy(np.stack([r["traj"] for r in sub], axis=0)).to(device)
    y = torch.tensor([r["label"] for r in sub], dtype=torch.long)
    sids = [r["sample_id"] for r in sub]
    return X, y, sids


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TLSTMEncoderV2(nn.Module):
    """M12 trajectory [B, L, 4096] -> Layer-L2 -> LSTM(4096, 128, 2 layers)
    -> last layer hidden state [B, 128]. ~4.3M params.
    """

    architecture_version = "t_lstm_v2_encoder"

    def __init__(
        self,
        *,
        input_dim: int,
        embed_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 3:
            raise ValueError(
                f"TLSTMEncoderV2 expects [B, L, H], got {tuple(trajectory.shape)}"
            )
        traj_norm = strict_l2_normalize(
            trajectory,
            stage="t_lstm_v2_layer_input",
        )
        _, (h_n, _) = self.lstm(traj_norm)
        return h_n[-1]  # [B, embed_dim] last layer final hidden


class TLSTMStage1Wrapper(nn.Module):
    """Encoder + temp head Linear(128, 2) for Stage-1 C/A pretraining."""

    def __init__(self, encoder: TLSTMEncoderV2) -> None:
        super().__init__()
        self.encoder = encoder
        self.temp_head = nn.Linear(encoder.embed_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temp_head(self.encoder(x))


# ---------------------------------------------------------------------------
# Training helpers (shared with SP-MLP v2 logic)
# ---------------------------------------------------------------------------


def _eval_classifier(model, X, y, *, batch_size, device):
    n = X.shape[0]
    if n == 0:
        return None
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X[i:i + batch_size]
            logits = model(xb)
            p = F.softmax(logits, dim=-1)[:, 1]
            probs.append(p.detach().cpu().numpy())
    probs = np.concatenate(probs)
    y_np = y.cpu().numpy()
    preds = (probs >= 0.5).astype(np.int64)
    bal, pc = _balanced_per_class_acc(preds, y_np)
    return {
        "accuracy": float(accuracy_score(y_np, preds)),
        "balanced_acc": bal,
        "macro_f1": float(f1_score(y_np, preds, zero_division=0)),
        "ap": (
            float(average_precision_score(y_np, probs))
            if len(set(y_np.tolist())) > 1 else 0.0
        ),
        "roc_auc": (
            float(roc_auc_score(y_np, probs))
            if len(set(y_np.tolist())) > 1 else 0.0
        ),
        "per_class_acc": pc,
        "probs": probs,
        "preds": preds,
    }


def _train_classifier(
    model,
    X_tr, y_tr,
    X_va, y_va,
    X_te, y_te,
    *,
    device,
    max_epochs,
    batch_size,
    lr,
    seed,
    select_metric: str = "val_balanced_acc",
):
    set_deterministic_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model.to(device)
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=min(batch_size, max(1, X_tr.shape[0])),
        shuffle=True,
        generator=g,
    )

    best_score = -1.0
    best_metrics = None
    best_state = None
    last_state = None
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optim.step()

        train_loss = _eval_loss(model, X_tr, y_tr, batch_size=batch_size)
        val_loss = _eval_loss(model, X_va, y_va, batch_size=batch_size)
        val_m = _eval_classifier(model, X_va, y_va, batch_size=batch_size, device=device) or {}
        te_m = _eval_classifier(model, X_te, y_te, batch_size=batch_size, device=device) or {}

        # M-A1-R5-2: single-code-path. The selection metric is ALWAYS
        # val_balanced_acc; selecting on test leaks the test set.
        assert select_metric == "val_balanced_acc", (
            f"select_metric must be 'val_balanced_acc' (got {select_metric!r})"
        )
        val_score = float(val_m.get("balanced_acc", 0.0) or 0.0)
        score = val_score

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_balanced_acc": val_m.get("balanced_acc"),
            "val_macro_f1": val_m.get("macro_f1"),
            "val_ap": val_m.get("ap"),
            "val_roc_auc": val_m.get("roc_auc"),
            "test_balanced_acc": te_m.get("balanced_acc"),
            "test_macro_f1": te_m.get("macro_f1"),
            "test_ap": te_m.get("ap"),
            "test_roc_auc": te_m.get("roc_auc"),
        }
        history.append(epoch_record)
        print(
            f"[ep{epoch:03d}] loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_bal={val_m.get('balanced_acc', 0):.4f} "
            f"test_bal={te_m.get('balanced_acc', 0):.4f} "
            f"test_ap={te_m.get('ap', 0):.4f}",
            file=sys.stderr, flush=True,
        )

        last_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        if score > best_score:
            best_score = score
            best_metrics = {
                "epoch": epoch,
                "selection_metric": select_metric,
                "val_balanced_acc": val_m.get("balanced_acc", 0.0),
                "val_macro_f1": val_m.get("macro_f1", 0.0),
                "val_ap": val_m.get("ap", 0.0),
                "val_roc_auc": val_m.get("roc_auc", 0.0),
                "test_balanced_acc": te_m.get("balanced_acc", 0.0),
                "test_macro_f1": te_m.get("macro_f1", 0.0),
                "test_ap": te_m.get("ap", 0.0),
                "test_roc_auc": te_m.get("roc_auc", 0.0),
                "per_class_acc": te_m.get("per_class_acc", {"class_0": 0.0, "class_1": 0.0}),
            }
            best_state = copy.deepcopy(last_state)

    if best_metrics is None:
        raise RuntimeError("no best epoch recorded (max_epochs=0?)")
    best_metrics["best_epoch"] = best_metrics["epoch"]
    best_metrics["final_epoch"] = max_epochs
    return best_metrics, best_state, history


# ---------------------------------------------------------------------------
# Stage 1 runner
# ---------------------------------------------------------------------------


def run_stage1(args, *, cache_index, sample_types, split_of, device, out_dir):
    t0 = time.time()
    prompt_ids = _load_prompt_ids(Path(args.prompt_set))
    rows = _build_ca_rows_m12traj(
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
        "train": _stack_rows(rows, "relation_train", device),
        "val": _stack_rows(rows, "relation_val", device),
        "test": _stack_rows(rows, "official_test", device),
    }
    X_tr, y_tr, _ = splits["train"]
    X_va, y_va, _ = splits["val"]
    X_te, y_te, _ = splits["test"]
    print(
        f"[stage1] tensor shapes: train={tuple(X_tr.shape)} val={tuple(X_va.shape)} test={tuple(X_te.shape)}",
        file=sys.stderr, flush=True,
    )

    input_dim = int(X_tr.shape[-1])
    encoder = TLSTMEncoderV2(
        input_dim=input_dim,
        embed_dim=args.embed_dim,
        num_layers=args.num_sequence_layers,
        dropout=args.dropout,
    )
    model = TLSTMStage1Wrapper(encoder)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stage1] T-LSTM v2 trainable params = {n_params:,}", file=sys.stderr, flush=True)

    best_metrics, best_state, history = _train_classifier(
        model, X_tr, y_tr, X_va, y_va, X_te, y_te,
        device=device,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        select_metric="val_balanced_acc",
    )

    enc_state = {
        k[len("encoder."):]: v for k, v in best_state.items() if k.startswith("encoder.")
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": enc_state,
            "architecture_version": "t_lstm_v2_encoder",
            "input_dim": int(input_dim),
            "embed_dim": int(args.embed_dim),
            "num_layers": int(args.num_sequence_layers),
            "dropout": float(args.dropout),
            "stage": "ca_pretrain",
            "best_epoch": int(best_metrics["best_epoch"]),
        },
        out_dir / "encoder.pt",
    )
    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": history}, f, indent=2, sort_keys=True)

    metrics = {
        "experiment": "t_lstm_v2_stage1_ca",
        "model_key": args.model_key,
        "method": "t_lstm_v2",
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
# Stage 2 runner
# ---------------------------------------------------------------------------


def run_stage2(args, *, cache_index, sample_types, split_of, misread_labels, device, out_dir):
    t0 = time.time()
    enc_ckpt_path = Path(args.encoder_checkpoint)
    if not enc_ckpt_path.exists():
        raise FileNotFoundError(f"encoder ckpt missing: {enc_ckpt_path}")
    enc_ckpt = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
    enc_state = enc_ckpt.get("encoder_state_dict", enc_ckpt)
    input_dim = int(enc_ckpt.get("input_dim"))
    embed_dim = int(enc_ckpt.get("embed_dim"))
    num_layers = int(enc_ckpt.get("num_layers", 2))
    dropout = float(enc_ckpt.get("dropout", 0.1))
    encoder = TLSTMEncoderV2(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    encoder.load_state_dict(enc_state, strict=True)
    encoder.to(device).eval()
    for p_ in encoder.parameters():
        p_.requires_grad_(False)
    print(
        f"[stage2] loaded encoder: input_dim={input_dim} embed_dim={embed_dim} "
        f"num_layers={num_layers} best_epoch={enc_ckpt.get('best_epoch')}",
        file=sys.stderr, flush=True,
    )

    prompt_ids = _load_prompt_ids(Path(args.prompt_set))
    rows = _build_mn_rows_m12traj(
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
        "train": _stack_rows(rows, "relation_train", device),
        "val": _stack_rows(rows, "relation_val", device),
        "test": _stack_rows(rows, "official_test", device),
    }
    X_tr_raw, y_tr, sid_tr = splits["train"]
    X_va_raw, y_va, sid_va = splits["val"]
    X_te_raw, y_te, sid_te = splits["test"]

    with torch.no_grad():
        Z_tr_chunks = []
        for i in range(0, X_tr_raw.shape[0], args.batch_size):
            Z_tr_chunks.append(encoder(X_tr_raw[i:i + args.batch_size]).detach())
        Z_tr = torch.cat(Z_tr_chunks, dim=0)
        Z_va_chunks = []
        for i in range(0, X_va_raw.shape[0], args.batch_size):
            Z_va_chunks.append(encoder(X_va_raw[i:i + args.batch_size]).detach())
        Z_va = torch.cat(Z_va_chunks, dim=0)
        Z_te_chunks = []
        for i in range(0, X_te_raw.shape[0], args.batch_size):
            Z_te_chunks.append(encoder(X_te_raw[i:i + args.batch_size]).detach())
        Z_te = torch.cat(Z_te_chunks, dim=0)
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
        "experiment": "t_lstm_v2_stage2_mn",
        "model_key": args.model_key,
        "method": "t_lstm_v2",
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
        description="T-LSTM v2 (two-stage): LSTM(4096, 128, 2 layers) C/A pretrain + Linear(128, 2) M/N head.",
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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-sequence-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
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
    proto = "va" if args.model_key in ("qwen2_5_omni_7b", "gemma4_12b_it") else "vt"
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
