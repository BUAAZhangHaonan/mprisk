#!/usr/bin/env python3
"""canonical_rerun TME e2e v3-B trainer (single-stage M/N).

Architecture (TME_E2E_v3B):
    Reuse SphericalTMEV1.condition_encoder (SequentialTrajectoryEncoderV1:
    Layer-L2 + 1-layer GRU + Linear(256, 128) projection + L2 normalize)
    on each of the 3 conditions. Concatenate the 3 condition_z vectors into
    a 384-d feature. A fresh MLP head [384 -> 32 -> 2] (~12k + 66 params)
    classifies M/N.

    No OrderedLinearRelationV1 (no 3-d bottleneck).
    No spherical normalize on the concat (it would distort the per-cond
    normalization already applied inside the encoder).

Optional warm start:
    ``--tme-pa-checkpoint`` can point at a T1 PA checkpoint
    (outputs/canonical_rerun/T1_gru_ca_frozen/<MODEL>_seed<SEED>/best_checkpoint.pt).
    We extract the ``condition_encoder.*`` weights and load them into the
    encoder. The relation.* weights are discarded.

Training:
    AdamW(lr=5e-4, wd=1e-4) + class-balanced CE + clip 1.0 + 100 epochs +
    best=val_mn_acc (val-selected). Single optimizer group
    (encoder + head at same lr), same convention as T4p M12-only.

Input: 3-condition trajectory [B, 3, 36, 4096] from the prefill cache
(same as T1/T4 canonical TME pipeline).

Output:
    outputs/.../<run>/<MODEL>_seed<SEED>/
        metrics.json
        best_encoder.pt     # full state_dict
        best_head.pt        # head-only state_dict
        best_test_preds.pt  # preds at best epoch
        history.json
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

HERE = Path(__file__).resolve().parent
PROJ_ROOT = HERE.parent.parent
SRC = PROJ_ROOT / "src"
for _p in (str(SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from _seed import set_deterministic_seed  # noqa: E402
from mprisk.cache.hidden_state_cache import normalize_protocol  # noqa: E402
from mprisk.cache.prefill_extract import extract_t0_trajectory  # noqa: E402
from mprisk.data.manifests import read_final_manifest  # noqa: E402
from mprisk.representation.relation_models import (  # noqa: E402
    SequentialTrajectoryEncoderV1,
    strict_l2_normalize,
)
from _trainer_lib import (  # noqa: E402  # P5-B shared helpers
    CONDITIONS, COND_IDX,
    _load_prompt_ids, _scan_cache, _load_split_assignment,
    _load_sample_type_map, _load_misread_labels, _domain_of, _eval_loss,
)


# ---------------------------------------------------------------------------
# Data loading: 3-condition trajectory [B, 3, L, H]
# ---------------------------------------------------------------------------


def _build_3cond_traj(info, prompt_id):
    """Stack [M1, M2, M12] -> [3, L, H]."""
    bundle = []
    for cond in CONDITIONS:
        entry = info[cond].get(prompt_id)
        if entry is None:
            return None
        try:
            traj = extract_t0_trajectory(entry)  # [L, H]
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] t0 extraction failed: {exc}", file=sys.stderr)
            return None
        bundle.append(traj)
    return np.stack(bundle, axis=0).astype(np.float32)


def _build_mn_rows_3cond(
    *,
    cache_index,
    prompt_ids,
    misread_labels,
    sample_types,
    split_of,
):
    """M/N rows from 3-condition trajectory. Conflict-only.
    Each row is one (sid, pid) pair with traj [3, L, H] and label {MISREAD:1, NON_MISREAD:0}.
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
            arr = _build_3cond_traj(info, pid)
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
        return torch.empty(0, 0, 0, 0, device=device, dtype=torch.float32), torch.empty(0, dtype=torch.long), []
    X = torch.from_numpy(np.stack([r["traj"] for r in sub], axis=0)).to(device)
    y = torch.tensor([r["label"] for r in sub], dtype=torch.long)
    sids = [r["sample_id"] for r in sub]
    return X, y, sids


# ---------------------------------------------------------------------------
# Model: TME e2e v3-B (shared GRU encoder + 3-cond concat, no bottleneck)
# ---------------------------------------------------------------------------


class TME_E2E_v3B(nn.Module):
    """TME e2e v3: shared GRU condition_encoder + 3-cond concat, no relation."""

    architecture_version = "tme_e2e_v3b"

    def __init__(
        self,
        *,
        input_dim: int,
        sequence_hidden_dim: int = 256,
        embed_dim: int = 128,
        dropout: float = 0.1,
        head_hidden_dim: int = 32,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.sequence_hidden_dim = int(sequence_hidden_dim)
        self.embed_dim = int(embed_dim)
        self.encoder = SequentialTrajectoryEncoderV1(
            input_dim=input_dim,
            sequence_hidden_dim=sequence_hidden_dim,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        # Head: 3 * embed_dim = 384 -> 32 -> 2
        self.head = nn.Sequential(
            nn.Linear(3 * embed_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, n_classes),
        )

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        """trajectories: [B, 3, L, H] -> logits [B, n_classes]."""
        cond_z = self.encoder(trajectories)  # [B, 3, embed_dim]
        z = cond_z.flatten(start_dim=1)  # [B, 3*embed_dim]
        return self.head(z)


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------


def _aggregate_per_sample(probs, labels, sample_ids):
    seen: dict[str, int] = {}
    unique: list[str] = []
    for s in sample_ids:
        if s not in seen:
            seen[s] = len(unique)
            unique.append(s)
    n_uniq = len(unique)
    agg_prob = np.zeros(n_uniq, dtype=np.float32)
    agg_count = np.zeros(n_uniq, dtype=np.int32)
    agg_label = np.zeros(n_uniq, dtype=np.int64)
    for i, sid in enumerate(sample_ids):
        j = seen[sid]
        agg_prob[j] += float(probs[i])
        agg_count[j] += 1
        agg_label[j] = int(labels[i])
    agg_prob = agg_prob / np.maximum(agg_count, 1.0)
    return agg_prob, agg_label, unique


def _eval_mn(model, X, y, sample_ids, *, batch_size, device):
    n = X.shape[0]
    if n == 0:
        return None
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X[i:i + batch_size]
            logits = model(xb)
            p = F.softmax(logits, dim=-1)[:, 1]
            probs.append(p.detach().cpu().numpy())
    probs = np.concatenate(probs)
    y_np = y.cpu().numpy()
    agg_prob, agg_label, agg_sids = _aggregate_per_sample(probs, y_np, sample_ids)
    pred = (agg_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(agg_label, pred)),
        "balanced_acc": float(balanced_accuracy_score(agg_label, pred)),
        "macro_f1": float(f1_score(agg_label, pred, zero_division=0)),
        "ap": (
            float(average_precision_score(agg_label, agg_prob))
            if len(set(agg_label.tolist())) > 1 else 0.0
        ),
        "roc_auc": (
            float(roc_auc_score(agg_label, agg_prob))
            if len(set(agg_label.tolist())) > 1 else 0.0
        ),
        "positive_rate": float(agg_label.mean()),
        "n": int(len(agg_label)),
        "n_rows": int(n),
        "agg_prob": agg_prob,
        "agg_label": agg_label,
        "agg_sample_ids": agg_sids,
    }


def train_one_epoch(model, X_tr, y_tr, *, optimizer, batch_size, rng, class_weights=None):
    model.train()
    n_train = X_tr.shape[0]
    idx = rng.permutation(n_train)
    total_loss = 0.0
    n_batches = 0
    for i in range(0, n_train, batch_size):
        b = idx[i:i + batch_size]
        xb = X_tr[b]
        yb = y_tr[b].to(xb.device)
        logits = model(xb)
        if class_weights is not None:
            loss = F.cross_entropy(logits, yb, weight=class_weights.to(xb.device))
        else:
            loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
        )
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train_tme_v3b(
    *,
    splits,
    device,
    max_epochs,
    batch_size,
    lr,
    weight_decay,
    dropout,
    sequence_hidden_dim,
    embed_dim,
    head_hidden_dim,
    seed,
    tme_pa_checkpoint: Path | None,
):
    set_deterministic_seed(seed)

    X_tr = splits["train"][0]
    if X_tr.shape[0] == 0:
        raise RuntimeError("empty train split")
    input_dim = int(X_tr.shape[-1])

    model = TME_E2E_v3B(
        input_dim=input_dim,
        sequence_hidden_dim=sequence_hidden_dim,
        embed_dim=embed_dim,
        dropout=dropout,
        head_hidden_dim=head_hidden_dim,
        n_classes=2,
    ).to(device)

    # Optional warm start from T1 PA checkpoint.
    warm_start_used = False
    if tme_pa_checkpoint is not None:
        ckpt = torch.load(tme_pa_checkpoint, map_location="cpu", weights_only=False)
        # C-A1-R5-4: refuse to warm-start the GRU encoder from a checkpoint
        # built on a different architecture (e.g. an LSTM checkpoint). The
        # encoder state-dict shape will not match and load_state_dict(strict=True)
        # below would fail with a cryptic tensor-mismatch error.
        arch = ckpt.get("architecture_version")
        if arch != "layer_l2_gru_linear_relation_v1":
            raise ValueError(
                f"--tme-pa-checkpoint has architecture_version={arch!r}; "
                "train_tme_e2e only supports GRU warm-start "
                "(layer_l2_gru_linear_relation_v1)."
            )
        full_state = ckpt.get("model_state_dict", ckpt)
        enc_state = {
            k[len("condition_encoder."):]: v
            for k, v in full_state.items()
            if k.startswith("condition_encoder.")
        }
        if not enc_state:
            raise RuntimeError(
                f"PA checkpoint {tme_pa_checkpoint} has no condition_encoder.* keys"
            )
        model.encoder.load_state_dict(enc_state, strict=True)
        warm_start_used = True
        print(
            f"[warm_start] loaded condition_encoder from {tme_pa_checkpoint} "
            f"({len(enc_state)} keys)",
            file=sys.stderr, flush=True,
        )

    # m-A1-R5-7: rely on PyTorch's default nn.Linear init (Kaiming-uniform
    # with a=sqrt(5)); explicitly forcing Kaiming-normal with nonlinearity
    # "relu" does not match the actual head activations and would only
    # perturb convergence without a principled reason.

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] TME v3-B trainable params = {n_params:,}", file=sys.stderr, flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable params")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    X_tr, y_tr, _sid_tr = splits["train"]
    X_va, y_va, sid_va = splits["val"]
    X_te, y_te, sid_te = splits["test"]

    # Class-balanced CE: compute global class weights from full train set
    # (inverse-frequency, normalized to mean=1). This counters the heavy
    # MISREAD/NON_MISREAD imbalance per S3.2 of the paper.
    y_tr_cpu = y_tr.detach().cpu()
    class_counts = torch.bincount(y_tr_cpu, minlength=2).float()
    class_weights = y_tr_cpu.shape[0] / (2.0 * class_counts + 1e-8)
    class_weights = class_weights / class_weights.mean()
    print(
        f"[setup] class-balanced CE: counts={class_counts.tolist()} "
        f"weights={class_weights.tolist()}",
        file=sys.stderr, flush=True,
    )

    rng = np.random.RandomState(seed)

    best_val_acc = -1.0
    best_metrics = None
    best_state = None
    best_test_payload = None
    best_epoch = -1
    history: list[dict] = []
    last_state = None

    for epoch in range(1, max_epochs + 1):
        avg_loss = train_one_epoch(
            model,
            X_tr,
            y_tr,
            optimizer=optimizer,
            batch_size=batch_size,
            rng=rng,
            class_weights=class_weights,
        )
        val_loss = _eval_loss(model, X_va, y_va, batch_size=batch_size)
        val_metrics = _eval_mn(model, X_va, y_va, sid_va, batch_size=batch_size, device=device) or {}
        te_metrics = _eval_mn(model, X_te, y_te, sid_te, batch_size=batch_size, device=device) or {}

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(avg_loss),
            "val_loss": float(val_loss),
            "val_mn_acc": val_metrics.get("accuracy"),
            "val_mn_balanced_acc": val_metrics.get("balanced_acc"),
            "val_mn_ap": val_metrics.get("ap"),
            "val_mn_f1": val_metrics.get("macro_f1"),
            "val_mn_auc": val_metrics.get("roc_auc"),
            "test_mn_acc": te_metrics.get("accuracy"),
            "test_mn_balanced_acc": te_metrics.get("balanced_acc"),
            "test_mn_ap": te_metrics.get("ap"),
            "test_mn_f1": te_metrics.get("macro_f1"),
            "test_mn_auc": te_metrics.get("roc_auc"),
        }
        history.append(epoch_record)
        print(
            f"[ep{epoch:03d}] loss={avg_loss:.4f} val_loss={val_loss:.4f} "
            f"val_ap={val_metrics.get('ap', 0):.4f} "
            f"val_bal={val_metrics.get('balanced_acc', 0):.4f} "
            f"test_ap={te_metrics.get('ap', 0):.4f} "
            f"test_acc={te_metrics.get('accuracy', 0):.4f} "
            f"test_bal={te_metrics.get('balanced_acc', 0):.4f}",
            file=sys.stderr, flush=True,
        )

        # best.pt = highest val_mn_acc (val-selected; avoids test-set leakage).
        val_acc_for_pick = float(val_metrics.get("accuracy", 0.0) or 0.0)
        if val_acc_for_pick > best_val_acc:
            best_val_acc = val_acc_for_pick
            best_metrics = {
                "epoch": epoch,
                "selection_metric": "val_mn_acc",
                "val_mn_acc": val_metrics.get("accuracy"),
                "val_mn_balanced_acc": val_metrics.get("balanced_acc"),
                "val_mn_ap": val_metrics.get("ap"),
                "val_mn_f1": val_metrics.get("macro_f1"),
                "val_mn_auc": val_metrics.get("roc_auc"),
                "test_mn_acc": te_metrics.get("accuracy"),
                "test_mn_balanced_acc": te_metrics.get("balanced_acc"),
                "test_mn_ap": te_metrics.get("ap"),
                "test_mn_f1": te_metrics.get("macro_f1"),
                "test_mn_auc": te_metrics.get("roc_auc"),
                "n_train_rows": int(X_tr.shape[0]),
                "n_val_rows": int(X_va.shape[0]),
                "n_test_rows": int(X_te.shape[0]),
                "n_train": int(len(set(_sid_tr))),
                "n_val": int(val_metrics.get("n", 0)),
                "n_test": int(te_metrics.get("n", 0)),
            }
            best_state = {"model": copy.deepcopy(model.state_dict())}
            best_test_payload = {
                "preds": (te_metrics["agg_prob"] >= 0.5).astype(np.int32),
                "probs": te_metrics["agg_prob"].astype(np.float32),
                "labels": te_metrics["agg_label"].astype(np.int32),
                "sample_ids": list(te_metrics["agg_sample_ids"]),
                "epoch": int(epoch),
            }
            best_epoch = epoch

        last_state = {
            "model": copy.deepcopy(model.state_dict()),
            "epoch": epoch,
        }

    if best_metrics is None:
        raise RuntimeError("no best epoch recorded (max_epochs=0?)")
    best_metrics["best_epoch"] = best_epoch
    best_metrics["final_epoch"] = max_epochs

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # M-A1-R5-7: previously we monkey-patched these onto the model instance
    # and read them back via getattr in main(); we now return them directly
    # so the data flow is explicit and the model is left untouched.
    return (
        best_metrics,
        {"history": history},
        model,
        n_params,
        warm_start_used,
        last_state,
        best_test_payload,
    )


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------


def save_artifacts(out_dir, *, model, args, best_metrics, history_payload, n_params, warm_start_used, tme_pa_checkpoint):
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            },
            "architecture_version": model.architecture_version,
            "input_dim": int(model.input_dim),
            "sequence_hidden_dim": int(model.sequence_hidden_dim),
            "embed_dim": int(model.embed_dim),
            "head_hidden_dim": int(args.head_hidden_dim),
            "dropout": float(args.dropout),
            "warm_start_checkpoint": str(tme_pa_checkpoint) if tme_pa_checkpoint else None,
            "warm_start_used": bool(warm_start_used),
            "best_epoch": int(best_metrics["best_epoch"]),
        },
        out_dir / "best_encoder.pt",
    )

    torch.save(
        {
            "head_state_dict": {
                k: v.detach().cpu().clone()
                for k, v in model.head.state_dict().items()
            },
            "head_in_dim": int(model.head[0].in_features),
            "head_hidden_dim": int(args.head_hidden_dim),
        },
        out_dir / "best_head.pt",
    )

    metrics = {
        "experiment": "tme_e2e_v3b",
        "model_key": args.model_key,
        "method": "tme_v3b",
        "encoder_mode": "shared_gru_3cond_concat_e2e",
        "warm_start_checkpoint": str(tme_pa_checkpoint) if tme_pa_checkpoint else None,
        "warm_start_used": bool(warm_start_used),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "sequence_hidden_dim": int(args.sequence_hidden_dim),
        "embed_dim": int(args.embed_dim),
        "head_hidden_dim": int(args.head_hidden_dim),
        "max_epochs": int(args.max_epochs),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "split_assignment": str(args.split_assignment),
        "n_trainable_params": int(n_params),
        "best_epoch": int(best_metrics["best_epoch"]),
        "best_test_mn_acc": float(best_metrics["test_mn_acc"]),
        "best_test_mn_ap": float(best_metrics["test_mn_ap"]),
        "best_test_mn_f1": float(best_metrics["test_mn_f1"]),
        "best_metrics": best_metrics,
        "history_last_run": history_payload,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "TME e2e v3-B: shared GRU encoder + 3-cond concat + MLP head [384->32->2]. "
            "Single-stage M/N, AdamW + plain CE, 200 epochs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", choices=("misread",), default="misread")
    p.add_argument("--model-key", required=True)
    p.add_argument("--split-assignment", required=True)
    p.add_argument("--misread-judgments", required=True)
    p.add_argument("--cache-roots", nargs="+", required=True)
    p.add_argument("--prompt-set", required=True)
    p.add_argument("--main-manifest", required=True)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260717)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--sequence-hidden-dim", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--head-hidden-dim", type=int, default=32)
    p.add_argument(
        "--tme-pa-checkpoint", default=None,
        help="optional T1 PA checkpoint for warm start (condition_encoder loaded)",
    )
    p.add_argument("--output-dir", required=True)
    return p


def main():
    args = _build_parser().parse_args()
    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}", file=sys.stderr, flush=True)

    prompt_ids = _load_prompt_ids(Path(args.prompt_set))
    if not prompt_ids:
        raise RuntimeError(f"no enabled prompts in {args.prompt_set}")
    cache_index = _scan_cache(
        [Path(p) for p in args.cache_roots],
        model_key=args.model_key,
        prompt_ids=prompt_ids,
    )
    print(f"[setup] cache covers {len(cache_index)} samples", file=sys.stderr, flush=True)
    proto = "va" if args.model_key in ("qwen2_5_omni_7b", "gemma4_12b_it") else "vt"
    sample_types = _load_sample_type_map(Path(args.main_manifest), protocol=proto)
    split_of = _load_split_assignment(Path(args.split_assignment))

    misread_labels = _load_misread_labels(Path(args.misread_judgments))
    print(f"[setup] MN labels = {len(misread_labels)}", file=sys.stderr, flush=True)
    rows = _build_mn_rows_3cond(
        cache_index=cache_index,
        prompt_ids=prompt_ids,
        misread_labels=misread_labels,
        sample_types=sample_types,
        split_of=split_of,
    )
    n_tr = sum(1 for r in rows if r["split"] == "relation_train")
    n_va = sum(1 for r in rows if r["split"] == "relation_val")
    n_te = sum(1 for r in rows if r["split"] == "official_test")
    print(f"[setup] rows: train={n_tr} val={n_va} test={n_te}", file=sys.stderr, flush=True)
    if n_tr == 0 or n_te == 0:
        raise RuntimeError(
            f"empty split: train={n_tr} test={n_te}; "
            "check --split-assignment / --misread-judgments / --cache-roots"
        )

    splits = {
        "train": _stack_rows(rows, "relation_train", device),
        "val": _stack_rows(rows, "relation_val", device),
        "test": _stack_rows(rows, "official_test", device),
    }
    print(
        f"[setup] tensor shapes: train={tuple(splits['train'][0].shape)} "
        f"val={tuple(splits['val'][0].shape)} test={tuple(splits['test'][0].shape)}",
        file=sys.stderr, flush=True,
    )

    tme_pa_ckpt = Path(args.tme_pa_checkpoint) if args.tme_pa_checkpoint else None
    if tme_pa_ckpt is not None and not tme_pa_ckpt.exists():
        raise FileNotFoundError(f"TME PA checkpoint not found: {tme_pa_ckpt}")

    # M-A1-R5-7: train_tme_v3b now returns last_state + best_test_payload as
    # explicit tuple elements instead of monkey-patching them onto the model.
    (
        best_metrics,
        history_payload,
        model,
        n_params,
        warm_start_used,
        last_snapshot,
        best_test_payload,
    ) = train_tme_v3b(
        splits=splits,
        device=device,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        sequence_hidden_dim=args.sequence_hidden_dim,
        embed_dim=args.embed_dim,
        head_hidden_dim=args.head_hidden_dim,
        seed=args.seed,
        tme_pa_checkpoint=tme_pa_ckpt,
    )

    out_dir = Path(args.output_dir)
    save_artifacts(
        out_dir,
        model=model,
        args=args,
        best_metrics=best_metrics,
        history_payload=history_payload,
        n_params=n_params,
        warm_start_used=warm_start_used,
        tme_pa_checkpoint=tme_pa_ckpt,
    )

    # last_encoder.pt -- final-epoch snapshot
    if last_snapshot is not None:
        torch.save(
            {
                "model_state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in last_snapshot["model"].items()
                },
                "architecture_version": model.architecture_version,
                "epoch": int(last_snapshot["epoch"]),
            },
            out_dir / "last_encoder.pt",
        )

    if best_test_payload is not None:
        torch.save(
            {
                "epoch": int(best_test_payload["epoch"]),
                "sample_ids": list(best_test_payload["sample_ids"]),
                "labels": np.asarray(best_test_payload["labels"], dtype=np.int32),
                "preds": np.asarray(best_test_payload["preds"], dtype=np.int32),
                "probs": np.asarray(best_test_payload["probs"], dtype=np.float32),
                "task": "misread",
                "selection_metric": "val_mn_acc",
            },
            out_dir / "best_test_preds.pt",
        )

    elapsed = time.time() - t0
    print(
        f"[done] task=misread "
        f"best_epoch={best_metrics['best_epoch']} "
        f"sel=val_mn_acc "
        f"test_acc={best_metrics['test_mn_acc']:.4f} "
        f"test_ap={best_metrics['test_mn_ap']:.4f} "
        f"test_f1={best_metrics['test_mn_f1']:.4f} "
        f"({elapsed:.1f}s)",
        file=sys.stderr, flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
