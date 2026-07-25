"""Shared data/eval helpers for train_*.py scripts (P5-B + P7-C extractions).

These helpers were duplicated byte-identically across
scripts/train_sp_mlp.py, scripts/train_t_lstm.py and
scripts/train_tme_e2e.py. Centralizing them removes ~250 LOC of
copy-paste while preserving the exact behaviour of every call site
(no renames, no signature changes, no default changes).

Scope (intentionally minimal):
    * Cache-index loading (_scan_cache) and JSONL readers
      (_load_split_assignment, _load_misread_labels)
    * Manifest/sample-type loaders
    * Domain classifier (_domain_of)
    * Generic 2-class balanced-accuracy + cross-entropy eval helpers
    * Two-stage classifier train/eval loop (eval_classifier,
      train_classifier) -- added in P7-C after re-audit showed the
      sp_mlp and t_lstm copies were byte-identical modulo docstrings
      and one unused local.

Out of scope (left in each script):
    * _build_parser -- per-script argparse surface (encoder-type
      description, LSTM-only --num-sequence-layers/--dropout, etc.)
    * Model construction (SP-MLP vs T-LSTM vs TME)
    * run_stage1 / run_stage2 -- the per-architecture encoder
      construction, checkpoint schema, and (for LSTM) chunked
      batched encoding still differ enough that extraction would
      obscure more than it would dedupe.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from mprisk.cache.hidden_state_cache import normalize_protocol
from mprisk.cache.prefill_extract import extract_t0_trajectory
from mprisk.data.manifests import read_final_manifest
from mprisk.utils.seeds import set_deterministic_seed

CONDITIONS = ("M1", "M2", "M12")
COND_IDX = {"M1": 0, "M2": 1, "M12": 2}

__all__ = [
    "CONDITIONS",
    "COND_IDX",
    "load_prompt_ids",
    "scan_cache",
    "load_split_assignment",
    "load_sample_type_map",
    "load_misread_labels",
    "domain_of",
    "balanced_per_class_acc",
    "eval_loss",
    "eval_classifier",
    "train_classifier",
    "set_deterministic_seed",
]


# ---------------------------------------------------------------------------
# Data loading (identical across sp_mlp / t_lstm / tme_e2e)
# ---------------------------------------------------------------------------


def load_prompt_ids(prompt_set_path: Path) -> list[str]:
    import yaml
    with open(prompt_set_path, "r", encoding="utf-8") as f:
        ps = yaml.safe_load(f)
    return [t["prompt_id"] for t in ps["templates"] if t.get("enabled", True)]


def scan_cache(cache_roots, *, model_key, prompt_ids):
    from mprisk.cache.cache_manifest import _can_materialize_entry, _entry_from_row
    out: dict[str, dict[str, dict[str, object]]] = {}
    expected = set(prompt_ids)
    for root in cache_roots:
        if not root.exists():
            print(f"[warn] cache root missing: {root}", file=sys.stderr)
            continue
        manifest_path = root / "manifest.jsonl"
        if not manifest_path.exists():
            print(f"[warn] manifest.jsonl missing in {root}", file=sys.stderr)
            continue
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("model_key") != model_key:
                    continue
                cond = row.get("condition")
                if cond not in CONDITIONS:
                    continue
                pid = row.get("prompt_id") or (row.get("metadata") or {}).get("prompt_id")
                if pid is None or pid not in expected:
                    continue
                if not _can_materialize_entry(row):
                    continue
                entry = _entry_from_row(row, cache_root=root)
                slot = out.setdefault(entry.sample_id, {c: {} for c in CONDITIONS})
                slot[cond].setdefault(pid, entry)
    return out


def load_split_assignment(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split = row.get("representation_split", "")
            for sid in row.get("sample_ids", []):
                out[sid] = split
    return out


def load_sample_type_map(main_manifest: Path, protocol: str) -> dict[str, str]:
    rows = read_final_manifest(main_manifest)
    return {
        r.sample_id: r.sample_type
        for r in rows
        if normalize_protocol(r.protocol) == normalize_protocol(protocol)
    }


def load_misread_labels(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("sample_id")
            label = row.get("final_label")
            if sid and label:
                out[sid] = label
    return out


def domain_of(sid: str) -> str:
    return "gen" if sid.startswith("gen:") else "natural"


# ---------------------------------------------------------------------------
# Eval helpers (identical across sp_mlp / t_lstm; tme_e2e also uses eval_loss)
# ---------------------------------------------------------------------------


def balanced_per_class_acc(preds, labels):
    a_idx = labels == 0
    c_idx = labels == 1
    a_acc = float((preds[a_idx] == 0).mean()) if a_idx.any() else 0.0
    c_acc = float((preds[c_idx] == 1).mean()) if c_idx.any() else 0.0
    bal = balanced_accuracy_score(labels, preds) if len(labels) else 0.0
    return float(bal), {"class_0": a_acc, "class_1": c_acc}


def eval_loss(model, X, y, *, batch_size):
    if X.shape[0] == 0:
        return 0.0
    model.eval()
    total = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size]
            yb = y[i:i + batch_size].to(xb.device)
            logits = model(xb)
            total += float(F.cross_entropy(logits, yb).item())
            n_batches += 1
    return total / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Two-stage classifier train/eval loop (added in P7-C; identical between
# sp_mlp and t_lstm modulo docstrings and one unused local).
# ---------------------------------------------------------------------------


def eval_classifier(model, X, y, *, batch_size, device):
    """Compute metrics on (X, y). Returns dict or None if X is empty."""
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
    bal, pc = balanced_per_class_acc(preds, y_np)
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


def train_classifier(
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
    """Train a classifier; return (best_metrics, best_state, history).

    Plain Adam(lr) (no WD) + plain CE + clip 1.0 + no early stop.

    Selection metric defaults to val_balanced_acc (val-selected) to avoid
    test-set leakage when reporting final test numbers.
    """
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

        train_loss = eval_loss(model, X_tr, y_tr, batch_size=batch_size)
        val_loss = eval_loss(model, X_va, y_va, batch_size=batch_size)
        val_m = eval_classifier(model, X_va, y_va, batch_size=batch_size, device=device) or {}
        te_m = eval_classifier(model, X_te, y_te, batch_size=batch_size, device=device) or {}

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


# Re-export for legacy callers that imported the underscore-prefixed name
# from the script module. New code should use the unprefixed names above.
_load_prompt_ids = load_prompt_ids
_scan_cache = scan_cache
_load_split_assignment = load_split_assignment
_load_sample_type_map = load_sample_type_map
_load_misread_labels = load_misread_labels
_domain_of = domain_of
_balanced_per_class_acc = balanced_per_class_acc
_eval_loss = eval_loss
_eval_classifier = eval_classifier
_train_classifier = train_classifier
