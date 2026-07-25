"""Shared data/eval helpers for train_*.py scripts (P5-B extraction).

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

Out of scope (left in each script):
    * _build_parser -- per-script argparse surface
    * Model construction (SP-MLP vs T-LSTM vs TME)
    * run_stage1 / run_stage2 / _train_classifier -- the
      per-architecture training loops still differ enough that
      extraction would obscure more than it would dedupe.
"""
from __future__ import annotations

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

from mprisk.cache.hidden_state_cache import normalize_protocol
from mprisk.cache.prefill_extract import extract_t0_trajectory
from mprisk.data.manifests import read_final_manifest

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
