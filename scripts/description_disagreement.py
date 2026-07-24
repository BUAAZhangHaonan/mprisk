"""Description Disagreement baseline (simplified).

Paper definition: M1 description vs M2 description 的连续语义距离作为分数。

Simplified implementation (no extra generation needed):
- Use the frozen condition_z (M1, M2, M12) from TME training.
- The spherical distance d(M1, M2) serves as a proxy for "the model would
  produce very different descriptions in each single-modality condition".
- Evaluate Acc / Macro-F1 / AP on Conflict samples using Phase 1 misread labels.

This is a geometric proxy; a full text-based DD would regenerate M1/M2
descriptions. For quick paper placeholder we use this distance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def load_frozen_with_condition_z(path: Path):
    """Returns dict[sample_id] -> list of {M1, M2, M12 vectors, split, label_id} per prompt row."""
    by_sid: dict[str, list] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            sid = d["sample_id"]
            cz = d["condition_z"]  # dict with M1/M2/M12 keys
            by_sid.setdefault(sid, []).append({
                "M1": np.asarray(cz["M1"], dtype=np.float32),
                "M2": np.asarray(cz["M2"], dtype=np.float32),
                "M12": np.asarray(cz["M12"], dtype=np.float32),
                "split": d.get("representation_split") or "",
                "label_id": d.get("label_id"),
            })
    return by_sid


def spherical_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic distance on unit sphere."""
    cos = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1.0, 1.0))
    return float(np.arccos(cos))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frozen", required=True)
    p.add_argument("--misread", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    frozen = load_frozen_with_condition_z(Path(args.frozen))
    misread = {}
    with open(args.misread) as f:
        for line in f:
            d = json.loads(line)
            misread[d["sample_id"]] = 1 if d.get("final_label") == "MISREAD" else 0

    # Conflict only: per sample average d(M1, M2) across prompts as score
    rows = []
    for sid, prompts in frozen.items():
        if sid not in misread:
            continue
        # Conflict only
        if prompts[0]["label_id"] != 1:
            continue
        dists = []
        for p_row in prompts:
            dists.append(spherical_distance(p_row["M1"], p_row["M2"]))
        rows.append({
            "sample_id": sid,
            "score": float(np.mean(dists)),
            "label": misread[sid],
            "split": prompts[0]["split"],
        })

    print(f"loaded {len(rows)} Conflict samples", file=sys.stderr)

    test = [r for r in rows if r["split"] == "official_test"]
    train = [r for r in rows if r["split"] != "official_test"]
    if not test:
        raise ValueError(
            f"no official_test rows in {len(rows)} samples; check representation_split field"
        )

    train_scores = np.asarray([r["score"] for r in train])
    train_labels = np.asarray([r["label"] for r in train])
    test_scores = np.asarray([r["score"] for r in test])
    test_labels = np.asarray([r["label"] for r in test])

    # Higher score = higher disagreement = predict MISREAD
    # Threshold on train: pick threshold maximizing F1 on train
    best_thresh = 0.5
    best_f1 = -1
    candidates = np.linspace(train_scores.min(), train_scores.max(), 50)
    for t in candidates:
        pred = (train_scores >= t).astype(int)
        f1 = f1_score(train_labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    test_pred = (test_scores >= best_thresh).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(test_labels, test_pred)),
        "macro_f1": float(f1_score(test_labels, test_pred, zero_division=0)),
        "ap": float(average_precision_score(test_labels, test_scores)),
        "roc_auc": float(roc_auc_score(test_labels, test_scores)) if len(set(test_labels)) > 1 else 0.5,
        "best_threshold": best_thresh,
        "n_train": len(train),
        "n_test": len(test),
        "misread_rate": float(test_labels.mean()),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "desc_disagree_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
