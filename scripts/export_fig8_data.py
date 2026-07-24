#!/usr/bin/env python
"""Aggregate v2 two-stage baseline outputs into Table 3 summary + Fig.8 data.

Reads a tree of the form:
    <input>/{indomain,crossdomain}_<method>_r<ratio>/
        pretrain_metrics.json
        mn_metrics.json
        embeddings.npz      (in-domain, ratio=1.0 used for UMAP/TSNE)

Produces:
    summary_table3.json   (ca_acc / mn_acc per method, per domain, per ratio)
    fig8_<model>.npz      (UMAP coords + curves expected by fig8 templates)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Optional deps; pick whichever is available.
try:
    import umap  # type: ignore
    _HAVE_UMAP = True
except Exception:
    _HAVE_UMAP = False

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

METHOD_ORDER = ["sp_mlp", "t_mlp", "tme"]
METHOD_DISPLAY = {
    "sp_mlp": "Single-Point",
    "t_mlp": "Trajectory MLP",
    "tme": "Trajectory Manifold Encoder",
}
FRACTIONS = [0.1, 0.25, 0.5, 1.0]
RUN_RE = re.compile(r"^(indomain|crossdomain)_(sp_mlp|t_mlp|tme)_r([0-9.]+)$")


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to read {p}: {exc}", file=sys.stderr)
        return None


def discover_runs(root: Path):
    """Return {(domain, method, ratio_str): run_dir} for all matches."""
    runs = {}
    if not root.is_dir():
        return runs
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = RUN_RE.match(child.name)
        if not m:
            continue
        domain = "in_domain" if m.group(1) == "indomain" else "cross_domain"
        method, ratio = m.group(2), m.group(3)
        runs[(domain, method, ratio)] = child
    return runs


def _acc(metrics: dict | None, key: str) -> float | None:
    if metrics is None:
        return None
    v = metrics.get(key)
    return float(v) if v is not None else None


def build_summary(root: Path, model_key: str, runs: dict) -> dict:
    summary = {
        "model_key": model_key,
        "methods": METHOD_ORDER,
        "in_domain": {},
        "cross_domain": {},
        "conflict_budget": {m: {} for m in METHOD_ORDER},
    }

    # Best per-method in-domain / cross-domain (any ratio).
    for method in METHOD_ORDER:
        for domain in ("in_domain", "cross_domain"):
            best_ca = best_mn = -1.0
            found = False
            for (d, m, _r), run in runs.items():
                if d != domain or m != method:
                    continue
                found = True
                pre = _read_json(run / "pretrain_metrics.json")
                mn = _read_json(run / "mn_metrics.json")
                ca = _acc(pre, "test_balanced_accuracy_ac")
                macc = _acc(mn, "test_balanced_accuracy_mn")
                if ca is not None and ca > best_ca:
                    best_ca = ca
                if macc is not None and macc > best_mn:
                    best_mn = macc
            summary[domain][method] = {
                "ca_acc": (round(best_ca, 4) if found else None),
                "mn_acc": (round(best_mn, 4) if found else None),
            }

    # Conflict-budget table: ratio -> ca/mn for each method (in-domain).
    for method in METHOD_ORDER:
        for frac in FRACTIONS:
            ratio_str = str(frac)
            run = runs.get(("in_domain", method, ratio_str))
            entry = {"ca": None, "mn": None}
            if run is not None:
                pre = _read_json(run / "pretrain_metrics.json")
                mn = _read_json(run / "mn_metrics.json")
                ca = _acc(pre, "test_balanced_accuracy_ac")
                macc = _acc(mn, "test_balanced_accuracy_mn")
                if ca is not None:
                    entry["ca"] = round(ca, 4)
                if macc is not None:
                    entry["mn"] = round(macc, 4)
            summary["conflict_budget"][method][ratio_str] = entry

    return summary


def _project(xy: np.ndarray) -> np.ndarray:
    """Reduce [N, D] -> [N, 2] via UMAP (preferred) or TSNE fallback."""
    xy = np.asarray(xy, dtype=np.float32)
    if _HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            metric="euclidean", random_state=0)
        return reducer.fit_transform(xy).astype(np.float32)
    # Fallback: TSNE with sane defaults; PCA pre-trunc if very wide.
    n = xy.shape[0]
    perplexity = max(5.0, min(30.0, (n - 1) / 3.0))
    return TSNE(n_components=2, perplexity=perplexity, init="pca",
                random_state=0).fit_transform(xy).astype(np.float32)


def _knn_purity(xy: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Leave-one-out 5-NN predicted-label purity."""
    n = xy.shape[0]
    if n <= k + 1 or len(set(labels.tolist())) < 2:
        return float("nan")
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(xy, labels)
    pred = clf.predict(xy)
    return float(np.mean(pred == labels))


def build_fig8(root: Path, model_key: str, runs: dict) -> dict:
    """Build the structure expected by fig8_representation_quality.py."""
    method_names = [METHOD_DISPLAY[m] for m in METHOD_ORDER]
    target_xy = []
    y_target = None
    silhouette = np.full(len(METHOD_ORDER), np.nan, dtype=np.float32)
    knn_purity = np.full(len(METHOD_ORDER), np.nan, dtype=np.float32)
    cross_bal_acc = np.full(len(METHOD_ORDER), np.nan, dtype=np.float32)
    in_curves = np.full((len(METHOD_ORDER), len(FRACTIONS)), np.nan, dtype=np.float32)
    cross_curves = np.full((len(METHOD_ORDER), len(FRACTIONS)), np.nan, dtype=np.float32)

    for mi, method in enumerate(METHOD_ORDER):
        # In-domain ratio=1.0 embeddings drive UMAP/TSNE + geometry metrics.
        run = runs.get(("in_domain", method, "1.0"))
        coords = np.zeros((0, 2), dtype=np.float32)
        if run is not None and (run / "embeddings.npz").exists():
            with np.load(run / "embeddings.npz", allow_pickle=True) as nz:
                emb = nz["embeddings"].astype(np.float32)
                labels = np.asarray(nz["labels"])
                test_mask = np.asarray(nz["test_mask"]).astype(bool)
            emb_test = emb[test_mask] if test_mask.any() else emb
            labels_test = labels[test_mask] if test_mask.any() else labels
            # Map string labels (MISREAD/NON_MISREAD, Conflict/Aligned) -> int.
            uniq = {lab: i for i, lab in enumerate(sorted(set(labels_test.tolist())))}
            y_int = np.array([uniq[v] for v in labels_test], dtype=np.int64)
            try:
                coords = _project(emb_test)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] projection failed for {method}: {exc}", file=sys.stderr)
                coords = np.zeros((emb_test.shape[0], 2), dtype=np.float32)
            if len(set(y_int.tolist())) >= 2 and coords.shape[0] > 5:
                try:
                    silhouette[mi] = float(silhouette_score(coords, y_int))
                except Exception:  # noqa: BLE001
                    pass
            knn_purity[mi] = _knn_purity(coords, y_int, k=5)
            # Capture y_target from first method that produced valid labels.
            if y_target is None and coords.shape[0] > 0:
                y_target = y_int

        # Always append this method's coords (may be empty if probe failed).
        target_xy.append(coords.astype(np.float32))

        # In-domain conflict-budget curve (mn acc per fraction).
        for fi, frac in enumerate(FRACTIONS):
            run_f = runs.get(("in_domain", method, str(frac)))
            if run_f is not None:
                mn = _read_json(run_f / "mn_metrics.json")
                v = _acc(mn, "test_balanced_accuracy_mn")
                if v is not None:
                    in_curves[mi, fi] = v
            run_c = runs.get(("cross_domain", method, str(frac)))
            if run_c is not None:
                mn = _read_json(run_c / "mn_metrics.json")
                v = _acc(mn, "test_balanced_accuracy_mn")
                if v is not None:
                    cross_curves[mi, fi] = v
                    if fi == FRACTIONS.index(1.0):
                        cross_bal_acc[mi] = v

    # Normalize target_xy to [3, N_min, 2] by truncating to smallest N.
    # If a method produced no coords (e.g. probe failed), substitute zeros so
    # np.stack succeeds; downstream silhouette/knn_purity already NaN for it.
    nonzero = [c for c in target_xy if c.shape[0] > 0]
    n_min = min((c.shape[0] for c in nonzero), default=0)
    if n_min == 0:
        target_arr = np.zeros((3, 0, 2), dtype=np.float32)
        y_target = np.zeros((0,), dtype=np.int64)
    else:
        normalized = []
        for c in target_xy:
            if c.shape[0] == 0:
                # Substitute zeros — no real signal for this method.
                normalized.append(np.zeros((n_min, 2), dtype=np.float32))
            else:
                normalized.append(c[:n_min])
        target_arr = np.stack(normalized, axis=0).astype(np.float32)
        y_target = (y_target[:n_min] if y_target is not None
                    else np.zeros((n_min,), dtype=np.int64))

    return {
        "method_names": np.asarray(method_names, dtype=object),
        "target_xy": target_arr,
        "y_target": np.asarray(y_target, dtype=np.int64),
        "fractions": np.asarray(FRACTIONS, dtype=np.float32),
        "in_curves": in_curves,
        "cross_curves": cross_curves,
        "silhouette": silhouette,
        "knn_purity": knn_purity,
        "cross_bal_acc": cross_bal_acc,
        "reducer": np.asarray(["umap" if _HAVE_UMAP else "tsne"]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                    help="experiment output tree (model_<stamp>)")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--output-summary", required=True, type=Path)
    ap.add_argument("--output-fig8", required=True, type=Path)
    args = ap.parse_args(argv)

    if not args.input.is_dir():
        print(f"[error] input tree not found: {args.input}", file=sys.stderr)
        return 1

    runs = discover_runs(args.input)
    if not runs:
        print(f"[error] no run directories found under {args.input}", file=sys.stderr)
        return 1

    summary = build_summary(args.input, args.model_key, runs)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] wrote {args.output_summary}")

    fig8 = build_fig8(args.input, args.model_key, runs)
    args.output_fig8.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_fig8, **fig8)
    reducer = "UMAP" if _HAVE_UMAP else "TSNE (fallback)"
    print(f"[fig8] wrote {args.output_fig8} (reducer={reducer})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
