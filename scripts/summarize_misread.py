"""Aggregate misread-judgment JSONL files into a summary.json.

Reads judgments.jsonl files (output of judge_misread.py) and the matching
descriptions, joins them with the sample-type/protocol info from the
description files, and computes:

  - per-model misread rate (overall, Conflict-only, Aligned-only)
  - per-model agreement across the 3 V4-flash judgments
    (Cohen's kappa on {MISREAD, NON_MISREAD} after mapping UNCERTAIN to
    the majority decision, plus simple agreement %)
  - how many samples went to V4-pro arbitration
  - wall-clock totals for generation + judgment (if metadata is available)

Layout (new, per-model subdir):
    <misread-dir>/<model>/descriptions.jsonl
    <misread-dir>/<model>/judgments.jsonl

Legacy flat layout (backwards-compat fallback):
    <misread-dir>/descriptions_<model>.jsonl
    <misread-dir>/judgments_<model>.jsonl

Output: outputs/v2/misread/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


# Canonical list of subject models. The summary traverses these by name so
# we get deterministic output and clear warnings when a model is missing.
KNOWN_MODELS = ["qwen3_vl_8b", "internvl3_5_8b", "qwen2_5_omni_7b"]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    dropped = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                dropped += 1
                if dropped <= 3:
                    print(f"[warn] dropped malformed line {i}: {exc}", file=sys.stderr)
                continue
    if dropped:
        print(f"[warn] dropped {dropped} malformed lines total", file=sys.stderr)
    return rows


def _resolve_model_files(misread_dir: Path, model: str) -> tuple[Path | None, Path | None]:
    """Find description + judgment files for ``model`` under ``misread_dir``.

    Returns ``(desc_path, judge_path)`` where either may be None if not
    found. Tries the per-model subdir layout first, then the legacy flat
    layout.
    """
    # Per-model subdir (current layout).
    desc_sub = misread_dir / model / "descriptions.jsonl"
    judge_sub = misread_dir / model / "judgments.jsonl"
    # Legacy flat layout.
    desc_flat = misread_dir / f"descriptions_{model}.jsonl"
    judge_flat = misread_dir / f"judgments_{model}.jsonl"

    desc = desc_sub if desc_sub.exists() else (desc_flat if desc_flat.exists() else None)
    judge = judge_sub if judge_sub.exists() else (judge_flat if judge_flat.exists() else None)
    return desc, judge


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa on paired label lists. Returns NaN-ish 0.0 on degenerate input."""
    assert len(labels_a) == len(labels_b), "label lists must be same length"
    n = len(labels_a)
    if n == 0:
        return 0.0
    labels = sorted(set(labels_a) | set(labels_b))
    # Observed agreement.
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    # Expected by chance.
    counts_a = {l: labels_a.count(l) / n for l in labels}
    counts_b = {l: labels_b.count(l) / n for l in labels}
    expected = sum(counts_a[l] * counts_b[l] for l in labels)
    if expected >= 1.0:
        return 1.0
    denom = 1.0 - expected
    if denom <= 1e-12:
        return 0.0
    return (observed - expected) / denom


def _pairwise_kappa(decisions: list[list[str]]) -> float:
    """Average pairwise Cohen's kappa across N annotators."""
    if len(decisions) < 2:
        return 0.0
    kappas: list[float] = []
    for i in range(len(decisions)):
        for j in range(i + 1, len(decisions)):
            kappas.append(_cohen_kappa(decisions[i], decisions[j]))
    return mean(kappas) if kappas else 0.0


def _agreement_pct(decisions: list[list[str]]) -> float:
    """Fraction of samples where all annotators agree."""
    if not decisions:
        return 0.0
    n = len(decisions[0])
    if n == 0:
        return 0.0
    agreed = 0
    for sample_idx in range(n):
        labels = {decisions[i][sample_idx] for i in range(len(decisions))}
        if len(labels) == 1:
            agreed += 1
    return agreed / n


def _summarize_model(
    model_key: str,
    description_rows: list[dict],
    judgment_rows: list[dict],
) -> dict:
    """Build the per-model summary block."""
    # Index sample metadata by sample_id.
    meta: dict[str, dict] = {}
    for r in description_rows:
        meta[r["sample_id"]] = {
            "sample_type": r.get("sample_type", ""),
            "protocol": r.get("protocol", ""),
        }
    # Index judgments by sample_id.
    judgments_by_sid: dict[str, dict] = {r["sample_id"]: r for r in judgment_rows}

    n_total = 0
    n_error = 0
    n_by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "misread": 0, "non_misread": 0, "error": 0}
    )
    n_arbitration = 0
    flash_decisions_matrix: list[list[str]] = []  # per-flash lists, aligned by sample
    agreement_ratios: list[float] = []

    for sid, meta_row in meta.items():
        j = judgments_by_sid.get(sid)
        n_total += 1
        st = meta_row["sample_type"] or "Unknown"
        n_by_type[st]["total"] += 1
        if j is None:
            n_by_type[st]["error"] += 1
            continue
        final = j.get("final_label", "ERROR")
        if final == "ERROR":
            n_error += 1
            n_by_type[st]["error"] += 1
            continue
        if final == "MISREAD":
            n_by_type[st]["misread"] += 1
        elif final == "NON_MISREAD":
            n_by_type[st]["non_misread"] += 1
        if j.get("arbitrator_used"):
            n_arbitration += 1
        agreement_ratios.append(float(j.get("agreement_ratio", 0.0)))
        # Collect per-flash decisions, mapping UNCERTAIN to final for kappa only.
        flashes = j.get("flash", [])
        if flashes:
            for f_idx, f in enumerate(flashes):
                while len(flash_decisions_matrix) <= f_idx:
                    flash_decisions_matrix.append([])
                # Kappa uses majority-mapped labels (treat UNCERTAIN as final).
                d = f.get("decision", "UNCERTAIN")
                if d == "UNCERTAIN":
                    d = final
                flash_decisions_matrix[f_idx].append(d)

    def _rates(bucket: dict[str, int]) -> dict[str, float]:
        valid = bucket["misread"] + bucket["non_misread"]
        return {
            "n_total": bucket["total"],
            "n_valid": valid,
            "n_misread": bucket["misread"],
            "n_non_misread": bucket["non_misread"],
            "n_error": bucket["error"],
            "misread_rate": (bucket["misread"] / valid) if valid > 0 else None,
        }

    summary = {
        "model_key": model_key,
        "n_samples_in_manifest": n_total,
        "n_errors": n_error,
        "n_arbitration": n_arbitration,
        "overall": _rates({
            "total": sum(b["total"] for b in n_by_type.values()),
            "misread": sum(b["misread"] for b in n_by_type.values()),
            "non_misread": sum(b["non_misread"] for b in n_by_type.values()),
            "error": sum(b["error"] for b in n_by_type.values()),
        }),
        "by_sample_type": {k: _rates(v) for k, v in n_by_type.items()},
        "judge_agreement": {
            "mean_agreement_ratio": (
                mean(agreement_ratios) if agreement_ratios else None
            ),
            "unanimous_pct": _agreement_pct(flash_decisions_matrix) if flash_decisions_matrix else None,
            "mean_pairwise_cohen_kappa": (
                _pairwise_kappa(flash_decisions_matrix) if flash_decisions_matrix else None
            ),
        },
    }
    return summary


def _scan_wall_clock(description_rows: list[dict]) -> dict | None:
    """Per-sample elapsed_seconds -> generation wall clock estimate."""
    elapsed = [float(r.get("elapsed_seconds", 0.0)) for r in description_rows
               if r.get("elapsed_seconds") is not None]
    if not elapsed:
        return None
    return {
        "n_samples": len(elapsed),
        "total_seconds": sum(elapsed),
        "mean_seconds_per_sample": mean(elapsed),
        "max_seconds_per_sample": max(elapsed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--misread-dir", required=True,
        help="Directory containing per-model subdirs "
             "(<model>/descriptions.jsonl, <model>/judgments.jsonl) "
             "or legacy flat files (descriptions_<model>.jsonl, "
             "judgments_<model>.jsonl).",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    misread_dir = Path(args.misread_dir)
    if not misread_dir.exists():
        print(f"misread-dir does not exist: {misread_dir}", file=sys.stderr)
        return 1

    # Detect any model subdirs not in KNOWN_MODELS so we still pick up new
    # models added later without code changes.
    discovered: list[str] = list(KNOWN_MODELS)
    if misread_dir.is_dir():
        for sub in misread_dir.iterdir():
            if sub.is_dir() and sub.name not in discovered:
                # only include if it actually has a descriptions.jsonl or
                # judgments.jsonl inside
                if (sub / "descriptions.jsonl").exists() or (sub / "judgments.jsonl").exists():
                    discovered.append(sub.name)
    # Also pick up legacy flat judgment files not covered above.
    for jf in misread_dir.glob("judgments_*.jsonl"):
        model_key = jf.stem.replace("judgments_", "")
        if model_key not in discovered:
            discovered.append(model_key)

    per_model: dict[str, dict] = {}
    missing: list[str] = []
    for model in discovered:
        desc_path, judge_path = _resolve_model_files(misread_dir, model)
        if desc_path is None and judge_path is None:
            missing.append(model)
            continue
        if desc_path is None:
            print(f"[summary] WARN: no descriptions file for {model}", file=sys.stderr)
            desc_rows: list[dict] = []
        else:
            desc_rows = _load_jsonl(desc_path)
        if judge_path is None:
            print(f"[summary] WARN: no judgments file for {model}", file=sys.stderr)
            judge_rows: list[dict] = []
        else:
            judge_rows = _load_jsonl(judge_path)
        if not desc_rows and not judge_rows:
            print(f"[summary] WARN: both files empty for {model}", file=sys.stderr)
            missing.append(model)
            continue
        per_model[model] = _summarize_model(model, desc_rows, judge_rows)
        per_model[model]["generation_wall_clock"] = _scan_wall_clock(desc_rows)

    if not per_model:
        print(f"[summary] no usable per-model files under {misread_dir}", file=sys.stderr)
        return 1

    for m in missing:
        print(f"[summary] WARN: missing files for model {m}", file=sys.stderr)

    out = {
        "schema": "mprisk_v2_misread_summary_v1",
        "per_model": per_model,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True, ensure_ascii=False)
    print(f"[summary] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
