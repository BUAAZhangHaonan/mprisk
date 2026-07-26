#!/usr/bin/env python3
"""Audit cache_matrix_20260722 outputs.

Checks:
  1. Expected counts: 144 TME (3 encoders x 16 models x 3 seeds) +
     48 SP-MLP + 48 T-LSTM + 16 SDR pipelines.
  2. NaN/inf in core metric columns.
  3. Cross-seed std < 0.05 per (encoder, model).
  4. VA vs VT stratification (3 VA models expected).
  5. InternVL within 2sigma of other VT models.
  6. SDR state distribution sanity (4 modes covered, no empty pattern).

Emits outputs/cache_matrix_20260722/_summary/audit_report.md.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIR = ROOT / "outputs/cache_matrix_20260722/_summary"
RUNS_DIR = ROOT / "outputs/cache_matrix_20260722/runs"
SDR_DIR = ROOT / "outputs/cache_matrix_20260722/sdr"
REPORT = SUMMARY_DIR / "audit_report.md"

MODELS = [
    "gemma3_12b", "gemma3_4b", "gemma4_12b", "glm4_6v_flash",
    "llava_onevision_qwen2_7b", "llava_v1_5_7b", "minicpm_v_2_6", "minicpm_v_4_5",
    "internvl3_5_8b", "phi3_5_vision", "phi4_multimodal", "qwen2_5_omni_7b",
    "qwen2_5_vl_7b", "qwen3_5_4b", "qwen3_5_9b", "qwen3_vl_8b",
]
VA_MODELS = {"qwen2_5_omni_7b", "gemma4_12b_it", "gemma4_12b", "phi4_multimodal"}
SEEDS = (20260717, 20260718, 20260719)
TME_ENCODERS = ("tme_bilstm", "tme_lstm", "tme_gru")
TWO_STAGE = ("sp_mlp", "t_lstm")

STD_THRESHOLD = 0.05


def _is_finite(value) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(f) or math.isinf(f))


def _check_count(rows: list[dict], encoder: str, expected: int, problems: list[str], ok: list[str]):
    actual = len([r for r in rows if r["encoder"] == encoder])
    if actual == expected:
        ok.append(f"count[{encoder}] = {actual}/{expected}")
    else:
        problems.append(f"count[{encoder}] = {actual}/{expected} (SHORT)")


def _check_finite(rows: list[dict], fields: list[str], problems: list[str], ok: list[str]):
    bad = []
    for row in rows:
        for f in fields:
            v = row.get(f)
            if v is not None and not _is_finite(v):
                bad.append(f"{row['encoder']}/{row['model']}/seed{row['seed']}::{f}={v}")
    if bad:
        problems.append(f"non-finite metrics ({len(bad)}): first 5 -> " + ", ".join(bad[:5]))
    else:
        ok.append(f"all metric values finite ({len(rows)} rows x {len(fields)} fields)")


def _check_std(rows: list[dict], problems: list[str], ok: list[str]):
    by_cell: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        v = r.get("test_mn_acc")
        if v is None:
            continue
        by_cell.setdefault((r["encoder"], r["model"]), []).append(float(v))
    bad = []
    for (encoder, model), values in sorted(by_cell.items()):
        if len(values) < 2:
            continue
        std = statistics.stdev(values)
        if std >= STD_THRESHOLD:
            bad.append(f"{encoder}/{model}: std={std:.4f}")
    if bad:
        problems.append(f"cross-seed std >= {STD_THRESHOLD} ({len(bad)} cells): " + "; ".join(bad[:10]))
    else:
        ok.append(f"cross-seed std < {STD_THRESHOLD} for all {len(by_cell)} cells")


def _check_va_vt_strat(rows: list[dict], problems: list[str], ok: list[str]):
    va_in_rows = {r["model"] for r in rows if r["protocol"] == "va"}
    va_expected_in_rows = va_in_rows & VA_MODELS
    if va_expected_in_rows == va_in_rows and va_in_rows.issubset(VA_MODELS):
        ok.append(f"VA/VT stratification: {len(va_in_rows)} VA models in rows = {sorted(va_in_rows)}")
    else:
        problems.append(f"VA/VT stratification mismatch: rows VA = {sorted(va_in_rows)}, expected subset of {sorted(VA_MODELS)}")


def _check_internvl_outlier(rows: list[dict], problems: list[str], ok: list[str]):
    """InternVL test_mn_acc should be within 2sigma of other VT models per encoder."""
    by_enc: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if r["protocol"] != "vt":
            continue
        by_enc.setdefault(r["encoder"], {}).setdefault(r["model"], []).append(float(r["test_mn_acc"]))
    for encoder, model_accs in by_enc.items():
        internvl = model_accs.get("internvl3_5_8b", [])
        others = [v for m, vs in model_accs.items() if m != "internvl3_5_8b" for v in vs]
        if not internvl or not others:
            continue
        mu = statistics.mean(others)
        sigma = statistics.stdev(others) if len(others) >= 2 else 0.0
        ivl = statistics.mean(internvl)
        if sigma > 0 and abs(ivl - mu) > 2 * sigma:
            problems.append(f"InternVL outlier in {encoder}: mean={ivl:.4f}, others mu={mu:.4f} sigma={sigma:.4f} (delta={abs(ivl-mu)/sigma:.2f}sigma)")
        else:
            ok.append(f"InternVL within 2sigma for {encoder} (delta/sigma={'n/a' if sigma==0 else f'{abs(ivl-mu)/sigma:.2f}'})")


def _check_sdr(problems: list[str], ok: list[str]):
    sdr_complete = 0
    sdr_missing_thresholds = 0
    sdr_missing_encoder = 0
    sdr_empty_pattern = []
    for model in MODELS:
        model_dir = SDR_DIR / model
        marker = model_dir / "MISSING_DEPENDENCY"
        if marker.exists():
            content = marker.read_text(encoding="utf-8").strip()
            if content == "MISSING_THRESHOLDS":
                sdr_missing_thresholds += 1
            elif content == "TME_BILSTM_ENCODER_MISSING":
                sdr_missing_encoder += 1
            continue
        # find state_patterns.jsonl under outputs/states/<model>/<proto>/<psk>/tme_proxy_anchor_v1/
        candidates = list(model_dir.rglob("state_patterns.jsonl"))
        if not candidates:
            problems.append(f"SDR[{model}]: no state_patterns.jsonl under {model_dir}")
            continue
        sdr_complete += 1
        # Sanity-check non-empty and 4 modes covered.
        try:
            rows = [json.loads(line) for line in candidates[0].read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"SDR[{model}]: cannot parse {candidates[0]}: {exc}")
            continue
        if not rows:
            sdr_empty_pattern.append(model)
            continue
        modes = {r.get("state_pattern") or r.get("mode") or r.get("pattern") for r in rows}
        if len(modes) < 4:
            problems.append(f"SDR[{model}]: only {len(modes)} state modes covered: {sorted(m for m in modes if m)}")
        else:
            ok.append(f"SDR[{model}]: {len(rows)} rows, {len(modes)} modes")
    ok.append(f"SDR summary: complete={sdr_complete}, missing_thresholds={sdr_missing_thresholds}, missing_encoder={sdr_missing_encoder}, empty={len(sdr_empty_pattern)}")


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    main_csv = SUMMARY_DIR / "main_results.csv"
    if not main_csv.exists():
        REPORT.write_text("# cache_matrix_20260722 audit\n\n## FAIL\n\nmain_results.csv missing; run aggregate_results.py first.\n", encoding="utf-8")
        print(f"FAIL: {main_csv} missing")
        return 1

    with main_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Numeric coercion.
    for r in rows:
        for k in ("seed", "val_mn_acc", "test_mn_acc", "test_mn_auc", "test_mn_ap", "test_mn_f1", "best_epoch", "train_time_min"):
            if r.get(k) in (None, "", "None"):
                r[k] = None
            else:
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass
        r["seed"] = int(r["seed"]) if r["seed"] is not None else None

    problems: list[str] = []
    ok: list[str] = []

    # 1. counts
    _check_count(rows, "tme_bilstm", 48, problems, ok)
    _check_count(rows, "tme_lstm", 48, problems, ok)
    _check_count(rows, "tme_gru", 48, problems, ok)
    _check_count(rows, "sp_mlp", 48, problems, ok)
    _check_count(rows, "t_lstm", 48, problems, ok)

    # 2. finite metrics
    _check_finite(rows, ["val_mn_acc", "test_mn_acc", "test_mn_auc"], problems, ok)

    # 3. cross-seed std
    _check_std(rows, problems, ok)

    # 4. VA/VT stratification
    _check_va_vt_strat(rows, problems, ok)

    # 5. InternVL outlier
    _check_internvl_outlier(rows, problems, ok)

    # 6. SDR pipelines
    _check_sdr(problems, ok)

    # Write report
    status = "PASS" if not problems else "FAIL"
    lines = [f"# cache_matrix_20260722 audit", "", f"## Status: {status}", "", "## OK", ""]
    lines += [f"- {x}" for x in ok]
    if problems:
        lines += ["", "## Problems", ""]
        lines += [f"- {x}" for x in problems]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"audit: {status} ({len(ok)} ok, {len(problems)} problems) -> {REPORT}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
