#!/usr/bin/env python3
"""Audit cache_matrix_20260722 outputs.

Self-contained scanner over outputs/cache_matrix_20260722/runs/.
Layout (post 2026-07 refactor):
  - C/A TME  : runs/ca_tme_{gru,lstm,bilstm}/<model>_seed<seed>/train_metrics.json
  - M/N TME  : runs/mn_tme_{e2e,frozen}/<model>_seed<seed>/metrics.json
  - SP-MLP   : runs/sp_mlp/<model>_seed<seed>/mn_metrics.json (+pretrain_metrics.json)
  - T-LSTM   : runs/t_lstm/<model>_seed<seed>/mn_metrics.json (+pretrain_metrics.json)

C/A metrics expose best_val_balanced_accuracy_ac (no test split; cross-attn supervised).
M/N + SP-MLP + T-LSTM expose best_metrics.test_mn_acc (and best_test_mn_acc top-level).

Checks:
  1. Expected cell counts (13 models x 3 seeds = 39 for ca_tme_*, mn_tme_*; 16x3=48 for sp_mlp/t_lstm).
  2. NaN/inf in core metric columns.
  3. Cross-seed std < STD_THRESHOLD per (encoder, model).
  4. C/A and M/N models agree with the canonical 13-model list.
  5. (Optional) InternVL within 2sigma of other VT models per encoder.

Emits outputs/cache_matrix_20260722/_summary/audit_report.md.
"""
from __future__ import annotations

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

# 13-model canonical list (newer cache_matrix_20260722 drop). SP-MLP / T-LSTM
# still cover 16 (legacy 3 VA extra); we don't enforce 13 on those.
CANONICAL_MODELS = [
    "gemma3_12b", "gemma3_4b", "gemma4_12b", "glm4_6v_flash",
    "llava_onevision_qwen2_7b", "minicpm_v_2_6", "minicpm_v_4_5",
    "internvl3_5_8b", "qwen2_5_omni_7b", "qwen2_5_vl_7b",
    "qwen3_5_4b", "qwen3_5_9b", "qwen3_vl_8b",
]
SEEDS = (20260717, 20260718, 20260719)

# (encoder_name, dir_name, metrics_file, primary_key, primary_kind, expected_count)
# primary_kind: "val_ac" -> top-level best_val_balanced_accuracy_ac (C/A)
#               "test_mn" -> best_metrics.test_mn_acc with best_test_mn_acc fallback
ENCODER_SPECS = [
    ("ca_tme_gru",    "ca_tme_gru",    "train_metrics.json", "best_val_balanced_accuracy_ac", "val_ac",  39),
    ("ca_tme_lstm",   "ca_tme_lstm",   "train_metrics.json", "best_val_balanced_accuracy_ac", "val_ac",  39),
    ("ca_tme_bilstm", "ca_tme_bilstm", "train_metrics.json", "best_val_balanced_accuracy_ac", "val_ac",  39),
    ("mn_tme_e2e",    "mn_tme_e2e",    "metrics.json",       "best_metrics.test_mn_acc",      "test_mn", 39),
    ("mn_tme_frozen", "mn_tme_frozen", "metrics.json",       "best_metrics.test_mn_acc",      "test_mn", 39),
    ("sp_mlp",        "sp_mlp",        "mn_metrics.json",    "best_test_mn_acc",              "test_mn_top", 48),
    ("t_lstm",        "t_lstm",        "mn_metrics.json",    "best_test_mn_acc",              "test_mn_top", 48),
]

STD_THRESHOLD = 0.05


def _is_finite(value) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(f) or math.isinf(f))


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _extract_primary(metrics: dict, kind: str) -> float | None:
    if metrics is None:
        return None
    if kind == "val_ac":
        v = metrics.get("best_val_balanced_accuracy_ac")
        return float(v) if v is not None else None
    if kind == "test_mn":
        best = metrics.get("best_metrics") or {}
        v = best.get("test_mn_acc")
        if v is None:
            v = metrics.get("best_test_mn_acc")
        return float(v) if v is not None else None
    if kind == "test_mn_top":
        v = metrics.get("best_test_mn_acc")
        if v is None:
            best = metrics.get("best_metrics") or {}
            v = best.get("test_mn_acc")
        return float(v) if v is not None else None
    return None


def _scan_encoder(enc_name: str, dir_name: str, mfile: str, primary_key: str, kind: str) -> list[dict]:
    enc_dir = RUNS_DIR / dir_name
    rows: list[dict] = []
    if not enc_dir.exists():
        return rows
    for run_dir in sorted(enc_dir.iterdir()):
        if not run_dir.is_dir() or "_seed" not in run_dir.name:
            continue
        model, _, seed = run_dir.name.partition("_seed")
        if not seed:
            continue
        metrics = _load_json(run_dir / mfile)
        rows.append({
            "encoder": enc_name,
            "model": model,
            "seed": int(seed),
            "primary": _extract_primary(metrics, kind),
            "metrics_loaded": metrics is not None,
            "run_dir": str(run_dir.relative_to(ROOT)),
        })
    return rows


def _check_count(rows: list[dict], expected: int, enc_name: str, problems: list[str], ok: list[str]):
    actual = len(rows)
    if actual == expected:
        ok.append(f"count[{enc_name}] = {actual}/{expected}")
    else:
        problems.append(f"count[{enc_name}] = {actual}/{expected} (SHORT)")


def _check_missing_metrics(rows: list[dict], enc_name: str, problems: list[str], ok: list[str]):
    missing = [r for r in rows if not r["metrics_loaded"]]
    if missing:
        problems.append(
            f"missing/unparseable metrics in {enc_name} ({len(missing)}): "
            + ", ".join(f"{r['model']}/seed{r['seed']}" for r in missing[:5])
        )
    else:
        ok.append(f"all {enc_name} metrics.json loaded ({len(rows)} cells)")


def _check_finite(rows: list[dict], enc_name: str, problems: list[str], ok: list[str]):
    bad = [r for r in rows if r["primary"] is not None and not _is_finite(r["primary"])]
    if bad:
        problems.append(
            f"non-finite primary metric in {enc_name} ({len(bad)}): "
            + ", ".join(f"{r['model']}/seed{r['seed']}={r['primary']}" for r in bad[:5])
        )
    else:
        n_with = sum(1 for r in rows if r["primary"] is not None)
        ok.append(f"{enc_name} primary metrics finite ({n_with}/{len(rows)})")


def _check_std(rows: list[dict], enc_name: str, problems: list[str], ok: list[str]):
    by_cell: dict[str, list[float]] = {}
    for r in rows:
        if r["primary"] is None:
            continue
        by_cell.setdefault(r["model"], []).append(float(r["primary"]))
    bad = []
    covered = 0
    for model, values in sorted(by_cell.items()):
        if len(values) < 2:
            continue
        covered += 1
        std = statistics.stdev(values)
        if std >= STD_THRESHOLD:
            bad.append(f"{model}: std={std:.4f}")
    if bad:
        problems.append(
            f"cross-seed std >= {STD_THRESHOLD} in {enc_name} ({len(bad)} cells): "
            + "; ".join(bad[:10])
        )
    else:
        ok.append(f"{enc_name} cross-seed std < {STD_THRESHOLD} ({covered} cells checked)")


def _check_models(rows: list[dict], enc_name: str, problems: list[str], ok: list[str]):
    seen = {r["model"] for r in rows}
    expected = set(CANONICAL_MODELS)
    missing = expected - seen
    extra = seen - expected
    if missing:
        problems.append(f"{enc_name} missing canonical models: {sorted(missing)}")
    elif extra:
        ok.append(f"{enc_name} has {len(seen)} models (extra: {sorted(extra)})")
    else:
        ok.append(f"{enc_name} canonical 13-model set present")


def _check_internvl(rows: list[dict], enc_name: str, problems: list[str], ok: list[str]):
    by_model: dict[str, list[float]] = {}
    for r in rows:
        if r["primary"] is None:
            continue
        by_model.setdefault(r["model"], []).append(float(r["primary"]))
    internvl = by_model.get("internvl3_5_8b", [])
    others = [v for m, vs in by_model.items() if m != "internvl3_5_8b" for v in vs]
    if not internvl or not others or len(others) < 2:
        return
    mu = statistics.mean(others)
    sigma = statistics.stdev(others)
    ivl = statistics.mean(internvl)
    if sigma > 0 and abs(ivl - mu) > 2 * sigma:
        problems.append(
            f"InternVL outlier in {enc_name}: mean={ivl:.4f}, others mu={mu:.4f} sigma={sigma:.4f} "
            f"(delta={abs(ivl-mu)/sigma:.2f}sigma)"
        )
    else:
        ok.append(
            f"{enc_name} InternVL within 2sigma (delta/sigma="
            f"{'n/a' if sigma==0 else f'{abs(ivl-mu)/sigma:.2f}'})"
        )


def _check_sdr(problems: list[str], ok: list[str]):
    sdr_complete = 0
    sdr_missing_thresholds = 0
    sdr_missing_encoder = 0
    sdr_empty = []
    for model in CANONICAL_MODELS:
        model_dir = SDR_DIR / model
        marker = model_dir / "MISSING_DEPENDENCY"
        if marker.exists():
            content = marker.read_text(encoding="utf-8").strip()
            if content == "MISSING_THRESHOLDS":
                sdr_missing_thresholds += 1
            elif content == "TME_BILSTM_ENCODER_MISSING":
                sdr_missing_encoder += 1
            continue
        candidates = list(model_dir.rglob("state_patterns.jsonl"))
        if not candidates:
            problems.append(f"SDR[{model}]: no state_patterns.jsonl under {model_dir}")
            continue
        sdr_complete += 1
        try:
            rows = [json.loads(line) for line in candidates[0].read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"SDR[{model}]: cannot parse {candidates[0]}: {exc}")
            continue
        if not rows:
            sdr_empty.append(model)
            continue
        modes = {r.get("state_pattern") or r.get("mode") or r.get("pattern") for r in rows}
        if len(modes) < 4:
            problems.append(f"SDR[{model}]: only {len(modes)} state modes covered: {sorted(m for m in modes if m)}")
        else:
            ok.append(f"SDR[{model}]: {len(rows)} rows, {len(modes)} modes")
    ok.append(
        f"SDR summary: complete={sdr_complete}, missing_thresholds={sdr_missing_thresholds}, "
        f"missing_encoder={sdr_missing_encoder}, empty={len(sdr_empty)}"
    )


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    ok: list[str] = []

    all_rows: list[dict] = []
    for enc_name, dir_name, mfile, pkey, kind, expected in ENCODER_SPECS:
        rows = _scan_encoder(enc_name, dir_name, mfile, pkey, kind)
        all_rows.extend(rows)
        _check_count(rows, expected, enc_name, problems, ok)
        _check_missing_metrics(rows, enc_name, problems, ok)
        _check_finite(rows, enc_name, problems, ok)
        _check_models(rows, enc_name, problems, ok)
        _check_std(rows, enc_name, problems, ok)
        _check_internvl(rows, enc_name, problems, ok)

    _check_sdr(problems, ok)

    status = "PASS" if not problems else "FAIL"
    lines = [
        "# cache_matrix_20260722 audit", "",
        f"## Status: {status}", "",
        "Encoders audited (directly from runs/):",
    ]
    for enc_name, dir_name, mfile, _, _, expected in ENCODER_SPECS:
        n = sum(1 for r in all_rows if r["encoder"] == enc_name)
        lines.append(f"  - {enc_name} (`runs/{dir_name}/*/{mfile}`): {n}/{expected}")
    lines += ["", "## OK", ""]
    lines += [f"- {x}" for x in ok]
    if problems:
        lines += ["", "## Problems", ""]
        lines += [f"- {x}" for x in problems]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"audit: {status} ({len(ok)} ok, {len(problems)} problems) -> {REPORT}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
