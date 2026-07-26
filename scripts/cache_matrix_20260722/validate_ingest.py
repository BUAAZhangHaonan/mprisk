#!/usr/bin/env python3
"""Validate cache_matrix_20260722 source manifests for all 16 models."""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/home/team/zhanghaonan/TAFFC/mprisk-v2")
CACHE_BASE = Path("/home/team/zhanghaonan/TAFFC/mprisk/outputs/cache_matrix_20260722")
LOG_DIR = REPO_ROOT / "outputs/cache_matrix_20260722/_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

VA_MODELS = {"gemma4_12b", "gemma4_12b_it", "phi4_multimodal", "qwen2_5_omni_7b"}
INTERNVL = "internvl3_5_8b"
EXPECTED_CONDS = {"M1", "M2", "M12"}

MODELS = [
    "gemma3_12b", "gemma3_4b", "gemma4_12b", "glm4_6v_flash",
    "llava_onevision_qwen2_7b", "llava_v1_5_7b", "minicpm_v_2_6",
    "minicpm_v_4_5", "internvl3_5_8b", "phi3_5_vision", "phi4_multimodal",
    "qwen2_5_omni_7b", "qwen2_5_vl_7b", "qwen3_5_4b", "qwen3_5_9b",
    "qwen3_vl_8b",
]


def manifest_path_for(model):
    if model == INTERNVL:
        return CACHE_BASE / "cache_manifests/internvl3_5_8b/manifest.jsonl"
    return CACHE_BASE / "source/{}/manifest.jsonl".format(model)


def validate(model):
    proto = "va" if model in VA_MODELS else "vt"
    path = manifest_path_for(model)
    summary = {
        "model": model,
        "manifest_path": str(path),
        "exists": path.exists(),
        "protocol": proto,
    }
    if not path.exists():
        summary["error"] = "manifest not found: {}".format(path)
        return summary

    n_entries = 0
    hidden_dims = set()
    layer_counts = set()
    conditions = set()
    prompts = set()
    sample_ids = set()
    protocols_seen = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_entries += 1
            hidden_dims.add(rec.get("hidden_dim"))
            layer_counts.add(rec.get("layer_count"))
            conditions.add(rec.get("condition"))
            prompts.add(rec.get("prompt_id"))
            sample_ids.add(rec.get("sample_id"))
            protocols_seen.add(rec.get("protocol"))

    summary.update({
        "n_entries": n_entries,
        "n_conditions": len(conditions),
        "conditions": sorted(c for c in conditions if c),
        "n_prompts": len(prompts),
        "prompts": sorted(p for p in prompts if p),
        "n_sample_ids": len(sample_ids),
        "hidden_dim": sorted(h for h in hidden_dims if h is not None),
        "layer_count": sorted(l for l in layer_counts if l is not None),
        "protocols_seen": sorted(p for p in protocols_seen if p),
    })

    expected = 46416 if proto == "va" else 45024
    summary["expected_entries"] = expected
    summary["entries_ok"] = (n_entries == expected)
    summary["conditions_ok"] = (conditions == EXPECTED_CONDS)
    summary["prompts_ok"] = (len(prompts) == 8)
    return summary


def main():
    print("{:<28} {:<6} {:<8} {:<8} {:<8} {:<5}".format("model", "proto", "entries", "hidden", "layers", "ok"))
    print("-" * 70)
    failures = 0
    for model in MODELS:
        s = validate(model)
        out = LOG_DIR / "ingest_{}.json".format(model)
        out.write_text(json.dumps(s, indent=2))
        ok = bool(s.get("entries_ok") and s.get("conditions_ok") and s.get("prompts_ok"))
        if not ok:
            failures += 1
        hd = s.get("hidden_dim", ["?"])
        lc = s.get("layer_count", ["?"])
        hd_str = hd[0] if isinstance(hd, list) and hd else "?"
        lc_str = lc[0] if isinstance(lc, list) and lc else "?"
        ne = s.get("n_entries", "?")
        pr = s.get("protocol", "?")
        ok_str = "OK" if ok else "FAIL"
        print("{:<28} {:<6} {:<8} {:<8} {:<8} {:<5}".format(model, pr, ne, hd_str, lc_str, ok_str))
    print("-" * 70)
    print("Done. {} models, {} failures.".format(len(MODELS), failures))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
