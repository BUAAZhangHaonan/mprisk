#!/usr/bin/env python3
"""Generate the user's 16-model results table from misread outputs.

Reads outputs/v2/misread/<model>/judgments.jsonl for each model, computes:
- Aligned Accuracy: fraction of Aligned samples correctly judged as non-misread
- Conflict Accuracy: fraction of Conflict samples correctly judged as misread
- Preferred Modality: M1 if vision-dominated, M2 if text/audio-dominated, balanced otherwise

Outputs both CSV and markdown table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MISREAD_DIR = Path(__file__).resolve().parents[3] / "outputs/v2/misread"

# Display names matching user's table
DISPLAY = {
    "gemma3_4b":                "Gemma-3-4B",
    "gemma3_12b":               "Gemma-3-12B",
    "glm4_6v_flash":            "GLM-4.6V-Flash",
    "internvl3_5_8b":           "InternVL3.5-8B",
    "llava_v1_5_7b":            "LLaVA-v1.5-7B",
    "llava_onevision_qwen2_7b": "LLaVA-OneVision-7B",
    "minicpm_v_2_6":            "MiniCPM-V-2.6",
    "minicpm_v_4_5":            "MiniCPM-V-4.5",
    "phi3_5_vision":            "Phi-3.5-Vision",
    "qwen2_5_vl_7b":            "Qwen2.5-VL-7B",
    "qwen3_vl_8b":              "Qwen3-VL-8B",
    "qwen3_5_4b":               "Qwen3.5-4B",
    "qwen3_5_9b":               "Qwen3.5-9B",
    "gemma4_12b":               "Gemma-4-12B",
    "gemma4_12b_it":            "Gemma-4-12B-IT",
    "phi4_multimodal":          "Phi-4-Multimodal",
    "qwen2_5_omni_7b":          "Qwen2.5-Omni-7B",
}

# User-listed model order (16 models)
ORDER = [
    "gemma3_4b", "gemma3_12b", "glm4_6v_flash", "internvl3_5_8b",
    "llava_v1_5_7b", "llava_onevision_qwen2_7b", "minicpm_v_2_6",
    "minicpm_v_4_5", "phi3_5_vision", "qwen2_5_vl_7b", "qwen3_vl_8b",
    "qwen3_5_4b", "qwen3_5_9b", "gemma4_12b", "phi4_multimodal",
    "qwen2_5_omni_7b",
]


def compute_model_stats(model_key: str, misread_dir: Path | None = None) -> dict | None:
    misread_dir = misread_dir or MISREAD_DIR
    jpath = misread_dir / model_key / "judgments.jsonl"
    if not jpath.exists():
        return None
    rows = []
    with jpath.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    aligned_total = aligned_correct = 0
    conflict_total = conflict_correct = 0
    for r in rows:
        st = r.get("sample_type", "")
        final = r.get("final_label", r.get("label", "")).upper()
        if st == "Aligned":
            aligned_total += 1
            if final in ("NON_MISREAD", "NON-MISREAD"):
                aligned_correct += 1
        elif st == "Conflict":
            conflict_total += 1
            if final == "MISREAD":
                conflict_correct += 1

    aligned_acc = (aligned_correct / aligned_total * 100) if aligned_total else 0.0
    conflict_acc = (conflict_correct / conflict_total * 100) if conflict_total else 0.0

    # Preferred modality heuristic: equal accuracy → balanced; conflict >> aligned → M2 (text/audio dominates, model misjudges by following vision); aligned >> conflict → M1; otherwise unclear.
    diff = aligned_acc - conflict_acc
    if abs(diff) < 5:
        pref = "balanced"
    elif diff > 15:
        pref = "M1 (vision)"
    elif diff < -15:
        pref = "M2 (text/audio)"
    else:
        pref = "unclear"

    return {
        "model_key": model_key,
        "display": DISPLAY.get(model_key, model_key),
        "aligned_acc": aligned_acc,
        "aligned_total": aligned_total,
        "aligned_correct": aligned_correct,
        "conflict_acc": conflict_acc,
        "conflict_total": conflict_total,
        "conflict_correct": conflict_correct,
        "preferred_modality": pref,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--misread-dir",
        type=Path,
        default=MISREAD_DIR,
        help=f"Directory containing per-model misread subdirs (default: {MISREAD_DIR}).",
    )
    args = parser.parse_args()
    misread_dir: Path = args.misread_dir

    results = []
    for k in ORDER:
        s = compute_model_stats(k, misread_dir=misread_dir)
        if s is None:
            results.append({"model_key": k, "display": DISPLAY.get(k, k),
                            "aligned_acc": None, "conflict_acc": None,
                            "preferred_modality": "MISSING"})
        else:
            results.append(s)

    # CSV
    csv_path = misread_dir / "table_misread_accuracy.csv"
    with csv_path.open("w") as f:
        f.write("model,aligned_accuracy,conflict_accuracy,preferred_modality,aligned_n,conflict_n\n")
        for r in results:
            aa = f"{r['aligned_acc']:.2f}" if r.get('aligned_acc') is not None else "N/A"
            ca = f"{r['conflict_acc']:.2f}" if r.get('conflict_acc') is not None else "N/A"
            f.write(f"{r['display']},{aa},{ca},{r.get('preferred_modality','N/A')},"
                    f"{r.get('aligned_total','N/A')},{r.get('conflict_total','N/A')}\n")
    print(f"wrote {csv_path}")

    # Markdown
    md_path = misread_dir / "table_misread_accuracy.md"
    with md_path.open("w") as f:
        f.write("# Misread Accuracy by Model\n\n")
        f.write("| Model | Aligned Accuracy (%) | Conflict Accuracy (%) | Preferred Modality | N (Aligned/Conflict) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        valid_a = []
        valid_c = []
        for r in results:
            if r.get('aligned_acc') is not None:
                valid_a.append(r['aligned_acc'])
            if r.get('conflict_acc') is not None:
                valid_c.append(r['conflict_acc'])
            aa = f"{r['aligned_acc']:.2f}" if r.get('aligned_acc') is not None else "—"
            ca = f"{r['conflict_acc']:.2f}" if r.get('conflict_acc') is not None else "—"
            ns = f"{r.get('aligned_total','—')}/{r.get('conflict_total','—')}"
            f.write(f"| {r['display']} | {aa} | {ca} | {r.get('preferred_modality','—')} | {ns} |\n")
        if valid_a:
            avg_a = sum(valid_a) / len(valid_a)
            avg_c = sum(valid_c) / len(valid_c)
            f.write(f"| **Macro Average** | **{avg_a:.2f}** | **{avg_c:.2f}** | — | — |\n")
    print(f"wrote {md_path}")

    # Brief stdout summary
    print("\n=== Summary ===")
    for r in results:
        aa = f"{r['aligned_acc']:.1f}%" if r.get('aligned_acc') is not None else "MISSING"
        ca = f"{r['conflict_acc']:.1f}%" if r.get('conflict_acc') is not None else "MISSING"
        print(f"  {r['display']:<22}  aligned={aa:<8}  conflict={ca:<8}  {r.get('preferred_modality','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
