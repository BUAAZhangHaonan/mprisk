"""Generate model descriptions for M12 (full video) input + judge Misread.

Two-stage pipeline:
  Stage A: For each sample + subject model, run M12 generation with fixed prompt,
           save diagnostic_affect_description.
  Stage B: For each (sample, subject model), run DeepSeek 3x flash + pro arbitration
           to assign MISREAD / NON_MISREAD.

Stage A is GPU-bound, uses three model wrappers.
Stage B is API-bound, runs concurrently.

Usage:
  # Stage A (per model, GPU 0 or 1)
  CUDA_VISIBLE_DEVICES=0 python scripts/generate_misread_inputs.py \\
      --delivery-root /home/team/lvshuyang/prompt-make/delivery_20260716 \\
      --model-key qwen3_vl_8b \\
      --output outputs/v2/misread/descriptions/qwen3_vl_8b.jsonl

  # Stage B (CPU, parallel via httpx)
  python scripts/judge_misread.py \\
      --descriptions-glob 'outputs/v2/misread/descriptions/*.jsonl' \\
      --output outputs/v2/misread/labels.jsonl \\
      --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mprisk_viz.misread import judge_many  # noqa: E402


def load_descriptions(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--descriptions-glob", required=True,
        help="Glob pattern for stage-A description JSONL files.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--n-flash", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--flash-model", default="deepseek-v4-flash")
    parser.add_argument("--pro-model", default="deepseek-v4-pro")
    args = parser.parse_args()

    import glob

    paths = sorted(glob.glob(args.descriptions_glob))
    if not paths:
        print(f"no description files match {args.descriptions_glob}", file=sys.stderr)
        return 1
    print(f"[v2-misread] loading {len(paths)} description files", flush=True)

    tasks: list[dict] = []
    for p in paths:
        for row in load_descriptions(p):
            if not row.get("gt_describe") or not row.get("diagnostic_description"):
                continue
            tasks.append({
                "sample_id": row["sample_id"],
                "subject_model_key": row["subject_model_key"],
                "protocol": row["protocol"],
                "gt_description": row["gt_describe"],
                "diagnostic_description": row["diagnostic_description"],
            })
    print(f"[v2-misread] {len(tasks)} tasks queued", flush=True)

    out_path = asyncio.run(judge_many(
        tasks=tasks,
        output_path=args.output,
        max_concurrency=args.concurrency,
        flash_model=args.flash_model,
        pro_model=args.pro_model,
        n_flash=args.n_flash,
        confidence_threshold=args.confidence_threshold,
    ))
    print(f"[v2-misread] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
