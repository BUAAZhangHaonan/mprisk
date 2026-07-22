#!/usr/bin/env python3
"""Run one Conflict and one Aligned Phi-4 VA prefill smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mprisk.models.phi4_mm import Phi4MmWrapper
from mprisk.models.qwen_omni import build_condition_request
from mprisk.prompts.template_bank import load_equiv_prompt_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-num-segments", type=int, default=2)
    parser.add_argument("--prompt-set", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = _select_rows(args.manifest)
    prompts = _load_prompts(args.prompt_set)
    wrapper = Phi4MmWrapper(
        model_key="phi4_multimodal",
        model_path=args.model_path,
        device=args.device,
        dtype="bfloat16",
        attn_implementation="eager",
        video_num_segments=args.video_num_segments,
    )
    results = []
    try:
        wrapper.load()
        for row in rows:
            for prompt_id, prompt_text, prompt_set_key in prompts:
                for condition in ("M1", "M2", "M12"):
                    request = build_condition_request(
                        sample_id=str(row["sample_id"]),
                        model_key="phi4_multimodal",
                        protocol="va",
                        condition=condition,
                        dataset_key=str(row["source_dataset"]),
                        split=str(row["split"]),
                        media_paths={
                            str(key): str(value)
                            for key, value in row["media_paths"].items()
                        },
                        transcript=None,
                        task_prompt=prompt_text,
                        prompt_set_key=prompt_set_key,
                        prompt_id=prompt_id,
                        joint_audio_mode="embedded_video",
                        video_fps=1.0,
                    )
                    result = wrapper.extract_prefill(request)
                    results.append(
                        {
                            "sample_id": row["sample_id"],
                            "sample_type": row["sample_type"],
                            "condition": condition,
                            "prompt_id": prompt_id,
                            "shape": [result.layer_count, result.hidden_dim],
                            "token_count": result.token_count,
                            "t0_token_index": result.t0_token_index,
                            "finite": bool(
                                result.trajectory.dtype.name == "float32"
                                and np.isfinite(result.trajectory).all()
                            ),
                            "elapsed_seconds": result.provenance["elapsed_seconds"],
                            "peak_gpu_memory_bytes": result.provenance[
                                "peak_gpu_memory_bytes"
                            ],
                            "input_mode": {"M1": 1, "M2": 2, "M12": 3}[condition],
                        }
                    )
    finally:
        wrapper.close()
    expected_tasks = 2 * len(prompts) * 3
    if len(results) != expected_tasks or {
        row["sample_type"] for row in results
    } != {"Conflict", "Aligned"}:
        raise RuntimeError("Phi-4 smoke did not cover two sample types and three conditions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema": "mprisk_phi4_real_smoke_v1",
                "model_path": str(args.model_path.resolve()),
                "manifest": str(args.manifest.resolve()),
                "prompt_set": None if args.prompt_set is None else str(args.prompt_set.resolve()),
                "video_num_segments": args.video_num_segments,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output": str(args.output), "tasks": len(results)}))
    return 0


def _select_rows(path: Path) -> list[dict]:
    selected = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_type = str(row.get("sample_type"))
            if sample_type in {"Conflict", "Aligned"} and sample_type not in selected:
                if all(Path(value).is_file() for value in row["media_paths"].values()):
                    selected[sample_type] = row
            if len(selected) == 2:
                break
    if set(selected) != {"Conflict", "Aligned"}:
        raise ValueError("Manifest must contain accessible Conflict and Aligned VA samples")
    return [selected["Conflict"], selected["Aligned"]]


def _load_prompts(path: Path | None) -> list[tuple[str, str, str]]:
    if path is None:
        return [
            (
                "canonical",
                "Based on the complete input, describe the person's overall emotional state "
                "in one concise sentence.",
                "phi4_real_smoke_v1",
            )
        ]
    prompt_set = load_equiv_prompt_set(path)
    if not prompt_set.active or prompt_set.protocol.lower() != "va":
        raise ValueError("Phi-4 formal smoke requires an active VA prompt set")
    templates = prompt_set.enabled_templates()
    if len(templates) != 8:
        raise ValueError("Phi-4 formal smoke requires exactly eight enabled prompts")
    return [
        (template.prompt_id, template.template_text, prompt_set.key)
        for template in templates
    ]


if __name__ == "__main__":
    raise SystemExit(main())
