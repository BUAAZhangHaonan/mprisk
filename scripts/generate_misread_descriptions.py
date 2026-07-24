"""V2 Misread description generator: M12 (full video) + canonical prompt.

For each sample, runs the subject model on the full video input + canonical
emotion description prompt, and saves the response. Output is consumed by
scripts/judge_misread.py.

Supports three model families via direct transformers calls (the mprisk
wrappers only implement extract_prefill, not generate, for qwen_vl/internvl).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from tqdm import tqdm

CANONICAL_PROMPT = (
    "Based on the complete input, describe the person's overall emotional state "
    "in one concise sentence. Do not address the person, give advice, or explain "
    "your reasoning."
)

MODEL_PATHS = {
    "qwen3_vl_8b":      "/home/team/lvshuyang/Models/Qwen3-VL-8B-Instruct",
    "internvl3_5_8b":   "/home/team/lvshuyang/Models/InternVL3_5-8B",
    "qwen2_5_omni_7b":  "/home/team/lvshuyang/Models/Qwen/Qwen2.5-Omni-7B",
}

MODEL_FAMILIES = {
    "qwen3_vl_8b":     "qwen_vl",
    "internvl3_5_8b":  "internvl",
    "qwen2_5_omni_7b": "qwen_omni",
}


@dataclass(frozen=True)
class DeliverySample:
    sample_id: str
    source_id: str
    protocol: str           # "VT" or "VA"
    sample_type: str        # "Conflict" or "Aligned"
    media_paths: dict[str, str]
    text_content: str
    gt_emotion: str
    surface_emotion: str | None
    gt_describe: str        # the GT 4-segment description


def load_delivery(delivery_root: str | Path) -> list[DeliverySample]:
    """Load all four manifests from delivery_20260716."""
    root = Path(delivery_root)
    samples: list[DeliverySample] = []
    for fname in ("vt_a_manifest.jsonl", "vt_c_manifest.jsonl",
                  "va_a_manifest.jsonl", "va_c_manifest.jsonl"):
        path = root / fname
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                samples.append(DeliverySample(
                    sample_id=row["sample_id"],
                    source_id=str(row.get("source_id", "")),
                    protocol=str(row["protocol"]),
                    sample_type=str(row["sample_type"]),
                    media_paths=dict(row.get("media_paths", {})),
                    text_content=str(row.get("text_content", "")),
                    gt_emotion=str(row.get("gt_emotion", "")),
                    surface_emotion=row.get("surface_emotion"),
                    gt_describe=str(row.get("gt_describe", "")),
                ))
    return samples


def _build_messages_qwen_vl(sample: DeliverySample) -> list[dict]:
    video_path = sample.media_paths.get("vision", "")
    text = sample.text_content
    if sample.protocol == "VT":
        user_content = [
            {"type": "video", "video": f"file://{video_path}", "fps": 1.0},
            {"type": "text", "text": f"Text: {text}\n\n{CANONICAL_PROMPT}"},
        ]
    else:  # VA — video already carries the speech
        user_content = [
            {"type": "video", "video": f"file://{video_path}", "fps": 1.0},
            {"type": "text", "text": CANONICAL_PROMPT},
        ]
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]


def _build_messages_internvl(sample: DeliverySample) -> tuple[str, list[Any]]:
    """InternVL accepts a question string + video path; returns (question, video_list)."""
    video_path = sample.media_paths.get("vision", "")
    text = sample.text_content
    if sample.protocol == "VT":
        question = (
            f"<video>\n"
            f"Text: {text}\n\n"
            f"{CANONICAL_PROMPT}"
        )
    else:
        question = f"<video>\n{CANONICAL_PROMPT}"
    return question, [video_path]


def _build_messages_qwen_omni(sample: DeliverySample) -> tuple[list[dict], bool]:
    video_path = sample.media_paths.get("vision", "")
    text = sample.text_content
    use_audio = (sample.protocol == "VA")
    if sample.protocol == "VT":
        content = [
            {"type": "video", "video": f"file://{video_path}", "fps": 1.0},
            {"type": "text", "text": f"Text: {text}\n\n{CANONICAL_PROMPT}"},
        ]
    else:
        content = [
            {"type": "video", "video": f"file://{video_path}", "fps": 1.0,
             "audio": True},
            {"type": "text", "text": CANONICAL_PROMPT},
        ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]
    return messages, use_audio


def _load_qwen_vl(model_path: str, device: str):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return model, processor


def _load_internvl(model_path: str, device: str):
    from transformers import AutoModel, AutoTokenizer
    from mprisk.models.internvl import load_image, load_video  # reuse helpers
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map=device, low_cpu_mem_usage=True, use_flash_attn=False
    ).eval()
    return model, tokenizer


def _load_qwen_omni(model_path: str, device: str):
    from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    return model, processor


def _generate_qwen_vl(model, processor, sample: DeliverySample, max_new_tokens: int) -> str:
    messages = _build_messages_qwen_vl(sample)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def _generate_internvl(model, tokenizer, sample: DeliverySample, max_new_tokens: int,
                       video_fps: float = 1.0) -> str:
    from mprisk.models.internvl import load_video
    question, video_list = _build_messages_internvl(sample)
    video_path = video_list[0] if video_list else ""
    frames = load_video(video_path, fps=video_fps)
    pixel_values = torch.stack(frames).to(model.device, dtype=torch.bfloat16)
    n_frames = len(frames)
    response, _ = model.chat(
        tokenizer=tokenizer,
        question=question,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_patches_list=[n_frames],
        history=None,
        return_history=True,
    )
    return response.strip()


def _generate_qwen_omni(model, processor, sample: DeliverySample, max_new_tokens: int) -> str:
    messages, use_audio = _build_messages_qwen_omni(sample)
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = processor.process_mm_info(
        messages, use_audio_in_video=use_audio
    )
    inputs = processor(
        text=[chat], audio=audios, images=images, videos=videos, padding=True,
        return_tensors="pt", use_audio_in_video=use_audio,
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_audio_in_video=use_audio,
        )
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def generate_descriptions(
    *,
    model_key: str,
    delivery_root: str | Path,
    output_path: str | Path,
    device: str = "cuda:0",
    max_new_tokens: int = 64,
    limit: int | None = None,
    skip_existing: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_PATHS[model_key]
    family = MODEL_FAMILIES[model_key]

    print(f"[gen-desc] loading {model_key} ({family}) from {model_path}", flush=True)
    if family == "qwen_vl":
        model, processor = _load_qwen_vl(model_path, device)
    elif family == "internvl":
        model, processor = _load_internvl(model_path, device)
    elif family == "qwen_omni":
        model, processor = _load_qwen_omni(model_path, device)
    else:
        raise ValueError(f"unknown family {family}")

    samples = load_delivery(delivery_root)
    print(f"[gen-desc] loaded {len(samples)} delivery samples", flush=True)

    existing: set[str] = set()
    if skip_existing and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    existing.add(json.loads(line)["sample_id"])
                except Exception:
                    pass
        print(f"[gen-desc] resuming — {len(existing)} already done", flush=True)

    target = samples[:limit] if limit else samples
    out_f = output_path.open("a" if skip_existing else "w", encoding="utf-8")
    started = time.perf_counter()
    done = 0
    errors = 0

    with out_f:
        for s in tqdm(target, desc=model_key):
            if s.sample_id in existing:
                done += 1
                continue
            try:
                if family == "qwen_vl":
                    text = _generate_qwen_vl(model, processor, s, max_new_tokens)
                elif family == "internvl":
                    text = _generate_internvl(model, processor, s, max_new_tokens)
                else:
                    text = _generate_qwen_omni(model, processor, s, max_new_tokens)
                row = {
                    "schema": "mprisk_v2_diagnostic_description_v1",
                    "sample_id": s.sample_id,
                    "subject_model_key": model_key,
                    "protocol": s.protocol,
                    "sample_type": s.sample_type,
                    "source_id": s.source_id,
                    "gt_emotion": s.gt_emotion,
                    "gt_describe": s.gt_describe,
                    "diagnostic_description": text,
                    "max_new_tokens": max_new_tokens,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                out_f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                out_f.flush()
                done += 1
            except Exception as exc:
                errors += 1
                print(f"[gen-desc] {s.sample_id} failed: {type(exc).__name__}: {exc}",
                      flush=True)
    print(f"[gen-desc] done model={model_key} done={done} errors={errors}", flush=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True,
                        choices=list(MODEL_PATHS.keys()))
    parser.add_argument("--delivery-root",
                        default="/home/team/lvshuyang/prompt-make/delivery_20260716")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()
    generate_descriptions(
        model_key=args.model_key,
        delivery_root=args.delivery_root,
        output_path=args.output,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
