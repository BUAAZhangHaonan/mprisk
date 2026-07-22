"""Immutable, processor-only context plans for LLaVA-v1.5 video simulation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mprisk.models.qwen_omni import build_condition_request
from mprisk.prompts.compiler import compile_prompt
from mprisk.prompts.template_bank import load_equiv_prompt_set

FRAME_PLAN_SCHEMA = "mprisk_llava_v15_frame_plan_v1"
CONTEXT_BUDGET_SCHEMA = "mprisk_llava_v15_context_budget_contract_v1"
FRAME_SELECTION_SCHEMA = "mprisk_llava_v15_shared_frame_selection_v1"
CONTEXT_BUDGET_MODE = "per_sample_shared_max_legal"
FRAME_PROTOCOL = "per_sample_shared_uniform_temporal_samples_v1"
SAMPLING_METHOD = "uniform_midpoint_decord_v1"
SELECTION_CONDITIONS = ("M1", "M12")


def build_frame_plan(
    *,
    manifest_path: str | Path,
    prompt_set_path: str | Path,
    model_path: str | Path,
    model_key: str,
    max_candidate_frames: int = 8,
) -> dict[str, Any]:
    manifest = Path(manifest_path).expanduser().resolve()
    prompt_path = Path(prompt_set_path).expanduser().resolve()
    checkpoint = Path(model_path).expanduser().resolve()
    if max_candidate_frames != 8:
        raise ValueError("LLaVA-v1.5 frame planning requires max_candidate_frames=8")
    rows = _read_jsonl(manifest)
    if len({str(row.get("sample_id")) for row in rows}) != len(rows):
        raise ValueError("Frame-plan manifest sample IDs must be unique")
    prompt_set = load_equiv_prompt_set(prompt_path)
    if not prompt_set.active or prompt_set.protocol.lower() != "vt":
        raise ValueError("LLaVA-v1.5 frame planning requires an active VT prompt set")
    templates = prompt_set.enabled_templates()
    prompt_ids = [template.prompt_id for template in templates]
    if len(prompt_ids) != 8 or len(set(prompt_ids)) != 8:
        raise ValueError("LLaVA-v1.5 frame planning requires exactly eight prompt IDs")
    prompt_ids_sha256 = _sha256_json(prompt_ids)
    config_path = checkpoint / "config.json"
    config = _read_json(config_path)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("LLaVA-v1.5 checkpoint has no text_config")
    max_position_embeddings = text_config.get("max_position_embeddings")
    if (
        not isinstance(max_position_embeddings, int)
        or isinstance(max_position_embeddings, bool)
        or max_position_embeddings <= 0
    ):
        raise ValueError("LLaVA-v1.5 max_position_embeddings must be positive")

    from transformers import LlavaProcessor

    processor = LlavaProcessor.from_pretrained(checkpoint, local_files_only=True)
    image_tokens_per_frame = _image_tokens_per_frame(processor)
    entries = [
        _plan_sample(
            row=row,
            templates=templates,
            prompt_set_key=prompt_set.key,
            prompt_ids=prompt_ids,
            prompt_ids_sha256=prompt_ids_sha256,
            processor=processor,
            image_tokens_per_frame=image_tokens_per_frame,
            model_key=model_key,
            max_candidate_frames=max_candidate_frames,
            max_position_embeddings=max_position_embeddings,
        )
        for row in rows
    ]
    return {
        "schema": FRAME_PLAN_SCHEMA,
        "model_key": model_key,
        "family": "llava_v15",
        "context_budget_mode": CONTEXT_BUDGET_MODE,
        "frame_protocol": FRAME_PROTOCOL,
        "sampling_method": SAMPLING_METHOD,
        "max_candidate_frames": max_candidate_frames,
        "max_position_embeddings": max_position_embeddings,
        "no_truncation": True,
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "prompt_set_path": str(prompt_path),
        "prompt_set_sha256": _sha256(prompt_path),
        "prompt_set_key": prompt_set.key,
        "prompt_ids": prompt_ids,
        "prompt_ids_sha256": prompt_ids_sha256,
        "model_path": str(checkpoint),
        "model_config_sha256": _sha256(config_path),
        "processor_class": type(processor).__name__,
        "image_tokens_per_frame": image_tokens_per_frame,
        "entries": entries,
    }


def write_frame_plan(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    validate_frame_plan(payload)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if not destination.is_file():
            raise FileExistsError(f"Frame-plan destination is not a file: {destination}")
        if destination.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(
                f"Refusing to replace immutable frame plan: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale frame-plan temporary exists: {temporary}")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def load_frame_plan(path: str | Path) -> dict[str, Any]:
    payload = _read_json(Path(path).expanduser().resolve())
    validate_frame_plan(payload)
    return payload


def index_frame_plan(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validate_frame_plan(payload)
    return {str(entry["sample_id"]): entry for entry in payload["entries"]}


def validate_frame_plan(payload: dict[str, Any]) -> None:
    if payload.get("schema") != FRAME_PLAN_SCHEMA:
        raise ValueError(f"Frame plan schema must be {FRAME_PLAN_SCHEMA}")
    if payload.get("family") != "llava_v15":
        raise ValueError("Frame plan family must be llava_v15")
    if payload.get("context_budget_mode") != CONTEXT_BUDGET_MODE:
        raise ValueError("Frame plan context-budget mode mismatch")
    if payload.get("frame_protocol") != FRAME_PROTOCOL:
        raise ValueError("Frame plan frame protocol mismatch")
    if payload.get("max_candidate_frames") != 8:
        raise ValueError("Frame plan max_candidate_frames must be 8")
    context_limit = payload.get("max_position_embeddings")
    if not isinstance(context_limit, int) or isinstance(context_limit, bool):
        raise ValueError("Frame plan max_position_embeddings must be an integer")
    if payload.get("no_truncation") is not True:
        raise ValueError("Frame plan must prohibit truncation")
    prompt_ids = payload.get("prompt_ids")
    if (
        not isinstance(prompt_ids, list)
        or len(prompt_ids) != 8
        or len(set(prompt_ids)) != 8
        or payload.get("prompt_ids_sha256") != _sha256_json(prompt_ids)
    ):
        raise ValueError("Frame plan prompt identity is invalid")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Frame plan entries must be a non-empty list")
    sample_ids = [str(entry.get("sample_id")) for entry in entries]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Frame plan sample IDs must be unique")
    for entry in entries:
        _validate_entry(
            entry,
            context_limit=context_limit,
            prompt_ids=prompt_ids,
            prompt_ids_sha256=str(payload["prompt_ids_sha256"]),
        )


def _plan_sample(
    *,
    row: dict[str, Any],
    templates: list[Any],
    prompt_set_key: str,
    prompt_ids: list[str],
    prompt_ids_sha256: str,
    processor: Any,
    image_tokens_per_frame: int,
    model_key: str,
    max_candidate_frames: int,
    max_position_embeddings: int,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or "")
    if not sample_id:
        raise ValueError("Frame-plan row has no sample_id")
    media = row.get("media_paths")
    if not isinstance(media, dict) or not isinstance(media.get("vision"), str):
        raise ValueError(f"Frame-plan sample {sample_id} has no vision path")
    video_path = Path(media["vision"]).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    source_total_frames = _video_frame_count(video_path)
    if source_total_frames < max_candidate_frames:
        raise ValueError(
            f"LLaVA-v1.5 sample {sample_id} has only {source_total_frames} frames"
        )
    transcript = "" if row.get("text_content") is None else str(row["text_content"])
    prompts = [
        (
            template.prompt_id,
            compile_prompt(template, {"sample_text": transcript}),
        )
        for template in templates
    ]
    request_texts: dict[str, list[str]] = {condition: [] for condition in SELECTION_CONDITIONS}
    for prompt_id, task_prompt in prompts:
        for condition in SELECTION_CONDITIONS:
            request = build_condition_request(
                sample_id=sample_id,
                model_key=model_key,
                protocol="vt",
                condition=condition,
                dataset_key=str(row.get("source_dataset") or "unknown"),
                split=str(row.get("split") or "all"),
                media_paths={str(key): str(value) for key, value in media.items()},
                transcript=transcript,
                task_prompt=task_prompt,
                prompt_set_key=prompt_set_key,
                prompt_id=prompt_id,
            )
            request_texts[condition].append(_request_text(request.messages))

    candidate_condition_max: dict[str, dict[str, int]] = {}
    candidate_max: dict[str, int] = {}
    for frames in range(1, max_candidate_frames + 1):
        condition_maxima = {
            condition: max(
                _count_tokens(
                    processor,
                    text=text,
                    frames=frames,
                    image_tokens_per_frame=image_tokens_per_frame,
                )
                for text in request_texts[condition]
            )
            for condition in SELECTION_CONDITIONS
        }
        key = str(frames)
        candidate_condition_max[key] = condition_maxima
        candidate_max[key] = max(condition_maxima.values())
    legal = [
        frames
        for frames in range(1, max_candidate_frames + 1)
        if candidate_max[str(frames)] <= max_position_embeddings
    ]
    if not legal:
        raise ValueError(f"LLaVA-v1.5 sample {sample_id} has no legal frame count")
    selected = max(legal)
    indices = _uniform_midpoint_indices(source_total_frames, selected)
    context_contract = {
        "schema": CONTEXT_BUDGET_SCHEMA,
        "sample_id": sample_id,
        "mode": CONTEXT_BUDGET_MODE,
        "max_position_embeddings": max_position_embeddings,
        "max_candidate_frames": max_candidate_frames,
        "selected_frames": selected,
        "conditions": list(SELECTION_CONDITIONS),
        "prompt_set_key": prompt_set_key,
        "prompt_ids": prompt_ids,
        "prompt_ids_sha256": prompt_ids_sha256,
        "candidate_max_token_counts": candidate_max,
        "candidate_condition_max_token_counts": candidate_condition_max,
        "selected_max_token_count": candidate_max[str(selected)],
        "selection_rule": "largest_f_with_all_p8_m1_m12_tokens_lte_context",
        "no_truncation": True,
    }
    frame_contract = {
        "schema": FRAME_SELECTION_SCHEMA,
        "sample_id": sample_id,
        "sampling_method": SAMPLING_METHOD,
        "video_path": str(video_path),
        "source_total_frames": source_total_frames,
        "selected_frames": selected,
        "frame_indices": indices,
        "frame_indices_sha256": _sha256_json(indices),
        "shared_conditions": list(SELECTION_CONDITIONS),
        "prompt_ids_sha256": prompt_ids_sha256,
    }
    entry = {
        "sample_id": sample_id,
        "context_budget_contract": context_contract,
        "frame_selection_contract": frame_contract,
    }
    _validate_entry(
        entry,
        context_limit=max_position_embeddings,
        prompt_ids=prompt_ids,
        prompt_ids_sha256=prompt_ids_sha256,
    )
    return entry


def _validate_entry(
    entry: dict[str, Any],
    *,
    context_limit: int,
    prompt_ids: list[str],
    prompt_ids_sha256: str,
) -> None:
    sample_id = str(entry.get("sample_id") or "")
    context = entry.get("context_budget_contract")
    frames = entry.get("frame_selection_contract")
    if not sample_id or not isinstance(context, dict) or not isinstance(frames, dict):
        raise ValueError("Frame-plan entry is incomplete")
    if context.get("schema") != CONTEXT_BUDGET_SCHEMA:
        raise ValueError(f"Invalid context contract for {sample_id}")
    if frames.get("schema") != FRAME_SELECTION_SCHEMA:
        raise ValueError(f"Invalid frame-selection contract for {sample_id}")
    if context.get("sample_id") != sample_id or frames.get("sample_id") != sample_id:
        raise ValueError(f"Frame-plan sample binding mismatch for {sample_id}")
    if context.get("mode") != CONTEXT_BUDGET_MODE:
        raise ValueError(f"Invalid context-budget mode for {sample_id}")
    if context.get("max_position_embeddings") != context_limit:
        raise ValueError(f"Context limit mismatch for {sample_id}")
    if context.get("max_candidate_frames") != 8:
        raise ValueError(f"Max candidate frames mismatch for {sample_id}")
    if context.get("conditions") != list(SELECTION_CONDITIONS):
        raise ValueError(f"Selection conditions mismatch for {sample_id}")
    if context.get("prompt_ids") != prompt_ids:
        raise ValueError(f"Prompt IDs mismatch for {sample_id}")
    if context.get("prompt_ids_sha256") != prompt_ids_sha256:
        raise ValueError(f"Prompt SHA mismatch for {sample_id}")
    if context.get("no_truncation") is not True:
        raise ValueError(f"Truncation must be disabled for {sample_id}")
    candidate_max = context.get("candidate_max_token_counts")
    candidate_by_condition = context.get("candidate_condition_max_token_counts")
    expected_keys = {str(value) for value in range(1, 9)}
    if not isinstance(candidate_max, dict) or set(candidate_max) != expected_keys:
        raise ValueError(f"Candidate token maxima are incomplete for {sample_id}")
    if (
        not isinstance(candidate_by_condition, dict)
        or set(candidate_by_condition) != expected_keys
    ):
        raise ValueError(f"Candidate condition maxima are incomplete for {sample_id}")
    for key in expected_keys:
        value = candidate_max[key]
        conditions = candidate_by_condition[key]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not isinstance(conditions, dict)
            or set(conditions) != set(SELECTION_CONDITIONS)
            or any(
                not isinstance(item, int) or isinstance(item, bool)
                for item in conditions.values()
            )
            or value != max(conditions.values())
        ):
            raise ValueError(f"Invalid candidate token maximum for {sample_id}/F{key}")
    selected = context.get("selected_frames")
    if (
        not isinstance(selected, int)
        or isinstance(selected, bool)
        or not 1 <= selected <= 8
    ):
        raise ValueError(f"Invalid selected frame count for {sample_id}")
    if candidate_max[str(selected)] > context_limit:
        raise ValueError(f"Selected frame count exceeds context for {sample_id}")
    if selected < 8 and candidate_max[str(selected + 1)] <= context_limit:
        raise ValueError(f"Selected frame count is not maximal for {sample_id}")
    if context.get("selected_max_token_count") != candidate_max[str(selected)]:
        raise ValueError(f"Selected token maximum mismatch for {sample_id}")
    if context.get("selection_rule") != "largest_f_with_all_p8_m1_m12_tokens_lte_context":
        raise ValueError(f"Selection rule mismatch for {sample_id}")
    if frames.get("sampling_method") != SAMPLING_METHOD:
        raise ValueError(f"Sampling method mismatch for {sample_id}")
    video_path = frames.get("video_path")
    if (
        not isinstance(video_path, str)
        or not video_path
        or not Path(video_path).is_absolute()
    ):
        raise ValueError(f"Video path must be absolute for {sample_id}")
    if frames.get("selected_frames") != selected:
        raise ValueError(f"Frame contracts disagree for {sample_id}")
    if frames.get("shared_conditions") != list(SELECTION_CONDITIONS):
        raise ValueError(f"Shared conditions mismatch for {sample_id}")
    if frames.get("prompt_ids_sha256") != prompt_ids_sha256:
        raise ValueError(f"Frame prompt SHA mismatch for {sample_id}")
    source_total = frames.get("source_total_frames")
    indices = frames.get("frame_indices")
    if (
        not isinstance(source_total, int)
        or isinstance(source_total, bool)
        or source_total < 8
        or not isinstance(indices, list)
        or len(indices) != selected
        or indices != sorted(set(indices))
        or any(
            not isinstance(index, int) or index < 0 or index >= source_total
            for index in indices
        )
        or frames.get("frame_indices_sha256") != _sha256_json(indices)
    ):
        raise ValueError(f"Frame indices are invalid for {sample_id}")
    if indices != _uniform_midpoint_indices(source_total, selected):
        raise ValueError(f"Frame indices are not uniform midpoints for {sample_id}")


def _image_tokens_per_frame(processor: Any) -> int:
    image_processor = processor.image_processor
    crop = getattr(image_processor, "crop_size", None)
    height = _size_component(crop, "height")
    width = _size_component(crop, "width")
    if height is None or height != width:
        raise ValueError("LLaVA-v1.5 requires a fixed square crop")
    size = height
    patch_size = int(processor.patch_size)
    if size <= 0 or patch_size <= 0 or size % patch_size:
        raise ValueError("LLaVA-v1.5 image-token geometry is invalid")
    value = (size // patch_size) ** 2 + int(processor.num_additional_image_tokens)
    if processor.vision_feature_select_strategy == "default":
        value -= 1
    if value != 576:
        raise ValueError(f"Unexpected LLaVA-v1.5 image tokens per frame: {value}")
    return value


def _size_component(value: Any, key: str) -> int | None:
    raw = value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        return None
    return raw


def _count_tokens(
    processor: Any, *, text: str, frames: int, image_tokens_per_frame: int
) -> int:
    content = [{"type": "image"} for _ in range(frames)]
    content.append({"type": "text", "text": text})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    expanded = prompt.replace(
        processor.image_token,
        processor.image_token * image_tokens_per_frame,
    )
    encoded = processor.tokenizer(
        expanded,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
    )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("LLaVA-v1.5 tokenizer returned no input IDs")
    return len(input_ids)


def _request_text(messages: Any) -> str:
    parts: list[str] = []
    for message in messages:
        if str(message.get("role")) != "user":
            raise ValueError("LLaVA-v1.5 requests must contain user messages only")
        for item in message.get("content", []):
            item_type = str(item.get("type"))
            if item_type == "text":
                parts.append(str(item.get("text", "")))
            elif item_type != "video":
                raise ValueError(f"Unsupported LLaVA-v1.5 request item {item_type!r}")
    value = "\n".join(part for part in parts if part).strip()
    if not value:
        raise ValueError("LLaVA-v1.5 request has no text")
    return value


def _video_frame_count(path: Path) -> int:
    import decord

    reader = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)
    value = len(reader)
    if value <= 0:
        raise ValueError(f"Video has no frames: {path}")
    return value


def _uniform_midpoint_indices(total_frames: int, count: int) -> list[int]:
    return [
        min(total_frames - 1, int((index + 0.5) * total_frames / count))
        for index in range(count)
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected non-empty JSONL objects: {path}")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
