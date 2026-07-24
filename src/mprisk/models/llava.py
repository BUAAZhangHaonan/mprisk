"""LLaVA VT prefill wrappers with deterministic multi-image video simulation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.hf_visual_prefill import HfVisualPrefillWrapper
from mprisk.models.video_frame_utils import request_text_and_frames


class _LlavaFrameWrapper(HfVisualPrefillWrapper):
    contract_location = "text_config"
    dtype_location = "root"

    def __init__(self, *, video_num_segments: int = 8, **kwargs: Any) -> None:
        self.video_num_segments = int(video_num_segments)
        if not 1 <= self.video_num_segments <= 64:
            raise ValueError("LLaVA video_num_segments must be in [1, 64]")
        super().__init__(**kwargs)

    def _prepare_inputs(self, request: PrefillRequest) -> tuple[Any, dict[str, Any]]:
        text, images, provenance = request_text_and_frames(
            request,
            video_num_segments=self.video_num_segments,
        )
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": text})
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            kwargs["images"] = images
        return self.processor(**kwargs), provenance


class LlavaV15Wrapper(_LlavaFrameWrapper):
    family = "llava_v15"
    model_type = "llava"
    architecture = "LlavaForConditionalGeneration"
    processor_class = "LlavaProcessor"
    provenance_schema = "mprisk_llava_v15_prefill_provenance_v1"

    def __init__(self, *, video_num_segments: int = 8, **kwargs: Any) -> None:
        if int(video_num_segments) != 8:
            raise ValueError(
                "LLaVA-v1.5 requires an F8 candidate ceiling; the per-sample frame "
                "planner selects the largest legal shared M1/M12 frame count"
            )
        super().__init__(video_num_segments=video_num_segments, **kwargs)

    def _prepare_inputs(self, request: PrefillRequest) -> tuple[Any, dict[str, Any]]:
        context_contract, frame_contract = _llava_v15_runtime_contracts(
            request,
            max_position_embeddings=int(self._contract["max_position_embeddings"]),
            max_candidate_frames=self.video_num_segments,
        )
        selected_frames = int(context_contract["selected_frames"])
        text, images, provenance = request_text_and_frames(
            request,
            video_num_segments=selected_frames,
        )
        _validate_llava_v15_sampled_frames(
            request,
            provenance=provenance,
            frame_contract=frame_contract,
        )
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": text})
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            kwargs["images"] = images
        model_inputs = self.processor(**kwargs)
        _validate_llava_v15_processor_tokens(
            request,
            model_inputs=model_inputs,
            context_contract=context_contract,
        )
        return model_inputs, {
            **provenance,
            "context_budget_contract": context_contract,
            "frame_selection_contract": frame_contract,
        }

    def _load_contract(self) -> dict[str, Any]:
        """Derive the legacy Vicuna language contract from checkpoint tensors.

        This local HF conversion predates complete nested Llama config metadata.
        The safetensors index and embedding tensor are the exact checkpoint
        contract, so extraction fails closed if their layer map is incomplete.
        """
        config_path = self.model_path / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("model_type") != self.model_type:
            raise ValueError(f"Unexpected model_type in {config_path}")
        if payload.get("architectures") != [self.architecture]:
            raise ValueError(f"Unexpected architecture in {config_path}")
        text_config = payload.get("text_config")
        if not isinstance(text_config, dict) or text_config.get("model_type") != "llama":
            raise ValueError("LLaVA-v1.5 requires a nested Llama text_config")
        index_path = self.model_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("LLaVA-v1.5 weight index has no weight_map")
        layer_pattern = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
        layers = sorted(
            {
                int(match.group(1))
                for key in weight_map
                if (match := layer_pattern.match(str(key))) is not None
            }
        )
        if not layers or layers != list(range(layers[-1] + 1)):
            raise ValueError("LLaVA-v1.5 language layer index is not contiguous")
        embedding_key = "language_model.model.embed_tokens.weight"
        shard_name = weight_map.get(embedding_key)
        if not isinstance(shard_name, str):
            raise ValueError("LLaVA-v1.5 embedding tensor is absent from the index")
        from safetensors import safe_open

        with safe_open(
            self.model_path / shard_name,
            framework="pt",
            device="cpu",
        ) as handle:
            shape = tuple(int(value) for value in handle.get_slice(embedding_key).get_shape())
        if len(shape) != 2 or shape[1] <= 0:
            raise ValueError(f"Invalid LLaVA-v1.5 embedding shape: {shape}")
        dtype = str(payload.get("dtype") or payload.get("torch_dtype") or "")
        max_position_embeddings = int(text_config["max_position_embeddings"])
        if max_position_embeddings <= 0:
            raise ValueError("LLaVA-v1.5 max_position_embeddings must be positive")
        return {
            "num_hidden_layers": len(layers),
            "hidden_size": shape[1],
            "torch_dtype": dtype,
            "max_position_embeddings": max_position_embeddings,
        }

    def _load_dependencies(self) -> tuple[Any, Any]:
        import torch
        from transformers import LlavaForConditionalGeneration, LlavaProcessor

        processor = LlavaProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=getattr(torch, self.dtype_name),
            attn_implementation=self.attn_implementation,
            device_map={"": self.device},
            local_files_only=True,
        ).eval()
        return model, processor


def _llava_v15_runtime_contracts(
    request: PrefillRequest,
    *,
    max_position_embeddings: int,
    max_candidate_frames: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_contracts = request.runtime_contracts
    if not isinstance(runtime_contracts, Mapping):
        raise TypeError("LLaVA-v1.5 runtime_contracts must be a mapping")
    required = {"context_budget_contract", "frame_selection_contract"}
    if set(runtime_contracts) != required:
        raise ValueError(
            "LLaVA-v1.5 requires exactly context_budget_contract and "
            "frame_selection_contract"
        )
    context = runtime_contracts["context_budget_contract"]
    frame = runtime_contracts["frame_selection_contract"]
    if not isinstance(context, Mapping) or not isinstance(frame, Mapping):
        raise TypeError("LLaVA-v1.5 runtime contracts must be mappings")
    context = dict(context)
    frame = dict(frame)
    expected_context_keys = {
        "schema",
        "mode",
        "sample_id",
        "max_position_embeddings",
        "max_candidate_frames",
        "selected_frames",
        "conditions",
        "prompt_set_key",
        "prompt_ids",
        "prompt_ids_sha256",
        "candidate_max_token_counts",
        "candidate_condition_max_token_counts",
        "selected_max_token_count",
        "selection_rule",
        "no_truncation",
    }
    expected_frame_keys = {
        "schema",
        "sample_id",
        "sampling_method",
        "video_path",
        "selected_frames",
        "source_total_frames",
        "frame_indices",
        "frame_indices_sha256",
        "shared_conditions",
        "prompt_ids_sha256",
    }
    if set(context) != expected_context_keys:
        raise ValueError("Invalid LLaVA-v1.5 context_budget_contract fields")
    if set(frame) != expected_frame_keys:
        raise ValueError("Invalid LLaVA-v1.5 frame_selection_contract fields")
    if context["schema"] != "mprisk_llava_v15_context_budget_contract_v1":
        raise ValueError("Invalid LLaVA-v1.5 context budget schema")
    if context["mode"] != "per_sample_shared_max_legal":
        raise ValueError("Invalid LLaVA-v1.5 context budget mode")
    if frame["schema"] != "mprisk_llava_v15_shared_frame_selection_v1":
        raise ValueError("Invalid LLaVA-v1.5 frame selection schema")
    if frame["sampling_method"] != "uniform_midpoint_decord_v1":
        raise ValueError("Invalid LLaVA-v1.5 frame sampling method")
    if context["sample_id"] != request.sample_id or frame["sample_id"] != request.sample_id:
        raise ValueError("LLaVA-v1.5 runtime contract sample_id mismatch")
    if (
        _contract_int(
            context["max_position_embeddings"], "max_position_embeddings"
        )
        != max_position_embeddings
    ):
        raise ValueError("LLaVA-v1.5 context limit does not match the checkpoint")
    if (
        _contract_int(context["max_candidate_frames"], "max_candidate_frames")
        != max_candidate_frames
    ):
        raise ValueError("LLaVA-v1.5 candidate frame ceiling does not match the wrapper")
    selected_frames = _contract_int(context["selected_frames"], "selected_frames")
    if not 1 <= selected_frames <= max_candidate_frames:
        raise ValueError("LLaVA-v1.5 selected_frames is outside the candidate range")
    if _contract_int(frame["selected_frames"], "frame selected_frames") != selected_frames:
        raise ValueError("LLaVA-v1.5 selected_frames differs across runtime contracts")
    if list(context["conditions"]) != ["M1", "M12"]:
        raise ValueError("LLaVA-v1.5 context budget conditions must be M1 and M12")
    if list(frame["shared_conditions"]) != ["M1", "M12"]:
        raise ValueError("LLaVA-v1.5 shared frame conditions must be M1 and M12")
    if context["prompt_set_key"] != request.prompt_set_key:
        raise ValueError("LLaVA-v1.5 prompt-set binding mismatch")
    prompt_ids = list(context["prompt_ids"])
    if len(prompt_ids) != 8 or len(set(prompt_ids)) != 8 or not all(
        isinstance(value, str) and value for value in prompt_ids
    ):
        raise ValueError("LLaVA-v1.5 context budget requires eight unique prompt IDs")
    if request.prompt_id not in prompt_ids:
        raise ValueError("LLaVA-v1.5 request prompt is absent from the frame plan")
    prompt_sha = _canonical_sha256(prompt_ids)
    if context["prompt_ids_sha256"] != prompt_sha or frame["prompt_ids_sha256"] != prompt_sha:
        raise ValueError("LLaVA-v1.5 prompt ID signature mismatch")
    candidate_counts = context["candidate_max_token_counts"]
    condition_counts = context["candidate_condition_max_token_counts"]
    expected_candidate_keys = {
        str(value) for value in range(1, max_candidate_frames + 1)
    }
    if (
        not isinstance(candidate_counts, Mapping)
        or set(candidate_counts) != expected_candidate_keys
    ):
        raise ValueError("LLaVA-v1.5 candidate token counts must cover F1 through F8")
    if (
        not isinstance(condition_counts, Mapping)
        or set(condition_counts) != expected_candidate_keys
    ):
        raise ValueError("LLaVA-v1.5 condition token counts must cover F1 through F8")
    counts: dict[int, int] = {}
    for key in expected_candidate_keys:
        value = _contract_int(candidate_counts[key], f"candidate F{key} token maximum")
        by_condition = condition_counts[key]
        if not isinstance(by_condition, Mapping) or set(by_condition) != {"M1", "M12"}:
            raise ValueError("LLaVA-v1.5 candidate condition maxima require M1 and M12")
        condition_values = {
            condition: _contract_int(
                by_condition[condition],
                f"candidate F{key} {condition} token maximum",
            )
            for condition in ("M1", "M12")
        }
        if value <= 0 or any(item <= 0 for item in condition_values.values()):
            raise ValueError("LLaVA-v1.5 candidate token counts must be positive")
        if value != max(condition_values.values()):
            raise ValueError(
                "LLaVA-v1.5 candidate maximum must equal its M1/M12 maximum"
            )
        counts[int(key)] = value
    legal = [
        frames for frames, tokens in counts.items() if tokens <= max_position_embeddings
    ]
    if not legal:
        raise ValueError("LLaVA-v1.5 has no legal frame candidate for this sample")
    if selected_frames != max(legal):
        raise ValueError("LLaVA-v1.5 selected_frames is not the largest legal candidate")
    if (
        _contract_int(
            context["selected_max_token_count"], "selected_max_token_count"
        )
        != counts[selected_frames]
    ):
        raise ValueError("LLaVA-v1.5 selected token maximum does not match its candidate")
    if context["selection_rule"] != (
        "largest_f_with_all_p8_m1_m12_tokens_lte_context"
    ):
        raise ValueError("LLaVA-v1.5 context selection rule mismatch")
    if context["no_truncation"] is not True:
        raise ValueError("LLaVA-v1.5 context planning must forbid token truncation")
    indices = list(frame["frame_indices"])
    total_frames = _contract_int(frame["source_total_frames"], "source_total_frames")
    video_path = frame["video_path"]
    request_video_path = request.media_paths.get("vision")
    if (
        not isinstance(video_path, str)
        or not video_path
        or not Path(video_path).is_absolute()
        or not isinstance(request_video_path, str)
    ):
        raise ValueError("LLaVA-v1.5 frame plan requires a bound vision path")
    if Path(video_path).expanduser().resolve() != Path(request_video_path).expanduser().resolve():
        raise ValueError("LLaVA-v1.5 frame-plan video path differs from the request")
    if total_frames < max_candidate_frames or len(indices) != selected_frames:
        raise ValueError("Invalid LLaVA-v1.5 shared frame index contract")
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < total_frames
        for index in indices
    ):
        raise ValueError("LLaVA-v1.5 shared frame indices are outside the source video")
    expected_indices = [
        min(total_frames - 1, int((index + 0.5) * total_frames / selected_frames))
        for index in range(selected_frames)
    ]
    if indices != expected_indices:
        raise ValueError("LLaVA-v1.5 shared indices are not uniform midpoint samples")
    if frame["frame_indices_sha256"] != _canonical_sha256(indices):
        raise ValueError("LLaVA-v1.5 shared frame index signature mismatch")
    return context, frame


def _validate_llava_v15_sampled_frames(
    request: PrefillRequest,
    *,
    provenance: Mapping[str, Any],
    frame_contract: Mapping[str, Any],
) -> None:
    if request.condition == "M2":
        if int(provenance["actual_frames"]) != 0:
            raise ValueError("LLaVA-v1.5 M2 must not decode video frames")
        return
    if request.condition not in {"M1", "M12"}:
        raise ValueError(f"Unsupported LLaVA-v1.5 condition: {request.condition!r}")
    expected_frames = int(frame_contract["selected_frames"])
    indices = provenance["video_frame_indices"]
    totals = provenance["video_source_total_frames"]
    if int(provenance["actual_frames"]) != expected_frames:
        raise ValueError("LLaVA-v1.5 decoded frame count differs from the frame plan")
    if indices != [list(frame_contract["frame_indices"])]:
        raise ValueError("LLaVA-v1.5 decoded frame indices differ from the frame plan")
    if totals != [int(frame_contract["source_total_frames"])]:
        raise ValueError("LLaVA-v1.5 source frame count differs from the frame plan")


def _validate_llava_v15_processor_tokens(
    request: PrefillRequest,
    *,
    model_inputs: Mapping[str, Any],
    context_contract: Mapping[str, Any],
) -> None:
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None or getattr(attention_mask, "ndim", None) != 2:
        raise ValueError("LLaVA-v1.5 processor must return a two-dimensional attention mask")
    if int(attention_mask.shape[0]) != 1:
        raise ValueError("LLaVA-v1.5 processor token audit requires batch size one")
    token_count = int(attention_mask.shape[-1])
    max_position_embeddings = int(context_contract["max_position_embeddings"])
    if token_count > max_position_embeddings:
        raise ValueError(
            f"LLaVA-v1.5 {request.condition} processor produced {token_count} tokens, "
            f"exceeding the checkpoint limit {max_position_embeddings}"
        )
    if request.condition in {"M1", "M12"} and token_count > int(
        context_contract["selected_max_token_count"]
    ):
        raise ValueError(
            "LLaVA-v1.5 request token count exceeds the frame-plan candidate maximum"
        )
    if request.condition in {"M1", "M12"}:
        selected = str(context_contract["selected_frames"])
        condition_maximum = context_contract[
            "candidate_condition_max_token_counts"
        ][selected][request.condition]
        if token_count > int(condition_maximum):
            raise ValueError(
                "LLaVA-v1.5 request token count exceeds its condition maximum"
            )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contract_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"LLaVA-v1.5 {label} must be an integer")
    return value
