"""Exact prompt-prefix KV reuse for Qwen3-VL pre-generation states.

For one (sample, condition) we run 8 prompt forwards. The 8 prompts share
a long multimodal prefix (chat-template scaffold + video tokens + transcript
text + "Task:\n"). We compute the prefix once, materialise past_key_values,
then run each prompt as a cheap suffix-only forward.

This extractor is deliberately registered only for Qwen3-VL. Other model
families need independent, model-native cache contracts before registration.
"""

from __future__ import annotations

import copy
import os
import site
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from mprisk.models.base_wrapper import PrefillRequest, PrefillResult

_SAFETY_PREFIX_FRACTION = 0.5
"""If the common prefix is shorter than this fraction of the shortest full
sequence, the exact-cache contract is invalid. The 8 prompts share chat
template + media + transcript, so the common prefix should almost always
exceed 50% of total length."""


class QwenVlPromptKvPrefillExtractor:
    """Run 8 prompt forwards per (sample, condition) sharing one KV cache."""

    def __init__(self, wrapper: Any, *, verbose: bool = True) -> None:
        _require_isolated_python_environment()
        family = getattr(wrapper, "family", None)
        if family not in {"qwen_vl"}:
            raise ValueError(
                "QwenVlPromptKvPrefillExtractor supports only the qwen_vl family, "
                f"got family={family!r}"
            )
        self.wrapper = wrapper
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract_condition_batch(
        self,
        *,
        sample_row: dict,
        build_request_fn: Callable[..., PrefillRequest],
        prompt_texts: Sequence[str],
        condition: str,
        protocol: str,
        prompt_set_key: str,
        prompt_ids: Sequence[str],
        common_kwargs: Mapping[str, Any],
    ) -> list[PrefillResult]:
        """Return one PrefillResult per (prompt_id, prompt_text) pair."""
        if len(prompt_texts) != len(prompt_ids):
            raise ValueError("prompt_texts and prompt_ids must be parallel sequences")
        if not prompt_texts:
            raise ValueError("extract_condition_batch requires at least one prompt")

        model_key = self.wrapper.model_key
        dataset_key = str(sample_row.get("source_dataset", ""))
        split = str(sample_row.get("split", ""))
        sample_id = str(sample_row["sample_id"])
        media_paths = {str(k): str(v) for k, v in sample_row["media_paths"].items()}
        transcript = sample_row.get("text_content")
        transcript = None if transcript is None else str(transcript)

        requests: list[PrefillRequest] = []
        for prompt_text, prompt_id in zip(prompt_texts, prompt_ids, strict=True):
            request = build_request_fn(
                sample_id=sample_id,
                model_key=model_key,
                protocol=protocol,
                condition=condition,
                dataset_key=dataset_key,
                split=split,
                media_paths=media_paths,
                transcript=transcript,
                task_prompt=prompt_text,
                prompt_set_key=prompt_set_key,
                prompt_id=prompt_id,
                **dict(common_kwargs),
            )
            requests.append(request)

        return self._extract_with_kv_cache(requests)

    # ------------------------------------------------------------------
    # KV-cache implementation
    # ------------------------------------------------------------------
    def _extract_with_kv_cache(self, requests: list[PrefillRequest]) -> list[PrefillResult]:
        import torch

        if self.wrapper.model is None or self.wrapper.processor is None:
            self.wrapper.load()
        if self.wrapper.model is None or self.wrapper.processor is None:
            raise RuntimeError("Wrapper model/processor unavailable")

        started_at = time.perf_counter()
        with torch.inference_mode():
            model_inputs_per_prompt = [self._build_model_inputs(request) for request in requests]

            # Verify a meaningful common prefix exists. The 8 prompts share the
            # chat template scaffold + video tokens + transcript, so they must
            # agree on a long prefix.
            prefix_len = _longest_common_prefix_length(
                [
                    tuple(int(t) for t in mi["input_ids"][0].tolist())
                    for mi in model_inputs_per_prompt
                ]
            )
            shortest_len = min(int(mi["input_ids"].shape[-1]) for mi in model_inputs_per_prompt)
            if self.verbose:
                print(
                    "[QwenVlPromptKvPrefillExtractor] "
                    f"prefix_len={prefix_len} shortest_len={shortest_len} "
                    "longest_len="
                    f"{max(int(mi['input_ids'].shape[-1]) for mi in model_inputs_per_prompt)}"
                )

            if prefix_len < int(shortest_len * _SAFETY_PREFIX_FRACTION) or prefix_len == 0:
                raise RuntimeError(
                    f"Common prefix too short ({prefix_len}/{shortest_len}); "
                    "prompt texts probably differ before the expected split point."
                )

            _assert_identical_token_prefix(model_inputs_per_prompt, prefix_len)
            full_position_ids = [self._full_position_ids(mi) for mi in model_inputs_per_prompt]
            prefix_inputs = self._build_prefix_inputs(
                model_inputs_per_prompt[0],
                prefix_len,
                full_position_ids=full_position_ids[0],
            )
            if self.verbose:
                print(
                    "[QwenVlPromptKvPrefillExtractor] "
                    f"prefix keys={sorted(prefix_inputs.keys())} "
                    f"prefix_seq_len={prefix_len}"
                )

            outputs_prefix = self.wrapper.model(
                **prefix_inputs,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
                logits_to_keep=1,
            )
            prefix_cache = outputs_prefix.past_key_values
            if prefix_cache is None:
                raise RuntimeError("Prefix forward did not return past_key_values")
            if int(prefix_cache.get_seq_length()) != prefix_len:
                raise RuntimeError(
                    "Prefix cache length does not match the token prefix: "
                    f"{int(prefix_cache.get_seq_length())} != {prefix_len}"
                )
            try:
                results: list[PrefillResult] = []
                for request, full_inputs, position_ids in zip(
                    requests,
                    model_inputs_per_prompt,
                    full_position_ids,
                    strict=True,
                ):
                    suffix_cache = _clone_pristine_dynamic_cache(prefix_cache)
                    suffix_result = self._suffix_forward(
                        request=request,
                        full_inputs=full_inputs,
                        full_position_ids=position_ids,
                        prefix_len=prefix_len,
                        past_key_values=suffix_cache,
                    )
                    if int(prefix_cache.get_seq_length()) != prefix_len:
                        raise RuntimeError("A suffix forward mutated the pristine prefix cache")
                    results.append(suffix_result)
            finally:
                del prefix_cache
                del outputs_prefix

        elapsed = time.perf_counter() - started_at
        if self.verbose:
            print(
                f"[QwenVlPromptKvPrefillExtractor] batch of {len(requests)} "
                f"done in {elapsed:.2f}s "
                f"({elapsed / len(requests):.2f}s/prompt)"
            )
        return results

    # ------------------------------------------------------------------
    # Qwen3-VL specifics
    # ------------------------------------------------------------------
    def _build_model_inputs(self, request: PrefillRequest) -> Any:
        """Mirror QwenVlWrapper.extract_prefill preprocessing for one request."""
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        video_fps = _request_video_fps(request)
        if video_fps is not None:
            template_kwargs["fps"] = video_fps
        model_inputs = self.wrapper.processor.apply_chat_template(
            [dict(message) for message in request.messages],
            **template_kwargs,
        )
        return _move_inputs_to_device(model_inputs, self.wrapper.device)

    def _full_position_ids(self, full_inputs: Any) -> Any:
        """Compute the exact full-sequence M-RoPE positions used by a full prefill."""
        required = ("input_ids", "attention_mask", "mm_token_type_ids")
        missing = [key for key in required if full_inputs.get(key) is None]
        if missing:
            raise RuntimeError(f"Qwen3-VL processor output is missing {missing}")
        position_ids, _ = self.wrapper.model.model.get_rope_index(
            full_inputs["input_ids"],
            mm_token_type_ids=full_inputs["mm_token_type_ids"],
            image_grid_thw=full_inputs.get("image_grid_thw"),
            video_grid_thw=full_inputs.get("video_grid_thw"),
            attention_mask=full_inputs["attention_mask"],
        )
        if position_ids.ndim != 3 or int(position_ids.shape[0]) != 3:
            raise RuntimeError(
                f"Expected full Qwen3-VL M-RoPE positions with shape [3, B, L], "
                f"got {tuple(position_ids.shape)}"
            )
        if int(position_ids.shape[-1]) != int(full_inputs["input_ids"].shape[-1]):
            raise RuntimeError("Full M-RoPE positions do not match input_ids length")
        return position_ids

    def _build_prefix_inputs(
        self,
        full_inputs: Any,
        prefix_len: int,
        *,
        full_position_ids: Any,
    ) -> dict[str, Any]:
        """Slice the first prefix_len positions of input_ids / attention_mask.

        Media tensors (pixel_values_videos, video_grid_thw, image_grid_thw, etc.)
        are passed unchanged: they describe media in the prefix, and the vision
        encoder consumes them once. The suffix forward will not re-pass them.
        mm_token_type_ids is also sliced to match the prefix.
        """
        prefix_inputs: dict[str, Any] = {}
        for key, value in full_inputs.items():
            if key == "input_ids":
                prefix_inputs[key] = value[:, :prefix_len]
            elif key == "attention_mask":
                prefix_inputs[key] = value[:, :prefix_len]
            elif key == "mm_token_type_ids":
                prefix_inputs[key] = value[:, :prefix_len]
            else:
                prefix_inputs[key] = value
        prefix_inputs["position_ids"] = full_position_ids[..., :prefix_len]
        return prefix_inputs

    def _suffix_forward(
        self,
        *,
        request: PrefillRequest,
        full_inputs: Any,
        full_position_ids: Any,
        prefix_len: int,
        past_key_values: Any,
    ) -> PrefillResult:
        import torch

        full_input_ids = full_inputs["input_ids"]
        full_len = int(full_input_ids.shape[-1])
        suffix_len = full_len - prefix_len
        if suffix_len <= 0:
            raise RuntimeError(
                f"Suffix length non-positive: full_len={full_len}, prefix_len={prefix_len}"
        )

        suffix_input_ids = full_input_ids[:, prefix_len:]
        full_attention_for_suffix = full_inputs["attention_mask"][:, :full_len]
        if tuple(full_attention_for_suffix.shape) != (1, full_len):
            raise RuntimeError(
                f"Expected one full attention mask of shape {(1, full_len)}, "
                f"got {tuple(full_attention_for_suffix.shape)}"
            )
        suffix_position_ids = full_position_ids[..., prefix_len:full_len]
        if tuple(suffix_position_ids.shape) != (3, 1, suffix_len):
            raise RuntimeError(
                f"Expected suffix M-RoPE positions {(3, 1, suffix_len)}, "
                f"got {tuple(suffix_position_ids.shape)}"
            )

        # Suffix contains no video tokens, so do NOT pass media tensors. The
        # vision encoder has already been consumed during the prefix forward;
        # its outputs live in past_key_values now.
        suffix_inputs: dict[str, Any] = {
            "input_ids": suffix_input_ids,
            "attention_mask": full_attention_for_suffix,
            "position_ids": suffix_position_ids,
            "past_key_values": past_key_values,
            "output_hidden_states": True,
            "use_cache": True,
            "return_dict": True,
            "logits_to_keep": 1,
        }
        outputs = self.wrapper.model(**suffix_inputs)

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Suffix forward did not return hidden_states")
        layer_count = self.wrapper.expected_layer_count
        hidden_dim = self.wrapper.expected_hidden_dim
        expected_state_count = layer_count + 1
        if len(hidden_states) != expected_state_count:
            actual = len(hidden_states)
            raise RuntimeError(
                f"Suffix hidden_states count {actual} != expected {expected_state_count}"
            )
        t0_token_index = suffix_len - 1
        trajectory = torch.stack(
            [state[0, t0_token_index, :] for state in hidden_states[1:]],
            dim=0,
        )
        if tuple(trajectory.shape) != (layer_count, hidden_dim):
            raise RuntimeError(
                f"Suffix trajectory shape {tuple(trajectory.shape)} != "
                f"{(layer_count, hidden_dim)}"
            )
        if not torch.isfinite(trajectory).all().item():
            raise RuntimeError("Suffix trajectory contains non-finite values")

        # Build provenance that mirrors QwenVlWrapper.extract_prefill, with KV
        # markers appended so downstream code can tell these results came from
        # the cache-reuse path.
        base_provenance = self._base_provenance(request)
        base_provenance.update(
            {
                "kv_cache": True,
                "prefix_len": int(prefix_len),
                "suffix_len": int(suffix_len),
                "schema": base_provenance.get("schema", "mprisk_qwen3_vl_prefill_provenance_v1")
                + "_kv",
            }
        )
        trajectory_np = trajectory.detach().to(dtype=torch.float32, device="cpu").numpy()
        return PrefillResult(
            request=request,
            trajectory=np.asarray(trajectory_np, dtype=np.float32),
            token_count=full_len,
            t0_token_index=full_len - 1,
            provenance=base_provenance,
        )

    def _base_provenance(self, request: PrefillRequest) -> dict[str, Any]:
        """Minimal provenance; verification script only checks trajectory, so
        we omit expensive hashes here (QwenVlWrapper's full provenance is
        rebuilt for real-cache writes elsewhere)."""
        import torch

        return {
            "schema": "mprisk_qwen3_vl_prefill_provenance_v1_kv",
            "model_path": str(self.wrapper.model_path),
            "model_class": self.wrapper.model.__class__.__name__,
            "processor_class": self.wrapper.processor.__class__.__name__,
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "source_dtype": self.wrapper.dtype_name,
            "stored_dtype": "float32",
            "device": self.wrapper.device,
            "attn_implementation": self.wrapper.attn_implementation,
            "num_hidden_layers": self.wrapper.expected_layer_count,
            "hidden_size": self.wrapper.expected_hidden_dim,
            "hidden_state_index_offset": 1,
            "visual_input_types": _visual_input_types(request),
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _longest_common_prefix_length(sequences: list[tuple[int, ...]]) -> int:
    if not sequences:
        return 0
    reference = sequences[0]
    prefix_len = len(reference)
    for seq in sequences[1:]:
        bound = min(prefix_len, len(seq))
        mismatch = bound
        for i in range(bound):
            if seq[i] != reference[i]:
                mismatch = i
                break
        else:
            mismatch = bound
        prefix_len = min(prefix_len, mismatch)
        if prefix_len == 0:
            return 0
    return prefix_len


def _require_isolated_python_environment() -> None:
    if os.environ.get("PYTHONNOUSERSITE") != "1" or site.ENABLE_USER_SITE:
        raise RuntimeError(
            "KV extraction requires PYTHONNOUSERSITE=1 so Transformers cannot be imported "
            "from ~/.local. Set it before starting Python."
        )


def _assert_identical_token_prefix(model_inputs: Sequence[Any], prefix_len: int) -> None:
    import torch

    if prefix_len <= 0:
        raise ValueError("prefix_len must be positive")
    reference = model_inputs[0]
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        reference_value = reference.get(key)
        if reference_value is None:
            raise RuntimeError(f"Qwen3-VL processor output is missing {key}")
        reference_prefix = reference_value[..., :prefix_len]
        for prompt_index, inputs in enumerate(model_inputs[1:], start=1):
            candidate = inputs.get(key)
            if candidate is None:
                raise RuntimeError(f"Qwen3-VL processor output is missing {key}")
            if candidate.ndim != reference_value.ndim:
                raise RuntimeError(f"Prompt {prompt_index} has a different {key} rank")
            candidate_prefix = candidate[..., :prefix_len]
            if candidate_prefix.shape != reference_prefix.shape or not torch.equal(
                candidate_prefix, reference_prefix
            ):
                raise RuntimeError(
                    f"Prompt {prompt_index} does not share the exact {key} prefix"
                )


def _clone_pristine_dynamic_cache(prefix_cache: Any) -> Any:
    """Clone cache metadata/layers while sharing immutable prefix tensors.

    A suffix update replaces each copied layer's key/value tensor with a concatenated
    tensor, so the source cache remains unchanged without eagerly duplicating the long
    multimodal prefix.
    """
    layers = getattr(prefix_cache, "layers", None)
    if not isinstance(layers, list) or not layers:
        raise TypeError("Expected a non-empty Transformers DynamicCache")
    cloned = copy.copy(prefix_cache)
    cloned.layers = [copy.copy(layer) for layer in layers]
    for source, target in zip(layers, cloned.layers, strict=True):
        if not getattr(source, "is_initialized", False):
            raise RuntimeError("Cannot reuse an uninitialized prefix-cache layer")
        if getattr(source, "keys", None) is None or getattr(source, "values", None) is None:
            raise RuntimeError("Prefix-cache layer is missing keys or values")
        target.keys = source.keys
        target.values = source.values
    return cloned


def _request_video_fps(request: PrefillRequest) -> float | None:
    values = {
        float(item["fps"])
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, Mapping) and item.get("type") == "video" and "fps" in item
    }
    if len(values) > 1:
        raise ValueError("Request cannot mix video fps values")
    return next(iter(values), None)


def _visual_input_types(request: PrefillRequest) -> list[str]:
    return [
        str(item.get("type"))
        for message in request.messages
        for item in message.get("content", [])
        if isinstance(item, Mapping) and item.get("type") in {"image", "video"}
    ]


def _move_inputs_to_device(model_inputs: Any, device: str) -> Any:
    if hasattr(model_inputs, "to"):
        return model_inputs.to(device)
    if not isinstance(model_inputs, Mapping):
        raise TypeError("Processor output must be a BatchFeature or mapping")
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }
