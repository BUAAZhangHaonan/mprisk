"""MiniCPM-V VT prefill extraction using its remote-code multimodal forward."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mprisk.models.base_wrapper import PrefillRequest
from mprisk.models.hf_visual_prefill import HfVisualPrefillWrapper
from mprisk.models.video_frame_utils import request_text_and_frames


class MiniCpmVWrapper(HfVisualPrefillWrapper):
    family = "minicpm_v"
    model_type = "minicpmv"
    architecture = "MiniCPMV"
    processor_class = "MiniCPMVProcessor"
    provenance_schema = "mprisk_minicpm_v_prefill_provenance_v1"
    contract_location = "root"
    loaded_contract_location = "root"
    forward_logits_to_keep = False

    def __init__(
        self,
        *,
        video_num_segments: int = 8,
        enable_thinking: bool = False,
        **kwargs: Any,
    ) -> None:
        self.video_num_segments = int(video_num_segments)
        if not 1 <= self.video_num_segments <= 64:
            raise ValueError("MiniCPM-V video_num_segments must be in [1, 64]")
        self.enable_thinking = bool(enable_thinking)
        if self.enable_thinking:
            raise ValueError("MiniCPM-V prefill extraction requires thinking disabled")
        super().__init__(**kwargs)

    def _load_dependencies(self) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModel, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        remote_class = get_class_from_dynamic_module(
            "modeling_minicpmv.MiniCPMV",
            str(self.model_path),
            local_files_only=True,
        )
        if not hasattr(remote_class, "all_tied_weights_keys"):
            remote_class.all_tied_weights_keys = {}
        processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            dtype=getattr(torch, self.dtype_name),
            attn_implementation=self.attn_implementation,
            device_map={"": self.device},
            local_files_only=True,
        ).eval()
        return model, processor

    def _prepare_inputs(self, request: PrefillRequest) -> tuple[Any, dict[str, Any]]:
        import torch

        text, images, provenance = request_text_and_frames(
            request,
            video_num_segments=self.video_num_segments,
        )
        self._configure_tokenizer()
        placeholders = "\n".join("(<image>./</image>)" for _ in images)
        user_text = f"{placeholders}\n{text}" if placeholders else text
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self._contract_supports_thinking():
            template_kwargs["enable_thinking"] = False
        prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            **template_kwargs,
        )
        if images:
            model_inputs = self.processor(
                text=[prompt],
                images=[images],
                return_tensors="pt",
            )
        else:
            model_inputs = dict(
                self.processor.tokenizer(
                    [prompt],
                    return_tensors="pt",
                    padding=True,
                )
            )
            model_inputs.update(
                pixel_values=[[]],
                tgt_sizes=[],
                image_bound=[torch.empty((0, 2), dtype=torch.long)],
            )
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("MiniCPM-V processor did not return attention_mask")
        if model_inputs.get("position_ids") is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            model_inputs["position_ids"] = position_ids.masked_fill(
                attention_mask == 0, 0
            )
        provenance["thinking_enabled"] = False
        return model_inputs, provenance

    def _forward_model(self, model_inputs: Mapping[str, Any]) -> Any:
        if self.model is None:
            raise RuntimeError("MiniCPM-V model is not loaded")
        data_keys = (
            "input_ids",
            "pixel_values",
            "tgt_sizes",
            "image_bound",
            "position_ids",
            "temporal_ids",
        )
        data = {key: model_inputs[key] for key in data_keys if key in model_inputs}
        if "position_ids" not in data:
            raise ValueError("MiniCPM-V requires position_ids")
        return self.model(
            data=data,
            attention_mask=model_inputs.get("attention_mask"),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

    def _configure_tokenizer(self) -> None:
        tokenizer = self.processor.tokenizer
        special_tokens = {
            "im_start": "<image>",
            "im_end": "</image>",
            "ref_start": "<ref>",
            "ref_end": "</ref>",
            "box_start": "<box>",
            "box_end": "</box>",
            "quad_start": "<quad>",
            "quad_end": "</quad>",
            "slice_start": "<slice>",
            "slice_end": "</slice>",
            "im_id_start": "<image_id>",
            "im_id_end": "</image_id>",
        }
        for name, token in special_tokens.items():
            setattr(tokenizer, name, token)
        token_ids = {
            "im_start_id": "<image>",
            "im_end_id": "</image>",
            "slice_start_id": "<slice>",
            "slice_end_id": "</slice>",
            "im_id_start_id": "<image_id>",
            "im_id_end_id": "</image_id>",
            "newline_id": "\n",
        }
        for name, token in token_ids.items():
            setattr(tokenizer, name, int(tokenizer.convert_tokens_to_ids(token)))
        tokenizer.bos_id = int(tokenizer.bos_token_id)
        tokenizer.eos_id = int(tokenizer.eos_token_id)
        tokenizer.unk_id = int(tokenizer.unk_token_id)

    def _contract_supports_thinking(self) -> bool:
        return "4_5" in self.model_path.name or "4-5" in self.model_path.name
