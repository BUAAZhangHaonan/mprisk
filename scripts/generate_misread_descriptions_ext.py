"""Extended Misread description generator for 11 additional models.

Mirrors scripts/generate_misread_descriptions.py output schema but adds
handlers for: gemma3 (4b/12b), glm4v, llava_v15, llava_onevision, minicpm_v
(2.6/4.5), phi3_5_vision, gemma4_12b (VA alias), phi4_multimodal, qwen2_5_vl.

Each family uses AutoProcessor + apply_chat_template + model.generate when
possible; custom-code families (minicpmv, phi3v, phi4mm) fall back to their
own chat APIs.

Output JSONL per model: outputs/v2/misread/<model_key>/descriptions.jsonl
One row per sample, condition=M12, single canonical prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
MPRISK_V2_SRC = HERE.parent.parent / "src"
if str(MPRISK_V2_SRC) not in sys.path:
    sys.path.insert(0, str(MPRISK_V2_SRC))

from mprisk_viz.delivery_manifests import (  # noqa: E402
    DeliverySample,
    load_delivery_filtered,
)

CANONICAL_PROMPT = (
    "Based on the complete input, describe the person's overall emotional state "
    "in one concise sentence. Do not address the person, give advice, or explain "
    "your reasoning."
)

MAX_VIDEO_FRAMES = 8

# TODO: replace with configs/assets/model_assets.yaml lookup (model_key -> local_path).
MODEL_PATHS = {
    # VT
    "gemma3_4b":                "/home/team/lvshuyang/Models/gemma-3-4b-it",
    "gemma3_12b":               "/home/team/lvshuyang/Models/gemma-3-12b-it",
    "glm4_6v_flash":            "/home/team/lvshuyang/Models/GLM-4.6V-Flash",
    "llava_v1_5_7b":            "/home/team/lvshuyang/Models/llava-1.5-7b-hf",
    "llava_onevision_qwen2_7b": "/home/team/lvshuyang/Models/llava-onevision-qwen2-7b-ov-hf",
    "minicpm_v_2_6":            "/home/team/lvshuyang/Models/MiniCPM-V-2_6",
    "minicpm_v_4_5":            "/home/team/lvshuyang/Models/MiniCPM-V-4_5",
    "phi3_5_vision":            "/home/team/lvshuyang/Models/Phi-3.5-vision-instruct",
    "qwen2_5_vl_7b":            "/home/team/lvshuyang/Models/Qwen2.5-VL-7B-Instruct",
    "qwen3_5_4b":               "/home/team/lvshuyang/Models/Qwen3.5-4B",
    "qwen3_5_9b":               "/home/team/lvshuyang/Models/Qwen3.5-9B",
    # VA
    # NOTE: gemma4_12b intentionally omitted — same checkpoint as gemma4_12b_it
    # (which is already done). Descriptions get symlinked at the end.
    "phi4_multimodal":          "/home/team/lvshuyang/Models/Phi-4-multimodal-instruct",
}

MODEL_FAMILIES = {
    "gemma3_4b":                "gemma3",
    "gemma3_12b":               "gemma3",
    "glm4_6v_flash":            "glm4v",
    "llava_v1_5_7b":            "llava_v15",
    "llava_onevision_qwen2_7b": "llava_onevision",
    "minicpm_v_2_6":            "minicpmv",
    "minicpm_v_4_5":            "minicpmv",
    "phi3_5_vision":            "phi3v",
    "qwen2_5_vl_7b":            "qwen_vl",
    "qwen3_5_4b":               "qwen3_5",
    "qwen3_5_9b":               "qwen3_5",
    "phi4_multimodal":          "phi4mm",
}

MODEL_PROTOCOL = {
    "gemma3_4b":                "vt",
    "gemma3_12b":               "vt",
    "glm4_6v_flash":            "vt",
    "llava_v1_5_7b":            "vt",
    "llava_onevision_qwen2_7b": "vt",
    "minicpm_v_2_6":            "vt",
    "minicpm_v_4_5":            "vt",
    "phi3_5_vision":            "vt",
    "qwen2_5_vl_7b":            "vt",
    "qwen3_5_4b":               "vt",
    "qwen3_5_9b":               "vt",
    "phi4_multimodal":          "va",
}

DEFAULT_OUTPUT_DIR = Path("outputs/v2/misread")


# ---------------------------------------------------------------------------
# Frame sampling helpers
# ---------------------------------------------------------------------------

def _sample_video_frames(video_path: str, max_frames: int = MAX_VIDEO_FRAMES):
    import av
    import numpy as np
    container = av.open(video_path)
    all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    if not all_frames:
        raise ValueError(f"Video yielded no frames: {video_path}")
    if len(all_frames) <= max_frames:
        return all_frames
    idx = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
    return [all_frames[i] for i in idx]


def _audio_to_wav(source_path: str) -> str:
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path
    ext = Path(source_path).suffix.lower()
    if ext in {".wav", ".flac", ".mp3", ".ogg", ".opus"}:
        return source_path
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH; needed for audio extraction")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", source_path,
         "-vn", "-ac", "1", "-ar", "16000",
         "-loglevel", "error", tmp.name],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    return tmp.name


def _move_inputs_to_device(inputs: dict, device: str) -> dict:
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list):
            out[k] = [
                (t.to(device) if isinstance(t, torch.Tensor) else t)
                for t in v
            ]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Generic HF-native handler
# ---------------------------------------------------------------------------

def _build_messages_generic(sample: DeliverySample, *, use_video_token: bool,
                            use_frames: bool, single_frame: bool = False) -> list[dict]:
    video_path = sample.media_paths.get("vision", "")
    text = sample.text_content
    user_items: list[dict] = []
    if use_video_token:
        user_items.append({"type": "video", "video": video_path, "fps": 1.0})
    elif use_frames:
        if single_frame:
            # For image-only models (e.g. LLaVA-1.5) — sample middle frame only.
            frames = _sample_video_frames(video_path, max_frames=1)
            user_items.append({"type": "image", "image": frames[0]})
        else:
            frames = _sample_video_frames(video_path)
            for fr in frames:
                user_items.append({"type": "image", "image": fr})
    if sample.protocol == "VT" and text:
        user_items.append({"type": "text", "text": "Text: " + text + "\n\n" + CANONICAL_PROMPT})
    else:
        user_items.append({"type": "text", "text": CANONICAL_PROMPT})
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_items},
    ]


def _load_generic_native(model_path: str, device: str, model_cls_name: str):
    from transformers import AutoProcessor
    import transformers
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    ModelCls = getattr(transformers, model_cls_name, None)
    if ModelCls is None:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True, device_map={"": device},
        ).eval()
    else:
        model = ModelCls.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True, device_map={"": device},
        ).eval()
    return model, processor


def _generate_generic(model, processor, sample: DeliverySample,
                     max_new_tokens: int, *, use_video_token: bool,
                     use_frames: bool, single_frame: bool = False) -> str:
    messages = _build_messages_generic(sample, use_video_token=use_video_token,
                                       use_frames=use_frames,
                                       single_frame=single_frame)
    # Normalize nested-list system content to plain string for processors
    # that expect str (LLaVA, Gemma3, etc.).
    msg_san: list[dict] = []
    for m in messages:
        c = m["content"]
        if isinstance(c, list) and len(c) == 1 and c[0].get("type") == "text":
            msg_san.append({"role": m["role"], "content": c[0]["text"]})
        else:
            msg_san.append(dict(m))
    try:
        inputs = processor.apply_chat_template(
            msg_san, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
    except Exception:
        text = processor.apply_chat_template(
            msg_san, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = _move_inputs_to_device(inputs, model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    token_ids = out[0] if isinstance(out, tuple) else out
    new_tokens = token_ids[:, input_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# LlavaV15: image-only model, single PIL image passed separately to processor.
# Pattern mirrors mind/src/mind/models/wrappers.py:LlavaV15Wrapper.
# ---------------------------------------------------------------------------

def _load_llava_v15(model_path: str, device: str):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    try:
        processor.tokenizer.padding_side = "left"
    except AttributeError:
        pass
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device},
        local_files_only=True,
    ).eval()
    return model, processor


def _generate_llava_v15(model, processor, sample: DeliverySample,
                       max_new_tokens: int) -> str:
    from PIL import Image
    video_path = sample.media_paths.get("vision", "")
    text = sample.text_content
    # Sample one middle frame as PIL Image.
    frames = _sample_video_frames(video_path, max_frames=1)
    image = Image.fromarray(frames[0])

    if sample.protocol == "VT" and text:
        question = "Text: " + text + "\n\n" + CANONICAL_PROMPT
    else:
        question = CANONICAL_PROMPT

    # LLaVA-1.5 modern chat template expects [{"type":"image"}, {"type":"text"}]
    # WITHOUT inline image data — image is passed separately to processor.
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    inputs = _move_inputs_to_device(inputs, model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    token_ids = out[0] if isinstance(out, tuple) else out
    new_tokens = token_ids[:, input_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Gemma-4 (gemma4_12b, VA)
# ---------------------------------------------------------------------------

def _register_gemma4_unified_alias() -> None:
    """Register 'gemma4_unified' as an alias for 'gemma4' in CONFIG_MAPPING.
    Used by AutoConfig/AutoModel lookups. Other Auto* mappings (feature
    extractor, image processor) don't need patching if we load directly via
    the Gemma4 classes."""
    try:
        from transformers.models.auto import configuration_auto as cfg_auto
        if "gemma4_unified" not in cfg_auto.CONFIG_MAPPING_NAMES:
            cfg_auto.CONFIG_MAPPING_NAMES["gemma4_unified"] = "gemma4"
    except Exception:
        pass


def _load_gemma4(model_path: str, device: str):
    _register_gemma4_unified_alias()
    from transformers import AutoProcessor
    from transformers.models.gemma4 import (
        Gemma4Config,
        Gemma4ForConditionalGeneration,
    )
    cfg = Gemma4Config.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    try:
        processor.tokenizer.padding_side = "left"
    except AttributeError:
        pass
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=True,
        device_map={"": device},
    ).eval()
    return model, processor


def _generate_gemma4(model, processor, sample: DeliverySample,
                    max_new_tokens: int) -> str:
    import numpy as np
    audio_path = sample.media_paths.get("audio", "")
    video_path = sample.media_paths.get("vision", "")
    text = sample.text_content
    content: list[dict] = []
    if audio_path:
        content.append({"type": "audio", "audio": audio_path})
    if video_path:
        content.append({"type": "video", "video": video_path})
    if text and text.strip():
        content.append({"type": "text", "text": "Transcript:\n" + text.strip()})
    content.append({"type": "text", "text": CANONICAL_PROMPT})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": content},
    ]

    audio_wavs: list[str] = []
    video_arrays: list[Any] = []
    placeholder_messages: list[dict] = []
    for msg in messages:
        if not isinstance(msg.get("content"), list):
            placeholder_messages.append(msg)
            continue
        new_items: list[dict] = []
        for item in msg["content"]:
            if not isinstance(item, dict):
                new_items.append(item)
                continue
            itype = str(item.get("type"))
            if itype == "audio":
                wav_path = _audio_to_wav(str(item.get("audio", "")))
                audio_wavs.append(wav_path)
                new_items.append({"type": "audio", "audio": wav_path})
            elif itype == "video":
                frames = _sample_video_frames(str(item.get("video", "")))
                video_arrays.append(np.stack(frames))
                new_items.append({"type": "video"})
            else:
                new_items.append(item)
        placeholder_messages.append({"role": msg["role"], "content": new_items})

    prompt = processor.apply_chat_template(
        placeholder_messages, tokenize=False, add_generation_prompt=True,
    )
    processor_kwargs: dict[str, Any] = {
        "text": [prompt],
        "return_tensors": "pt",
        "padding": True,
    }
    if audio_wavs:
        processor_kwargs["audio"] = audio_wavs
    if video_arrays:
        processor_kwargs["videos"] = video_arrays
        processor_kwargs["num_frames"] = int(video_arrays[0].shape[0])
    inputs = processor(**processor_kwargs)
    inputs = _move_inputs_to_device(inputs, model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    token_ids = out[0] if isinstance(out, tuple) else out
    new_tokens = token_ids[:, input_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# MiniCPM-V (2.6 / 4.5)
# ---------------------------------------------------------------------------

def _patch_custom_post_init(cls) -> None:
    """Patch a custom-code model class so ``all_tied_weights_keys`` is set
    after __init__ (needed under transformers>=5.5 for InternVLChatModel,
    MiniCPMV, Phi3V, Phi4MM, etc.)."""
    if getattr(cls, "_ext_patched_post_init", False):
        return
    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            try:
                expanded = self.get_expanded_tied_weights_keys(all_submodels=False)
                self.all_tied_weights_keys = expanded
            except Exception:
                self.all_tied_weights_keys = {}

    cls.__init__ = patched_init
    cls._ext_patched_post_init = True


def _patch_phi4mm_model_class(model_path: str) -> None:
    """Phi-4-MM specific patches:
    1. Patch Phi4MMModel.prepare_inputs_for_generation (required before peft
       wraps it during model __init__).
    2. Wrap Phi4MMForCausalLM.__init__ with `torch.device('cpu')` context to
       bypass the meta-device context that transformers from_pretrained
       activates. Without this, NemoConvSubsampling calls int() on a meta
       tensor and crashes.
    3. Override get_expanded_tied_weights_keys to return {} — Phi4MM's
       `_tied_weights_keys` is a list, but transformers 5.5.3 expects a dict.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    try:
        Phi4MMModelCls = get_class_from_dynamic_module(
            "modeling_phi4mm.Phi4MMModel", model_path, trust_remote_code=True,
        )
        if not hasattr(Phi4MMModelCls, "prepare_inputs_for_generation"):
            def _phi4_prepare(self, input_ids, **kwargs):
                return {"input_ids": input_ids, **kwargs}
            Phi4MMModelCls.prepare_inputs_for_generation = _phi4_prepare
    except Exception as e:
        print(f"[ext] Phi4MMModel prepare patch failed: {e}", flush=True)

    try:
        Phi4MMCls = get_class_from_dynamic_module(
            "modeling_phi4mm.Phi4MMForCausalLM", model_path, trust_remote_code=True,
        )
        if getattr(Phi4MMCls, "_ext_patched_phi4mm", False):
            return
        original_init = Phi4MMCls.__init__

        def safe_get_expanded(self, all_submodels=False):
            return {}
        Phi4MMCls.get_expanded_tied_weights_keys = safe_get_expanded

        def patched_init(self, *args, **kwargs):
            with torch.device("cpu"):
                original_init(self, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}
        Phi4MMCls.__init__ = patched_init
        Phi4MMCls._ext_patched_phi4mm = True
        print("[ext] Phi4MMForCausalLM patched (cpu-device init + tied_weights)",
              flush=True)
    except Exception as e:
        print(f"[ext] Phi4MMForCausalLM patch failed: {e}", flush=True)


def _patch_dynamic_cache_seen_tokens() -> None:
    """Phi-3.5-Vision and Phi-4-MM custom code reads several attributes/methods
    that were renamed/removed in transformers 5.5+. Add them back as
    no-op-ish stubs that don't break arithmetic or cause huge allocations."""
    try:
        from transformers import DynamicCache
    except Exception:
        return
    if getattr(DynamicCache, "_ext_patched_seen_tokens", False):
        return
    try:
        def _seen_tokens_get(self):
            try:
                return int(self.get_seq_length())
            except Exception:
                return 0
        DynamicCache.seen_tokens = property(_seen_tokens_get)

        # Phi3v does `kv_seq_len += get_usable_length(kv_seq_len, layer_idx)`.
        # Returning 0 keeps kv_seq_len unchanged (DynamicCache is unbounded).
        def _get_usable_length(self, *_a, **_kw):
            return 0
        DynamicCache.get_usable_length = _get_usable_length

        # Return a moderately-large int. Phi3v uses this for buffer sizing;
        # huge values cause OOM, small values might cause re-allocations.
        def _get_max_length(self):
            return 8192
        DynamicCache.get_max_length = _get_max_length

        DynamicCache._ext_patched_seen_tokens = True
    except Exception as e:
        print(f"[ext] DynamicCache patch failed: {e}", flush=True)


def _disable_cache_class_support(model_cls) -> None:
    """Force a custom-code model to use the legacy tuple-of-tuples cache
    instead of the new Cache class. Phi3V/Phi4MM custom attention code
    expects the legacy format and breaks under transformers 5.5+'s Cache."""
    if getattr(model_cls, "_ext_disabled_cache_class", False):
        return
    try:
        model_cls._supports_cache_class = False
        model_cls._ext_disabled_cache_class = True
    except Exception as e:
        print(f"[ext] disable_cache_class failed: {e}", flush=True)


def _patch_custom_classes(model_path: str, target_cls_names: set[str],
                         disable_cache_class: bool = False) -> None:
    """Trigger trust_remote_code registration via AutoConfig, then patch any
    dynamically-registered class whose __name__ is in target_cls_names."""
    import importlib
    from transformers import AutoConfig
    try:
        from transformers import AUTO_MODEL_REGISTRY  # type: ignore
    except Exception:
        AUTO_MODEL_REGISTRY = None
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    auto_map = getattr(cfg, "auto_map", {}) or {}

    patched: set[str] = set()

    # Strategy 1: load each AutoModel entry in auto_map via dynamic class loader.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    for auto_key, qualified in auto_map.items():
        if not auto_key.startswith("AutoModel"):
            continue
        cls_name = qualified.split(".")[-1]
        if cls_name not in target_cls_names or cls_name in patched:
            continue
        try:
            cls = get_class_from_dynamic_module(qualified, model_path, trust_remote_code=True)
            if isinstance(cls, type):
                _patch_custom_post_init(cls)
                if disable_cache_class:
                    _disable_cache_class_support(cls)
                patched.add(cls_name)
        except Exception as e:
            print(f"[ext] get_class_from_dynamic_module failed for {qualified}: {e}",
                  flush=True)

    # Strategy 2: walk AUTO_MODEL_REGISTRY.
    if AUTO_MODEL_REGISTRY is not None:
        for cls in AUTO_MODEL_REGISTRY.values():
            name = getattr(cls, "__name__", "")
            if name in target_cls_names and name not in patched:
                _patch_custom_post_init(cls)
                if disable_cache_class:
                    _disable_cache_class_support(cls)
                patched.add(name)

    # Strategy 3: walk sys.modules.
    for name, mod in list(sys.modules.items()):
        if not any(tag in name for tag in ("modeling_", "minicpm", "phi3", "phi4")):
            continue
        for tgt in target_cls_names - patched:
            cls = getattr(mod, tgt, None)
            if cls is not None and isinstance(cls, type):
                _patch_custom_post_init(cls)
                if disable_cache_class:
                    _disable_cache_class_support(cls)
                patched.add(tgt)

    if not patched:
        print(f"[ext] WARN: no classes patched for {target_cls_names} at {model_path}",
              flush=True)
    else:
        print(f"[ext] patched classes: {patched}", flush=True)


def _load_minicpmv(model_path: str, device: str):
    _patch_custom_classes(model_path, {"MiniCPMV"})
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    # Load custom MiniCPMVTokenizerFast (has im_start_id/im_end_id props).
    tokenizer = None
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        TokCls = get_class_from_dynamic_module(
            "tokenization_minicpmv_fast.MiniCPMVTokenizerFast", model_path, trust_remote_code=True,
        )
        tokenizer = TokCls.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
    except Exception as e:
        print(f"[ext] minicpmv custom tokenizer load failed, falling back: {e}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
    model = AutoModel.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": device},
    ).eval()
    # Build the processor ourselves so we can force its tokenizer to be the
    # custom class. Otherwise chat() will lazily load AutoProcessor which
    # uses the default fast tokenizer (TokenizersBackend) missing im_start_id.
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True,
        )
        processor.tokenizer = tokenizer
        model.processor = processor
    except Exception as e:
        print(f"[ext] minicpmv processor override skipped: {e}", flush=True)
    return model, tokenizer


def _generate_minicpmv(model, tokenizer, sample: DeliverySample,
                      max_new_tokens: int) -> str:
    from PIL import Image
    video_path = sample.media_paths.get("vision", "")
    text = sample.text_content
    frames_np = _sample_video_frames(video_path, max_frames=MAX_VIDEO_FRAMES)
    # MiniCPM-V chat() expects a single representative image; we pass the
    # middle frame (most informative for emotion description).
    mid = len(frames_np) // 2
    image = Image.fromarray(frames_np[mid])

    if sample.protocol == "VT" and text:
        question = "Text: " + text + "\n\n" + CANONICAL_PROMPT
    else:
        question = CANONICAL_PROMPT

    msgs = [{"role": "user", "content": question}]
    out = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=False,
        max_new_tokens=max_new_tokens,
    )
    if isinstance(out, tuple):
        out = out[0]
    return str(out).strip()


# ---------------------------------------------------------------------------
# Phi-3.5-Vision
# ---------------------------------------------------------------------------

def _load_phi3v(model_path: str, device: str):
    _patch_custom_classes(model_path, {"Phi3VForCausalLM"}, disable_cache_class=True)
    _patch_dynamic_cache_seen_tokens()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    # num_crops=4 required for Phi-3.5-Vision processor (mind Phi35VisionWrapper).
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        num_crops=4,
    )
    # Pre-set _attn_implementation on config (mind Phi4MultimodalWrapper pattern).
    # Passing attn_implementation="eager" as kwarg alone does NOT propagate
    # through custom-code __init__ under transformers 5.5.3.
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
        device_map=None,
    ).eval()
    model = model.to(device)
    return model, processor


def _generate_phi3v(model, processor, sample: DeliverySample,
                   max_new_tokens: int) -> str:
    from PIL import Image
    video_path = sample.media_paths.get("vision", "")
    text = sample.text_content
    # Single middle frame as PIL.
    frames = _sample_video_frames(video_path, max_frames=1)
    image = Image.fromarray(frames[0])

    if sample.protocol == "VT" and text:
        question = "Text: " + text + "\n\n" + CANONICAL_PROMPT
    else:
        question = CANONICAL_PROMPT

    # Phi-3.5-Vision native prompt format (mind Phi35VisionWrapper.build_prompt).
    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = _move_inputs_to_device(inputs, model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=False,
        )
    token_ids = out[0] if isinstance(out, tuple) else out
    new_tokens = token_ids[:, input_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Phi-4-Multimodal (VA)
# ---------------------------------------------------------------------------

def _load_phi4mm(model_path: str, device: str):
    _patch_custom_classes(model_path, {"Phi4MMForCausalLM"}, disable_cache_class=True)
    _patch_dynamic_cache_seen_tokens()
    _patch_phi4mm_model_class(model_path)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    # Pre-set _attn_implementation on config (mind Phi4MultimodalWrapper pattern).
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
        device_map=None,
    ).eval()
    model = model.to(device)
    return model, processor


def _generate_phi4mm(model, processor, sample: DeliverySample,
                    max_new_tokens: int) -> str:
    from PIL import Image
    audio_path = sample.media_paths.get("audio", "")
    video_path = sample.media_paths.get("vision", "")
    text = sample.text_content

    # Single middle frame (mind Phi4MultimodalWrapper uses single image).
    frames = _sample_video_frames(video_path, max_frames=1)
    image = Image.fromarray(frames[0])

    if sample.protocol == "VT" and text:
        question = "Text: " + text + "\n\n" + CANONICAL_PROMPT
    else:
        # VA protocol — Phi-4-MM has audio input but we don't have raw audio
        # path in our pipeline; use the transcript-only fallback.
        question = (text + "\n\n" + CANONICAL_PROMPT) if text else CANONICAL_PROMPT

    # Phi-4-MM native prompt format (mind Phi4MultimodalWrapper.build_prompt).
    prompt = f"<|user|><|image_1|>{question}<|end|><|assistant|>"
    proc_kwargs: dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
    # Phi-4-MM uses audio for VA protocol; skip audio for now (no clean path).
    proc_kwargs["images"] = image
    inputs = processor(**proc_kwargs)
    inputs = _move_inputs_to_device(inputs, model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=False, num_logits_to_keep=0,
        )
    token_ids = out[0] if isinstance(out, tuple) else out
    new_tokens = token_ids[:, input_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Qwen3.5 (4B / 9B, VT). Native arch Qwen3_5ForConditionalGeneration.
# ---------------------------------------------------------------------------

def _load_qwen3_5(model_path: str, device: str):
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": device},
        local_files_only=True, trust_remote_code=True,
        ignore_mismatched_sizes=True,
    ).eval()
    return model, processor


def _generate_qwen3_5(model, processor, sample: DeliverySample,
                     max_new_tokens: int) -> str:
    """Qwen3.5-specific generate: forces enable_thinking=False so the chat
    template emits empty <think></think> tags instead of consuming the entire
    token budget on chain-of-thought."""
    video_path = sample.media_paths.get("vision", "")
    text = sample.text_content
    user_items: list[dict] = [{"type": "video", "video": video_path, "fps": 1.0}]
    if sample.protocol == "VT" and text:
        user_items.append({"type": "text", "text": "Text: " + text + "\n\n" + CANONICAL_PROMPT})
    else:
        user_items.append({"type": "text", "text": CANONICAL_PROMPT})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": user_items},
    ]
    # Keep all content as list-of-dicts: processing_utils.py:1807 iterates
    # message["content"] for every message assuming list-of-dicts.
    msg_san = [dict(m) for m in messages]
    inputs = processor.apply_chat_template(
        msg_san, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
        enable_thinking=False,
    )
    inputs = _move_inputs_to_device(inputs, model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    token_ids = out[0] if isinstance(out, tuple) else out
    new_tokens = token_ids[:, input_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_NATIVE_CLS = {
    "gemma3":          "Gemma3ForConditionalGeneration",
    "glm4v":           "Glm4vForConditionalGeneration",
    "llava_v15":       "LlavaForConditionalGeneration",
    "llava_onevision": "LlavaOnevisionForConditionalGeneration",
    "qwen_vl":         "Qwen2_5_VLForConditionalGeneration",
}


def _load_for_family(family: str, model_path: str, device: str):
    if family in _NATIVE_CLS:
        return _load_generic_native(model_path, device, _NATIVE_CLS[family])
    if family == "qwen3_5":
        return _load_qwen3_5(model_path, device)
    if family == "gemma4":
        return _load_gemma4(model_path, device)
    if family == "minicpmv":
        return _load_minicpmv(model_path, device)
    if family == "phi3v":
        return _load_phi3v(model_path, device)
    if family == "phi4mm":
        return _load_phi4mm(model_path, device)
    if family == "llava_v15":
        return _load_llava_v15(model_path, device)
    raise ValueError(f"unknown family {family}")


def _generate_for_family(family: str, gen_args, sample: DeliverySample,
                        max_new_tokens: int) -> str:
    # Families with native video support: pass video path via chat content.
    if family in {"glm4v", "llava_onevision", "qwen_vl"}:
        return _generate_generic(gen_args[0], gen_args[1], sample, max_new_tokens,
                                use_video_token=True, use_frames=False)
    # qwen3_5 needs enable_thinking=False to avoid wasting tokens on CoT.
    if family == "qwen3_5":
        return _generate_qwen3_5(*gen_args, sample, max_new_tokens)
    # gemma3: native multi-image, works with frame stack.
    if family == "gemma3":
        return _generate_generic(gen_args[0], gen_args[1], sample, max_new_tokens,
                                use_video_token=False, use_frames=True)
    if family == "llava_v15":
        return _generate_llava_v15(*gen_args, sample, max_new_tokens)
    if family == "gemma4":
        return _generate_gemma4(*gen_args, sample, max_new_tokens)
    if family == "minicpmv":
        return _generate_minicpmv(*gen_args, sample, max_new_tokens)
    if family == "phi3v":
        return _generate_phi3v(*gen_args, sample, max_new_tokens)
    if family == "phi4mm":
        return _generate_phi4mm(*gen_args, sample, max_new_tokens)
    raise ValueError(f"unknown family {family}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def generate_descriptions(
    *,
    model_key: str,
    output_path: str | Path | None = None,
    delivery_dir: str | Path | None = None,
    device: str = "cuda:0",
    max_new_tokens: int = 64,
    limit: int | None = None,
    skip_existing: bool = True,
    sample_type_filter: str | None = None,
) -> Path:
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / model_key / "descriptions.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_PATHS[model_key]
    family = MODEL_FAMILIES[model_key]
    protocol = MODEL_PROTOCOL[model_key]

    print(f"[ext] loading {model_key} ({family}, protocol={protocol.upper()}) "
          f"from {model_path}", flush=True)
    gen_args = _load_for_family(family, model_path, device)

    samples = load_delivery_filtered(protocol, output_dir=delivery_dir)
    print(f"[ext] loaded {len(samples)} delivery samples (protocol={protocol.upper()})",
          flush=True)
    if sample_type_filter:
        samples = [s for s in samples if s.sample_type == sample_type_filter]
        print(f"[ext] filtered to sample_type={sample_type_filter} -> {len(samples)}",
              flush=True)

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
        print(f"[ext] resuming - {len(existing)} already done", flush=True)

    target = samples[:limit] if limit else samples
    out_f = output_path.open("a" if skip_existing else "w", encoding="utf-8")
    started = time.perf_counter()
    done = 0
    errors = 0
    skipped = 0

    with out_f:
        for s in tqdm(target, desc=model_key):
            if s.sample_id in existing:
                skipped += 1
                continue
            t0 = time.perf_counter()
            try:
                text = _generate_for_family(family, gen_args, s, max_new_tokens)
                elapsed = time.perf_counter() - t0
                row = {
                    "schema": "mprisk_v2_diagnostic_description_v1",
                    "sample_id": s.sample_id,
                    "subject_model_key": model_key,
                    "protocol": s.protocol,
                    "sample_type": s.sample_type,
                    "condition": "M12",
                    "source_id": s.source_id,
                    "gt_emotion": s.gt_emotion,
                    "surface_emotion": s.surface_emotion,
                    "gt_describe": s.gt_describe,
                    "generated_description": text,
                    "diagnostic_description": text,
                    "max_new_tokens": max_new_tokens,
                    "elapsed_seconds": elapsed,
                }
                out_f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                out_f.flush()
                done += 1
            except Exception as exc:
                errors += 1
                elapsed = time.perf_counter() - t0
                err_row = {
                    "schema": "mprisk_v2_diagnostic_description_v1",
                    "sample_id": s.sample_id,
                    "subject_model_key": model_key,
                    "protocol": s.protocol,
                    "sample_type": s.sample_type,
                    "condition": "M12",
                    "source_id": s.source_id,
                    "gt_emotion": s.gt_emotion,
                    "surface_emotion": s.surface_emotion,
                    "gt_describe": s.gt_describe,
                    "generated_description": "",
                    "diagnostic_description": "",
                    "max_new_tokens": max_new_tokens,
                    "elapsed_seconds": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                out_f.write(json.dumps(err_row, sort_keys=True, ensure_ascii=False) + "\n")
                out_f.flush()
                print(f"[ext] {s.sample_id} failed: {type(exc).__name__}: {exc}",
                      flush=True)
    total_wall = time.perf_counter() - started
    print(f"[ext] done model={model_key} done={done} skipped={skipped} "
          f"errors={errors} wall={total_wall:.1f}s", flush=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=list(MODEL_PATHS.keys()), required=True)
    parser.add_argument("--delivery-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-type", choices=["Conflict", "Aligned"], default=None)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    generate_descriptions(
        model_key=args.model_key,
        output_path=args.output,
        delivery_dir=args.delivery_dir,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        skip_existing=args.skip_existing,
        sample_type_filter=args.sample_type,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
