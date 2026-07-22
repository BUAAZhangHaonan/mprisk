"""Model wrapper registry."""

from __future__ import annotations

from typing import TypeAlias

from mprisk.models.base_wrapper import BaseModelWrapper
from mprisk.models.gemma3 import Gemma3Wrapper
from mprisk.models.gemma4 import Gemma4Wrapper
from mprisk.models.glm4v import Glm4vWrapper
from mprisk.models.internvl import InternVlWrapper
from mprisk.models.llava import LlavaV15Wrapper
from mprisk.models.llava_onevision import LlavaOneVisionWrapper
from mprisk.models.minicpm_v import MiniCpmVWrapper
from mprisk.models.phi3_vision import Phi3VisionWrapper
from mprisk.models.phi4_mm import Phi4MmWrapper
from mprisk.models.qwen2_5_vl import Qwen2_5VlWrapper
from mprisk.models.qwen3_5 import Qwen3_5Wrapper
from mprisk.models.qwen_omni import QwenOmniWrapper
from mprisk.models.qwen_vl import QwenVlWrapper

WrapperFactory: TypeAlias = type[BaseModelWrapper]

REGISTRY: dict[str, WrapperFactory] = {
    Gemma3Wrapper.family: Gemma3Wrapper,
    Gemma4Wrapper.family: Gemma4Wrapper,
    Glm4vWrapper.family: Glm4vWrapper,
    InternVlWrapper.family: InternVlWrapper,
    LlavaOneVisionWrapper.family: LlavaOneVisionWrapper,
    LlavaV15Wrapper.family: LlavaV15Wrapper,
    MiniCpmVWrapper.family: MiniCpmVWrapper,
    Phi3VisionWrapper.family: Phi3VisionWrapper,
    Phi4MmWrapper.family: Phi4MmWrapper,
    Qwen2_5VlWrapper.family: Qwen2_5VlWrapper,
    Qwen3_5Wrapper.family: Qwen3_5Wrapper,
    QwenOmniWrapper.family: QwenOmniWrapper,
    QwenVlWrapper.family: QwenVlWrapper,
}


def register_wrapper(family: str, wrapper_cls: WrapperFactory) -> None:
    REGISTRY[family] = wrapper_cls


def get_wrapper(family: str) -> WrapperFactory:
    return REGISTRY[family]


def create_wrapper(family: str, **kwargs: object) -> BaseModelWrapper:
    return get_wrapper(family)(**kwargs)
