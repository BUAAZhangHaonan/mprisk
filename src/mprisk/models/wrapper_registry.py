"""Model wrapper registry."""

from __future__ import annotations

from typing import TypeAlias

from mprisk.models.base_wrapper import BaseModelWrapper
from mprisk.models.internvl import InternVlWrapper
from mprisk.models.qwen_omni import QwenOmniWrapper
from mprisk.models.qwen_vl import QwenVlWrapper

WrapperFactory: TypeAlias = type[BaseModelWrapper]

REGISTRY: dict[str, WrapperFactory] = {
    InternVlWrapper.family: InternVlWrapper,
    QwenOmniWrapper.family: QwenOmniWrapper,
    QwenVlWrapper.family: QwenVlWrapper,
}


def register_wrapper(family: str, wrapper_cls: WrapperFactory) -> None:
    REGISTRY[family] = wrapper_cls


def get_wrapper(family: str) -> WrapperFactory:
    return REGISTRY[family]


def create_wrapper(family: str, **kwargs: object) -> BaseModelWrapper:
    return get_wrapper(family)(**kwargs)
