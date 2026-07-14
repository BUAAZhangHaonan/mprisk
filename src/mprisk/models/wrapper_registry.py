"""Model wrapper registry."""

from __future__ import annotations

from typing import TypeAlias

from mprisk.models.base_wrapper import BaseModelWrapper
from mprisk.models.qwen_omni import QwenOmniWrapper

WrapperFactory: TypeAlias = type[BaseModelWrapper]

REGISTRY: dict[str, WrapperFactory] = {QwenOmniWrapper.family: QwenOmniWrapper}


def register_wrapper(family: str, wrapper_cls: WrapperFactory) -> None:
    REGISTRY[family] = wrapper_cls


def get_wrapper(family: str) -> WrapperFactory:
    return REGISTRY[family]


def create_wrapper(family: str, **kwargs: object) -> BaseModelWrapper:
    return get_wrapper(family)(**kwargs)
