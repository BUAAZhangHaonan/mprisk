"""Model wrapper registry."""

from __future__ import annotations

from typing import TypeAlias

from mprisk.models.base_wrapper import BaseModelWrapper

WrapperFactory: TypeAlias = type[BaseModelWrapper]

REGISTRY: dict[str, WrapperFactory] = {}


def register_wrapper(family: str, wrapper_cls: WrapperFactory) -> None:
    REGISTRY[family] = wrapper_cls


def get_wrapper(family: str) -> WrapperFactory:
    return REGISTRY[family]
