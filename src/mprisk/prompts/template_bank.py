"""Prompt template bank contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mprisk.config.loader import load_yaml


@dataclass(frozen=True, init=False)
class PromptTemplate:
    prompt_id: str
    template_text: str
    role: str
    enabled: bool
    protocol: str | None

    def __init__(
        self,
        prompt_id: str,
        protocol: str | None = None,
        text: str | None = None,
        *,
        template_text: str | None = None,
        role: str = "user",
        enabled: bool = True,
    ) -> None:
        resolved_text = template_text if template_text is not None else text
        if resolved_text is None:
            raise ValueError(f"PromptTemplate {prompt_id} requires template_text")
        object.__setattr__(self, "prompt_id", prompt_id)
        object.__setattr__(self, "template_text", resolved_text)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "protocol", protocol)

    @property
    def text(self) -> str:
        return self.template_text

    def render(self, **values: object) -> str:
        return self.template_text.format(**values)


@dataclass(frozen=True)
class EquivPromptSet:
    key: str
    protocol: str
    templates: list[PromptTemplate]
    version: str
    active: bool

    def enabled_templates(self) -> list[PromptTemplate]:
        return [template for template in self.templates if template.enabled]


def load_equiv_prompt_set(path: str | Path) -> EquivPromptSet:
    data = load_yaml(path)
    key = _required_str(data, "key", path)
    protocol = _required_str(data, "protocol", path)
    version = str(data.get("version", "v1"))
    active = bool(data.get("active", True))
    raw_templates = data.get("templates")
    if raw_templates == []:
        raise ValueError(f"Prompt set {key} templates are empty")
    if not isinstance(raw_templates, list):
        raise ValueError(f"Prompt set {key} requires templates as a non-empty list")

    templates = [
        _load_template(raw_template, prompt_set_key=key, protocol=protocol, index=index)
        for index, raw_template in enumerate(raw_templates)
    ]
    return EquivPromptSet(
        key=key,
        protocol=protocol,
        version=version,
        active=active,
        templates=templates,
    )


def _load_template(
    raw_template: Any,
    *,
    prompt_set_key: str,
    protocol: str,
    index: int,
) -> PromptTemplate:
    if not isinstance(raw_template, dict):
        raise ValueError(f"Template {index} in {prompt_set_key} must be a mapping")
    if "prompt_id" not in raw_template:
        raise ValueError(f"Template template[{index}] in {prompt_set_key} is missing prompt_id")
    prompt_id = _required_template_str(
        raw_template,
        "prompt_id",
        prompt_set_key,
        f"template[{index}]",
    )
    template_text = _required_template_str(raw_template, "template_text", prompt_set_key, prompt_id)
    role = _required_template_str(raw_template, "role", prompt_set_key, prompt_id)
    if "enabled" not in raw_template:
        raise ValueError(f"Template {prompt_id} in {prompt_set_key} is missing enabled")
    return PromptTemplate(
        prompt_id=prompt_id,
        protocol=protocol,
        template_text=template_text,
        role=role,
        enabled=bool(raw_template["enabled"]),
    )


def _required_str(data: dict[str, Any], field: str, path: str | Path) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Prompt set {path} requires {field}")
    return value


def _required_template_str(
    raw_template: dict[str, Any],
    field: str,
    prompt_set_key: str,
    prompt_id: str,
) -> str:
    value = raw_template.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Template {prompt_id} in {prompt_set_key} is missing {field}")
    return value
