from __future__ import annotations

from pathlib import Path

import pytest

from mprisk.prompts.compiler import compile_prompt
from mprisk.prompts.template_bank import EquivPromptSet, load_equiv_prompt_set


@pytest.mark.parametrize(
    ("path", "key", "protocol"),
    [
        ("configs/prompts/equiv_sets/vt_primary_v1.yaml", "vt_primary_v1", "vt"),
        ("configs/prompts/equiv_sets/va_aux_v1.yaml", "va_aux_v1", "va"),
        ("configs/prompts/equiv_sets/it_aux_v1.yaml", "it_aux_v1", "it"),
    ],
)
def test_loads_equiv_prompt_set_from_yaml(path: str, key: str, protocol: str) -> None:
    prompt_set = load_equiv_prompt_set(Path(path))

    assert isinstance(prompt_set, EquivPromptSet)
    assert prompt_set.key == key
    assert prompt_set.protocol == protocol
    assert prompt_set.version == "v1"
    assert prompt_set.active is True
    assert len(prompt_set.enabled_templates()) >= 2
    assert {template.role for template in prompt_set.enabled_templates()} == {"user"}


def test_compiler_renders_template_with_sample_fields() -> None:
    prompt_set = load_equiv_prompt_set(Path("configs/prompts/equiv_sets/va_aux_v1.yaml"))
    template = prompt_set.enabled_templates()[0]

    rendered = compile_prompt(
        template,
        {
            "sample_text": "A short clip shows a person speaking calmly.",
            "question": "What emotion is most likely present?",
        },
    )

    assert "A short clip shows a person speaking calmly." in rendered
    assert "What emotion is most likely present?" in rendered
