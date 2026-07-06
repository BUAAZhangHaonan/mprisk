from __future__ import annotations

from mprisk.prompts.template_bank import PromptTemplate


def test_prompt_template_renders_values() -> None:
    template = PromptTemplate(prompt_id="q001", protocol="vt", text="Sample: {sample_id}")
    assert template.render(sample_id="x") == "Sample: x"
