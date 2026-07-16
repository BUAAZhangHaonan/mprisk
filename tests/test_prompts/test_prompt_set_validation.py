from __future__ import annotations

from pathlib import Path

import pytest

from mprisk.prompts.compiler import compile_prompt
from mprisk.prompts.template_bank import PromptTemplate, load_equiv_prompt_set


def test_empty_template_set_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text(
        """
schema: mprisk_equiv_prompt_set_v1
key: empty_v1
protocol: vt
version: v1
active: true
templates: []
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="empty_v1.*templates.*empty"):
        load_equiv_prompt_set(path)


def test_template_missing_required_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad-template.yaml"
    path.write_text(
        """
schema: mprisk_equiv_prompt_set_v1
key: bad_v1
protocol: vt
version: v1
active: true
templates:
  - prompt_id: bad_t01
    role: user
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bad_t01.*template_text"):
        load_equiv_prompt_set(path)


def test_template_missing_prompt_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing-prompt-id.yaml"
    path.write_text(
        """
schema: mprisk_equiv_prompt_set_v1
key: bad_v1
protocol: vt
version: v1
active: true
templates:
  - template_text: "Text: {sample_text}"
    role: user
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="template\\[0\\].*prompt_id"):
        load_equiv_prompt_set(path)


def test_compile_prompt_reports_missing_sample_field() -> None:
    template = PromptTemplate(
        prompt_id="vt_primary_v1_t01",
        template_text="Text: {sample_text}\nQuestion: {question}",
        role="user",
        enabled=True,
    )

    with pytest.raises(ValueError, match="vt_primary_v1_t01.*question"):
        compile_prompt(template, {"sample_text": "The speaker sounds tense."})
