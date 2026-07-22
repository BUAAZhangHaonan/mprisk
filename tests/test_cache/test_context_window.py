from __future__ import annotations

from pathlib import Path

import pytest

from mprisk.cache.context_window import (
    context_ceiling_path,
    validate_sidecar_token_count,
)


@pytest.mark.parametrize(
    ("family", "path"),
    [
        ("gemma3", ("text_config", "max_position_embeddings")),
        ("glm4v", ("text_config", "max_position_embeddings")),
        ("internvl", ("llm_config", "max_position_embeddings")),
        ("llava_v15", ("text_config", "max_position_embeddings")),
        ("llava_onevision", ("text_config", "max_position_embeddings")),
        ("minicpm_v", ("max_position_embeddings",)),
        ("phi3_vision", ("max_position_embeddings",)),
        ("qwen2_5_vl", ("text_config", "max_position_embeddings")),
        ("qwen_vl", ("text_config", "max_position_embeddings")),
        ("qwen3_5", ("text_config", "max_position_embeddings")),
        ("gemma4", ("text_config", "max_position_embeddings")),
        ("phi4_multimodal", ("max_position_embeddings",)),
        (
            "qwen_omni",
            (
                "thinker_config",
                "text_config",
                "max_position_embeddings",
            ),
        ),
    ],
)
def test_context_ceiling_paths_are_family_explicit(
    family: str, path: tuple[str, ...]
) -> None:
    assert context_ceiling_path(family) == path


def test_context_ceiling_path_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="No context ceiling path"):
        context_ceiling_path("unknown")


def test_sidecar_context_validation_accepts_boundary(tmp_path: Path) -> None:
    sidecar = tmp_path / "entry.json"
    assert (
        validate_sidecar_token_count(
            model_key="model",
            sidecar_path=sidecar,
            manifest_token_count=4096,
            sidecar_payload={"entry": {"token_count": 4096}},
            context_ceiling=4096,
        )
        == 4096
    )


def test_sidecar_context_validation_rejects_overflow(tmp_path: Path) -> None:
    sidecar = tmp_path / "onevision.json"
    with pytest.raises(
        ValueError,
        match=(
            r"Context overflow for llava_onevision_qwen2_7b: "
            r"sidecar token_count 58236 exceeds max_position_embeddings 32768"
        ),
    ):
        validate_sidecar_token_count(
            model_key="llava_onevision_qwen2_7b",
            sidecar_path=sidecar,
            manifest_token_count=58236,
            sidecar_payload={"entry": {"token_count": 58236}},
            context_ceiling=32768,
        )


def test_sidecar_context_validation_rejects_manifest_mismatch(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Manifest/sidecar token_count mismatch"):
        validate_sidecar_token_count(
            model_key="model",
            sidecar_path=tmp_path / "entry.json",
            manifest_token_count=128,
            sidecar_payload={"entry": {"token_count": 127}},
            context_ceiling=4096,
        )
