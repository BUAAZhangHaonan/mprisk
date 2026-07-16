from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from mprisk.representation.training import load_training_config
from scripts.generate_seed_representation_configs import (
    MODELS,
    PROMPT_FILES,
    REPRESENTATIONS,
    generate_configs,
)


def test_generator_locks_every_seed_model_and_representation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    generated = generate_configs(repo_root, tmp_path)
    assert len(generated) == 27
    assert len(set(generated)) == 27
    for seed, prompt_files in PROMPT_FILES.items():
        for model_key, protocol in MODELS.items():
            prompt_path = repo_root / prompt_files[protocol]
            prompt_payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
            expected_ids = tuple(row["prompt_id"] for row in prompt_payload["templates"])
            for repr_key in REPRESENTATIONS:
                path = tmp_path / f"seed{seed}" / f"{model_key}_{repr_key}.yaml"
                config = load_training_config(path)
                assert config.seed == seed
                assert config.model_key == model_key
                assert config.protocol == protocol
                assert config.classification_objective == (
                    "proxy_anchor_only"
                    if repr_key == "tme_proxy_anchor_v1"
                    else "inverse_frequency_cross_entropy"
                )
                assert config.repr_key == repr_key
                assert config.prompt_set_key == prompt_payload["key"]
                assert config.expected_prompt_ids == expected_ids
                assert (
                    config.prompt_set_artifact_sha256
                    == hashlib.sha256(prompt_path.read_bytes()).hexdigest()
                )
