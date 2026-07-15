from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


MODELS = {
    "qwen3_vl_8b": "vt",
    "internvl3_5_8b": "vt",
    "qwen2_5_omni_7b": "va",
}
REPRESENTATIONS = (
    "single_point_binary_v1",
    "trajectory_mlp_binary_v1",
    "tme_proxy_anchor_v1",
)
PROMPT_FILES = {
    20260715: {
        "vt": "configs/prompts/equiv_sets/vt_p8_seed20260715.yaml",
        "va": "configs/prompts/equiv_sets/va_p8_seed20260715.yaml",
    },
    20260716: {
        "vt": "configs/prompts/equiv_sets/vt_p8_seed20260716.yaml",
        "va": "configs/prompts/equiv_sets/va_p8_seed20260716.yaml",
    },
    20260717: {
        "vt": "configs/prompts/equiv_sets/vt_main_p8_seed20260717.yaml",
        "va": "configs/prompts/equiv_sets/va_main_p8_seed20260717.yaml",
    },
}


def generate_configs(repo_root: Path, output_dir: Path) -> list[Path]:
    written: list[Path] = []
    for seed, protocol_files in PROMPT_FILES.items():
        for model_key, protocol in MODELS.items():
            prompt_path = repo_root / protocol_files[protocol]
            prompt_payload = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
            templates = prompt_payload.get("templates") or []
            prompt_ids = [str(row["prompt_id"]) for row in templates if row.get("enabled", True)]
            if len(prompt_ids) != 8 or len(set(prompt_ids)) != 8:
                raise ValueError(f"{prompt_path} must define exactly eight unique prompts")
            prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
            for repr_key in REPRESENTATIONS:
                payload: dict[str, Any] = {
                    "schema": "mprisk_representation_training_v3",
                    "key": f"{model_key}_{repr_key}_seed{seed}",
                    "architecture_version": (
                        "layer_l2_gru_linear_relation_v1"
                        if repr_key == "tme_proxy_anchor_v1"
                        else repr_key
                    ),
                    "repr_key": repr_key,
                    "model_key": model_key,
                    "prompt_set_key": str(prompt_payload["key"]),
                    "prompt_set_artifact_sha256": prompt_sha,
                    "expected_prompt_count": 8,
                    "expected_prompt_ids": prompt_ids,
                    "hidden_dim": 128,
                    "condition_dim": 64,
                    "relation_dim": 32,
                    "dropout": 0.1,
                    "max_epochs": 200,
                    "batch_size": 32,
                    "lr": 0.001,
                    "weight_decay": 0.0001,
                    "proxy_alpha": 32.0,
                    "proxy_margin": 0.1,
                    "patience": 20,
                    "min_delta": 0.0001,
                    "seed": seed,
                }
                destination = output_dir / f"seed{seed}" / f"{model_key}_{repr_key}.yaml"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                written.append(destination)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate immutable model x seed x representation training configs."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/experiments/seed_runs"),
    )
    args = parser.parse_args()
    paths = generate_configs(args.repo_root.resolve(), args.output_dir)
    print(f"generated={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
