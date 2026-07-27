"""Generate per-model tme_sdr YAML for cache_matrix_20260722 C/A training.

Uses qwen3_vl_8b_tme_sdr.yaml as the canonical template. Only model_key,
layer_count, input_dim vary per model; all hyperparameters are held constant
so this is a controlled model-vs-model comparison.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]

# 13 valid models (drop phi3_5_vision / phi4_multimodal / llava_v1_5_7b)
MODEL_META = {
    # VT (11)
    "gemma3_12b":               {"proto": "vt", "layer_count": 48, "hidden_dim": 3840},
    "gemma3_4b":                {"proto": "vt", "layer_count": 34, "hidden_dim": 2560},
    "glm4_6v_flash":            {"proto": "vt", "layer_count": 40, "hidden_dim": 4096},
    "llava_onevision_qwen2_7b": {"proto": "vt", "layer_count": 28, "hidden_dim": 3584},
    "minicpm_v_2_6":            {"proto": "vt", "layer_count": 28, "hidden_dim": 3584},
    "minicpm_v_4_5":            {"proto": "vt", "layer_count": 36, "hidden_dim": 4096},
    "internvl3_5_8b":           {"proto": "vt", "layer_count": 36, "hidden_dim": 4096},
    "qwen2_5_vl_7b":            {"proto": "vt", "layer_count": 28, "hidden_dim": 3584},
    "qwen3_5_4b":               {"proto": "vt", "layer_count": 32, "hidden_dim": 2560},
    "qwen3_5_9b":               {"proto": "vt", "layer_count": 32, "hidden_dim": 4096},
    "qwen3_vl_8b":              {"proto": "vt", "layer_count": 36, "hidden_dim": 4096},
    # VA (2)
    "gemma4_12b":               {"proto": "va", "layer_count": 48, "hidden_dim": 3840},
    "qwen2_5_omni_7b":          {"proto": "va", "layer_count": 28, "hidden_dim": 3584},
}

# Prompt-set SHA-256 (computed once; both proto share the same prompt IDs
# but different YAML files, so hashes differ).
HASHES = {
    "vt": hashlib.sha256(
        (REPO / "configs/prompts/equiv_sets/vt_main_p8_seed20260717.yaml").read_bytes()
    ).hexdigest(),
    "va": hashlib.sha256(
        (REPO / "configs/prompts/equiv_sets/va_main_p8_seed20260717.yaml").read_bytes()
    ).hexdigest(),
}

EXPECTED_PROMPT_IDS = [
    "pregen_risk_v1_p001",
    "pregen_risk_v1_p008",
    "pregen_risk_v1_p012",
    "pregen_risk_v1_p018",
    "pregen_risk_v1_p022",
    "pregen_risk_v1_p054",
    "pregen_risk_v1_p056",
    "pregen_risk_v1_p067",
]

# All hyperparameters are identical to qwen3_vl_8b_tme_sdr.yaml — no per-model tuning.
COMMON = {
    "schema": "mprisk_representation_training_v4",
    "architecture_version": "layer_l2_gru_linear_relation_v1",
    "repr_key": "tme_proxy_anchor_v1",
    "classification_objective": "proxy_anchor_only",
    "expected_prompt_count": 8,
    "expected_prompt_ids": EXPECTED_PROMPT_IDS,
    "seed": 20260717,
    "hidden_dim": 256,
    "condition_dim": 128,
    "relation_dim": 64,
    "dropout": 0.1,
    "encoder_type": "gru",
    "max_epochs": 100,
    "batch_size": 32,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "proxy_alpha": 32.0,
    "proxy_margin": 0.1,
    "sdr_aux_weight": 1.0,
    "sdr_margin_D": 0.6,
    "sdr_margin_R": 0.4,
    "sdr_warmup_epochs": 10,
    "enable_state_supervision": False,
    "d_supervision_weight": 0.0,
    "d_ranking_margin": 0.0,
    "angular_supervision_weight": 0.0,
    "angular_ranking_margin_rad": 0.0,
    "d_aux_samples_per_class": 0,
    "state_selection_min_d_gap": 0.0,
    "state_selection_min_raw_theta_gap_rad": 0.0,
    "state_selection_max_d_mannwhitney_p": 1.0,
    "state_selection_min_d_effect_size": 0.0,
    "patience": 10,
    "min_delta": 1.0e-4,
}


def main() -> int:
    out_dir = REPO / "configs/experiments/cache_matrix_20260722"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    skipped: list[str] = []
    for model, meta in MODEL_META.items():
        proto = meta["proto"]
        cfg = dict(COMMON)
        cfg["key"] = f"cache_matrix_{model}_tme_sdr_v1"
        cfg["model_key"] = model
        cfg["protocol"] = proto  # lowercase "vt"/"va" — validator rejects uppercase
        cfg["prompt_set_key"] = f"{proto}_main_p8_seed20260717"
        cfg["prompt_set_artifact_sha256"] = HASHES[proto]

        out = out_dir / f"{model}_tme_sdr.yaml"
        if out.exists():
            # Re-read existing and compare; skip if identical to keep mtimes stable.
            existing = yaml.safe_load(out.read_text())
            if existing == cfg:
                skipped.append(model)
                continue

        with out.open("w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
        written.append(model)

    print(f"Written: {len(written)} ({', '.join(written) if written else '-'})")
    print(f"Skipped (already matches): {len(skipped)} ({', '.join(skipped) if skipped else '-'})")
    print(f"Prompt set SHAs: vt={HASHES['vt'][:16]} va={HASHES['va'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
