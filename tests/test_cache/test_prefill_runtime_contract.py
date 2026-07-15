from __future__ import annotations

import json
from pathlib import Path

import yaml

from mprisk.assets.registry import index_assets, load_model_assets
from mprisk.cache.prefill_batch import build_batch_plan, build_parser
from mprisk.models.internvl import InternVlWrapper
from mprisk.models.qwen_omni import QwenOmniWrapper
from mprisk.models.qwen_vl import QwenVlWrapper
from mprisk.models.wrapper_registry import get_wrapper
from mprisk.prompts.template_bank import load_equiv_prompt_set

ASSET_CONFIG = Path("configs/assets/model_assets.yaml")
PROMPT_CONFIG = Path("configs/cache/prefill_main_p8_v1.yaml")
SMOKE_ROOT = Path("data/frozen/prefill_smoke_v1")


def test_fixed_prefill_models_resolve_from_versioned_asset_config() -> None:
    assets = index_assets(load_model_assets(ASSET_CONFIG, require_local_paths=True))

    assert assets["qwen3_vl_8b"].local_path == (
        "/home/team/lvshuyang/Models/Qwen3-VL-8B-Instruct"
    )
    assert assets["internvl3_5_8b"].local_path == (
        "/home/team/lvshuyang/Models/InternVL3_5-8B"
    )
    assert assets["qwen2_5_omni_7b"].local_path == (
        "/home/team/lvshuyang/Models/Qwen/Qwen2.5-Omni-7B"
    )
    assert get_wrapper("qwen_vl") is QwenVlWrapper
    assert get_wrapper("internvl") is InternVlWrapper
    assert get_wrapper("qwen_omni") is QwenOmniWrapper


def test_main_prompt_config_freezes_p8_seed_for_vt_and_va() -> None:
    config = yaml.safe_load(PROMPT_CONFIG.read_text(encoding="utf-8"))
    assert config["schema"] == "mprisk_prefill_main_p8_v1"
    assert config["seed"] == 20260717
    assert config["seed_semantics"] == "immutable_prompt_subset_selection_seed_not_run_date"
    assert set(config["models"]) == {
        "qwen3_vl_8b",
        "internvl3_5_8b",
        "qwen2_5_omni_7b",
    }
    for protocol in ("vt", "va"):
        prompt_path = Path(config["prompt_sets"][protocol])
        prompt_provenance = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
        assert prompt_provenance["selection_seed"] == 20260717
        assert prompt_provenance["selection_seed_semantics"] == (
            "immutable_prompt_subset_selection_seed_not_run_date"
        )
        prompt_set = load_equiv_prompt_set(prompt_path)
        assert prompt_set.protocol == protocol
        assert len(prompt_set.enabled_templates()) == 8


def test_each_model_smoke_manifest_has_one_conflict_and_one_aligned() -> None:
    expected_protocol = {
        "qwen3_vl_8b": "VT",
        "internvl3_5_8b": "VT",
        "qwen2_5_omni_7b": "VA",
    }
    for model_key, protocol in expected_protocol.items():
        rows = [
            json.loads(line)
            for line in (SMOKE_ROOT / f"{model_key}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 2
        assert {row["sample_type"] for row in rows} == {"Conflict", "Aligned"}
        assert {row["protocol"] for row in rows} == {protocol}
        assert all(row["use_in_main"] is True for row in rows)


def test_smoke_plan_is_exactly_six_condition_requests_times_p8() -> None:
    config = yaml.safe_load(PROMPT_CONFIG.read_text(encoding="utf-8"))
    args = build_parser().parse_args(
        [
            "--manifest",
            str(SMOKE_ROOT / "qwen3_vl_8b.jsonl"),
            "--prompt-set",
            config["prompt_sets"]["vt"],
            "--protocol",
            "vt",
            "--model-key",
            "qwen3_vl_8b",
            "--output-root",
            "/tmp/prefill-smoke",
        ]
    )
    plan = build_batch_plan(args)

    assert len(plan.tasks) == 6 * 8
    assert {task.prompt_id for task in plan.tasks} == set(plan.prompt_ids)
    assert all(task.row["sample_type"] != "Misread" for task in plan.tasks)
    assert plan.signature["family"] == "qwen_vl"
    assert plan.signature["model_path"] == (
        "/home/team/lvshuyang/Models/Qwen3-VL-8B-Instruct"
    )
