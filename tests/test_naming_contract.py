from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _active_text_files() -> list[Path]:
    roots = (ROOT / "src/mprisk", ROOT / "scripts", ROOT / "configs")
    suffixes = {".py", ".yaml", ".yml"}
    return [
        path
        for base in roots
        for path in base.rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and "legacy" not in path.relative_to(ROOT).parts
    ]


def test_removed_task_specific_entrypoints_do_not_reappear() -> None:
    forbidden_terms = (
        "_".join(("qwen", "omni", "m12")),
        "_".join(("prompt", "context", "v2")),
    )
    for path in _active_text_files():
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in relative
            assert term not in text


def test_gt_semantic_fields_use_scenario_context() -> None:
    ground_truth_source = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "src/mprisk/ground_truth/annotation_inputs.py",
            "src/mprisk/ground_truth/deepseek_gt.py",
        )
    )
    legacy_field = "_".join(("context", "text"))
    assert legacy_field not in ground_truth_source
    assert "scenario_context" in ground_truth_source
    assert "gt_input_schema_version" in ground_truth_source


def test_label_schemas_do_not_treat_conditions_as_modalities() -> None:
    for relative in (
        "configs/labels/sample_type_schema.yaml",
        "configs/labels/stage1_emotion_schema.yaml",
    ):
        payload = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        assert "dominant_modality" not in payload
        assert payload["joint_lean_direction"] == ["V", "T_or_A", "No-lean", "unclear"]
    stage_schema = yaml.safe_load(
        (ROOT / "configs/labels/stage1_emotion_schema.yaml").read_text(encoding="utf-8")
    )
    assert "m12_label" in stage_schema["fields"]
    assert "joint_label" not in stage_schema["fields"]


def test_active_paths_use_task_level_names() -> None:
    required = (
        "src/mprisk/diagnostic_affect/generation.py",
        "src/mprisk/ground_truth/annotation_inputs.py",
        "scripts/build_gt_annotation_input_pilot.py",
        "scripts/run_gt_description_generation.py",
        "configs/ground_truth/gt_description_generation_pilot_v1.yaml",
        "docs/NAMING_CONVENTIONS.md",
    )
    assert all((ROOT / relative).is_file() for relative in required)
