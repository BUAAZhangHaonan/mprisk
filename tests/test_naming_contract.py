from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXACT_INACTIVE_ROOTS = (("configs", "legacy"),)
IMMUTABLE_V1_SCHEMAS = {
    "configs/labels/sample_type_schema.yaml",
    "configs/labels/stage1_emotion_schema.yaml",
}
FROZEN_LEGACY_V1_CONSUMERS = {
    "src/mprisk/data/state_bundle.py",
    "src/mprisk/data/state_dataset.py",
}
LEGACY_V1_DOC = "docs/ANNOTATION_GUIDE.md"
LEGACY_V1_START = "<!-- naming-contract: legacy-v1-start -->"
LEGACY_V1_END = "<!-- naming-contract: legacy-v1-end -->"
LEGACY_PUBLIC_FIELDS = (
    "joint_label",
    "joint_specific_affect",
    "joint_is_clear",
    "joint_confidence",
    "dominant_modality",
)


def _is_exact_inactive_path(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    return any(parts[: len(root)] == root for root in EXACT_INACTIVE_ROOTS)


def _active_text_files() -> list[Path]:
    roots = (ROOT / "src/mprisk", ROOT / "scripts", ROOT / "configs")
    suffixes = {".py", ".yaml", ".yml"}
    return [
        path
        for base in roots
        for path in base.rglob("*")
        if path.is_file() and path.suffix in suffixes and not _is_exact_inactive_path(path)
    ]


def _documentation_files() -> list[Path]:
    return sorted(path for path in (ROOT / "docs").rglob("*.md") if path.is_file())


def _without_allowed_legacy_doc_block(path: Path, text: str) -> str:
    relative = path.relative_to(ROOT).as_posix()
    if relative != LEGACY_V1_DOC:
        assert LEGACY_V1_START not in text
        assert LEGACY_V1_END not in text
        return text

    assert text.count(LEGACY_V1_START) == 1
    assert text.count(LEGACY_V1_END) == 1
    start = text.index(LEGACY_V1_START)
    end = text.index(LEGACY_V1_END, start) + len(LEGACY_V1_END)
    return text[:start] + text[end:]


def test_removed_task_specific_entrypoints_do_not_reappear() -> None:
    forbidden_terms = (
        "_".join(("qwen", "omni", "m12")),
        "_".join(("prompt", "context", "v2")),
    )
    for path in [*_active_text_files(), *_documentation_files()]:
        relative = path.relative_to(ROOT).as_posix()
        text = (
            _without_allowed_legacy_doc_block(path, path.read_text(encoding="utf-8"))
            if path.suffix == ".md"
            else path.read_text(encoding="utf-8")
        )
        for term in forbidden_terms:
            assert term not in relative
            assert term not in text


def test_legacy_public_fields_are_confined_to_exact_v1_contracts() -> None:
    exact_file_exceptions = IMMUTABLE_V1_SCHEMAS | FROZEN_LEGACY_V1_CONSUMERS
    for path in [*_active_text_files(), *_documentation_files()]:
        relative = path.relative_to(ROOT).as_posix()
        if relative in exact_file_exceptions:
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".md":
            text = _without_allowed_legacy_doc_block(path, text)
        for term in LEGACY_PUBLIC_FIELDS:
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])"
            assert re.search(pattern, text) is None, f"{term} leaked into {relative}"

    state_dataset = (ROOT / "src/mprisk/data/state_dataset.py").read_text(encoding="utf-8")
    state_bundle = (ROOT / "src/mprisk/data/state_bundle.py").read_text(encoding="utf-8")
    assert state_dataset.count('"dominant_modality"') == 2
    assert state_dataset.count('"joint_label"') == 2
    assert state_bundle.count('"dominant_modality"') == 1


def test_v1_label_schemas_remain_immutable_and_v2_is_explicit() -> None:
    stage_v1 = yaml.safe_load(
        (ROOT / "configs/labels/stage1_emotion_schema.yaml").read_text(encoding="utf-8")
    )
    relation_v1 = yaml.safe_load(
        (ROOT / "configs/labels/sample_type_schema.yaml").read_text(encoding="utf-8")
    )
    assert stage_v1["schema"] == "mprisk_stage1_relation_schema_v1"
    assert "joint_label" in stage_v1["fields"]
    assert "m12_label" not in stage_v1["fields"]
    assert stage_v1["dominant_modality"] == ["M1", "M2", "balanced", "unclear"]
    assert relation_v1["schema"] == "mprisk_sample_type_schema_v1"
    assert "joint_label" in relation_v1["sample_types"]["Conflict"]["required"][4]
    assert relation_v1["dominant_modality"] == ["M1", "M2", "balanced", "unclear"]

    affect_v2 = yaml.safe_load(
        (ROOT / "configs/labels/condition_affect_annotation_schema_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    relation_v2 = yaml.safe_load(
        (ROOT / "configs/labels/sample_relation_schema_v2.yaml").read_text(encoding="utf-8")
    )
    assert affect_v2["schema"] == "mprisk_condition_affect_annotation_schema_v2"
    assert "m12_label" in affect_v2["fields"]
    assert "joint_label" not in affect_v2["fields"]
    assert "joint_lean_direction" not in affect_v2
    assert affect_v2["reference_dominant_modality"] == [
        "V",
        "T",
        "A",
        "Balanced",
        "Unclear",
    ]
    assert relation_v2["schema"] == "mprisk_sample_relation_schema_v2"
    assert "m12_label" in relation_v2["sample_types"]["Conflict"]["required"][4]
    assert "joint_lean_direction" not in relation_v2
    assert relation_v2["reference_dominant_modality"] == [
        "V",
        "T",
        "A",
        "Balanced",
        "Unclear",
    ]


def test_running_state_pipeline_does_not_auto_select_v2() -> None:
    runtime_text = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in sorted(FROZEN_LEGACY_V1_CONSUMERS)
    )
    for forbidden in (
        "condition_affect_annotation_schema_v2",
        "sample_relation_schema_v2",
        "mprisk_condition_affect_annotation_schema_v2",
        "mprisk_sample_relation_schema_v2",
    ):
        assert forbidden not in runtime_text


def test_gt_semantic_fields_use_scenario_context() -> None:
    ground_truth_source = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "src/mprisk/ground_truth/annotation_inputs.py",
            "src/mprisk/ground_truth/description_generation.py",
        )
    )
    legacy_field = "_".join(("context", "text"))
    assert legacy_field not in ground_truth_source
    assert "scenario_context" in ground_truth_source
    assert "gt_input_schema_version" in ground_truth_source


def test_active_paths_use_task_level_names() -> None:
    required = (
        "src/mprisk/diagnostic_affect/generation.py",
        "src/mprisk/judge/misread_judgment.py",
        "src/mprisk/ground_truth/annotation_inputs.py",
        "src/mprisk/ground_truth/description_generation.py",
        "src/mprisk/ground_truth/providers/base.py",
        "src/mprisk/ground_truth/providers/registry.py",
        "src/mprisk/ground_truth/providers/deepseek.py",
        "scripts/build_gt_annotation_input_pilot.py",
        "scripts/run_gt_description_generation.py",
        "configs/ground_truth/gt_description_generation_pilot_v3.yaml",
        "configs/experiments/diagnostic_affect_description_v2.yaml",
        "configs/judge/misread_judgment_v2.yaml",
        "configs/labels/condition_affect_annotation_schema_v2.yaml",
        "configs/labels/sample_relation_schema_v2.yaml",
        "docs/NAMING_CONVENTIONS.md",
    )
    assert all((ROOT / relative).is_file() for relative in required)


def test_active_judgment_and_diagnostic_paths_have_no_legacy_aliases() -> None:
    forbidden = (
        "src/mprisk/judge/reference_guided.py",
        "scripts/run_reference_guided_judge.py",
        "configs/judge/reference_guided_misread_v1.yaml",
        "configs/experiments/diagnostic_affect_description_v1.yaml",
    )
    assert all(not (ROOT / relative).exists() for relative in forbidden)
    judge_source = (ROOT / "src/mprisk/judge/misread_judgment.py").read_text(
        encoding="utf-8"
    )
    assert '"DIAGNOSTIC_AFFECT_DESCRIPTION"' in judge_source
    assert "diagnostic_affect_description_manifest_path" in judge_source


def test_gt_description_task_contract_is_not_vendor_scoped() -> None:
    task_source = (
        ROOT / "src/mprisk/ground_truth/description_generation.py"
    ).read_text(encoding="utf-8")
    provider_source = (
        ROOT / "src/mprisk/ground_truth/providers/deepseek.py"
    ).read_text(encoding="utf-8")
    for symbol in (
        "class GTDescriptionGenerationConfig",
        "class GTDescriptionGenerationTask",
        "class GTDescriptionGenerationResult",
        "class GTDescriptionGenerationLedger",
        "def prepare_tasks",
        "def run_gt_description_generation",
        "def verify_gt_description_generation",
    ):
        assert symbol in task_source
        assert symbol not in provider_source
    for vendor_symbol in (
        "DeepSeek",
        "DEEPSEEK_API_KEY",
        "api_url",
        "api_key_env",
        "thinking",
    ):
        assert vendor_symbol not in task_source
    registry_source = (
        ROOT / "src/mprisk/ground_truth/providers/registry.py"
    ).read_text(encoding="utf-8")
    assert "def get_provider" in registry_source
    assert '"deepseek"' in registry_source
    assert not (ROOT / "src/mprisk/ground_truth/deepseek_gt.py").exists()
