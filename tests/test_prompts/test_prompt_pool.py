from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from mprisk.prompts.pool import (
    CANONICAL_PROMPT,
    GLOBAL_POOL_SIZE,
    RAW_POOL_SIZE,
    SUBSET_SEEDS,
    SUBSET_SIZE,
    build_prompt_pool,
    filter_candidates,
    generate_seeded_subset,
    select_global_pool,
)
from scripts.generate_prompt_candidates import build_generation_plan

EXPECTED_CANONICAL = (
    "Based on the complete input, describe the person's overall emotional state in one concise "
    "sentence. Do not address the person, give advice, or explain your reasoning."
)


SAFE_WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "ember",
    "forest",
    "garden",
    "harbor",
    "island",
    "jungle",
    "kernel",
    "lantern",
    "meadow",
    "nectar",
    "orange",
    "prairie",
    "quartz",
    "river",
    "silver",
    "timber",
    "umber",
    "velvet",
    "willow",
    "xenon",
    "yellow",
    "zephyr",
]


def _candidate(text: str, *, review: bool = True) -> dict[str, object]:
    return {"template_text": text, "ai_semantic_review_pass": review}


def _valid_prompt(index: int) -> str:
    left = SAFE_WORDS[index % len(SAFE_WORDS)]
    right = SAFE_WORDS[(index // len(SAFE_WORDS)) % len(SAFE_WORDS)]
    return (
        "Summarize the person's overall inner state in one concise sentence using "
        f"{left} {right} wording."
    )


def _raw_candidates(count: int = RAW_POOL_SIZE) -> list[dict[str, object]]:
    return [_candidate(_valid_prompt(index)) for index in range(count)]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_canonical_prompt_is_exact() -> None:
    assert CANONICAL_PROMPT == EXPECTED_CANONICAL


def test_generation_plan_skeleton_records_canonical_prompt_and_decoding() -> None:
    plan = build_generation_plan(
        model_path="/models/local-small",
        output_path="data/processed/prompt_banks/pregen_risk_v1/raw384.jsonl",
        seed=20260715,
        temperature=0.8,
        top_p=0.9,
    )

    assert plan.count == RAW_POOL_SIZE
    assert plan.canonical_prompt == EXPECTED_CANONICAL
    assert plan.model_path == "/models/local-small"
    assert plan.temperature == 0.8
    assert plan.top_p == 0.9


def test_prompt_selection_and_equiv_configs_use_prompt_pool_contract() -> None:
    selection = yaml.safe_load(
        Path("configs/prompts/prompt_selection.yaml").read_text(encoding="utf-8")
    )

    assert selection["canonical_prompt"] == EXPECTED_CANONICAL
    assert selection["default_p"] == SUBSET_SIZE
    assert selection["global_pool_size"] == GLOBAL_POOL_SIZE
    assert selection["subset_seeds"] == list(SUBSET_SEEDS)
    assert selection["raw_candidate_count"] == RAW_POOL_SIZE

    for path in sorted(Path("configs/prompts/equiv_sets").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["canonical_prompt"] == EXPECTED_CANONICAL


def test_filter_candidates_normalizes_dedupes_and_records_rejections() -> None:
    rows = [
        _candidate("  Summarize   the person's overall inner state in one concise sentence. "),
        _candidate("Summarize the person's overall inner state in one concise sentence."),
        _candidate("Describe the person's happy mood in one concise sentence."),
        _candidate("Describe the person's overall state from the image in one concise sentence."),
        _candidate("Describe the person's overall state in one concise sentence."),
        _candidate("Question: describe the person's overall state in one concise sentence."),
        _candidate("Describe the person's overall state in one concise sentence.", review=False),
    ]

    accepted, rejections = filter_candidates(rows)

    assert accepted == ["Summarize the person's overall inner state in one concise sentence."]
    assert [rejection["reason"] for rejection in rejections] == [
        "duplicate",
        "forbidden_term",
        "forbidden_term",
        "word_count",
        "placeholder_or_leakage",
        "semantic_review_not_passed",
    ]


def test_select_global_pool_uses_tfidf_word_and_char_features() -> None:
    prompts = [_valid_prompt(index) for index in range(130)]

    selected = select_global_pool(prompts, pool_size=GLOBAL_POOL_SIZE)
    selected_again = select_global_pool(prompts, pool_size=GLOBAL_POOL_SIZE)

    assert len(selected) == GLOBAL_POOL_SIZE
    assert len(set(selected)) == GLOBAL_POOL_SIZE
    assert selected == selected_again
    assert set(selected).issubset(set(prompts))


def test_seeded_subsets_are_deterministic_and_drawn_from_global_pool() -> None:
    pool = [_valid_prompt(index) for index in range(GLOBAL_POOL_SIZE)]

    subset = generate_seeded_subset(pool, seed=SUBSET_SEEDS[0], subset_size=SUBSET_SIZE)
    subset_again = generate_seeded_subset(pool, seed=SUBSET_SEEDS[0], subset_size=SUBSET_SIZE)
    other_subset = generate_seeded_subset(pool, seed=SUBSET_SEEDS[1], subset_size=SUBSET_SIZE)

    assert subset == subset_again
    assert subset != other_subset
    assert len(subset) == SUBSET_SIZE
    assert set(subset).issubset(set(pool))


def test_build_prompt_pool_requires_exact_raw_count(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    _write_jsonl(raw_path, _raw_candidates(RAW_POOL_SIZE - 1))

    with pytest.raises(ValueError, match="exactly 384"):
        build_prompt_pool(raw_path, tmp_path / "out")


def test_build_prompt_pool_fails_when_accepted_candidates_are_less_than_128(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.jsonl"
    rows = [_candidate(_valid_prompt(index % 120)) for index in range(RAW_POOL_SIZE)]
    _write_jsonl(raw_path, rows)

    with pytest.raises(ValueError, match="at least 128 accepted"):
        build_prompt_pool(raw_path, tmp_path / "out")


def test_build_prompt_pool_writes_counts_provenance_and_placeholder_free_exports(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.jsonl"
    output_dir = tmp_path / "pool"
    _write_jsonl(raw_path, _raw_candidates())

    result = build_prompt_pool(raw_path, output_dir)

    assert len(result.raw384) == RAW_POOL_SIZE
    assert len(result.pool128) == GLOBAL_POOL_SIZE
    assert len(result.rejections) == 0
    assert set(result.subsets) == set(SUBSET_SEEDS)
    assert all(len(subset) == SUBSET_SIZE for subset in result.subsets.values())
    assert result.provenance["canonical_prompt"] == EXPECTED_CANONICAL
    assert result.provenance["raw_count"] == RAW_POOL_SIZE
    assert result.provenance["global_pool_size"] == GLOBAL_POOL_SIZE

    exported_pool = [
        json.loads(line)
        for line in (output_dir / "pool128.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(exported_pool) == GLOBAL_POOL_SIZE
    assert set(exported_pool[0]) == {"prompt_id", "template_text", "role", "enabled"}
    assert all(row["role"] == "user" and row["enabled"] is True for row in exported_pool)
    assert not any(
        "{" in row["template_text"] or "}" in row["template_text"] for row in exported_pool
    )

    for seed in SUBSET_SEEDS:
        subset_yaml = yaml.safe_load(
            (output_dir / f"subset_p8_seed{seed}.yaml").read_text(encoding="utf-8")
        )
        assert subset_yaml["canonical_prompt"] == EXPECTED_CANONICAL
        assert len(subset_yaml["templates"]) == SUBSET_SIZE
        assert all(
            "{" not in template["template_text"] and "}" not in template["template_text"]
            for template in subset_yaml["templates"]
        )
