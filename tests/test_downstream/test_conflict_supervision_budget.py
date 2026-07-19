from __future__ import annotations

from collections import Counter

import pytest

from mprisk.experiments.conflict_supervision_budget import (
    FRACTIONS,
    BudgetJob,
    BudgetMethod,
    BudgetPlanError,
    _ac_consolidated_row,
    retained_conflict_rows,
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_index = 0
    for split, sample_type, group_count in (
        ("relation_train", "Aligned", 12),
        ("relation_train", "Conflict", 10),
        ("relation_val", "Aligned", 3),
        ("relation_val", "Conflict", 3),
        ("aligned_calibration", "Aligned", 4),
        ("official_test", "Aligned", 4),
        ("official_test", "Conflict", 4),
    ):
        for group_index in range(group_count):
            group_id = f"{split}:{sample_type}:{group_index}"
            for prompt_index in range(8):
                rows.append(
                    {
                        "row_id": f"row-{row_index}",
                        "sample_id": f"sample-{split}-{sample_type}-{group_index}",
                        "sample_type": sample_type,
                        "split_group_id": group_id,
                        "representation_split": split,
                        "prompt_id": f"p{prompt_index + 1}",
                    }
                )
                row_index += 1
    return rows


def test_budget_subsets_are_nested_and_protect_every_held_out_row() -> None:
    rows = _rows()
    protected_before = {
        row["row_id"] for row in rows if row["representation_split"] != "relation_train"
    }
    previous: set[str] = set()
    expected_counts = {0.10: 1, 0.25: 3, 0.50: 5, 1.00: 10}
    for fraction in FRACTIONS:
        retained, metadata = retained_conflict_rows(rows, fraction=fraction, seed=17)
        kept = set(metadata["retained_conflict_group_ids"])
        assert previous <= kept
        previous = kept
        assert metadata["retained_conflict_group_count"] == expected_counts[fraction]
        assert protected_before <= {row["row_id"] for row in retained}
        train_types = Counter(
            row["sample_type"]
            for row in retained
            if row["representation_split"] == "relation_train"
        )
        assert set(train_types) == {"Aligned", "Conflict"}
    assert len(previous) == 10


def test_budget_subset_is_seed_deterministic() -> None:
    rows = _rows()
    first, first_metadata = retained_conflict_rows(rows, fraction=0.25, seed=20260717)
    second, second_metadata = retained_conflict_rows(rows, fraction=0.25, seed=20260717)

    assert first == second
    assert first_metadata == second_metadata
    assert len(first_metadata["retained_conflict_group_ids_sha256"]) == 64


def test_budget_rejects_unregistered_fraction() -> None:
    with pytest.raises(BudgetPlanError, match="unregistered"):
        retained_conflict_rows(_rows(), fraction=0.20, seed=17)


def test_budget_uses_registered_official_ac_metric_fields(tmp_path) -> None:
    row = _ac_consolidated_row(
        job=BudgetJob("qwen3_vl_8b", "vt", tmp_path / "relation", "a" * 64, ()),
        method=BudgetMethod("single_point", tmp_path / "config", "b" * 64),
        fraction=0.1,
        subset={
            "retained_conflict_group_count": 51,
            "available_conflict_group_count": 510,
        },
        metrics={"accuracy": 0.8, "macro_f1": 0.7, "auprc": 0.6},
    )

    assert row["accuracy"] == 0.8
    assert row["macro_f1"] == 0.7
    assert row["auprc"] == 0.6
    assert "balanced_accuracy" not in row
