from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from mprisk.experiments.delivery_representation import (
    DeliveryPlanError,
    _validate_cache_union,
    _write_geometry_metrics,
    load_delivery_plan,
)


def _unit(angle: float) -> list[float]:
    return [math.cos(angle), math.sin(angle)]


def test_pending_production_template_is_not_runnable() -> None:
    root = Path(__file__).parents[2]
    template = (
        root
        / "configs/downstream/delivery_20260716_seed20260717_tme_template_v1.yaml"
    )
    with pytest.raises(DeliveryPlanError, match="pending"):
        load_delivery_plan(template)


def test_geometry_metrics_report_d_angle_relation_clustering_and_proxy_angle(
    tmp_path: Path,
) -> None:
    score_rows = []
    frozen_rows = []
    relation_features = {
        "a0": [1.0, 0.0],
        "a1": [0.999, 0.045],
        "a2": [0.998, -0.063],
        "a3": [0.997, 0.077],
        "c0": [-1.0, 0.0],
        "c1": [-0.999, 0.045],
        "c2": [-0.998, -0.063],
        "c3": [-0.997, 0.077],
    }
    for index, (sample_id, feature) in enumerate(relation_features.items()):
        sample_type = "Aligned" if sample_id.startswith("a") else "Conflict"
        split = 0.08 if sample_type == "Aligned" else 0.7
        m1 = _unit(index * 0.01)
        m2 = _unit(index * 0.01 + split)
        score_rows.append({"sample_id": sample_id, "sample_type": sample_type, "D": split})
        frozen_rows.append(
            {
                "sample_id": sample_id,
                "sample_type": sample_type,
                "sample_relation_feature": feature,
                "embeddings": {
                    "M1": {"p1": m1, "p2": m1},
                    "M2": {"p1": m2, "p2": m2},
                    "M12": {"p1": m1, "p2": m1},
                },
            }
        )
    scores = tmp_path / "scores.jsonl"
    frozen = tmp_path / "frozen.jsonl"
    scores.write_text("".join(json.dumps(row) + "\n" for row in score_rows), encoding="utf-8")
    frozen.write_text("".join(json.dumps(row) + "\n" for row in frozen_rows), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"proxy_state_dict": {"proxies": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}},
        checkpoint,
    )

    output = _write_geometry_metrics(
        scores_path=scores,
        frozen_path=frozen,
        checkpoint_path=checkpoint,
        output_path=tmp_path / "metrics.json",
    )
    metrics = json.loads(output.read_text(encoding="utf-8"))

    assert metrics["metrics"]["D"]["conflict_minus_aligned"] > 0.5
    assert metrics["metrics"]["split_angle_deg"]["conflict_minus_aligned"] > 20.0
    assert metrics["relation_r_clustering"]["cosine_silhouette"] > 0.9
    assert metrics["relation_r_clustering"]["five_nn_label_purity"] >= 0.6
    assert metrics["relation_r_clustering"]["proxy_angular_separation_deg"] == pytest.approx(90.0)


def test_cache_union_gate_requires_exact_full_prefill_grid(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    plan_path = tmp_path / "configs/downstream/template.yaml"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "state.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "sample_type": sample_type,
                    "protocol": "VT",
                }
            )
            + "\n"
            for sample_id, sample_type in (("a", "Aligned"), ("c", "Conflict"))
        ),
        encoding="utf-8",
    )
    prompt_ids = [f"p{index}" for index in range(8)]
    entries = [
        {
            "sample_id": sample_id,
            "model_key": "qwen3_vl_8b",
            "protocol": "vt",
            "condition": condition,
            "prompt_set_key": "prompts",
            "prompt_id": prompt_id,
        }
        for sample_id in ("a", "c")
        for prompt_id in prompt_ids
        for condition in ("M1", "M2", "M12")
    ]
    job = {
        "model_key": "qwen3_vl_8b",
        "protocol": "vt",
        "state_manifest": {"path": str(manifest), "sha256": "unused"},
        "invalid_assets": None,
        "prompt_set": {
            "key": "prompts",
            "sha256": "p" * 64,
            "prompt_ids": prompt_ids,
        },
        "expected_counts": {
            "resolved_tasks": 48,
            "blocked_tasks": 0,
            "raw_tasks": 48,
        },
    }
    union_payload = {
        "schema": "mprisk_prefill_cache_union_v1",
        "entries": entries,
        "blocked_tasks": [],
        "provenance": {
            "prefill_strategy": "full_prefill",
            "prefill_strategy_version": "v1",
            "counts": {"resolved_tasks": 48, "blocked_tasks": 0, "raw_tasks": 48},
            "expected_signature": {
                "model_key": "qwen3_vl_8b",
                "protocol": "vt",
                "prompt_set_sha256": "p" * 64,
                "prompt_ids": prompt_ids,
                "conditions": ["M1", "M2", "M12"],
            },
        },
    }
    union = tmp_path / "union.json"
    union.write_text(json.dumps(union_payload), encoding="utf-8")

    _validate_cache_union(union, job, plan_path)
    union_payload["entries"].pop()
    union_payload["provenance"]["counts"]["resolved_tasks"] = 47
    union.write_text(json.dumps(union_payload), encoding="utf-8")
    with pytest.raises(DeliveryPlanError, match="task counts"):
        _validate_cache_union(union, job, plan_path)
