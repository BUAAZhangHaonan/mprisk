from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

import mprisk.experiments.delivery_representation as delivery_representation
from mprisk.cache.cache_union import UNION_SCHEMA
from mprisk.experiments.delivery_representation import (
    BASELINE_COMPLETION_SCHEMA,
    BASELINE_METHODS,
    DSTRONG_METHOD,
    DTHETA_METHOD,
    METHODS,
    PARTIALLY_RUNNABLE,
    PENDING_CACHE_UNION,
    RUNNABLE,
    SINGLE_POINT_METHOD,
    TRAJECTORY_MLP_METHOD,
    DeliveryJob,
    DeliveryPlanError,
    _load_job_identity_config,
    _normalize_method_selection,
    _selected_model_keys_for_load,
    _validate_cache_union,
    _validate_completion_marker,
    _write_geometry_metrics,
    bind_delivery_plan,
    load_delivery_plan,
)
from mprisk.representation.training import load_training_config


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


def test_production_template_training_config_hashes_match_files() -> None:
    root = Path(__file__).parents[2]
    template = yaml.safe_load(
        (
            root
            / "configs/downstream/delivery_20260716_seed20260717_tme_template_v1.yaml"
        ).read_text(encoding="utf-8")
    )

    for job in template["jobs"]:
        for method, spec in job["training_configs"].items():
            config_path = root / spec["path"]
            assert config_path.is_file(), (job["model_key"], method, config_path)
            assert hashlib.sha256(config_path.read_bytes()).hexdigest() == spec["sha256"], (
                job["model_key"],
                method,
            )


def test_production_baseline_template_is_native_ce_and_statically_valid() -> None:
    root = Path(__file__).parents[2]
    template_path = (
        root
        / "configs/downstream/delivery_20260716_seed20260717_baselines_template_v1.yaml"
    )
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))

    delivery_representation._validate_static_plan(
        template,
        template_path,
        expect_template=True,
    )
    assert template["resource_gate"] == {
        "device": "cuda:1",
        "max_gpu_memory_fraction": 0.75,
    }
    assert template["output_root"].endswith("/representation_baselines_v1")
    for job in template["jobs"]:
        assert tuple(job["training_configs"]) == BASELINE_METHODS
        for method, spec in job["training_configs"].items():
            config_path = root / spec["path"]
            config = load_training_config(config_path)
            assert hashlib.sha256(config_path.read_bytes()).hexdigest() == spec["sha256"]
            assert config.repr_key == method
            assert config.classification_objective == "inverse_frequency_cross_entropy"
            assert config.enable_state_supervision is False


def test_baseline_method_selection_cannot_cross_into_tme_group() -> None:
    assert _normalize_method_selection(
        {TRAJECTORY_MLP_METHOD, SINGLE_POINT_METHOD},
        available_methods=BASELINE_METHODS,
    ) == BASELINE_METHODS
    assert _normalize_method_selection(
        {SINGLE_POINT_METHOD},
        available_methods=BASELINE_METHODS,
    ) == (SINGLE_POINT_METHOD,)
    with pytest.raises(DeliveryPlanError, match="unknown method"):
        _normalize_method_selection(
            {DSTRONG_METHOD},
            available_methods=BASELINE_METHODS,
        )


def test_baseline_job_uses_registered_native_config_for_relation_identity() -> None:
    root = Path(__file__).parents[2]
    config = (
        root
        / "configs/experiments/seed_runs/seed20260717/"
        "qwen3_vl_8b_single_point_binary_v1.yaml"
    )
    job = DeliveryJob(
        model_key="qwen3_vl_8b",
        protocol="vt",
        seed=20260717,
        run_id="baseline",
        output_dir=root / "unused",
        state_manifest=root / "unused.jsonl",
        cache_union=root / "unused-union.json",
        cache_union_sha256="c" * 64,
        training_configs={
            SINGLE_POINT_METHOD: config,
            TRAJECTORY_MLP_METHOD: config.with_name(
                "qwen3_vl_8b_trajectory_mlp_binary_v1.yaml"
            ),
        },
    )

    identity = _load_job_identity_config(job)
    assert identity.repr_key == SINGLE_POINT_METHOD
    assert identity.model_key == job.model_key
    assert identity.protocol == job.protocol


def test_global_dstrong_v2_changes_only_d_supervision_weight() -> None:
    root = Path(__file__).parents[2]
    template = yaml.safe_load(
        (
            root
            / "configs/downstream/delivery_20260716_seed20260717_tme_template_v1.yaml"
        ).read_text(encoding="utf-8")
    )

    for job in template["jobs"]:
        configs = job["training_configs"]
        assert set(configs) == set(METHODS)
        base_path = root / configs[DTHETA_METHOD]["path"]
        strong_path = root / configs[DSTRONG_METHOD]["path"]
        base = load_training_config(base_path)
        strong = load_training_config(strong_path)
        assert strong.d_supervision_weight == pytest.approx(0.5)
        assert replace(strong, d_supervision_weight=base.d_supervision_weight) == base
        assert strong_path.name.endswith("_tme_pa_dstrong_v2.yaml")
        base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        strong_raw = yaml.safe_load(strong_path.read_text(encoding="utf-8"))
        assert base_raw["key"] != strong_raw["key"]


def test_dstrong_method_can_be_selected_without_running_infeasible_pilot() -> None:
    assert _normalize_method_selection({DSTRONG_METHOD}) == (DSTRONG_METHOD,)
    assert _normalize_method_selection(None) == METHODS
    with pytest.raises(DeliveryPlanError, match="unknown method"):
        _normalize_method_selection({"unknown"})


def test_completion_marker_requires_bound_method_config_and_cache_identity(
    tmp_path: Path,
) -> None:
    config = tmp_path / "dstrong.yaml"
    config.write_text("d_supervision_weight: 0.5\n", encoding="utf-8")
    union = tmp_path / "union.json"
    union.write_text("{}\n", encoding="utf-8")
    artifacts = {}
    for field in (
        "best_checkpoint",
        "official_frozen",
        "official_sdr_scores",
        "official_patterns",
        "geometry_metrics",
    ):
        path = tmp_path / f"{field}.json"
        path.write_text(f"{field}\n", encoding="utf-8")
        artifacts[field] = path
    job = DeliveryJob(
        model_key="qwen2_5_omni_7b",
        protocol="va",
        seed=20260717,
        run_id="run",
        output_dir=tmp_path / "output",
        state_manifest=tmp_path / "state.jsonl",
        cache_union=union,
        cache_union_sha256="c" * 64,
        training_configs={DSTRONG_METHOD: config},
    )
    completion = {
        "schema": "mprisk_delivery_tme_run_complete_v1",
        "model_key": job.model_key,
        "method": DSTRONG_METHOD,
        "training_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "cache_union_sha256": job.cache_union_sha256,
    }
    artifact_sha_fields = {
        "best_checkpoint": "best_checkpoint_sha256",
        "official_frozen": "official_frozen_sha256",
        "official_sdr_scores": "official_sdr_sha256",
        "official_patterns": "official_patterns_sha256",
        "geometry_metrics": "geometry_metrics_sha256",
    }
    for field, path in artifacts.items():
        completion[field] = str(path)
        completion[artifact_sha_fields[field]] = hashlib.sha256(path.read_bytes()).hexdigest()

    assert _validate_completion_marker(
        completion=completion,
        job=job,
        method=DSTRONG_METHOD,
        marker_path=tmp_path / "RUN_COMPLETE.json",
    ) == artifacts["official_sdr_scores"]
    completion["training_config_sha256"] = "0" * 64
    with pytest.raises(DeliveryPlanError, match="identity drift"):
        _validate_completion_marker(
            completion=completion,
            job=job,
            method=DSTRONG_METHOD,
            marker_path=tmp_path / "RUN_COMPLETE.json",
        )


def test_baseline_completion_marker_binds_union_and_forbids_proxy_state(
    tmp_path: Path,
) -> None:
    config = tmp_path / "baseline.yaml"
    config.write_text("repr_key: single_point_binary_v1\n", encoding="utf-8")
    union = tmp_path / "union.json"
    union.write_text("{}\n", encoding="utf-8")
    artifact_fields = (
        ("best_checkpoint", "best_checkpoint_sha256"),
        ("training_metrics", "training_metrics_sha256"),
        ("relation_dataset", "relation_dataset_sha256"),
        ("split_assignment", "split_assignment_sha256"),
        ("official_manifest", "official_manifest_sha256"),
        ("official_frozen_summary", "official_frozen_summary_sha256"),
        ("official_ac_metrics", "official_ac_metrics_sha256"),
    )
    completion = {
        "schema": BASELINE_COMPLETION_SCHEMA,
        "model_key": "qwen3_vl_8b",
        "method": SINGLE_POINT_METHOD,
        "repr_key": SINGLE_POINT_METHOD,
        "training_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "cache_union_sha256": "c" * 64,
        "classification_objective": "inverse_frequency_cross_entropy",
        "proxy_anchor_used": False,
        "state_indices_used": False,
        "misread_labels_used": False,
    }
    artifacts = {}
    for path_field, sha_field in artifact_fields:
        artifact = tmp_path / f"{path_field}.json"
        artifact.write_text(path_field + "\n", encoding="utf-8")
        completion[path_field] = str(artifact)
        completion[sha_field] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        artifacts[path_field] = artifact
    job = DeliveryJob(
        model_key="qwen3_vl_8b",
        protocol="vt",
        seed=20260717,
        run_id="run",
        output_dir=tmp_path / "output",
        state_manifest=tmp_path / "state.jsonl",
        cache_union=union,
        cache_union_sha256="c" * 64,
        training_configs={SINGLE_POINT_METHOD: config},
    )

    assert _validate_completion_marker(
        completion=completion,
        job=job,
        method=SINGLE_POINT_METHOD,
        marker_path=tmp_path / "RUN_COMPLETE.json",
    ) == artifacts["official_manifest"]
    completion["proxy_anchor_used"] = True
    with pytest.raises(DeliveryPlanError, match="violates method contract"):
        _validate_completion_marker(
            completion=completion,
            job=job,
            method=SINGLE_POINT_METHOD,
            marker_path=tmp_path / "RUN_COMPLETE.json",
        )


def test_partial_plan_requires_exact_explicit_model_selection() -> None:
    selected = {"qwen3_vl_8b", "qwen2_5_omni_7b"}
    payload = {
        "status": PARTIALLY_RUNNABLE,
        "selected_model_keys": sorted(selected),
    }

    with pytest.raises(DeliveryPlanError, match="requires explicit --model-key"):
        _selected_model_keys_for_load(payload, requested=None)
    with pytest.raises(DeliveryPlanError, match="exactly match"):
        _selected_model_keys_for_load(payload, requested={"qwen3_vl_8b"})
    assert _selected_model_keys_for_load(payload, requested=selected) == selected


def test_full_plan_allows_implicit_all_but_not_selective_skipping() -> None:
    all_models = {"qwen3_vl_8b", "internvl3_5_8b", "qwen2_5_omni_7b"}
    payload = {"status": RUNNABLE, "selected_model_keys": sorted(all_models)}

    assert _selected_model_keys_for_load(payload, requested=None) == all_models
    with pytest.raises(DeliveryPlanError, match="exactly match"):
        _selected_model_keys_for_load(payload, requested={"qwen2_5_omni_7b"})


def test_selective_binder_keeps_unselected_jobs_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "template.yaml"
    output = tmp_path / "partial.yaml"
    jobs = [
        {"model_key": model_key, "cache_union": PENDING_CACHE_UNION}
        for model_key in ("qwen3_vl_8b", "internvl3_5_8b", "qwen2_5_omni_7b")
    ]
    template.write_text(yaml.safe_dump({"jobs": jobs}), encoding="utf-8")
    unions = {}
    for model_key in ("qwen3_vl_8b", "qwen2_5_omni_7b"):
        union = tmp_path / f"{model_key}.json"
        union.write_text("{}\n", encoding="utf-8")
        unions[model_key] = union
    monkeypatch.setattr(delivery_representation, "_validate_static_plan", lambda *a, **k: None)
    monkeypatch.setattr(delivery_representation, "_validate_cache_union", lambda *a, **k: None)
    monkeypatch.setattr(delivery_representation, "_repo_root", lambda _path: tmp_path)
    monkeypatch.setattr(
        delivery_representation,
        "load_delivery_plan",
        lambda path, model_keys: yaml.safe_load(Path(path).read_text(encoding="utf-8")),
    )

    bound = bind_delivery_plan(
        template,
        cache_unions=unions,
        output_path=output,
        model_keys=set(unions),
    )

    assert bound["status"] == PARTIALLY_RUNNABLE
    assert bound["selected_model_keys"] == sorted(unions)
    by_model = {job["model_key"]: job for job in bound["jobs"]}
    assert by_model["internvl3_5_8b"]["cache_union"] == PENDING_CACHE_UNION
    assert isinstance(by_model["qwen3_vl_8b"]["cache_union"], dict)
    assert isinstance(by_model["qwen2_5_omni_7b"]["cache_union"], dict)


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
        "schema": UNION_SCHEMA,
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
