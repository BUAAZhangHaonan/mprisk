from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from mprisk.evaluation.downstream_metrics import (
    aggregate_three_seeds,
    evaluate_official_representation,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _official_row(sample_id: str, label_id: int, logits: list[float]) -> dict:
    return {
        "sample_id": sample_id,
        "sample_type": "Conflict" if label_id else "Aligned",
        "label_id": label_id,
        "model_key": "model",
        "protocol": "vt",
        "prompt_set_key": "p8",
        "representation_split": "official_test",
        "split_group_id": f"g-{sample_id}",
        "split_assignment_key": "split-v1",
        "split_assignment_sha256": "a" * 64,
        "mean_logits": logits,
    }


def test_official_ac_metrics_are_not_misread_metrics(tmp_path: Path) -> None:
    manifest = _write_jsonl(
        tmp_path / "features.jsonl",
        [_official_row("a", 0, [4.0, -4.0]), _official_row("c", 1, [-4.0, 4.0])],
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"repr_key": "single_point_binary_v1"}, checkpoint)
    payload = evaluate_official_representation(
        manifest_path=manifest, checkpoint_path=checkpoint, output_dir=tmp_path / "eval"
    )
    assert payload["task"] == "Conflict_vs_Aligned"
    assert payload["accuracy"] == 1.0
    assert payload["macro_f1"] == 1.0
    assert payload["auprc"] == 1.0


def test_tme_official_evaluation_rejects_zero_norm_relation(tmp_path: Path) -> None:
    row = {
        **_official_row("c", 1, [0.0, 1.0]),
        "sample_relation_feature": [0.0, 0.0],
    }
    manifest = _write_jsonl(tmp_path / "tme.jsonl", [row])
    checkpoint = tmp_path / "tme.pt"
    torch.save(
        {
            "repr_key": "tme_proxy_anchor_v1",
            "proxy_state_dict": {"proxies": torch.eye(2)},
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="norm must exceed"):
        evaluate_official_representation(
            manifest_path=manifest,
            checkpoint_path=checkpoint,
            output_dir=tmp_path / "tme-eval",
        )


def test_three_seed_aggregation_pairs_samples_and_uses_t_df2(tmp_path: Path) -> None:
    runs = []
    for offset, seed in enumerate((20260715, 20260716, 20260717)):
        run = tmp_path / str(seed)
        patterns = _write_jsonl(
            run / "patterns.jsonl",
            [
                {
                    **_official_row("a", 0, [1.0, 0.0]),
                    "S_mean": 0.1 + offset * 0.01,
                    "D": 0.2,
                    "R": -0.1,
                    "pattern": "Consensus",
                },
                {
                    **_official_row("c", 1, [0.0, 1.0]),
                    "S_mean": 0.4 + offset * 0.01,
                    "D": 0.8,
                    "R": 0.6,
                    "pattern": "Dominant",
                },
            ],
        )
        pattern_rows = [json.loads(line) for line in patterns.read_text().splitlines()]
        for row in pattern_rows:
            row["prompt_set_key"] = f"p8-{seed}"
        _write_jsonl(patterns, pattern_rows)
        calibration = run / "calibration.json"
        calibration.write_text(
            json.dumps(
                {
                    "calibration_split": "aligned_calibration",
                    "kappa": 0.5,
                    "tau": 0.3,
                    "model_key": "model",
                    "prompt_set_key": f"p8-{seed}",
                    "split_assignment_sha256": "a" * 64,
                }
            ),
            encoding="utf-8",
        )
        provenance = run / "provenance.json"
        provenance.write_text(
            json.dumps(
                {
                    "schema": "mprisk_official_test_state_provenance_v1",
                    "calibration_artifact": str(calibration),
                    "calibration_artifact_sha256": hashlib.sha256(
                        calibration.read_bytes()
                    ).hexdigest(),
                    "official_patterns_sha256": hashlib.sha256(patterns.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        metrics = {}
        for repr_key in (
            "single_point_binary_v1",
            "trajectory_mlp_binary_v1",
            "tme_proxy_anchor_v1",
        ):
            path = run / f"{repr_key}.json"
            path.write_text(
                json.dumps(
                    {
                        "task": "Conflict_vs_Aligned",
                        "accuracy": 0.7 + offset * 0.01,
                        "macro_f1": 0.6,
                        "auprc": 0.8,
                    }
                ),
                encoding="utf-8",
            )
            metrics[repr_key] = str(path)
        runs.append(
            {
                "seed": seed,
                "prompt_set_key": f"p8-{seed}",
                "state_patterns": str(patterns),
                "state_provenance": str(provenance),
                "classification_metrics": metrics,
            }
        )
    paths = aggregate_three_seeds(model_key="model", runs=runs, output_dir=tmp_path / "out")
    with paths["paired_samples"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["sample_id"] for row in rows} == {"a", "c"}
    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["confidence_interval"] == "two_sided_t_95_percent_df_2"
    assert provenance["pairing_unit"] == "model_key,sample_id"
    with paths["fig04"].open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert len(list(csv.DictReader(paths["fig05"].open(encoding="utf-8")))) == 8

    first_patterns = Path(runs[0]["state_patterns"])
    first_patterns.write_text(first_patterns.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bind the official pattern"):
        aggregate_three_seeds(model_key="model", runs=runs, output_dir=tmp_path / "tampered")
