"""Held-out Conflict/Aligned metrics and paired three-seed summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from mprisk.representation.relation_models import strict_l2_normalize
from mprisk.utils.io import write_json, write_jsonl

T_CRITICAL_DF2_975 = 4.302652729911275


def evaluate_official_representation(
    *,
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    rows = _read_jsonl(manifest_path)
    _require_official_rows(rows)
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    repr_key = str(checkpoint.get("repr_key", ""))
    checkpoint_sha = _sha256(Path(checkpoint_path))
    predictions: list[dict[str, Any]] = []
    if repr_key == "tme_proxy_anchor_v1":
        proxies = checkpoint.get("proxy_state_dict", {}).get("proxies")
        if not isinstance(proxies, torch.Tensor) or proxies.ndim != 2 or proxies.shape[0] != 2:
            raise ValueError("TME checkpoint requires exactly two Proxy Anchor proxies")
        proxies = proxies.float()
        proxy_norms = torch.linalg.vector_norm(proxies, dim=-1)
        if bool((proxy_norms <= 1e-12).any()):
            raise ValueError("TME Proxy Anchor contains a zero-norm proxy")
        proxies = proxies / proxy_norms.unsqueeze(-1)
        for row in rows:
            feature = torch.tensor(row["sample_relation_feature"], dtype=torch.float32)
            feature = strict_l2_normalize(
                feature.unsqueeze(0),
                stage="official_test_tme_relation",
                sample_ids=[str(row["sample_id"])],
            )[0]
            logits = feature @ proxies.T
            predictions.append(_prediction_row(row, logits, repr_key, checkpoint_sha))
    elif repr_key in {"single_point_binary_v1", "trajectory_mlp_binary_v1"}:
        for row in rows:
            logits = torch.tensor(row["mean_logits"], dtype=torch.float32)
            if logits.shape != (2,):
                raise ValueError("baseline official logits must have shape [2]")
            predictions.append(_prediction_row(row, logits, repr_key, checkpoint_sha))
    else:
        raise ValueError(f"unsupported representation for A/C evaluation: {repr_key}")
    labels = [int(row["label_id"]) for row in predictions]
    predicted = [int(row["prediction_id"]) for row in predictions]
    scores = [float(row["conflict_score"]) for row in predictions]
    metrics = _binary_metrics(labels, predicted, scores)
    output_root = Path(output_dir)
    prediction_path = write_jsonl(output_root / "official_test_predictions.jsonl", predictions)
    curve_rows = _curve_rows(labels, scores)
    curve_path = _write_csv(output_root / "official_test_roc_pr.csv", curve_rows)
    split_key = _one(rows, "split_assignment_key")
    split_sha = _one(rows, "split_assignment_sha256")
    payload = {
        "schema": "mprisk_official_test_ac_metrics_v1",
        "task": "Conflict_vs_Aligned",
        "positive_class": "Conflict",
        "selection_rule": "representation_split=official_test",
        "sample_count": len(rows),
        "sample_type_counts": dict(Counter(str(row["sample_type"]) for row in rows)),
        "model_key": _one(rows, "model_key"),
        "protocol": _one(rows, "protocol"),
        "prompt_set_key": _one(rows, "prompt_set_key"),
        "repr_key": repr_key,
        "split_assignment_key": split_key,
        "split_assignment_sha256": split_sha,
        "checkpoint_sha256": checkpoint_sha,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(Path(manifest_path)),
        "predictions": str(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
        "roc_pr": str(curve_path),
        "roc_pr_sha256": _sha256(curve_path),
        **metrics,
    }
    metrics_path = write_json(output_root / "official_test_metrics.json", payload)
    return {**payload, "metrics_path": str(metrics_path)}


def aggregate_three_seeds(
    *,
    model_key: str,
    runs: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Path]:
    if len(runs) != 3 or {int(run["seed"]) for run in runs} != {
        20260715,
        20260716,
        20260717,
    }:
        raise ValueError("paired aggregation requires exactly the three registered seeds")
    prompt_keys = {str(run["prompt_set_key"]) for run in runs}
    if len(prompt_keys) != 3:
        raise ValueError("each repeated measure must use a distinct registered prompt set")
    rows_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    thresholds: dict[int, dict[str, Any]] = {}
    split_identities: set[tuple[str, str]] = set()
    for run in runs:
        seed = int(run["seed"])
        pattern_path = Path(run["state_patterns"])
        rows = _read_jsonl(pattern_path)
        _require_official_rows(rows, expected_model_key=model_key, require_label=False)
        if _one(rows, "prompt_set_key") != str(run["prompt_set_key"]):
            raise ValueError("run prompt_set_key does not match state rows")
        rows_by_seed[seed] = {str(row["sample_id"]): row for row in rows}
        provenance = json.loads(Path(run["state_provenance"]).read_text(encoding="utf-8"))
        if provenance.get("schema") != "mprisk_official_test_state_provenance_v1":
            raise ValueError("state provenance schema mismatch")
        if provenance.get("official_patterns_sha256") != _sha256(pattern_path):
            raise ValueError("state provenance does not bind the official pattern artifact")
        calibration_path = Path(provenance["calibration_artifact"])
        if provenance.get("calibration_artifact_sha256") != _sha256(calibration_path):
            raise ValueError("state provenance does not bind the calibration artifact")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("calibration_split") != "aligned_calibration":
            raise ValueError("aggregation calibration did not use aligned_calibration")
        if calibration.get("model_key") != model_key:
            raise ValueError("calibration model identity mismatch")
        if calibration.get("prompt_set_key") != str(run["prompt_set_key"]):
            raise ValueError("calibration prompt identity mismatch")
        split_identity = (
            _one(rows, "split_assignment_key"),
            _one(rows, "split_assignment_sha256"),
        )
        if calibration.get("split_assignment_sha256") != split_identity[1]:
            raise ValueError("calibration split identity mismatch")
        split_identities.add(split_identity)
        thresholds[seed] = calibration
    if len(split_identities) != 1:
        raise ValueError("three prompt seeds must use the same registered sample split")
    sample_sets = [set(rows) for rows in rows_by_seed.values()]
    if any(sample_set != sample_sets[0] for sample_set in sample_sets[1:]):
        raise ValueError("three-seed aggregation requires exact paired sample IDs")

    paired_rows: list[dict[str, Any]] = []
    seed_level_rows: list[dict[str, Any]] = []
    seed_summary_rows: list[dict[str, Any]] = []
    for seed, sample_rows in sorted(rows_by_seed.items()):
        kappa = float(thresholds[seed]["kappa"])
        tau = float(thresholds[seed]["tau"])
        for row in sample_rows.values():
            base = {
                "model_key": model_key,
                "seed": seed,
                "sample_id": row["sample_id"],
                "sample_type": row["sample_type"],
                "S": float(row["S_mean"]),
                "D": float(row["D"]),
                "R": float(row["R"]),
                "abs_R": abs(float(row["R"])),
                "pattern": row["pattern"],
                "kappa": kappa,
                "tau": tau,
            }
            seed_level_rows.append(base)
        seed_summary_rows.extend(_seed_state_statistics(model_key, seed, sample_rows.values()))

    for sample_id in sorted(sample_sets[0]):
        rows = [rows_by_seed[seed][sample_id] for seed in sorted(rows_by_seed)]
        if len({row["sample_type"] for row in rows}) != 1:
            raise ValueError("paired sample_type differs across prompt seeds")
        values = {
            "S": [float(row["S_mean"]) for row in rows],
            "D": [float(row["D"]) for row in rows],
            "R": [float(row["R"]) for row in rows],
            "abs_R": [abs(float(row["R"])) for row in rows],
        }
        result: dict[str, Any] = {
            "model_key": model_key,
            "sample_id": sample_id,
            "sample_type": rows[0]["sample_type"],
            "seed_count": 3,
            "patterns": "|".join(str(row["pattern"]) for row in rows),
            "pattern_agreement_count": max(Counter(row["pattern"] for row in rows).values()),
            "stable_all_seeds": all(
                float(row["S_mean"]) <= float(thresholds[seed]["kappa"])
                for seed, row in zip(sorted(rows_by_seed), rows, strict=True)
            ),
            "direction_interpretable_all_seeds": all(
                float(row["D"]) > float(thresholds[seed]["tau"])
                for seed, row in zip(sorted(rows_by_seed), rows, strict=True)
            ),
        }
        for metric, metric_values in values.items():
            mean, sd, low, high = _mean_sd_ci(metric_values)
            result.update(
                {
                    f"{metric}_seed_mean": mean,
                    f"{metric}_sample_sd": sd,
                    f"{metric}_ci95_low": low,
                    f"{metric}_ci95_high": high,
                }
            )
        paired_rows.append(result)

    model_summary_rows = _aggregate_seed_statistics(seed_summary_rows)
    classification_rows: list[dict[str, Any]] = []
    for run in runs:
        for repr_key, metrics_path in sorted(run["classification_metrics"].items()):
            payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
            if payload.get("task") != "Conflict_vs_Aligned":
                raise ValueError("Misread or generic correctness metrics cannot enter A/C tables")
            classification_rows.append(
                {
                    "model_key": model_key,
                    "seed": int(run["seed"]),
                    "repr_key": repr_key,
                    "accuracy": payload["accuracy"],
                    "macro_f1": payload["macro_f1"],
                    "auprc": payload["auprc"],
                }
            )
    model_summary_rows.extend(_aggregate_classification_statistics(classification_rows))
    figure_sdr = [
        {
            key: row[key]
            for key in (
                "model_key",
                "sample_id",
                "sample_type",
                "S_seed_mean",
                "S_sample_sd",
                "S_ci95_low",
                "S_ci95_high",
                "D_seed_mean",
                "D_sample_sd",
                "D_ci95_low",
                "D_ci95_high",
                "abs_R_seed_mean",
                "abs_R_sample_sd",
                "abs_R_ci95_low",
                "abs_R_ci95_high",
            )
        }
        for row in paired_rows
    ]
    figure_patterns = [
        row
        for row in model_summary_rows
        if row.get("sample_type") in {"Aligned", "Conflict"}
        and str(row.get("metric", "")).startswith("pattern_")
    ]
    figure_geometry = [
        {
            "model_key": row["model_key"],
            "sample_id": row["sample_id"],
            "sample_type": row["sample_type"],
            "D_seed_mean": row["D_seed_mean"],
            "D_sample_sd": row["D_sample_sd"],
            "R_seed_mean": row["R_seed_mean"],
            "R_sample_sd": row["R_sample_sd"],
            "direction_interpretable_all_seeds": row["direction_interpretable_all_seeds"],
        }
        for row in paired_rows
        if row["stable_all_seeds"]
    ]

    output_root = Path(output_dir)
    paths = {
        "paired_samples": _write_csv(output_root / "paired_sample_statistics.csv", paired_rows),
        "seed_statistics": _write_csv(output_root / "seed_statistics.csv", seed_summary_rows),
        "model_statistics": _write_csv(
            output_root / "model_repeated_measure_statistics.csv", model_summary_rows
        ),
        "seed_level_state_rows": _write_csv(
            output_root / "seed_level_state_rows.csv", seed_level_rows
        ),
        "fig04": _write_csv(output_root / "paper_inputs/fig04_sdr.csv", figure_sdr),
        "fig05": _write_csv(output_root / "paper_inputs/fig05_state_patterns.csv", figure_patterns),
        "fig06": _write_csv(
            output_root / "paper_inputs/fig06_stable_geometry.csv", figure_geometry
        ),
        "table_ac": _write_csv(
            output_root / "paper_inputs/table_ac_classification.csv", classification_rows
        ),
    }
    provenance = {
        "schema": "mprisk_three_seed_paired_aggregation_v1",
        "model_key": model_key,
        "seeds": sorted(rows_by_seed),
        "pairing_unit": "model_key,sample_id",
        "seed_count": 3,
        "sample_count": len(sample_sets[0]),
        "standard_deviation": "sample_sd_ddof_1_across_three_prompt_seeds",
        "confidence_interval": "two_sided_t_95_percent_df_2",
        "t_critical": T_CRITICAL_DF2_975,
        "selection_rule": "official_test only",
        "calibration_rule": "independent aligned_calibration per seed",
        "artifacts": {key: str(path) for key, path in paths.items()},
        "artifact_sha256": {key: _sha256(path) for key, path in paths.items()},
    }
    paths["provenance"] = write_json(output_root / "aggregation_provenance.json", provenance)
    return paths


def _prediction_row(
    row: dict[str, Any], logits: torch.Tensor, repr_key: str, checkpoint_sha: str
) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1)
    prediction_id = int(probabilities.argmax())
    return {
        "schema": "mprisk_official_test_ac_prediction_v1",
        "sample_id": row["sample_id"],
        "sample_type": row["sample_type"],
        "label_id": int(row["label_id"]),
        "model_key": row["model_key"],
        "protocol": row["protocol"],
        "prompt_set_key": row["prompt_set_key"],
        "representation_split": row["representation_split"],
        "split_group_id": row["split_group_id"],
        "split_assignment_key": row["split_assignment_key"],
        "split_assignment_sha256": row["split_assignment_sha256"],
        "repr_key": repr_key,
        "checkpoint_sha256": checkpoint_sha,
        "conflict_score": float(probabilities[1]),
        "prediction_id": prediction_id,
        "prediction_label": "Conflict" if prediction_id else "Aligned",
    }


def _binary_metrics(
    labels: list[int], predictions: list[int], scores: list[float]
) -> dict[str, float]:
    if set(labels) != {0, 1}:
        raise ValueError("official A/C evaluation requires both classes")
    accuracy = sum(a == b for a, b in zip(labels, predictions, strict=True)) / len(labels)
    f1_values = []
    for label in (0, 1):
        tp = sum(a == label and b == label for a, b in zip(labels, predictions, strict=True))
        fp = sum(a != label and b == label for a, b in zip(labels, predictions, strict=True))
        fn = sum(a == label and b != label for a, b in zip(labels, predictions, strict=True))
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    grouped_scores: dict[float, list[int]] = defaultdict(list)
    for score, label in zip(scores, labels, strict=True):
        grouped_scores[float(score)].append(label)
    positives = sum(labels)
    true_positives = 0
    predicted_positives = 0
    average_precision = 0.0
    for score in sorted(grouped_scores, reverse=True):
        group = grouped_scores[score]
        previous_true_positives = true_positives
        true_positives += sum(group)
        predicted_positives += len(group)
        recall_increment = (true_positives - previous_true_positives) / positives
        average_precision += recall_increment * (true_positives / predicted_positives)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(sum(f1_values) / 2),
        "auprc": float(average_precision),
    }


def _curve_rows(labels: list[int], scores: list[float]) -> list[dict[str, float | str]]:
    thresholds = [math.inf] + sorted(set(scores), reverse=True) + [-math.inf]
    positives = sum(labels)
    negatives = len(labels) - positives
    rows: list[dict[str, float | str]] = []
    for threshold in thresholds:
        predicted = [score >= threshold for score in scores]
        tp = sum(label == 1 and value for label, value in zip(labels, predicted, strict=True))
        fp = sum(label == 0 and value for label, value in zip(labels, predicted, strict=True))
        rows.append(
            {
                "threshold": str(threshold),
                "tpr_recall": tp / positives,
                "fpr": fp / negatives,
                "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
            }
        )
    return rows


def _require_official_rows(
    rows: list[dict[str, Any]],
    *,
    expected_model_key: str | None = None,
    require_label: bool = True,
) -> None:
    if not rows or any(row.get("representation_split") != "official_test" for row in rows):
        raise ValueError("paper evaluation accepts official_test rows only")
    if expected_model_key and any(row.get("model_key") != expected_model_key for row in rows):
        raise ValueError("paper evaluation model identity mismatch")
    for row in rows:
        expected_label = int(row.get("sample_type") == "Conflict")
        if row.get("sample_type") not in {"Aligned", "Conflict"}:
            raise ValueError("paper evaluation requires Conflict/Aligned sample_type")
        if require_label and row.get("label_id") != expected_label:
            raise ValueError("official label_id must be derived from sample_type")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("paper evaluation requires unique non-empty sample IDs")


def _seed_state_statistics(
    model_key: str, seed: int, rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    materialized = list(rows)
    results: list[dict[str, Any]] = []
    for sample_type in ("All", "Aligned", "Conflict"):
        selected = (
            materialized
            if sample_type == "All"
            else [row for row in materialized if row["sample_type"] == sample_type]
        )
        for metric, values in (
            ("mean_S", [float(row["S_mean"]) for row in selected]),
            ("mean_D", [float(row["D"]) for row in selected]),
            ("mean_abs_R", [abs(float(row["R"])) for row in selected]),
        ):
            results.append(
                {
                    "model_key": model_key,
                    "seed": seed,
                    "sample_type": sample_type,
                    "metric": metric,
                    "value": sum(values) / len(values),
                    "sample_count": len(values),
                }
            )
        patterns = Counter(str(row["pattern"]) for row in selected)
        for pattern in ("Confusion", "Consensus", "Balanced", "Dominant"):
            results.append(
                {
                    "model_key": model_key,
                    "seed": seed,
                    "sample_type": sample_type,
                    "metric": f"pattern_{pattern}_proportion",
                    "value": patterns[pattern] / len(selected),
                    "sample_count": len(selected),
                }
            )
    return results


def _aggregate_seed_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sample_type"]), str(row["metric"]))].append(float(row["value"]))
    results = []
    for (sample_type, metric), values in sorted(grouped.items()):
        mean, sd, low, high = _mean_sd_ci(values)
        results.append(
            {
                "task": "State",
                "sample_type": sample_type,
                "metric": metric,
                "seed_mean": mean,
                "sample_sd": sd,
                "ci95_low": low,
                "ci95_high": high,
                "seed_count": 3,
            }
        )
    return results


def _aggregate_classification_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in ("accuracy", "macro_f1", "auprc"):
            grouped[(str(row["repr_key"]), metric)].append(float(row[metric]))
    results = []
    for (repr_key, metric), values in sorted(grouped.items()):
        mean, sd, low, high = _mean_sd_ci(values)
        results.append(
            {
                "task": "Conflict_vs_Aligned",
                "sample_type": "official_test",
                "repr_key": repr_key,
                "metric": metric,
                "seed_mean": mean,
                "sample_sd": sd,
                "ci95_low": low,
                "ci95_high": high,
                "seed_count": 3,
            }
        )
    return results


def _mean_sd_ci(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 3:
        raise ValueError("registered repeated-measure CI requires exactly three values")
    mean = sum(values) / 3
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / 2)
    half_width = T_CRITICAL_DF2_975 * sd / math.sqrt(3)
    return mean, sd, mean - half_width, mean + half_width


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row is not an object: {path}")
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if not rows:
        raise ValueError(f"refusing to write an empty data artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def _one(rows: list[dict[str, Any]], field: str) -> str:
    values = {str(row.get(field, "")) for row in rows}
    if len(values) != 1 or not next(iter(values)):
        raise ValueError(f"official rows require homogeneous {field}")
    return next(iter(values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
