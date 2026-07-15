"""Conflict-only Misread probe contract; no labels means machine-readable Pending."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mprisk.utils.io import write_json


def write_pending_conflict_misread_probe(output_dir: str | Path) -> Path:
    payload: dict[str, Any] = {
        "schema": "mprisk_conflict_only_misread_probe_v1",
        "status": "Pending Misread annotations",
        "labels_available": False,
        "eligible_sample_type": "Conflict",
        "excluded_sample_type": "Aligned",
        "representation_policy": "frozen_no_encoder_gradients",
        "split_policy": "group_disjoint_within_conflict_only",
        "probe_architecture": {
            "shared_across_representations": True,
            "layers": ["Linear(input_dim,128)", "GELU", "Dropout(0.1)", "Linear(128,2)"],
            "target": "Misread_vs_Non-misread",
        },
        "required_future_fields": [
            "sample_id",
            "split_group_id",
            "sample_type=Conflict",
            "misread_label",
        ],
        "metrics_when_available": ["Accuracy", "Macro-F1", "AUPRC", "Latency"],
        "generated_labels": 0,
        "pseudo_labels": 0,
        "training_started": False,
    }
    return write_json(Path(output_dir) / "PENDING.json", payload)
