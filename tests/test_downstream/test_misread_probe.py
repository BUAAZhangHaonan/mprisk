from __future__ import annotations

import json
from pathlib import Path

from mprisk.evaluation.misread_probe import write_pending_conflict_misread_probe


def test_pending_probe_never_generates_labels_or_starts_training(tmp_path: Path) -> None:
    path = write_pending_conflict_misread_probe(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "Pending Misread annotations"
    assert payload["eligible_sample_type"] == "Conflict"
    assert payload["representation_policy"] == "frozen_no_encoder_gradients"
    assert payload["generated_labels"] == 0
    assert payload["pseudo_labels"] == 0
    assert payload["training_started"] is False
