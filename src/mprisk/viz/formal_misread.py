"""Fail-closed readers for publication-bound Misread experiment roots.

Exploratory files are deliberately not discovered.  A consumer sees evidence only
through an explicit root containing a completed marker and hashed artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mprisk.data.misread_labels import verify_imported_labels

COMPLETE_MARKER = "COMPLETE.json"
COMPLETE = "complete"
PARTIAL_REVIEW = "partial_manual_review_required"
ROOT_SCHEMAS = {
    "labels": "mprisk_formal_misread_labels_root_v1",
    "probes": "mprisk_formal_misread_probe_root_v1",
    "budgets": "mprisk_formal_conflict_budget_root_v1",
}
FORMAL_MODELS = ("qwen2_5_omni_7b", "qwen3_vl_8b", "internvl3_5_8b")
FORMAL_METHODS = ("Single-Point", "Trajectory MLP", "TME")
FORMAL_BUDGETS = (10, 25, 50, 100)
LABEL_PROTOCOL = {
    "judge_model": "deepseek-v4-flash",
    "temperature": 0.0,
    "confidence_threshold": 0.5,
    "judges_per_sample": 1,
    "uncertain_policy": "manual_review_required",
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class FormalRoot:
    kind: str
    root: Path
    marker_path: Path
    marker: dict[str, Any]

    def artifacts(self, role: str) -> list[dict[str, Any]]:
        return [artifact for artifact in self.marker["artifacts"] if artifact["role"] == role]

    @property
    def marker_sha256(self) -> str:
        return sha256(self.marker_path)


def load_formal_root(root: str | Path | None, *, kind: str) -> FormalRoot | None:
    """Return only a complete, fully hashed formal root.

    A missing/in-progress root is Pending.  A marker that claims completion but
    violates the contract is an error; it is never silently downgraded.
    """
    if kind not in ROOT_SCHEMAS:
        raise ValueError(f"unknown formal result kind: {kind}")
    if root is None:
        return None
    path = Path(root)
    marker_path = path / COMPLETE_MARKER
    if not marker_path.is_file():
        return None
    if kind == "labels":
        return _load_imported_labels(path, marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    status = marker.get("status")
    accepted = {PARTIAL_REVIEW} if kind == "labels" else {COMPLETE}
    if status not in accepted:
        return None
    if marker.get("schema") != ROOT_SCHEMAS[kind]:
        raise ValueError(f"{kind} marker schema mismatch")
    if marker.get("dataset_id") != "delivery_20260716":
        raise ValueError(f"{kind} marker must bind delivery_20260716")
    if not _is_sha256(marker.get("split_assignment_sha256")):
        raise ValueError(f"{kind} marker requires split_assignment_sha256")
    command = marker.get("generated_command")
    if not isinstance(command, list) or not command or any(not str(item) for item in command):
        raise ValueError(f"{kind} marker requires generated_command argv")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{kind} marker requires artifacts")
    identities: set[tuple[str, str]] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError(f"{kind} artifact must be an object")
        role = artifact.get("role")
        relative = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(role, str) or not role or not isinstance(relative, str) or not relative:
            raise ValueError(f"{kind} artifact identity is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{kind} artifact path must stay within its root")
        if not _is_sha256(digest):
            raise ValueError(f"{kind} artifact hash is invalid")
        identity = (role, str(relative_path))
        if identity in identities:
            raise ValueError(f"{kind} marker contains a duplicate artifact")
        identities.add(identity)
        artifact_path = path / relative_path
        if not artifact_path.is_file() or sha256(artifact_path) != digest:
            raise ValueError(f"{kind} artifact checksum mismatch: {artifact_path}")
    result = FormalRoot(kind=kind, root=path, marker_path=marker_path, marker=marker)
    _validate_root_identity(result)
    return result


def _load_imported_labels(path: Path, marker_path: Path) -> FormalRoot:
    """Adapt the immutable importer contract without weakening its verification."""
    verified = verify_imported_labels(path)
    if verified.get("status") not in {PARTIAL_REVIEW, COMPLETE}:
        raise ValueError("formal labels are neither complete nor review-bounded")
    provenance_path = path / "provenance.json"
    checksums_path = path / "artifact_checksums.json"
    summary_path = path / "summary.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = provenance.get("judge_protocol") or {}
    normalized_protocol = {
        "judge_model": protocol.get("judge_model"),
        "temperature": protocol.get("temperature"),
        "confidence_threshold": protocol.get("confidence_threshold"),
        "judges_per_sample": protocol.get("n_flash"),
        "uncertain_policy": "manual_review_required",
    }
    input_artifacts = provenance.get("input_artifacts") or {}
    split = input_artifacts.get("split_assignment") or {}
    delivery = input_artifacts.get("delivery_manifest") or {}
    if Path(str(delivery.get("path", ""))).name != "unified_sample_manifest.jsonl":
        raise ValueError("formal labels do not bind the authoritative delivery manifest")
    artifacts: list[dict[str, Any]] = []
    for relative, evidence in (checksums.get("artifacts") or {}).items():
        role = "labels" if relative.startswith("labels/") else Path(relative).stem
        artifact: dict[str, Any] = {
            "role": role,
            "path": relative,
            "sha256": evidence.get("sha256"),
        }
        if role == "labels":
            artifact["model"] = Path(relative).stem
        artifacts.append(artifact)
    marker = {
        **verified,
        "dataset_id": "delivery_20260716",
        "split_assignment_sha256": split.get("sha256"),
        "label_protocol": normalized_protocol,
        "artifacts": artifacts,
        "row_count": verified.get("counts", {}).get("rows"),
        "model_row_counts": {
            model: evidence.get("overall", {}).get("rows")
            for model, evidence in (summary.get("models") or {}).items()
        },
    }
    result = FormalRoot(
        kind="labels",
        root=path,
        marker_path=marker_path,
        marker=marker,
    )
    _validate_root_identity(result)
    return result


def _validate_root_identity(root: FormalRoot) -> None:
    marker = root.marker
    if root.kind == "labels":
        if root.marker.get("eligible_subset_complete") is not True:
            raise ValueError("formal labels must complete their eligible subset")
        if marker.get("label_protocol") != LABEL_PROTOCOL:
            raise ValueError("formal labels do not match the locked judge protocol")
        if not set(FORMAL_MODELS).issubset(set(marker.get("models") or ())):
            raise ValueError("formal labels must bind the three representative models")
        representative = [
            artifact
            for artifact in root.artifacts("labels")
            if artifact.get("model") in FORMAL_MODELS
        ]
        if len(representative) != len(FORMAL_MODELS):
            raise ValueError("formal labels require one artifact per representative model")
    elif root.kind == "probes":
        if tuple(marker.get("methods") or ()) != FORMAL_METHODS:
            raise ValueError("formal probes must bind the three representation methods")
        if len(root.artifacts("probe_metrics")) != 1:
            raise ValueError("formal probes require one probe_metrics artifact")
    else:
        if tuple(marker.get("methods") or ()) != FORMAL_METHODS:
            raise ValueError("formal budgets must bind the three representation methods")
        if tuple(marker.get("budget_pct") or ()) != FORMAL_BUDGETS:
            raise ValueError("formal budgets must bind 10/25/50/100 percent")
        if len(root.artifacts("budget_metrics")) != 1:
            raise ValueError("formal budgets require one budget_metrics artifact")


def read_jsonl_artifacts(root: FormalRoot, role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in root.artifacts(role):
        path = root.root / artifact["path"]
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def read_csv_artifact(root: FormalRoot, role: str) -> list[dict[str, str]]:
    artifacts = root.artifacts(role)
    if len(artifacts) != 1:
        raise ValueError(f"{root.kind} requires exactly one {role} artifact")
    with (root.root / artifacts[0]["path"]).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def canonical_label_rows(root: FormalRoot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in root.artifacts("labels"):
        if artifact.get("model") not in FORMAL_MODELS:
            continue
        path = root.root / artifact["path"]
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    canonical: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    artifact_models = {
        str(item.get("model"))
        for item in root.artifacts("labels")
        if item.get("model") in FORMAL_MODELS
    }
    if artifact_models != set(FORMAL_MODELS):
        raise ValueError("formal label artifacts require an explicit unique model identity")
    for row in rows:
        model = str(row.get("subject_model_key") or row.get("model"))
        sample_id = str(row.get("sample_id") or "")
        raw_label = row.get("imported_label")
        label = "" if raw_label is None else str(raw_label)
        protocol = str(row.get("protocol") or "").upper()
        sample_type = str(row.get("sample_type") or "")
        if model not in FORMAL_MODELS or not sample_id:
            raise ValueError("formal label row identity is invalid")
        if protocol not in {"VT", "VA"}:
            raise ValueError("formal label row protocol is invalid")
        if sample_type not in {"Aligned", "Conflict"}:
            raise ValueError("formal label row sample_type is invalid")
        blocked = row.get("blocked") is True
        needs_review = row.get("needs_manual_review") is True
        label_eligible = row.get("label_eligible") is True
        probe_eligible = row.get("probe_eligible") is True
        if label_eligible or probe_eligible:
            if label not in {"MISREAD", "NON_MISREAD"} or blocked or needs_review:
                raise ValueError("eligible formal label row is unresolved or blocked")
        elif label and label not in {"MISREAD", "NON_MISREAD", "UNCERTAIN"}:
            raise ValueError("ineligible formal label row has an invalid label")
        if probe_eligible and not label_eligible:
            raise ValueError("probe eligibility requires label eligibility")
        flashes = row.get("flash")
        confidence = row.get("confidence", row.get("judge_confidence"))
        if confidence is None and isinstance(flashes, list) and len(flashes) == 1:
            confidence = flashes[0].get("confidence")
            if flashes[0].get("judge_model") != LABEL_PROTOCOL["judge_model"]:
                raise ValueError("formal label row judge identity mismatch")
        if confidence is None and not (label_eligible or probe_eligible):
            parsed_confidence: float | None = None
        else:
            parsed_confidence = float(confidence)
            if not 0.0 <= parsed_confidence <= 1.0:
                raise ValueError("formal label confidence is invalid")
        key = (model, sample_id)
        if key in seen:
            raise ValueError("formal labels contain duplicate model/sample rows")
        seen.add(key)
        canonical.append(
            {
                "model": model,
                "sample_id": sample_id,
                "protocol": protocol,
                "sample_type": sample_type,
                "label": label,
                "confidence": parsed_confidence,
                "blocked": blocked,
                "needs_manual_review": needs_review,
                "label_eligible": label_eligible,
                "probe_eligible": probe_eligible,
            }
        )
    model_row_counts = root.marker.get("model_row_counts") or {}
    if any(not isinstance(model_row_counts.get(model), int) for model in FORMAL_MODELS):
        raise ValueError("formal labels require per-model row counts")
    expected = sum(model_row_counts[model] for model in FORMAL_MODELS)
    if len(canonical) != expected:
        raise ValueError("formal representative label row_count is incomplete")
    return canonical


PROBE_FIELDS = (
    "model",
    "protocol",
    "method",
    "seed",
    "accuracy",
    "macro_f1",
    "auprc",
    "latency_ms",
    "n_train",
    "n_val",
    "n_test",
    "test_sample_ids_sha256",
    "label_artifact_sha256",
    "status",
)
BUDGET_FIELDS = (
    "model",
    "protocol",
    "method",
    "budget_pct",
    "seed",
    "accuracy",
    "macro_f1",
    "auprc",
    "n_conflict_supervision",
    "n_aligned_supervision",
    "n_train",
    "n_val",
    "n_test",
    "test_sample_ids_sha256",
    "label_artifact_sha256",
    "encoder_checkpoint_sha256",
    "status",
)


def canonical_metric_rows(
    root: FormalRoot, *, role: str, fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    raw = read_csv_artifact(root, role)
    if not raw:
        raise ValueError(f"formal {root.kind} metrics are empty")
    if tuple(raw[0]) != fields:
        raise ValueError(f"formal {root.kind} CSV header mismatch")
    canonical: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in raw:
        if row["status"] != "Ready" or row["method"] not in FORMAL_METHODS:
            raise ValueError(f"formal {root.kind} row status/method is invalid")
        if row["model"] not in FORMAL_MODELS or row["protocol"] not in {"VT", "VA"}:
            raise ValueError(f"formal {root.kind} row model/protocol is invalid")
        for name in ("test_sample_ids_sha256", "label_artifact_sha256"):
            if not _is_sha256(row[name]):
                raise ValueError(f"formal {root.kind} row requires {name}")
        if "encoder_checkpoint_sha256" in row and not _is_sha256(row["encoder_checkpoint_sha256"]):
            raise ValueError("formal budget row requires encoder_checkpoint_sha256")
        item: dict[str, Any] = dict(row)
        for name in ("accuracy", "macro_f1", "auprc"):
            item[name] = float(row[name])
            if not 0.0 <= item[name] <= 1.0:
                raise ValueError(f"formal {root.kind} {name} is outside [0,1]")
        for name in ("seed", "n_train", "n_val", "n_test"):
            item[name] = int(row[name])
        if role == "probe_metrics":
            item["latency_ms"] = (
                None if row["latency_ms"] == "" else float(row["latency_ms"])
            )
            if item["latency_ms"] is not None and item["latency_ms"] < 0:
                raise ValueError("formal probe latency_ms must be non-negative")
            key = (row["model"], row["method"], row["seed"])
        else:
            for name in ("budget_pct", "n_conflict_supervision", "n_aligned_supervision"):
                item[name] = int(row[name])
            if item["budget_pct"] not in FORMAL_BUDGETS:
                raise ValueError("formal budget row has an unregistered percentage")
            key = (row["model"], row["method"], row["budget_pct"], row["seed"])
        if key in seen:
            raise ValueError(f"formal {root.kind} metrics contain duplicate rows")
        seen.add(key)
        canonical.append(item)
    by_model: dict[str, set[str]] = {}
    for row in canonical:
        by_model.setdefault(row["model"], set()).add(row["test_sample_ids_sha256"])
    if any(len(values) != 1 for values in by_model.values()):
        raise ValueError(f"formal {root.kind} rows must share one test intersection per model")
    return canonical
