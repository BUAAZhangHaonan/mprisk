"""Strict import of externally generated Diagnostic Misread labels.

The importer treats the source experiment tree as immutable evidence.  It
joins every model result to the authoritative delivery manifest and
representation split assignment, excludes unresolved judge/asset cases from
probe use, and publishes a checksummed directory exactly once.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_V2_ROOT = Path("/home/team/zhanghaonan/TAFFC/mprisk-v2")
DEFAULT_DELIVERY_MANIFEST = Path(
    "outputs/datasets/delivery_20260716/manifests/unified_sample_manifest.jsonl"
)
DEFAULT_SPLIT_ASSIGNMENT = Path(
    "outputs/datasets/delivery_20260716/splits/representation_split_assignment_v1.jsonl"
)
DEFAULT_INVALID_ASSETS = Path("outputs/datasets/delivery_20260716/manifests/invalid_assets.jsonl")
DEFAULT_OUTPUT_ROOT = Path("outputs/labels/delivery_20260716_single_flash_v1")
CONFIDENCE_THRESHOLD = 0.5

DELIVERY_MANIFEST_SHA256 = "353dcfa73bf327bb2cde2316689bd1470a0c1202866c050dcb85268a07c916c3"
SPLIT_ASSIGNMENT_SHA256 = "e4008c99e70c22e54120573799ce87998445a28de6e82639023581728446df41"
INVALID_ASSETS_SHA256 = "98bb527f553bec4f490320f61b40495d011b8ec095051f69fc5af37ed2c5e8de"


@dataclass(frozen=True)
class ModelImportSpec:
    model_key: str
    protocol: str
    expected_descriptions: int
    expected_judgments: int
    descriptions_sha256: str
    judgments_sha256: str
    representative_probe_model: bool


DEFAULT_MODEL_SPECS = (
    ModelImportSpec(
        "qwen3_vl_8b",
        "VT",
        1876,
        1876,
        "6ed3300f23a62ddd2ba1d4ab16a180ddbd07a577f16527ff82a92310e6136d20",
        "cbe8cd878d135ffe9bc04cf477855be6bc26407114bcb0a5e9319c06f0673738",
        True,
    ),
    ModelImportSpec(
        "internvl3_5_8b",
        "VT",
        1876,
        1876,
        "3c22ab5fdcc8d954ce39782773e0f09cd35731ef400ee267005ec9c1958f839e",
        "c9427ed6f0b28cc082a02819ce23f94445edf598ce4b1b50fcd033bbbfddca0b",
        True,
    ),
    ModelImportSpec(
        "qwen2_5_omni_7b",
        "VA",
        1939,
        1934,
        "e52ddd7967fd049f19e99eafcf129a77d2f12893aacb35f8fdbfea3a864ec61f",
        "412493aab36e65b6a3a4507d6af7d1dc32cdc31c19747a681499ad3b7aa01c77",
        True,
    ),
    ModelImportSpec(
        "gemma4_12b_it",
        "VA",
        1939,
        1934,
        "24a0febb69bd60afae011d255c10ceda32661942190643808ff03a119091e91f",
        "f9d06598e96c7cb693229b47d49e0503688b6d3316cb27bd6704b14c9a114268",
        False,
    ),
)


@dataclass(frozen=True)
class MisreadImportResult:
    output_root: Path
    marker_path: Path
    total_rows: int
    probe_eligible_rows: int
    manual_review_rows: int
    blocked_rows: int


def import_single_flash_labels(
    *,
    v2_root: str | Path = DEFAULT_V2_ROOT,
    delivery_manifest: str | Path = DEFAULT_DELIVERY_MANIFEST,
    split_assignment: str | Path = DEFAULT_SPLIT_ASSIGNMENT,
    invalid_assets: str | Path = DEFAULT_INVALID_ASSETS,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_specs: Sequence[ModelImportSpec] = DEFAULT_MODEL_SPECS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    expected_delivery_sha256: str | None = DELIVERY_MANIFEST_SHA256,
    expected_split_sha256: str | None = SPLIT_ASSIGNMENT_SHA256,
    expected_invalid_assets_sha256: str | None = INVALID_ASSETS_SHA256,
) -> MisreadImportResult:
    """Import pinned single-Flash labels into a new immutable output directory."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    source_root = Path(v2_root).expanduser().resolve()
    manifest_path = Path(delivery_manifest).expanduser().resolve()
    split_path = Path(split_assignment).expanduser().resolve()
    invalid_path = Path(invalid_assets).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace immutable label import: {output}")
    for path in (manifest_path, split_path, invalid_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output == source_root or source_root in output.parents:
        raise ValueError("Output root must not be inside the read-only V2 source tree")

    input_paths: dict[str, Path] = {
        "delivery_manifest": manifest_path,
        "split_assignment": split_path,
        "invalid_assets": invalid_path,
    }
    for spec in model_specs:
        model_dir = source_root / "outputs/v2/misread" / spec.model_key
        input_paths[f"{spec.model_key}.descriptions"] = model_dir / "descriptions.jsonl"
        input_paths[f"{spec.model_key}.judgments"] = model_dir / "judgments_single_flash.jsonl"
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing import evidence: {missing}")

    source_hashes_before = {key: _sha256(path) for key, path in input_paths.items()}
    _require_sha(
        "delivery manifest", source_hashes_before["delivery_manifest"], expected_delivery_sha256
    )
    _require_sha(
        "split assignment", source_hashes_before["split_assignment"], expected_split_sha256
    )
    _require_sha(
        "invalid assets", source_hashes_before["invalid_assets"], expected_invalid_assets_sha256
    )
    for spec in model_specs:
        _require_sha(
            f"{spec.model_key} descriptions",
            source_hashes_before[f"{spec.model_key}.descriptions"],
            spec.descriptions_sha256,
        )
        _require_sha(
            f"{spec.model_key} judgments",
            source_hashes_before[f"{spec.model_key}.judgments"],
            spec.judgments_sha256,
        )

    manifest_rows = _read_unique_jsonl(manifest_path, "sample_id")
    manifest_by_id = {str(row["sample_id"]): row for row in manifest_rows}
    split_by_sample = _load_splits(split_path, manifest_by_id)
    invalid_rows = _read_unique_jsonl(invalid_path, "sample_id")
    invalid_by_id = {str(row["sample_id"]): row for row in invalid_rows}
    for sample_id, row in invalid_by_id.items():
        manifest = manifest_by_id.get(sample_id)
        if manifest is None:
            raise ValueError(f"Invalid-asset sample is absent from delivery manifest: {sample_id}")
        if manifest.get("protocol") != "VA" or row.get("reason") != "missing_audio_stream":
            raise ValueError(f"Unexpected invalid-asset contract for {sample_id}")

    temp_parent = output.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=temp_parent))
    summaries: dict[str, Any] = {}
    output_files: list[Path] = []
    all_output_rows: list[dict[str, Any]] = []
    try:
        labels_dir = staging / "labels"
        labels_dir.mkdir()
        for spec in model_specs:
            descriptions_path = input_paths[f"{spec.model_key}.descriptions"]
            judgments_path = input_paths[f"{spec.model_key}.judgments"]
            descriptions = _read_unique_jsonl(descriptions_path, "sample_id")
            judgments = _read_unique_jsonl(judgments_path, "sample_id")
            _require_count(spec, "descriptions", len(descriptions), spec.expected_descriptions)
            _require_count(spec, "judgments", len(judgments), spec.expected_judgments)
            descriptions_by_id = {str(row["sample_id"]): row for row in descriptions}
            judgments_by_id = {str(row["sample_id"]): row for row in judgments}
            expected_ids = {
                sample_id
                for sample_id, row in manifest_by_id.items()
                if row.get("protocol") == spec.protocol
            }
            if set(descriptions_by_id) != expected_ids:
                _raise_set_mismatch(
                    spec.model_key, "descriptions", expected_ids, set(descriptions_by_id)
                )
            blocked_ids = expected_ids & set(invalid_by_id)
            expected_judgment_ids = expected_ids - blocked_ids
            if set(judgments_by_id) != expected_judgment_ids:
                _raise_set_mismatch(
                    spec.model_key,
                    "judgments",
                    expected_judgment_ids,
                    set(judgments_by_id),
                )

            imported_rows = []
            for sample_id in sorted(expected_ids):
                manifest = manifest_by_id[sample_id]
                description = descriptions_by_id[sample_id]
                _validate_description(description, manifest, spec)
                blocked = sample_id in blocked_ids
                if blocked:
                    if description.get("diagnostic_description") or not description.get("error"):
                        raise ValueError(
                            f"Blocked sample must carry an explicit generation error: {sample_id}"
                        )
                    row = _blocked_row(
                        manifest=manifest,
                        description=description,
                        invalid=invalid_by_id[sample_id],
                        spec=spec,
                    )
                else:
                    split = split_by_sample.get(sample_id)
                    if split is None:
                        raise ValueError(
                            f"Non-blocked sample has no representation split: {sample_id}"
                        )
                    judgment = judgments_by_id[sample_id]
                    row = _labeled_row(
                        manifest=manifest,
                        split=split,
                        description=description,
                        judgment=judgment,
                        spec=spec,
                        confidence_threshold=confidence_threshold,
                    )
                imported_rows.append(row)
            label_path = labels_dir / f"{spec.model_key}.jsonl"
            _write_jsonl(label_path, imported_rows)
            output_files.append(label_path)
            all_output_rows.extend(imported_rows)
            summaries[spec.model_key] = _summarize_model(imported_rows, spec)

        summary = {
            "schema": "mprisk_misread_label_import_summary_v1",
            "confidence_threshold": confidence_threshold,
            "models": summaries,
            "totals": _summarize_totals(all_output_rows),
        }
        summary_path = staging / "summary.json"
        _write_json(summary_path, summary)
        output_files.append(summary_path)

        source_hashes_after = {key: _sha256(path) for key, path in input_paths.items()}
        if source_hashes_after != source_hashes_before:
            raise RuntimeError("Read-only source evidence changed during import")
        provenance = {
            "schema": "mprisk_misread_label_import_provenance_v1",
            "source_read_only": True,
            "judge_protocol": {
                "judge_model": "deepseek-v4-flash",
                "n_flash": 1,
                "pro_arbitration": False,
                "temperature": 0.0,
                "confidence_threshold": confidence_threshold,
                "manual_review_rule": "raw decision UNCERTAIN or confidence < threshold",
            },
            "input_artifacts": {
                key: {"path": str(path), "sha256": source_hashes_before[key]}
                for key, path in sorted(input_paths.items())
            },
            "output_policy": {
                "blocked_assets_are_not_labels": True,
                "manual_review_rows_have_null_imported_label": True,
                "probe_requires_representative_model_and_finalized_label": True,
            },
        }
        provenance_path = staging / "provenance.json"
        _write_json(provenance_path, provenance)
        output_files.append(provenance_path)

        checksums = {
            str(path.relative_to(staging)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(output_files)
        }
        checksums_path = staging / "artifact_checksums.json"
        _write_json(
            checksums_path,
            {"schema": "mprisk_artifact_checksums_v1", "artifacts": checksums},
        )
        has_unresolved = bool(
            summary["totals"]["needs_manual_review"] or summary["totals"]["blocked"]
        )
        marker = {
            "schema": "mprisk_formal_misread_labels_root_v1",
            "status": "partial_manual_review_required" if has_unresolved else "complete",
            "eligible_subset_complete": True,
            "resolved_count": summary["totals"]["label_eligible"],
            "unresolved_count": (
                summary["totals"]["needs_manual_review"] + summary["totals"]["blocked"]
            ),
            "artifact_checksums_sha256": _sha256(checksums_path),
            "counts": summary["totals"],
            "models": [spec.model_key for spec in model_specs],
        }
        marker_path = staging / "COMPLETE.json"
        _write_json(marker_path, marker)
        marker_sha = _sha256(marker_path)
        (staging / "COMPLETE.json.sha256").write_text(
            f"{marker_sha}  COMPLETE.json\n", encoding="utf-8"
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    verify_imported_labels(output)
    totals = summary["totals"]
    return MisreadImportResult(
        output_root=output,
        marker_path=output / "COMPLETE.json",
        total_rows=int(totals["rows"]),
        probe_eligible_rows=int(totals["probe_eligible"]),
        manual_review_rows=int(totals["needs_manual_review"]),
        blocked_rows=int(totals["blocked"]),
    )


def verify_imported_labels(output_root: str | Path) -> dict[str, Any]:
    """Verify marker and every materialized artifact without rewriting anything."""
    root = Path(output_root).expanduser().resolve()
    marker_path = root / "COMPLETE.json"
    marker_sha_path = root / "COMPLETE.json.sha256"
    checksums_path = root / "artifact_checksums.json"
    for path in (marker_path, marker_sha_path, checksums_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    sidecar_parts = marker_sha_path.read_text(encoding="utf-8").strip().split()
    if len(sidecar_parts) != 2 or sidecar_parts[1] != "COMPLETE.json":
        raise ValueError("Malformed COMPLETE.json.sha256")
    if _sha256(marker_path) != sidecar_parts[0]:
        raise ValueError("COMPLETE marker SHA-256 mismatch")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != "mprisk_formal_misread_labels_root_v1":
        raise ValueError("Unexpected formal label root schema")
    if marker.get("status") not in {"complete", "partial_manual_review_required"}:
        raise ValueError("Label import marker has an invalid status")
    if marker.get("eligible_subset_complete") is not True:
        raise ValueError("Eligible label subset is not marked complete")
    if marker.get("artifact_checksums_sha256") != _sha256(checksums_path):
        raise ValueError("Artifact checksum manifest SHA-256 mismatch")
    checksum_doc = json.loads(checksums_path.read_text(encoding="utf-8"))
    for relative, evidence in checksum_doc.get("artifacts", {}).items():
        path = root / relative
        if not path.is_file() or _sha256(path) != evidence.get("sha256"):
            raise ValueError(f"Imported label artifact mismatch: {relative}")
        if path.stat().st_size != evidence.get("bytes"):
            raise ValueError(f"Imported label artifact size mismatch: {relative}")
    return marker


def _read_unique_jsonl(path: Path, key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line in {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Missing string {key} in {path}:{line_number}")
            value = str(row[key])
            if value in seen:
                raise ValueError(f"Duplicate {key} {value!r} in {path}")
            seen.add(value)
            rows.append(row)
    return rows


def _load_splits(
    path: Path, manifest_by_id: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    assignments = _read_unique_jsonl(path, "split_group_id")
    by_sample: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        sample_ids = assignment.get("sample_ids")
        if not isinstance(sample_ids, list) or not sample_ids:
            raise ValueError(f"Invalid split sample_ids for {assignment['split_group_id']}")
        for sample_id in sample_ids:
            if not isinstance(sample_id, str) or sample_id in by_sample:
                raise ValueError(f"Duplicate/invalid split sample ID: {sample_id!r}")
            manifest = manifest_by_id.get(sample_id)
            if manifest is None:
                raise ValueError(f"Split sample is absent from delivery manifest: {sample_id}")
            if assignment["split_group_id"] != manifest.get("split_group_id"):
                raise ValueError(f"Split group mismatch for {sample_id}")
            if assignment.get("master_split") != manifest.get("split"):
                raise ValueError(f"Master split mismatch for {sample_id}")
            by_sample[sample_id] = assignment
    return by_sample


def _validate_description(
    row: dict[str, Any], manifest: dict[str, Any], spec: ModelImportSpec
) -> None:
    sample_id = str(manifest["sample_id"])
    required = {
        "schema": "mprisk_v2_diagnostic_description_v1",
        "subject_model_key": spec.model_key,
        "protocol": spec.protocol,
        "sample_type": manifest["sample_type"],
        "condition": "M12",
        "source_id": manifest["source_id"],
    }
    for key, expected in required.items():
        if row.get(key) != expected:
            raise ValueError(
                f"Description {key} mismatch for {sample_id}: {row.get(key)!r} != {expected!r}"
            )
    if row.get("gt_describe") != manifest.get("gt_describe"):
        raise ValueError(f"GT description mismatch for {sample_id}")


def _labeled_row(
    *,
    manifest: dict[str, Any],
    split: dict[str, Any],
    description: dict[str, Any],
    judgment: dict[str, Any],
    spec: ModelImportSpec,
    confidence_threshold: float,
) -> dict[str, Any]:
    sample_id = str(manifest["sample_id"])
    required = {
        "schema": "mprisk_v2_misread_label_v1",
        "sample_id": sample_id,
        "subject_model_key": spec.model_key,
        "protocol": spec.protocol,
        "arbitrator_used": False,
        "agreement_ratio": 1.0,
        "pro_arbitration": None,
    }
    for key, expected in required.items():
        if judgment.get(key) != expected:
            raise ValueError(
                f"Judgment {key} mismatch for {sample_id}: {judgment.get(key)!r} != {expected!r}"
            )
    flashes = judgment.get("flash")
    if not isinstance(flashes, list) or len(flashes) != 1 or not isinstance(flashes[0], dict):
        raise ValueError(f"Single-Flash judgment must have exactly one flash result: {sample_id}")
    flash = flashes[0]
    if flash.get("judge_model") != "deepseek-v4-flash":
        raise ValueError(f"Unexpected judge model for {sample_id}")
    decision = flash.get("decision")
    if decision not in {"MISREAD", "NON_MISREAD", "UNCERTAIN"}:
        raise ValueError(f"Invalid raw judge decision for {sample_id}: {decision!r}")
    confidence = flash.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise ValueError(f"Invalid judge confidence for {sample_id}")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Judge confidence outside [0,1] for {sample_id}")
    source_final = judgment.get("final_label")
    if source_final not in {"MISREAD", "NON_MISREAD"}:
        raise ValueError(f"Invalid source final label for {sample_id}")
    if decision in {"MISREAD", "NON_MISREAD"} and source_final != decision:
        raise ValueError(f"Source final label disagrees with raw Flash decision: {sample_id}")
    reasons = []
    if decision == "UNCERTAIN":
        reasons.append("raw_judge_uncertain")
    if confidence < confidence_threshold:
        reasons.append("judge_confidence_below_threshold")
    needs_manual = bool(reasons)
    imported_label = None if needs_manual else source_final
    diagnostic = str(description.get("diagnostic_description") or "")
    if not diagnostic or description.get("error"):
        raise ValueError(f"Non-blocked sample lacks a diagnostic description: {sample_id}")
    return {
        "schema": "mprisk_imported_misread_label_v1",
        "sample_id": sample_id,
        "source_id": manifest["source_id"],
        "subject_model_key": spec.model_key,
        "protocol": spec.protocol,
        "sample_type": manifest["sample_type"],
        "split_group_id": manifest["split_group_id"],
        "master_split": manifest["split"],
        "representation_split": split["representation_split"],
        "representative_probe_model": spec.representative_probe_model,
        "judge_model": "deepseek-v4-flash",
        "judge_confidence": confidence,
        "raw_judge_decision": decision,
        "source_final_label": source_final,
        "imported_label": imported_label,
        "needs_manual_review": needs_manual,
        "manual_review_reasons": reasons,
        "blocked": False,
        "blocked_reason": None,
        "label_eligible": imported_label is not None,
        "probe_eligible": spec.representative_probe_model and imported_label is not None,
        "diagnostic_description_sha256": _sha256_text(diagnostic),
        "gt_description_sha256": _sha256_text(str(manifest["gt_describe"])),
    }


def _blocked_row(
    *,
    manifest: dict[str, Any],
    description: dict[str, Any],
    invalid: dict[str, Any],
    spec: ModelImportSpec,
) -> dict[str, Any]:
    return {
        "schema": "mprisk_imported_misread_label_v1",
        "sample_id": manifest["sample_id"],
        "source_id": manifest["source_id"],
        "subject_model_key": spec.model_key,
        "protocol": spec.protocol,
        "sample_type": manifest["sample_type"],
        "split_group_id": manifest["split_group_id"],
        "master_split": manifest["split"],
        "representation_split": None,
        "representative_probe_model": spec.representative_probe_model,
        "judge_model": None,
        "judge_confidence": None,
        "raw_judge_decision": None,
        "source_final_label": None,
        "imported_label": None,
        "needs_manual_review": False,
        "manual_review_reasons": [],
        "blocked": True,
        "blocked_reason": invalid["reason"],
        "generation_error": description["error"],
        "label_eligible": False,
        "probe_eligible": False,
        "diagnostic_description_sha256": None,
        "gt_description_sha256": _sha256_text(str(manifest["gt_describe"])),
    }


def _summarize_model(rows: list[dict[str, Any]], spec: ModelImportSpec) -> dict[str, Any]:
    by_type = {
        sample_type: _summarize_rows([row for row in rows if row["sample_type"] == sample_type])
        for sample_type in ("Aligned", "Conflict")
    }
    return {
        "protocol": spec.protocol,
        "representative_probe_model": spec.representative_probe_model,
        "overall": _summarize_rows(rows),
        "by_sample_type": by_type,
    }


def _summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    labels = Counter(row["imported_label"] for row in rows if row["imported_label"] is not None)
    eligible = labels["MISREAD"] + labels["NON_MISREAD"]
    return {
        "rows": len(rows),
        "misread": labels["MISREAD"],
        "non_misread": labels["NON_MISREAD"],
        "label_eligible": eligible,
        "misread_rate": labels["MISREAD"] / eligible if eligible else None,
        "needs_manual_review": sum(bool(row["needs_manual_review"]) for row in rows),
        "blocked": sum(bool(row["blocked"]) for row in rows),
        "probe_eligible": sum(bool(row["probe_eligible"]) for row in rows),
    }


def _summarize_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize_rows(rows)
    summary["models"] = len({row["subject_model_key"] for row in rows})
    return summary


def _require_count(spec: ModelImportSpec, name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"{spec.model_key} {name} count {actual} != expected {expected}")


def _raise_set_mismatch(model: str, name: str, expected: set[str], actual: set[str]) -> None:
    missing = sorted(expected - actual)[:10]
    extra = sorted(actual - expected)[:10]
    raise ValueError(f"{model} {name} sample set mismatch; missing={missing}, extra={extra}")


def _require_sha(name: str, actual: str, expected: str | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
