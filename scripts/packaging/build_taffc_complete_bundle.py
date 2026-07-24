#!/usr/bin/env python3
"""Build or audit the canonical 16-model, two-domain TAFFC delivery.

The delivery contract is data driven.  Every model/domain cell explicitly
declares the cache, ground-truth description, diagnostic description, and
Misread label artifact together with its checksum and provenance evidence.
Pending cells are valid for readiness reporting, but a bundle can never be
published while any cell is pending or invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

MATRIX_SCHEMA = "taffc_complete_bundle_matrix_v3"
READINESS_SCHEMA = "taffc_complete_bundle_readiness_v1"
INVENTORY_SCHEMA = "taffc_complete_bundle_inventory_v3"
VALIDATION_SCHEMA = "taffc_complete_bundle_validation_v2"
ARTIFACT_TYPES = ("cache", "ground_truth", "diagnostic_description", "misread_labels")
DOMAINS = ("source", "target")
PROTOCOLS = ("VT", "VA")
CONDITIONS = ("M1", "M2", "M12")
PROMPT_COUNT = 8
EXPECTED_SAMPLES = {
    ("source", "VT"): 1876,
    ("source", "VA"): 1934,
    ("target", "VT"): 2035,
    ("target", "VA"): 2190,
}
MODEL_PROTOCOLS = {
    "gemma3_4b": "VT",
    "gemma3_12b": "VT",
    "glm4_6v_flash": "VT",
    "internvl3_5_8b": "VT",
    "llava_v1_5_7b": "VT",
    "llava_onevision_qwen2_7b": "VT",
    "minicpm_v_2_6": "VT",
    "minicpm_v_4_5": "VT",
    "phi3_5_vision": "VT",
    "qwen2_5_vl_7b": "VT",
    "qwen3_vl_8b": "VT",
    "qwen3_5_4b": "VT",
    "qwen3_5_9b": "VT",
    "gemma4_12b": "VA",
    "phi4_multimodal": "VA",
    "qwen2_5_omni_7b": "VA",
}
SHA256_HEX = frozenset("0123456789abcdef")


class BundleContractError(RuntimeError):
    """Raised when the declared delivery contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class ArtifactSpec:
    status: str
    path: Path | None
    sha256: str | None
    provenance_path: Path | None
    provenance_sha256: str | None
    reason: str | None


@dataclass(frozen=True)
class JobSpec:
    domain: str
    model_key: str
    protocol: str
    expected_samples: int
    artifacts: dict[str, ArtifactSpec]

    @property
    def job_id(self) -> str:
        return f"{self.domain}:{self.model_key}"


@dataclass(frozen=True)
class MatrixContract:
    config_path: Path
    artifact_root: Path
    dataset_manifests: dict[tuple[str, str], ArtifactSpec]
    jobs: tuple[JobSpec, ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    require(
        actual == expected,
        (
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        ),
    )


def _nonempty(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty text")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    text = _nonempty(value, label).lower()
    require(len(text) == 64 and set(text) <= SHA256_HEX, f"{label} must be a SHA-256")
    return text


def _resolve(root: Path, value: Any, label: str) -> Path:
    text = _nonempty(value, label)
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_artifact(value: Any, *, root: Path, label: str) -> ArtifactSpec:
    require(isinstance(value, dict), f"{label} must be an object")
    status = value.get("status")
    require(status in {"ready", "pending"}, f"{label}.status must be ready or pending")
    if status == "pending":
        _exact_keys(value, {"status", "reason"}, label)
        return ArtifactSpec(
            "pending", None, None, None, None, _nonempty(value["reason"], f"{label}.reason")
        )
    _exact_keys(
        value,
        {"status", "path", "sha256", "provenance_path", "provenance_sha256"},
        label,
    )
    return ArtifactSpec(
        "ready",
        _resolve(root, value["path"], f"{label}.path"),
        _sha(value["sha256"], f"{label}.sha256"),
        _resolve(root, value["provenance_path"], f"{label}.provenance_path"),
        _sha(value["provenance_sha256"], f"{label}.provenance_sha256"),
        None,
    )


def load_contract(path: str | Path) -> MatrixContract:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "matrix config must be an object")
    _exact_keys(
        payload,
        {"schema", "artifact_root", "prompt_count", "conditions", "datasets", "models", "jobs"},
        "matrix",
    )
    require(payload["schema"] == MATRIX_SCHEMA, f"matrix schema must be {MATRIX_SCHEMA}")
    require(payload["prompt_count"] == PROMPT_COUNT, "prompt_count must be 8")
    require(tuple(payload["conditions"]) == CONDITIONS, "conditions must be M1,M2,M12")
    root = _resolve(config_path.parent, payload["artifact_root"], "artifact_root")

    datasets_raw = payload["datasets"]
    require(
        isinstance(datasets_raw, list) and len(datasets_raw) == 4,
        "datasets must contain four domain/protocol rows",
    )
    dataset_manifests: dict[tuple[str, str], ArtifactSpec] = {}
    for index, row in enumerate(datasets_raw):
        require(isinstance(row, dict), f"datasets[{index}] must be an object")
        _exact_keys(
            row, {"domain", "protocol", "expected_samples", "manifest"}, f"datasets[{index}]"
        )
        domain = _nonempty(row["domain"], "dataset.domain").lower()
        protocol = _nonempty(row["protocol"], "dataset.protocol").upper()
        key = (domain, protocol)
        require(key in EXPECTED_SAMPLES, f"unsupported dataset cell {key}")
        require(key not in dataset_manifests, f"duplicate dataset cell {key}")
        require(row["expected_samples"] == EXPECTED_SAMPLES[key], f"dataset {key} count drift")
        manifest = _load_artifact(row["manifest"], root=root, label=f"dataset {key} manifest")
        require(manifest.status == "ready", f"dataset {key} manifest cannot be pending")
        dataset_manifests[key] = manifest
    require(set(dataset_manifests) == set(EXPECTED_SAMPLES), "dataset matrix is incomplete")

    models = payload["models"]
    require(isinstance(models, list), "models must be a list")
    observed_models: dict[str, str] = {}
    for index, row in enumerate(models):
        require(isinstance(row, dict), f"models[{index}] must be an object")
        _exact_keys(row, {"model_key", "protocol"}, f"models[{index}]")
        model_key = _nonempty(row["model_key"], "model_key")
        protocol = _nonempty(row["protocol"], "model.protocol").upper()
        require(model_key not in observed_models, f"duplicate model {model_key}")
        observed_models[model_key] = protocol
    require(observed_models == MODEL_PROTOCOLS, "models must equal the canonical 16-model panel")

    jobs_raw = payload["jobs"]
    require(
        isinstance(jobs_raw, list) and len(jobs_raw) == 32,
        "jobs must contain 16 models x 2 domains",
    )
    jobs: list[JobSpec] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(jobs_raw):
        require(isinstance(row, dict), f"jobs[{index}] must be an object")
        _exact_keys(
            row,
            {"domain", "model_key", "protocol", "expected_samples", "artifacts"},
            f"jobs[{index}]",
        )
        domain = _nonempty(row["domain"], "job.domain").lower()
        model_key = _nonempty(row["model_key"], "job.model_key")
        protocol = _nonempty(row["protocol"], "job.protocol").upper()
        key = (domain, model_key)
        require(domain in DOMAINS, f"unsupported domain {domain}")
        require(model_key in MODEL_PROTOCOLS, f"unsupported model {model_key}")
        require(protocol == MODEL_PROTOCOLS[model_key], f"{key} protocol drift")
        require(key not in seen, f"duplicate job {key}")
        seen.add(key)
        expected = EXPECTED_SAMPLES[(domain, protocol)]
        require(row["expected_samples"] == expected, f"{key} expected_samples must be {expected}")
        artifacts_raw = row["artifacts"]
        require(isinstance(artifacts_raw, dict), f"{key}.artifacts must be an object")
        _exact_keys(artifacts_raw, set(ARTIFACT_TYPES), f"{key}.artifacts")
        artifacts = {
            kind: _load_artifact(artifacts_raw[kind], root=root, label=f"{key}.{kind}")
            for kind in ARTIFACT_TYPES
        }
        jobs.append(JobSpec(domain, model_key, protocol, expected, artifacts))
    expected_jobs = {(domain, model) for domain in DOMAINS for model in MODEL_PROTOCOLS}
    require(seen == expected_jobs, "job matrix is incomplete")
    phi4_source = next(
        job for job in jobs if job.domain == "source" and job.model_key == "phi4_multimodal"
    )
    require(
        "misread_labels" in phi4_source.artifacts, "Phi-4 source Misread label cell is required"
    )
    return MatrixContract(config_path, root, dataset_manifests, tuple(jobs))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BundleContractError(f"{path}:{line_number}: invalid JSON") from exc
            require(isinstance(row, dict), f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def _verify_file(spec: ArtifactSpec, label: str) -> list[str]:
    errors: list[str] = []
    assert spec.status == "ready"
    assert spec.path is not None and spec.sha256 is not None
    assert spec.provenance_path is not None and spec.provenance_sha256 is not None
    for path, expected, name in (
        (spec.path, spec.sha256, label),
        (spec.provenance_path, spec.provenance_sha256, f"{label} provenance"),
    ):
        if not path.is_file():
            errors.append(f"{name} missing: {path}")
        elif sha256_file(path) != expected:
            errors.append(f"{name} SHA mismatch: {path}")
    return errors


def _ids(rows: Sequence[dict[str, Any]], label: str) -> set[str]:
    values: list[str] = []
    for row in rows:
        sample_id = row.get("sample_id")
        require(isinstance(sample_id, str) and sample_id, f"{label} row has no sample_id")
        values.append(sample_id)
    require(len(values) == len(set(values)), f"{label} has duplicate sample_id")
    return set(values)


def _validate_dataset(
    spec: ArtifactSpec, *, domain: str, protocol: str, expected_samples: int
) -> tuple[set[str], list[str]]:
    errors = _verify_file(spec, f"{domain}/{protocol} dataset")
    if errors:
        return set(), errors
    assert spec.path is not None
    try:
        rows = read_jsonl(spec.path)
        require(len(rows) == expected_samples, f"{domain}/{protocol} dataset count drift")
        sample_ids = _ids(rows, f"{domain}/{protocol} dataset")
        require(
            {str(row.get("protocol", "")).upper() for row in rows} == {protocol},
            f"{domain}/{protocol} dataset protocol drift",
        )
        require(
            {row.get("sample_type") for row in rows} <= {"Conflict", "Aligned"},
            f"{domain}/{protocol} dataset sample_type drift",
        )
        return sample_ids, []
    except (BundleContractError, OSError, ValueError) as exc:
        return set(), [str(exc)]


def _validate_cache(spec: ArtifactSpec, *, job: JobSpec, sample_ids: set[str]) -> list[str]:
    errors = _verify_file(spec, f"{job.job_id} cache")
    if errors:
        return errors
    assert spec.path is not None
    try:
        payload = json.loads(spec.path.read_text(encoding="utf-8"))
        require(isinstance(payload, dict), f"{job.job_id} cache index must be an object")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            manifest_path = payload.get("manifest_path")
            require(
                isinstance(manifest_path, str),
                f"{job.job_id} cache has no entries or manifest_path",
            )
            manifest = Path(manifest_path)
            if not manifest.is_absolute():
                manifest = (spec.path.parent / manifest).resolve()
            entries = read_jsonl(manifest)
        expected_tasks = job.expected_samples * PROMPT_COUNT * len(CONDITIONS)
        require(len(entries) == expected_tasks, f"{job.job_id} cache task count drift")
        require(
            _ids(entries, f"{job.job_id} cache entries") == sample_ids,
            f"{job.job_id} cache sample set drift",
        )
        task_keys: set[tuple[str, str, str]] = set()
        prompts: set[str] = set()
        for row in entries:
            condition = str(row.get("condition"))
            prompt_id = str(row.get("prompt_id"))
            require(condition in CONDITIONS, f"{job.job_id} cache condition drift")
            require(prompt_id and prompt_id != "None", f"{job.job_id} cache prompt_id missing")
            prompts.add(prompt_id)
            key = (str(row["sample_id"]), prompt_id, condition)
            require(key not in task_keys, f"{job.job_id} duplicate cache task {key}")
            task_keys.add(key)
            checksum = row.get("checksum")
            require(
                isinstance(checksum, str) and len(checksum) == 64,
                f"{job.job_id} cache checksum missing",
            )
        require(len(prompts) == PROMPT_COUNT, f"{job.job_id} cache must contain eight prompts")
        return []
    except (BundleContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        return [str(exc)]


def _validate_ground_truth(spec: ArtifactSpec, *, job: JobSpec, sample_ids: set[str]) -> list[str]:
    errors = _verify_file(spec, f"{job.job_id} ground truth")
    if errors:
        return errors
    assert spec.path is not None
    try:
        rows = read_jsonl(spec.path)
        require(len(rows) == job.expected_samples, f"{job.job_id} GT count drift")
        require(_ids(rows, f"{job.job_id} GT") == sample_ids, f"{job.job_id} GT sample set drift")
        for row in rows:
            value = row.get("GT_DESCRIPTION")
            require(
                isinstance(value, str) and value.strip(), f"{job.job_id} GT_DESCRIPTION missing"
            )
        return []
    except (BundleContractError, OSError, ValueError) as exc:
        return [str(exc)]


def _validate_diagnostic(spec: ArtifactSpec, *, job: JobSpec, sample_ids: set[str]) -> list[str]:
    errors = _verify_file(spec, f"{job.job_id} diagnostic description")
    if errors:
        return errors
    assert spec.path is not None
    try:
        rows = read_jsonl(spec.path)
        require(len(rows) == job.expected_samples, f"{job.job_id} diagnostic count drift")
        require(
            _ids(rows, f"{job.job_id} diagnostic") == sample_ids,
            f"{job.job_id} diagnostic sample set drift",
        )
        for row in rows:
            subject = row.get("subject_model_key", row.get("model_key"))
            require(subject == job.model_key, f"{job.job_id} diagnostic model drift")
            value = row.get("diagnostic_affect_description", row.get("diagnostic_description"))
            require(
                isinstance(value, str) and value.strip(),
                f"{job.job_id} diagnostic description missing",
            )
        return []
    except (BundleContractError, OSError, ValueError) as exc:
        return [str(exc)]


def _validate_misread(spec: ArtifactSpec, *, job: JobSpec, sample_ids: set[str]) -> list[str]:
    errors = _verify_file(spec, f"{job.job_id} Misread labels")
    if errors:
        return errors
    assert spec.path is not None
    try:
        rows = read_jsonl(spec.path)
        require(len(rows) == job.expected_samples, f"{job.job_id} Misread count drift")
        require(
            _ids(rows, f"{job.job_id} Misread") == sample_ids,
            f"{job.job_id} Misread sample set drift",
        )
        for row in rows:
            subject = row.get("subject_model_key", row.get("model_key"))
            require(subject == job.model_key, f"{job.job_id} Misread model drift")
            label = row.get("final_label", row.get("imported_label"))
            require(
                label in {"MISREAD", "NON_MISREAD"},
                f"{job.job_id} unresolved or invalid Misread label",
            )
        return []
    except (BundleContractError, OSError, ValueError) as exc:
        return [str(exc)]


def audit_contract(contract: MatrixContract) -> dict[str, Any]:
    dataset_ids: dict[tuple[str, str], set[str]] = {}
    dataset_records: list[dict[str, Any]] = []
    failures: list[str] = []
    for key in sorted(contract.dataset_manifests):
        spec = contract.dataset_manifests[key]
        ids, errors = _validate_dataset(
            spec,
            domain=key[0],
            protocol=key[1],
            expected_samples=EXPECTED_SAMPLES[key],
        )
        dataset_ids[key] = ids
        dataset_records.append(
            {
                "domain": key[0],
                "protocol": key[1],
                "expected_samples": EXPECTED_SAMPLES[key],
                "status": "invalid" if errors else "ready",
                "errors": errors,
            }
        )
        failures.extend(errors)

    job_records: list[dict[str, Any]] = []
    pending: list[dict[str, str]] = []
    for job in sorted(contract.jobs, key=lambda item: (item.domain, item.model_key)):
        artifact_records: dict[str, Any] = {}
        ids = dataset_ids[(job.domain, job.protocol)]
        for kind in ARTIFACT_TYPES:
            spec = job.artifacts[kind]
            if spec.status == "pending":
                record = {"status": "pending", "reason": spec.reason}
                pending.append({"job_id": job.job_id, "artifact": kind, "reason": str(spec.reason)})
            else:
                if kind == "cache":
                    errors = _validate_cache(spec, job=job, sample_ids=ids)
                elif kind == "ground_truth":
                    errors = _validate_ground_truth(spec, job=job, sample_ids=ids)
                elif kind == "diagnostic_description":
                    errors = _validate_diagnostic(spec, job=job, sample_ids=ids)
                else:
                    errors = _validate_misread(spec, job=job, sample_ids=ids)
                record = {
                    "status": "invalid" if errors else "ready",
                    "path": str(spec.path),
                    "sha256": spec.sha256,
                    "provenance_path": str(spec.provenance_path),
                    "provenance_sha256": spec.provenance_sha256,
                    "errors": errors,
                }
                failures.extend(errors)
            artifact_records[kind] = record
        job_records.append(
            {
                "job_id": job.job_id,
                "domain": job.domain,
                "model_key": job.model_key,
                "protocol": job.protocol,
                "expected_samples": job.expected_samples,
                "artifacts": artifact_records,
            }
        )
    ready = not pending and not failures
    return {
        "schema": READINESS_SCHEMA,
        "status": "READY" if ready else "PENDING" if pending and not failures else "BLOCKED",
        "publishable": ready,
        "matrix_sha256": sha256_file(contract.config_path),
        "contract": {
            "models": len(MODEL_PROTOCOLS),
            "domains": list(DOMAINS),
            "jobs": len(contract.jobs),
            "prompt_count": PROMPT_COUNT,
            "conditions": list(CONDITIONS),
            "expected_samples": {
                f"{domain}/{protocol}": count
                for (domain, protocol), count in EXPECTED_SAMPLES.items()
            },
            "target_intersection_policy": "protocol_specific_no_intersection",
            "phi4_source_misread_required": True,
        },
        "datasets": dataset_records,
        "jobs": job_records,
        "pending": pending,
        "failures": failures,
        "counts": {
            "pending_artifacts": len(pending),
            "invalid_checks": len(failures),
            "ready_artifacts": sum(
                record["status"] == "ready"
                for job in job_records
                for record in job["artifacts"].values()
            ),
            "total_artifacts": len(contract.jobs) * len(ARTIFACT_TYPES),
        },
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_readiness(report: Mapping[str, Any], path: Path) -> tuple[Path, Path]:
    json_path = path.resolve()
    md_path = json_path.with_suffix(".md")
    _atomic_text(json_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    lines = [
        "# TAFFC bundle readiness",
        "",
        f"- Status: **{report['status']}**",
        f"- Publishable: **{str(report['publishable']).lower()}**",
        f"- Jobs: {report['contract']['jobs']}",
        (
            f"- Ready artifacts: {report['counts']['ready_artifacts']}/"
            f"{report['counts']['total_artifacts']}"
        ),
        f"- Pending artifacts: {report['counts']['pending_artifacts']}",
        f"- Invalid checks: {report['counts']['invalid_checks']}",
        "",
        "## Pending",
        "",
    ]
    pending = report["pending"]
    lines.extend(f"- `{row['job_id']}` / `{row['artifact']}`: {row['reason']}" for row in pending)
    if not pending:
        lines.append("- None")
    lines.extend(["", "## Failures", ""])
    failures = report["failures"]
    lines.extend(f"- {error}" for error in failures)
    if not failures:
        lines.append("- None")
    _atomic_text(md_path, "\n".join(lines) + "\n")
    return json_path, md_path


def _safe_relative_name(job: JobSpec, kind: str, path: Path) -> Path:
    suffix = "".join(path.suffixes) or ".bin"
    return Path("artifacts") / job.domain / job.model_key / f"{kind}{suffix}"


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> None:
    require(source.is_file(), f"missing source artifact: {source}")
    require(sha256_file(source) == expected_sha256, f"source artifact SHA drift: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    require(sha256_file(target) == expected_sha256, f"copied artifact SHA drift: {target}")


def build_bundle(contract: MatrixContract, output: Path) -> Path:
    report = audit_contract(contract)
    if not report["publishable"]:
        raise BundleContractError(
            "complete bundle publication is fail-closed: "
            f"pending={len(report['pending'])} invalid={len(report['failures'])}"
        )
    output = output.expanduser().resolve()
    require(not output.exists(), f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    inventory: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "matrix_sha256": report["matrix_sha256"],
        "contract": report["contract"],
        "datasets": {},
        "jobs": {},
    }
    try:
        for (domain, protocol), spec in sorted(contract.dataset_manifests.items()):
            assert spec.path is not None and spec.sha256 is not None
            assert spec.provenance_path is not None and spec.provenance_sha256 is not None
            manifest_target = Path("datasets") / domain / protocol.lower() / "manifest.jsonl"
            provenance_target = Path("datasets") / domain / protocol.lower() / "provenance.json"
            _copy_verified(spec.path, staging / manifest_target, spec.sha256)
            _copy_verified(
                spec.provenance_path, staging / provenance_target, spec.provenance_sha256
            )
            inventory["datasets"][f"{domain}/{protocol}"] = {
                "expected_samples": EXPECTED_SAMPLES[(domain, protocol)],
                "manifest": manifest_target.as_posix(),
                "manifest_sha256": spec.sha256,
                "provenance": provenance_target.as_posix(),
                "provenance_sha256": spec.provenance_sha256,
            }
        for job in sorted(contract.jobs, key=lambda item: (item.domain, item.model_key)):
            artifacts: dict[str, Any] = {}
            for kind, spec in job.artifacts.items():
                assert spec.path is not None and spec.sha256 is not None
                assert spec.provenance_path is not None and spec.provenance_sha256 is not None
                target = _safe_relative_name(job, kind, spec.path)
                provenance_target = target.with_name(target.stem + ".provenance.json")
                _copy_verified(spec.path, staging / target, spec.sha256)
                _copy_verified(
                    spec.provenance_path,
                    staging / provenance_target,
                    spec.provenance_sha256,
                )
                artifacts[kind] = {
                    "path": target.as_posix(),
                    "sha256": spec.sha256,
                    "provenance_path": provenance_target.as_posix(),
                    "provenance_sha256": spec.provenance_sha256,
                }
            inventory["jobs"][job.job_id] = {
                "domain": job.domain,
                "model_key": job.model_key,
                "protocol": job.protocol,
                "expected_samples": job.expected_samples,
                "artifacts": artifacts,
            }
        inventory_path = staging / "inventory.json"
        _atomic_text(
            inventory_path,
            json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        checksums = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
            and path.name not in {"SHA256SUMS", "validation_report.json", "README.md"}
        }
        _atomic_text(
            staging / "SHA256SUMS",
            "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())),
        )
        validation = {
            "schema": VALIDATION_SCHEMA,
            "status": "PASS",
            "matrix_sha256": report["matrix_sha256"],
            "inventory_sha256": sha256_file(inventory_path),
            "payload_file_count": len(checksums),
            "readiness": report,
        }
        _atomic_text(
            staging / "validation_report.json",
            json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        _atomic_text(
            staging / "README.md",
            "# TAFFC complete bundle\n\n"
            "This bundle contains the canonical 16-model panel in both source and target "
            "domains. VT uses 1,876 source and 2,035 target samples; VA uses 1,934 source "
            "and 2,190 target samples. Target protocols remain separate and are not forced "
            "onto an incorrect intersection. Every cache, GT description, diagnostic "
            "description, and Misread label artifact is bound to its SHA-256 and provenance "
            "record. `validation_report.json` proves that no pending artifact was published.\n",
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verify_bundle(output)
    return output


def _safe_bundle_member(root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    require(
        not relative.is_absolute() and ".." not in relative.parts, f"unsafe bundle path: {value}"
    )
    target = (root / Path(*relative.parts)).resolve()
    require(target == root or root in target.parents, f"bundle path escapes root: {value}")
    return target


def verify_bundle(output: Path) -> dict[str, Any]:
    root = output.expanduser().resolve()
    require(root.is_dir(), f"bundle does not exist: {root}")
    inventory_path = root / "inventory.json"
    validation_path = root / "validation_report.json"
    sums_path = root / "SHA256SUMS"
    for path in (inventory_path, validation_path, sums_path, root / "README.md"):
        require(path.is_file(), f"bundle control file missing: {path}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    require(inventory.get("schema") == INVENTORY_SCHEMA, "inventory schema mismatch")
    require(validation.get("schema") == VALIDATION_SCHEMA, "validation schema mismatch")
    require(validation.get("status") == "PASS", "validation status is not PASS")
    require(
        validation.get("inventory_sha256") == sha256_file(inventory_path), "inventory SHA mismatch"
    )
    require(inventory.get("contract", {}).get("jobs") == 32, "bundle job count is not 32")
    require(len(inventory.get("jobs", {})) == 32, "bundle inventory job matrix is incomplete")
    require(
        set(inventory["jobs"])
        == {f"{domain}:{model}" for domain in DOMAINS for model in MODEL_PROTOCOLS},
        "bundle inventory job identities differ from canonical matrix",
    )
    expected_sums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, name = line.split("  ", 1)
        require(name not in expected_sums, f"duplicate checksum entry: {name}")
        expected_sums[name] = _sha(digest, f"checksum {name}")
    for name, digest in expected_sums.items():
        path = _safe_bundle_member(root, name)
        require(path.is_file(), f"checksummed file missing: {name}")
        require(sha256_file(path) == digest, f"payload checksum mismatch: {name}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "validation_report.json", "README.md"}
    }
    require(actual == set(expected_sums), "SHA256SUMS does not cover the exact payload set")
    require(validation.get("payload_file_count") == len(expected_sums), "payload file count drift")
    return {
        "status": "PASS",
        "jobs": 32,
        "models": 16,
        "payload_files": len(expected_sums),
        "bundle": str(root),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config",
        type=Path,
        default=Path("configs/packaging/complete_bundle_matrix.yaml"),
    )
    result.add_argument("--output", type=Path)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--readiness-report", type=Path)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.verify_only:
        require(args.output is not None, "--verify-only requires --output")
        print(json.dumps(verify_bundle(args.output), sort_keys=True))
        return 0
    contract = load_contract(args.config)
    if args.readiness_report:
        report = audit_contract(contract)
        paths = write_readiness(report, args.readiness_report)
        print(
            json.dumps(
                {"status": report["status"], "reports": [str(path) for path in paths]},
                sort_keys=True,
            )
        )
        return 0
    require(args.output is not None, "--build requires --output")
    output = build_bundle(contract, args.output)
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
