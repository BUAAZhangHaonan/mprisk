"""Queue manifest dataclasses, exceptions, and YAML loaders."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mprisk.config.loader import load_yaml

QUEUE_SCHEMA = "mprisk_prefill_dependency_queue_v1"
CLASS_CODE = {"A": "Conflict", "C": "Aligned"}
CLASS_CODE_SEMANTICS = {"A": "sample_type.Conflict", "C": "sample_type.Aligned"}


class GateFailure(RuntimeError):
    """Raised when a dependency gate can no longer succeed without intervention."""


class QueueExecutionError(RuntimeError):
    """Raised when a queued extraction job fails its runtime contract."""


class CapacityFailure(GateFailure):
    """Raised when projected cache artifacts would exceed the capacity limit."""


class QueueLockError(GateFailure):
    """Raised when another process owns the dependent-queue scope."""


@dataclass(frozen=True)
class GateJob:
    model_key: str
    ledger: Path
    expected_tasks: int
    runtime_cache_key: str


@dataclass(frozen=True)
class UpstreamConfig:
    tmux_session: str
    pid: int | None
    heartbeat_max_age_seconds: float
    heartbeat_paths: tuple[Path, ...]


@dataclass(frozen=True)
class MainGate:
    runtime_record: Path
    upstream: UpstreamConfig
    jobs: tuple[GateJob, ...]


@dataclass(frozen=True)
class FollowupJob:
    job_id: str
    seed: int
    model_key: str
    protocol: str
    manifest: Path
    prompt_set: Path
    output_root: Path
    log_path: Path
    expected_tasks: int
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class CapacityOutput:
    output_root: Path
    expected_tasks: int


@dataclass(frozen=True)
class CapacityModel:
    model_key: str
    calibration_root: Path
    outputs: tuple[CapacityOutput, ...]


@dataclass(frozen=True)
class CapacityGate:
    filesystem_path: Path
    max_projected_utilization: float
    models: tuple[CapacityModel, ...]


@dataclass(frozen=True)
class CapacityStatus:
    safe: bool
    filesystem_path: Path
    total_bytes: int
    used_bytes: int
    free_bytes: int
    projected_bytes: int
    projected_used_bytes: int
    projected_utilization: float
    total_inodes: int
    free_inodes: int
    projected_inodes: int
    projected_inode_utilization: float
    max_projected_utilization: float
    models: tuple[dict[str, Any], ...]

    def require_safe(self) -> None:
        if self.safe:
            return
        raise CapacityFailure(
            "Projected cache utilization is "
            f"{self.projected_utilization:.2%} bytes and "
            f"{self.projected_inode_utilization:.2%} inodes; "
            f"limit is {self.max_projected_utilization:.2%}"
        )


@dataclass(frozen=True)
class QueueManifest:
    source_path: Path
    physical_gpu: int
    device: str
    python: Path
    extract_script: Path
    runtime_record: Path
    lock_path: Path
    capacity_gate: CapacityGate
    main_gate: MainGate
    followup_jobs: tuple[FollowupJob, ...]


@dataclass(frozen=True)
class GateStatus:
    ready: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class UpstreamStatus:
    running: bool
    reason: str
    heartbeat_age_seconds: float | None
    seconds_until_stale: float
    process_pid: int | None = None


class EventWatcher(Protocol):
    def wait(self, timeout_seconds: float | None = None) -> None: ...

    def close(self) -> None: ...


JobExecutor = Callable[..., None]
WatcherFactory = Callable[[Sequence[Path], int | None], EventWatcher]
UpstreamChecker = Callable[[QueueManifest], UpstreamStatus]


def load_queue_manifest(path: str | Path) -> QueueManifest:
    source_path = Path(path).expanduser()
    data = load_yaml(source_path)
    if data.get("schema") != QUEUE_SCHEMA:
        raise ValueError(f"Queue manifest schema must be {QUEUE_SCHEMA}")
    physical_gpu = _positive_int(data, "physical_gpu", allow_zero=True)
    device = _required_str(data, "device")
    if device != "cuda:0":
        raise ValueError("Dependent queue requires process-local device cuda:0")
    main_raw = _required_mapping(data, "main_gate")
    upstream_raw = _required_mapping(main_raw, "upstream")
    capacity_raw = _required_mapping(data, "capacity_gate")
    gate_jobs = tuple(_load_gate_job(item) for item in _required_list(main_raw, "jobs"))
    jobs = tuple(_load_followup_job(item) for item in _required_list(data, "followup_jobs"))
    if not gate_jobs or not jobs:
        raise ValueError("Dependent queue requires main-gate and follow-up jobs")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Follow-up job IDs must be unique")
    if len({job.output_root for job in jobs}) != len(jobs):
        raise ValueError("Follow-up output roots must be unique")
    return QueueManifest(
        source_path=source_path.resolve(),
        physical_gpu=physical_gpu,
        device=device,
        python=Path(_required_str(data, "python")).expanduser(),
        extract_script=Path(_required_str(data, "extract_script")).expanduser(),
        runtime_record=Path(_required_str(data, "runtime_record")).expanduser(),
        lock_path=Path(_required_str(data, "lock_path")).expanduser(),
        capacity_gate=_load_capacity_gate(capacity_raw),
        main_gate=MainGate(
            runtime_record=Path(_required_str(main_raw, "runtime_record")).expanduser(),
            upstream=_load_upstream_config(upstream_raw),
            jobs=gate_jobs,
        ),
        followup_jobs=jobs,
    )


def _load_gate_job(data: Any) -> GateJob:
    if not isinstance(data, dict):
        raise ValueError("Main-gate jobs must be mappings")
    return GateJob(
        model_key=_required_str(data, "model_key"),
        ledger=Path(_required_str(data, "ledger")).expanduser(),
        expected_tasks=_positive_int(data, "expected_tasks"),
        runtime_cache_key=_required_str(data, "runtime_cache_key"),
    )


def _load_upstream_config(data: dict[str, Any]) -> UpstreamConfig:
    heartbeat_max_age = data.get("heartbeat_max_age_seconds")
    if (
        not isinstance(heartbeat_max_age, int | float)
        or isinstance(heartbeat_max_age, bool)
        or heartbeat_max_age <= 0
    ):
        raise ValueError("heartbeat_max_age_seconds must be positive")
    heartbeat_values = _required_list(data, "heartbeat_paths")
    if not heartbeat_values or not all(
        isinstance(item, str) and item for item in heartbeat_values
    ):
        raise ValueError("heartbeat_paths must contain non-empty strings")
    heartbeat_paths = tuple(Path(item).expanduser() for item in heartbeat_values)
    pid = data.get("pid")
    if pid is not None and (
        not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
    ):
        raise ValueError("upstream.pid must be a positive integer when provided")
    return UpstreamConfig(
        tmux_session=_required_str(data, "tmux_session"),
        pid=pid,
        heartbeat_max_age_seconds=float(heartbeat_max_age),
        heartbeat_paths=heartbeat_paths,
    )


def _load_capacity_gate(data: dict[str, Any]) -> CapacityGate:
    maximum = data.get("max_projected_utilization")
    if not isinstance(maximum, int | float) or isinstance(maximum, bool):
        raise ValueError("max_projected_utilization must be numeric")
    if not 0 < float(maximum) < 1:
        raise ValueError("max_projected_utilization must be between zero and one")
    models = tuple(_load_capacity_model(item) for item in _required_list(data, "models"))
    if not models:
        raise ValueError("capacity_gate.models must not be empty")
    return CapacityGate(
        filesystem_path=Path(_required_str(data, "filesystem_path")).expanduser(),
        max_projected_utilization=float(maximum),
        models=models,
    )


def _load_capacity_model(data: Any) -> CapacityModel:
    if not isinstance(data, dict):
        raise ValueError("Capacity models must be mappings")
    outputs = tuple(_load_capacity_output(item) for item in _required_list(data, "outputs"))
    if not outputs:
        raise ValueError("Capacity model outputs must not be empty")
    return CapacityModel(
        model_key=_required_str(data, "model_key"),
        calibration_root=Path(_required_str(data, "calibration_root")).expanduser(),
        outputs=outputs,
    )


def _load_capacity_output(data: Any) -> CapacityOutput:
    if not isinstance(data, dict):
        raise ValueError("Capacity outputs must be mappings")
    return CapacityOutput(
        output_root=Path(_required_str(data, "output_root")).expanduser(),
        expected_tasks=_positive_int(data, "expected_tasks"),
    )


def _load_followup_job(data: Any) -> FollowupJob:
    if not isinstance(data, dict):
        raise ValueError("Follow-up jobs must be mappings")
    extra_args = data.get("extra_args", [])
    if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
        raise ValueError("extra_args must be a list of strings")
    protocol = _required_str(data, "protocol")
    if protocol not in {"vt", "va"}:
        raise ValueError(f"Unsupported follow-up protocol: {protocol}")
    return FollowupJob(
        job_id=_required_str(data, "job_id"),
        seed=_positive_int(data, "seed"),
        model_key=_required_str(data, "model_key"),
        protocol=protocol,
        manifest=Path(_required_str(data, "manifest")).expanduser(),
        prompt_set=Path(_required_str(data, "prompt_set")).expanduser(),
        output_root=Path(_required_str(data, "output_root")).expanduser(),
        log_path=Path(_required_str(data, "log_path")).expanduser(),
        expected_tasks=_positive_int(data, "expected_tasks"),
        extra_args=tuple(extra_args),
    )


def _required_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _positive_int(data: dict[str, Any], key: str, *, allow_zero: bool = False) -> int:
    value = data.get(key)
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GateFailure(f"Runtime record must contain an object: {path}")
    return data
