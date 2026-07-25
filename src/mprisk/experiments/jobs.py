"""Dataclasses and constants describing downstream cache/plan jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mprisk.experiments._io_utils import _load_yaml

PLAN_SCHEMA = "mprisk_downstream_queue_v1"
CONDITIONS = ("M1", "M2", "M12")
OFFICIAL_TEST = "official_test"
CALIBRATION = "aligned_calibration"
TRAINING_SPLITS = frozenset({"relation_train", "relation_val"})
REPRESENTATIONS = (
    "single_point_binary_v1",
    "trajectory_mlp_binary_v1",
    "tme_proxy_anchor_v1",
)


class CacheNotReady(RuntimeError):
    """A recoverable state: extraction has not completed yet."""


@dataclass(frozen=True)
class CacheJob:
    seed: int
    model_key: str
    protocol: str
    source_manifest: Path
    prompt_set: Path
    cache_root: Path
    expected_tasks: int

    @property
    def prompt_set_key(self) -> str:
        return str(_load_yaml(self.prompt_set)["key"])

    @property
    def run_key(self) -> str:
        return f"seed{self.seed}/{self.model_key}/{self.prompt_set_key}"


@dataclass(frozen=True)
class AllowedExternalGpuContext:
    process_name: str
    command_substring: str
    max_process_count: int
    max_gpu_memory_mib_per_process: float


@dataclass(frozen=True)
class DownstreamPlan:
    repo_root: Path
    jobs: tuple[CacheJob, ...]
    split_assignment: Path
    config_root: Path
    output_root: Path
    physical_gpu: int
    device: str
    max_gpu_memory_fraction: float
    poll_seconds: int
    lock_path: Path
    retention_seed: int
    retention_fractions: tuple[float, ...]
    producer_tmux_sessions: tuple[str, ...]
    producer_command_substrings: tuple[str, ...]
    allowed_external_gpu_contexts: tuple[AllowedExternalGpuContext, ...]
    max_external_gpu_context_memory_mib: float
