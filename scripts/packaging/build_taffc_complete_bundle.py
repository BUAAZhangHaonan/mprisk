#!/usr/bin/env python3
"""Build and verify the canonical TAFFC complete delivery directory.

This entrypoint is intentionally tied to the frozen 2026-07-16 delivery and
fails closed when any source count, identifier set, task matrix, path, stream,
or checksum differs from the declared contract.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence


EXPECTED_SILENT_IDS = {f"gen:accept_a_va:S{number:04d}" for number in range(544, 549)}
PROMPT_COUNT = 8
CONDITIONS = {"M1", "M2", "M12"}
CONTROL_FILES = {"SHA256SUMS", "file_provenance.tsv"}

FORMAL_LABEL_MODELS = {
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
    "qwen2_5_omni_7b": "VA",
}

UNION_CACHE_SPECS = {
    "qwen3_vl_8b": {
        "protocol": "VT",
        "samples": 1876,
        "tasks": 45024,
        "blocked": 0,
        "path": "outputs/prefill_cache/production_unions/qwen3_vl_8b/"
        "vt_delivery_p8_seed20260717/union_v2.json",
        "sha256": "16aedab3a02d993467828b6d1b0b3f5882a3fb24d47908ae41c3bbe31d8c3ab4",
    },
    "internvl3_5_8b": {
        "protocol": "VT",
        "samples": 1876,
        "tasks": 45024,
        "blocked": 0,
        "path": "outputs/prefill_cache/production_unions/internvl3_5_8b/"
        "vt_delivery_p8_seed20260717/union_v2.json",
        "sha256": "d091276b1de8efe5a50bdcf0e58e29b6a1bd2759bd8d1fdb0a0e781d137829fb",
    },
    "qwen2_5_omni_7b": {
        "protocol": "VA",
        "samples": 1934,
        "tasks": 46416,
        "blocked": 120,
        "path": "outputs/prefill_cache/production_unions/qwen2_5_omni_7b/"
        "va_delivery_p8_seed20260717/union_v2.json",
        "sha256": "7a1a778a8c0995eed05dd90325bb2ca4bc77b3550370b34f5bfc5b947607056e",
    },
}

FULL_CACHE_SPECS = {
    "qwen3_5_4b": {
        "protocol": "VT",
        "path": "outputs/prefill_cache/qwen3_5_4b/vt_main_p8_seed20260717",
        "manifest_sha256": "6169a0031501ae3e097e2ef7315c1d4d12dc76e489253e4c754df3b7671eff8c",
        "full_rows": 93864,
        "full_samples": 3911,
        "valid_rows": 45024,
        "valid_samples": 1876,
    },
    "gemma4_12b": {
        "protocol": "VA",
        "path": "outputs/prefill_cache/gemma4_12b_it/va_delivery_p8_seed20260717",
        "manifest_sha256": "5f638f49c63404af1313d17ed3e04f1b4a23d548a56668610a5e876a0253e62e",
        "full_rows": 46496,
        "full_samples": 1939,
        "valid_rows": 46416,
        "valid_samples": 1934,
    },
}

STATE_SPECS = {
    "qwen3_vl_8b": {
        "protocol": "VT",
        "rows": 1876,
        "sha256": "99184c89cd86f7af3107da77486fbef68848da3b478ef8a515d99c8912ccf001",
    },
    "internvl3_5_8b": {
        "protocol": "VT",
        "rows": 1876,
        "sha256": "df2e2aabeb8bc3c96ac29a367f3217db66217b1b49dc5b5116759d21ed1b194e",
    },
    "qwen2_5_omni_7b": {
        "protocol": "VA",
        "rows": 1934,
        "sha256": "e34a3fb66e493cc38aa58987ff81d4de23adf5b30de71eaaeaea2a1595ba0f5b",
    },
}

STATE_EVIDENCE_FILES = (
    "RUN_COMPLETE.json",
    "calibration/thresholds.json",
    "training/best_checkpoint.pt",
    "training/train_config.yaml",
    "training/train_metrics.json",
    "training/convergence_history.json",
    "official_test/frozen_tme_representations.jsonl",
    "official_test/sdr_scores.jsonl",
    "official_test/state_patterns.jsonl",
    "official_test/geometry_metrics.json",
    "official_test/provenance.json",
    "state_all_registered_splits/sdr_scores.jsonl",
    "state_all_registered_splits/state_patterns.jsonl",
    "state_all_registered_splits/state_summary.json",
    "state_all_registered_splits/sdr_score_summary.json",
)


class BundleError(RuntimeError):
    """Raised for any contract mismatch."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def normalize_rel(value: str | PurePosixPath) -> str:
    path = PurePosixPath(value)
    require(not path.is_absolute(), f"package path must be relative: {value}")
    require(".." not in path.parts, f"package path escapes bundle: {value}")
    require(str(path) not in {"", "."}, f"empty package path: {value}")
    require("\n" not in str(path) and "\r" not in str(path), f"newline in path: {value}")
    return path.as_posix()


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    require(bool(result), f"cannot derive safe filename from {value!r}")
    return result


def id_set(rows: Sequence[dict[str, Any]], source: str) -> set[str]:
    values = [row.get("sample_id") for row in rows]
    require(all(isinstance(value, str) and value for value in values), f"missing sample_id in {source}")
    require(len(values) == len(set(values)), f"duplicate sample_id in {source}")
    return set(values)


def count_types(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("sample_type")) for row in rows).items()))


def validate_task_matrix(
    rows: Sequence[dict[str, Any]], sample_ids: set[str], source: str
) -> None:
    require(len(rows) == len(sample_ids) * PROMPT_COUNT * len(CONDITIONS), f"{source}: task count mismatch")
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str, str]] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        require(sample_id in sample_ids, f"{source}: unexpected sample {sample_id}")
        identity = (sample_id, row.get("prompt_id"), row.get("condition"))
        require(identity not in identities, f"{source}: duplicate task {identity}")
        identities.add(identity)
        by_sample[sample_id].append(row)
    for sample_id, sample_rows in by_sample.items():
        prompts = {row.get("prompt_id") for row in sample_rows}
        conditions = {row.get("condition") for row in sample_rows}
        require(len(prompts) == PROMPT_COUNT, f"{source}: {sample_id} prompt count != 8")
        require(conditions == CONDITIONS, f"{source}: {sample_id} conditions mismatch")
        require(len(sample_rows) == PROMPT_COUNT * len(CONDITIONS), f"{source}: {sample_id} rows != 24")


def path_under(path: Path, root: Path, source: str) -> Path:
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    try:
        return resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise BundleError(f"{source}: {resolved} is outside {root_resolved}") from exc


def resolve_cache_reference(row: dict[str, Any], value: str, source: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve(strict=True)
    root = Path(str(row.get("cache_root", "")))
    require(root.is_absolute(), f"{source}: relative cache_root cannot resolve {value}")
    return (root / path).resolve(strict=True)


@dataclass(frozen=True)
class ProvenanceRecord:
    source: str
    mode: str


class BundleBuilder:
    def __init__(
        self,
        repo_root: Path,
        output: Path,
        workers: int,
        dry_run: bool,
        probe_streams: bool,
        resume_existing: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve(strict=True)
        self.output = output.absolute()
        self.staging = output.parent / f".{output.name}.building"
        self.workers = workers
        self.dry_run = dry_run
        self.probe_streams = probe_streams
        self.resume_existing = resume_existing
        self.records: dict[str, ProvenanceRecord] = {}
        self.expected_sha: dict[str, str] = {}
        self._record_lock = threading.Lock()
        self.checks: list[dict[str, Any]] = []
        self.inventory: dict[str, Any] = {
            "schema": "taffc_complete_bundle_inventory_v1",
            "bundle_name": output.name,
            "scope": {
                "generated_dataset": "3810 valid in-domain protocol rows",
                "natural_dataset": "CH-SIMS v2 cross-domain protocol views",
                "cache_models": list(UNION_CACHE_SPECS) + list(FULL_CACHE_SPECS),
                "formal_misread_models": list(FORMAL_LABEL_MODELS),
                "state_models": list(STATE_SPECS),
                "state_not_computed": ["qwen3_5_4b", "gemma4_12b"],
                "tme_not_trained": ["qwen3_5_4b", "gemma4_12b"],
            },
        }
        self.dataset_rows: dict[str, list[dict[str, Any]]] = {}
        self.dataset_ids: dict[str, set[str]] = {}

    def check(self, name: str, detail: Any) -> None:
        self.checks.append({"name": name, "status": "PASS", "detail": detail})
        print(f"PASS {name}: {detail}", flush=True)

    def prepare(self) -> None:
        require(self.repo_root.joinpath("pyproject.toml").is_file(), "repo root lacks pyproject.toml")
        require(not self.output.exists(), f"target already exists: {self.output}")
        if self.resume_existing:
            require(not self.dry_run, "cannot resume staging during dry-run")
            require(self.staging.is_dir(), f"resume staging does not exist: {self.staging}")
        else:
            require(not self.staging.exists(), f"staging already exists: {self.staging}")
        if not self.dry_run and not self.resume_existing:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.staging.mkdir(parents=False, exist_ok=False)
        self.check(
            "preflight",
            {
                "repo_root": str(self.repo_root),
                "output_absent": True,
                "resume_existing_staging": self.resume_existing,
            },
        )

    def _target(self, rel: str) -> Path:
        rel = normalize_rel(rel)
        target = self.staging.joinpath(*PurePosixPath(rel).parts)
        require(target == self.staging / Path(rel), f"unexpected package target mapping: {rel}")
        return target

    def _register(self, rel: str, source: str, mode: str) -> None:
        rel = normalize_rel(rel)
        with self._record_lock:
            existing = self.records.get(rel)
            require(existing is None, f"duplicate package path: {rel}; existing={existing}; source={source}")
            self.records[rel] = ProvenanceRecord(source=source, mode=mode)

    def write_bytes(self, rel: str, payload: bytes, source: str, mode: str = "generated") -> None:
        self._register(rel, source, mode)
        if self.dry_run:
            return
        target = self._target(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.resume_existing and target.exists():
            require(target.is_file() and not target.is_symlink(), f"bad resumed generated file: {target}")
            if target.read_bytes() != payload:
                replacement = target.with_name(f".{target.name}.resume-replacement")
                require(not replacement.exists(), f"stale generated replacement exists: {replacement}")
                with replacement.open("xb") as handle:
                    handle.write(payload)
                os.replace(replacement, target)
            return
        with target.open("xb") as handle:
            handle.write(payload)

    def write_text(self, rel: str, text: str, source: str, mode: str = "generated") -> None:
        self.write_bytes(rel, text.encode("utf-8"), source, mode)

    def write_json(self, rel: str, value: Any, source: str, mode: str = "generated") -> None:
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self.write_text(rel, payload, source, mode)

    def write_jsonl(
        self,
        rel: str,
        rows: Iterable[dict[str, Any]],
        source: str,
        mode: str = "generated",
    ) -> None:
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        self.write_text(rel, payload, source, mode)

    def link(self, source: Path, rel: str, nonzero: bool = False) -> None:
        source = source.resolve(strict=True)
        require(source.is_file(), f"source is not a file: {source}")
        require(not source.is_symlink(), f"source symlink is forbidden: {source}")
        if nonzero:
            require(source.stat().st_size > 0, f"zero-size payload: {source}")
        target_device = (
            self.staging.stat().st_dev
            if not self.dry_run
            else self.output.parent.resolve(strict=True).stat().st_dev
        )
        source_stat = source.stat()
        if source_stat.st_dev == target_device and source_stat.st_uid == os.geteuid():
            mode = "hardlink"
        elif source_stat.st_dev == target_device:
            mode = "copy_cross_owner"
        else:
            mode = "copy_cross_device"
        self._register(rel, str(source), mode)
        if self.dry_run:
            return
        target = self._target(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.resume_existing and target.exists():
            require(target.is_file() and not target.is_symlink(), f"bad resumed payload: {target}")
            target_stat = target.stat()
            if mode == "hardlink":
                require(
                    (source_stat.st_dev, source_stat.st_ino)
                    == (target_stat.st_dev, target_stat.st_ino),
                    f"resumed hardlink inode mismatch: {source} -> {target}",
                )
            else:
                require(source_stat.st_size == target_stat.st_size, f"resumed copied size mismatch: {target}")
                require(sha256_file(source) == sha256_file(target), f"resumed copied SHA mismatch: {target}")
            return
        if mode == "hardlink":
            os.link(source, target)
        else:
            with source.open("rb") as source_handle, target.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
        target_stat = target.stat()
        if mode == "hardlink":
            require(
                (source_stat.st_dev, source_stat.st_ino) == (target_stat.st_dev, target_stat.st_ino),
                f"hardlink inode mismatch: {source} -> {target}",
            )
        else:
            require(source_stat.st_size == target_stat.st_size, f"copied size mismatch: {source} -> {target}")

    def link_many(self, items: Sequence[tuple[Path, str, bool]], label: str) -> None:
        print(f"LINK {label}: {len(items)} files", flush=True)
        if self.dry_run or len(items) < 64:
            for index, item in enumerate(items, 1):
                self.link(*item)
                if index % 25000 == 0:
                    print(f"LINK {label}: {index}/{len(items)}", flush=True)
            return
        for start in range(0, len(items), 2000):
            batch = items[start : start + 2000]
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                list(executor.map(lambda item: self.link(*item), batch))
            completed = min(start + len(batch), len(items))
            if completed % 20000 == 0 or completed == len(items):
                print(f"LINK {label}: {completed}/{len(items)}", flush=True)

    def register_expected_sha(self, rel: str, checksum: str, source: str) -> None:
        require(re.fullmatch(r"[0-9a-f]{64}", checksum) is not None, f"invalid checksum in {source}")
        existing = self.expected_sha.get(rel)
        require(existing in (None, checksum), f"conflicting expected checksum for {rel}")
        self.expected_sha[rel] = checksum

    def _probe_one(self, item: tuple[Path, bool, bool]) -> tuple[str, set[str]]:
        path, need_video, need_audio = item
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        streams = {row.get("codec_type") for row in json.loads(result.stdout).get("streams", [])}
        if need_video:
            require("video" in streams, f"missing video stream: {path}")
        if need_audio:
            require("audio" in streams, f"missing audio stream: {path}")
        return str(path), {str(value) for value in streams if value}

    def probe_media(self, requirements: dict[Path, tuple[bool, bool]], label: str) -> None:
        if not self.probe_streams:
            self.check(f"{label}_stream_probe", "disabled only by explicit CLI flag")
            return
        items = [(path, flags[0], flags[1]) for path, flags in sorted(requirements.items(), key=lambda x: str(x[0]))]
        print(f"PROBE {label}: {len(items)} media files", flush=True)
        with ThreadPoolExecutor(max_workers=min(self.workers, 4)) as executor:
            for index, _ in enumerate(executor.map(self._probe_one, items), 1):
                if index % 500 == 0:
                    print(f"PROBE {label}: {index}/{len(items)}", flush=True)
        self.check(f"{label}_stream_probe", len(items))

    def build_datasets(self) -> None:
        delivery = self.repo_root / "data/processed/manifests/delivery_20260716"
        vt_source = delivery / "vt_filtered.jsonl"
        va_source = delivery / "va_filtered.jsonl"
        vt_rows = read_jsonl(vt_source)
        va_all_rows = read_jsonl(va_source)
        require(len(vt_rows) == 1876, "generated VT row count must be 1876")
        require(count_types(vt_rows) == {"Aligned": 1144, "Conflict": 732}, "generated VT type counts mismatch")
        require(len(va_all_rows) == 1939, "source generated VA row count must be 1939")
        excluded = [row for row in va_all_rows if row.get("sample_id") in EXPECTED_SILENT_IDS]
        va_rows = [row for row in va_all_rows if row.get("sample_id") not in EXPECTED_SILENT_IDS]
        require(id_set(excluded, "excluded VA") == EXPECTED_SILENT_IDS, "silent VA ID set mismatch")
        require(len(va_rows) == 1934, "generated valid VA row count must be 1934")
        require(count_types(va_rows) == {"Aligned": 1093, "Conflict": 841}, "generated VA type counts mismatch")
        vt_ids = id_set(vt_rows, "generated VT")
        va_ids = id_set(va_rows, "generated VA")
        require(vt_ids.isdisjoint(va_ids), "generated VT and VA sample IDs overlap")

        requirements: dict[Path, tuple[bool, bool]] = {}
        media_targets: dict[Path, str] = {}
        rewritten: dict[str, list[dict[str, Any]]] = {"VT": [], "VA": []}
        links: dict[str, tuple[Path, str, bool]] = {}
        for protocol, rows in (("VT", vt_rows), ("VA", va_rows)):
            for row in rows:
                require(row.get("protocol") == protocol, f"generated {protocol} protocol mismatch")
                paths = row.get("media_paths")
                require(isinstance(paths, dict) and paths, f"missing media_paths for {row.get('sample_id')}")
                output_row = copy.deepcopy(row)
                for modality, raw_path in paths.items():
                    source = Path(str(raw_path)).resolve(strict=True)
                    require(source.is_file() and source.stat().st_size > 0, f"bad media: {source}")
                    target_rel = media_targets.get(source)
                    if target_rel is None:
                        suffix = source.suffix.lower() or ".bin"
                        target_rel = f"datasets/generated_3810/media/{protocol}/{safe_name(row['sample_id'])}{suffix}"
                        require(target_rel not in links, f"generated media target collision: {target_rel}")
                        media_targets[source] = target_rel
                        links[target_rel] = (source, target_rel, True)
                    output_row["media_paths"][modality] = target_rel
                    old_video, old_audio = requirements.get(source, (False, False))
                    requirements[source] = (old_video or modality == "vision", old_audio or protocol == "VA" or modality == "audio")
                rewritten[protocol].append(output_row)
        require(len(media_targets) == 3810, "generated media must contain 3810 unique files")

        silent_requirements: dict[Path, tuple[bool, bool]] = {}
        excluded_ledger: list[dict[str, Any]] = []
        for row in excluded:
            source = Path(str(row["media_paths"]["vision"])).resolve(strict=True)
            require(source.is_file() and source.stat().st_size > 0, f"bad excluded media: {source}")
            silent_requirements[source] = (True, False)
            ledger_row = copy.deepcopy(row)
            ledger_row["exclusion_reason"] = "missing_audio_stream"
            excluded_ledger.append(ledger_row)
        if self.probe_streams:
            for path, _flags in silent_requirements.items():
                _, streams = self._probe_one((path, True, False))
                require("audio" not in streams, f"excluded silent file unexpectedly has audio: {path}")
            self.check("generated_excluded_silent_streams", len(silent_requirements))
        self.probe_media(requirements, "generated_valid")
        self.link_many(list(links.values()), "generated media")

        self.write_jsonl("datasets/generated_3810/manifests/vt.jsonl", rewritten["VT"], str(vt_source), "generated_rewrite")
        self.write_jsonl("datasets/generated_3810/manifests/va.jsonl", rewritten["VA"], str(va_source), "generated_rewrite")
        self.write_jsonl(
            "datasets/generated_3810/manifests/all.jsonl",
            [*rewritten["VT"], *rewritten["VA"]],
            f"derived from {vt_source} and {va_source}",
            "generated_rewrite",
        )
        self.write_jsonl(
            "datasets/generated_3810/manifests/excluded_silent_va.jsonl",
            sorted(excluded_ledger, key=lambda row: row["sample_id"]),
            str(va_source),
            "generated_exclusion_ledger",
        )
        self.link(vt_source, "provenance/datasets/generated_3810/vt_filtered.original.jsonl", True)
        self.link(va_source, "provenance/datasets/generated_3810/va_filtered.original.jsonl", True)
        self.write_text(
            "datasets/generated_3810/README.md",
            "# Generated in-domain dataset\n\n"
            "Canonical valid population: 3,810 protocol rows (VT 1,876; VA 1,934). "
            "The five listed VA Conflict samples are excluded because their files have no audio stream. "
            "All `media_paths` in the delivery manifests are relative to the bundle root.\n",
            "packaging contract",
        )
        self.dataset_rows["VT"] = vt_rows
        self.dataset_rows["VA"] = va_rows
        self.dataset_ids["VT"] = vt_ids
        self.dataset_ids["VA"] = va_ids
        self.inventory["generated_3810"] = {
            "protocol_rows": 3810,
            "unique_sample_ids": 3810,
            "unique_media": 3810,
            "VT": {"rows": 1876, "sample_types": count_types(vt_rows)},
            "VA": {"rows": 1934, "sample_types": count_types(va_rows)},
            "excluded_silent_va": sorted(EXPECTED_SILENT_IDS),
        }
        self.check("generated_3810", self.inventory["generated_3810"])

        merged = self.repo_root / "data/processed/manifests/protocol_manifests_merged"
        natural_sources = {
            "VT": merged / "vt_merged_primary.jsonl",
            "VA": merged / "va_merged_primary.jsonl",
        }
        natural_rows: dict[str, list[dict[str, Any]]] = {}
        for protocol, source in natural_sources.items():
            natural_rows[protocol] = [row for row in read_jsonl(source) if row.get("source_dataset") == "ch_sims_v2"]
        require(len(natural_rows["VT"]) == 2035, "CH-SIMS VT rows must be 2035")
        require(len(natural_rows["VA"]) == 2190, "CH-SIMS VA rows must be 2190")
        require(count_types(natural_rows["VT"]) == {"Aligned": 1888, "Conflict": 147}, "CH-SIMS VT types mismatch")
        require(count_types(natural_rows["VA"]) == {"Aligned": 2141, "Conflict": 49}, "CH-SIMS VA types mismatch")
        natural_ids = id_set([*natural_rows["VT"], *natural_rows["VA"]], "CH-SIMS protocol rows")
        require(len(natural_ids) == 4225, "CH-SIMS protocol IDs must total 4225")
        media_root = Path("/home/team/lvshuyang/TAFFC/mprisk/curation/outputs/media_cropped")
        natural_media_targets: dict[Path, str] = {}
        natural_requirements: dict[Path, tuple[bool, bool]] = {}
        natural_links: dict[str, tuple[Path, str, bool]] = {}
        natural_rewritten: dict[str, list[dict[str, Any]]] = {"VT": [], "VA": []}
        for protocol in ("VT", "VA"):
            for row in natural_rows[protocol]:
                require(row.get("protocol") == protocol, f"CH-SIMS {protocol} protocol mismatch")
                output_row = copy.deepcopy(row)
                for modality, raw_path in row["media_paths"].items():
                    source = Path(str(raw_path)).resolve(strict=True)
                    relative = path_under(source, media_root, "CH-SIMS media")
                    target_rel = f"datasets/ch_sims_v2_cross_domain/media/{relative.as_posix()}"
                    previous = natural_media_targets.setdefault(source, target_rel)
                    require(previous == target_rel, f"CH-SIMS target conflict for {source}")
                    natural_links[target_rel] = (source, target_rel, True)
                    output_row["media_paths"][modality] = target_rel
                    old_video, old_audio = natural_requirements.get(source, (False, False))
                    natural_requirements[source] = (old_video or modality == "vision", old_audio or protocol == "VA" or modality == "audio")
                natural_rewritten[protocol].append(output_row)
        require(len(natural_media_targets) == 2445, "CH-SIMS unique media must be 2445")
        self.probe_media(natural_requirements, "ch_sims_v2")
        self.link_many(list(natural_links.values()), "CH-SIMS media")
        for protocol in ("VT", "VA"):
            lower = protocol.lower()
            source = natural_sources[protocol]
            self.write_jsonl(
                f"datasets/ch_sims_v2_cross_domain/manifests/{lower}.jsonl",
                natural_rewritten[protocol],
                str(source),
                "generated_filter_rewrite",
            )
            self.write_jsonl(
                f"provenance/datasets/ch_sims_v2/{lower}.filtered.original.jsonl",
                natural_rows[protocol],
                str(source),
                "generated_filter",
            )
            self.link(source, f"provenance/datasets/ch_sims_v2/{source.name}.original", True)
        self.write_jsonl(
            "datasets/ch_sims_v2_cross_domain/manifests/all.jsonl",
            [*natural_rewritten["VT"], *natural_rewritten["VA"]],
            "filtered CH-SIMS VT and VA protocol manifests",
            "generated_filter_rewrite",
        )
        self.write_text(
            "datasets/ch_sims_v2_cross_domain/README.md",
            "# CH-SIMS v2 cross-domain natural dataset\n\n"
            "This package contains 4,225 protocol rows: 2,035 VT and 2,190 VA. "
            "They reference 2,445 unique cropped media files; overlap between protocol views is expected.\n",
            "packaging contract",
        )
        self.inventory["ch_sims_v2_cross_domain"] = {
            "protocol_rows": 4225,
            "unique_sample_ids": 4225,
            "unique_media": 2445,
            "VT": {"rows": 2035, "sample_types": count_types(natural_rows["VT"])},
            "VA": {"rows": 2190, "sample_types": count_types(natural_rows["VA"])},
        }
        self.dataset_rows["CH_SIMS_VT"] = natural_rows["VT"]
        self.dataset_ids["CH_SIMS_VT"] = {row["sample_id"] for row in natural_rows["VT"]}
        self.check("ch_sims_v2_cross_domain", self.inventory["ch_sims_v2_cross_domain"])

    def _union_root_maps(
        self, model: str, data: dict[str, Any]
    ) -> tuple[dict[Path, str], dict[Path, str]]:
        sources = data.get("provenance", {}).get("sources")
        require(isinstance(sources, list) and sources, f"{model}: union provenance sources missing")
        root_targets: dict[Path, str] = {}
        evidence_targets: dict[Path, str] = {}
        source_ids: set[str] = set()
        for source in sources:
            source_id = safe_name(str(source.get("source_id")))
            require(source_id not in source_ids, f"{model}: duplicate source_id {source_id}")
            source_ids.add(source_id)
            root = Path(str(source.get("cache_root"))).resolve(strict=True)
            root_targets[root] = f"caches/{model}/source_cache/{source_id}"
            evidence = Path(str(source.get("evidence_path"))).resolve(strict=True)
            evidence_targets[evidence] = f"caches/{model}/evidence/{source_id}.json"
        return root_targets, evidence_targets

    @staticmethod
    def _map_union_path(path: Path, root_targets: dict[Path, str], source: str) -> str:
        resolved = path.resolve(strict=True)
        candidates: list[tuple[int, str]] = []
        for root, target_root in root_targets.items():
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            candidates.append((len(root.parts), f"{target_root}/{relative.as_posix()}"))
        require(bool(candidates), f"{source}: cache path not under declared roots: {path}")
        return max(candidates, key=lambda item: item[0])[1]

    def build_union_cache(self, model: str, spec: dict[str, Any]) -> None:
        source_index = self.repo_root / str(spec["path"])
        require(sha256_file(source_index) == spec["sha256"], f"{model}: union source SHA mismatch")
        data = json.loads(source_index.read_text(encoding="utf-8"))
        entries = data.get("entries")
        blocked = data.get("blocked_tasks")
        require(isinstance(entries, list) and isinstance(blocked, list), f"{model}: malformed union")
        sample_ids = {row.get("sample_id") for row in entries}
        expected_ids = self.dataset_ids[str(spec["protocol"])]
        require(sample_ids == expected_ids, f"{model}: union sample set mismatch")
        require(len(entries) == spec["tasks"], f"{model}: union task count mismatch")
        validate_task_matrix(entries, expected_ids, f"{model} union")
        require(len(blocked) == spec["blocked"], f"{model}: blocked task count mismatch")
        if model == "qwen2_5_omni_7b":
            require({row.get("sample_id") for row in blocked} == EXPECTED_SILENT_IDS, "Omni blocked sample IDs mismatch")
            require(Counter(row.get("sample_id") for row in blocked) == Counter({sample_id: 24 for sample_id in EXPECTED_SILENT_IDS}), "Omni blocked tasks must be 24 per silent sample")
            require({row.get("reason") for row in blocked} == {"missing_audio_stream"}, "Omni blocked reasons mismatch")

        root_targets, evidence_targets = self._union_root_maps(model, data)
        link_map: dict[str, tuple[Path, str, bool]] = {}
        for root, target_root in root_targets.items():
            for path in sorted(item for item in root.iterdir() if item.is_file()):
                target = f"{target_root}/{path.name}"
                link_map[target] = (path, target, False)
        for source, target in evidence_targets.items():
            link_map[target] = (source, target, True)

        rewritten = copy.deepcopy(data)
        for original, output in zip(entries, rewritten["entries"], strict=True):
            shard_source = Path(str(original["shard_path"])).resolve(strict=True)
            sidecar_source = Path(str(original["metadata"]["sidecar_path"])).resolve(strict=True)
            shard_target = self._map_union_path(shard_source, root_targets, model)
            sidecar_target = self._map_union_path(sidecar_source, root_targets, model)
            link_map[shard_target] = (shard_source, shard_target, True)
            link_map[sidecar_target] = (sidecar_source, sidecar_target, True)
            self.register_expected_sha(shard_target, str(original["checksum"]), f"{model} union")
            output["cache_root"] = self._map_union_path(Path(str(original["cache_root"])), root_targets, model)
            output["shard_path"] = shard_target
            output["metadata"]["sidecar_path"] = sidecar_target
            source_provenance = output.get("source_provenance")
            require(isinstance(source_provenance, dict), f"{model}: source_provenance missing")
            source_provenance["source_cache_root"] = self._map_union_path(
                Path(str(original["source_provenance"]["source_cache_root"])), root_targets, model
            )
            source_provenance["ledger_path"] = self._map_union_path(
                Path(str(original["source_provenance"]["ledger_path"])), root_targets, model
            )
            source_provenance["sidecar_path"] = sidecar_target
        for original, output in zip(data["provenance"]["sources"], rewritten["provenance"]["sources"], strict=True):
            output["cache_root"] = self._map_union_path(Path(str(original["cache_root"])), root_targets, model)
            output["ledger_path"] = self._map_union_path(Path(str(original["ledger_path"])), root_targets, model)
            evidence_source = Path(str(original["evidence_path"])).resolve(strict=True)
            output["evidence_path"] = evidence_targets[evidence_source]

        self.link_many(list(link_map.values()), f"{model} union cache")
        self.link(source_index, f"provenance/caches/{model}/union_v2.original.json", True)
        self.write_json(
            f"caches/{model}/index/union.package.json",
            rewritten,
            str(source_index),
            "generated_cache_path_rewrite",
        )
        self.write_jsonl(
            f"caches/{model}/index/blocked_tasks.jsonl",
            blocked,
            str(source_index),
            "generated_cache_ledger",
        )
        self.inventory.setdefault("caches", {})[model] = {
            "kind": "production_union",
            "protocol": spec["protocol"],
            "samples": len(sample_ids),
            "successful_tasks": len(entries),
            "blocked_tasks": len(blocked),
            "source_roots": len(root_targets),
            "tensor_and_metadata_files": len(link_map),
        }
        self.check(f"cache_{model}", self.inventory["caches"][model])

    def _rewrite_local_cache_row(
        self, model: str, source_root: Path, row: dict[str, Any]
    ) -> tuple[dict[str, Any], str, str]:
        output = copy.deepcopy(row)
        shard_source = resolve_cache_reference(row, str(row["shard_path"]), model)
        sidecar_source = resolve_cache_reference(row, str(row["metadata"]["sidecar_path"]), model)
        shard_relative = path_under(shard_source, source_root, model)
        sidecar_relative = path_under(sidecar_source, source_root, model)
        cache_root_relative = path_under(Path(str(row["cache_root"])), source_root, model)
        shard_target = f"caches/{model}/source_cache/{shard_relative.as_posix()}"
        sidecar_target = f"caches/{model}/source_cache/{sidecar_relative.as_posix()}"
        output["cache_root"] = f"caches/{model}/source_cache/{cache_root_relative.as_posix()}"
        output["shard_path"] = shard_target
        output["metadata"]["sidecar_path"] = sidecar_target
        return output, shard_target, sidecar_target

    def _read_failed_tasks(self, database: Path) -> list[dict[str, Any]]:
        connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT task_id, sample_id, model_key, protocol, prompt_set_key, prompt_id, "
                "condition, sample_type, source_dataset, status, attempts, error_type, error_message "
                "FROM tasks WHERE status = 'failed' ORDER BY sample_id, prompt_id, condition"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def build_full_cache(self, model: str, spec: dict[str, Any]) -> None:
        source_root = (self.repo_root / str(spec["path"])).resolve(strict=True)
        manifest = source_root / "manifest.jsonl"
        require(sha256_file(manifest) == spec["manifest_sha256"], f"{model}: source manifest SHA mismatch")
        rows = read_jsonl(manifest)
        require(len(rows) == spec["full_rows"], f"{model}: full manifest row count mismatch")
        require(len({row["sample_id"] for row in rows}) == spec["full_samples"], f"{model}: full sample count mismatch")

        tree_items: list[tuple[Path, str, bool]] = []
        excluded_backup_files = 0
        excluded_transient_sqlite_files = 0
        for path in sorted(source_root.rglob("*")):
            relative = path.relative_to(source_root)
            if ".bak_wrong_proto" in relative.parts:
                if path.is_file():
                    excluded_backup_files += 1
                continue
            if path.name.endswith((".sqlite3-shm", ".sqlite3-wal")):
                if path.is_file():
                    excluded_transient_sqlite_files += 1
                continue
            require(not path.is_symlink(), f"{model}: source cache symlink forbidden: {path}")
            if path.is_file():
                target = f"caches/{model}/source_cache/{relative.as_posix()}"
                tree_items.append((path, target, False))
        self.link_many(tree_items, f"{model} complete source cache")

        rewritten_rows: list[dict[str, Any]] = []
        for row in rows:
            output, shard_target, sidecar_target = self._rewrite_local_cache_row(model, source_root, row)
            require((self.staging / shard_target).stat().st_size > 0 if not self.dry_run else resolve_cache_reference(row, str(row["shard_path"]), model).stat().st_size > 0, f"{model}: zero tensor")
            require((self.staging / sidecar_target).stat().st_size > 0 if not self.dry_run else resolve_cache_reference(row, str(row["metadata"]["sidecar_path"]), model).stat().st_size > 0, f"{model}: zero sidecar")
            self.register_expected_sha(shard_target, str(row["checksum"]), f"{model} manifest")
            rewritten_rows.append(output)

        if model == "qwen3_5_4b":
            valid_rows = [row for row in rows if row.get("dataset_key") == "delivery_20260716"]
            natural_rows = [row for row in rows if row.get("dataset_key") == "ch_sims_v2"]
            require({row["sample_id"] for row in valid_rows} == self.dataset_ids["VT"], "Qwen3.5 generated sample set mismatch")
            require({row["sample_id"] for row in natural_rows} == self.dataset_ids["CH_SIMS_VT"], "Qwen3.5 CH-SIMS sample set mismatch")
            validate_task_matrix(valid_rows, self.dataset_ids["VT"], "Qwen3.5 generated subset")
            validate_task_matrix(natural_rows, self.dataset_ids["CH_SIMS_VT"], "Qwen3.5 CH-SIMS subset")
            valid_rewritten = [row for row in rewritten_rows if row.get("dataset_key") == "delivery_20260716"]
            require(len(valid_rewritten) == spec["valid_rows"], "Qwen3.5 valid row count mismatch")
            self.write_jsonl(
                f"caches/{model}/index/generated_valid_manifest.package.jsonl",
                valid_rewritten,
                str(manifest),
                "generated_cache_filter_rewrite",
            )
            extra = {"ch_sims_tasks": len(natural_rows), "failed_tasks": 0}
        else:
            valid_rows = [row for row in rows if row.get("sample_id") not in EXPECTED_SILENT_IDS]
            excluded_success = [row for row in rows if row.get("sample_id") in EXPECTED_SILENT_IDS]
            require({row["sample_id"] for row in valid_rows} == self.dataset_ids["VA"], "Gemma valid sample set mismatch")
            validate_task_matrix(valid_rows, self.dataset_ids["VA"], "Gemma valid subset")
            require(len(excluded_success) == 80, "Gemma excluded successful tasks must be 80")
            require(Counter(row["sample_id"] for row in excluded_success) == Counter({sample_id: 16 for sample_id in EXPECTED_SILENT_IDS}), "Gemma excluded successes must be 16 per silent sample")
            require({row["condition"] for row in excluded_success} == {"M1", "M12"}, "Gemma silent success conditions mismatch")
            failed = self._read_failed_tasks(source_root / "batch_state.sqlite3")
            require(len(failed) == 40, "Gemma failed task count must be 40")
            require(Counter(row["sample_id"] for row in failed) == Counter({sample_id: 8 for sample_id in EXPECTED_SILENT_IDS}), "Gemma failures must be 8 per silent sample")
            require({row["condition"] for row in failed} == {"M2"}, "Gemma failed tasks must all be M2")
            valid_rewritten = [row for row in rewritten_rows if row.get("sample_id") not in EXPECTED_SILENT_IDS]
            excluded_rewritten = [row for row in rewritten_rows if row.get("sample_id") in EXPECTED_SILENT_IDS]
            require(len(valid_rewritten) == spec["valid_rows"], "Gemma valid row count mismatch")
            self.write_jsonl(
                f"caches/{model}/index/generated_valid_manifest.package.jsonl",
                valid_rewritten,
                str(manifest),
                "generated_cache_filter_rewrite",
            )
            self.write_jsonl(
                f"caches/{model}/index/excluded_silent_successes.package.jsonl",
                excluded_rewritten,
                str(manifest),
                "generated_cache_exclusion_ledger",
            )
            self.write_jsonl(
                f"caches/{model}/index/failed_tasks.jsonl",
                failed,
                str(source_root / "batch_state.sqlite3"),
                "generated_sqlite_failure_ledger",
            )
            extra = {"excluded_silent_successful_tasks": 80, "failed_tasks": 40}

        self.write_jsonl(
            f"caches/{model}/index/full_manifest.package.jsonl",
            rewritten_rows,
            str(manifest),
            "generated_cache_path_rewrite",
        )
        self.inventory.setdefault("caches", {})[model] = {
            "kind": "complete_source_cache",
            "protocol": spec["protocol"],
            "full_samples": spec["full_samples"],
            "full_successful_tasks": spec["full_rows"],
            "valid_generated_samples": spec["valid_samples"],
            "valid_generated_tasks": spec["valid_rows"],
            "source_files": len(tree_items),
            "excluded_backup_files": excluded_backup_files,
            "excluded_transient_sqlite_files": excluded_transient_sqlite_files,
            **extra,
        }
        self.check(f"cache_{model}", self.inventory["caches"][model])

    def build_caches(self) -> None:
        for model, spec in UNION_CACHE_SPECS.items():
            self.build_union_cache(model, spec)
        for model, spec in FULL_CACHE_SPECS.items():
            self.build_full_cache(model, spec)
        self.write_text(
            "caches/README.md",
            "# Hidden-state caches\n\n"
            "Each package index uses paths relative to the bundle root. Qwen3-VL, InternVL, and "
            "Qwen2.5-Omni contain the exact valid generated-set union. Qwen3.5 contains its complete "
            "mixed generated plus CH-SIMS cache and an exact generated-only index. Gemma contains the "
            "complete source cache except `.bak_wrong_proto`, with valid, excluded-success, and failure ledgers.\n",
            "packaging contract",
        )

    def build_labels(self) -> None:
        label_root = Path("/home/team/zhanghaonan/TAFFC/mprisk-v2/outputs/v2/misread")
        summaries: dict[str, Any] = {}
        for model, protocol in FORMAL_LABEL_MODELS.items():
            source = label_root / model / "judgments.jsonl"
            rows = read_jsonl(source)
            expected_ids = self.dataset_ids[protocol]
            require(len(rows) == len(expected_ids), f"{model}: formal label row count mismatch")
            require(id_set(rows, f"{model} labels") == expected_ids, f"{model}: formal label coverage mismatch")
            require({row.get("protocol") for row in rows} == {protocol}, f"{model}: formal label protocol mismatch")
            require({row.get("subject_model_key") for row in rows} == {model}, f"{model}: label subject mismatch")
            require({row.get("final_label") for row in rows} <= {"MISREAD", "NON_MISREAD"}, f"{model}: invalid final_label")
            source_types = {row["sample_id"]: row["sample_type"] for row in self.dataset_rows[protocol]}
            by_type_label = Counter((source_types[row["sample_id"]], row["final_label"]) for row in rows)
            summaries[model] = {
                "protocol": protocol,
                "rows": len(rows),
                "final_labels": dict(sorted(Counter(row["final_label"] for row in rows).items())),
                "by_sample_type": {
                    sample_type: {
                        label: by_type_label[(sample_type, label)]
                        for label in ("MISREAD", "NON_MISREAD")
                    }
                    for sample_type in ("Aligned", "Conflict")
                },
            }
            self.link(source, f"misread_labels/{model}/judgments.jsonl", True)
        require(set(summaries) == set(FORMAL_LABEL_MODELS), "formal label model set mismatch")
        self.write_json("misread_labels/counts.json", summaries, "derived from formal judgments.jsonl files")
        self.write_text(
            "misread_labels/README.md",
            "# Formal Misread labels\n\n"
            "Only the 15 formal `judgments.jsonl` files are included. Single-flash intermediates, "
            "`gemma4_12b_it`, and `phi4_multimodal` are deliberately excluded.\n",
            "packaging contract",
        )
        self.inventory["formal_misread_labels"] = summaries
        self.check("formal_misread_labels", {"models": len(summaries), "all_exact_coverage": True})

    def build_states(self) -> None:
        base = self.repo_root / "outputs/downstream/delivery_20260716/seed20260717/tme_ablation_v1"
        summaries: dict[str, Any] = {}
        for model, spec in STATE_SPECS.items():
            method_root = base / model / "tme_pa_dstrong_v2"
            state_source = method_root / "state_all_registered_splits/state_patterns.jsonl"
            require(sha256_file(state_source) == spec["sha256"], f"{model}: canonical state SHA mismatch")
            rows = read_jsonl(state_source)
            require(len(rows) == spec["rows"], f"{model}: state row count mismatch")
            require(id_set(rows, f"{model} states") == self.dataset_ids[str(spec["protocol"])], f"{model}: state-to-dataset coverage mismatch")
            require({row.get("model_key") for row in rows} == {model}, f"{model}: state model mismatch")
            require({str(row.get("protocol")).upper() for row in rows} == {spec["protocol"]}, f"{model}: state protocol mismatch")
            for row in rows:
                require(all(key in row for key in ("S_M1", "S_M2", "S_M12", "S_mean", "D", "R", "pattern")), f"{model}: state index field missing")

            run_complete = json.loads((method_root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
            require(run_complete.get("model_key") == model, f"{model}: RUN_COMPLETE model mismatch")
            require(run_complete.get("method") == "tme_pa_dstrong_v2", f"{model}: method mismatch")
            require(run_complete.get("misread_labels_used") is False, f"{model}: state run must not use Misread labels")
            training_config = Path(str(run_complete["training_config"])).resolve(strict=True)
            cache_union = Path(str(run_complete["cache_union"])).resolve(strict=True)
            checkpoint = Path(str(run_complete["best_checkpoint"])).resolve(strict=True)
            require(sha256_file(training_config) == run_complete["training_config_sha256"], f"{model}: training config SHA mismatch")
            require(sha256_file(cache_union) == run_complete["cache_union_sha256"], f"{model}: cache union SHA mismatch")
            require(sha256_file(checkpoint) == run_complete["best_checkpoint_sha256"], f"{model}: checkpoint SHA mismatch")

            for relative in STATE_EVIDENCE_FILES:
                source = method_root / relative
                self.link(source, f"states/{model}/method_evidence/{relative}", True)
            self.link(training_config, f"states/{model}/method_evidence/source_training_config.yaml", True)
            evidence = {
                "schema": "taffc_state_evidence_v1",
                "model": model,
                "method": "tme_pa_dstrong_v2",
                "state_patterns": f"states/{model}/method_evidence/state_all_registered_splits/state_patterns.jsonl",
                "state_patterns_sha256": spec["sha256"],
                "training_config": f"states/{model}/method_evidence/source_training_config.yaml",
                "training_config_sha256": run_complete["training_config_sha256"],
                "best_checkpoint": f"states/{model}/method_evidence/training/best_checkpoint.pt",
                "best_checkpoint_sha256": run_complete["best_checkpoint_sha256"],
                "cache_union": f"caches/{model}/index/union.package.json",
                "source_cache_union_sha256": run_complete["cache_union_sha256"],
                "misread_labels_used": False,
            }
            self.write_json(
                f"states/{model}/evidence_manifest.json",
                evidence,
                str(method_root / "RUN_COMPLETE.json"),
                "generated_state_evidence_index",
            )
            summaries[model] = {
                "protocol": spec["protocol"],
                "rows": len(rows),
                "sample_types": count_types(rows),
                "patterns": dict(sorted(Counter(row["pattern"] for row in rows).items())),
                "state_patterns_sha256": spec["sha256"],
            }
        scope_text = (
            "# State/TME scope\n\n"
            "State indices and state patterns are included only for `qwen3_vl_8b`, "
            "`internvl3_5_8b`, and `qwen2_5_omni_7b`.\n\n"
            "For `qwen3_5_4b` and `gemma4_12b`: **state indices/patterns are NOT COMPUTED and "
            "TME is NOT TRAINED**. Their delivery scope is cache plus formal Misread labels only.\n"
        )
        self.write_text("states/SCOPE.md", scope_text, "explicit user scope")
        self.inventory["states"] = summaries
        self.check("canonical_states", {"models": list(summaries), "all_exact_coverage": True})

    def write_control_payload(self) -> None:
        readme = (
            "# TAFFC complete bundle 2026-07-21\n\n"
            "This directory is the canonical, fail-closed delivery for the 3,810-row generated "
            "in-domain set, the CH-SIMS v2 cross-domain natural set, five hidden-state caches, "
            "15 formal Misread label sets, and state outputs for exactly three models.\n\n"
            "All dataset media and cache paths are relative to the bundle root. Large source files "
            "use hardlinks when source ownership permits it; cross-owner media are byte-copied. "
            "There are no symlinks. `SHA256SUMS` covers every file except itself "
            "and `file_provenance.tsv`, which are control manifests whose recursion is intentionally "
            "excluded. Run the builder with `--verify-only` for full checksum and coverage validation.\n\n"
            "Qwen3.5-4B and Gemma4-12B have cache and Misread labels only. Their state indices are "
            "NOT COMPUTED and their TME is NOT TRAINED.\n"
        )
        self.write_text("README.md", readme, "packaging contract")
        self.write_json("inventory.json", self.inventory, "derived verified inventory")
        report = {
            "schema": "taffc_complete_bundle_validation_v1",
            "status": "PASS",
            "checks": self.checks,
            "checksum_policy": {
                "algorithm": "SHA-256",
                "excluded_control_files": sorted(CONTROL_FILES),
                "verification_command": "python3 scripts/packaging/build_taffc_complete_bundle.py --verify-only --output <bundle>",
            },
        }
        self.write_json("validation_report.json", report, "builder validation results")
        lines = ["# Validation report", "", "Overall status: **PASS**", ""]
        for check in self.checks:
            detail = json.dumps(check["detail"], ensure_ascii=False, sort_keys=True)
            lines.append(f"- PASS `{check['name']}`: {detail}")
        lines.extend(["", "Qwen3.5-4B and Gemma4-12B state/TME: **NOT COMPUTED / NOT TRAINED**.", ""])
        self.write_text("validation_report.md", "\n".join(lines), "builder validation results")

    def finalize(self) -> None:
        if self.dry_run:
            self.check("dry_run_complete", {"planned_files": len(self.records), "source_contracts": "PASS"})
            print(json.dumps(self.inventory, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
            return
        actual = {
            path.relative_to(self.staging).as_posix()
            for path in self.staging.rglob("*")
            if path.is_file()
        }
        symlinks = [path for path in self.staging.rglob("*") if path.is_symlink()]
        require(not symlinks, f"delivery contains symlinks: {symlinks[:5]}")
        require(actual == set(self.records), f"pre-checksum file registry mismatch: actual={len(actual)} registered={len(self.records)}")

        rel_paths = sorted(self.records)
        hashes: dict[str, str] = {}
        sizes: dict[str, tuple[int, int]] = {}
        print(f"HASH payload: {len(rel_paths)} files", flush=True)
        for start in range(0, len(rel_paths), 1000):
            batch = rel_paths[start : start + 1000]
            with ThreadPoolExecutor(max_workers=min(self.workers, 4)) as executor:
                batch_hashes = list(executor.map(lambda rel: sha256_file(self._target(rel)), batch))
            for rel, digest in zip(batch, batch_hashes, strict=True):
                hashes[rel] = digest
                stat = self._target(rel).stat()
                sizes[rel] = (stat.st_size, stat.st_blocks * 512)
                expected = self.expected_sha.get(rel)
                require(expected in (None, digest), f"cache checksum mismatch for {rel}: {digest} != {expected}")
            completed = min(start + len(batch), len(rel_paths))
            if completed % 10000 == 0 or completed == len(rel_paths):
                print(f"HASH payload: {completed}/{len(rel_paths)}", flush=True)

        sha_lines = [f"{hashes[rel]}  {rel}" for rel in rel_paths]
        sha_path = self.staging / "SHA256SUMS"
        sha_path.write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
        provenance_path = self.staging / "file_provenance.tsv"
        with provenance_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["relative_path", "source_path", "size_bytes", "allocated_bytes", "sha256", "link_mode"])
            for rel in rel_paths:
                record = self.records[rel]
                writer.writerow([rel, record.source, sizes[rel][0], sizes[rel][1], hashes[rel], record.mode])

        all_files = {
            path.relative_to(self.staging).as_posix()
            for path in self.staging.rglob("*")
            if path.is_file()
        }
        require(all_files == set(self.records) | CONTROL_FILES, "final file set differs from registry plus controls")
        os.replace(self.staging, self.output)
        print(f"BUILD_COMPLETE {self.output}", flush=True)

    def run(self) -> None:
        self.prepare()
        self.build_datasets()
        self.build_caches()
        self.build_labels()
        self.build_states()
        self.write_control_payload()
        self.finalize()


def _verify_package_indexes(output: Path) -> dict[str, Any]:
    vt_rows = read_jsonl(output / "datasets/generated_3810/manifests/vt.jsonl")
    va_rows = read_jsonl(output / "datasets/generated_3810/manifests/va.jsonl")
    require(len(vt_rows) == 1876 and count_types(vt_rows) == {"Aligned": 1144, "Conflict": 732}, "packaged VT mismatch")
    require(len(va_rows) == 1934 and count_types(va_rows) == {"Aligned": 1093, "Conflict": 841}, "packaged VA mismatch")
    ids = {"VT": id_set(vt_rows, "packaged VT"), "VA": id_set(va_rows, "packaged VA")}
    for row in [*vt_rows, *va_rows]:
        for path in row["media_paths"].values():
            require(not Path(path).is_absolute(), f"absolute packaged media path: {path}")
            target = output / path
            require(target.is_file() and target.stat().st_size > 0, f"bad packaged media: {target}")
    ch_vt = read_jsonl(output / "datasets/ch_sims_v2_cross_domain/manifests/vt.jsonl")
    ch_va = read_jsonl(output / "datasets/ch_sims_v2_cross_domain/manifests/va.jsonl")
    require(len(ch_vt) == 2035 and len(ch_va) == 2190, "packaged CH-SIMS count mismatch")
    require(len({value for row in [*ch_vt, *ch_va] for value in row["media_paths"].values()}) == 2445, "packaged CH-SIMS media count mismatch")

    for model, protocol in FORMAL_LABEL_MODELS.items():
        rows = read_jsonl(output / f"misread_labels/{model}/judgments.jsonl")
        require(id_set(rows, f"packaged {model} labels") == ids[protocol], f"packaged {model} label coverage mismatch")
    for model, spec in STATE_SPECS.items():
        rows = read_jsonl(output / f"states/{model}/method_evidence/state_all_registered_splits/state_patterns.jsonl")
        require(id_set(rows, f"packaged {model} states") == ids[str(spec["protocol"])], f"packaged {model} state coverage mismatch")

    for model, spec in UNION_CACHE_SPECS.items():
        data = json.loads((output / f"caches/{model}/index/union.package.json").read_text(encoding="utf-8"))
        entries = data["entries"]
        require(len(entries) == spec["tasks"], f"packaged {model} union task mismatch")
        validate_task_matrix(entries, ids[str(spec["protocol"])], f"packaged {model} union")
        for row in entries:
            for value in (row["shard_path"], row["metadata"]["sidecar_path"]):
                require(not Path(value).is_absolute(), f"absolute packaged cache path: {value}")
                target = output / value
                require(target.is_file() and target.stat().st_size > 0, f"bad packaged cache payload: {target}")
    q35 = read_jsonl(output / "caches/qwen3_5_4b/index/generated_valid_manifest.package.jsonl")
    gemma = read_jsonl(output / "caches/gemma4_12b/index/generated_valid_manifest.package.jsonl")
    validate_task_matrix(q35, ids["VT"], "packaged Qwen3.5 generated cache")
    validate_task_matrix(gemma, ids["VA"], "packaged Gemma valid cache")
    require(len(read_jsonl(output / "caches/gemma4_12b/index/excluded_silent_successes.package.jsonl")) == 80, "packaged Gemma excluded success mismatch")
    require(len(read_jsonl(output / "caches/gemma4_12b/index/failed_tasks.jsonl")) == 40, "packaged Gemma failure mismatch")
    require(not (output / "states/qwen3_5_4b").exists(), "Qwen3.5 state directory must not exist")
    require(not (output / "states/gemma4_12b").exists(), "Gemma state directory must not exist")
    return {"generated": 3810, "ch_sims_protocol_rows": 4225, "formal_models": 15, "state_models": 3}


def verify_bundle(output: Path, workers: int) -> None:
    output = output.resolve(strict=True)
    require(output.is_dir(), f"not a bundle directory: {output}")
    symlinks = [path for path in output.rglob("*") if path.is_symlink()]
    require(not symlinks, f"bundle contains symlinks: {symlinks[:5]}")
    sha_path = output / "SHA256SUMS"
    provenance_path = output / "file_provenance.tsv"
    require(sha_path.is_file() and provenance_path.is_file(), "control manifests missing")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(sha_path.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, rel = line.partition("  ")
        require(separator == "  " and re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"bad SHA256SUMS line {line_number}")
        rel = normalize_rel(rel)
        require(rel not in expected, f"duplicate SHA path: {rel}")
        expected[rel] = digest
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    require(actual_files == set(expected) | CONTROL_FILES, "SHA256SUMS file coverage mismatch")

    rel_paths = sorted(expected)
    print(f"VERIFY SHA-256: {len(rel_paths)} files", flush=True)
    actual_hashes: dict[str, str] = {}
    for start in range(0, len(rel_paths), 1000):
        batch = rel_paths[start : start + 1000]
        with ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
            values = list(executor.map(lambda rel: sha256_file(output / rel), batch))
        for rel, digest in zip(batch, values, strict=True):
            require(digest == expected[rel], f"SHA mismatch: {rel}")
            actual_hashes[rel] = digest
        completed = min(start + len(batch), len(rel_paths))
        if completed % 10000 == 0 or completed == len(rel_paths):
            print(f"VERIFY SHA-256: {completed}/{len(rel_paths)}", flush=True)

    with provenance_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require({row["relative_path"] for row in rows} == set(expected), "provenance path coverage mismatch")
    for row in rows:
        rel = row["relative_path"]
        require(row["sha256"] == actual_hashes[rel], f"provenance SHA mismatch: {rel}")
        target = output / rel
        require(int(row["size_bytes"]) == target.stat().st_size, f"provenance size mismatch: {rel}")
        if row["link_mode"] == "hardlink":
            source = Path(row["source_path"])
            require(source.is_file(), f"hardlink source missing: {source}")
            source_stat = source.stat()
            target_stat = target.stat()
            require((source_stat.st_dev, source_stat.st_ino) == (target_stat.st_dev, target_stat.st_ino), f"hardlink provenance mismatch: {rel}")
    logical = _verify_package_indexes(output)
    print(json.dumps({"status": "PASS", "sha_files": len(expected), **logical}, sort_keys=True), flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/deliveries/taffc_complete_bundle_20260721"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="Audit every source and planned package path without writing")
    parser.add_argument("--verify-only", action="store_true", help="Verify an existing bundle and recompute all SHA-256 digests")
    parser.add_argument(
        "--resume-existing-staging",
        action="store_true",
        help="Validate and reuse the fixed staging directory after an interrupted build",
    )
    parser.add_argument(
        "--skip-media-stream-probe",
        action="store_true",
        help="Skip ffprobe only when a separate validated stream audit is supplied",
    )
    args = parser.parse_args(argv)
    require(args.workers > 0, "workers must be positive")
    require(not (args.dry_run and args.verify_only), "--dry-run and --verify-only are mutually exclusive")
    require(not (args.resume_existing_staging and args.verify_only), "cannot resume staging in verify-only mode")
    if not args.output.is_absolute():
        args.output = args.repo_root / args.output
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_only:
        verify_bundle(args.output, args.workers)
        return 0
    builder = BundleBuilder(
        repo_root=args.repo_root,
        output=args.output,
        workers=args.workers,
        dry_run=args.dry_run,
        probe_streams=not args.skip_media_stream_probe,
        resume_existing=args.resume_existing_staging,
    )
    builder.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BundleError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"FATAL: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
