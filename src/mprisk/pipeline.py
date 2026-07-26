"""V2 pipeline orchestrator.

For each (model, seed=20260717):
  1. Build state dataset + bundles from cache
  2. Train TME (Proxy Anchor, full layer trajectory)
  3. Export frozen embeddings (condition-level z + sample-level r)
  4. Compute spherical SDR scores
  5. Calibrate thresholds (v2 tunable quantile)
  6. Assign four state patterns
  7. Emit per-model summary JSON for downstream plotting

V2 deviations from mprisk mainline (clearly marked):
  - Single seed (no three-seed aggregation)
  - Threshold quantile tunable per protocol for "interpretable" pattern distribution
  - Sample filter optional (drop extreme-S outliers that wreck KDE)
  - Output root is outputs/v2/, never touches mprisk outputs
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
# IMPORTANT: this module manipulates sys.path at import time so that the
# ``mprisk`` and ``scripts`` packages resolve when ``mprisk_viz.pipeline``
# is imported from arbitrary cwd. Deferring this to a main() guard would
# break callers that import this module from outside the project root
# (e.g. tests invoked from /tmp). Keep these insertions here and make sure
# to invoke the pipeline from the project root, or pre-populate PYTHONPATH.
MPRISK_SRC = HERE.parent.parent / "src"
if str(MPRISK_SRC) not in sys.path:
    sys.path.insert(0, str(MPRISK_SRC))
SCRIPTS_DIR = HERE.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mprisk.data.protocol_views import normalize_protocol
from mprisk.data.state_bundle import build_state_bundles
from mprisk.data import state_dataset as _state_dataset_mod
from mprisk.data.state_dataset import build_state_dataset
from mprisk.representation.relation_dataset import build_relation_dataset
from mprisk.representation.training import (
    TrainingConfig,
    export_frozen_representations,
    train_trajectory_encoder,
)
from mprisk.state import spherical as _spherical_mod
from mprisk.state.spherical import compute_spherical_state
from mprisk.state.patterns import assign_state, StateThresholds
from mprisk.state.thresholds import calibrate_registered_aligned_thresholds
from mprisk.data.manifests import read_jsonl
from mprisk.utils.io import write_json, write_jsonl

def _v2_relaxed_shape_check(entries, sample_id):
    """V2 only requires layer_count and hidden_dim to match across M1/M2/M12."""
    present = {entry.condition for entry in entries}
    missing = [c for c in ("M1", "M2", "M12") if c not in present]
    if missing:
        raise ValueError(f"Missing cache entries for {sample_id}: {', '.join(missing)}")
    shapes = {(entry.layer_count, entry.hidden_dim) for entry in entries}
    if len(shapes) != 1:
        raise ValueError(
            f"Cache entry layer/hidden differs for {sample_id}: {shapes}"
        )


_V2_PATCHES_INSTALLED = False


def install_v2_pipeline_patches() -> None:
    """Apply v2-specific monkey-patches to the canonical ``mprisk`` library.

    Deferred to an explicit call (invoked by ``run_v2_for_model``) instead of
    firing at import time, so ``import mprisk_viz.pipeline`` is side effect
    free. Tests that import this module for smoke checks must not corrupt
    publication library state (BOOTSTRAP_REPLICATES, build_representation_model,
    batch loss, shape check).

    Idempotent: re-entry is a no-op so repeated calls from a long-lived
    process are safe.
    """
    global _V2_PATCHES_INSTALLED
    if _V2_PATCHES_INSTALLED:
        return
    _V2_PATCHES_INSTALLED = True

    # V2 speed-up: smaller bootstrap replicates (200 vs 2000).
    # SDR formulas stay exactly as paper (no normalization of D or delta);
    # cross-model normalization uses S/kappa, D/tau, R/delta downstream.
    _spherical_mod.BOOTSTRAP_REPLICATES = 200

    # Swap GRU-TME for LSTM-TME.
    from mprisk_viz.lstm_tme import install_v2_tme_factory
    install_v2_tme_factory()

    # Replace PA-only batch loss with SDR-aware hinge (Conflict push apart).
    from mprisk_viz.sdr_loss import install_sdr_aware_loss
    install_sdr_aware_loss(
        aux_weight=1.0,
        margin_D=0.60,   # push Conflict d(M1,M2) > Aligned by >=0.6 rad (~34 deg)
        margin_R=0.40,
        warmup_epochs=10,
    )

    # Relax cache entry shape check: v2 only requires layer_count + hidden_dim match.
    _state_dataset_mod._require_consistent_entry_shape = _v2_relaxed_shape_check


@dataclass(frozen=True)
class V2ModelSpec:
    model_key: str
    protocol: str
    cache_root: str
    prompt_set: str
    prompt_set_key: str
    main_manifest: str
    smoke_manifest: str
    train_config: str


@dataclass(frozen=True)
class V2PipelineResult:
    model_key: str
    protocol: str
    checkpoint_path: str
    embedding_manifest_path: str
    sdr_scores_path: str
    state_patterns_path: str
    thresholds_path: str
    summary_path: str
    sample_count: int


def _quantile_thresholds(
    sdr_rows: list[dict[str, Any]],
    *,
    kappa_quantile: float,
    tau_quantile: float,
) -> dict[str, Any]:
    """Calibrate kappa/tau on aligned_calibration rows using v2-chosen quantiles.

    Mprisk mainline uses 0.95 by default. V2 lowers it to get a more balanced
    four-pattern distribution (so Consensus does not dominate every plot).
    """
    aligned = [r for r in sdr_rows if r.get("calibration_split") == "aligned_calibration"]
    if not aligned:
        aligned = [r for r in sdr_rows if r.get("sample_type") == "Aligned"]
    if not aligned:
        raise ValueError("no aligned_calibration rows found for threshold calibration")
    s_values = np.array([float(r["S_mean"]) for r in aligned], dtype=np.float64)
    d_values = np.array([float(r["D"]) for r in aligned], dtype=np.float64)
    kappa = float(np.quantile(s_values, kappa_quantile))
    tau = float(np.quantile(d_values, tau_quantile))
    return {
        "schema": "mprisk_spherical_calibration_v2",
        "kappa": kappa,
        "tau": tau,
        "kappa_quantile": kappa_quantile,
        "tau_quantile": tau_quantile,
        "delta_policy": "per_sample_synchronous_prompt_bootstrap_1.96se",
        "calibration_split": "aligned_calibration",
        "n_calibration_rows": len(aligned),
    }


def run_v2_for_model(
    *,
    spec: V2ModelSpec,
    split_assignment: str | Path,
    output_root: str | Path,
    cache_root: str | Path,
    prompt_cache_manifest: str | Path | None = None,
    prompt_conditioned_cache_manifest: str | Path | None = None,
    unified_cache_manifest: str | Path | None = None,
    kappa_quantile: float = 0.80,
    tau_quantile: float = 0.50,
    max_epochs: int = 300,
    patience: int = 30,
    device: str = "cpu",
    resume_checkpoint: str | Path | None = None,
) -> V2PipelineResult:
    """Run the full v2 pipeline for one (model, seed) pair.

    Callers may pass any combination of the three cache manifests. If any of
    ``prompt_cache_manifest`` / ``prompt_conditioned_cache_manifest`` /
    ``unified_cache_manifest`` is missing, all three are auto-built from
    ``cache_root`` via :func:`mprisk_viz.setup_helper.setup_v2_cache_manifests`.
    """
    install_v2_pipeline_patches()

    protocol = normalize_protocol(spec.protocol)
    out = Path(output_root)
    train_dir = out / "checkpoints" / spec.model_key
    train_dir.mkdir(parents=True, exist_ok=True)

    if (
        prompt_cache_manifest is None
        or prompt_conditioned_cache_manifest is None
        or unified_cache_manifest is None
    ):
        if (
            prompt_cache_manifest is not None
            or prompt_conditioned_cache_manifest is not None
            or unified_cache_manifest is not None
        ):
            raise ValueError(
                "run_v2_for_model requires all three cache manifests to be set "
                "together: prompt_cache_manifest, prompt_conditioned_cache_manifest, "
                "and unified_cache_manifest. Pass either all three or none (auto-build)."
            )
        from mprisk_viz.setup_helper import setup_v2_cache_manifests
        print(f"[v2][{spec.model_key}] auto-building cache manifests...", flush=True)
        setup_out = setup_v2_cache_manifests(
            cache_root=cache_root,
            prompt_set_path=spec.prompt_set,
            model_key=spec.model_key,
            output_root=out / "cache_manifests" / spec.model_key,
        )
        prompt_cache_manifest = setup_out["prompt_cache_manifest"]
        prompt_conditioned_cache_manifest = setup_out["prompt_conditioned_cache_manifest"]
        unified_cache_manifest = setup_out["unified_full_cache_manifest"]

    with open(spec.train_config, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    allowed = {
        "repr_key", "model_key", "protocol", "classification_objective",
        "prompt_set_key", "prompt_set_artifact_sha256",
        "expected_prompt_count", "expected_prompt_ids",
        "hidden_dim", "condition_dim", "relation_dim", "dropout",
        "max_epochs", "batch_size", "lr", "weight_decay",
        "proxy_alpha", "proxy_margin", "patience", "min_delta", "seed",
    }
    config_dict = {k: v for k, v in raw_config.items() if k in allowed}
    if "expected_prompt_ids" in config_dict and isinstance(config_dict["expected_prompt_ids"], list):
        config_dict["expected_prompt_ids"] = tuple(config_dict["expected_prompt_ids"])
    config_dict["max_epochs"] = max_epochs
    config_dict["patience"] = patience
    config_dict["seed"] = int(config_dict.get("seed", 20260717))
    config = TrainingConfig(**config_dict)

    print(f"[v2][{spec.model_key}] building state dataset...", flush=True)
    state_dataset_result = build_state_dataset(
        manifest_paths=[Path(spec.main_manifest)],
        cache_root=Path(cache_root),
        model_key=spec.model_key,
        protocol=protocol,
        split_assignment_path=Path(split_assignment),
        output_dir=out / "state_data" / spec.model_key / protocol,
        manifest_path=Path(unified_cache_manifest),
    )

    print(f"[v2][{spec.model_key}] building state bundles...", flush=True)
    bundle_result = build_state_bundles(
        state_dataset_manifest_path=state_dataset_result.manifest_path,
        prompt_cache_manifest_path=Path(prompt_cache_manifest),
        prompt_conditioned_cache_manifest_path=Path(prompt_conditioned_cache_manifest),
        model_key=spec.model_key,
        protocol=protocol,
        prompt_set_path=Path(spec.prompt_set),
        prompt_set_key=spec.prompt_set_key,
        output_root=out / "state_bundles",
    )

    print(f"[v2][{spec.model_key}] building relation dataset...", flush=True)
    relation_dataset_result = build_relation_dataset(
        bundle_manifest_path=bundle_result.manifest_path,
        output_dir=out / "relation_data" / spec.model_key / protocol / spec.prompt_set_key,
        prompt_set_key=config.prompt_set_key,
        prompt_set_artifact_sha256=config.prompt_set_artifact_sha256,
        expected_prompt_count=config.expected_prompt_count,
        expected_prompt_ids=config.expected_prompt_ids,
    )

    print(f"[v2][{spec.model_key}] training TME (max_epochs={max_epochs}, patience={patience})...",
          flush=True)
    training_result = train_trajectory_encoder(
        dataset_path=relation_dataset_result.dataset_path,
        config=config,
        output_dir=train_dir,
        resume_checkpoint=resume_checkpoint,
        device=device,
    )
    print(f"[v2][{spec.model_key}] TME best epoch={training_result.metrics.get('best_epoch')} "
          f"best_val_bal_acc={training_result.metrics.get('best_val_balanced_accuracy_ac', 0.0):.4f} "
          f"stop={training_result.metrics.get('stop_reason')}", flush=True)

    print(f"[v2][{spec.model_key}] exporting frozen embeddings...", flush=True)
    embedding_dir = out / "embeddings" / spec.model_key / protocol / spec.prompt_set_key
    spherical_path = embedding_dir / "spherical_embedding_manifest.jsonl"
    if spherical_path.exists():
        from mprisk.representation.training import FrozenRepresentationExportResult
        print(f"[v2][{spec.model_key}] reusing existing spherical embeddings: {spherical_path}",
              flush=True)
        embedding_result = FrozenRepresentationExportResult(
            manifest_path=embedding_dir / "frozen_representations.jsonl",
            bundle_manifest_path=spherical_path,
            summary_path=embedding_dir / "frozen_representation_summary.json",
            count=_count_lines(spherical_path),
        )
    else:
        embedding_result = export_frozen_representations(
            dataset_path=relation_dataset_result.dataset_path,
            checkpoint_path=training_result.checkpoint_path,
            output_dir=embedding_dir,
        )

    print(f"[v2][{spec.model_key}] computing SDR scores...", flush=True)
    sdr_rows = []
    for row in read_jsonl(embedding_result.bundle_manifest_path):
        sdr = compute_spherical_state(row)
        sdr["representation_split"] = row.get("representation_split", "")
        sdr["master_split"] = row.get("master_split", "")
        sdr["calibration_split"] = row.get("calibration_split", "") or row.get("representation_split", "")
        sdr_rows.append(sdr)
    sdr_path = out / "state_data" / spec.model_key / protocol / "sdr_scores.jsonl"
    write_jsonl(sdr_path, sdr_rows)

    print(f"[v2][{spec.model_key}] calibrating thresholds "
          f"(kappa_q={kappa_quantile}, tau_q={tau_quantile})...", flush=True)
    thresholds = _quantile_thresholds(
        sdr_rows,
        kappa_quantile=kappa_quantile,
        tau_quantile=tau_quantile,
    )
    thresholds_path = out / "state_data" / spec.model_key / protocol / "thresholds.json"
    write_json(thresholds_path, thresholds)

    print(f"[v2][{spec.model_key}] assigning four state patterns...", flush=True)
    state_thresholds = StateThresholds(
        kappa=thresholds["kappa"],
        tau=thresholds["tau"],
        delta=None,
    )
    pattern_rows = []
    for row in sdr_rows:
        pattern = assign_state(
            row["S_mean"], row["D"], row["R"],
            state_thresholds,
            delta_i=row.get("delta_i"),
        ).value
        pattern_rows.append({**row, "pattern": pattern})
    patterns_path = out / "state_data" / spec.model_key / protocol / "state_patterns.jsonl"
    write_jsonl(patterns_path, pattern_rows)

    summary = {
        "model_key": spec.model_key,
        "protocol": protocol,
        "prompt_set_key": spec.prompt_set_key,
        "checkpoint_path": str(training_result.checkpoint_path),
        "embedding_manifest_path": str(embedding_result.bundle_manifest_path),
        "sdr_scores_path": str(sdr_path),
        "state_patterns_path": str(patterns_path),
        "thresholds_path": str(thresholds_path),
        "training_metrics": training_result.metrics,
        "thresholds": thresholds,
        "sample_count": len(pattern_rows),
        "pattern_counts": _count_patterns(pattern_rows),
        "sample_type_counts": _count_field(pattern_rows, "sample_type"),
    }
    summary_path = out / "state_data" / spec.model_key / protocol / "v2_summary.json"
    write_json(summary_path, summary)
    print(f"[v2][{spec.model_key}] done. summary={summary_path}", flush=True)

    return V2PipelineResult(
        model_key=spec.model_key,
        protocol=protocol,
        checkpoint_path=str(training_result.checkpoint_path),
        embedding_manifest_path=str(embedding_result.bundle_manifest_path),
        sdr_scores_path=str(sdr_path),
        state_patterns_path=str(patterns_path),
        thresholds_path=str(thresholds_path),
        summary_path=str(summary_path),
        sample_count=len(pattern_rows),
    )


def _count_patterns(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        p = str(r.get("pattern", ""))
        counts[p] = counts.get(p, 0) + 1
    return counts


def _count_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        v = str(r.get(field, ""))
        counts[v] = counts.get(v, 0) + 1
    return counts


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a file without leaking the file handle."""
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)
