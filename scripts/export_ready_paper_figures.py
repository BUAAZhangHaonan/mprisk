from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from mprisk.viz.bundle_figures import UMAP_CONFIG, export_bundle_figures
from mprisk.viz.figure_inputs import (
    FIGURE_CSV_FIELDS,
    PENDING,
    _atomic_csv,
    _write_provenance,
    build_state_figure_inputs,
    write_pending_figure_inputs,
)

DELIVERY_SCHEMA = "mprisk_delivery_tme_run_complete_v1"
BASELINE_SCHEMA = "mprisk_downstream_run_complete_v1"
DELIVERY = "delivery_20260716"
SEED = 20260717
STATE_METHOD = "tme_pa_dstrong_v2"
DEFAULT_DOWNSTREAM_ROOT = Path(
    "outputs/downstream/delivery_20260716/seed20260717/tme_ablation_v1"
)
MODELS = (
    ("qwen3_vl_8b", "VT", "Qwen3-VL-8B"),
    ("internvl3_5_8b", "VT", "InternVL3.5-8B"),
    ("qwen2_5_omni_7b", "VA", "Qwen2.5-Omni-7B"),
)
BASELINE_REPRESENTATIONS = (
    ("single_point_binary_v1", "Single-Point", "penultimate_feature"),
    ("trajectory_mlp_binary_v1", "Trajectory MLP", "penultimate_feature"),
)
TME_REPRESENTATION = (STATE_METHOD, "TME", "sample_relation_feature")


@dataclass(frozen=True)
class DeliveryStateArtifacts:
    model_key: str
    method_root: Path
    scores: Path
    patterns: Path
    thresholds: Path
    frozen_tme: Path


@dataclass(frozen=True)
class FigureRepresentation:
    label: str
    feature_field: str
    source: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_marker(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"completion marker must be a JSON object: {path}")
    return payload


def _marker_file(
    marker: dict[str, Any],
    *,
    marker_path: Path,
    path_field: str,
    sha_field: str,
    expected_path: Path | None = None,
) -> Path:
    value = marker.get(path_field)
    digest = marker.get(sha_field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{marker_path} is missing {path_field}")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"{marker_path} is missing a valid {sha_field}")
    path = Path(value).resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise RuntimeError(
            f"{marker_path} {path_field} does not match the registered delivery path"
        )
    if not path.is_file():
        raise RuntimeError(f"completion marker artifact is missing: {path}")
    if _sha256(path) != digest:
        raise RuntimeError(f"completion marker artifact checksum drift: {path}")
    return path


def _delivery_state_artifacts(
    root: Path,
    model_key: str,
    *,
    method: str = STATE_METHOD,
) -> DeliveryStateArtifacts | None:
    method_root = root / model_key / method
    marker_path = method_root / "RUN_COMPLETE.json"
    if not marker_path.is_file():
        return None
    marker = _load_marker(marker_path)
    expected_identity = {
        "schema": DELIVERY_SCHEMA,
        "delivery": DELIVERY,
        "seed": SEED,
        "model_key": model_key,
        "method": method,
        "misread_labels_used": False,
    }
    for field, expected in expected_identity.items():
        if marker.get(field) != expected:
            raise RuntimeError(
                f"{marker_path} identity mismatch for {field}: "
                f"expected {expected!r}, found {marker.get(field)!r}"
            )

    _marker_file(
        marker,
        marker_path=marker_path,
        path_field="training_config",
        sha_field="training_config_sha256",
    )
    _marker_file(
        marker,
        marker_path=marker_path,
        path_field="cache_union",
        sha_field="cache_union_sha256",
    )
    expected_artifacts = {
        "best_checkpoint": (
            "best_checkpoint_sha256",
            method_root / "training/best_checkpoint.pt",
        ),
        "official_frozen": (
            "official_frozen_sha256",
            method_root / "official_test/frozen_tme_representations.jsonl",
        ),
        "official_sdr_scores": (
            "official_sdr_sha256",
            method_root / "official_test/sdr_scores.jsonl",
        ),
        "official_patterns": (
            "official_patterns_sha256",
            method_root / "official_test/state_patterns.jsonl",
        ),
        "geometry_metrics": (
            "geometry_metrics_sha256",
            method_root / "official_test/geometry_metrics.json",
        ),
    }
    checked = {
        field: _marker_file(
            marker,
            marker_path=marker_path,
            path_field=field,
            sha_field=sha_field,
            expected_path=expected,
        )
        for field, (sha_field, expected) in expected_artifacts.items()
    }
    thresholds = (method_root / "calibration/thresholds.json").resolve()
    if not thresholds.is_file():
        raise RuntimeError(f"completed state run is missing calibration thresholds: {thresholds}")
    return DeliveryStateArtifacts(
        model_key=model_key,
        method_root=method_root.resolve(),
        scores=checked["official_sdr_scores"],
        patterns=checked["official_patterns"],
        thresholds=thresholds,
        frozen_tme=checked["official_frozen"],
    )


def _ready_state_artifacts(
    root: Path,
    *,
    method: str = STATE_METHOD,
) -> dict[str, DeliveryStateArtifacts] | None:
    artifacts: dict[str, DeliveryStateArtifacts] = {}
    for model_key, _protocol, _label in MODELS:
        state = _delivery_state_artifacts(root, model_key, method=method)
        if state is None:
            return None
        artifacts[model_key] = state
    return artifacts


def _validate_baseline_source(
    source: Path,
    *,
    model_key: str,
    repr_key: str,
) -> Path | None:
    source = source.resolve()
    marker_path = source.parent.parent / "RUN_COMPLETE.json"
    if not marker_path.is_file():
        return None
    marker = _load_marker(marker_path)
    expected_identity = {
        "schema": BASELINE_SCHEMA,
        "seed": SEED,
        "model_key": model_key,
        "repr_key": repr_key,
    }
    for field, expected in expected_identity.items():
        if marker.get(field) != expected:
            raise RuntimeError(
                f"{marker_path} identity mismatch for {field}: "
                f"expected {expected!r}, found {marker.get(field)!r}"
            )
    return _marker_file(
        marker,
        marker_path=marker_path,
        path_field="official_manifest",
        sha_field="official_manifest_sha256",
        expected_path=source,
    )


def _ready_fig08_sources(
    root: Path,
    state_artifacts: dict[str, DeliveryStateArtifacts],
    *,
    single_point_path: Path | None = None,
    trajectory_mlp_path: Path | None = None,
) -> tuple[FigureRepresentation, ...] | None:
    model_key = "qwen3_vl_8b"
    overrides = {
        "single_point_binary_v1": single_point_path,
        "trajectory_mlp_binary_v1": trajectory_mlp_path,
    }
    sources: list[FigureRepresentation] = []
    for repr_key, label, feature_field in BASELINE_REPRESENTATIONS:
        default = (
            root
            / model_key
            / repr_key
            / "official_test/frozen_baseline_representations.jsonl"
        )
        source = overrides[repr_key] or default
        validated = _validate_baseline_source(
            source,
            model_key=model_key,
            repr_key=repr_key,
        )
        if validated is None:
            return None
        sources.append(
            FigureRepresentation(
                label=label,
                feature_field=feature_field,
                source=validated,
            )
        )
    sources.append(
        FigureRepresentation(
            label=TME_REPRESENTATION[1],
            feature_field=TME_REPRESENTATION[2],
            source=state_artifacts[model_key].frozen_tme,
        )
    )
    return tuple(sources)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _finite_feature(item: dict[str, object], field: str, source: Path) -> list[float]:
    feature = item.get(field)
    if not isinstance(feature, list) or not feature:
        raise RuntimeError(f"missing representation feature {field} in {source}")
    values = [float(value) for value in feature]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"non-finite representation feature {field} in {source}")
    return values


def _build_fig08(
    representations: tuple[FigureRepresentation, ...],
    output_dir: Path,
    command: list[str],
) -> Path:
    rows_by_representation: dict[str, list[dict[str, object]]] = {}
    identities: dict[str, dict[str, str]] = {}
    for representation in representations:
        rows: list[dict[str, object]] = []
        seen: set[str] = set()
        current: dict[str, str] = {}
        for item in _read_jsonl(representation.source):
            if item.get("representation_split") != "official_test":
                continue
            sample_id = str(item["sample_id"])
            if sample_id in seen:
                raise RuntimeError(
                    f"duplicate official_test sample {sample_id} in {representation.source}"
                )
            seen.add(sample_id)
            sample_type = str(item["sample_type"])
            if sample_type not in {"Aligned", "Conflict"}:
                raise RuntimeError(
                    f"unexpected sample_type {sample_type!r} in {representation.source}"
                )
            current[sample_id] = sample_type
            feature = _finite_feature(
                item,
                representation.feature_field,
                representation.source,
            )
            rows.append(
                {
                    "panel": "ac",
                    "representation": representation.label,
                    "model": "qwen3_vl_8b",
                    "protocol": "VT",
                    "seed": str(SEED),
                    "sample_id": sample_id,
                    "sample_type": sample_type,
                    "representation_split": "official_test",
                    "feature": json.dumps(feature, separators=(",", ":")),
                    "status": "Ready",
                }
            )
        if not rows:
            raise RuntimeError(
                f"Fig. 8 source contains no official_test representations: "
                f"{representation.source}"
            )
        rows_by_representation[representation.label] = rows
        identities[representation.label] = current
    reference = identities["Single-Point"]
    if any(identity != reference for identity in identities.values()):
        raise RuntimeError("Fig. 8 representations do not share the exact official_test set")

    rows = [
        row
        for representation in ("Single-Point", "Trajectory MLP", "TME")
        for row in rows_by_representation[representation]
    ]
    sources = [representation.source for representation in representations]
    output = output_dir / "fig08_representation_comparison.csv"
    _atomic_csv(
        output,
        FIGURE_CSV_FIELDS["fig08_representation_comparison"],
        rows,
    )
    _write_provenance(
        output,
        figure_key="fig08_representation_comparison",
        generated_command=command,
        sources=sources,
        sample_masks={
            "ac": "qwen3_vl_8b/VT/seed20260717/representation_split=official_test",
            "misread": "Pending Misread annotations",
            "conflict_retention": "Pending Conflict-retention sensitivity artifacts",
        },
        thresholds=None,
        source_sample_count=len(rows),
        included_sample_count=len(rows),
        extra={
            "representation_split": "official_test",
            "representative_backbone": {
                "model": "qwen3_vl_8b",
                "protocol": "VT",
                "seed": str(SEED),
            },
            "misread_status": PENDING,
            "umap": {
                "package": "umap-learn",
                "version": version("umap-learn"),
                **UMAP_CONFIG,
            },
        },
    )
    return output


def _write_pending_fig08(output_dir: Path, command: list[str]) -> Path:
    output = output_dir / "fig08_representation_comparison.csv"
    _atomic_csv(
        output,
        FIGURE_CSV_FIELDS["fig08_representation_comparison"],
        [],
    )
    _write_provenance(
        output,
        figure_key="fig08_representation_comparison",
        generated_command=command,
        sources=[],
        sample_masks={
            "ac": "Pending complete frozen Single-Point/Trajectory MLP/TME artifacts",
            "misread": "Pending Misread annotations",
            "conflict_retention": "Pending Conflict-retention sensitivity artifacts",
        },
        thresholds=None,
        source_sample_count=0,
        included_sample_count=0,
        status=PENDING,
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export artifact-backed Fig. 4-8 inputs.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--downstream-root",
        type=Path,
        default=DEFAULT_DOWNSTREAM_ROOT,
    )
    parser.add_argument("--state-method", choices=(STATE_METHOD,), default=STATE_METHOD)
    parser.add_argument(
        "--fig08-single-point",
        type=Path,
        help="Completed qwen3 official_test Single-Point representation manifest.",
    )
    parser.add_argument(
        "--fig08-trajectory-mlp",
        type=Path,
        help="Completed qwen3 official_test Trajectory MLP representation manifest.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/paper/figure_map.yaml"))
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    downstream = (
        args.downstream_root.resolve()
        if args.downstream_root.is_absolute()
        else (root / args.downstream_root).resolve()
    )
    config = args.config.resolve() if args.config.is_absolute() else root / args.config
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]

    state_artifacts = _ready_state_artifacts(downstream, method=args.state_method)
    if state_artifacts is None:
        print(
            f"PENDING: all three {args.state_method} delivery state artifacts are not complete",
            flush=True,
        )
        return 0

    ordered = [state_artifacts[model_key] for model_key, _protocol, _label in MODELS]
    output_dir = root / "outputs/paper_exports/figures"
    build_state_figure_inputs(
        sdr_scores_path=[item.scores for item in ordered],
        state_patterns_path=[item.patterns for item in ordered],
        thresholds_path=[item.thresholds for item in ordered],
        output_dir=output_dir,
        generated_command=command,
    )
    fig08_sources = _ready_fig08_sources(
        downstream,
        state_artifacts,
        single_point_path=args.fig08_single_point,
        trajectory_mlp_path=args.fig08_trajectory_mlp,
    )
    if fig08_sources is None:
        _write_pending_fig08(output_dir, command)
        fig08_status = PENDING
    else:
        _build_fig08(fig08_sources, output_dir, command)
        fig08_status = "Ready"
    write_pending_figure_inputs(config, generated_command=command)
    result = export_bundle_figures(config)
    result["delivery_export"] = {
        "state_method": args.state_method,
        "fig04_fig07": "Ready",
        "fig08": fig08_status,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
