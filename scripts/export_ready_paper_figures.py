from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

from mprisk.viz.bundle_figures import UMAP_CONFIG, export_bundle_figures
from mprisk.viz.figure_inputs import (
    _atomic_csv,
    _write_provenance,
    build_state_figure_inputs,
)

MODELS = (
    ("qwen3_vl_8b", "VT", "Qwen3-VL-8B"),
    ("internvl3_5_8b", "VT", "InternVL3.5-8B"),
    ("qwen2_5_omni_7b", "VA", "Qwen2.5-Omni-7B"),
)
REPRESENTATIONS = (
    ("single_point_binary_v1", "Single-Point", "penultimate_feature"),
    ("trajectory_mlp_binary_v1", "Trajectory MLP", "penultimate_feature"),
    ("tme_proxy_anchor_v1", "TME", "sample_relation_feature"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_run(root: Path, model: str) -> Path | None:
    candidates = sorted((root / "seed20260717" / model).glob("*"))
    return candidates[0] if len(candidates) == 1 else None


def _ready_paths(root: Path) -> tuple[list[Path], list[Path], list[Path]] | None:
    score_paths: list[Path] = []
    pattern_paths: list[Path] = []
    threshold_paths: list[Path] = []
    for model, _protocol, _label in MODELS:
        run = _find_run(root, model)
        if run is None:
            return None
        if any(
            not (run / repr_key / "RUN_COMPLETE.json").is_file()
            for repr_key, _label, _field in REPRESENTATIONS
        ):
            return None
        score = run / "tme_proxy_anchor_v1/official_test/sdr_scores.jsonl"
        pattern = run / "tme_proxy_anchor_v1/official_test/state_patterns.jsonl"
        threshold = run / "tme_proxy_anchor_v1/calibration/thresholds.json"
        if not all(path.is_file() for path in (score, pattern, threshold)):
            return None
        score_paths.append(score)
        pattern_paths.append(pattern)
        threshold_paths.append(threshold)
    return score_paths, pattern_paths, threshold_paths


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _build_fig08(root: Path, output_dir: Path, command: list[str]) -> Path:
    run = _find_run(root, "qwen3_vl_8b")
    if run is None:
        raise RuntimeError("qwen3 representative run is not uniquely discoverable")
    rows: list[dict[str, object]] = []
    sources: list[Path] = []
    for repr_key, label, feature_field in REPRESENTATIONS:
        if repr_key == "tme_proxy_anchor_v1":
            source = run / repr_key / "official_test/frozen_tme_representations.jsonl"
        else:
            source = run / repr_key / "official_test/frozen_baseline_representations.jsonl"
        if not source.is_file():
            raise RuntimeError(f"missing Fig. 8 representation artifact: {source}")
        sources.append(source)
        for item in _read_jsonl(source):
            if item.get("representation_split") != "official_test":
                continue
            feature = item.get(feature_field)
            if not isinstance(feature, list) or not feature:
                raise RuntimeError(f"missing finite representation feature in {source}")
            rows.append(
                {
                    "panel": "ac",
                    "representation": label,
                    "model": "qwen3_vl_8b",
                    "protocol": "VT",
                    "seed": "20260717",
                    "sample_id": str(item["sample_id"]),
                    "sample_type": str(item["sample_type"]),
                    "representation_split": "official_test",
                    "feature": json.dumps(feature, separators=(",", ":")),
                    "status": "Ready",
                }
            )
    output = output_dir / "fig08_representation_comparison.csv"
    fields = [
        "panel", "representation", "model", "protocol", "seed", "sample_id",
        "sample_type", "representation_split", "feature", "status",
    ]
    _atomic_csv(output, fields, rows)
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
                "model": "qwen3_vl_8b", "protocol": "VT", "seed": "20260717"
            },
            "umap": {
                "package": "umap-learn",
                "version": version("umap-learn"),
                **UMAP_CONFIG,
            },
        },
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resume artifact-backed Fig. 4-8 export.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--downstream-root",
        type=Path,
        default=Path("outputs/downstream/three_seed_v1"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/paper/figure_map.yaml"))
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    downstream = (
        args.downstream_root
        if args.downstream_root.is_absolute()
        else root / args.downstream_root
    )
    config = args.config if args.config.is_absolute() else root / args.config
    ready = _ready_paths(downstream)
    if ready is None:
        print("PENDING: seed20260717 state artifacts are not complete", flush=True)
        return 0
    score_paths, pattern_paths, threshold_paths = ready
    output_dir = root / "outputs/paper_exports/figures"
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    build_state_figure_inputs(
        sdr_scores_path=score_paths,
        state_patterns_path=pattern_paths,
        thresholds_path=threshold_paths,
        output_dir=output_dir,
        generated_command=command,
    )
    _build_fig08(downstream, output_dir, command)
    result = export_bundle_figures(config)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
