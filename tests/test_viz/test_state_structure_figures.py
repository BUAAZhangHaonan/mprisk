from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from mprisk.viz.bundle_figures import UMAP_CONFIG
from mprisk.viz.figure_inputs import PROVENANCE_SCHEMA, provenance_path
from mprisk.viz.state_structure_figures import FIGURES, PENDING, export_state_structure_figures

MODELS = ("qwen2_5_omni_7b", "qwen3_vl_8b", "internvl3_5_8b")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(
    root: Path,
    figure_key: str,
    rows: list[dict[str, object]],
    extra: dict[str, object],
) -> None:
    source = root / f"{figure_key}.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifact = root / f"{figure_key}.artifact"
    artifact.write_text(figure_key, encoding="utf-8")
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "figure_key": figure_key,
        "status": "Ready",
        "generated_command": ["pytest", figure_key],
        "sources": [{"path": str(artifact), "sha256": _sha(artifact)}],
        **extra,
    }
    provenance_path(source).write_text(json.dumps(provenance), encoding="utf-8")


def _state_thresholds() -> dict[str, dict[str, float]]:
    return {model: {"kappa": 1.0, "tau": 0.5} for model in MODELS}


def _prepare_sources(root: Path) -> None:
    fig04: list[dict[str, object]] = []
    for model_index, model in enumerate(MODELS):
        for sample_type, shift in (("Aligned", 0.0), ("Conflict", 0.2)):
            for sample_index in range(4):
                s = 0.1 + 0.01 * sample_index + shift
                d = 0.8 + 0.02 * model_index + shift
                r = (-0.6 if sample_type == "Aligned" else 0.7) + 0.01 * sample_index
                base = {
                    "sample_id": f"{model}-{sample_type}-{sample_index}",
                    "model": model,
                    "sample_type": sample_type,
                    "S": s,
                    "D": d,
                    "R": r,
                }
                for metric, value in (("S", s), ("D", d), ("abs_R", abs(r))):
                    fig04.append({**base, "metric": metric, "value": value})
    common = {
        "thresholds_by_model": _state_thresholds(),
        "representation_split": "official_test",
    }
    _write_source(
        root,
        "fig04_sdr_distributions",
        fig04,
        {
            **common,
            "sample_masks": {
                "S": "representation_split=official_test",
                "D": "representation_split=official_test and S<=kappa",
                "abs_R": "representation_split=official_test and S<=kappa and D>tau",
            },
        },
    )

    fig05: list[dict[str, object]] = []
    for model in MODELS:
        for sample_type in ("Aligned", "Conflict"):
            for pattern in ("Consensus", "Balanced", "Dominant", "Confusion"):
                fig05.append(
                    {
                        "model": model,
                        "sample_type": sample_type,
                        "pattern": pattern,
                        "count": 1,
                        "total": 4,
                        "proportion": 0.25,
                    }
                )
    _write_source(
        root,
        "fig05_four_state_stacks",
        fig05,
        {**common, "sample_masks": {"patterns": "representation_split=official_test"}},
    )

    fig06: list[dict[str, object]] = []
    for model in MODELS:
        for sample_type, r in (("Aligned", -0.4), ("Conflict", 0.7)):
            for sample_index in range(4):
                fig06.append(
                    {
                        "sample_id": f"{model}-{sample_type}-{sample_index}",
                        "model": model,
                        "sample_type": sample_type,
                        "S": 0.1,
                        "D": 0.8,
                        "R": r + 0.01 * sample_index,
                        "stable": "true",
                        "direction_emphasized": "true",
                        "lean": "V" if r > 0 else "T/A",
                    }
                )
    _write_source(
        root,
        "fig06_stable_d_signed_r",
        fig06,
        {
            **common,
            "sample_masks": {
                "points": "S<=kappa",
                "direction_emphasis": "S<=kappa and D>tau",
            },
        },
    )

    fig07 = [
        {
            "panel": "bias",
            "model": model,
            "sample_id": f"{model}-{sample_index}",
            "sample_type": "Conflict",
            "S": 0.1,
            "D": 0.8,
            "R": 0.7 - 0.05 * sample_index,
            "direction_emphasized": "true",
            "status": "Ready",
        }
        for model in MODELS
        for sample_index in range(4)
    ]
    _write_source(
        root,
        "fig07_misread_bias",
        fig07,
        {
            **common,
            "sample_masks": {
                "misread": PENDING,
                "bias": "representation_split=official_test and sample_type=Conflict and S<=kappa",
                "direction_emphasis": "D>tau",
            },
        },
    )

    fig08: list[dict[str, object]] = []
    for representation_index, representation in enumerate(
        ("Single-Point", "Trajectory MLP", "TME")
    ):
        for sample_index in range(18):
            conflict = sample_index >= 9
            sign = 1.0 if conflict else -1.0
            fig08.append(
                {
                    "panel": "ac",
                    "representation": representation,
                    "model": "qwen3_vl_8b",
                    "protocol": "VT",
                    "seed": "20260717",
                    "sample_id": f"sample-{sample_index}",
                    "sample_type": "Conflict" if conflict else "Aligned",
                    "representation_split": "official_test",
                    "feature": json.dumps(
                        [
                            sign * (1.0 + representation_index * 0.1),
                            sample_index * 0.03 + 0.2,
                            sign * 0.4 + sample_index * 0.01,
                        ]
                    ),
                    "status": "Ready",
                }
            )
    _write_source(
        root,
        "fig08_representation_comparison",
        fig08,
        {
            "sample_masks": {
                "ac": "qwen3_vl_8b/VT/seed20260717/representation_split=official_test",
                "misread": PENDING,
                "conflict_retention": "Pending Conflict-retention sensitivity artifacts",
            },
            "umap": {
                "package": "umap-learn",
                "version": importlib.metadata.version("umap-learn"),
                **UMAP_CONFIG,
            },
        },
    )


def test_state_structure_exports_separately_with_real_only_coordinates_and_pending_panels(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "original-inputs"
    input_root = tmp_path / "template-inputs"
    output_root = tmp_path / "template-figures"
    _prepare_sources(source_root)
    original_hashes = {path: _sha(path) for path in source_root.glob("fig*.csv*") if path.is_file()}

    result = export_state_structure_figures(
        source_root=source_root,
        input_root=input_root,
        output_root=output_root,
        generated_command=["pytest", "template state-structure"],
    )

    assert set(result["figures"]) == set(FIGURES)
    assert all(_sha(path) == digest for path, digest in original_hashes.items())
    for item in result["figures"].values():
        assert Path(item["pdf"]).read_bytes().startswith(b"%PDF-")
        assert Path(item["png"]).is_file()
        assert Path(item["input"]).parent == input_root
        images = subprocess.run(
            ["pdfimages", "-list", item["pdf"]],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[2:]
        assert not [line for line in images if line.strip()]
        sidecar = json.loads(Path(item["provenance"]).read_text())
        assert sidecar["state_structure"]["synthetic_data_used"] is False
        assert all(Path(source["path"]).parent == source_root for source in sidecar["sources"])

    fig08_rows = list(csv.DictReader((input_root / "fig08_representation_comparison.csv").open()))
    assert "feature" not in fig08_rows[0]
    assert {row["panel"] for row in fig08_rows} == {"ac_umap"}
    assert {row["sample_type"] for row in fig08_rows} == {"Aligned", "Conflict"}

    for key in ("fig05_four_state_stacks", "fig07_misread_bias", "fig08_representation_comparison"):
        pdf = Path(result["figures"][key]["pdf"])
        completed = subprocess.run(
            ["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True
        )
        assert PENDING in completed.stdout
