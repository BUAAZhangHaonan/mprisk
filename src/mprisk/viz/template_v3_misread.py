"""Template-v3 figures and tables with formal Misread evidence only.

The v3 export is additive.  It never rewrites the registered figures, template-v2,
or the original paper tables.  Missing/in-progress formal roots produce explicit
Pending panels; a root claiming usable evidence is validated fail-closed.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from mprisk.viz.bundle_figures import MODEL_SPECS, _load_figure_input  # noqa: E402
from mprisk.viz.formal_misread import (  # noqa: E402
    BUDGET_FIELDS,
    FORMAL_BUDGETS,
    FORMAL_METHODS,
    FORMAL_MODELS,
    PROBE_FIELDS,
    FormalRoot,
    canonical_label_rows,
    canonical_metric_rows,
    load_formal_root,
    sha256,
)
from mprisk.viz.template_v2 import (  # noqa: E402
    GRID,
    STATE_COLORS,
    _pending_panel,
    _project_fig08_rows,
    _set_style,
    _with_normalized_split,
)

SCHEMA = "mprisk_template_v3_misread_export_v1"
PENDING = "Pending formal Misread results"
FIGURE_KEYS = (
    "fig05_state_patterns_template_v3_misread",
    "fig07_misread_associations_template_v3_misread",
    "fig08_representation_quality_template_v3_misread",
)


def export_template_v3_misread(
    *,
    source_root: str | Path = "outputs/paper_exports/figures",
    labels_root: str | Path | None = None,
    probes_root: str | Path | None = None,
    budgets_root: str | Path | None = None,
    input_root: str | Path = "outputs/paper_exports/figures/template_v3_misread",
    output_root: str | Path = "paper/figures/generated/template_v3_misread",
    table_input_root: str | Path = "outputs/paper_exports/tables/template_v3_misread",
    table_output_root: str | Path = "paper/tables/generated/template_v3_misread",
    generated_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    source_dir = Path(source_root)
    input_dir = Path(input_root)
    output_dir = Path(output_root)
    table_input_dir = Path(table_input_root)
    table_output_dir = Path(table_output_root)
    for path in (input_dir, output_dir, table_input_dir, table_output_dir):
        path.mkdir(parents=True, exist_ok=True)
    if source_dir.resolve() == input_dir.resolve():
        raise ValueError("template-v3 inputs must not overwrite registered source inputs")

    command = list(generated_command or [sys.executable, "export_template_v3_misread"])
    labels = load_formal_root(labels_root, kind="labels")
    probes = load_formal_root(probes_root, kind="probes")
    budgets = load_formal_root(budgets_root, kind="budgets")
    label_rows = canonical_label_rows(labels) if labels else None
    probe_rows = (
        canonical_metric_rows(probes, role="probe_metrics", fields=PROBE_FIELDS) if probes else None
    )
    budget_rows = (
        canonical_metric_rows(budgets, role="budget_metrics", fields=BUDGET_FIELDS)
        if budgets
        else None
    )
    _validate_cross_root_links(labels, probes, budgets, probe_rows, budget_rows)

    fig05_source = _ready_source(source_dir, "fig05_four_state_stacks")
    fig07_source = _ready_source(source_dir, "fig07_misread_bias")
    fig08_source = _ready_source(source_dir, "fig08_representation_comparison")
    state_rows = _registered_jsonl(fig05_source[1], "state_patterns.jsonl")
    sdr_rows = _registered_jsonl(fig07_source[1], "sdr_scores.jsonl")
    _validate_registered_state_rows(state_rows, require_pattern=True)
    _validate_registered_state_rows(sdr_rows, require_pattern=False)

    excluded = _exclusion_counts(label_rows)
    fig05_rows = _fig05_rows(fig05_source[0], state_rows, label_rows)
    fig07_rows = _fig07_rows(fig07_source[0], fig07_source[1], sdr_rows, state_rows, label_rows)
    fig08_rows = _fig08_rows(fig08_source[0], fig08_source[1], budget_rows, budgets)
    source_records = _source_records(source_dir, labels, probes, budgets)

    figures: dict[str, Any] = {}
    for key, rows, renderer, readiness in (
        (
            FIGURE_KEYS[0],
            fig05_rows,
            lambda: _render_fig05(fig05_rows, labels is not None, excluded),
            {"misread": labels is not None},
        ),
        (
            FIGURE_KEYS[1],
            fig07_rows,
            lambda: _render_fig07(fig07_rows, fig07_source[1], labels is not None, excluded),
            {"misread": labels is not None, "bias": True},
        ),
        (
            FIGURE_KEYS[2],
            fig08_rows,
            lambda: _render_fig08(fig08_rows, budgets is not None),
            {"ca_umap": True, "budget": budgets is not None},
        ),
    ):
        csv_path = input_dir / f"{key}.csv"
        _write_csv(csv_path, rows)
        provenance = _write_provenance(
            csv_path,
            key=key,
            command=command,
            sources=source_records,
            readiness=readiness,
            excluded=excluded,
        )
        figure = renderer()
        pdf_path = output_dir / f"{key}.pdf"
        png_path = output_dir / f"{key}.png"
        figure.savefig(pdf_path, format="pdf", facecolor="white")
        figure.savefig(png_path, format="png", dpi=100, facecolor="white")
        plt.close(figure)
        if not pdf_path.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError(f"failed to produce an openable PDF: {pdf_path}")
        figures[key] = {
            "input": str(csv_path),
            "input_sha256": sha256(csv_path),
            "provenance": str(provenance),
            "provenance_sha256": sha256(provenance),
            "pdf": str(pdf_path),
            "pdf_sha256": sha256(pdf_path),
            "png": str(png_path),
            "png_sha256": sha256(png_path),
            "readiness": readiness,
        }

    tables = _export_tables(
        labels=labels,
        probes=probes,
        label_rows=label_rows,
        probe_rows=probe_rows,
        stable_bias_rows=fig07_rows,
        input_root=table_input_dir,
        output_root=table_output_dir,
        command=command,
        sources=source_records,
        excluded=excluded,
    )
    manifest = {
        "schema": SCHEMA,
        "source_root": str(source_dir),
        "input_root": str(input_dir),
        "output_root": str(output_dir),
        "formal_roots": {
            "labels": _root_record(labels),
            "probes": _root_record(probes),
            "budgets": _root_record(budgets),
        },
        "excluded_labels": excluded,
        "figures": figures,
        "tables": tables,
    }
    manifest_path = input_dir / "template_v3_misread_export.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _ready_source(root: Path, key: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status, rows, provenance = _load_figure_input(key, root / f"{key}.csv")
    if status != "Ready" or not rows:
        raise ValueError(f"template-v3 requires registered Ready source: {key}")
    return rows, provenance


def _registered_jsonl(provenance: dict[str, Any], basename: str) -> list[dict[str, Any]]:
    matches = [
        item for item in provenance.get("sources", []) if Path(item["path"]).name == basename
    ]
    if not matches:
        raise ValueError(f"registered provenance lacks {basename}")
    rows: list[dict[str, Any]] = []
    for source in matches:
        path = Path(source["path"])
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError(f"registered state artifact checksum mismatch: {path}")
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _validate_registered_state_rows(rows: list[dict[str, Any]], *, require_pattern: bool) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        model = row.get("model_key")
        key = (str(model), str(row.get("sample_id")))
        if (
            model not in FORMAL_MODELS
            or row.get("representation_split") != "official_test"
            or row.get("sample_type") not in {"Aligned", "Conflict"}
            or key in seen
        ):
            raise ValueError("registered state rows violate official-test identity")
        if require_pattern and row.get("pattern") not in {
            "Confusion",
            "Consensus",
            "Balanced",
            "Dominant",
        }:
            raise ValueError("registered state row has an invalid State Pattern")
        seen.add(key)


def _label_index(
    rows: list[dict[str, Any]] | None, *, eligibility: str
) -> dict[tuple[str, str], str]:
    if rows is None:
        return {}
    return {(row["model"], row["sample_id"]): row["label"] for row in rows if row[eligibility]}


def _fig05_rows(
    aggregate_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    output = [
        {
            "panel": "state_patterns",
            "model": row["model"],
            "sample_type": row["sample_type"],
            "pattern": row["pattern"],
            "label": "",
            "count": int(row["count"]),
            "total": int(row["total"]),
            "proportion": float(row["proportion"]),
        }
        for row in aggregate_rows
    ]
    labels = _label_index(label_rows, eligibility="label_eligible")
    if not labels:
        return output
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for row in state_rows:
        if row["sample_type"] != "Conflict":
            continue
        key = (row["model_key"], row["sample_id"])
        if key not in labels:
            continue
        pattern = str(row["pattern"])
        grouped[(pattern, labels[key])] += 1
        totals[pattern] += 1
    for pattern in ("Confusion", "Consensus", "Balanced", "Dominant"):
        if totals[pattern] == 0:
            continue
        for label in ("NON_MISREAD", "MISREAD"):
            count = grouped[(pattern, label)]
            output.append(
                {
                    "panel": "misread_by_pattern",
                    "model": "pooled",
                    "sample_type": "Conflict",
                    "pattern": pattern,
                    "label": label,
                    "count": count,
                    "total": totals[pattern],
                    "proportion": count / totals[pattern],
                }
            )
    return output


def _binned_rows(rows: list[dict[str, Any]], *, panel: str, value_key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in FORMAL_MODELS:
        selected = sorted(
            (row for row in rows if row["model"] == model), key=lambda row: row[value_key]
        )
        if not selected:
            continue
        for bin_index, indexes in enumerate(np.array_split(np.arange(len(selected)), 5), start=1):
            group = [selected[int(index)] for index in indexes]
            misread = sum(row["label"] == "MISREAD" for row in group)
            output.append(
                {
                    "panel": panel,
                    "model": model,
                    "sample_id": "",
                    "sample_type": "Conflict",
                    "S": "",
                    "D": "",
                    "R": "",
                    "S_over_kappa": "",
                    "D_over_tau": "",
                    "pattern": "",
                    "label": "",
                    "shape": "",
                    "bin_index": bin_index,
                    "x": float(np.median([row[value_key] for row in group])),
                    "n": len(group),
                    "misread_count": misread,
                    "misread_rate": misread / len(group),
                    "status": "Ready",
                }
            )
    return output


def _fig07_rows(
    stable_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    sdr_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized_stable = _with_normalized_split(stable_rows, provenance)
    labels = _label_index(label_rows, eligibility="label_eligible")
    patterns = {(row["model_key"], row["sample_id"]): row["pattern"] for row in state_rows}
    thresholds = provenance["thresholds_by_model"]
    joined: list[dict[str, Any]] = []
    for row in sdr_rows:
        if row["sample_type"] != "Conflict":
            continue
        key = (row["model_key"], row["sample_id"])
        if key not in labels:
            continue
        model = row["model_key"]
        joined.append(
            {
                "model": model,
                "sample_id": row["sample_id"],
                "label": labels[key],
                "pattern": patterns[key],
                "S_over_kappa": float(row["S_mean"]) / float(thresholds[model]["kappa"]),
                "D_over_tau": float(row["D"]) / float(thresholds[model]["tau"]),
            }
        )
    output: list[dict[str, Any]] = []
    if labels:
        output.extend(_binned_rows(joined, panel="misread_by_S", value_key="S_over_kappa"))
        output.extend(_binned_rows(joined, panel="misread_by_D", value_key="D_over_tau"))
        for model in FORMAL_MODELS:
            for pattern in ("Confusion", "Consensus", "Balanced", "Dominant"):
                group = [
                    row for row in joined if row["model"] == model and row["pattern"] == pattern
                ]
                if not group:
                    continue
                misread = sum(row["label"] == "MISREAD" for row in group)
                output.append(
                    {
                        "panel": "misread_by_pattern",
                        "model": model,
                        "sample_id": "",
                        "sample_type": "Conflict",
                        "S": "",
                        "D": "",
                        "R": "",
                        "S_over_kappa": "",
                        "D_over_tau": "",
                        "pattern": pattern,
                        "label": "",
                        "shape": "",
                        "bin_index": "",
                        "x": "",
                        "n": len(group),
                        "misread_count": misread,
                        "misread_rate": misread / len(group),
                        "status": "Ready",
                    }
                )
    for row in normalized_stable:
        key = (row["model"], row["sample_id"])
        if labels and key not in labels:
            continue
        label = labels.get(key, "")
        output.append(
            {
                "panel": "bias",
                "model": row["model"],
                "sample_id": row["sample_id"],
                "sample_type": "Conflict",
                "S": float(row["S"]),
                "D": float(row["D"]),
                "R": float(row["R"]),
                "S_over_kappa": "",
                "D_over_tau": float(row["D_over_tau"]),
                "pattern": "",
                "label": label,
                "shape": "X" if label == "MISREAD" else "o",
                "bin_index": "",
                "x": "",
                "n": "",
                "misread_count": "",
                "misread_rate": "",
                "status": "Ready",
            }
        )
    return output


def _fig08_rows(
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    budget_rows: list[dict[str, Any]] | None,
    budget_root: FormalRoot | None,
) -> list[dict[str, Any]]:
    projected = _project_fig08_rows(rows, provenance)
    output = [
        {
            **row,
            "budget_pct": "",
            "accuracy": "",
            "macro_f1": "",
            "auprc": "",
            "n_conflict_supervision": "",
            "n_aligned_supervision": "",
            "n_train": "",
            "n_val": "",
            "n_test": "",
        }
        for row in projected
    ]
    if budget_rows is None or budget_root is None:
        return output
    representative = str(budget_root.marker.get("representative_model") or "")
    if representative != "qwen3_vl_8b":
        raise ValueError("Fig. 8 formal budget root must bind qwen3_vl_8b")
    seeds = tuple(int(seed) for seed in budget_root.marker.get("seeds") or ())
    if not seeds:
        raise ValueError("formal budget root requires registered seeds")
    selected = [row for row in budget_rows if row["model"] == representative]
    expected = {
        (method, budget, seed)
        for method in FORMAL_METHODS
        for budget in FORMAL_BUDGETS
        for seed in seeds
    }
    observed = {(row["method"], row["budget_pct"], row["seed"]) for row in selected}
    if observed != expected:
        raise ValueError("formal Fig. 8 budget rows do not cover the registered Cartesian grid")
    for row in selected:
        output.append(
            {
                "panel": "budget",
                "representation": row["method"],
                "model": row["model"],
                "protocol": row["protocol"],
                "seed": row["seed"],
                "sample_id": "",
                "sample_type": "Conflict",
                "representation_split": "official_test",
                "umap_x": "",
                "umap_y": "",
                "silhouette_original": "",
                "knn5_purity_original": "",
                "status": "Ready",
                "budget_pct": row["budget_pct"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "auprc": row["auprc"],
                "n_conflict_supervision": row["n_conflict_supervision"],
                "n_aligned_supervision": row["n_aligned_supervision"],
                "n_train": row["n_train"],
                "n_val": row["n_val"],
                "n_test": row["n_test"],
            }
        )
    return output


def _render_fig05(rows: list[dict[str, Any]], ready: bool, excluded: dict[str, int]) -> Any:
    _set_style()
    figure = plt.figure(figsize=(14.48, 10.86), dpi=100)
    figure.suptitle(
        "Figure 5. State Pattern distributions and Misread prevalence",
        fontsize=24,
        fontweight="bold",
        y=0.975,
    )
    grid = figure.add_gridspec(1, 2, left=0.08, right=0.97, bottom=0.18, top=0.82, wspace=0.28)
    axis = figure.add_subplot(grid[0, 0])
    patterns = ("Consensus", "Balanced", "Dominant", "Confusion")
    base = [row for row in rows if row["panel"] == "state_patterns"]
    totals = {
        kind: sum(int(row["count"]) for row in base if row["sample_type"] == kind)
        for kind in ("Aligned", "Conflict")
    }
    bottoms = np.zeros(2)
    for pattern in patterns:
        counts = np.asarray(
            [
                sum(
                    int(row["count"])
                    for row in base
                    if row["sample_type"] == kind and row["pattern"] == pattern
                )
                for kind in ("Aligned", "Conflict")
            ]
        )
        values = counts / np.asarray([totals["Aligned"], totals["Conflict"]])
        axis.bar(
            (0, 1),
            values,
            bottom=bottoms,
            color=STATE_COLORS[pattern],
            edgecolor="#333333",
            label=pattern,
        )
        bottoms += values
    axis.set_xticks(
        (0, 1), (f"Aligned\n(n={totals['Aligned']:,})", f"Conflict\n(n={totals['Conflict']:,})")
    )
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_ylabel("Model-sample observations (%)")
    axis.set_title("State Pattern composition in\nAligned and Conflict samples", fontweight="bold")
    axis.grid(axis="y", color=GRID, linestyle=(0, (2, 3)))
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    axis.text(-0.16, 1.10, "(a)", transform=axis.transAxes, fontsize=22, fontweight="bold")
    right = figure.add_subplot(grid[0, 1])
    if not ready:
        _pending_panel(
            right,
            "Misread composition within each\nState Pattern (Conflict only)",
            xlabel="State Pattern",
            ylabel="Conflict samples (%)",
            xticks=(0, 1, 2, 3),
            xticklabels=("Confusion", "Consensus", "Balanced", "Dominant"),
        )
    else:
        right_rows = [row for row in rows if row["panel"] == "misread_by_pattern"]
        x = np.arange(4)
        bottoms = np.zeros(4)
        for label, color, display in (
            ("NON_MISREAD", "#7185A4", "Non-misread"),
            ("MISREAD", "#C95D61", "Misread"),
        ):
            values = np.asarray(
                [
                    sum(
                        float(row["proportion"])
                        for row in right_rows
                        if row["pattern"] == pattern and row["label"] == label
                    )
                    for pattern in ("Confusion", "Consensus", "Balanced", "Dominant")
                ]
            )
            right.bar(x, values, bottom=bottoms, color=color, edgecolor="#333333", label=display)
            bottoms += values
        right.set_xticks(x, ("Confusion", "Consensus", "Balanced", "Dominant"))
        right.set_ylim(0, 1)
        right.yaxis.set_major_formatter(PercentFormatter(1))
        right.set_ylabel("Conflict samples (%)")
        right.set_title(
            "Misread composition within each\nState Pattern (Conflict only)", fontweight="bold"
        )
        right.grid(axis="y", color=GRID, linestyle=(0, (2, 3)))
        right.legend(frameon=False)
    right.text(-0.16, 1.10, "(b)", transform=right.transAxes, fontsize=22, fontweight="bold")
    if ready:
        exclusion_note = (
            "Formal eligible subset; excluded "
            f"unresolved={excluded['unresolved']}, blocked={excluded['blocked']}."
        )
        figure.text(
            0.5,
            0.035,
            exclusion_note,
            ha="center",
            fontsize=12,
            fontstyle="italic",
        )
    return figure


def _render_fig07(
    rows: list[dict[str, Any]], provenance: dict[str, Any], ready: bool, excluded: dict[str, int]
) -> Any:
    _set_style()
    figure = plt.figure(figsize=(14.48, 10.86), dpi=100)
    figure.suptitle(
        "Figure 7. State-Misread associations and model-specific modality bias",
        fontsize=23,
        fontweight="bold",
        y=0.978,
    )
    grid = figure.add_gridspec(
        2, 3, left=0.05, right=0.93, bottom=0.17, top=0.88, wspace=0.30, hspace=0.42
    )
    colors = dict(zip(FORMAL_MODELS, ("#C95D61", "#2F5597", "#E69F00"), strict=True))
    for column, (panel, title, xlabel) in enumerate(
        (
            ("misread_by_S", r"State Dispersion $S/\kappa$ vs Misread", r"$S/\kappa$"),
            (
                "misread_by_D",
                r"Modality Split $\mathcal{D}/\tau$ vs Misread",
                r"$\mathcal{D}/\tau$",
            ),
            ("misread_by_pattern", "State Pattern vs Misread", "State Pattern"),
        )
    ):
        axis = figure.add_subplot(grid[0, column])
        if not ready:
            _pending_panel(
                axis,
                title,
                xlabel=xlabel,
                ylabel="Misread Rate (%)",
                xticks=(0, 1, 2, 3) if column == 2 else (0, 0.5, 1, 1.5, 2),
                xticklabels=("Confusion", "Consensus", "Balanced", "Dominant")
                if column == 2
                else None,
            )
        else:
            selected = [row for row in rows if row["panel"] == panel]
            for model, label in MODEL_SPECS:
                group = [row for row in selected if row["model"] == model]
                if column == 2:
                    mapping = {
                        name: i
                        for i, name in enumerate(("Confusion", "Consensus", "Balanced", "Dominant"))
                    }
                    x = [mapping[row["pattern"]] for row in group]
                else:
                    x = [float(row["x"]) for row in group]
                axis.plot(
                    x,
                    [100 * float(row["misread_rate"]) for row in group],
                    marker="o",
                    label=label,
                    color=colors[model],
                )
            axis.set_title(title, fontweight="bold")
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Misread Rate (%)")
            axis.set_ylim(0, 100)
            axis.grid(color=GRID, linestyle=(0, (2, 3)))
            if column == 2:
                axis.set_xticks(
                    range(4), ("Confusion", "Consensus", "Balanced", "Dominant"), rotation=18
                )
            axis.legend(fontsize=8, frameon=False)
        axis.text(
            -0.16,
            1.10,
            f"({chr(97 + column)})",
            transform=axis.transAxes,
            fontsize=17,
            fontweight="bold",
        )
    cmap = LinearSegmentedColormap.from_list("lean", ["#2455A4", "#F8F8F8", "#E52B2B"])
    norm = Normalize(-1, 1)
    for column, (model, label) in enumerate(MODEL_SPECS):
        axis = figure.add_subplot(grid[1, column])
        selected = [row for row in rows if row["panel"] == "bias" and row["model"] == model]
        if ready:
            for outcome, marker in (("NON_MISREAD", "o"), ("MISREAD", "X")):
                group = [row for row in selected if row["label"] == outcome]
                axis.scatter(
                    [float(row["D_over_tau"]) for row in group],
                    [float(row["R"]) for row in group],
                    c=[float(row["R"]) for row in group],
                    cmap=cmap,
                    norm=norm,
                    marker=marker,
                    s=28 if marker == "o" else 42,
                    alpha=0.68,
                    edgecolors="#333333",
                    linewidths=0.3,
                )
        else:
            axis.scatter(
                [float(row["D_over_tau"]) for row in selected],
                [float(row["R"]) for row in selected],
                c=[float(row["R"]) for row in selected],
                cmap=cmap,
                norm=norm,
                s=22,
                alpha=0.65,
                edgecolors="#333333",
                linewidths=0.2,
            )
        axis.axvline(1, color="#888", linestyle=(0, (5, 4)))
        axis.axhline(0, color="#888", linestyle=(0, (2, 3)))
        axis.set_title(label, fontweight="bold")
        axis.set_xlabel(r"$\mathcal{D}/\tau$")
        axis.set_ylabel(r"Signed Joint Lean $\mathcal{R}$" if column == 0 else "")
        axis.set_ylim(-1.05, 1.05)
        axis.grid(color=GRID, linestyle=(0, (2, 3)), alpha=0.55)
        axis.text(
            -0.16,
            1.08,
            f"({chr(100 + column)})",
            transform=axis.transAxes,
            fontsize=17,
            fontweight="bold",
        )
    color_axis = figure.add_axes([0.945, 0.22, 0.014, 0.23])
    color_edges = np.linspace(-1.0, 1.0, 257)
    color_values = ((color_edges[:-1] + color_edges[1:]) / 2.0).reshape(-1, 1)
    color_axis.pcolormesh(
        np.asarray([0.0, 1.0]),
        color_edges,
        color_values,
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=False,
    )
    color_axis.set_xlim(0.0, 1.0)
    color_axis.set_ylim(-1.0, 1.0)
    color_axis.set_xticks(())
    color_axis.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    color_axis.yaxis.tick_right()
    if ready:
        figure.legend(
            handles=[
                Line2D([0], [0], marker="o", ls="", color="#555", label="Non-misread"),
                Line2D([0], [0], marker="X", ls="", color="#555", label="Misread"),
            ],
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.085),
        )
        exclusion_note = (
            "Formal eligible subset; excluded "
            f"unresolved={excluded['unresolved']}, blocked={excluded['blocked']}."
        )
        figure.text(
            0.5,
            0.025,
            exclusion_note,
            ha="center",
            fontsize=11,
            fontstyle="italic",
        )
    return figure


def _render_fig08(rows: list[dict[str, Any]], ready: bool) -> Any:
    _set_style()
    figure = plt.figure(figsize=(14.48, 10.86), dpi=100)
    figure.suptitle(
        "Figure 8. Frozen representation quality and Conflict-supervision sensitivity",
        fontsize=23,
        fontweight="bold",
        y=0.978,
    )
    grid = figure.add_gridspec(
        2, 3, left=0.05, right=0.965, bottom=0.12, top=0.88, wspace=0.28, hspace=0.62
    )
    for column, method in enumerate(FORMAL_METHODS):
        axis = figure.add_subplot(grid[0, column])
        top = [row for row in rows if row["panel"] == "ac_umap" and row["representation"] == method]
        for sample_type, color in (("Aligned", "#154BFF"), ("Conflict", "#FF2020")):
            group = [row for row in top if row["sample_type"] == sample_type]
            axis.scatter(
                [float(row["umap_x"]) for row in group],
                [float(row["umap_y"]) for row in group],
                s=16,
                color=color,
                alpha=0.75,
                edgecolors="#333",
                linewidths=0.2,
                label=sample_type,
            )
        axis.set_title(method, fontweight="bold")
        axis.set_xlabel("UMAP-1")
        axis.set_ylabel("UMAP-2")
        axis.grid(color=GRID, linestyle=(0, (2, 3)), alpha=0.55)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=10)
        axis.text(
            -0.16,
            1.10,
            f"({chr(97 + column)})",
            transform=axis.transAxes,
            fontsize=17,
            fontweight="bold",
        )
        bottom = figure.add_subplot(grid[1, column])
        budget = [
            row for row in rows if row["panel"] == "budget" and row["representation"] == method
        ]
        if not ready:
            _pending_panel(
                bottom,
                f"{method}\nConflict supervision sensitivity",
                xlabel="Conflict supervision retained (%)",
                ylabel="Misread AUPRC",
                xticks=FORMAL_BUDGETS,
                xticklabels=tuple(str(item) for item in FORMAL_BUDGETS),
                title_size=15,
            )
        else:
            means = []
            cis = []
            for pct in FORMAL_BUDGETS:
                values = np.asarray(
                    [float(row["auprc"]) for row in budget if int(row["budget_pct"]) == pct]
                )
                means.append(float(values.mean()))
                cis.append(
                    float(1.96 * values.std(ddof=1) / math.sqrt(values.size))
                    if values.size > 1
                    else 0.0
                )
            means_array = np.asarray(means)
            cis_array = np.asarray(cis)
            bottom.plot(FORMAL_BUDGETS, means_array, marker="o", color="#2F5597")
            bottom.fill_between(
                FORMAL_BUDGETS,
                means_array - cis_array,
                means_array + cis_array,
                color="#2F5597",
                alpha=0.18,
            )
            bottom.set_title(
                f"{method}\nConflict supervision sensitivity", fontweight="bold", fontsize=15
            )
            bottom.set_xlabel("Conflict supervision retained (%)")
            bottom.set_ylabel("Misread AUPRC")
            bottom.set_xticks(FORMAL_BUDGETS)
            bottom.set_ylim(0, 1)
            bottom.grid(color=GRID, linestyle=(0, (2, 3)))
        bottom.text(
            -0.16,
            1.08,
            f"({chr(100 + column)})",
            transform=bottom.transAxes,
            fontsize=17,
            fontweight="bold",
        )
    return figure


def _export_tables(
    *,
    labels: FormalRoot | None,
    probes: FormalRoot | None,
    label_rows: list[dict[str, Any]] | None,
    probe_rows: list[dict[str, Any]] | None,
    stable_bias_rows: list[dict[str, Any]],
    input_root: Path,
    output_root: Path,
    command: list[str],
    sources: list[dict[str, str]],
    excluded: dict[str, int],
) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    tab1_columns = (
        "Model / Protocol",
        "Diagnostic Acc. (Aligned)",
        "Diagnostic Acc. (Conflict)",
        "Dominant Modality Signature",
    )
    tab1 = []
    if labels and label_rows:
        eligible = [row for row in label_rows if row["label_eligible"]]
        for model, label in MODEL_SPECS:
            protocol = "VA" if model == "qwen2_5_omni_7b" else "VT"
            values = []
            for sample_type in ("Aligned", "Conflict"):
                group = [
                    row
                    for row in eligible
                    if row["model"] == model and row["sample_type"] == sample_type
                ]
                values.append(
                    f"{sum(row['label'] == 'NON_MISREAD' for row in group) / len(group):.3f}"
                    if group
                    else "Pending"
                )
            bias = [
                row
                for row in stable_bias_rows
                if row["panel"] == "bias" and row["model"] == model and float(row["D_over_tau"]) > 1
            ]
            v = sum(float(row["R"]) > 0 for row in bias)
            other = len(bias) - v
            name = "Visual" if v >= other else ("Audio" if protocol == "VA" else "Text")
            share = max(v, other) / len(bias) if bias else 0
            tab1.append(
                dict(
                    zip(
                        tab1_columns,
                        (f"{label} / {protocol}", values[0], values[1], f"{name} ({share:.1%})"),
                        strict=True,
                    )
                )
            )
    tables["tab01_cross_backbone_results_template_v3_misread"] = _write_table(
        "tab01_cross_backbone_results_template_v3_misread",
        tab1_columns,
        tab1,
        input_root,
        output_root,
        command,
        sources,
        labels is not None,
        {"excluded_labels": excluded},
    )
    tab2_columns = ("Method", "Accuracy", "Macro-F1", "AUPRC", "Latency")
    tab2 = []
    if probes and probe_rows:
        for method in FORMAL_METHODS:
            group = [row for row in probe_rows if row["method"] == method]

            def summary(name: str, rows: list[dict[str, Any]] = group) -> str:
                values = np.asarray([float(row[name]) for row in rows])
                return (
                    f"{values.mean():.3f} +/- {values.std(ddof=1):.3f}"
                    if values.size > 1
                    else f"{values.mean():.3f}"
                )

            tab2.append(
                dict(
                    zip(
                        tab2_columns,
                        (
                            method,
                            summary("accuracy"),
                            summary("macro_f1"),
                            summary("auprc"),
                            summary("latency_ms") + " ms",
                        ),
                        strict=True,
                    )
                )
            )
    tables["tab02_conflict_misread_baselines_template_v3_misread"] = _write_table(
        "tab02_conflict_misread_baselines_template_v3_misread",
        tab2_columns,
        tab2,
        input_root,
        output_root,
        command,
        sources,
        probes is not None,
        {},
    )
    return tables


def _write_table(
    key: str,
    columns: tuple[str, ...],
    rows: list[dict[str, Any]],
    input_root: Path,
    output_root: Path,
    command: list[str],
    sources: list[dict[str, str]],
    ready: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    csv_path = input_root / f"{key}.csv"
    _write_csv(csv_path, rows, fieldnames=columns)
    provenance = {
        "schema": "mprisk_template_v3_table_input_v1",
        "table_key": key,
        "status": "Ready" if ready else "Pending",
        "columns": list(columns),
        "input_sha256": sha256(csv_path),
        "row_count": len(rows),
        "generated_command": command,
        "sources": sources,
        **extra,
    }
    sidecar = csv_path.with_suffix(".csv.provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tex_path = output_root / f"{key}.tex"
    display = rows or [{column: "Pending" for column in columns}]
    lines = [
        "% Artifact-backed template-v3 table.",
        "\\begin{tabular}{l" + "c" * (len(columns) - 1) + "}",
        "\\toprule",
        " & ".join(columns) + r" \\",
        "\\midrule",
    ]
    lines.extend(
        " & ".join(str(row[column]).replace("%", r"\%") for column in columns) + r" \\"
        for row in display
    )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": provenance["status"],
        "input": str(csv_path),
        "input_sha256": sha256(csv_path),
        "provenance": str(sidecar),
        "output": str(tex_path),
        "output_sha256": sha256(tex_path),
    }


def _validate_cross_root_links(
    labels: FormalRoot | None,
    probes: FormalRoot | None,
    budgets: FormalRoot | None,
    probe_rows: list[dict[str, Any]] | None,
    budget_rows: list[dict[str, Any]] | None,
) -> None:
    if (probes or budgets) and labels is None:
        raise ValueError("formal probe/budget roots require the formal label root")
    if labels is None:
        return
    label_hashes = {item["model"]: item["sha256"] for item in labels.artifacts("labels")}
    for root, rows in ((probes, probe_rows), (budgets, budget_rows)):
        if root is None or rows is None:
            continue
        if root.marker["split_assignment_sha256"] != labels.marker["split_assignment_sha256"]:
            raise ValueError("formal roots use different split assignments")
        for row in rows:
            if row["label_artifact_sha256"] != label_hashes[row["model"]]:
                raise ValueError("formal metric row does not bind its model label artifact")


def _exclusion_counts(rows: list[dict[str, Any]] | None) -> dict[str, int]:
    if rows is None:
        return {"unresolved": 0, "blocked": 0, "label_eligible": 0, "probe_eligible": 0}
    return {
        "unresolved": sum(row["needs_manual_review"] for row in rows),
        "blocked": sum(row["blocked"] for row in rows),
        "label_eligible": sum(row["label_eligible"] for row in rows),
        "probe_eligible": sum(row["probe_eligible"] for row in rows),
    }


def _source_records(source_root: Path, *roots: FormalRoot | None) -> list[dict[str, str]]:
    records = []
    for key in ("fig05_four_state_stacks", "fig07_misread_bias", "fig08_representation_comparison"):
        for path in (source_root / f"{key}.csv", source_root / f"{key}.csv.provenance.json"):
            records.append({"path": str(path.resolve()), "sha256": sha256(path)})
    for root in roots:
        if root:
            records.append({"path": str(root.marker_path.resolve()), "sha256": root.marker_sha256})
            records.extend(
                {"path": str((root.root / item["path"]).resolve()), "sha256": item["sha256"]}
                for item in root.marker["artifacts"]
            )
    return records


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _write_provenance(
    path: Path,
    *,
    key: str,
    command: list[str],
    sources: list[dict[str, str]],
    readiness: dict[str, bool],
    excluded: dict[str, int],
) -> Path:
    provenance = {
        "schema": "mprisk_template_v3_figure_input_v1",
        "figure_key": key,
        "status": "Ready" if all(readiness.values()) else "Partial",
        "input_sha256": sha256(path),
        "row_count": sum(1 for _ in csv.DictReader(path.open(encoding="utf-8"))),
        "generated_command": command,
        "sources": sources,
        "panel_readiness": readiness,
        "excluded_labels": excluded,
        "synthetic_data_used": False,
    }
    sidecar = path.with_suffix(".csv.provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def _root_record(root: FormalRoot | None) -> dict[str, Any] | None:
    return (
        None
        if root is None
        else {
            "path": str(root.root),
            "marker": str(root.marker_path),
            "marker_sha256": root.marker_sha256,
            "schema": root.marker["schema"],
            "status": root.marker["status"],
        }
    )
