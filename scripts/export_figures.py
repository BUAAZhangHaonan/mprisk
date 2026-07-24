"""V2 figure export (Fig.4, Fig.5, Fig.6) as PDFs + input CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mprisk_viz.plotting import (
    load_state_patterns,
    plot_fig04_sdr_distributions,
    plot_fig05_pattern_stacks,
    plot_fig06_stable_d_r,
)


def _load_thresholds(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--figures-dir", default=None,
                        help="Override figures dir; default = output_root/figures")
    parser.add_argument("--split", default="official_test")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_root = Path(cfg["output_root"])
    fig_dir = Path(args.figures_dir) if args.figures_dir else output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_root / "figure_inputs"
    csv_dir.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, dict] = {}
    from mprisk.data.protocol_views import normalize_protocol
    for mk, mc in cfg["models"].items():
        protocol = normalize_protocol(mc["protocol"])
        patterns_path = (
            output_root / "state_data" / mk / protocol / "state_patterns.jsonl"
        )
        thresholds_path = (
            output_root / "state_data" / mk / protocol / "thresholds.json"
        )
        if not patterns_path.exists():
            print(f"[v2-fig] skip {mk}: {patterns_path} missing", flush=True)
            continue
        df = load_state_patterns(patterns_path)
        df.to_csv(csv_dir / f"{mk}_state_patterns.csv", index=False)
        dfs[mk] = df
        thresholds[mk] = _load_thresholds(thresholds_path)

    if not dfs:
        print("[v2-fig] no models ready; nothing to plot", flush=True)
        return 1

    print(f"[v2-fig] plotting for models: {list(dfs.keys())}", flush=True)

    fig04 = plot_fig04_sdr_distributions(
        dfs, fig_dir / "fig04_sdr_distributions.pdf",
        split_filter=args.split,
    )
    fig05 = plot_fig05_pattern_stacks(
        dfs, fig_dir / "fig05_four_state_stacks.pdf",
        split_filter=args.split,
    )
    fig06 = plot_fig06_stable_d_r(
        dfs, thresholds, fig_dir / "fig06_stable_d_signed_r.pdf",
        split_filter=args.split,
    )

    summary = {
        "models": list(dfs.keys()),
        "split": args.split,
        "figures": {
            "fig04": str(fig04),
            "fig05": str(fig05),
            "fig06": str(fig06),
        },
        "sample_counts": {mk: int(len(df)) for mk, df in dfs.items()},
    }
    summary_path = fig_dir / "figure_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[v2-fig] summary: {summary_path}", flush=True)
    for p in [fig04, fig05, fig06]:
        print(f"  PDF: {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
