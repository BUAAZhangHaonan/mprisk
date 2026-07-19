from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mprisk.viz.template_v3_misread import export_template_v3_misread


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export additive template-v3 figures/tables from formal Misread roots."
    )
    parser.add_argument("--source-root", type=Path, default=Path("outputs/paper_exports/figures"))
    parser.add_argument("--labels-root", type=Path)
    parser.add_argument("--probes-root", type=Path)
    parser.add_argument("--budgets-root", type=Path)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/paper_exports/figures/template_v3_misread"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper/figures/generated/template_v3_misread"),
    )
    parser.add_argument(
        "--table-input-root",
        type=Path,
        default=Path("outputs/paper_exports/tables/template_v3_misread"),
    )
    parser.add_argument(
        "--table-output-root",
        type=Path,
        default=Path("paper/tables/generated/template_v3_misread"),
    )
    args = parser.parse_args()
    result = export_template_v3_misread(
        source_root=args.source_root,
        labels_root=args.labels_root,
        probes_root=args.probes_root,
        budgets_root=args.budgets_root,
        input_root=args.input_root,
        output_root=args.output_root,
        table_input_root=args.table_input_root,
        table_output_root=args.table_output_root,
        generated_command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
