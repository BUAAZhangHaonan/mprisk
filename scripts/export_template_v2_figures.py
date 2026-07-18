from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mprisk.viz.template_v2 import export_template_v2_figures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export real-data Fig. 4-8 using the template-v2 visual grammar."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/paper_exports/figures"),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/paper_exports/figures/template_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper/figures/generated/template_v2"),
    )
    args = parser.parse_args()
    result = export_template_v2_figures(
        source_root=args.source_root,
        input_root=args.input_root,
        output_root=args.output_root,
        generated_command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
