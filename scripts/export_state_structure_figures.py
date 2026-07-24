from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mprisk.viz.state_structure_figures import export_state_structure_figures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export real-data Fig. 4-8 using the canonical state-structure visual grammar."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/paper_exports/figures"),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/paper_exports/figures/state_structure"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("paper/figures/generated/state_structure"),
    )
    args = parser.parse_args()
    result = export_state_structure_figures(
        source_root=args.source_root,
        input_root=args.input_root,
        output_root=args.output_root,
        generated_command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
