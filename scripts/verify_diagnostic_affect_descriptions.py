from __future__ import annotations

import argparse
import json
from pathlib import Path

from mprisk.diagnostic_affect.generation import verify_diagnostic_affect_descriptions


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Diagnostic Affect Description artifacts.")
    parser.add_argument("--manifest-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subject-model-key", required=True)
    parser.add_argument("--protocol", required=True, choices=("VT", "VA"))
    parser.add_argument("--condition", default="M12", choices=("M12",))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_diagnostic_affect_descriptions(
                manifest_path=args.manifest_path,
                output_root=args.output_root,
                subject_model_key=args.subject_model_key,
                protocol=args.protocol,
                condition=args.condition,
                dataset=args.dataset,
                split=args.split,
                strict_full=not args.smoke,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
