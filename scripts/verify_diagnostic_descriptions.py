from __future__ import annotations

import argparse
import json
from pathlib import Path

from mprisk.diagnostic_descriptions.qwen_omni_m12 import verify_description_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify strict diagnostic-description artifacts.")
    parser.add_argument("--eligible-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_description_artifacts(
                eligible_path=args.eligible_path,
                output_root=args.output_root,
                strict_full=not args.smoke,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
