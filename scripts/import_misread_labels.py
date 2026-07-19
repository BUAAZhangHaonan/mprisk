from __future__ import annotations

import argparse
import json

from mprisk.data.misread_labels import (
    DEFAULT_DELIVERY_MANIFEST,
    DEFAULT_INVALID_ASSETS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SPLIT_ASSIGNMENT,
    DEFAULT_V2_ROOT,
    import_single_flash_labels,
    verify_imported_labels,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strictly import pinned delivery_20260716 single-Flash Misread labels."
    )
    parser.add_argument("--v2-root", default=str(DEFAULT_V2_ROOT))
    parser.add_argument("--delivery-manifest", default=str(DEFAULT_DELIVERY_MANIFEST))
    parser.add_argument("--split-assignment", default=str(DEFAULT_SPLIT_ASSIGNMENT))
    parser.add_argument("--invalid-assets", default=str(DEFAULT_INVALID_ASSETS))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        marker = verify_imported_labels(args.output_root)
        print(json.dumps(marker, sort_keys=True))
        return 0
    result = import_single_flash_labels(
        v2_root=args.v2_root,
        delivery_manifest=args.delivery_manifest,
        split_assignment=args.split_assignment,
        invalid_assets=args.invalid_assets,
        output_root=args.output_root,
        confidence_threshold=args.confidence_threshold,
    )
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "marker": str(result.marker_path),
                "rows": result.total_rows,
                "probe_eligible": result.probe_eligible_rows,
                "needs_manual_review": result.manual_review_rows,
                "blocked": result.blocked_rows,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
