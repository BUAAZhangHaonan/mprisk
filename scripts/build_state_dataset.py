from __future__ import annotations

import argparse
from pathlib import Path

from mprisk.data.state_dataset import build_state_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a state-data manifest from final labels and full-cache entries."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        dest="manifests",
        required=True,
        help="Final label manifest JSONL. Pass more than once for conflict+aligned.",
    )
    parser.add_argument("--cache-root", default=".", help="Project root or cache root.")
    parser.add_argument(
        "--cache-manifest",
        default=None,
        help="Path to unified_full_cache_manifest.json, relative to cache root if not absolute.",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="Path to extraction_ledger.csv, relative to cache root if not absolute.",
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--split-assignment", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_state_dataset(
        manifest_paths=[Path(path) for path in args.manifests],
        cache_root=args.cache_root,
        manifest_path=args.cache_manifest,
        ledger_path=args.ledger,
        model_key=args.model_key,
        protocol=args.protocol,
        split_assignment_path=args.split_assignment,
        output_dir=args.output_dir,
    )
    print(f"state_dataset_manifest={result.manifest_path}")
    print(f"state_dataset_summary={result.summary_path}")
    print(f"missing_cache_rows={result.missing_path}")
    print(f"resolved_rows={result.resolved_count}")
    print(f"missing_cache_rows={result.missing_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
