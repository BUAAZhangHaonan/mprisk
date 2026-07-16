from __future__ import annotations

import argparse

from mprisk.evaluation.misread_probe import write_pending_conflict_misread_probe


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write the strict Pending contract for the future Conflict-only probe."
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    path = write_pending_conflict_misread_probe(args.output_dir)
    print(f"pending_probe={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
