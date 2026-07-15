from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "LEGACY DISABLED: generic correctness and mixed A/C rows cannot be used as "
            "Misread labels. Use run_conflict_misread_probe.py after annotations exist."
        )
    )
    parser.parse_args()
    parser.error(
        "legacy state-to-error analysis is disabled; Misread requires independent "
        "Conflict-only annotations and frozen representations"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
