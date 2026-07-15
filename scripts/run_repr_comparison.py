from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "LEGACY DISABLED: this comparison consumes generic state-to-error outputs. "
            "Use the official Conflict/Aligned downstream queue instead."
        )
    )
    parser.parse_args()
    parser.error(
        "legacy representation comparison is disabled; it cannot be used as a Misread paper input"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
