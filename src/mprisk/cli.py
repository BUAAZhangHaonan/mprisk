"""Shared fail-closed behavior for commands that have not been implemented."""

from __future__ import annotations

import argparse


def scaffold_main(command_name: str) -> int:
    """Validate scaffold wiring, but never report an unimplemented task as completed."""

    parser = argparse.ArgumentParser(
        description=f"{command_name} is retained as an unavailable scaffold command"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate that the unavailable command is wired and fails closed.",
    )
    args = parser.parse_args()
    if args.dry_run:
        print(f"{command_name}: unavailable scaffold; dry-run wiring valid")
        return 0
    parser.error(
        f"{command_name} is not implemented; use a documented active entry point instead"
    )
