"""Shared scaffold command-line behavior."""

from __future__ import annotations

import argparse


def scaffold_main(command_name: str) -> int:
    parser = argparse.ArgumentParser(description=f"{command_name} scaffold command")
    parser.add_argument("--dry-run", action="store_true", help="Validate command wiring only.")
    args = parser.parse_args()
    if args.dry_run:
        print(f"{command_name}: dry run")
        return 0
    print(f"{command_name}: scaffold command")
    return 0
