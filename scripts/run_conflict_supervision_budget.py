#!/usr/bin/env python3
"""Run delivery-locked 10/25/50/100% Conflict-supervision experiments."""

from __future__ import annotations

import argparse

from mprisk.experiments.conflict_supervision_budget import (
    run_conflict_supervision_budget,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument(
        "--method",
        action="append",
        choices=("single_point", "trajectory_mlp", "tme"),
        dest="methods",
    )
    args = parser.parse_args()
    marker = run_conflict_supervision_budget(
        args.config,
        model_keys=set(args.models) if args.models else None,
        method_names=set(args.methods) if args.methods else None,
    )
    print(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
