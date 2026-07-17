#!/usr/bin/env python3
"""Run the bound delivery representation queue to completion."""

from __future__ import annotations

import argparse

from mprisk.experiments.delivery_representation import run_delivery_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    return run_delivery_plan(args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
