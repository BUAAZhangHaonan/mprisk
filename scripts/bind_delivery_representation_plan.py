#!/usr/bin/env python3
"""Bind immutable cache unions to the delivery representation plan."""

from __future__ import annotations

import argparse

from mprisk.experiments.delivery_representation import bind_delivery_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--cache-union", action="append", required=True, metavar="MODEL=PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    unions: dict[str, str] = {}
    for item in args.cache_union:
        if "=" not in item:
            parser.error("--cache-union must use MODEL=PATH")
        model_key, path = item.split("=", 1)
        if not model_key or not path or model_key in unions:
            parser.error("cache-union model keys and paths must be non-empty and unique")
        unions[model_key] = path
    plan = bind_delivery_plan(args.template, cache_unions=unions, output_path=args.output)
    print(f"RUNNABLE {plan.path} jobs={len(plan.jobs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
