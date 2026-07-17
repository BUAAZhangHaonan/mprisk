#!/usr/bin/env python3
"""Bind immutable cache unions to the delivery representation plan."""

from __future__ import annotations

import argparse

from mprisk.experiments.delivery_representation import bind_delivery_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument(
        "--model-key",
        action="append",
        choices=("qwen3_vl_8b", "internvl3_5_8b", "qwen2_5_omni_7b"),
        help="Bind exactly this model job; repeat for multiple jobs. Omit only for all jobs.",
    )
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
    if args.model_key is not None and len(args.model_key) != len(set(args.model_key)):
        parser.error("--model-key values must be unique")
    selected = set(args.model_key) if args.model_key is not None else None
    plan = bind_delivery_plan(
        args.template,
        cache_unions=unions,
        output_path=args.output,
        model_keys=selected,
    )
    print(
        f"RUNNABLE {plan.path} selected={','.join(plan.selected_model_keys)} "
        f"pending={','.join(plan.pending_model_keys) or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
