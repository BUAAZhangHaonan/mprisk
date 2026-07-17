#!/usr/bin/env python3
"""Run the bound delivery representation queue to completion."""

from __future__ import annotations

import argparse

from mprisk.experiments.delivery_representation import run_delivery_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--model-key",
        action="append",
        choices=("qwen3_vl_8b", "internvl3_5_8b", "qwen2_5_omni_7b"),
        help="Run exactly the jobs bound in a partial plan; repeat for multiple jobs.",
    )
    args = parser.parse_args()
    if args.model_key is not None and len(args.model_key) != len(set(args.model_key)):
        parser.error("--model-key values must be unique")
    selected = set(args.model_key) if args.model_key is not None else None
    return run_delivery_plan(args.plan, model_keys=selected)


if __name__ == "__main__":
    raise SystemExit(main())
