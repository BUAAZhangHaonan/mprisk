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
    parser.add_argument(
        "--method-key",
        action="append",
        choices=(
            "tme_pa_only_v1",
            "tme_pa_dtheta_v1",
            "tme_pa_dstrong_v2",
            "single_point_binary_v1",
            "trajectory_mlp_binary_v1",
        ),
        help="Run only these registered methods; omit to run every method in the plan.",
    )
    args = parser.parse_args()
    if args.model_key is not None and len(args.model_key) != len(set(args.model_key)):
        parser.error("--model-key values must be unique")
    selected = set(args.model_key) if args.model_key is not None else None
    if args.method_key is not None and len(args.method_key) != len(set(args.method_key)):
        parser.error("--method-key values must be unique")
    selected_methods = set(args.method_key) if args.method_key is not None else None
    return run_delivery_plan(
        args.plan,
        model_keys=selected,
        method_keys=selected_methods,
    )


if __name__ == "__main__":
    raise SystemExit(main())
