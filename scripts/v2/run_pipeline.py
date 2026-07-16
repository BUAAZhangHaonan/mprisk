"""V2 pipeline CLI entry.

Usage:
    python scripts/v2/run_pipeline.py --config configs/v2/pipeline.yaml

Optional per-model override:
    python scripts/v2/run_pipeline.py --config ... --model qwen3_vl_8b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mprisk_v2.pipeline import V2ModelSpec, run_v2_for_model


def load_spec(model_key: str, cfg: dict) -> V2ModelSpec:
    mc = cfg["models"][model_key]
    return V2ModelSpec(
        model_key=model_key,
        protocol=mc["protocol"],
        cache_root=mc["cache_root"],
        prompt_set=mc["prompt_set"],
        prompt_set_key=mc["prompt_set_key"],
        main_manifest=mc["main_manifest"],
        smoke_manifest=mc.get("smoke_manifest", ""),
        train_config=mc["train_config"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None,
                        help="Run only this model key; default = all in config")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint if present")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_root = cfg["output_root"]
    Path(output_root).mkdir(parents=True, exist_ok=True)

    model_keys = [args.model] if args.model else list(cfg["models"].keys())

    results = []
    for mk in model_keys:
        spec = load_spec(mk, cfg)
        resume_ckpt = None
        if args.resume:
            cand = Path(output_root) / "checkpoints" / mk / "last_checkpoint.pt"
            if cand.exists():
                resume_ckpt = str(cand)
                print(f"[v2][{mk}] resuming from {cand}", flush=True)
        result = run_v2_for_model(
            spec=spec,
            split_assignment=cfg["split_assignment"],
            output_root=output_root,
            cache_root=spec.cache_root,
            prompt_cache_manifest=cfg.get("prompt_cache_manifest"),
            prompt_conditioned_cache_manifest=cfg.get("prompt_conditioned_cache_manifest"),
            kappa_quantile=cfg.get("kappa_quantile", 0.80),
            tau_quantile=cfg.get("tau_quantile", 0.50),
            max_epochs=cfg.get("max_epochs", 300),
            patience=cfg.get("patience", 30),
            device=args.device,
            resume_checkpoint=resume_ckpt,
        )
        results.append(result)
        print(f"[v2][{mk}] => {result.summary_path}", flush=True)
    print("[v2] all models done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
