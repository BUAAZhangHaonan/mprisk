from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.data.state_bundle import build_state_bundles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prompt-conditioned state bundle manifests."
    )
    parser.add_argument(
        "--state-dataset-manifest",
        required=True,
        help="Input state_dataset_manifest.jsonl.",
    )
    parser.add_argument(
        "--prompt-cache-manifest",
        required=True,
        help="Input prompt cache manifest JSONL.",
    )
    parser.add_argument(
        "--prompt-conditioned-cache-manifest",
        required=True,
        help="Input prompt-conditioned cache manifest JSONL.",
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument(
        "--prompt-set",
        default=None,
        help="Prompt set YAML. If omitted, --prompt-set-key is resolved in --prompt-set-dir.",
    )
    parser.add_argument("--prompt-set-key", default=None)
    parser.add_argument("--prompt-set-dir", default="configs/prompts/equiv_sets")
    parser.add_argument("--output-root", default="outputs/state_bundles")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_state_bundles(
        state_dataset_manifest_path=Path(args.state_dataset_manifest),
        prompt_cache_manifest_path=Path(args.prompt_cache_manifest),
        prompt_conditioned_cache_manifest_path=Path(args.prompt_conditioned_cache_manifest),
        model_key=args.model_key,
        protocol=args.protocol,
        prompt_set_path=Path(args.prompt_set) if args.prompt_set else None,
        prompt_set_key=args.prompt_set_key,
        prompt_set_dir=Path(args.prompt_set_dir),
        output_root=Path(args.output_root),
    )
    print(f"bundle_manifest={result.manifest_path}")
    print(f"bundle_summary={result.summary_path}")
    print(f"missing_prompt_cache_rows={result.missing_path}")
    print(f"complete_samples={result.complete_count}")
    print(f"missing_samples={result.missing_count}")
    print(f"prompt_count={result.prompt_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
