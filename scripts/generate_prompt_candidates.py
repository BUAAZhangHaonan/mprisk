from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mprisk.prompts.pool import CANONICAL_PROMPT, RAW_POOL_SIZE


@dataclass(frozen=True)
class PromptCandidateGenerationPlan:
    schema: str
    model_path: str
    output_path: str
    count: int
    seed: int
    temperature: float
    top_p: float
    max_new_tokens: int
    canonical_prompt: str
    instruction: str


def build_generation_plan(
    *,
    model_path: str | Path,
    output_path: str | Path,
    count: int = RAW_POOL_SIZE,
    seed: int = 20260715,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_new_tokens: int = 256,
) -> PromptCandidateGenerationPlan:
    return PromptCandidateGenerationPlan(
        schema="mprisk_prompt_candidate_generation_plan_v1",
        model_path=str(model_path),
        output_path=str(output_path),
        count=count,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        canonical_prompt=CANONICAL_PROMPT,
        instruction=(
            "Generate diverse English candidate instructions for the same task as the "
            "canonical prompt. Each candidate must be one concise sentence and must not "
            "name modalities, examples, labels, conflict, ambiguity, or specific emotions."
        ),
    )


def write_generation_plan(plan: PromptCandidateGenerationPlan, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a local-model prompt-candidate generation plan."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--plan-path", required=True)
    parser.add_argument("--count", type=int, default=RAW_POOL_SIZE)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_generation_plan(
        model_path=Path(args.model_path),
        output_path=Path(args.output_path),
        count=args.count,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    plan_path = write_generation_plan(plan, args.plan_path)
    print(f"generation_plan={plan_path}")
    print("local_model_generation=not_started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
