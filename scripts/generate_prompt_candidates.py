from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mprisk.prompts.pool import CANONICAL_PROMPT, RAW_POOL_SIZE

_VERB_GROUPS = (
    ("Describe", "Summarize", "State"),
    ("Capture", "Characterize", "Convey"),
    ("Report", "Write", "Provide"),
    ("Express", "Render", "Condense"),
    ("Present", "Frame", "Give"),
    ("Identify", "Portray", "Outline"),
)
_INPUT_GROUPS = (
    ("complete input", "full input", "entire input"),
    ("all given input", "complete input", "whole input"),
    ("provided input as a whole", "full input", "complete input"),
    ("entire supplied input", "all input", "complete input"),
)


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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_hashes(model_path: Path) -> dict[str, Any]:
    patterns = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "model.safetensors.index.json",
        "*.safetensors",
        "*.bin",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(model_path.glob(pattern)))
    unique = []
    seen = set()
    for path in files:
        if path.is_file() and path not in seen:
            unique.append(path)
            seen.add(path)
    return {
        "files": [
            {
                "path": str(path),
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in unique
        ]
    }


def load_model(model_path: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def _chat(tokenizer, system: str, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _decode_json_list(text: str, *, expected_count: int, field: str = "strings") -> list[Any]:
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"Model returned {field} payload that is not a JSON list")
    if len(payload) != expected_count:
        raise ValueError(f"Expected {expected_count} {field}, got {len(payload)}")
    return payload


def _generate_json_list(
    tokenizer,
    model,
    *,
    system: str,
    user: str,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    do_sample: bool,
) -> tuple[str, list[Any]]:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    prompt = _chat(tokenizer, system, user)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
    try:
        return raw, json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Model returned invalid JSON: {raw[:500]}") from error


def generation_prompt(batch_size: int, batch_index: int) -> str:
    if batch_size == 1:
        verbs = ", ".join(_VERB_GROUPS[batch_index % len(_VERB_GROUPS)])
        input_phrases = ", ".join(_INPUT_GROUPS[batch_index % len(_INPUT_GROUPS)])
        return f"""
Return valid JSON only. Do not use markdown. Do not use a code fence.
The first character must be [ and the last character must be ].
The top-level JSON value must be an array containing exactly one string.
Do not return alternatives. Do not return more than one string.
The string must be a candidate instruction, not an answer.
The string must ask another system to perform the same task as this canonical task:
{CANONICAL_PROMPT}

Rules for the one string:
- 12 to 22 English words. Count the words before returning the JSON.
- One sentence only.
- Ask for only one concise description of the person's overall emotional state.
- Use the phrase "overall emotional state" or a very close paraphrase.
- Include the idea of using the complete input.
- Include the phrase "in one concise sentence".
- Start with an instruction verb.
- Do not address the person.
- Do not give advice.
- Do not ask for reasoning or explanation.
- Do not mention examples, labels, modalities, conflict, ambiguity, sarcasm, or specific
  emotion names.
- Do not use the words narrative, content, material, context, or text; use input instead.
- Do not include placeholders, the word question, or sample_text.
- Preferred instruction verbs for variation: {verbs}.
- Preferred complete-input wording for variation: {input_phrases}.
- Do not copy the canonical prompt exactly.
- Use private variation id {batch_index} to choose fresh wording, but do not mention the id.
""".strip()
    return f"""
Return valid JSON only. Do not use markdown. Do not use a code fence.
The first character must be [ and the last character must be ].
The top-level JSON value must be an array of strings, not an object.
Return exactly {batch_size} different English strings.
Each string must be a candidate instruction, not an answer.
Each string must ask another system to perform the same task as this canonical task:
{CANONICAL_PROMPT}

Rules for every string:
- 12 to 22 English words. Count the words before returning the JSON.
- One sentence only.
- Ask for only one concise description of the person's overall emotional state.
- Use the phrase "overall emotional state" or a very close paraphrase.
- Include the idea of using the complete input.
- Include the phrase "in one concise sentence" in every string.
- Start each string with an instruction verb such as Describe, Summarize, State, Capture,
  Characterize, Convey, Report, Write, or Provide.
- Do not address the person.
- Do not give advice.
- Do not ask for reasoning or explanation.
- Do not mention examples, labels, modalities, conflict, ambiguity, sarcasm, or specific
  emotion names.
- Do not use the words narrative, content, material, context, or text; use input instead.
- Do not include placeholders, the word question, or sample_text.
- Use varied wording across the list and across candidate number {batch_index}.
- Do not include candidate numbers, placeholders, brackets, or braces inside any string.
""".strip()


def review_prompt(items: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "task": (
                "For each candidate, decide if it is fully equivalent to the canonical "
                "instruction."
            ),
            "canonical_prompt": CANONICAL_PROMPT,
            "pass_criteria": [
                (
                    "It asks only for one concise sentence describing the person's overall "
                    "emotional state."
                ),
                "It does not add advice, reasoning, explanation, labels, examples, or extra tasks.",
                "It does not rely on a particular modality.",
                "It is not an answer; it is an instruction.",
            ],
            "output_contract": (
                "Return JSON list only, same length/order, each object exactly with keys "
                "raw_index, passed, reason."
            ),
            "candidates": items,
        },
        ensure_ascii=False,
    )


def generate_candidates(
    *,
    model_path: str | Path,
    output_dir: str | Path,
    count: int = RAW_POOL_SIZE,
    batch_size: int = 8,
    review_batch_size: int = 16,
    seed: int = 20260715,
    temperature: float = 0.95,
    top_p: float = 0.92,
    max_new_tokens: int = 768,
) -> dict[str, Any]:
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_model_candidates_384.jsonl"
    review_path = output_dir / "ai_reviews.jsonl"
    plan_path = output_dir / "generation_plan.json"
    provenance_path = output_dir / "generation_provenance.json"
    log_path = output_dir / "generation_log.jsonl"

    if count != RAW_POOL_SIZE:
        raise ValueError(f"This milestone requires exactly {RAW_POOL_SIZE} candidates")
    if count % batch_size != 0:
        raise ValueError("count must be divisible by batch_size")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    tokenizer, model = load_model(model_path)
    import torch
    import transformers

    plan = build_generation_plan(
        model_path=model_path,
        output_path=raw_path,
        count=count,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    write_generation_plan(plan, plan_path)

    provenance = {
        "schema": "mprisk_prompt_candidate_generation_provenance_v1",
        "model_path": str(model_path),
        "model_hashes": model_hashes(model_path),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": str(model.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "count": count,
        "batch_size": batch_size,
        "review_batch_size": review_batch_size,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "canonical_prompt": CANONICAL_PROMPT,
        "generator": "scripts.generate_prompt_candidates.generate_candidates",
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    raw_rows: list[dict[str, Any]] = []
    with (
        raw_path.open("w", encoding="utf-8") as raw_handle,
        log_path.open("w", encoding="utf-8") as log_handle,
    ):
        for batch_index in range(count // batch_size):
            batch_seed = seed + batch_index
            started = time.time()
            raw_text, _payload = _generate_json_list(
                tokenizer,
                model,
                system="You are a strict JSON API. You output only valid JSON and no markdown.",
                user=generation_prompt(batch_size, batch_index),
                seed=batch_seed,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )
            candidates = _decode_json_list(raw_text, expected_count=batch_size, field="candidates")
            for value in candidates:
                if not isinstance(value, str):
                    raise ValueError(f"Batch {batch_index} contained a non-string candidate")
            for text in candidates:
                raw_index = len(raw_rows)
                row = {
                    "schema": "mprisk_prompt_candidate_raw_v1",
                    "raw_index": raw_index,
                    "batch_index": batch_index,
                    "seed": batch_seed,
                    "text": text,
                    "model_path": str(model_path),
                    "decoding": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                    },
                }
                raw_rows.append(row)
                raw_handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            log_handle.write(
                json.dumps(
                    {
                        "event": "generate_batch",
                        "batch_index": batch_index,
                        "seed": batch_seed,
                        "count": len(candidates),
                        "elapsed_seconds": time.time() - started,
                        "raw_response": raw_text,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )
            raw_handle.flush()
            log_handle.flush()

    review_rows: list[dict[str, Any]] = []
    with (
        review_path.open("w", encoding="utf-8") as review_handle,
        log_path.open("a", encoding="utf-8") as log_handle,
    ):
        for start in range(0, len(raw_rows), review_batch_size):
            chunk = raw_rows[start : start + review_batch_size]
            batch_seed = seed + 10000 + start
            started = time.time()
            review_items = [{"raw_index": row["raw_index"], "text": row["text"]} for row in chunk]
            raw_text, _payload = _generate_json_list(
                tokenizer,
                model,
                system=(
                    "You are a strict JSON reviewer. You output only valid JSON and no "
                    "markdown."
                ),
                user=review_prompt(review_items),
                seed=batch_seed,
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            reviews = _decode_json_list(raw_text, expected_count=len(chunk), field="reviews")
            by_index = {row["raw_index"]: row for row in chunk}
            for review in reviews:
                if not isinstance(review, dict):
                    raise ValueError("Review payload contains a non-object item")
                if set(review) != {"raw_index", "passed", "reason"}:
                    raise ValueError(f"Review object has invalid keys: {sorted(review)}")
                raw_index = int(review["raw_index"])
                if raw_index not in by_index:
                    raise ValueError(f"Review raw_index out of batch: {raw_index}")
                if not isinstance(review["passed"], bool) or not isinstance(review["reason"], str):
                    raise ValueError("Review object must contain bool passed and string reason")
                source = by_index[raw_index]
                row = {
                    **source,
                    "ai_semantic_review_pass": review["passed"],
                    "semantic_review": {
                        "passed": review["passed"],
                        "reason": review["reason"],
                        "review_model_path": str(model_path),
                        "review_seed": batch_seed,
                        "review_temperature": 0.0,
                        "review_top_p": 1.0,
                    },
                }
                review_rows.append(row)
                review_handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            log_handle.write(
                json.dumps(
                    {
                        "event": "review_batch",
                        "start": start,
                        "count": len(reviews),
                        "seed": batch_seed,
                        "elapsed_seconds": time.time() - started,
                        "raw_response": raw_text,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )
            review_handle.flush()
            log_handle.flush()

    if len(raw_rows) != count or len(review_rows) != count:
        raise ValueError("Generation/review counts did not match requested count")
    return {
        "raw_path": str(raw_path),
        "review_path": str(review_path),
        "plan_path": str(plan_path),
        "provenance_path": str(provenance_path),
        "log_path": str(log_path),
        "count": count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and AI-review local prompt candidates with a local model."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-path", default=None, help="Deprecated alias; use --output-dir.")
    parser.add_argument("--plan-path", default=None, help="Optional extra copy of generation plan.")
    parser.add_argument("--count", type=int, default=RAW_POOL_SIZE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--review-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--write-plan-only",
        action="store_true",
        help="Only write a generation plan without loading the model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    raw_output = (
        Path(args.output_path)
        if args.output_path
        else output_dir / "raw_model_candidates_384.jsonl"
    )
    plan_path = Path(args.plan_path) if args.plan_path else output_dir / "generation_plan.json"
    if args.write_plan_only:
        plan = build_generation_plan(
            model_path=Path(args.model_path),
            output_path=raw_output,
            count=args.count,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        written = write_generation_plan(plan, plan_path)
        print(
            json.dumps(
                {"generation_plan": str(written), "local_model_generation": "not_started"}
            )
        )
        return 0

    result = generate_candidates(
        model_path=Path(args.model_path),
        output_dir=output_dir,
        count=args.count,
        batch_size=args.batch_size,
        review_batch_size=args.review_batch_size,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    if args.plan_path:
        write_generation_plan(
            build_generation_plan(
                model_path=Path(args.model_path),
                output_path=raw_output,
                count=args.count,
                seed=args.seed,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            ),
            args.plan_path,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
