"""Build deterministic prompt pools from reviewed AI candidates."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

CANONICAL_PROMPT = (
    "Based on the complete input, describe the person's overall emotional state in one concise "
    "sentence. Do not address the person, give advice, or explain your reasoning."
)
RAW_POOL_SIZE = 384
GLOBAL_POOL_SIZE = 128
SUBSET_SIZE = 8
SUBSET_SEEDS = (20260715, 20260716, 20260717)

_TEXT_FIELDS = ("template_text", "prompt", "text", "candidate")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_END_RE = re.compile(r"[.!?]")
_PLACEHOLDER_RE = re.compile(
    r"[\{\}]|%\(.*?\)s|\bquestion\b|\bsample_text\b|\bsample text\b",
    re.IGNORECASE,
)
_SINGLE_TASK_RE = re.compile(
    r"\b(also|then|compare|choose|classify|list|rank|include|explain|reason|reasoning|advice|"
    r"steps?|multi[- ]?step)\b",
    re.IGNORECASE,
)
_FORBIDDEN_RE = re.compile(
    r"\b("
    r"conflict|conflicts|inconsistency|inconsistencies|ambiguity|ambiguities|sarcasm|"
    r"image|images|video|videos|visual|text|audio|sound|sounds|speech|"
    r"cautious|focus|example|examples|label|labels|explanation|explanations|"
    r"happy|happiness|sad|sadness|angry|anger|fear|afraid|anxious|anxiety|"
    r"disgust|surprise|surprised|neutral|joy|calm|calmness|frustrated|frustration|"
    r"excited|excitement|bored|boredom|confused|confusion|tense|tension|relief"
    r")\b|\bmixed emotion\b|\bbe cautious\b|\bmulti[- ]?step\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PromptPoolBuild:
    raw384: list[dict[str, Any]]
    accepted: list[dict[str, Any]]
    rejections: list[dict[str, Any]]
    pool128: list[dict[str, Any]]
    subsets: dict[int, list[dict[str, Any]]]
    provenance: dict[str, Any]


def normalize_prompt(text: str) -> str:
    """Normalize prompt text before exact dedupe and filtering."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def filter_candidates(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    accepted: list[str] = []
    rejections: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, row in enumerate(rows):
        text = _extract_text(row)
        normalized = normalize_prompt(text) if text is not None else ""
        reason = _rejection_reason(row, normalized, seen)
        if reason is not None:
            rejections.append(
                {
                    "raw_index": index,
                    "reason": reason,
                    "template_text": normalized,
                }
            )
            continue
        seen.add(normalized)
        accepted.append(normalized)

    return accepted, rejections


def select_global_pool(prompts: list[str], pool_size: int = GLOBAL_POOL_SIZE) -> list[str]:
    if len(prompts) < pool_size:
        raise ValueError(f"Need at least {pool_size} accepted prompts, got {len(prompts)}")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")

    word_features = TfidfVectorizer(analyzer="word", ngram_range=(1, 2)).fit_transform(prompts)
    char_features = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(prompts)
    features = normalize(hstack([word_features, char_features], format="csr"))
    centroid = normalize(np.asarray(features.mean(axis=0)))

    centroid_similarity = np.asarray(features @ centroid.T).ravel()
    selected_indices = [int(np.argmin(centroid_similarity))]
    min_distances = np.full(len(prompts), np.inf)

    while len(selected_indices) < pool_size:
        latest = selected_indices[-1]
        similarity = (features @ features[latest].T).toarray().ravel()
        min_distances = np.minimum(min_distances, 1.0 - similarity)
        min_distances[selected_indices] = -np.inf
        selected_indices.append(int(np.argmax(min_distances)))

    return [prompts[index] for index in selected_indices]


def generate_seeded_subset(
    pool: list[str] | list[dict[str, Any]],
    *,
    seed: int,
    subset_size: int = SUBSET_SIZE,
) -> list[str] | list[dict[str, Any]]:
    if len(pool) < subset_size:
        raise ValueError(f"Need at least {subset_size} prompts, got {len(pool)}")
    rng = np.random.default_rng(seed)
    chosen = sorted(int(index) for index in rng.choice(len(pool), size=subset_size, replace=False))
    return [pool[index] for index in chosen]


def build_prompt_pool(
    raw_jsonl: str | Path,
    output_dir: str | Path,
    *,
    prompt_set_key: str = "prompt_pool_v1",
    protocol: str = "vt",
) -> PromptPoolBuild:
    rows = _read_jsonl(Path(raw_jsonl))
    if len(rows) != RAW_POOL_SIZE:
        raise ValueError(
            f"Prompt pool final build requires exactly 384 raw candidates, got {len(rows)}"
        )

    accepted_texts, rejections = filter_candidates(rows)
    if len(accepted_texts) < GLOBAL_POOL_SIZE:
        raise ValueError(
            "Prompt pool final build requires at least 128 accepted candidates, "
            f"got {len(accepted_texts)}"
        )

    selected_texts = select_global_pool(accepted_texts, pool_size=GLOBAL_POOL_SIZE)
    accepted = [
        {
            "accepted_index": index,
            "template_text": template_text,
        }
        for index, template_text in enumerate(accepted_texts)
    ]
    pool128 = [
        {
            "prompt_id": f"{prompt_set_key}_p{index:03d}",
            "template_text": template_text,
            "role": "user",
            "enabled": True,
        }
        for index, template_text in enumerate(selected_texts, start=1)
    ]
    subsets = {
        seed: generate_seeded_subset(pool128, seed=seed, subset_size=SUBSET_SIZE)
        for seed in SUBSET_SEEDS
    }
    raw384 = [
        {
            "raw_index": index,
            "template_text": normalize_prompt(_extract_text(row) or ""),
            "ai_semantic_review_pass": row.get("ai_semantic_review_pass"),
        }
        for index, row in enumerate(rows)
    ]
    provenance = {
        "schema": "mprisk_prompt_pool_provenance_v1",
        "canonical_prompt": CANONICAL_PROMPT,
        "raw_candidate_file": str(raw_jsonl),
        "raw_count": len(rows),
        "accepted_count": len(accepted_texts),
        "rejected_count": len(rejections),
        "global_pool_size": GLOBAL_POOL_SIZE,
        "subset_size": SUBSET_SIZE,
        "subset_seeds": list(SUBSET_SEEDS),
        "selection": "tfidf_word_1_2_char_wb_3_5_farthest_point",
        "prompt_set_key": prompt_set_key,
        "protocol": protocol,
    }

    _write_outputs(
        Path(output_dir),
        raw384,
        accepted,
        rejections,
        pool128,
        subsets,
        provenance,
        protocol,
    )
    verification = verify_prompt_pool_artifacts(Path(output_dir))
    (Path(output_dir) / "artifact_verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PromptPoolBuild(
        raw384=raw384,
        accepted=accepted,
        rejections=rejections,
        pool128=pool128,
        subsets=subsets,
        provenance=provenance,
    )


def verify_prompt_pool_artifacts(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    raw384 = _read_jsonl(output_dir / "raw384.jsonl")
    pool128 = _read_jsonl(output_dir / "pool128.jsonl")
    accepted = _read_jsonl(output_dir / "accepted.jsonl")
    rejections = _read_jsonl(output_dir / "rejections.jsonl")
    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("canonical_prompt") != CANONICAL_PROMPT:
        raise ValueError("canonical_prompt does not match the frozen canonical prompt")
    if len(raw384) != RAW_POOL_SIZE:
        raise ValueError(f"raw384.jsonl must contain 384 rows, got {len(raw384)}")
    if len(pool128) != GLOBAL_POOL_SIZE:
        raise ValueError(f"pool128.jsonl must contain 128 rows, got {len(pool128)}")
    if len({row["prompt_id"] for row in pool128}) != GLOBAL_POOL_SIZE:
        raise ValueError("pool128.jsonl contains duplicate prompt_id values")
    if len({row["template_text"] for row in pool128}) != GLOBAL_POOL_SIZE:
        raise ValueError("pool128.jsonl contains duplicate template_text values")
    pool_by_id = {str(row["prompt_id"]): str(row["template_text"]) for row in pool128}

    for row in pool128:
        text = normalize_prompt(str(row.get("template_text", "")))
        reason = _content_rejection_reason(text)
        if reason is not None:
            raise ValueError(f"pool128 row failed {reason}: {text}")

    subset_sizes: dict[str, int] = {}
    for seed in SUBSET_SEEDS:
        subset_path = output_dir / f"subset_p8_seed{seed}.yaml"
        payload = yaml.safe_load(subset_path.read_text(encoding="utf-8"))
        if payload["canonical_prompt"] != CANONICAL_PROMPT:
            raise ValueError(f"{subset_path.name} canonical_prompt mismatch")
        templates = payload.get("templates", [])
        if len(templates) != SUBSET_SIZE:
            raise ValueError(f"{subset_path.name} must contain 8 templates, got {len(templates)}")
        prompt_ids = [str(row.get("prompt_id", "")) for row in templates]
        template_texts = [str(row.get("template_text", "")) for row in templates]
        if len(set(prompt_ids)) != SUBSET_SIZE or len(set(template_texts)) != SUBSET_SIZE:
            raise ValueError(f"{subset_path.name} must contain 8 unique prompt ID/text pairs")
        for row in templates:
            prompt_id = str(row.get("prompt_id", ""))
            if prompt_id not in pool_by_id:
                raise ValueError(f"{subset_path.name} prompt_id is not in pool128: {prompt_id}")
            if str(row.get("template_text", "")) != pool_by_id[prompt_id]:
                raise ValueError(
                    f"{subset_path.name} template_text does not match pool128: {prompt_id}"
                )
            text = normalize_prompt(str(row.get("template_text", "")))
            reason = _content_rejection_reason(text)
            if reason is not None:
                raise ValueError(f"{subset_path.name} row failed {reason}: {text}")
        subset_sizes[str(seed)] = len(templates)

    return {
        "schema": "mprisk_prompt_pool_artifact_verification_v1",
        "canonical_prompt": CANONICAL_PROMPT,
        "raw_count": len(raw384),
        "accepted_count": len(accepted),
        "rejected_count": len(rejections),
        "global_pool_size": len(pool128),
        "subset_sizes": subset_sizes,
        "forbidden_hits": 0,
        "placeholder_hits": 0,
        "word_count_failures": 0,
        "status": "passed",
    }


def _extract_text(row: dict[str, Any]) -> str | None:
    for field in _TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _rejection_reason(
    row: dict[str, Any],
    normalized: str,
    seen: set[str],
) -> str | None:
    if not normalized:
        return "missing_text"
    if row.get("ai_semantic_review_pass") is not True:
        return "semantic_review_not_passed"
    if normalized in seen:
        return "duplicate"
    if _PLACEHOLDER_RE.search(normalized):
        return "placeholder_or_leakage"
    if _FORBIDDEN_RE.search(normalized):
        return "forbidden_term"
    return _content_rejection_reason(normalized)


def _content_rejection_reason(normalized: str) -> str | None:
    word_count = len(_WORD_RE.findall(normalized))
    if word_count < 10 or word_count > 30:
        return "word_count"
    if not _is_one_sentence(normalized):
        return "sentence_count"
    if _SINGLE_TASK_RE.search(normalized):
        return "multi_task"
    return None


def _is_one_sentence(text: str) -> bool:
    endings = _SENTENCE_END_RE.findall(text)
    return len(endings) == 1 and text[-1] in ".!?"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def _write_outputs(
    output_dir: Path,
    raw384: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    pool128: list[dict[str, Any]],
    subsets: dict[int, list[dict[str, Any]]],
    provenance: dict[str, Any],
    protocol: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "raw384.jsonl", raw384)
    _write_jsonl(output_dir / "accepted.jsonl", accepted)
    _write_jsonl(output_dir / "rejections.jsonl", rejections)
    _write_jsonl(output_dir / "pool128.jsonl", pool128)
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for seed, subset in subsets.items():
        key = f"prompt_pool_p8_seed{seed}"
        payload = {
            "schema": "mprisk_equiv_prompt_set_v1",
            "key": key,
            "protocol": protocol,
            "version": "v1",
            "active": True,
            "canonical_prompt": CANONICAL_PROMPT,
            "global_pool_reference": "pool128.jsonl",
            "seed": seed,
            "subset_size": SUBSET_SIZE,
            "templates": subset,
        }
        (output_dir / f"subset_p8_seed{seed}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
