"""V2 Misread judgment: 3× DeepSeek-V4-flash + V4-pro arbitration.

Pipeline (per sample):
  1. Subject model generates a description from the M12 input (full video + text/audio).
  2. Three independent DeepSeek-V4-flash judgments compare GT_DESCRIPTION vs model output.
  3. If unanimous NON_MISREAD or unanimous MISREAD with all confidences >= threshold,
     the label is final.
  4. Otherwise, DeepSeek-V4-pro makes the final arbitration call.

Output schema: see _FINAL_LABEL below.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

DEFAULT_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_FLASH_MODEL = "deepseek-v4-flash"
DEFAULT_PRO_MODEL = "deepseek-v4-pro"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_REQUEST_TIMEOUT = 90.0

JUDGE_SYSTEM_PROMPT = (
    "You are a strict affective computing judge. Compare the reference description "
    "(the ground-truth emotion description) with the diagnostic description (what the "
    "model under test produced).\n\n"
    "Return MISREAD when the diagnostic is led by surface cues, contradicts the primary "
    "affect, wrongly compresses distinct affects into one, omits a decisive component, "
    "or gives a confidently opposite account of the persons internal state.\n"
    "Return NON_MISREAD when the core affect is compatible, synonymous, a valid "
    "simplification, or accurately hedged.\n"
    "Return UNCERTAIN only when the comparison genuinely cannot decide.\n\n"
    "Return STRICT JSON with exactly these keys: decision (one of MISREAD, NON_MISREAD, "
    "UNCERTAIN), confidence (float in [0,1]), rationale (one short sentence)."
)


class JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["MISREAD", "NON_MISREAD", "UNCERTAIN"]
    confidence: float
    rationale: str

    @field_validator("confidence")
    @classmethod
    def _check_conf(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        return float(v)


@dataclass(frozen=True)
class FlashJudgment:
    judge_model: str
    decision: str
    confidence: float
    rationale: str
    raw_response: str


@dataclass(frozen=True)
class FinalMisreadLabel:
    sample_id: str
    subject_model_key: str
    protocol: str
    final_label: str            # "MISREAD" | "NON_MISREAD"
    arbitrator_used: bool
    flash_decisions: list[str]
    flash_confidences: list[float]
    flash_rationales: list[str]
    pro_decision: str | None
    pro_confidence: float | None
    pro_rationale: str | None
    agreement_ratio: float


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DeepSeekJudgeClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str = DEFAULT_API_URL,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _post(self, *, model: str, gt: str, diag: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _canonical_json({
                        "GT_DESCRIPTION": gt,
                        "DIAGNOSTIC_AFFECT_DESCRIPTION": diag,
                    }),
                },
            ],
            "temperature": DEFAULT_TEMPERATURE,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        try:
            resp = await self.client.post(self.api_url, headers=self.headers, json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RuntimeError(f"transport error: {type(exc).__name__}") from exc
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        body = resp.json()
        choices = body.get("choices") or []
        if len(choices) != 1 or "message" not in choices[0]:
            raise RuntimeError(f"unexpected response shape: {body}")
        content = choices[0]["message"].get("content")
        if not isinstance(content, str):
            raise RuntimeError("content is not a string")
        return content

    async def judge_once(self, *, model: str, gt: str, diag: str) -> FlashJudgment:
        raw = await self._post(model=model, gt=gt, diag=diag)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Some models wrap JSON in ```json fences.
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise RuntimeError(f"non-JSON response: {raw[:200]}") from exc
            data = json.loads(m.group(0))
        parsed = JudgeResponse.model_validate(data)
        return FlashJudgment(
            judge_model=model,
            decision=parsed.decision,
            confidence=parsed.confidence,
            rationale=parsed.rationale,
            raw_response=raw,
        )

    async def judge_with_arbitration(
        self,
        *,
        sample_id: str,
        subject_model_key: str,
        protocol: str,
        gt_description: str,
        diagnostic_description: str,
        flash_model: str = DEFAULT_FLASH_MODEL,
        pro_model: str = DEFAULT_PRO_MODEL,
        n_flash: int = 3,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> FinalMisreadLabel:
        """Run n_flash independent flash judgments; arbitrate with pro if unclear."""
        tasks = [
            self.judge_once(model=flash_model, gt=gt_description, diag=diagnostic_description)
            for _ in range(n_flash)
        ]
        flashes = await asyncio.gather(*tasks, return_exceptions=True)
        ok_flash: list[FlashJudgment] = []
        for f in flashes:
            if isinstance(f, Exception):
                # Treat failed call as UNCERTAIN with 0 confidence.
                ok_flash.append(FlashJudgment(
                    judge_model=flash_model,
                    decision="UNCERTAIN",
                    confidence=0.0,
                    rationale=f"API error: {type(f).__name__}: {f}",
                    raw_response="",
                ))
            else:
                ok_flash.append(f)

        decisions = [j.decision for j in ok_flash]
        confs = [j.confidence for j in ok_flash]
        rationales = [j.rationale for j in ok_flash]
        decision_counts = Counter(decisions)

        # Unanimous + all confident -> final
        top_decision, top_count = decision_counts.most_common(1)[0]
        all_confident = all(c >= confidence_threshold for c in confs)
        unanimous_strong = (
            top_count == n_flash
            and top_decision in {"MISREAD", "NON_MISREAD"}
            and all_confident
        )
        majority_strong = (
            top_count >= 2
            and top_decision in {"MISREAD", "NON_MISREAD"}
            and all(c >= confidence_threshold for c, d in zip(confs, decisions) if d == top_decision)
        )

        pro_decision = None
        pro_confidence = None
        pro_rationale = None
        arbitrator_used = False
        if unanimous_strong or majority_strong:
            final_label = top_decision
        else:
            # Arbitrate with V4-pro
            arbitrator_used = True
            arb = await self.judge_once(
                model=pro_model,
                gt=gt_description,
                diag=diagnostic_description,
            )
            pro_decision = arb.decision
            pro_confidence = arb.confidence
            pro_rationale = arb.rationale
            if arb.decision in {"MISREAD", "NON_MISREAD"}:
                final_label = arb.decision
            else:
                # Tie-break by majority of all decisions including pro
                all_decisions = decisions + [arb.decision]
                final_label = Counter(all_decisions).most_common(1)[0][0]
                if final_label == "UNCERTAIN":
                    final_label = "MISREAD"  # conservative default for downstream risk

        agreement_ratio = float(top_count / n_flash)
        return FinalMisreadLabel(
            sample_id=sample_id,
            subject_model_key=subject_model_key,
            protocol=protocol,
            final_label=final_label,
            arbitrator_used=arbitrator_used,
            flash_decisions=decisions,
            flash_confidences=confs,
            flash_rationales=rationales,
            pro_decision=pro_decision,
            pro_confidence=pro_confidence,
            pro_rationale=pro_rationale,
            agreement_ratio=agreement_ratio,
        )

    async def close(self) -> None:
        await self.client.aclose()


def label_to_dict(label: FinalMisreadLabel) -> dict[str, Any]:
    return {
        "schema": "mprisk_v2_misread_label_v1",
        "sample_id": label.sample_id,
        "subject_model_key": label.subject_model_key,
        "protocol": label.protocol,
        "final_label": label.final_label,
        "arbitrator_used": label.arbitrator_used,
        "agreement_ratio": label.agreement_ratio,
        "flash": [
            {
                "judge_model": label.judge_model if False else "deepseek-v4-flash",
                "decision": d,
                "confidence": c,
                "rationale": r,
            }
            for d, c, r in zip(
                label.flash_decisions,
                label.flash_confidences,
                label.flash_rationales,
            )
        ],
        "pro_arbitration": (
            None
            if label.pro_decision is None
            else {
                "judge_model": "deepseek-v4-pro",
                "decision": label.pro_decision,
                "confidence": label.pro_confidence,
                "rationale": label.pro_rationale,
            }
        ),
    }


async def judge_many(
    *,
    tasks: list[dict[str, Any]],
    output_path: str | Path,
    max_concurrency: int = 8,
    flash_model: str = DEFAULT_FLASH_MODEL,
    pro_model: str = DEFAULT_PRO_MODEL,
    n_flash: int = 3,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    api_key: str | None = None,
) -> Path:
    """Run judgment on many samples concurrently. Each task is a dict with:
       sample_id, subject_model_key, protocol, gt_description, diagnostic_description.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = DeepSeekJudgeClient(api_key=api_key)
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(t: dict[str, Any]) -> dict[str, Any] | None:
        async with sem:
            try:
                lbl = await client.judge_with_arbitration(
                    sample_id=t["sample_id"],
                    subject_model_key=t["subject_model_key"],
                    protocol=t["protocol"],
                    gt_description=t["gt_description"],
                    diagnostic_description=t["diagnostic_description"],
                    flash_model=flash_model,
                    pro_model=pro_model,
                    n_flash=n_flash,
                    confidence_threshold=confidence_threshold,
                )
                return label_to_dict(lbl)
            except Exception as exc:
                return {
                    "schema": "mprisk_v2_misread_label_v1",
                    "sample_id": t["sample_id"],
                    "subject_model_key": t["subject_model_key"],
                    "protocol": t["protocol"],
                    "final_label": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }

    try:
        with output_path.open("w", encoding="utf-8") as out_f:
            n = len(tasks)
            done = 0
            for fut in asyncio.as_completed([_one(t) for t in tasks]):
                row = await fut
                if row is not None:
                    out_f.write(json.dumps(row, sort_keys=True) + "\n")
                    out_f.flush()
                done += 1
                if done % 50 == 0 or done == n:
                    print(f"[v2-misread] judged {done}/{n}", flush=True)
    finally:
        await client.close()
    return output_path
