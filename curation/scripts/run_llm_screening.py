from __future__ import annotations

import argparse
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from curation.scripts.common import read_jsonl, write_jsonl


class ScreeningProvider(ABC):
    @abstractmethod
    def screen_view(self, candidate: dict[str, Any], view: str) -> dict[str, Any]:
        raise NotImplementedError


class MockProvider(ScreeningProvider):
    def screen_view(self, candidate: dict[str, Any], view: str) -> dict[str, Any]:
        label_key = {"M1": "m1_label", "M2": "m2_label", "M12": "joint_label"}[view]
        label = candidate.get(label_key) or "uncertain"
        is_clear = label not in {"uncertain", "invalid", ""}
        return {
            "label": label if label in {"positive", "negative", "neutral", "uncertain", "invalid"} else "uncertain",
            "specific_affect": candidate.get(f"{view.lower()}_specific_affect", ""),
            "is_clear": is_clear,
            "confidence": 0.75 if is_clear else 0.35,
            "evidence": f"{view} mock evidence",
            "quality_flags": [],
        }


class OpenRouterGeminiProvider(ScreeningProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get("OPENROUTER_GEMINI_MODEL", "google/gemini-3.1-pro-preview")
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", base_url)
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL")
        self.app_name = app_name or os.environ.get("OPENROUTER_APP_NAME", "mprisk-curation")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter Gemini screening")

    def screen_view(self, candidate: dict[str, Any], view: str) -> dict[str, Any]:
        prompt = build_prompt(candidate, view)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return compact JSON only. Do not include chain-of-thought."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name
        request = Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed: {exc.code} {body}") from exc
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return normalize_view_output(parsed)


def build_prompt(candidate: dict[str, Any], view: str) -> str:
    return json.dumps(
        {
            "task": "screen multimodal affect sample view",
            "view": view,
            "allowed_labels": ["positive", "negative", "neutral", "uncertain", "invalid"],
            "candidate": candidate,
            "required_output": {
                "label": "positive|negative|neutral|uncertain|invalid",
                "specific_affect": "short label",
                "is_clear": True,
                "confidence": 0.0,
                "evidence": "short phrase only",
                "quality_flags": [],
            },
        },
        ensure_ascii=False,
    )


def normalize_view_output(output: dict[str, Any]) -> dict[str, Any]:
    label = output.get("label", "uncertain")
    if label not in {"positive", "negative", "neutral", "uncertain", "invalid"}:
        label = "uncertain"
    confidence = float(output.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    return {
        "label": label,
        "specific_affect": str(output.get("specific_affect", "")),
        "is_clear": bool(output.get("is_clear", False)),
        "confidence": confidence,
        "evidence": str(output.get("evidence", ""))[:240],
        "quality_flags": list(output.get("quality_flags", [])),
    }


def infer_dominant(view_outputs: dict[str, dict[str, Any]]) -> str:
    joint = view_outputs["M12"]["label"]
    m1 = view_outputs["M1"]["label"]
    m2 = view_outputs["M2"]["label"]
    if joint == m1 and joint != m2:
        return "M1"
    if joint == m2 and joint != m1:
        return "M2"
    if joint == m1 == m2:
        return "balanced"
    return "unclear"


def screen_candidate(candidate: dict[str, Any], provider: ScreeningProvider) -> dict[str, Any]:
    view_outputs = {view: provider.screen_view(candidate, view) for view in ("M1", "M2", "M12")}
    labels = {view: output["label"] for view, output in view_outputs.items()}
    clears = {view: output["is_clear"] for view, output in view_outputs.items()}
    if not all(clears.values()):
        sample_type = "Ambiguous"
    elif labels["M1"] != labels["M2"]:
        sample_type = "Conflict"
    elif labels["M1"] == labels["M2"] == labels["M12"]:
        sample_type = "Aligned"
    else:
        sample_type = "Ambiguous"
    quality_flags = sorted(
        {
            flag
            for output in view_outputs.values()
            for flag in output.get("quality_flags", [])
        }
    )
    return {
        "sample_id": candidate["sample_id"],
        "protocol": candidate.get("protocol", ""),
        "view_outputs": view_outputs,
        "sample_type_suggestion": sample_type,
        "dominant_modality_suggestion": infer_dominant(view_outputs),
        "quality_flags": quality_flags,
        "needs_human_review": True,
    }


def build_provider(name: str) -> ScreeningProvider:
    if name == "mock":
        return MockProvider()
    if name in {"openrouter", "gemini", "openrouter-gemini"}:
        return OpenRouterGeminiProvider()
    raise ValueError(f"Unknown screening provider: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", default=os.environ.get("MPRISK_SCREENING_PROVIDER", "mock"))
    args = parser.parse_args()
    provider = build_provider(args.provider)
    rows = [screen_candidate(candidate, provider) for candidate in read_jsonl(args.input)]
    write_jsonl(Path(args.output), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
