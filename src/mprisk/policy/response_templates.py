"""Response template placeholders."""

from __future__ import annotations


def format_policy_context(policy_name: str, evidence: str) -> str:
    return f"Policy: {policy_name}\nEvidence: {evidence}"
