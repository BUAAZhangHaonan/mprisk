"""Route state patterns to response policies."""

from __future__ import annotations

from mprisk.state.patterns import StatePattern


POLICY_BY_STATE = {
    StatePattern.CONFUSION: "confusion_cautious",
    StatePattern.CONSENSUS: "consensus_direct",
    StatePattern.BALANCED: "balanced_two_sided",
    StatePattern.DOMINANT: "dominant_with_guardrail",
}


def route_policy(pattern: StatePattern) -> str:
    return POLICY_BY_STATE[pattern]
