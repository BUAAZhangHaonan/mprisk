from __future__ import annotations

from mprisk.state.patterns import StatePattern, StateThresholds, assign_state


def test_assign_state_patterns() -> None:
    thresholds = StateThresholds(kappa=0.2, tau=0.5, delta=0.3)
    assert assign_state(0.3, 1.0, 0.0, thresholds) == StatePattern.CONFUSION
    assert assign_state(0.1, 0.4, 0.0, thresholds) == StatePattern.CONSENSUS
    assert assign_state(0.1, 0.8, 0.1, thresholds) == StatePattern.BALANCED
    assert assign_state(0.1, 0.8, 0.5, thresholds) == StatePattern.DOMINANT
