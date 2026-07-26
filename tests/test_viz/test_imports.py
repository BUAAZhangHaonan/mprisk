"""Smoke tests for mprisk package import integrity.

These tests verify that every public module in :mod:`mprisk` (relocated viz modules) imports
cleanly and that previously-fixed typos (e.g. ``CONFLICT_COLOR`` instead
of the original ``CONFICT_COLOR``) do not regress.
"""

from __future__ import annotations


def test_pipeline_imports():
    import mprisk.pipeline  # noqa: F401


def test_misread_imports():
    import mprisk.misread  # noqa: F401


def test_sdr_loss_imports():
    import mprisk.representation.sdr_loss  # noqa: F401


def test_lstm_tme_imports():
    import mprisk.representation.lstm_tme  # noqa: F401


def test_plotting_imports():
    import mprisk.plotting  # noqa: F401


def test_baselines_imports():
    import mprisk.representation.baselines  # noqa: F401


def test_setup_helper_imports():
    import mprisk.setup_helper  # noqa: F401


def test_spherical_norm_imports():
    import mprisk.state.spherical_norm  # noqa: F401


def test_delivery_manifests_imports():
    import mprisk.data.delivery_manifests  # noqa: F401


def test_plotting_conflict_color_attribute():
    """Regression guard for the CONFICT_COLOR typo (Round 5 M3 fix).

    The constant was originally misspelled ``CONFICT_COLOR``; it must now
    be exposed as ``CONFLICT_COLOR`` with the legacy name removed so
    future typos fail loudly at import time rather than silently falling
    back to a NameError at runtime.
    """
    from mprisk import plotting
    assert hasattr(plotting, "CONFLICT_COLOR")
    assert isinstance(plotting.CONFLICT_COLOR, str)
    assert plotting.CONFLICT_COLOR.startswith("#")
    # Legacy misspelling must NOT survive.
    assert not hasattr(plotting, "CONFICT_COLOR"), (
        "plotting module still exposes the legacy misspelled CONFICT_COLOR; "
        "rename to CONFLICT_COLOR everywhere."
    )
