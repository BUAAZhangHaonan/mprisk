"""Smoke tests for mprisk_viz package import integrity.

These tests verify that every public module in :mod:`mprisk_viz` imports
cleanly and that previously-fixed typos (e.g. ``CONFLICT_COLOR`` instead
of the original ``CONFICT_COLOR``) do not regress.
"""

from __future__ import annotations


def test_pipeline_imports():
    import mprisk_viz.pipeline  # noqa: F401


def test_misread_imports():
    import mprisk_viz.misread  # noqa: F401


def test_sdr_loss_imports():
    import mprisk_viz.sdr_loss  # noqa: F401


def test_lstm_tme_imports():
    import mprisk_viz.lstm_tme  # noqa: F401


def test_plotting_imports():
    import mprisk_viz.plotting  # noqa: F401


def test_baselines_imports():
    import mprisk_viz.baselines  # noqa: F401


def test_setup_helper_imports():
    import mprisk_viz.setup_helper  # noqa: F401


def test_spherical_norm_imports():
    import mprisk_viz.spherical_norm  # noqa: F401


def test_delivery_manifests_imports():
    import mprisk_viz.delivery_manifests  # noqa: F401


def test_plotting_conflict_color_attribute():
    """Regression guard for the CONFICT_COLOR typo (Round 5 M3 fix).

    The constant was originally misspelled ``CONFICT_COLOR``; it must now
    be exposed as ``CONFLICT_COLOR`` with the legacy name removed so
    future typos fail loudly at import time rather than silently falling
    back to a NameError at runtime.
    """
    from mprisk_viz import plotting
    assert hasattr(plotting, "CONFLICT_COLOR")
    assert isinstance(plotting.CONFLICT_COLOR, str)
    assert plotting.CONFLICT_COLOR.startswith("#")
    # Legacy misspelling must NOT survive.
    assert not hasattr(plotting, "CONFICT_COLOR"), (
        "plotting module still exposes the legacy misspelled CONFICT_COLOR; "
        "rename to CONFLICT_COLOR everywhere."
    )
