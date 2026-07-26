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


def test_sdr_hinge_loss_imports():
    """The SDR-aware hinge loss is now inlined into mainline losses.

    Previously this smoke test imported ``mprisk.representation.sdr_loss``
    (a separate viz module that monkey-patched
    ``training._batch_loss_and_outputs`` via install_sdr_aware_loss). After
    P2.3 the hinge lives in the mainline losses module as
    ``SphericalSDRHingeLoss`` and the viz pipeline enables it declaratively
    through TrainingConfig.sdr_aux_weight.
    """
    from mprisk.representation.losses import SphericalSDRHingeLoss
    assert SphericalSDRHingeLoss.__name__ == "SphericalSDRHingeLoss"


def test_pipeline_no_install_legacy_pipeline_patches():
    """install_v2_pipeline_patches must be gone (legacy).

    All four legacy monkey-patches (BOOTSTRAP_REPLICATES, encoder factory,
    SDR hinge, strict shape) are now expressed declaratively through
    function arguments or TrainingConfig fields, so the install entry
    point is removed entirely.
    """
    import mprisk.pipeline as pipeline
    assert not hasattr(pipeline, "install_v2_pipeline_patches"), (
        "mprisk.pipeline still exposes install_v2_pipeline_patches; the "
        "monkey-patch entry point must be removed now that all v2 patches "
        "are inlined."
    )
    assert not hasattr(pipeline, "_V2_PATCHES_INSTALLED"), (
        "mprisk.pipeline still exposes _V2_PATCHES_INSTALLED guard."
    )


def test_bilstm_tme_class_imports():
    """The bi-LSTM TME class is now inlined into relation_models.

    Previously this smoke test imported ``mprisk.representation.lstm_tme``
    (a separate module with the viz ``SphericalTME_BiLSTM`` and the
    ``install_v2_tme_factory`` monkey-patch). After P2.2 the class lives
    in the mainline relation_models module as ``SphericalTME_BiLSTM`` and
    the viz pipeline selects it declaratively via encoder_type=\"bilstm\".
    """
    from mprisk.representation.relation_models import (
        LSTMSequentialEncoderBi,
        SphericalTME_BiLSTM,
        TME_ARCHITECTURE_BILSTM_V1,
    )
    assert SphericalTME_BiLSTM.architecture_version == TME_ARCHITECTURE_BILSTM_V1
    assert SphericalTME_BiLSTM.__name__ == "SphericalTME_BiLSTM"
    assert LSTMSequentialEncoderBi.__name__ == "LSTMSequentialEncoderBi"


def test_plotting_imports():
    import mprisk.plotting  # noqa: F401


def test_baselines_imports():
    import mprisk.representation.baselines  # noqa: F401


def test_setup_helper_imports():
    import mprisk.setup_helper  # noqa: F401



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
