from __future__ import annotations

import pytest

from mprisk.data.protocol_views import expand_protocol_views, normalize_protocol


def test_normalize_protocol_accepts_upper_and_lower_case() -> None:
    assert normalize_protocol("VT") == "VT"
    assert normalize_protocol("va") == "VA"
    assert normalize_protocol("it") == "IT"


def test_expand_protocol_views_adds_canonical_view_metadata_and_media_paths() -> None:
    expanded = expand_protocol_views(
        "vt",
        views={
            "M1": {"label": "positive"},
            "M2": {"label": "negative"},
            "M12": {"label": "negative", "modality": "custom_joint"},
        },
        media_paths={"vision": "clip.mp4", "text": "caption.txt"},
    )

    assert expanded["M1"] == {
        "view": "M1",
        "protocol": "VT",
        "modality": "vision",
        "label": "positive",
        "media_paths": {"vision": "clip.mp4"},
    }
    assert expanded["M2"]["modality"] == "text"
    assert expanded["M2"]["media_paths"] == {"text": "caption.txt"}
    assert expanded["M12"]["modality"] == "custom_joint"
    assert expanded["M12"]["media_paths"] == {"vision": "clip.mp4", "text": "caption.txt"}


def test_expand_protocol_views_requires_m1_m2_m12() -> None:
    with pytest.raises(ValueError, match="M12"):
        expand_protocol_views("VA", views={"M1": {}, "M2": {}})
