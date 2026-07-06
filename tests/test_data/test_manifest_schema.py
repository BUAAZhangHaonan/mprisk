from __future__ import annotations

from mprisk.data.manifests import read_jsonl


def test_processed_manifest_placeholders_are_readable() -> None:
    rows = read_jsonl("data/processed/manifests/unified_sample_manifest.jsonl")
    assert rows[0]["schema"] == "mprisk_sample_manifest_v1"
