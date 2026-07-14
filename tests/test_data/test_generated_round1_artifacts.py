from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "data/frozen/generated_round1_v1"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_generated_round1_frozen_artifacts_match_contract() -> None:
    archive_rows = _read_jsonl(FROZEN / "archive_manifest.jsonl")
    eligible_rows = _read_jsonl(FROZEN / "gt_eligible.jsonl")
    provenance = json.loads((FROZEN / "provenance.json").read_text(encoding="utf-8"))

    assert len(archive_rows) == 652
    assert len(eligible_rows) == 162
    assert Counter(row["source_archive"] for row in archive_rows) == {
        "accept_a_svt": 371,
        "accept_a_va": 52,
        "accept_c_svt": 142,
        "accept_c_va": 87,
    }
    assert Counter(row["source_archive"] for row in eligible_rows) == {
        "accept_a_svt": 64,
        "accept_a_va": 8,
        "accept_c_svt": 77,
        "accept_c_va": 13,
    }
    assert Counter(row["context_source"] for row in eligible_rows) == {
        "setting": 126,
        "trigger": 36,
    }
    assert Counter(row["anchor"]["source_kind"] for row in eligible_rows) == {
        "recorded": 74,
        "official_template": 88,
    }
    keys = [(row["source_archive"], row["original_variant_id"]) for row in archive_rows]
    assert len(keys) == len(set(keys)) == 652
    assert len({row["sample_id"] for row in archive_rows}) == 652
    assert provenance["counts"]["total"] == 652
    assert provenance["counts"]["gt_eligible"] == 162
    assert provenance["media"]["large_media_committed"] is False
    assert {
        archive: payload["sha256"] for archive, payload in provenance["source_indexes"].items()
    } == {
        "accept_a_svt": "fab21963b23c44a71f7b5d40892ae2aad8fb5bada997b07ece1a11fbe9312e4f",
        "accept_a_va": "55b3e4f2f3878e4eddfd59b17bc223abbbfc3a1a5a0e29b0dffadcf3568e9de5",
        "accept_c_svt": "abd96cadce15e41f89d1dbc5be1d8b3088ffd12f67d5dfa34c5b6245b04cfe5c",
        "accept_c_va": "5d75f03addac52fe6112a6a3e7cd9c18ff8956b22842abf6df472090158cadaf",
    }
    assert {key: payload["sha256"] for key, payload in provenance["official_sources"].items()} == {
        "archetypes_glm": "0a7b627bd9c564f7b95287385171ec8f5f5481cdcace2816b2da1603be4a829b",
        "c_emotion_variants": "6b9749afa1029471d9f65b26144da0c9d0d2576e0d1cd4b33bba05d2cf47e85e",
    }
    assert {
        (row["source_archive"], row["source_sample_id"])
        for row in archive_rows
        if row["media"]["derivation"] == "ffmpeg_stream_copy_no_audio"
    } == {
        ("accept_a_svt", "S0114"),
        ("accept_a_svt", "S0115"),
        ("accept_a_svt", "S0116"),
    }
    for artifact in provenance["artifacts"].values():
        path = ROOT / artifact["path"]
        assert len(path.read_bytes()) == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
