from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from mprisk.judge.misread_judgment import (
    MISREAD_JUDGMENT_PROMPT,
    MisreadJudgeConfig,
    MisreadJudgeLedger,
    MisreadJudgmentValidationError,
    build_tasks,
    export_final_labels,
    import_human_decisions,
    load_api_key,
    run_misread_judgment,
    validate_misread_judgment_response,
    verify_misread_judgment_artifacts,
)


def _write_inputs(tmp_path: Path, count: int = 8) -> tuple[Path, Path]:
    references = tmp_path / "reference.jsonl"
    diagnostics = tmp_path / "diagnostics.jsonl"
    reference_rows = []
    diagnostic_rows = []
    for index in range(count):
        sample_id = f"sample-{index:03d}"
        reference_rows.append({"sample_id": sample_id, "GT_DESCRIPTION": f"Reference {index}."})
        diagnostic_rows.append(
            {
                "sample_id": sample_id,
                "DIAGNOSTIC_AFFECT_DESCRIPTION": f"Diagnostic description {index}.",
                "subject_model_key": "subject_model",
                "protocol": "VT",
                "split": "test",
            }
        )
    references.write_text("".join(json.dumps(row) + "\n" for row in reference_rows))
    diagnostics.write_text("".join(json.dumps(row) + "\n" for row in diagnostic_rows))
    return references, diagnostics


def _config(tmp_path: Path, references: Path, diagnostics: Path) -> MisreadJudgeConfig:
    return MisreadJudgeConfig(
        schema_name="mprisk_misread_judgment_config_v2",
        run_id="misread-judgment-test-v2",
        status="ready",
        judge_model="deepseek-v4-flash",
        subject_model_key="subject_model",
        protocol="VT",
        split="test",
        api_url="https://example.invalid/chat/completions",
        temperature=0,
        confidence_threshold=0.5,
        gt_description_manifest_path=references,
        diagnostic_affect_description_manifest_path=diagnostics,
        output_root=tmp_path / "output",
        request_timeout_seconds=30.0,
    )


def test_blind_payload_contains_only_two_descriptions_and_fixed_protocol(tmp_path: Path) -> None:
    references, diagnostics = _write_inputs(tmp_path)
    tasks = build_tasks(_config(tmp_path, references, diagnostics))

    request = tasks[0].request
    assert request["messages"][0]["content"] == MISREAD_JUDGMENT_PROMPT
    payload = json.loads(request["messages"][1]["content"])
    assert set(payload) == {"GT_DESCRIPTION", "DIAGNOSTIC_AFFECT_DESCRIPTION"}
    encoded = json.dumps(request).lower()
    for forbidden in ("sample-", "archetype", "trigger", "dialogue", "surface_emotion"):
        assert forbidden not in encoded
    assert not re.search(r"\\b(?:vt|va)\\b", encoded)
    assert request["model"] == "deepseek-v4-flash"
    assert request["temperature"] == 0


def test_strict_response_parser_rejects_repairable_invalid_values() -> None:
    valid = validate_misread_judgment_response(
        json.dumps(
            {
                "decision": "UNCERTAIN",
                "confidence": 0.5,
                "rationale": "The reference does not settle the comparison.",
            }
        )
    )
    assert valid["decision"] == "UNCERTAIN"
    assert (
        validate_misread_judgment_response(
            json.dumps(
                {
                    "decision": "MISREAD",
                    "confidence": 0,
                    "rationale": "The comparison supports this judgment.",
                }
            )
        )["confidence"]
        == 0
    )
    assert (
        validate_misread_judgment_response(
            json.dumps(
                {
                    "decision": "NON_MISREAD",
                    "confidence": 1,
                    "rationale": "The comparison supports this judgment.",
                }
            )
        )["confidence"]
        == 1
    )
    for invalid in (
        "not json",
        json.dumps({"decision": "MISREAD", "confidence": 0.5}),
        json.dumps(
            {
                "decision": "MISREAD",
                "confidence": 0.5,
                "rationale": "One sentence.",
                "extra": True,
            }
        ),
        json.dumps({"decision": "MAYBE", "confidence": 0.5, "rationale": "One sentence."}),
        json.dumps({"decision": "MISREAD", "confidence": 1.2, "rationale": "One sentence."}),
        json.dumps({"decision": "MISREAD", "confidence": True, "rationale": "One sentence."}),
        json.dumps(
            {"decision": "MISREAD", "confidence": float("nan"), "rationale": "One sentence."}
        ),
        json.dumps(
            {"decision": "MISREAD", "confidence": float("inf"), "rationale": "One sentence."}
        ),
        json.dumps({"decision": "MISREAD", "confidence": 0.5, "rationale": "First. Second."}),
    ):
        with pytest.raises(MisreadJudgmentValidationError):
            validate_misread_judgment_response(invalid)


def test_only_deepseek_api_key_is_accepted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("GLM_API_KEY", "forbidden")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_api_key(_config(tmp_path, *_write_inputs(tmp_path)))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "accepted")
    assert load_api_key(_config(tmp_path, *_write_inputs(tmp_path))) == "accepted"


def test_missing_gt_description_fails_before_any_client_call(tmp_path: Path) -> None:
    references, diagnostics = _write_inputs(tmp_path)
    rows = [json.loads(line) for line in references.read_text().splitlines()]
    rows[0].pop("GT_DESCRIPTION")
    references.write_text("".join(json.dumps(row) + "\n" for row in rows))

    class Client:
        calls = 0

        async def complete(self, task):
            self.calls += 1
            raise AssertionError("client must not be called")

    client = Client()
    with pytest.raises(ValueError, match="GT_DESCRIPTION"):
        asyncio.run(
            run_misread_judgment(
                config=_config(tmp_path, references, diagnostics), client=client
            )
        )
    assert client.calls == 0


def test_ledger_rejects_signature_mismatch_and_exports_atomic_outputs(tmp_path: Path) -> None:
    references, diagnostics = _write_inputs(tmp_path)
    config = _config(tmp_path, references, diagnostics)
    ledger = MisreadJudgeLedger(config.output_root / "batch_state.sqlite3")
    ledger.prepare({"signature": "one"})
    with pytest.raises(ValueError, match="signature"):
        ledger.prepare({"signature": "two"})
    ledger.close()


def test_queue_manual_import_and_final_labels(tmp_path: Path) -> None:
    references, diagnostics = _write_inputs(tmp_path)
    config = _config(tmp_path, references, diagnostics)

    class Client:
        async def complete(self, task):
            if task.sample_id == "sample-000":
                decision, confidence = "UNCERTAIN", 1.0
            elif task.sample_id == "sample-001":
                decision, confidence = "NON_MISREAD", 0.5
            elif task.sample_id == "sample-002":
                decision, confidence = "MISREAD", 0.49
            else:
                decision, confidence = "NON_MISREAD", 0.9
            return json.dumps(
                {
                    "decision": decision,
                    "confidence": confidence,
                    "rationale": "The comparison supports this judgment.",
                }
            )

        async def close(self):
            return None

    result = asyncio.run(run_misread_judgment(config=config, client=Client()))
    assert result["completed"] == 8
    verified = verify_misread_judgment_artifacts(config, require_complete=True)
    assert verified["queue_count"] == 2
    decisions = tmp_path / "human.jsonl"
    decisions.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"sample_id": "sample-000", "final_decision": "MISREAD"},
                {"sample_id": "sample-002", "final_decision": "NON_MISREAD"},
            )
        )
    )
    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text(
        json.dumps({"sample_id": "sample-000", "final_decision": "MISREAD"}) + "\n"
    )
    with pytest.raises(ValueError, match="exactly cover"):
        import_human_decisions(config, incomplete)
    extra = tmp_path / "extra.jsonl"
    extra.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"sample_id": "sample-000", "final_decision": "MISREAD"},
                {"sample_id": "sample-002", "final_decision": "NON_MISREAD"},
                {"sample_id": "sample-003", "final_decision": "MISREAD"},
            )
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="exactly cover"):
        import_human_decisions(config, extra)
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"sample_id": "sample-000", "final_decision": "MISREAD"},
                {"sample_id": "sample-000", "final_decision": "MISREAD"},
            )
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="invalid or duplicate"):
        import_human_decisions(config, duplicate)
    uncertain = tmp_path / "uncertain.jsonl"
    uncertain.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"sample_id": "sample-000", "final_decision": "UNCERTAIN"},
                {"sample_id": "sample-002", "final_decision": "NON_MISREAD"},
            )
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="invalid or duplicate"):
        import_human_decisions(config, uncertain)
    (config.output_root / "human_decisions.jsonl").write_text(incomplete.read_text())
    with pytest.raises(ValueError, match="exactly cover"):
        export_final_labels(config)
    import_human_decisions(config, decisions)
    final = export_final_labels(config)
    assert len(final) == 8
    assert final[0]["binary_label"] == 1


def test_empty_review_queue_requires_an_empty_human_decision_set(tmp_path: Path) -> None:
    references, diagnostics = _write_inputs(tmp_path)
    config = _config(tmp_path, references, diagnostics)

    class Client:
        async def complete(self, task):
            return json.dumps(
                {
                    "decision": "NON_MISREAD",
                    "confidence": 0.5,
                    "rationale": "The comparison supports this judgment.",
                }
            )

        async def close(self):
            return None

    asyncio.run(run_misread_judgment(config=config, client=Client()))
    assert verify_misread_judgment_artifacts(config, require_complete=True)["queue_count"] == 0
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    import_human_decisions(config, empty)
    assert len(export_final_labels(config)) == 8


def test_explicit_retry_retries_failures_without_repeating_completed_records(
    tmp_path: Path,
) -> None:
    references, diagnostics = _write_inputs(tmp_path)
    config = _config(tmp_path, references, diagnostics)

    class Client:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.failed = False

        async def complete(self, task):
            self.calls.append(task.sample_id)
            if task.sample_id == "sample-000" and not self.failed:
                self.failed = True
                return "invalid"
            return json.dumps(
                {
                    "decision": "NON_MISREAD",
                    "confidence": 0.9,
                    "rationale": "The comparison supports this judgment.",
                }
            )

    client = Client()
    first = asyncio.run(run_misread_judgment(config=config, client=client))
    assert (first["completed"], first["failed"]) == (7, 1)
    second = asyncio.run(
        run_misread_judgment(config=config, client=client, retry_failed=True)
    )
    assert (second["completed"], second["failed"]) == (8, 0)
    assert client.calls.count("sample-000") == 2
    assert len(client.calls) == 9


def test_pending_config_cannot_start_judgment(tmp_path: Path) -> None:
    references, diagnostics = _write_inputs(tmp_path)
    config = _config(tmp_path, references, diagnostics).model_copy(
        update={"status": "pending"}
    )
    with pytest.raises(ValueError, match="pending required manifests"):
        asyncio.run(run_misread_judgment(config=config))


def test_legacy_text_field_is_rejected(tmp_path: Path) -> None:
    references, diagnostics = _write_inputs(tmp_path)
    rows = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    rows[0]["text"] = rows[0].pop("DIAGNOSTIC_AFFECT_DESCRIPTION")
    diagnostics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="DIAGNOSTIC_AFFECT_DESCRIPTION"):
        build_tasks(_config(tmp_path, references, diagnostics))
