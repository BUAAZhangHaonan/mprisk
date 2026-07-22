from __future__ import annotations

import copy
import hashlib
import json

import pytest

from mprisk.cache.llava_v15_frame_plan import (
    CONTEXT_BUDGET_MODE,
    CONTEXT_BUDGET_SCHEMA,
    FRAME_PLAN_SCHEMA,
    FRAME_PROTOCOL,
    FRAME_SELECTION_SCHEMA,
    SAMPLING_METHOD,
    _uniform_midpoint_indices,
    index_frame_plan,
    load_frame_plan,
    validate_frame_plan,
    write_frame_plan,
)


def _sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _payload() -> dict:
    prompt_ids = [f"p{index}" for index in range(1, 9)]
    prompt_sha = _sha(prompt_ids)
    candidate_by_condition = {
        str(frames): {"M1": 500 * frames, "M12": 500 * frames + 10}
        for frames in range(1, 9)
    }
    candidate_max = {
        key: max(values.values()) for key, values in candidate_by_condition.items()
    }
    selected = 8
    indices = _uniform_midpoint_indices(80, selected)
    sample_id = "sample-1"
    return {
        "schema": FRAME_PLAN_SCHEMA,
        "model_key": "llava_v1_5_7b",
        "family": "llava_v15",
        "context_budget_mode": CONTEXT_BUDGET_MODE,
        "frame_protocol": FRAME_PROTOCOL,
        "sampling_method": SAMPLING_METHOD,
        "max_candidate_frames": 8,
        "max_position_embeddings": 4096,
        "no_truncation": True,
        "prompt_ids": prompt_ids,
        "prompt_ids_sha256": prompt_sha,
        "entries": [
            {
                "sample_id": sample_id,
                "context_budget_contract": {
                    "schema": CONTEXT_BUDGET_SCHEMA,
                    "sample_id": sample_id,
                    "mode": CONTEXT_BUDGET_MODE,
                    "max_position_embeddings": 4096,
                    "max_candidate_frames": 8,
                    "selected_frames": selected,
                    "conditions": ["M1", "M12"],
                    "prompt_set_key": "vt-main-p8",
                    "prompt_ids": prompt_ids,
                    "prompt_ids_sha256": prompt_sha,
                    "candidate_max_token_counts": candidate_max,
                    "candidate_condition_max_token_counts": candidate_by_condition,
                    "selected_max_token_count": candidate_max[str(selected)],
                    "selection_rule": (
                        "largest_f_with_all_p8_m1_m12_tokens_lte_context"
                    ),
                    "no_truncation": True,
                },
                "frame_selection_contract": {
                    "schema": FRAME_SELECTION_SCHEMA,
                    "sample_id": sample_id,
                    "sampling_method": SAMPLING_METHOD,
                    "video_path": "/data/sample.mp4",
                    "source_total_frames": 80,
                    "selected_frames": selected,
                    "frame_indices": indices,
                    "frame_indices_sha256": _sha(indices),
                    "shared_conditions": ["M1", "M12"],
                    "prompt_ids_sha256": prompt_sha,
                },
            }
        ],
    }


def test_frame_plan_round_trip_is_immutable(tmp_path) -> None:
    payload = _payload()
    path = tmp_path / "plan.json"

    write_frame_plan(payload, path)
    assert load_frame_plan(path) == payload
    assert index_frame_plan(payload)["sample-1"]["sample_id"] == "sample-1"
    write_frame_plan(payload, path)

    changed = copy.deepcopy(payload)
    changed["entries"][0]["frame_selection_contract"]["video_path"] = "/other.mp4"
    with pytest.raises(FileExistsError, match="immutable frame plan"):
        write_frame_plan(changed, path)


def test_frame_plan_rejects_nonmaximal_selection() -> None:
    payload = _payload()
    contract = payload["entries"][0]["context_budget_contract"]
    contract["selected_frames"] = 7
    contract["selected_max_token_count"] = contract["candidate_max_token_counts"]["7"]
    frames = payload["entries"][0]["frame_selection_contract"]
    frames["selected_frames"] = 7
    frames["frame_indices"] = _uniform_midpoint_indices(80, 7)
    frames["frame_indices_sha256"] = _sha(frames["frame_indices"])

    with pytest.raises(ValueError, match="not maximal"):
        validate_frame_plan(payload)


def test_frame_plan_requires_overflow_at_selected_plus_one() -> None:
    payload = _payload()
    contract = payload["entries"][0]["context_budget_contract"]
    for condition in ("M1", "M12"):
        contract["candidate_condition_max_token_counts"]["8"][condition] = 4500
    contract["candidate_max_token_counts"]["8"] = 4500
    contract["selected_frames"] = 7
    contract["selected_max_token_count"] = contract["candidate_max_token_counts"]["7"]
    frames = payload["entries"][0]["frame_selection_contract"]
    frames["selected_frames"] = 7
    frames["frame_indices"] = _uniform_midpoint_indices(80, 7)
    frames["frame_indices_sha256"] = _sha(frames["frame_indices"])

    validate_frame_plan(payload)

    contract["candidate_max_token_counts"]["8"] = 4096
    for condition in ("M1", "M12"):
        contract["candidate_condition_max_token_counts"]["8"][condition] = 4096
    with pytest.raises(ValueError, match="not maximal"):
        validate_frame_plan(payload)


@pytest.mark.parametrize("selected", [6, 7])
def test_frame_plan_accepts_observed_source_and_target_frame_counts(selected) -> None:
    payload = _payload()
    contract = payload["entries"][0]["context_budget_contract"]
    for frames in range(selected + 1, 9):
        for condition in ("M1", "M12"):
            contract["candidate_condition_max_token_counts"][str(frames)][
                condition
            ] = 4200 + frames
        contract["candidate_max_token_counts"][str(frames)] = 4200 + frames
    contract["selected_frames"] = selected
    contract["selected_max_token_count"] = contract["candidate_max_token_counts"][
        str(selected)
    ]
    frames = payload["entries"][0]["frame_selection_contract"]
    frames["selected_frames"] = selected
    frames["frame_indices"] = _uniform_midpoint_indices(80, selected)
    frames["frame_indices_sha256"] = _sha(frames["frame_indices"])

    validate_frame_plan(payload)
