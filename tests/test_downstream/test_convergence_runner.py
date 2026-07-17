from __future__ import annotations

from types import SimpleNamespace

from mprisk.experiments import downstream
from mprisk.representation.training import TrainingConfig


def test_runner_extends_epoch_boundary_until_early_stopping(tmp_path, monkeypatch) -> None:
    config = TrainingConfig(
        repr_key="tme_proxy_anchor_v1",
        model_key="model",
        protocol="vt",
        classification_objective="proxy_anchor_only",
        prompt_set_key="p8",
        prompt_set_artifact_sha256="a" * 64,
        expected_prompt_count=1,
        expected_prompt_ids=("p1",),
        d_supervision_weight=0.2,
        d_ranking_margin=0.25,
        angular_supervision_weight=0.2,
        angular_ranking_margin_rad=0.08726646259971647,
        d_aux_samples_per_class=1,
        max_epochs=5,
    )
    calls = []

    def fake_train(**kwargs):
        calls.append(kwargs["config"].max_epochs)
        last = kwargs["output_dir"] / "last_checkpoint.pt"
        last.parent.mkdir(parents=True, exist_ok=True)
        last.touch()
        reason = "max_epochs" if len(calls) == 1 else "early_stopping"
        return SimpleNamespace(
            metrics={
                "final_epoch": calls[-1],
                "best_epoch": 3,
                "stop_reason": reason,
            },
            last_checkpoint_path=last,
        )

    monkeypatch.setattr(downstream, "train_trajectory_encoder", fake_train)
    result = downstream._train_until_converged(
        dataset_path=tmp_path / "dataset.jsonl",
        config=config,
        output_dir=tmp_path / "training",
        device="cpu",
    )
    assert calls == [5, 10]
    assert result.metrics["stop_reason"] == "early_stopping"
