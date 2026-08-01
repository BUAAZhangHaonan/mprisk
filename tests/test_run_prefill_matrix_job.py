from __future__ import annotations

import sys
import types

import pytest

from scripts import run_prefill_matrix_job as launcher


def _clear_allocator_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)


def test_allocator_default_is_set_before_torch_use(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_allocator_environment(monkeypatch)
    events: list[object] = []
    original = launcher._configure_cuda_allocator

    def configure() -> dict[str, str]:
        events.append("configure")
        return original()

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: events.append("is_available") or True,
        device_count=lambda: events.append("device_count") or 1,
        set_per_process_memory_fraction=lambda fraction, device: events.append(
            ("set_fraction", fraction, device)
        ),
    )
    fake_batch = types.ModuleType("mprisk.cache.prefill_batch")
    fake_batch.main = lambda args: events.append(("extract", args)) or 17
    monkeypatch.setattr(launcher, "_configure_cuda_allocator", configure)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "mprisk.cache.prefill_batch", fake_batch)

    result = launcher.main(
        ["--gpu-memory-fraction", "0.88", "--", "--manifest", "fixture.jsonl"]
    )

    assert result == 17
    assert events[0] == "configure"
    assert events[1:4] == [
        "is_available",
        "device_count",
        ("set_fraction", 0.88, 0),
    ]
    assert launcher.os.environ["PYTORCH_CUDA_ALLOC_CONF"] == (
        "expandable_segments:True"
    )


def test_allocator_accepts_compatible_existing_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_allocator_environment(monkeypatch)
    value = "backend:native,expandable_segments:True"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", value)

    assert launcher._configure_cuda_allocator() == {
        "PYTORCH_CUDA_ALLOC_CONF": value
    }


@pytest.mark.parametrize(
    "value",
    [
        "expandable_segments:False",
        "backend:native",
        "expandable_segments",
        "expandable_segments:True,expandable_segments:True",
    ],
)
def test_allocator_rejects_missing_false_or_invalid_requirement(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    _clear_allocator_environment(monkeypatch)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", value)

    with pytest.raises(RuntimeError):
        launcher._configure_cuda_allocator()


def test_allocator_rejects_conflicting_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_allocator_environment(monkeypatch)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF",
        "backend:native,expandable_segments:True",
    )

    with pytest.raises(RuntimeError, match="conflicting options"):
        launcher._configure_cuda_allocator()
