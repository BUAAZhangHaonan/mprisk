import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_qwen_vl_prefill_generation.py"
SPEC = importlib.util.spec_from_file_location("benchmark_qwen_vl_prefill_generation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_p1_speedup_is_undefined():
    assert MODULE._speedup(100.0, 50.0, 1) is None


def test_p2_speedup_is_ratio():
    assert MODULE._speedup(100.0, 50.0, 2) == 2.0


def test_stats_is_deterministic():
    stats = MODULE._stats([100.0, 200.0, 300.0])
    assert stats["mean_ms"] == 200.0
    assert stats["median_ms"] == 200.0
    assert stats["std_ms"] > 0


def test_schema_and_generation_boundary_are_explicit():
    assert MODULE.SCHEMA.endswith("timing_v1")
    assert "M12" in MODULE.CONDITIONS
