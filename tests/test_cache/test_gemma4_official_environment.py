from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mprisk.assets.registry import index_assets, load_model_assets
from mprisk.cache.cache_matrix_queue import (
    build_model_environment,
    load_matrix_config,
)


def test_gemma4_official_environment_resolves_unified_auto_classes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    matrix = load_matrix_config(
        repo_root / "configs/cache/complete_cache_matrix_20260722.yaml"
    )
    model = next(item for item in matrix.models if item.model_key == "gemma4_12b")
    asset = index_assets(load_model_assets(matrix.asset_config))[model.model_key]
    model_path = asset.local_model_path.resolve()
    environment_root = model.python.parent.parent.resolve()
    code = r"""
import hashlib
import inspect
import json
import site
import sys
from pathlib import Path

import transformers
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

model_path = Path(sys.argv[1])
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
model_class = AutoModelForMultimodalLM._model_mapping[type(config)]

def describe(value):
    source = Path(inspect.getfile(value)).resolve()
    return {
        "name": value.__name__,
        "module": value.__module__,
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }

print(json.dumps({
    "python_no_user_site": bool(sys.flags.no_user_site),
    "site_enable_user_site": bool(site.ENABLE_USER_SITE),
    "sys_executable": str(Path(sys.executable).resolve()),
    "transformers_path": str(Path(transformers.__file__).resolve()),
    "transformers_version": transformers.__version__,
    "config": describe(type(config)),
    "processor": describe(type(processor)),
    "model": describe(model_class),
}, sort_keys=True))
"""

    completed = subprocess.run(
        [str(model.python), "-c", code, str(model_path)],
        cwd=matrix.repo_root,
        env=build_model_environment(matrix, model, model.gpu_lane),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    evidence = json.loads(completed.stdout)

    assert evidence["python_no_user_site"] is True
    assert evidence["site_enable_user_site"] is False
    assert Path(evidence["sys_executable"]).is_relative_to(environment_root)
    assert Path(evidence["transformers_path"]).is_relative_to(environment_root)
    assert "/.local/" not in evidence["transformers_path"]
    assert evidence["transformers_version"] == "5.10.2"
    expected = {
        "config": (
            "Gemma4UnifiedConfig",
            "transformers.models.gemma4_unified.configuration_gemma4_unified",
        ),
        "processor": (
            "Gemma4UnifiedProcessor",
            "transformers.models.gemma4_unified.processing_gemma4_unified",
        ),
        "model": (
            "Gemma4UnifiedForConditionalGeneration",
            "transformers.models.gemma4_unified.modeling_gemma4_unified",
        ),
    }
    for role, (name, module) in expected.items():
        value = evidence[role]
        assert value["name"] == name
        assert value["module"] == module
        assert Path(value["source_path"]).is_relative_to(environment_root)
        assert len(value["source_sha256"]) == 64
