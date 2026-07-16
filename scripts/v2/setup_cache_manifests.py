"""V2 setup: derive prompt_cache + prompt_conditioned_cache manifests from cache.

Reads:
  - cache_root/manifest.jsonl (the cache ledger)
  - prompt_set YAML (8 prompt IDs)

Writes:
  - output_root/prompt_cache/manifest.jsonl
  - output_root/prompt_conditioned/<model>/<protocol>/<prompt_set_key>/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
SCRIPTS = HERE.parent.parent / "scripts"
for p in [SRC, SCRIPTS]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mprisk.cache.prompt_cache import write_prompt_cache_manifest
from mprisk.cache.prompt_conditioned_cache import (
    prompt_conditioned_entry_from_row,
    write_prompt_conditioned_manifest,
)
from mprisk.prompts.prompt_cache_builder import build_prompt_cache_manifest_row
from mprisk.prompts.template_bank import PromptTemplate
from mprisk.utils.io import write_json


def _load_prompt_templates(prompt_set_path: str | Path) -> tuple[str, str, list[PromptTemplate]]:
    with open(prompt_set_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    key = d["key"]
    protocol = d["protocol"]
    templates = []
    for t in d["templates"]:
        if not t.get("enabled", True):
            continue
        templates.append(PromptTemplate(
            prompt_id=t["prompt_id"],
            role=t.get("role", "user"),
            template_text=t["template_text"],
        ))
    return key, protocol, templates


def setup_v2_cache_manifests(
    *,
    cache_root: str | Path,
    prompt_set_path: str | Path,
    model_key: str,
    output_root: str | Path,
) -> dict:
    cache_root = Path(cache_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    prompt_set_key, protocol, templates = _load_prompt_templates(prompt_set_path)
    print(f"[v2-setup] prompt_set={prompt_set_key} protocol={protocol} "
          f"n_prompts={len(templates)}", flush=True)

    pc_rows = [
        build_prompt_cache_manifest_row(
            model_key=model_key,
            prompt_set_key=prompt_set_key,
            protocol=protocol,
            template=t,
        )
        for t in templates
    ]
    pc_path = output_root / "prompt_cache_manifest.jsonl"
    write_prompt_cache_manifest(pc_path, pc_rows)
    print(f"[v2-setup] wrote prompt_cache_manifest: {pc_path}", flush=True)

    source_manifest = cache_root / "manifest.jsonl"
    if not source_manifest.exists():
        raise FileNotFoundError(f"cache manifest not found: {source_manifest}")

    entries = []
    seen_prompt_ids = set()
    skipped = 0
    selected = 0
    with open(source_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("model_key") != model_key:
                skipped += 1
                continue
            if row.get("prompt_set_key") != prompt_set_key:
                skipped += 1
                continue
            if row.get("protocol") != protocol:
                skipped += 1
                continue
            try:
                entry = prompt_conditioned_entry_from_row(row)
                entries.append(entry)
                selected += 1
                seen_prompt_ids.add(row.get("prompt_id"))
            except (TypeError, ValueError) as exc:
                skipped += 1
                continue

    print(f"[v2-setup] prompt_conditioned: selected={selected} skipped={skipped} "
          f"seen_prompts={len(seen_prompt_ids)}/{len(templates)}", flush=True)

    pccond_dir = (
        output_root / "prompt_conditioned_cache" / model_key / protocol / prompt_set_key
    )
    pccond_dir.mkdir(parents=True, exist_ok=True)
    pccond_path = write_prompt_conditioned_manifest(pccond_dir / "manifest.jsonl", entries)
    summary_path = write_json(
        pccond_dir / "summary.json",
        {
            "cache_root": str(cache_root),
            "model_key": model_key,
            "protocol": protocol,
            "prompt_set_key": prompt_set_key,
            "selected_rows": selected,
            "skipped_rows": skipped,
            "expected_prompt_ids": [t.prompt_id for t in templates],
            "seen_prompt_ids": sorted(seen_prompt_ids),
            "manifest_path": str(pccond_path),
        },
    )
    return {
        "prompt_cache_manifest": str(pc_path),
        "prompt_conditioned_cache_manifest": str(pccond_path),
        "prompt_conditioned_summary": str(summary_path),
        "selected_rows": selected,
        "seen_prompt_ids": sorted(seen_prompt_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = setup_v2_cache_manifests(
        cache_root=args.cache_root,
        prompt_set_path=args.prompt_set,
        model_key=args.model_key,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
