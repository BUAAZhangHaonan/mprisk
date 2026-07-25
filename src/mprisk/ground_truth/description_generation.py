"""Provider-independent, strict, resumable GT Description generation."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path

from mprisk.ground_truth.providers.base import (
    GTDescriptionProvider,
    GTDescriptionProviderRequest,
    GTDescriptionProviderResponse,
    PermanentProviderError,
    TransientProviderError,
)
from mprisk.ground_truth.providers.registry import get_provider
from mprisk.utils.io import now_iso as _now

# Public re-exports -- keep historical import path stable.
from mprisk.ground_truth._ledger import GTDescriptionGenerationLedger
from mprisk.ground_truth._materialize import _export
from mprisk.ground_truth._plan import (
    CONFIG_SCHEMA,
    OUTPUT_SCHEMA,
    PROVENANCE_SCHEMA,
    GTDescriptionGenerationConfig,
    GTDescriptionGenerationResult,
    GTDescriptionGenerationTask,
    GTDescriptionValidationError,
)
from mprisk.ground_truth._planner import (
    _resolve_repo_path,
    _validate_model_input,
    load_config,
    prepare_tasks,
)
from mprisk.ground_truth._verifier import (
    _text,
    validate_gt_description_content,
    verify_gt_description_generation,
)

__all__ = [
    "CONFIG_SCHEMA",
    "GTDescriptionGenerationConfig",
    "GTDescriptionGenerationLedger",
    "GTDescriptionGenerationResult",
    "GTDescriptionGenerationTask",
    "GTDescriptionValidationError",
    "OUTPUT_SCHEMA",
    "PROVENANCE_SCHEMA",
    "_export",
    "_resolve_repo_path",
    "_text",
    "_validate_model_input",
    "load_config",
    "prepare_tasks",
    "run_gt_description_generation",
    "validate_gt_description_content",
    "verify_gt_description_generation",
]


async def run_gt_description_generation(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    retry_failed: bool = False,
    provider: GTDescriptionProvider | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> GTDescriptionGenerationResult:
    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_config(config_file)
    tasks = prepare_tasks(root, config)
    task_by_id = {task.sample_id: task for task in tasks}
    output_root = _resolve_repo_path(root, config.output_root)
    ledger = GTDescriptionGenerationLedger(output_root / "batch_state.sqlite3")
    ledger.prepare(tasks)
    owns_provider = provider is None
    if provider is None:
        provider = get_provider(
            config.provider_key,
            config.gt_generator_model,
            config.provider_settings,
        )
    semaphore = asyncio.Semaphore(config.concurrency)

    async def worker(sample_id: str) -> None:
        task = task_by_id[sample_id]
        async with semaphore:
            last_error: Exception | None = None
            for retry_index in range(len(config.retry_delays_seconds) + 1):
                attempt = ledger.start(sample_id)
                started = _now()
                response: GTDescriptionProviderResponse | None = None
                try:
                    request = GTDescriptionProviderRequest(
                        model=config.gt_generator_model,
                        system_prompt=task.system_prompt,
                        model_input=task.model_input,
                    )
                    response = await provider.complete(request)
                    description = validate_gt_description_content(
                        response.content,
                        min_words=config.min_words,
                        max_words=config.max_words,
                    )
                    result = {**asdict(response), "GT_DESCRIPTION": description}
                    ledger.finish_attempt(sample_id, attempt, started, "completed", result)
                    ledger.complete(sample_id, result)
                    return
                except TransientProviderError as exc:
                    last_error = exc
                    ledger.finish_attempt(
                        sample_id, attempt, started, "transient_error", exc=exc
                    )
                    if retry_index < len(config.retry_delays_seconds):
                        await sleep(config.retry_delays_seconds[retry_index])
                        continue
                    break
                except PermanentProviderError as exc:
                    last_error = exc
                    ledger.finish_attempt(
                        sample_id,
                        attempt,
                        started,
                        "failed",
                        response=None if response is None else asdict(response),
                        exc=exc,
                    )
                    break
                except GTDescriptionValidationError as exc:
                    last_error = exc
                    ledger.finish_attempt(
                        sample_id,
                        attempt,
                        started,
                        "failed",
                        response=None if response is None else asdict(response),
                        exc=exc,
                    )
                    break
            if last_error is None:
                raise RuntimeError(f"GT task ended without a result: {sample_id}")
            ledger.fail(sample_id, last_error)

    try:
        runnable_ids = ledger.pending_ids(include_failed=retry_failed)
        await asyncio.gather(*(worker(sample_id) for sample_id in runnable_ids))
        _export(output_root, ledger, config, config_file)
        counts = Counter(row["status"] for row in ledger.rows())
        return GTDescriptionGenerationResult(
            total=len(ledger.rows()),
            completed=counts["completed"],
            failed=counts["failed"],
            pending=counts["pending"],
            output_root=output_root,
        )
    finally:
        if owns_provider:
            await provider.close()
        ledger.close()
