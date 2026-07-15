# Reference-Guided Misread Judge

`scripts/run_reference_guided_judge.py` compares each frozen `GT_DESCRIPTION` with the matching
Qwen M12 diagnostic description using the fixed `deepseek-v4-flash` model and `temperature: 0`.
The service request is blinded: it contains only those two descriptions and the fixed comparison
protocol. It never includes sample identifiers, source metadata, modality labels, dialogue,
archetypes, triggers, surface labels, or model names.

The judge accepts only exact JSON with `MISREAD`, `NON_MISREAD`, or `UNCERTAIN`, a confidence in
`[0,1]`, and one short rationale sentence. Invalid responses are explicit failures. There is no
key fallback: the only accepted credential is `DEEPSEEK_API_KEY`.

All `UNCERTAIN` results and all confidence values below the versioned `0.85` threshold enter
`human_review_queue.jsonl`. This is a round-one provisional operational threshold, not a
paper-validated threshold. Human decisions must exactly cover that queue and can only be
`MISREAD` or `NON_MISREAD`; only then can `final_binary_labels.jsonl` be exported.

The run first validates that the frozen GT and M12 manifests have exactly the same 162 sample IDs
and that no `GT_DESCRIPTION` is missing. SQLite preserves request hashes, raw responses, attempts,
and failures for resume. JSONL exports and provenance are atomic. Do not commit the runtime SQLite
files, API logs, or credentials.

```bash
DEEPSEEK_API_KEY=... \
  /home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python \
  scripts/run_reference_guided_judge.py
```

Verification does not call the API:

```bash
/home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python \
  scripts/run_reference_guided_judge.py --verify
```
