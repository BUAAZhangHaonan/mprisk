# DeepSeek GT generation

This stage joins the frozen 162-row GT-eligible manifest to the canonical
archetype dictionary and sends only `archetype`, `trigger_context`, `dialogue`,
and nullable `surface_emotion` to `deepseek-v4-flash`.

The A and C prompts are fixed separately. Thinking is disabled, temperature is
zero, and the response must be exact JSON with only `GT_DESCRIPTION`. Invalid
JSON, extra keys, non-declarative text, or multi-sentence text fails without
repair. Only timeouts, transport failures, 408/409/429, and 5xx responses retry.
The sole accepted credential is `DEEPSEEK_API_KEY`; no alternate provider key is used.

The SQLite ledger resumes interrupted work. Exports are atomic and keep raw
request/response metadata, attempts, failures, provenance, and a review-status
sidecar. The final manifest copies every eligible field unchanged and adds only
`GT_DESCRIPTION`; all labels remain `preliminary_ai_draft` and `pending_human`
in the sidecar.

```bash
python scripts/run_deepseek_gt.py --mode pilot
python scripts/run_deepseek_gt.py --mode full
python scripts/verify_deepseek_gt.py --require-complete
```

Resume is strict: the default run processes only `pending` rows. A failed row is
never retried implicitly; use `--retry-failed` after reviewing its recorded error.

## Prompt-context v2 pilot

The versioned v2 context resolver records one of three explicit sources in order:
`setting`, then a natural non-template `trigger`, then `source_row.ltx2_prompt`.
The source name is retained in the frozen input manifest and raw prompt text is
never relabeled as a setting or trigger. The deterministic pilot contains two rows for
`class_code.A=sample_type.Conflict` and two rows for
`class_code.C=sample_type.Aligned` in each VT and VA cell, and has its own config,
expected count, manifest hash, output root, and ledger signature. The v1 162-row path is
unchanged.

```bash
python scripts/build_prompt_context_v2_pilot.py
python scripts/run_deepseek_gt.py \
  --config configs/ground_truth/deepseek_gt_prompt_context_v2_pilot.yaml \
  --mode pilot
```

The v2 request sends only `archetype` (including canonical meaning), `dialogue`,
`context`, and nullable `surface_emotion`. `context_source`, protocol, media,
assignments, labels, and future model outputs remain outside the request body.
