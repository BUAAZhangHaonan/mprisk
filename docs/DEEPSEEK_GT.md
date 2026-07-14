# DeepSeek GT generation

This stage joins the frozen 162-row GT-eligible manifest to the canonical
archetype dictionary and sends only `archetype`, `trigger_context`, `dialogue`,
and nullable `surface_emotion` to `deepseek-v4-flash`.

The A and C prompts are fixed separately. Thinking is disabled, temperature is
zero, and the response must be exact JSON with only `GT_DESCRIPTION`. Invalid
JSON, extra keys, non-declarative text, or multi-sentence text fails without
repair. Only timeouts, transport failures, 408/409/429, and 5xx responses retry.

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
