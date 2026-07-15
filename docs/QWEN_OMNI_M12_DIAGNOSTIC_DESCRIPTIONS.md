# Qwen2.5-Omni M12 Diagnostic Affect Descriptions

`scripts/generate_diagnostic_descriptions.py` produces one greedy, Thinker-only M12
description for each frozen strict-eligible generated sample. The canonical prompt is fixed in
`mprisk.diagnostic_descriptions.qwen_omni_m12`. VT receives silent video plus dialogue and the
canonical prompt with `use_audio_in_video=False`. VA receives video plus its original embedded
audio and the canonical prompt with `use_audio_in_video=True`; VA never receives dialogue.

The CLI records request/input/media/prompt/model/config hashes, attempts, failures, raw newly
generated token IDs, EOS metadata, and raw decoded text in a SQLite resume ledger. A changed
signature is a hard failure. The exported `manifest.jsonl`, `summary.json`, `failures.jsonl`, and
`provenance.json` are atomic. The SQLite database and WAL files are runtime state and are not
committed.

Run GPU 1 through one tmux session by exposing it as process-local `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
  /home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python \
  scripts/generate_diagnostic_descriptions.py
```

Smoke runs use the same formal CLI with one VT and one VA `--sample-id`, but must use an isolated
`--output-root`. `--smoke` deterministically selects exactly one VT and one VA input; explicit
sample IDs are accepted only with `--smoke` and must still contain exactly one of each protocol.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
  /home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python \
  scripts/generate_diagnostic_descriptions.py --smoke \
  --output-root outputs/diagnostics/qwen_omni_m12_smoke_v1
PYTHONNOUSERSITE=1 /home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python \
  scripts/verify_diagnostic_descriptions.py \
  --eligible-path data/frozen/generated_round1_v1/gt_eligible.jsonl \
  --output-root outputs/diagnostics/qwen_omni_m12_smoke_v1 --smoke
```

The configuration fixes the model path, `max_new_tokens`, video sampling rate, and SDPA setting.
The generation request admits only `do_sample=False`, `num_beams=1`, and that fixed token budget;
it never supplies temperature or top-p. The verifier requires exactly VT141/VA21 in full mode, or
one VT plus one VA in smoke mode. It checks signatures, output field shape, output sentence form,
and hashes for the atomically written manifest, failure, attempt, and summary artifacts.

## Recorded Smoke Evidence

The formal CLI smoke run on 2026-07-15 used `gen:accept_a_svt:S0001` and
`gen:accept_a_va:S0006`. It completed both records with `finish_reason="eos"` and tokenizer EOS
`151645`; the verifier passed with manifest SHA-256
`0059120b879fb2a6cd6da4f78f92a03331647e16853d4440d0c787d9841ace9f`.

- VT: `The person appears to be in a relaxed and content state.`
  Token IDs: `785,1697,7952,311,387,304,264,30367,323,2213,1584,13,151645`.
- VA: `The person appears to be feeling anxious.`
  Token IDs: `785,1697,7952,311,387,8266,37000,13,151645`.

## Frozen Full Run

The formal GPU 1 run completed on 2026-07-15 with 162 successful records: VT141 and VA21.
Every record finished on EOS and has non-empty generated token IDs. The generation-time sum was
91.256 seconds, the peak allocated GPU memory was `18318550016` bytes, and the frozen manifest
SHA-256 is `1c178cb7acfb80359ecad01f893c094f442179434f9f0a92cc8bca98e9e9083a`.

The tracked full-run artifacts are in
`outputs/diagnostics/qwen2_5_omni_7b_m12_v1/`. Runtime SQLite state, WAL files, and the tmux log
remain untracked.
