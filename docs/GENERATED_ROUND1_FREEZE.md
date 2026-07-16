# Generated round-one freeze

`generated_round1_v1` is an immutable snapshot of all 652 accepted generated
records. It is separate from `delivery_20260714`; the previous 4,754-row delivery
and its 529 generated rows are not changed.

The source archives are:

| Source archive | Rows | Protocol | GT eligible |
| --- | ---: | --- | ---: |
| `accept_a_svt` | 371 | VT | 64 |
| `accept_a_va` | 52 | VA | 8 |
| `accept_c_svt` | 142 | VT | 77 |
| `accept_c_va` | 87 | VA | 13 |

The immutable primary key is `(source_archive, original_variant_id)`. The archive
manifest stores the complete source row, source-line digest, source-index digest,
and hashes for both the recorded primary media and the actual model-input media.
It references media by absolute path; media files are not committed to Git.
For five source rows whose `files.primary` already names a verified
`*.silent.mp4` while `files.silent` is null, that recorded primary is the strict
model input and is marked `recorded_primary_silent`.

## GT-eligible boundary

A row is eligible only when all three contracts hold:

1. `dialogue_text` is a non-empty string.
2. Context is the original non-empty `setting`; otherwise it is a non-empty
   `trigger` that is not the template code `T1`, `T2`, `T3`, or `T4`.
3. The row has a complete recorded archetype anchor, or a C row matches exactly
   one official prompt-make C template by `(gt_emotion, setting, dialogue_text)`
   and exactly one C emotion entry in `ARCHETYPES_GLM`.

No context is recovered from `ltx2_prompt`, and no fuzzy matching is allowed.
The resulting artifact has 162 rows: 64/8/77/13 in the source order above.

## Deterministic silent media

`accept_a_svt` records `S0114`, `S0115`, and `S0116` only have their recorded
audio-bearing primary file. The builder creates a repository-external silent copy
with FFmpeg video stream copy (`-c:v copy -an`), removes metadata, and verifies
with FFprobe that the result has a video stream and no audio stream. A rerun
regenerates a temporary copy and requires its SHA-256 to match the existing copy.

## Build

```bash
python scripts/freeze_generated_round1.py
```

The builder writes `archive_manifest.jsonl`, `gt_eligible.jsonl`, and
`provenance.json` under `data/frozen/generated_round1_v1`. Existing byte-identical
outputs are accepted. Existing outputs with different bytes cause a hard failure.
