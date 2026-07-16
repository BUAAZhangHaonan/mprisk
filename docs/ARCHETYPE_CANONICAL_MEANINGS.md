# Archetype canonical meanings

`archetype_canonical_meanings_v1` binds every generated round-one sample to one
scene-independent semantic archetype. It is derived only from the immutable
`generated_round1_v1` snapshot and the prompt-make source definitions whose
SHA-256 values were locked by the archive-freeze provenance.

- A meanings use the official `ARCHETYPES_GLM` `desc`, `gt`, and `surface`
  fields. The description is only whitespace-normalized and terminated as one
  sentence. It is not expanded by a model or by hand.
- C meanings use the exact `EMOTION_VARIANTS` tuple vocabulary and state only
  that all modalities consistently express the recorded emotion. C entries
  always have `surface_emotion: null`.
- `archetype_semantic_assignments_v1.jsonl` gives every one of the 652 frozen
  samples exactly one dictionary key and marks the 162 GT-eligible rows.
- Recorded prompt-make name and surface variants are accepted only through the
  explicit versioned alias maps in the dictionary config; suffix stripping and
  fuzzy normalization are not used.
- Invalid or non-sentence official A descriptions are written to the versioned
  review queue and block build/verification. There is no generated fallback.

Build and verify with:

```bash
python scripts/build_archetype_canonical_meanings.py
python scripts/verify_archetype_canonical_meanings.py
```

The builder accepts byte-identical reruns and rejects any attempt to change an
existing artifact under the same dictionary version.
