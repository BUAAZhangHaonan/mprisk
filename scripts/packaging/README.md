# TAFFC complete-bundle packaging

`build_taffc_complete_bundle.py` is the frozen, fail-closed builder for the
2026-07-21 handoff. It packages the canonical 3,810 generated protocol rows,
CH-SIMS v2 cross-domain views, five hidden-state caches, 15 formal Misread label
sets, and state/TME evidence for exactly Qwen3-VL-8B, InternVL3.5-8B, and
Qwen2.5-Omni-7B.

Qwen3.5-4B and Gemma4-12B are cache/Misread-only. The builder never computes
their state indices and never trains TME.

Run the source and path audit first:

```bash
python3 scripts/packaging/build_taffc_complete_bundle.py --dry-run
```

Run the focused unit tests with:

```bash
python3 -m pytest -q scripts/packaging/test_build_taffc_complete_bundle.py
```

Build in tmux because linking and hashing hundreds of thousands of cache files
is long-running:

```bash
bash scripts/packaging/run_taffc_complete_bundle_tmux.sh dry-run
bash scripts/packaging/run_taffc_complete_bundle_tmux.sh build
bash scripts/packaging/run_taffc_complete_bundle_tmux.sh verify
```

Independently recompute every payload SHA and re-check package coverage:

```bash
python3 scripts/packaging/build_taffc_complete_bundle.py \
  --verify-only \
  --output outputs/deliveries/taffc_complete_bundle_20260721
```

The final directory contains no symlinks. Large source artifacts use hardlinks
when the source has the same owner and filesystem. Linux protected-hardlink
rules prevent cross-owner links, so those media are byte-copied and recorded as
`copy_cross_owner` in the provenance manifest. Package-relative media/cache
indexes are generated alongside untouched source manifests under `provenance/`.
Transient SQLite `-shm` and `-wal` sidecars are excluded; the failure ledger is
read with SQLite immutable mode so the audit cannot create new sidecars.
