# TAFFC complete-bundle packaging

`build_taffc_complete_bundle.py` is the frozen, fail-closed builder for the
2026-07-21 handoff. It packages the canonical 3,810 generated protocol rows,
CH-SIMS v2 cross-domain views, five hidden-state caches, 15 formal Misread label
sets, and state/TME evidence for exactly Qwen3-VL-8B, InternVL3.5-8B, and
Qwen2.5-Omni-7B.

Qwen3.5-4B and Gemma4-12B are cache/Misread-only. The builder never computes
their state indices and never trains TME.

The active cache hierarchy is dataset-first:

```text
caches/
  generated_set/
    qwen3_vl_8b/          # 1876 samples, 45024 tasks
    internvl3_5_8b/       # 1876 samples, 45024 tasks
    qwen2_5_omni_7b/      # 1934 samples, 46416 tasks
    qwen3_5_4b/           # 1876 samples, 45024 tasks
    gemma4_12b/           # 1934 samples, 46416 tasks
  natural_set/
    ch_sims_v2/
      qwen3_5_4b/         # 2035 samples, 48840 tasks
```

Only Qwen3.5 appears under `natural_set`. Its source mixed manifest is split by
the exact generated and CH-SIMS ID sets, and the two active payload sets are
required to be disjoint. The source mixed manifest, SQLite ledger, and summary
are retained under `provenance/caches/qwen3_5_4b/mixed_original`; its packaged
manifest has valid package-relative references into the two active subtrees.

For Qwen3-VL, InternVL, and Qwen2.5-Omni, `new-only` and
`overlap-frozen-v2` remain internal union provenance only. They are not exposed
as dataset categories. Gemma's five silent IDs, 80 partial successes, and 40
failures are absent from the active cache and retained only under `provenance/`.

Run the source and path audit first:

```bash
python3 scripts/packaging/build_taffc_complete_bundle.py --dry-run
```

Run the focused unit tests with:

```bash
python3 -m pytest -q scripts/packaging/test_build_taffc_complete_bundle.py
```

Build and promote in tmux because linking and hashing hundreds of thousands of
cache files is long-running. The candidate is built without touching the
current final bundle. An independent verify run fully rehashes the candidate.
Promotion requires that run's durable exit-0/PASS record, uses Linux
`renameat2(RENAME_EXCHANGE)` for an atomic directory swap, fully verifies the
new final path, and only then deletes the old bundle. Immediately after the
exchange, promotion rewrites the four root identity/report files through the
same control-metadata code path used by `--refresh-control-metadata`, so a
candidate name or path cannot survive at the canonical final path. Between
exchange and final verification, the old bundle is preserved at
`.taffc_complete_bundle_20260721.backup_pre_dataset_reorg_20260721`:

```bash
bash scripts/packaging/run_taffc_complete_bundle_tmux.sh reorg-dry-run
bash scripts/packaging/run_taffc_complete_bundle_tmux.sh reorg-build
bash scripts/packaging/run_taffc_complete_bundle_tmux.sh reorg-verify
bash scripts/packaging/run_taffc_complete_bundle_tmux.sh reorg-promote
```

Independently recompute every payload SHA and re-check package coverage:

```bash
python3 scripts/packaging/build_taffc_complete_bundle.py \
  --verify-only \
  --output outputs/deliveries/taffc_complete_bundle_20260721
```

`SHA256SUMS` and `file_provenance.tsv` cover immutable payload and package
manifests only. The six root controls (`README.md`, `inventory.json`, both
validation reports, `SHA256SUMS`, and `file_provenance.tsv`) are excluded from
that payload set. Refresh canonical identity and checksum-policy metadata
without reading or hashing payload content with:

```bash
python3 scripts/packaging/build_taffc_complete_bundle.py \
  --refresh-control-metadata \
  --output outputs/deliveries/taffc_complete_bundle_20260721
```

The refresh command also migrates the one legacy bundle format that placed the
four identity/report files inside the checksum manifests. It verifies only
those small legacy control files, removes their rows, and preserves every
payload path and digest byte-for-byte.

The final directory contains no symlinks. Large source artifacts use hardlinks
when the source has the same owner and filesystem. Linux protected-hardlink
rules prevent cross-owner links, so those media are byte-copied and recorded as
`copy_cross_owner` in the provenance manifest. Package-relative media/cache
indexes are generated alongside untouched source manifests under `provenance/`.
Transient SQLite `-shm` and `-wal` sidecars are excluded; the failure ledger is
read with SQLite immutable mode so the audit cannot create new sidecars.
