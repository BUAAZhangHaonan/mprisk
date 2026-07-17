# Delivery 20260716 prefill-cache union

The delivery cache is an immutable view. It never copies or edits a source shard,
sidecar, or SQLite ledger. Historical main-dataset caches are archival and must not
enter this view because they do not contain historical extractor-code identity.

For each model, the view combines two disjoint delivery roots:

- `delivery-new-only`: samples that were not present in the earlier dataset;
- `delivery-overlap-reextracted`: overlapping samples extracted again with the exact
  same `mprisk-v2` full-prefill code and runtime signature.

Before building the view, record evidence for each completed source ledger:

```bash
python scripts/build_prefill_cache_union.py record-evidence \
  --source-id qwen3-vl-delivery-new-only \
  --ledger /absolute/source/root/batch_state.sqlite3 \
  --cache-root /absolute/source/root \
  --code-root /home/team/zhanghaonan/TAFFC/mprisk-v2 \
  --output /absolute/provenance/qwen3-vl-new-only-evidence.json

python scripts/build_prefill_cache_union.py record-source \
  --source-id qwen3-vl-delivery-new-only \
  --ledger /absolute/source/root/batch_state.sqlite3 \
  --cache-root /absolute/source/root \
  --evidence /absolute/provenance/qwen3-vl-new-only-evidence.json \
  --output /absolute/provenance/qwen3-vl-new-only-source.json
```

Repeat both commands for the re-extracted overlap root. Evidence records bind the
logical ledger content, exact semantic source-file hashes, Git object hashes, prompt
and asset hashes, preprocessing signature, every weight shard referenced by the model
index, and all runtime config/processor/tokenizer assets. Each model file records both
size and SHA-256, summarized by one model-asset fingerprint. All source evidence
fingerprints must be identical. There is no compatibility fallback or inferred equivalence.

Build VT views from `manifests/vt_primary.jsonl` with exactly 45,024 resolved tasks
per VL model. Build the VA view from `manifests/va_state_valid.jsonl` with exactly
46,416 resolved tasks. Pass `manifests/invalid_assets.jsonl` to the VA command so the
five invalid source rows remain visible as 120 blocked tasks; they are never exposed
as cache entries. The VA raw-source accounting is therefore 46,536 tasks.

```bash
python scripts/build_prefill_cache_union.py build \
  --manifest outputs/datasets/delivery_20260716/manifests/va_state_valid.jsonl \
  --blocked-manifest outputs/datasets/delivery_20260716/manifests/invalid_assets.jsonl \
  --prompt-set configs/prompts/equiv_sets/va_main_p8_seed20260717.yaml \
  --protocol va \
  --model-key qwen2_5_omni_7b \
  --source /absolute/provenance/omni-new-only-source.json \
  --source /absolute/provenance/omni-overlap-source.json \
  --expected-resolved-tasks 46416 \
  --expected-blocked-tasks 120 \
  --expected-raw-tasks 46536 \
  --checksum-workers 8 \
  --output outputs/prefill_cache_unions/delivery_20260716/qwen2_5_omni_7b.json
```

The builder fails on missing, duplicate, non-completed usable tasks, incompatible
extractor evidence, request changes, source-signature changes, broken sidecars,
checksum mismatches, tensor-shape mismatches, or model-artifact mismatches. The new
delivery split is written only into the union entry. Original source dataset and
split values remain under `source_provenance`.
