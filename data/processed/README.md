# Processed Data

Processed data is represented by manifests, prompt banks, labels, and small paper input exports.

## Frozen Delivery 20260714

The current final manifests come verbatim from the frozen
`delivery_20260714.tar.gz` machine-screened delivery. Its archive hash, member hashes,
fixed counts, source boundaries, media policy, and current annotation waiver are recorded in:

- `manifests/delivery_20260714.provenance.json`

The delivered `use_in_main` values are authoritative for the current experiments. The
delivery has no multi-annotator statistics yet; their mean and standard deviation are pending
future annotation work and do not block the current run.

Run the strict acceptance check and deterministic derivation with:

```bash
python scripts/build_manifests.py --repo-root .
```

This command verifies the source archive, every tracked delivery artifact, all referenced media,
the explicit annotation waiver, source boundaries, and variety-text exclusions before writing
the train/validation/test and VT/VA protocol manifests.
