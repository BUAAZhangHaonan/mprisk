# Canonical complete-bundle protocol

`configs/packaging/complete_bundle_matrix.yaml` is the only input contract
for the final TAFFC delivery. It fixes the full panel at 16 models and creates
one source and one target job for every model.

The protocol-specific sample counts are:

| Domain | VT | VA |
| --- | ---: | ---: |
| Source | 1,876 | 1,934 |
| Target | 2,035 | 2,190 |

VT and VA target manifests are separate protocol datasets. The builder rejects
an attempt to replace the 2,190-row target VA contract with a 2,035-row
intersection.

Every one of the 32 jobs declares four artifacts:

1. hidden-state cache index;
2. `GT_DESCRIPTION` manifest;
3. diagnostic affect description manifest;
4. resolved Misread label manifest.

A ready artifact must bind its payload path and SHA-256 plus an independent
provenance path and SHA-256. A pending artifact contains only a reason. Pending
records are accepted only by readiness reporting; `--build` fails before
creating an output directory if any record is pending or invalid. In
particular, missing target GT never produces placeholder descriptions,
pseudo-labels, or a nominally complete bundle.

The source Phi-4 Misread label is a required matrix cell. It cannot be omitted
by changing a model-count constant.

Generate the current readiness report:

```bash
python scripts/packaging/build_taffc_complete_bundle.py \
  --config configs/packaging/complete_bundle_matrix.yaml \
  --readiness-report outputs/deliveries/complete_bundle_readiness.json
```

Build only after every cell has been changed to a verified `ready` record:

```bash
python scripts/packaging/build_taffc_complete_bundle.py \
  --config configs/packaging/complete_bundle_matrix.yaml \
  --output outputs/deliveries/taffc_complete_bundle_final \
  --build
```

Verify a published bundle independently:

```bash
python scripts/packaging/build_taffc_complete_bundle.py \
  --output outputs/deliveries/taffc_complete_bundle_final \
  --verify-only
```

The output `inventory.json` preserves the 32 job identities and the SHA/provenance
pair for every artifact. `SHA256SUMS` covers the exact payload set, and
`validation_report.json` embeds the complete readiness audit that authorized
publication.
