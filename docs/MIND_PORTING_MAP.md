# MIND Porting Map

This repository reuses three engineering ideas from MIND.

## Asset Registry

MIND keeps model assets in a unified registry and validates them before large runs. `mprisk` mirrors that with:

- `configs/assets/model_assets.yaml`
- `src/mprisk/assets/registry.py`
- `scripts/verify_assets.py`
- `outputs/assets/*`

## Full-Cache Surface

MIND uses a unified full-cache manifest as the downstream source of truth. `mprisk` uses the same idea for three-condition pre-generation cache:

- `M1`
- `M2`
- `M12`

The authoritative files are:

- `outputs/full_cache/manifests/unified_full_cache_manifest.json`
- `outputs/full_cache/manifests/extraction_ledger.csv`

## Trajectory Representation

MIND analyzes a full-layer prefill trajectory instead of one hidden-state point. `mprisk` adopts the same object for the revised paper:

```text
H(x) = (h_1, ..., h_L)
u_l = h_l / ||h_l||_2
T(x) = (u_1, ..., u_L)
```

The TAFFC revision uses this idea to answer the single-layer and linear-probe concerns. The paper does not need to import MIND's full Stage A/B/C/D narrative.
