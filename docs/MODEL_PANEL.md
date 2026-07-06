# Model Panel

The model panel is managed through `configs/assets/model_assets.yaml`.

## Initial Models

- Qwen2.5-Omni-7B-Instruct for `VA`.
- Qwen3-VL-8B-Instruct for `VT`.
- InternVL3 5-8B for `VT`.

## Expansion Surface

The repository keeps a wrapper registry so that additional model families can share the same cache contract. The intended expansion follows the MIND pattern: verify assets first, then add them to the extraction panel.

## Required Model Contract

Each model wrapper must provide:

- stable model key and family;
- prompt compilation support;
- prefill hidden-state extraction;
- first-token logits when available;
- cache sidecar metadata;
- deterministic smoke validation.
