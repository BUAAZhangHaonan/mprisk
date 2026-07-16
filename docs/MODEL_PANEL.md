# Model Panel

The frozen experiment panel contains 16 locally available checkpoints. Its machine-readable
source of truth is `configs/assets/model_assets.yaml`; protocol and experiment configs may only
reference keys declared there.

## V-T: text plus image or video

| Model | Scale | Video input | Output policy |
| --- | ---: | --- | --- |
| Gemma-3-4B | 4B | multiple images simulate video | direct Instruct output |
| Gemma-3-12B | 12B | multiple images simulate video | direct Instruct output |
| GLM-4.6V-Flash | ~9B | native video | Thinking disabled with `enable_thinking=false` |
| InternVL3.5-8B | 8B | native video or multiple images | direct Instruct output |
| LLaVA-v1.5-7B | 7B | multiple images simulate video | direct Instruct output |
| LLaVA-OneVision-7B | 7B | native video or multiple images | direct Instruct output |
| MiniCPM-V-2.6 | 8B | multiple images or extracted video frames | direct Instruct output |
| MiniCPM-V-4.5 | 8B | native video or multiple images | Thinking disabled with `enable_thinking=false` |
| Phi-3.5-Vision | 4.2B | multiple images simulate video | direct Instruct output |
| Qwen2.5-VL-7B | 7B | native video or multiple images | direct Instruct output |
| Qwen3-VL-8B-Instruct | 8B | native video or multiple images | the Instruct checkpoint is mandatory |
| Qwen3.5-4B | 4B | native video or multiple images | Thinking disabled with `enable_thinking=false` |
| Qwen3.5-9B | 9B | native video or multiple images | Thinking disabled with `enable_thinking=false` |

`GLM-4.6V-Flash` is the official checkpoint name for the roughly 9B GLM-4.6V variant.
`Qwen/Qwen3-VL-8B-Instruct` is pinned explicitly; the separate Thinking checkpoint is not part
of this panel.

## V-A / V-T-A: vision plus audio or omni input

| Model | Scale | Modalities and video boundary | Output policy |
| --- | ---: | --- | --- |
| Gemma-4-12B | 12B | text, image, native video, and audio | Thinking disabled with `enable_thinking=false` |
| Phi-4-Multimodal-Instruct | 5.6B | text, image, and audio; video requires at most 64 extracted frames | direct Instruct output |
| Qwen2.5-Omni-7B | 7B | text, image, native video, and audio | direct Instruct output |

Phi-4-Multimodal-Instruct does not accept MP4 or MOV as native video. A video must be decoded
into a multi-image sequence before model input, with no more than 64 frames. This frame path is
not classified as native video support.

## Thinking policy

Every asset has both `thinking.enabled: false` and `policy.allow_thinking: false`. Models that
support a runtime Thinking switch record its exact disabling argument. Models without such a
switch use their direct-response or Instruct checkpoint.

## Local assets

The configured model root is `/home/team/lvshuyang/Models`. Every one of the 16 asset entries
resolves to a checkpoint directory containing `config.json`. Qwen2.5-Omni-7B is located at
`/home/team/lvshuyang/Models/Qwen/Qwen2.5-Omni-7B`.

## Required wrapper contract

Each model wrapper must provide:

- stable model key and family;
- prompt compilation support;
- prefill hidden-state extraction;
- first-token logits when available;
- cache sidecar metadata;
- deterministic smoke validation.
