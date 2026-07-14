# Qwen2.5-Omni prefill extraction

This extractor loads only `Qwen2_5OmniThinkerForConditionalGeneration`. It does not load
the Talker, call `generate`, or retain the full token-by-layer cache. Each artifact stores
the 28 Thinker block states at the final non-padding conditioning token.

## Isolated environment on 6403

The validated environment reuses the CUDA Torch and Transformers packages from
`mind-py311`, while installing Qwen-specific media dependencies only inside a venv:

```bash
PYTHONNOUSERSITE=1 /home/team/zhanghaonan/miniconda3/envs/mind-py311/bin/python \
  -m venv --system-site-packages \
  /home/team/zhanghaonan/.venvs/mprisk-qwen-omni
PYTHONNOUSERSITE=1 /home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python \
  -m pip install 'qwen-omni-utils[decord]==0.0.9'
cd /home/team/zhanghaonan/TAFFC/mprisk
PYTHONNOUSERSITE=1 /home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python \
  -m pip install --no-deps -e .
```

Validated versions are Torch `2.6.0+cu124`, Transformers `4.57.1`, Accelerate `1.6.0`,
and qwen-omni-utils `0.0.9`.

## Conditioning definitions

- VT: M1 is vision, M2 is transcript text, and M12 is vision plus transcript. Video audio
  is disabled for all three views.
- VA: M1 is silent video, M2 is audio, and M12 is video plus audio.
- VTA: transcript text is shared context; M1 is vision plus text, M2 is audio plus text,
  and M12 is vision plus audio plus text.

For media whose vision and audio paths point to the same MP4, use
`--joint-audio-mode embedded_video`. M1 passes the video with
`use_audio_in_video=False`, so its audio track is not decoded. M2 passes the same MP4 as
an explicit audio item. M12 passes it as video with `use_audio_in_video=True`. For separate
vision and audio files, use `--joint-audio-mode separate_file`.

## One-sample extraction

Run GPU work in tmux and expose exactly one physical GPU. With physical GPU 1 exposed,
the process-local device remains `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
  /home/team/zhanghaonan/.venvs/mprisk-qwen-omni/bin/python \
  scripts/extract_prefill_cache.py \
  --manifest data/processed/manifests/protocol_manifests/va_aux.jsonl \
  --sample-id ch_sims_v2:VA:video_0001:0001 \
  --protocol va \
  --conditions M1 M2 M12 \
  --task-prompt 'Identify the main emotion and briefly explain the evidence.' \
  --joint-audio-mode embedded_video \
  --model-path /home/team/lvshuyang/Models/Qwen/Qwen2.5-Omni-7B \
  --device cuda:0 \
  --attn-implementation sdpa \
  --video-fps 1 \
  --max-pixels 602112 \
  --output-root outputs/full_cache
```

The CLI is batch-size one by construction. Multiple sample IDs are processed sequentially.
Existing sample/model/protocol/condition entries fail unless `--overwrite` is explicit.
