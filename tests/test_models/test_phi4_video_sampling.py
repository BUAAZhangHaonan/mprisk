from __future__ import annotations

import subprocess

import numpy as np

from mprisk.models.phi4_mm import _uniform_video_sample_ffmpeg


def test_phi4_ffmpeg_sampler_decodes_exact_midpoint_indices(tmp_path):
    video = tmp_path / "twenty_frames.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=16x12:rate=10",
            "-frames:v",
            "20",
            "-c:v",
            "ffv1",
            str(video),
        ],
        check=True,
    )

    frames, metadata = _uniform_video_sample_ffmpeg(str(video), 8)
    decoded_all = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    all_frames = np.frombuffer(decoded_all.stdout, dtype=np.uint8).reshape(
        20, 12, 16, 3
    )

    assert len(frames) == 8
    assert [frame.size for frame in frames] == [(16, 12)] * 8
    assert metadata["video_backend"] == "ffmpeg"
    assert metadata["total_num_frames"] == 20
    assert metadata["frames_indices"] == [1, 3, 6, 8, 11, 13, 16, 18]
    np.testing.assert_array_equal(
        np.stack([np.asarray(frame) for frame in frames]),
        all_frames[metadata["frames_indices"]],
    )
