from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

from jascue_video_lab.media import create_analysis_proxy, extract_frame, probe_video


def test_probe_and_extract_preserve_semantic_request_vs_pts(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "color=c=blue:s=320x180:r=10:d=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    media = probe_video(video)
    assert media.video.coded_width == 320
    assert media.video.display_height == 180
    assert media.duration_ms == 2000
    frame = extract_frame(video, 555, tmp_path / "frame.png")
    assert frame.requested_time_ms == 555
    assert frame.frame_time_ms == 600
    assert frame.frame_pts != frame.frame_time_ms
    assert (frame.width, frame.height) == (320, 180)


def test_probe_preserves_non_square_sample_aspect_ratio(tmp_path: Path) -> None:
    video = tmp_path / "anamorphic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=720x576:r=10:d=0.2",
            "-vf",
            "setsar=16/15",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )

    media = probe_video(video)

    assert media.video.sample_aspect_ratio.numerator == 16
    assert media.video.sample_aspect_ratio.denominator == 15
    assert media.video.display_sample_aspect_ratio == media.video.sample_aspect_ratio


def test_extract_frame_maps_nonzero_stream_pts_to_local_playback_time(
    tmp_path: Path,
) -> None:
    video = tmp_path / "nonzero-start.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:r=10:d=2",
            "-vf",
            "setpts=PTS+5/TB",
            "-copyts",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    media = probe_video(video)
    assert media.video.start_pts is not None
    assert media.video.start_pts > 0

    frame = extract_frame(video, 555, tmp_path / "nonzero-frame.png")

    source_time_base = Fraction(
        media.video.time_base.numerator,
        media.video.time_base.denominator,
    )
    local_time_ms = round(
        Fraction(frame.frame_pts - media.video.start_pts) * source_time_base * 1000
    )
    assert frame.requested_time_ms == 555
    assert frame.frame_time_ms == local_time_ms == 600
    assert frame.frame_pts > media.video.start_pts


def test_analysis_proxy_preserves_duration_and_has_independent_identity(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "color=c=purple:s=640x360:r=20:d=2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    source = probe_video(video)
    proxy, record = create_analysis_proxy(video, tmp_path / "analysis-proxy.mp4", max_side=480)
    assert proxy.video.display_width == 480
    assert abs(proxy.duration_ms - source.duration_ms) <= 100
    assert proxy.asset_id != source.asset_id
    assert record["source_asset_id"] == source.asset_id
    assert record["proxy_asset_id"] == proxy.asset_id
