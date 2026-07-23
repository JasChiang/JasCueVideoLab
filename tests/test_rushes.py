from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from jascue_video_lab.media import probe_video
from jascue_video_lab.models import RushesEditPlan
from jascue_video_lab.rushes import _crop_filter, _segment_bounds, create_rushes_catalog
from jascue_video_lab.sam_tracking import _normalize_shot_manifest
from jascue_video_lab.shots import ShotSegment, detect_shots_ffmpeg


def _make_color_video(path: Path, color: str, duration: float = 2) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=320x180:r=10:d={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_ffmpeg_scdet_preserves_exact_boundary_pts(tmp_path: Path) -> None:
    video = tmp_path / "cuts.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x180:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x180:r=10:d=2",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    manifest = detect_shots_ffmpeg(video, threshold=4)
    assert [boundary.frame_time_ms for boundary in manifest.boundaries] == [2000, 4000]
    assert all(boundary.frame_pts > 0 for boundary in manifest.boundaries)
    assert [(shot.start_time_ms, shot.end_time_ms) for shot in manifest.shots] == [
        (0, 2000),
        (2000, 4000),
        (4000, 6000),
    ]


def test_ffmpeg_scdet_uses_local_time_with_nonzero_stream_start_pts(
    tmp_path: Path,
) -> None:
    video = tmp_path / "nonzero-start-cuts.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x180:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:r=10:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x180:r=10:d=2",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0,setpts=PTS+5/TB[v]",
            "-map",
            "[v]",
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

    detected = detect_shots_ffmpeg(video, threshold=4)
    assert [boundary.frame_time_ms for boundary in detected.boundaries] == [2000, 4000]
    time_base = Fraction(
        media.video.time_base.numerator,
        media.video.time_base.denominator,
    )
    assert [boundary.frame_pts for boundary in detected.boundaries] == [
        media.video.start_pts + round(Fraction(2, 1) / time_base),
        media.video.start_pts + round(Fraction(4, 1) / time_base),
    ]

    normalized = _normalize_shot_manifest(
        detected,
        duration_ms=media.duration_ms,
        source_start_pts=media.video.start_pts,
        time_base_numerator=media.video.time_base.numerator,
        time_base_denominator=media.video.time_base.denominator,
    )
    assert normalized.timeline_basis == "local_ms_from_decoded_pts"
    assert normalized.source_start_pts == media.video.start_pts
    assert normalized.source_time_base == media.video.time_base
    assert normalized.shots[0].start_frame_pts == media.video.start_pts
    assert [(shot.start_time_ms, shot.end_time_ms) for shot in normalized.shots] == [
        (0, 2000),
        (2000, 4000),
        (4000, 6000),
    ]


def test_catalog_uses_immutable_frame_ids_without_model_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _make_color_video(source / "A001.MP4", "red")
    _make_color_video(source / "A002.MP4", "blue")
    catalog = create_rushes_catalog(source, tmp_path / "catalog", sample_interval_ms=1000)
    assert len(catalog.clips) == 2
    assert [frame.frame_id for frame in catalog.frames] == [
        "RF000001",
        "RF000002",
        "RF000003",
        "RF000004",
    ]
    assert Path(catalog.analysis_reel_path).exists()
    assert all((tmp_path / "catalog" / frame.image_path).exists() for frame in catalog.frames)


def test_catalog_keeps_one_frame_for_sub_interval_clip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _make_color_video(source / "SHORT.MP4", "green", duration=0.4)
    catalog = create_rushes_catalog(source, tmp_path / "catalog", sample_interval_ms=2000)
    assert len(catalog.clips) == 1
    assert [frame.frame_id for frame in catalog.frames] == ["RF000001"]
    assert catalog.frames[0].requested_time_ms == 0
    assert (tmp_path / "catalog" / catalog.frames[0].image_path).exists()


def test_segment_handles_are_clamped_to_ffmpeg_shot() -> None:
    shot = ShotSegment(
        shot_id="shot-0002",
        start_time_ms=4000,
        end_time_ms=8000,
        start_frame_pts=400,
        boundary_source="ffmpeg_scdet",
        boundary_score=12.0,
    )
    assert _segment_bounds(
        center_ms=4500,
        requested_duration_ms=3000,
        clip_duration_ms=10000,
        shot=shot,
    ) == (4000, 7000)


def test_vertical_focus_maps_to_fixed_crop_not_fake_tracking() -> None:
    left, dimensions = _crop_filter("9:16", "left")
    right, _ = _crop_filter("9:16", "right")
    assert dimensions == (1080, 1920)
    assert "crop=1080:1920:0:0" in left
    assert "crop=1080:1920:iw-ow:0" in right


def test_edit_plan_requires_both_aspects() -> None:
    payload = {
        "project_id": "p",
        "catalog_id": "c",
        "summary": "s",
        "timelines": [],
        "uncertainties": [],
        "model_provenance": {
            "model_id": "gemini-3.5-flash",
            "api": "gemini_interactions",
            "sdk": "google-genai",
            "sdk_version": "x",
            "interaction_id": None,
            "run_id": "r",
            "generated_at": "now",
        },
    }
    with pytest.raises(ValueError, match="exactly one"):
        RushesEditPlan.model_validate(payload)
