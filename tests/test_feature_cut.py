from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from jascue_video_lab.feature_cut import (
    _piecewise_expression,
    _concat_segments,
    _render_source_segment,
    _render_text_layer,
)
from jascue_video_lab.models import FeatureChapterBrief, FeatureChapterSelect, FeatureEditBrief


def test_feature_brief_requires_unique_chapter_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        FeatureEditBrief(
            project_id="test",
            title="test",
            target_duration_seconds=60,
            chapters=[
                FeatureChapterBrief(
                    feature_id="same",
                    title="one",
                    detail_lines=[],
                    target_duration_seconds=4,
                ),
                FeatureChapterBrief(
                    feature_id="same",
                    title="two",
                    detail_lines=[],
                    target_duration_seconds=4,
                ),
            ],
        )


def test_feature_brief_can_disable_titles_and_choose_primary_center_crop() -> None:
    brief = FeatureEditBrief(
        project_id="clean-cut",
        title="clean",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id="hero",
                title="hero",
                detail_lines=[],
                target_duration_seconds=6,
                vertical_primary_target_description="presenter holding the white phone",
                vertical_crop_mode="primary_center",
            )
        ],
    )
    assert brief.render_title_overlays is False
    assert brief.chapters[0].vertical_crop_mode == "primary_center"


def test_tracked_reframe_requires_target_and_nonzero_intent() -> None:
    payload = {
        "feature_id": "ui",
        "evidence_status": "supported",
        "horizontal_frame_id": "RF000001",
        "vertical_frame_id": "RF000002",
        "observed_visual_evidence": "phone screen",
        "selection_reason": "clear",
        "horizontal_strategy": "tracked_reframe",
        "horizontal_zoom_intent": "none",
        "horizontal_target_description": None,
        "vertical_strategy": "fit_with_background",
        "vertical_target_description": None,
        "quality_risks": [],
        "confidence": 0.9,
    }
    with pytest.raises(ValidationError, match="requires a zoom intent"):
        FeatureChapterSelect.model_validate(payload)


def test_piecewise_expression_is_ffmpeg_escaped() -> None:
    expression = _piecewise_expression([0.0, 0.5, 1.0], [100.0, 150.0, 130.0])
    assert "lt(t\\,0.500)" in expression
    assert "if(" in expression


def test_dynamic_crop_filter_renders_video_and_audio(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=30:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    chapter = FeatureChapterBrief(
        feature_id="demo",
        title="動態安全裁切",
        detail_lines=["保留指定主體"],
        target_duration_seconds=3,
    )
    overlay = tmp_path / "overlay.png"
    _render_text_layer(chapter, overlay, dimensions=(1080, 1920))
    expression = _piecewise_expression([0.0, 1.0, 2.0], [400.0, 900.0, 500.0])
    output = tmp_path / "vertical.mp4"
    _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=2000,
        overlay_path=overlay,
        base_filter=(
            "[0:v]fps=30,scale=3414:1920,"
            f"crop=1080:1920:x='{expression}':y=0,setsar=1[base]"
        ),
        output_path=output,
    )
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(stream["codec_type"] == "audio" for stream in streams)

    clean_output = tmp_path / "vertical-clean.mp4"
    _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=500,
        overlay_path=None,
        base_filter=(
            "[0:v]fps=30,scale=3414:1920,"
            "crop=1080:1920:x=500:y=0,setsar=1[base]"
        ),
        output_path=clean_output,
    )
    assert clean_output.exists()
    assert not (tmp_path / ".vertical-clean.partial.mp4").exists()


def test_concat_decodes_each_mp4_instead_of_stream_copy(tmp_path: Path) -> None:
    segments: list[Path] = []
    for index, frequency in enumerate((440, 660)):
        segment = tmp_path / f"segment-{index}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={'red' if index == 0 else 'blue'}:s=320x180:r=30:d=1",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate=48000:duration=1",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-pix_fmt",
                "yuv420p",
                str(segment),
            ],
            check=True,
        )
        segments.append(segment)
    output = tmp_path / "joined.mp4"
    _concat_segments(segments, output)
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(completed.stdout) == pytest.approx(2.0, abs=0.08)
