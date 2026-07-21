from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jascue_video_lab.feature_cut import (
    _chapter_bounds_with_approved_trim,
    _load_approved_trim_decisions,
    _piecewise_expression,
    _concat_segments,
    _render_source_segment,
    _render_text_layer,
    _segment_variant_fingerprint,
    _usable_track_centers,
)
from jascue_video_lab.models import (
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureEditBrief,
    TrimIntentDecision,
    RushClip,
    RushFrame,
)
from jascue_video_lab.shots import ShotManifest, ShotSegment
from jascue_video_lab.storage import write_json


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


def test_segment_cache_key_changes_with_source_or_tracking_geometry() -> None:
    base = {
        "source_sha256": "a" * 64,
        "start_ms": 1000,
        "end_ms": 3000,
        "filter_graph": "crop=1080:1920:x=100:y=0",
        "geometry": {"applied_strategy": "tracked_crop"},
        "track_fingerprint": "b" * 64,
    }
    original = _segment_variant_fingerprint(**base)
    assert original != _segment_variant_fingerprint(
        **{**base, "track_fingerprint": "c" * 64}
    )
    assert original != _segment_variant_fingerprint(
        **{**base, "source_sha256": "d" * 64}
    )


def test_feature_cut_refuses_unreviewed_trim_decision(tmp_path) -> None:
    path = tmp_path / "proposed.json"
    decision = TrimIntentDecision(
        source_asset_id="sha256:" + "a" * 64,
        event_id="event-1",
        shot_id="shot-0001",
        usable=False,
        first_included_frame=None,
        last_included_frame=None,
        exclusive_out_frame=None,
        hold_start_frame=None,
        hold_end_frame=None,
        source_in_ms=None,
        source_out_ms=None,
        source_in_pts=None,
        source_out_pts=None,
        handle_in_ms=None,
        handle_out_ms=None,
        tail_intent="uncertain",
        proposal_path="/tmp/proposal.json",
        catalog_path="/tmp/catalog.json",
    )
    write_json(path, decision)

    with pytest.raises(ValueError, match="human-approved"):
        _load_approved_trim_decisions([path])


def test_feature_cut_applies_only_matching_approved_trim_bounds(tmp_path) -> None:
    clip = RushClip(
        clip_id="clip-1",
        path="/tmp/source.mp4",
        sha256="a" * 64,
        duration_ms=10_000,
        width=1920,
        height=1080,
        frame_rate="30/1",
        size_bytes=1,
    )
    frame = RushFrame(
        frame_id="RF000001",
        clip_id=clip.clip_id,
        requested_time_ms=5000,
        image_path="/tmp/frame.jpg",
    )
    evidence = {
        "frame_id": "DF000001",
        "requested_time_ms": 3000,
        "frame_time_ms": 3003,
        "frame_pts": 90,
        "frame_hash": "b" * 64,
    }
    decision = TrimIntentDecision.model_validate(
        {
            "source_asset_id": "sha256:" + clip.sha256,
            "event_id": "event-1",
            "shot_id": "shot-0001",
            "usable": True,
            "first_included_frame": evidence,
            "last_included_frame": {**evidence, "frame_id": "DF000002", "frame_time_ms": 7007},
            "exclusive_out_frame": {
                **evidence,
                "frame_id": "DF000003",
                "frame_time_ms": 7250,
                "frame_pts": 220,
            },
            "hold_start_frame": None,
            "hold_end_frame": None,
            "source_in_ms": 3003,
            "source_out_ms": 7250,
            "source_in_pts": 90,
            "source_out_pts": 220,
            "handle_in_ms": 2250,
            "handle_out_ms": 8250,
            "tail_intent": "natural_pause",
            "approval_status": "approved",
            "requires_human_review": False,
            "human_review": {
                "reviewer": "reviewer",
                "reviewed_at": "2026-07-21T00:00:00Z",
                "decision": "approved",
                "notes": "verified",
            },
            "proposal_path": "/tmp/proposal.json",
            "catalog_path": "/tmp/catalog.json",
        }
    )
    shot_cache = {
        clip.clip_id: ShotManifest(
            video_path=clip.path,
            duration_ms=clip.duration_ms,
            detector="test",
            threshold=4,
            generated_at="2026-07-21T00:00:00Z",
            boundaries=[],
            shots=[
                ShotSegment(
                    shot_id="shot-0001",
                    start_time_ms=0,
                    end_time_ms=10_000,
                    start_frame_pts=0,
                    boundary_source="video_start",
                    boundary_score=None,
                )
            ],
        )
    }

    start_ms, end_ms, shot_id, audit = _chapter_bounds_with_approved_trim(
        frame,
        clip,
        2.0,
        shot_cache,
        tmp_path,
        4.0,
        [(tmp_path / "approved.json", decision)],
    )

    assert (start_ms, end_ms, shot_id) == (3003, 7250, "shot-0001")
    assert audit["trim_method"] == "human_approved_frame_id_pts"
    assert audit["trim_event_id"] == "event-1"


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
                vertical_primary_target_description="reviewer-selected foreground subject",
                vertical_crop_mode="primary_center",
            )
        ],
    )
    assert brief.render_title_overlays is False
    assert brief.chapters[0].vertical_crop_mode == "primary_center"


def test_feature_brief_can_forbid_blurred_vertical_fallback() -> None:
    brief = FeatureEditBrief(
        project_id="clean-cut",
        title="clean",
        target_duration_seconds=60,
        vertical_fallback_strategy="center_crop",
        chapters=[
            FeatureChapterBrief(
                feature_id="hero",
                title="hero",
                detail_lines=[],
                target_duration_seconds=6,
            )
        ],
    )
    assert brief.vertical_fallback_strategy == "center_crop"


def test_tracked_reframe_requires_target_and_nonzero_intent() -> None:
    payload = {
        "feature_id": "ui",
        "evidence_status": "supported",
        "horizontal_frame_id": "RF000001",
        "vertical_frame_id": "RF000002",
        "observed_visual_evidence": "selected subject remains visible",
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


def test_track_centers_are_rebased_to_render_segment_time() -> None:
    track = SimpleNamespace(
        analysis_start_ms=5000,
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=5100,
                tracking_state="tracked",
                center_2d=[300.0, 500.0],
                derived_tracking_box=[200, 200, 400, 800],
            ),
            SimpleNamespace(
                analysis_sample_time_ms=6100,
                tracking_state="low_confidence",
                center_2d=[500.0, 500.0],
                derived_tracking_box=[400, 200, 600, 800],
            ),
        ],
    )
    times, centers, boxes = _usable_track_centers(track)  # type: ignore[arg-type]
    assert times == pytest.approx([0.1, 1.1])
    assert centers == [300.0, 500.0]
    assert boxes == [[200, 200, 400, 800], [400, 200, 600, 800]]


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


def test_video_only_source_gets_explicit_synthetic_silence(tmp_path: Path) -> None:
    source = tmp_path / "video-only.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=30:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    output = tmp_path / "review-segment.mp4"
    audio_origin = _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=1000,
        overlay_path=None,
        base_filter="[0:v]fps=30,scale=320:180,setsar=1[base]",
        output_path=output,
    )
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = {stream["codec_type"] for stream in json.loads(completed.stdout)["streams"]}
    assert audio_origin == "synthetic_silence"
    assert stream_types == {"video", "audio"}
