from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from jascue_video_lab.cli import build_parser
from jascue_video_lab.media import probe_video, sha256_file
from jascue_video_lab.models import (
    SegmentationModelProvenance,
    SegmentationSample,
    SegmentationTrack,
    SemanticIdentityStatus,
    TrackingState,
)
from jascue_video_lab.multi_tracking import render_multi_segmentation_review
from jascue_video_lab.storage import write_json


def _source_with_audio(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=160x90:r=30:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    return path


def _source_pts(time_ms: int, start_pts: int, time_base: Fraction) -> int:
    return start_pts + round(Fraction(time_ms, 1000) / time_base)


def _track(
    root: Path,
    source: Path,
    *,
    target: str,
    box: list[int],
) -> Path:
    media = probe_video(source)
    frames_dir = root / "analysis-frames"
    masks_dir = root / "masks"
    frames_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    source_start_pts = media.video.start_pts or 0
    time_base = Fraction(
        media.video.time_base.numerator, media.video.time_base.denominator
    )
    samples: list[SegmentationSample] = []
    for index, time_ms in enumerate((0, 500, 1000, 1500)):
        frame_path = frames_dir / f"{index:06d}.jpg"
        Image.new("RGB", (160, 90), (40 + index * 20, 55, 80)).save(frame_path)
        mask_path = masks_dir / f"{index:06d}.png"
        mask = Image.new("L", (160, 90), 0)
        draw = ImageDraw.Draw(mask)
        x_min, y_min, x_max, y_max = box
        draw.rectangle(
            (
                round(x_min * 160 / 1000),
                round(y_min * 90 / 1000),
                round(x_max * 160 / 1000),
                round(y_max * 90 / 1000),
            ),
            fill=255,
        )
        mask.save(mask_path)
        samples.append(
            SegmentationSample(
                sample_index=index,
                analysis_sample_time_ms=time_ms,
                source_pts=_source_pts(time_ms, source_start_pts, time_base),
                timing_basis="decoded_source_pts",
                mask_path=f"masks/{index:06d}.png",
                mask_sha256=sha256_file(mask_path),
                mask_area_pixels=1200,
                mask_area_ratio=0.08333333,
                connected_components=1,
                derived_tracking_box=box,
                center_2d=[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                mean_positive_probability=0.95,
                scene_cut_score=None,
                shot_boundary=False,
                tracking_state=TrackingState.TRACKED,
                state_reasons=[],
                semantic_identity_status=SemanticIdentityStatus.NOT_REVALIDATED,
            )
        )
    track = SegmentationTrack(
        method="bbox_seed_sam2_video_mask_propagation",
        asset_id=media.asset_id,
        video_path=str(source.resolve()),
        target_description=target,
        seed_source="test bbox",
        seed_time_ms=500,
        seed_sample_index=1,
        semantic_seed_box=box,
        seed_prompt_type="box",
        sam_prompt_box=box,
        sam_prompt_mask_polygon_xy=None,
        seed_box_padding_ratio=0,
        refined_seed_mask_path="masks/000001.png",
        analysis_fps=2,
        analysis_width=160,
        analysis_height=90,
        analysis_start_ms=0,
        analysis_end_ms=2000,
        source_start_pts=source_start_pts,
        source_time_base=media.video.time_base,
        timing_warning="test timing",
        semantic_warning="test semantic warning",
        total_samples=4,
        state_counts={TrackingState.TRACKED: 4},
        elapsed_seconds=0,
        effective_fps=2,
        model_provenance=SegmentationModelProvenance(
            model_id="sam2.1_hiera_tiny",
            implementation="test",
            implementation_revision="test",
            checkpoint_sha256="a" * 64,
            device="cpu",
            torch_version="test",
            generated_at="2026-07-21T00:00:00Z",
        ),
        samples=samples,
    )
    path = root / "segmentation-track.json"
    write_json(path, track)
    return path


def _fixture_tracks(tmp_path: Path) -> tuple[Path, Path]:
    source = _source_with_audio(tmp_path / "source.mp4")
    left = _track(
        tmp_path / "left",
        source,
        target="the independently selected left object",
        box=[100, 200, 400, 800],
    )
    right = _track(
        tmp_path / "right",
        source,
        target="the independently selected right object",
        box=[600, 200, 900, 800],
    )
    return left, right


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda data: data.update(asset_id="sha256:" + "b" * 64), "asset_id"),
        (lambda data: data.update(analysis_fps=4), "analysis_fps"),
        (lambda data: data.update(analysis_end_ms=1900), "analysis_end_ms"),
        (lambda data: data.update(analysis_width=161), "analysis_width"),
        (
            lambda data: data["samples"][2].update(
                source_pts=data["samples"][2]["source_pts"] + 1
            ),
            "samples",
        ),
    ],
)
def test_multi_track_alignment_fails_closed(
    tmp_path: Path, mutate, expected: str
) -> None:
    left, right = _fixture_tracks(tmp_path)
    payload = json.loads(right.read_text(encoding="utf-8"))
    mutate(payload)
    right.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        render_multi_segmentation_review(
            track_json_paths=[left, right],
            labels=["Object A", "Object B"],
            output_dir=tmp_path / "review",
        )


def test_multi_track_alignment_rejects_a_different_video_path(tmp_path: Path) -> None:
    left, right = _fixture_tracks(tmp_path)
    alternate_source = tmp_path / "alternate-source.mp4"
    shutil.copyfile(tmp_path / "source.mp4", alternate_source)
    payload = json.loads(right.read_text(encoding="utf-8"))
    payload["video_path"] = str(alternate_source)
    right.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="video_path"):
        render_multi_segmentation_review(
            track_json_paths=[left, right],
            labels=["Object A", "Object B"],
            output_dir=tmp_path / "review",
        )


def test_multi_track_review_is_normal_duration_h264_and_muxes_audio(
    tmp_path: Path,
) -> None:
    left, right = _fixture_tracks(tmp_path)
    output_dir = tmp_path / "review"
    manifest = render_multi_segmentation_review(
        track_json_paths=[left, right],
        labels=["Selected object A", "Selected object B"],
        output_dir=output_dir,
        display_fps=30,
    )

    assert manifest.interpretation == "manual_review_visualization_not_accuracy"
    assert manifest.analysis_fps == 2
    assert manifest.display_fps == 30
    assert manifest.output_codec_name == "h264"
    assert manifest.output_pixel_format == "yuv420p"
    assert manifest.output_frame_rate.numerator / manifest.output_frame_rate.denominator == 30
    assert manifest.audio_muxed is True
    assert manifest.output_duration_ms == pytest.approx(2000, abs=70)
    assert manifest.output_video_duration_ms == pytest.approx(2000, abs=40)
    assert manifest.output_frame_count >= 59
    assert [member.label for member in manifest.members] == [
        "Selected object A",
        "Selected object B",
    ]
    assert manifest.members[0].color_rgb != manifest.members[1].color_rgb
    assert "not an accuracy measurement" in manifest.warning
    assert (output_dir / "multi-track-review.mp4").exists()
    assert (output_dir / "multi-track-review.json").exists()


def test_multi_track_review_preserves_irregular_sample_timing(tmp_path: Path) -> None:
    left, right = _fixture_tracks(tmp_path)
    irregular_times = [0, 400, 1200, 1700]
    media = probe_video(tmp_path / "source.mp4")
    source_start_pts = media.video.start_pts or 0
    time_base = Fraction(
        media.video.time_base.numerator, media.video.time_base.denominator
    )
    for track_path in (left, right):
        payload = json.loads(track_path.read_text(encoding="utf-8"))
        for sample, time_ms in zip(payload["samples"], irregular_times, strict=True):
            sample["analysis_sample_time_ms"] = time_ms
            sample["source_pts"] = _source_pts(time_ms, source_start_pts, time_base)
        track_path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = render_multi_segmentation_review(
        track_json_paths=[left, right],
        labels=["Object A", "Object B"],
        output_dir=tmp_path / "review-irregular",
        display_fps=30,
    )

    assert manifest.output_video_duration_ms == pytest.approx(2000, abs=40)
    assert manifest.output_frame_count >= 59


def test_multi_track_cli_preserves_track_and_label_order() -> None:
    args = build_parser().parse_args(
        [
            "render-multi-sam21",
            "first.json",
            "second.json",
            "--label",
            "First",
            "--label",
            "Second",
            "--display-fps",
            "30",
            "--output-dir",
            "review",
        ]
    )

    assert args.track_json == [Path("first.json"), Path("second.json")]
    assert args.label == ["First", "Second"]
    assert args.display_fps == 30
