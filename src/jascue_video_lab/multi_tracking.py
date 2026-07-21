from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from .geometry import box_iou, center_distance
from .media import has_audio_stream, probe_video, sha256_file
from .models import (
    MultiSegmentationReviewManifest,
    MultiSegmentationReviewMember,
    Rational,
    SegmentationSample,
    SegmentationTrackAgreementReport,
    SegmentationTrackAgreementSample,
    SegmentationTrack,
    SharedSam21AnalysisFramesManifest,
)
from .overlay import _overlay_font
from .storage import utc_now, write_json


_REVIEW_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 255, 170),
    (255, 75, 166),
    (255, 202, 58),
    (64, 156, 255),
    (255, 127, 39),
    (166, 99, 255),
    (76, 223, 255),
    (255, 96, 96),
)

_REVIEW_WARNING = (
    "This synchronized overlay is a manual-review visualization of per-target "
    "tracker proposals. It is not an accuracy measurement, independent ground truth, "
    "or production SpatialTrack data."
)

_AGREEMENT_WARNING = (
    "These symmetric A/B measurements quantify agreement between two tracker outputs. "
    "Neither track is ground truth, so higher agreement does not establish accuracy."
)


def _load_track(path: Path) -> tuple[Path, SegmentationTrack]:
    resolved = path.expanduser().resolve(strict=True)
    return resolved, SegmentationTrack.model_validate_json(
        resolved.read_text(encoding="utf-8")
    )


def _alignment_signature(track: SegmentationTrack) -> dict[str, object]:
    return {
        "asset_id": track.asset_id,
        "video_path": str(Path(track.video_path).expanduser().resolve(strict=True)),
        "analysis_fps": track.analysis_fps,
        "analysis_start_ms": track.analysis_start_ms,
        "analysis_end_ms": track.analysis_end_ms,
        "analysis_width": track.analysis_width,
        "analysis_height": track.analysis_height,
        "source_start_pts": track.source_start_pts,
        "source_time_base": (
            track.source_time_base.model_dump(mode="json")
            if track.source_time_base is not None
            else None
        ),
        "samples": [
            {
                "sample_index": sample.sample_index,
                "source_pts": sample.source_pts,
                "analysis_sample_time_ms": sample.analysis_sample_time_ms,
            }
            for sample in track.samples
        ],
    }


def validate_segmentation_track_alignment(
    tracks: Sequence[SegmentationTrack],
) -> None:
    """Require exact source and decoded-sample lineage before combining tracks."""
    if len(tracks) < 2:
        raise ValueError("multi-track review requires at least two segmentation tracks")
    for index, track in enumerate(tracks, start=1):
        if track.analysis_end_ms is None:
            raise ValueError(f"track {index} has no bounded analysis_end_ms")
        if any(sample.source_pts is None for sample in track.samples):
            raise ValueError(f"track {index} contains samples without decoded source PTS")
        if [sample.sample_index for sample in track.samples] != list(
            range(track.total_samples)
        ):
            raise ValueError(f"track {index} sample indices are not contiguous from zero")
        sample_times = [sample.analysis_sample_time_ms for sample in track.samples]
        if any(current >= following for current, following in zip(sample_times, sample_times[1:])):
            raise ValueError(f"track {index} sample times are not strictly increasing")
        source_pts = [sample.source_pts for sample in track.samples]
        if any(current >= following for current, following in zip(source_pts, source_pts[1:])):
            raise ValueError(f"track {index} source PTS values are not strictly increasing")
    reference = _alignment_signature(tracks[0])
    for index, track in enumerate(tracks[1:], start=2):
        candidate = _alignment_signature(track)
        mismatches = {
            key: {"expected": reference[key], "actual": candidate[key]}
            for key in reference
            if candidate[key] != reference[key]
        }
        if mismatches:
            raise ValueError(
                f"segmentation track {index} is not exactly aligned: {mismatches}"
            )


def _contained_artifact_path(track_dir: Path, relative_path: str) -> Path:
    resolved_root = track_dir.resolve(strict=True)
    resolved = (resolved_root / relative_path).resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"track artifact escapes its directory: {relative_path}") from error
    return resolved


def _validate_frame_dimensions(path: Path, track: SegmentationTrack) -> None:
    with Image.open(path) as frame:
        if frame.size != (track.analysis_width, track.analysis_height):
            raise ValueError(f"analysis frame dimensions do not match track: {path}")


def _resolve_review_frames(
    *,
    resolved_track_paths: Sequence[Path],
    tracks: Sequence[SegmentationTrack],
    analysis_frames_dir: Path | None,
) -> tuple[Path, list[Path], str | None]:
    """Resolve independent or shared frames, fully validating shared provenance."""
    reference = tracks[0]
    if analysis_frames_dir is None:
        if any(
            track.shared_session_id is not None
            or track.analysis_frames_manifest_sha256 is not None
            for track in tracks
        ):
            raise ValueError(
                "shared SAM tracks require an explicit analysis_frames_dir so the "
                "immutable frame manifest can be validated"
            )
        frames_dir = (resolved_track_paths[0].parent / "analysis-frames").resolve()
        frame_paths: list[Path] = []
        for sample_index in range(reference.total_samples):
            frame_path = (frames_dir / f"{sample_index:06d}.jpg").resolve(strict=True)
            try:
                frame_path.relative_to(frames_dir)
            except ValueError as error:
                raise ValueError(f"analysis frame escapes its directory: {frame_path}") from error
            _validate_frame_dimensions(frame_path, reference)
            frame_paths.append(frame_path)
        return frames_dir, frame_paths, None

    frames_dir = analysis_frames_dir.expanduser().resolve(strict=True)
    if not frames_dir.is_dir():
        raise ValueError(f"analysis frames path is not a directory: {frames_dir}")
    session_ids = {track.shared_session_id for track in tracks}
    if None in session_ids or len(session_ids) != 1:
        raise ValueError(
            "explicit analysis frames require tracks from one shared SAM session"
        )
    expected_manifest_hashes = {
        track.analysis_frames_manifest_sha256 for track in tracks
    }
    if None in expected_manifest_hashes or len(expected_manifest_hashes) != 1:
        raise ValueError(
            "shared tracks must contain one matching analysis frame manifest hash"
        )
    expected_manifest_hash = next(iter(expected_manifest_hashes))
    assert expected_manifest_hash is not None
    manifest_path = (frames_dir.parent / "analysis-frames-manifest.json").resolve()
    try:
        manifest_path.relative_to(frames_dir.parent)
    except ValueError as error:
        raise ValueError("analysis frame manifest escapes the shared session directory") from error
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(f"analysis frame manifest hash mismatch: {manifest_path}")
    manifest = SharedSam21AnalysisFramesManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if len(manifest.frames) != reference.total_samples:
        raise ValueError("analysis frame manifest sample count does not match tracks")

    frame_paths = []
    for frame_record, sample in zip(manifest.frames, reference.samples, strict=True):
        if (
            frame_record.sample_index != sample.sample_index
            or frame_record.analysis_sample_time_ms != sample.analysis_sample_time_ms
            or frame_record.source_pts != sample.source_pts
        ):
            raise ValueError(
                "analysis frame manifest index/time/source PTS does not match tracks"
            )
        frame_path = (frames_dir.parent / frame_record.path).resolve(strict=True)
        try:
            frame_path.relative_to(frames_dir)
        except ValueError as error:
            raise ValueError(
                f"analysis frame path is outside the explicit directory: {frame_record.path}"
            ) from error
        if not frame_path.is_file():
            raise ValueError(f"analysis frame path is not a file: {frame_path}")
        if sha256_file(frame_path) != frame_record.sha256:
            raise ValueError(f"analysis frame hash mismatch: {frame_path}")
        _validate_frame_dimensions(frame_path, reference)
        frame_paths.append(frame_path)
    return frames_dir, frame_paths, actual_manifest_hash


def _render_overlay_frame(
    *,
    frame_path: Path,
    tracks: Sequence[SegmentationTrack],
    track_dirs: Sequence[Path],
    labels: Sequence[str],
    colors: Sequence[tuple[int, int, int]],
    sample_index: int,
    output_path: Path,
) -> None:
    with Image.open(frame_path).convert("RGBA") as source:
        image = source.copy()
    font = _overlay_font(max(12, round(min(image.size) / 35)))
    line_width = max(2, image.width // 400)

    for track, track_dir, label, color in zip(
        tracks, track_dirs, labels, colors, strict=True
    ):
        sample = track.samples[sample_index]
        if sample.mask_path is not None:
            mask_path = _contained_artifact_path(track_dir, sample.mask_path)
            if sample.mask_sha256 is None or sha256_file(mask_path) != sample.mask_sha256:
                raise ValueError(f"mask hash mismatch: {mask_path}")
            with Image.open(mask_path).convert("L") as source_mask:
                if source_mask.size != image.size:
                    raise ValueError(
                        f"mask dimensions do not match analysis frame: {mask_path}"
                    )
                alpha = source_mask.point(lambda value: 92 if value else 0)
            color_layer = Image.new("RGBA", image.size, (*color, 0))
            color_layer.putalpha(alpha)
            image = Image.alpha_composite(image, color_layer)

        box = sample.derived_tracking_box
        if box is None:
            continue
        x_min, y_min, x_max, y_max = box
        pixels = (
            round(x_min * image.width / 1000),
            round(y_min * image.height / 1000),
            round(x_max * image.width / 1000),
            round(y_max * image.height / 1000),
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle(pixels, outline=color, width=line_width)
        text = f"{label} | {sample.tracking_state.value}"
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = min(max(0, pixels[0]), max(0, image.width - text_width - 12))
        text_y = max(0, pixels[1] - text_height - 12)
        draw.rectangle(
            (text_x, text_y, text_x + text_width + 10, text_y + text_height + 8),
            fill=(*color, 220),
        )
        draw.text((text_x + 5, text_y + 3), text, fill="black", font=font)

    draw = ImageDraw.Draw(image)
    legend = "  ".join(f"{index + 1}. {label}" for index, label in enumerate(labels))
    legend_box = draw.textbbox((0, 0), legend, font=font)
    legend_height = legend_box[3] - legend_box[1]
    draw.rectangle((0, 0, image.width, legend_height + 18), fill="#101820cc")
    x = 12
    for label, color in zip(labels, colors, strict=True):
        item = f"■ {label}  "
        draw.text((x, 7), item, fill=color, font=font)
        item_box = draw.textbbox((0, 0), item, font=font)
        x += item_box[2] - item_box[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=90)


def _render_review_video(
    *,
    overlays_dir: Path,
    source_video: Path,
    output_path: Path,
    sample_times_ms: Sequence[int],
    display_fps: float,
    start_ms: int,
    end_ms: int,
    source_has_audio: bool,
) -> None:
    duration_seconds = (end_ms - start_ms) / 1000
    if not sample_times_ms:
        raise ValueError("multi-track review requires at least one sampled frame")
    if any(not start_ms <= value < end_ms for value in sample_times_ms):
        raise ValueError("sample times must remain inside the review interval")
    if any(
        current >= following
        for current, following in zip(sample_times_ms, sample_times_ms[1:])
    ):
        raise ValueError("sample times must be strictly increasing")

    # FFmpeg's concat demuxer preserves each decoded sample's real timeline gap.
    # The first selected frame is held back to the interval start; subsequent
    # frames begin at their actual analysis_sample_time_ms. Repeating the last
    # entry makes the final duration effective instead of silently truncating it.
    timeline_path = output_path.parent / "overlay-timeline.ffconcat"
    lines = ["ffconcat version 1.0"]
    for index, sample_time_ms in enumerate(sample_times_ms):
        frame_path = (overlays_dir / f"{index:06d}.jpg").resolve(strict=True)
        if "\n" in str(frame_path) or "\r" in str(frame_path):
            raise ValueError("overlay frame path contains a newline")
        escaped_path = str(frame_path).replace("'", "'\\''")
        next_time_ms = (
            sample_times_ms[index + 1]
            if index + 1 < len(sample_times_ms)
            else end_ms
        )
        frame_start_ms = start_ms if index == 0 else sample_time_ms
        frame_duration = (next_time_ms - frame_start_ms) / 1000
        if frame_duration <= 0:
            raise ValueError("overlay frame duration must be positive")
        lines.extend(
            [
                f"file '{escaped_path}'",
                "option framerate 1000",
                f"duration {frame_duration:.9f}",
            ]
        )
    last_path = (overlays_dir / f"{len(sample_times_ms) - 1:06d}.jpg").resolve(
        strict=True
    )
    escaped_last_path = str(last_path).replace("'", "'\\''")
    lines.extend([f"file '{escaped_last_path}'", "option framerate 1000"])
    timeline_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The output fps filter duplicates VFR review samples for broadly compatible
    # playback; it does not claim that SAM inferred at display_fps.
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(timeline_path),
        "-ss",
        f"{start_ms / 1000:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-vf",
        f"fps={display_fps},scale=in_range=pc:out_range=tv,format=yuv420p",
        "-t",
        f"{duration_seconds:.6f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "cfr",
    ]
    if source_has_audio:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.extend(["-movflags", "+faststart", str(output_path)])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"multi-track review render failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )


def _probe_review_video(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration:stream=codec_type,codec_name,pix_fmt,avg_frame_rate,"
                "duration,nb_frames"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not inspect rendered review video: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    video_streams = [
        stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"
    ]
    if len(video_streams) != 1:
        raise RuntimeError("rendered review must contain exactly one video stream")
    video = video_streams[0]
    rate = Fraction(str(video["avg_frame_rate"]))
    video_duration = video.get("duration")
    frame_count = video.get("nb_frames")
    if video_duration in {None, "N/A"} or frame_count in {None, "N/A"}:
        raise RuntimeError("rendered review lacks video duration or frame-count metadata")
    return {
        "duration_ms": round(float(payload["format"]["duration"]) * 1000),
        "video_duration_ms": round(float(video_duration) * 1000),
        "frame_count": int(frame_count),
        "codec_name": video.get("codec_name"),
        "pixel_format": video.get("pix_fmt"),
        "frame_rate": Rational(numerator=rate.numerator, denominator=rate.denominator),
        "audio_muxed": any(
            stream.get("codec_type") == "audio" for stream in payload.get("streams", [])
        ),
    }


def render_multi_segmentation_review(
    *,
    track_json_paths: Sequence[Path],
    labels: Sequence[str],
    output_dir: Path,
    display_fps: float = 30.0,
    analysis_frames_dir: Path | None = None,
) -> MultiSegmentationReviewManifest:
    """Combine aligned SAM tracks into one manual-review MP4."""
    if len(track_json_paths) < 2:
        raise ValueError("multi-track review requires at least two track JSON files")
    if len(labels) != len(track_json_paths):
        raise ValueError("provide exactly one label for each segmentation track")
    normalized_labels = [label.strip() for label in labels]
    if any(not label for label in normalized_labels):
        raise ValueError("multi-track review labels must be non-empty")
    if len(normalized_labels) != len(set(normalized_labels)):
        raise ValueError("multi-track review labels must be unique")
    if len(track_json_paths) > len(_REVIEW_COLORS):
        raise ValueError(f"multi-track review supports at most {len(_REVIEW_COLORS)} tracks")

    loaded = [_load_track(path) for path in track_json_paths]
    resolved_paths = [item[0] for item in loaded]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("segmentation track JSON paths must be unique")
    tracks = [item[1] for item in loaded]
    validate_segmentation_track_alignment(tracks)
    reference = tracks[0]
    if display_fps < reference.analysis_fps or display_fps > 60:
        raise ValueError("display_fps must be within [analysis_fps, 60]")

    source_video = Path(reference.video_path).expanduser().resolve(strict=True)
    source_media = probe_video(source_video)
    if source_media.asset_id != reference.asset_id:
        raise ValueError("source video content no longer matches the track asset_id")
    if reference.analysis_end_ms is None:
        raise ValueError("reference track has no bounded analysis_end_ms")
    if reference.analysis_end_ms > source_media.duration_ms:
        raise ValueError("track analysis interval exceeds source video duration")

    resolved_frames_dir, frame_paths, frames_manifest_sha256 = _resolve_review_frames(
        resolved_track_paths=resolved_paths,
        tracks=tracks,
        analysis_frames_dir=analysis_frames_dir,
    )

    output_dir = output_dir.expanduser().resolve()
    overlays_dir = output_dir / "overlays"
    if overlays_dir.exists() and any(overlays_dir.iterdir()):
        raise FileExistsError(f"combined overlay directory is not empty: {overlays_dir}")
    overlays_dir.mkdir(parents=True, exist_ok=True)
    colors = _REVIEW_COLORS[: len(tracks)]
    track_dirs = [path.parent for path in resolved_paths]
    for sample_index, frame_path in enumerate(frame_paths):
        _render_overlay_frame(
            frame_path=frame_path,
            tracks=tracks,
            track_dirs=track_dirs,
            labels=normalized_labels,
            colors=colors,
            sample_index=sample_index,
            output_path=overlays_dir / f"{sample_index:06d}.jpg",
        )

    output_video = output_dir / "multi-track-review.mp4"
    source_has_audio = has_audio_stream(source_video)
    _render_review_video(
        overlays_dir=overlays_dir,
        source_video=source_video,
        output_path=output_video,
        sample_times_ms=[sample.analysis_sample_time_ms for sample in reference.samples],
        display_fps=display_fps,
        start_ms=reference.analysis_start_ms,
        end_ms=reference.analysis_end_ms,
        source_has_audio=source_has_audio,
    )
    rendered = _probe_review_video(output_video)
    if rendered["codec_name"] != "h264" or rendered["pixel_format"] != "yuv420p":
        raise RuntimeError("review output is not H.264 yuv420p")
    if rendered["audio_muxed"] != source_has_audio:
        raise RuntimeError("review output audio presence does not match the source interval")
    expected_duration_ms = reference.analysis_end_ms - reference.analysis_start_ms
    duration_tolerance_ms = math.ceil(1000 / display_fps) + 5
    if abs(int(rendered["video_duration_ms"]) - expected_duration_ms) > duration_tolerance_ms:
        raise RuntimeError("review video stream does not cover the complete analysis interval")
    minimum_frames = max(1, math.floor(expected_duration_ms * display_fps / 1000) - 1)
    if int(rendered["frame_count"]) < minimum_frames:
        raise RuntimeError("review video stream contains too few frames for its interval")

    members = [
        MultiSegmentationReviewMember(
            label=label,
            color_rgb=color,
            target_description=track.target_description,
            track_json_path=str(path),
            track_json_sha256=sha256_file(path),
            seed_time_ms=track.seed_time_ms,
        )
        for path, track, label, color in zip(
            resolved_paths, tracks, normalized_labels, colors, strict=True
        )
    ]
    manifest = MultiSegmentationReviewManifest(
        artifact_type="multi_segmentation_track_review",
        interpretation="manual_review_visualization_not_accuracy",
        asset_id=reference.asset_id,
        source_video_path=str(source_video),
        source_video_sha256=source_media.sha256,
        analysis_fps=reference.analysis_fps,
        display_fps=display_fps,
        analysis_width=reference.analysis_width,
        analysis_height=reference.analysis_height,
        analysis_start_ms=reference.analysis_start_ms,
        analysis_end_ms=reference.analysis_end_ms,
        total_samples=reference.total_samples,
        analysis_frames_dir=str(resolved_frames_dir),
        analysis_frames_manifest_sha256=frames_manifest_sha256,
        audio_muxed=bool(rendered["audio_muxed"]),
        output_video_path=str(output_video),
        output_video_sha256=sha256_file(output_video),
        output_duration_ms=int(rendered["duration_ms"]),
        output_video_duration_ms=int(rendered["video_duration_ms"]),
        output_frame_count=int(rendered["frame_count"]),
        output_codec_name="h264",
        output_pixel_format="yuv420p",
        output_frame_rate=rendered["frame_rate"],
        warning=_REVIEW_WARNING,
        generated_at=utc_now(),
        members=members,
    )
    write_json(output_dir / "multi-track-review.json", manifest)
    return manifest


def _mask_bits(
    *, track_dir: Path, sample: SegmentationSample, width: int, height: int
) -> bytes | None:
    mask_path_value = sample.mask_path
    if mask_path_value is None:
        return None
    mask_path = _contained_artifact_path(track_dir, mask_path_value)
    expected_hash = sample.mask_sha256
    if expected_hash is None or sha256_file(mask_path) != expected_hash:
        raise ValueError(f"mask hash mismatch: {mask_path}")
    with Image.open(mask_path).convert("L") as mask:
        if mask.size != (width, height):
            raise ValueError(f"mask dimensions do not match track: {mask_path}")
        binary = mask.point(lambda value: 255 if value else 0).convert("1")
        bits = binary.tobytes()
    if not any(bits):
        raise ValueError(f"non-empty mask artifact contains no positive pixels: {mask_path}")
    return bits


def _mask_iou(left: bytes | None, right: bytes | None) -> float | None:
    if left is None and right is None:
        return None
    if left is None or right is None:
        return 0.0
    if len(left) != len(right):
        raise ValueError("aligned mask buffers have different lengths")
    intersection = sum((a & b).bit_count() for a, b in zip(left, right, strict=True))
    union = sum((a | b).bit_count() for a, b in zip(left, right, strict=True))
    if union == 0:
        raise ValueError("non-empty aligned masks have an empty union")
    return intersection / union


def compare_aligned_segmentation_tracks(
    track_a_json_path: Path,
    track_b_json_path: Path,
    output_path: Path,
) -> SegmentationTrackAgreementReport:
    """Measure symmetric agreement between two exactly aligned mask tracks."""
    path_a, track_a = _load_track(track_a_json_path)
    path_b, track_b = _load_track(track_b_json_path)
    if path_a == path_b:
        raise ValueError("track A and track B JSON paths must be different")
    validate_segmentation_track_alignment([track_a, track_b])
    source_video = Path(track_a.video_path).expanduser().resolve(strict=True)
    if probe_video(source_video).asset_id != track_a.asset_id:
        raise ValueError("source video content no longer matches the track asset_id")

    rows: list[SegmentationTrackAgreementSample] = []
    for sample_a, sample_b in zip(track_a.samples, track_b.samples, strict=True):
        bits_a = _mask_bits(
            track_dir=path_a.parent,
            sample=sample_a,
            width=track_a.analysis_width,
            height=track_a.analysis_height,
        )
        bits_b = _mask_bits(
            track_dir=path_b.parent,
            sample=sample_b,
            width=track_b.analysis_width,
            height=track_b.analysis_height,
        )
        mask_iou = _mask_iou(bits_a, bits_b)
        has_both_boxes = (
            sample_a.derived_tracking_box is not None
            and sample_b.derived_tracking_box is not None
        )
        bbox_agreement = (
            box_iou(sample_a.derived_tracking_box, sample_b.derived_tracking_box)
            if has_both_boxes
            else None
        )
        distance = (
            center_distance(sample_a.derived_tracking_box, sample_b.derived_tracking_box)
            if has_both_boxes
            else None
        )
        rows.append(
            SegmentationTrackAgreementSample(
                sample_index=sample_a.sample_index,
                analysis_sample_time_ms=sample_a.analysis_sample_time_ms,
                source_pts=sample_a.source_pts,
                tracking_state_a=sample_a.tracking_state,
                tracking_state_b=sample_b.tracking_state,
                state_agreement=sample_a.tracking_state == sample_b.tracking_state,
                mask_iou=round(mask_iou, 6) if mask_iou is not None else None,
                bbox_iou=(
                    round(bbox_agreement, 6) if bbox_agreement is not None else None
                ),
                center_distance_normalized=(
                    round(distance, 6) if distance is not None else None
                ),
            )
        )

    mask_ious = [row.mask_iou for row in rows if row.mask_iou is not None]
    bbox_ious = [row.bbox_iou for row in rows if row.bbox_iou is not None]
    distances = [
        row.center_distance_normalized
        for row in rows
        if row.center_distance_normalized is not None
    ]
    state_agreement_samples = sum(row.state_agreement for row in rows)
    report = SegmentationTrackAgreementReport(
        artifact_type="segmentation_track_agreement_report",
        interpretation="peer_agreement_not_accuracy",
        asset_id=track_a.asset_id,
        track_a_path=str(path_a),
        track_a_sha256=sha256_file(path_a),
        track_a_target_description=track_a.target_description,
        track_b_path=str(path_b),
        track_b_sha256=sha256_file(path_b),
        track_b_target_description=track_b.target_description,
        total_samples=len(rows),
        mask_iou_samples=len(mask_ious),
        mean_mask_iou=(round(sum(mask_ious) / len(mask_ious), 6) if mask_ious else None),
        bbox_iou_samples=len(bbox_ious),
        mean_bbox_iou=(round(sum(bbox_ious) / len(bbox_ious), 6) if bbox_ious else None),
        center_distance_samples=len(distances),
        mean_center_distance_normalized=(
            round(sum(distances) / len(distances), 6) if distances else None
        ),
        state_agreement_samples=state_agreement_samples,
        state_agreement_rate=round(state_agreement_samples / len(rows), 6),
        warning=_AGREEMENT_WARNING,
        generated_at=utc_now(),
        samples=rows,
    )
    write_json(output_path.expanduser().resolve(), report)
    return report
