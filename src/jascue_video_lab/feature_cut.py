from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

from .billing import summarize_usage_and_list_price
from .gemini import GeminiLabClient, MODEL_ID, VISUAL_EVIDENCE_SYSTEM_INSTRUCTION
from .grounding_selection import (
    require_grounding_request_match,
    require_tracking_seed_candidate,
)
from .media import extract_frame, has_audio_stream, probe_video, sha256_file
from .models import (
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    GeminiNativeGroundingProposal,
    GroundingProposal,
    RushClip,
    RushFrame,
    RushesCatalog,
    SegmentationTrack,
    TrackingState,
)
from .overlay import draw_grounding_overlay
from .rushes import _segment_bounds
from .sam_tracking import (
    SAM21_CONFIG,
    SAM21_IMPLEMENTATION_REVISION,
    require_bbox_track_request_match,
    track_bbox_sam21,
)
from .schema import gemini_response_schema
from .shots import ShotManifest, detect_shots_ffmpeg
from .storage import read_json, utc_now, write_json


_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)
_RENDER_PIPELINE_VERSION = "feature-cut-v2-primary-center-atomic"
_TRACKING_MAX_SIDE = 960
_TRACKING_DEVICE = "cpu"
_TRACKING_SEED_BOX_PADDING_RATIO = 0.04


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _render_text_layer(
    chapter: FeatureChapterBrief,
    output_path: Path,
    *,
    dimensions: tuple[int, int],
    missing_evidence: bool = False,
    opaque: bool = False,
) -> None:
    width, height = dimensions
    image = Image.new("RGBA", dimensions, (11, 14, 18, 255 if opaque else 0))
    draw = ImageDraw.Draw(image)
    title_font = _font(54 if width > height else 48)
    detail_font = _font(34 if width > height else 31)
    label_font = _font(23 if width > height else 24)
    panel_height = round(height * (0.35 if width < height else 0.30))
    top = height - panel_height
    draw.rectangle((0, top, width, height), fill=(8, 12, 16, 218 if not opaque else 255))
    draw.rectangle((0, top, 14 if width > height else 10, height), fill=(29, 196, 96, 255))
    margin = 64 if width > height else 48
    y = top + 36
    for line in _wrap_text(draw, chapter.title, title_font, width - margin * 2):
        draw.text((margin, y), line, font=title_font, fill="white")
        y += title_font.size + 9
    y += 5
    for detail in chapter.detail_lines:
        for line in _wrap_text(draw, detail, detail_font, width - margin * 2):
            draw.text((margin, y), line, font=detail_font, fill=(220, 231, 225, 255))
            y += detail_font.size + 6
    if missing_evidence:
        label = "CATALOG 中未找到直接功能示範畫面"
        box = draw.textbbox((0, 0), label, font=label_font)
        label_width = box[2] - box[0] + 28
        draw.rounded_rectangle(
            (margin, max(22, top - 58), margin + label_width, max(22, top - 58) + 42),
            radius=10,
            fill=(211, 70, 70, 235),
        )
        draw.text((margin + 14, max(22, top - 51)), label, font=label_font, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _chapter_bounds(
    frame: RushFrame,
    clip: RushClip,
    duration_seconds: float,
    shot_cache: dict[str, ShotManifest],
    shots_dir: Path,
    scdet_threshold: float,
) -> tuple[int, int, str]:
    if clip.clip_id not in shot_cache:
        shot_cache[clip.clip_id] = detect_shots_ffmpeg(
            Path(clip.path),
            threshold=scdet_threshold,
            output_path=shots_dir / f"{clip.clip_id}.json",
        )
    shot = next(
        item
        for item in shot_cache[clip.clip_id].shots
        if item.start_time_ms <= frame.requested_time_ms < item.end_time_ms
    )
    start_ms, end_ms = _segment_bounds(
        center_ms=frame.requested_time_ms,
        requested_duration_ms=round(duration_seconds * 1000),
        clip_duration_ms=clip.duration_ms,
        shot=shot,
    )
    return start_ms, end_ms, shot.shot_id


def _run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _run_segment_encoder(command: list[str]) -> None:
    try:
        _run_ffmpeg(command)
    except subprocess.CalledProcessError:
        if "h264_videotoolbox" not in command:
            raise
        fallback = ["libx264" if value == "h264_videotoolbox" else value for value in command]
        _run_ffmpeg(fallback)


def _render_source_segment(
    *,
    source_path: Path,
    start_ms: int,
    end_ms: int,
    overlay_path: Path | None,
    base_filter: str,
    output_path: Path,
    source_has_audio: bool | None = None,
) -> str:
    duration = (end_ms - start_ms) / 1000
    audio_fade_out = max(0.0, duration - 0.12)
    if source_has_audio is None:
        source_has_audio = has_audio_stream(source_path)
    if overlay_path is None:
        filter_graph = base_filter + ";[base]null[v]"
        overlay_input: list[str] = []
    else:
        filter_graph = (
            base_filter
            + ";[1:v]format=rgba[card];"
            + "[base][card]overlay=0:0:shortest=1[v]"
        )
        overlay_input = ["-loop", "1", "-i", str(overlay_path)]
    if source_has_audio:
        audio_input: list[str] = []
        audio_map = "0:a:0"
        audio_origin = "source"
    else:
        silence_input_index = 2 if overlay_path is not None else 1
        audio_input = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        audio_map = f"{silence_input_index}:a:0"
        audio_origin = "synthetic_silence"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.partial.mp4")
    _run_segment_encoder(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-i",
            str(source_path),
            *overlay_input,
            *audio_input,
            "-t",
            f"{duration:.3f}",
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            audio_map,
            "-af",
            (
                "volume=0.58,afade=t=in:st=0:d=0.08,"
                f"afade=t=out:st={audio_fade_out:.3f}:d=0.12"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "8M",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]
    )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temporary_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ]
    )
    temporary_path.replace(output_path)
    return audio_origin


def _render_missing_segment(
    chapter: FeatureChapterBrief,
    output_path: Path,
    overlay_path: Path,
    dimensions: tuple[int, int],
) -> None:
    _render_text_layer(
        chapter,
        overlay_path,
        dimensions=dimensions,
        missing_evidence=True,
        opaque=True,
    )
    duration = chapter.target_duration_seconds
    _run_segment_encoder(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(overlay_path),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            f"{duration:.3f}",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "8M",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            str(output_path),
        ]
    )


def _concat_segments(segment_paths: Sequence[Path], output_path: Path) -> None:
    if not segment_paths:
        raise ValueError("cannot concatenate an empty segment list")
    inputs: list[str] = []
    filter_inputs: list[str] = []
    for index, path in enumerate(segment_paths):
        inputs.extend(["-i", str(path.resolve())])
        filter_inputs.extend([f"[{index}:v:0]", f"[{index}:a:0]"])
    filter_graph = "".join(filter_inputs) + f"concat=n={len(segment_paths)}:v=1:a=1[v][a]"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.partial.mp4")
    _run_segment_encoder(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "8M",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]
    )
    expected_duration = sum(_probe_duration_seconds(path) for path in segment_paths)
    actual_duration = _probe_duration_seconds(temporary_path)
    if abs(actual_duration - expected_duration) > 0.25:
        raise RuntimeError(
            f"assembled duration mismatch: expected={expected_duration:.3f}s "
            f"actual={actual_duration:.3f}s"
        )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temporary_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ]
    )
    temporary_path.replace(output_path)


def _output_media_metadata(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_name,codec_type,width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio = next(
        (stream for stream in payload["streams"] if stream["codec_type"] == "audio"),
        None,
    )
    return {
        "sha256": sha256_file(path),
        "duration_seconds": float(payload["format"]["duration"]),
        "size_bytes": int(payload["format"]["size"]),
        "video_codec": video["codec_name"],
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_rate": video["r_frame_rate"],
        "video_frames": int(video["nb_frames"]),
        "has_audio": audio is not None,
        "audio_codec": audio["codec_name"] if audio is not None else None,
    }


def _probe_duration_seconds(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(completed.stdout)["format"]["duration"])


def _segment_is_valid(
    path: Path, *, expected_duration: float, dimensions: tuple[int, int]
) -> bool:
    if not path.exists():
        return False
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return False
    try:
        payload = json.loads(probe.stdout)
        stream = payload["streams"][0]
        duration = float(payload["format"]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if (stream["width"], stream["height"]) != dimensions:
        return False
    if abs(duration - expected_duration) > 0.15:
        return False
    decode = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    return decode.returncode == 0


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _track_geometry_fingerprint(track: SegmentationTrack) -> str:
    """Fingerprint every consumed tracking sample and its model/source provenance."""
    return _stable_fingerprint(track.model_dump(mode="json"))


def _segment_variant_fingerprint(
    *,
    source_sha256: str,
    start_ms: int,
    end_ms: int,
    filter_graph: str,
    geometry: dict[str, Any],
    track_fingerprint: str | None,
) -> str:
    return _stable_fingerprint(
        {
            "contract_version": "feature-segment-render-v2",
            "source_sha256": source_sha256,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "filter_graph": filter_graph,
            "geometry": geometry,
            "track_fingerprint": track_fingerprint,
        }
    )


def _usable_track_centers(track: SegmentationTrack) -> tuple[list[float], list[float], list[list[int]]]:
    usable_states = {TrackingState.TRACKED, TrackingState.LOW_CONFIDENCE}
    times: list[float] = []
    centers: list[float] = []
    boxes: list[list[int]] = []
    for sample in track.samples:
        if (
            sample.tracking_state in usable_states
            and sample.center_2d is not None
            and sample.derived_tracking_box is not None
        ):
            times.append((sample.analysis_sample_time_ms - track.analysis_start_ms) / 1000)
            centers.append(float(sample.center_2d[0]))
            boxes.append([int(value) for value in sample.derived_tracking_box])
    return times, centers, boxes


def _smooth(values: Sequence[float], alpha: float = 0.34) -> list[float]:
    if not values:
        return []
    forward = [float(values[0])]
    for value in values[1:]:
        forward.append(alpha * float(value) + (1 - alpha) * forward[-1])
    backward = [forward[-1]]
    for value in reversed(forward[:-1]):
        backward.append(alpha * value + (1 - alpha) * backward[-1])
    return list(reversed(backward))


def _piecewise_expression(times: Sequence[float], values: Sequence[float]) -> str:
    if not times or len(times) != len(values):
        raise ValueError("crop expression needs aligned non-empty times and values")
    if len(times) == 1:
        return f"{values[0]:.3f}"
    expression = f"{values[-1]:.3f}"
    for index in range(len(times) - 2, -1, -1):
        t0, t1 = times[index], times[index + 1]
        x0, x1 = values[index], values[index + 1]
        delta = max(0.001, t1 - t0)
        linear = f"{x0:.3f}+({x1 - x0:.3f})*(t-{t0:.3f})/{delta:.3f}"
        expression = f"if(lt(t\\,{t1:.3f})\\,{linear}\\,{expression})"
    return expression


def _horizontal_filter_from_track(
    track: SegmentationTrack, zoom_intent: str
) -> tuple[str, dict[str, Any]]:
    times, centers_x, boxes = _usable_track_centers(track)
    if len(times) < 2:
        return _horizontal_original_filter(), {
            "requested_zoom": None,
            "geometry_safe_max_zoom": None,
            "applied_zoom": 1.0,
            "fallback_reason": "fewer_than_two_usable_tracking_samples",
        }
    requested = {"subtle": 1.12, "detail": 1.35}[zoom_intent]
    max_width = max(box[2] - box[0] for box in boxes)
    max_height = max(box[3] - box[1] for box in boxes)
    safe_max = min(2.0, 1000 / (max_width * 1.45), 1000 / (max_height * 1.45))
    applied = max(1.0, min(requested, safe_max))
    if applied < 1.035:
        return _horizontal_original_filter(), {
            "requested_zoom": requested,
            "geometry_safe_max_zoom": round(safe_max, 4),
            "applied_zoom": 1.0,
            "fallback_reason": "mask_geometry_left_no_safe_zoom_margin",
        }
    scaled_width = int(math.ceil(1920 * applied / 2) * 2)
    scaled_height = int(math.ceil(1080 * applied / 2) * 2)
    smooth_x = _smooth(centers_x)
    centers_y = [
        (box[1] + box[3]) / 2
        for box in boxes
    ]
    smooth_y = _smooth(centers_y)
    x_values = [
        max(0.0, min(scaled_width - 1920, center * scaled_width / 1000 - 960))
        for center in smooth_x
    ]
    y_values = [
        max(0.0, min(scaled_height - 1080, center * scaled_height / 1000 - 540))
        for center in smooth_y
    ]
    x_expression = _piecewise_expression(times, x_values)
    y_expression = _piecewise_expression(times, y_values)
    return (
        f"[0:v]fps=30,scale={scaled_width}:{scaled_height},"
        f"crop=1920:1080:x='{x_expression}':y='{y_expression}',setsar=1[base]",
        {
            "requested_zoom": requested,
            "geometry_safe_max_zoom": round(safe_max, 4),
            "applied_zoom": round(applied, 4),
            "fallback_reason": None,
        },
    )


def _horizontal_original_filter() -> str:
    return (
        "[0:v]fps=30,scale=1920:1080:force_original_aspect_ratio=increase,"
        "crop=1920:1080,setsar=1[base]"
    )


def _vertical_filter_from_track(
    track: SegmentationTrack,
    *,
    allow_subject_clipping: bool = False,
) -> tuple[str, dict[str, Any]]:
    times, centers_x, boxes = _usable_track_centers(track)
    if len(times) < 2:
        return _vertical_fit_filter(), {
            "applied_strategy": "fit_with_background",
            "fallback_reason": "fewer_than_two_usable_tracking_samples",
        }
    scaled_width = 3414
    crop_width_normalized = 1080 * 1000 / scaled_width
    max_target_width = max(box[2] - box[0] for box in boxes)
    if not allow_subject_clipping and max_target_width * 1.08 > crop_width_normalized:
        return _vertical_fit_filter(), {
            "applied_strategy": "fit_with_background",
            "fallback_reason": "tracked_subject_too_wide_for_safe_9x16_crop",
            "subject_clipping_allowed": False,
        }
    x_values = [
        max(0.0, min(scaled_width - 1080, center * scaled_width / 1000 - 540))
        for center in _smooth(centers_x)
    ]
    x_expression = _piecewise_expression(times, x_values)
    return (
        f"[0:v]fps=30,scale={scaled_width}:1920,"
        f"crop=1080:1920:x='{x_expression}':y=0,setsar=1[base]",
        {
            "applied_strategy": "tracked_crop",
            "fallback_reason": None,
            "subject_clipping_allowed": allow_subject_clipping,
        },
    )


def _vertical_fit_filter() -> str:
    return (
        "[0:v]fps=30,split=2[background_source][foreground_source];"
        "[background_source]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,gblur=sigma=28[background];"
        "[foreground_source]scale=1080:-2[foreground];"
        "[background][foreground]overlay=(W-w)/2:(H-h)/2,setsar=1[base]"
    )


def _build_track(
    *,
    client: GeminiLabClient,
    clip: RushClip,
    frame: RushFrame,
    start_ms: int,
    end_ms: int,
    feature_id: str,
    event_description: str,
    target_description: str,
    checkpoint_path: Path,
    grounding_prompt: str,
    output_dir: Path,
    run_id: str,
    analysis_fps: float,
    scdet_threshold: float,
) -> tuple[GroundingProposal, SegmentationTrack]:
    track_root = output_dir / "sam21"
    exact_frame_path = output_dir / "evidence-frame.png"
    exact_frame = extract_frame(Path(clip.path), frame.requested_time_ms, exact_frame_path)
    media = probe_video(Path(clip.path))
    grounding_key = {
        "contract_version": "exact-frame-grounding-v2",
        "model_id": MODEL_ID,
        "source_asset_id": media.asset_id,
        "temperature": client.temperature,
        "feature_id": feature_id,
        "frame_hash": exact_frame.frame_hash,
        "frame_pts": exact_frame.frame_pts,
        "frame_time_ms": exact_frame.frame_time_ms,
        "source_width": exact_frame.width,
        "source_height": exact_frame.height,
        "entity_id": "reframe_subject",
        "event_description": event_description,
        "target_description": target_description,
        "prompt_sha256": hashlib.sha256(grounding_prompt.encode("utf-8")).hexdigest(),
        "system_instruction_sha256": hashlib.sha256(
            VISUAL_EVIDENCE_SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "response_schema_sha256": hashlib.sha256(
            json.dumps(
                gemini_response_schema(GeminiNativeGroundingProposal),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "thinking_level": "low",
    }
    grounding_fingerprint = hashlib.sha256(
        json.dumps(
            grounding_key,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    grounding_dir = output_dir / "grounding" / f"bbox-{grounding_fingerprint[:16]}"
    write_json(
        grounding_dir / "request-key.json",
        {**grounding_key, "request_fingerprint": grounding_fingerprint},
    )
    grounding_path = grounding_dir / "grounding.json"
    if grounding_path.exists():
        proposal = GroundingProposal.model_validate(read_json(grounding_path))
        require_grounding_request_match(
            proposal,
            asset_id=media.asset_id,
            event_id=feature_id,
            entity_id="reframe_subject",
            frame_pts=exact_frame.frame_pts,
            frame_time_ms=exact_frame.frame_time_ms,
            frame_hash=exact_frame.frame_hash,
            source_width=exact_frame.width,
            source_height=exact_frame.height,
            model_id=MODEL_ID,
        )
        frame_time_ms = proposal.frame_time_ms
    else:
        proposal = client.ground_frame(
            media=media,
            frame=exact_frame,
            event_id=feature_id,
            event_description=event_description,
            entity_id="reframe_subject",
            target_description=target_description,
            prompt_template=grounding_prompt,
            run_id=run_id,
            output_dir=grounding_dir,
        )
        frame_time_ms = exact_frame.frame_time_ms
    debug_path = grounding_dir / "debug.png"
    if not debug_path.exists():
        draw_grounding_overlay(exact_frame_path, proposal, debug_path)
    if debug_path.exists():
        shutil.copy2(debug_path, output_dir / "grounding-debug.png")
    if not proposal.visible or not proposal.candidates:
        raise ValueError(f"Gemini could not ground reframe subject for {feature_id}")
    selected_seed = require_tracking_seed_candidate(proposal)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    seed_manifest = {
        "contract_version": "bbox-seed-v2-exact-pts",
        "asset_id": proposal.asset_id,
        "event_id": proposal.event_id,
        "entity_id": proposal.entity_id,
        "target_description": target_description,
        "frame_hash": proposal.frame_hash,
        "frame_pts": proposal.frame_pts,
        "candidate_number": selected_seed.candidate_number,
        "candidate_index": selected_seed.candidate_index,
        "candidate_selection_source": selected_seed.selection_source,
        "box_2d": list(selected_seed.candidate.box_2d),
        "seed_type": "gemini_bbox",
        "source_start_ms": start_ms,
        "source_end_ms": end_ms,
        "normalized_seed_shot_start_ms": start_ms,
        "normalized_seed_shot_end_ms": end_ms,
        "analysis_fps": analysis_fps,
        "analysis_max_side": _TRACKING_MAX_SIDE,
        "ffmpeg_scdet_threshold": scdet_threshold,
        "seed_box_padding_ratio": _TRACKING_SEED_BOX_PADDING_RATIO,
        "device_request": _TRACKING_DEVICE,
        "sam_config": SAM21_CONFIG,
        "sam_implementation_revision": SAM21_IMPLEMENTATION_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
    }
    seed_fingerprint = hashlib.sha256(
        json.dumps(seed_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    track_dir = track_root / f"bbox-{seed_fingerprint[:16]}"
    seed_manifest_path = track_dir / "seed-selection.json"
    write_json(seed_manifest_path, {**seed_manifest, "seed_fingerprint": seed_fingerprint})
    track_path = track_dir / "segmentation-track.json"
    if track_path.exists():
        track = SegmentationTrack.model_validate(read_json(track_path))
    else:
        track = track_bbox_sam21(
            video_path=Path(clip.path),
            checkpoint_path=checkpoint_path,
            seed_time_ms=frame_time_ms,
            seed_box_2d=selected_seed.candidate.box_2d,
            target_description=target_description,
            output_dir=track_dir,
            seed_source=str(seed_manifest_path),
            asset_id=proposal.asset_id,
            seed_frame_pts=proposal.frame_pts,
            seed_frame_sha256=proposal.frame_hash,
            seed_source_width=proposal.source_width,
            seed_source_height=proposal.source_height,
            analysis_fps=analysis_fps,
            max_side=_TRACKING_MAX_SIDE,
            device=_TRACKING_DEVICE,
            ffmpeg_scdet_threshold=scdet_threshold,
            seed_box_padding_ratio=_TRACKING_SEED_BOX_PADDING_RATIO,
            allowed_start_ms=start_ms,
            allowed_end_ms=end_ms,
        )
    require_bbox_track_request_match(
        track,
        video_path=Path(clip.path),
        asset_id=proposal.asset_id,
        target_description=target_description,
        seed_time_ms=frame_time_ms,
        seed_box_2d=selected_seed.candidate.box_2d,
        seed_box_padding_ratio=_TRACKING_SEED_BOX_PADDING_RATIO,
        analysis_fps=analysis_fps,
        analysis_start_ms=start_ms,
        analysis_end_ms=end_ms,
        checkpoint_sha256=checkpoint_sha256,
        seed_frame_pts=proposal.frame_pts,
        seed_frame_sha256=proposal.frame_hash,
        seed_source_width=proposal.source_width,
        seed_source_height=proposal.source_height,
    )
    return proposal, track


def _render_review_html(
    output_dir: Path,
    brief: FeatureEditBrief,
    plan: FeatureEditPlan,
    manifest: dict[str, Any],
) -> None:
    overlay_note = (
        "成片不燒錄實驗字卡；使用者 brief 只作審查 metadata。"
        if not brief.render_title_overlays
        else "成片字卡來自使用者 editorial brief。"
    )
    rows: list[str] = []
    by_id = {chapter.feature_id: chapter for chapter in plan.chapters}
    for brief_chapter in brief.chapters:
        selected = by_id[brief_chapter.feature_id]
        vertical = next(
            item for item in manifest["vertical"]["chapters"] if item["feature_id"] == brief_chapter.feature_id
        )
        horizontal = next(
            item for item in manifest["horizontal"]["chapters"] if item["feature_id"] == brief_chapter.feature_id
        )
        debug_path = vertical.get("grounding_debug")
        debug_link = (
            f'<a href="{html.escape(str(Path(debug_path).relative_to(output_dir.resolve())))}">bbox</a>'
            if debug_path
            else "—"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(brief_chapter.title)}</td>"
            f"<td>{html.escape(selected.evidence_status)}</td>"
            f"<td>{html.escape(str(selected.horizontal_frame_id))}</td>"
            f"<td>{html.escape(str(horizontal.get('applied_zoom', 1.0)))}</td>"
            f"<td>{html.escape(str(selected.vertical_frame_id))}</td>"
            f"<td>{html.escape(vertical['applied_strategy'])}</td>"
            f"<td>{debug_link}</td>"
            f"<td>{html.escape(selected.observed_visual_evidence)}</td>"
            f"<td>{html.escape('; '.join(selected.quality_risks) or 'none')}</td>"
            "</tr>"
        )
    (output_dir / "index.html").write_text(
        """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Feature cut review</title>
<style>body{font:15px system-ui;background:#101214;color:#eee;max-width:1500px;margin:24px auto;padding:0 20px}section{background:#1b1f24;padding:20px;margin:20px 0;border-radius:12px}video{width:min(100%,960px);max-height:76vh;background:#000}table{border-collapse:collapse;width:100%}th,td{border:1px solid #3b424a;padding:8px;text-align:left;vertical-align:top}a{color:#71e59c}</style>
<h1>Feature cut review</h1><p>"""
        + html.escape(overlay_note)
        + " 畫面證據、frame ID、Gemini bbox、SAM tracking 與 fallback 分開保存。</p>"
        + f"<section><h2>16:9</h2><video controls src=\"{html.escape(str(Path(manifest['horizontal']['output_path']).relative_to(output_dir.resolve())))}\"></video></section>"
        + f"<section><h2>9:16</h2><video controls src=\"{html.escape(str(Path(manifest['vertical']['output_path']).relative_to(output_dir.resolve())))}\"></video></section>"
        + "<table><thead><tr><th>chapter</th><th>evidence</th><th>16:9 frame</th><th>zoom</th><th>9:16 frame</th><th>vertical</th><th>debug</th><th>observed evidence</th><th>risks</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></html>",
        encoding="utf-8",
    )


def run_feature_cut_experiment(
    *,
    catalog_path: Path,
    brief_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    plan_prompt: str,
    grounding_prompt: str,
    temperature: float = 0.2,
    scdet_threshold: float = 4.0,
    sam_analysis_fps: float = 2.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    brief = FeatureEditBrief.model_validate(read_json(brief_path))
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    brief_by_id = {chapter.feature_id: chapter for chapter in brief.chapters}
    timings: dict[str, float] = {}
    started = monotonic()
    reel_path = Path(catalog.analysis_reel_path)
    reel_media = probe_video(reel_path)
    upload_dir = catalog_path.parent / "file-cache" / reel_media.sha256 / "upload"
    client = GeminiLabClient(temperature=temperature)
    plan_dir = output_dir / "gemini-plan"
    try:
        stage = monotonic()
        uploaded, reused = client.ensure_video_upload(reel_path, upload_dir)
        timings["file_api_seconds"] = round(monotonic() - stage, 3)
        plan_path = plan_dir / "feature_edit_plan.json"
        if plan_path.exists():
            plan = FeatureEditPlan.model_validate(read_json(plan_path))
            expected_ids = [chapter.feature_id for chapter in brief.chapters]
            actual_ids = [chapter.feature_id for chapter in plan.chapters]
            if (
                plan.project_id != brief.project_id
                or plan.catalog_id != catalog.catalog_id
                or actual_ids != expected_ids
            ):
                raise ValueError("saved feature plan does not match the current brief/catalog")
            timings["gemini_plan_seconds"] = 0.0
            plan_reused = True
        else:
            stage = monotonic()
            plan = client.plan_feature_edit(
                catalog=catalog,
                brief=brief,
                uploaded=uploaded,
                prompt_template=plan_prompt,
                run_id=f"feature-plan-{uuid.uuid4().hex[:8]}",
                run_dir=plan_dir,
            )
            timings["gemini_plan_seconds"] = round(monotonic() - stage, 3)
            plan_reused = False
        shot_cache: dict[str, ShotManifest] = {}
        shots_dir = output_dir / "shots"
        horizontal_segments: list[Path] = []
        vertical_segments: list[Path] = []
        render_config = {
            "pipeline_version": _RENDER_PIPELINE_VERSION,
            "brief": brief.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "sam_analysis_fps": sam_analysis_fps,
            "scdet_threshold": scdet_threshold,
        }
        render_key = hashlib.sha256(
            json.dumps(render_config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        render_variant = (
            f"with-titles-{render_key}"
            if brief.render_title_overlays
            else f"clean-{render_key}"
        )
        manifest: dict[str, Any] = {
            "project_id": brief.project_id,
            "catalog_id": catalog.catalog_id,
            "render_title_overlays": brief.render_title_overlays,
            "render_pipeline_version": _RENDER_PIPELINE_VERSION,
            "render_cache_key": render_key,
            "horizontal": {"chapters": []},
            "vertical": {"chapters": []},
        }
        track_cache: dict[tuple[str, str, int, int], tuple[GroundingProposal, SegmentationTrack, Path]] = {}
        source_audio_cache: dict[str, bool] = {}
        stage = monotonic()
        for index, selected in enumerate(plan.chapters):
            brief_chapter = brief_by_id[selected.feature_id]
            horizontal_overlay = output_dir / "overlays" / "16x9" / f"{index:02d}.png"
            vertical_overlay = output_dir / "overlays" / "9x16" / f"{index:02d}.png"
            horizontal_segment = (
                output_dir / "segments" / render_variant / "16x9" / f"{index:02d}.mp4"
            )
            vertical_segment = (
                output_dir / "segments" / render_variant / "9x16" / f"{index:02d}.mp4"
            )
            if selected.evidence_status == "not_found":
                if not _segment_is_valid(
                    horizontal_segment,
                    expected_duration=brief_chapter.target_duration_seconds,
                    dimensions=(1920, 1080),
                ):
                    _render_missing_segment(
                        brief_chapter, horizontal_segment, horizontal_overlay, (1920, 1080)
                    )
                if not _segment_is_valid(
                    vertical_segment,
                    expected_duration=brief_chapter.target_duration_seconds,
                    dimensions=(1080, 1920),
                ):
                    _render_missing_segment(
                        brief_chapter, vertical_segment, vertical_overlay, (1080, 1920)
                    )
                horizontal_entry = {
                    "feature_id": selected.feature_id,
                    "source_frame_id": None,
                    "applied_zoom": 1.0,
                    "fallback_reason": "catalog_evidence_not_found",
                    "audio_origin": "synthetic_silence",
                }
                vertical_entry = {
                    "feature_id": selected.feature_id,
                    "source_frame_id": None,
                    "applied_strategy": "graphic_missing_evidence_card",
                    "fallback_reason": "catalog_evidence_not_found",
                    "audio_origin": "synthetic_silence",
                }
            else:
                horizontal_frame = frames[selected.horizontal_frame_id or ""]
                horizontal_clip = clips[horizontal_frame.clip_id]
                h_start, h_end, h_shot = _chapter_bounds(
                    horizontal_frame,
                    horizontal_clip,
                    brief_chapter.target_duration_seconds,
                    shot_cache,
                    shots_dir,
                    scdet_threshold,
                )
                vertical_frame = frames[selected.vertical_frame_id or ""]
                vertical_clip = clips[vertical_frame.clip_id]
                for source_clip in (horizontal_clip, vertical_clip):
                    if source_clip.sha256 not in source_audio_cache:
                        source_audio_cache[source_clip.sha256] = has_audio_stream(
                            Path(source_clip.path)
                        )
                horizontal_source_has_audio = source_audio_cache[horizontal_clip.sha256]
                vertical_source_has_audio = source_audio_cache[vertical_clip.sha256]
                v_start, v_end, v_shot = _chapter_bounds(
                    vertical_frame,
                    vertical_clip,
                    brief_chapter.target_duration_seconds,
                    shot_cache,
                    shots_dir,
                    scdet_threshold,
                )
                if brief.render_title_overlays:
                    _render_text_layer(
                        brief_chapter, horizontal_overlay, dimensions=(1920, 1080)
                    )
                    _render_text_layer(
                        brief_chapter, vertical_overlay, dimensions=(1080, 1920)
                    )
                horizontal_filter = _horizontal_original_filter()
                horizontal_geometry = {
                    "requested_zoom": None,
                    "geometry_safe_max_zoom": None,
                    "applied_zoom": 1.0,
                    "fallback_reason": None,
                }
                horizontal_debug: Path | None = None
                horizontal_track_fingerprint: str | None = None
                if selected.horizontal_strategy == "tracked_reframe":
                    target = selected.horizontal_target_description or ""
                    cache_key = (horizontal_frame.frame_id, target, h_start, h_end)
                    track_root = output_dir / "geometry" / selected.feature_id / "horizontal"
                    try:
                        if cache_key not in track_cache:
                            proposal, track = _build_track(
                                client=client,
                                clip=horizontal_clip,
                                frame=horizontal_frame,
                                start_ms=h_start,
                                end_ms=h_end,
                                feature_id=selected.feature_id,
                                event_description=(
                                    brief_chapter.title + "；" + selected.observed_visual_evidence
                                ),
                                target_description=target,
                                checkpoint_path=checkpoint_path,
                                grounding_prompt=grounding_prompt,
                                output_dir=track_root,
                                run_id=f"feature-h-{uuid.uuid4().hex[:8]}",
                                analysis_fps=sam_analysis_fps,
                                scdet_threshold=scdet_threshold,
                            )
                            track_cache[cache_key] = (proposal, track, track_root)
                        _, track, track_root = track_cache[cache_key]
                        horizontal_track_fingerprint = _track_geometry_fingerprint(track)
                        horizontal_filter, horizontal_geometry = _horizontal_filter_from_track(
                            track, selected.horizontal_zoom_intent
                        )
                        horizontal_debug = track_root / "grounding-debug.png"
                    except Exception as error:
                        horizontal_geometry = {
                            "requested_zoom": (
                                1.12 if selected.horizontal_zoom_intent == "subtle" else 1.35
                            ),
                            "geometry_safe_max_zoom": None,
                            "applied_zoom": 1.0,
                            "fallback_reason": (
                                f"tracking_or_grounding_failed:{type(error).__name__}:{error}"
                            ),
                        }
                horizontal_segment_fingerprint = _segment_variant_fingerprint(
                    source_sha256=horizontal_clip.sha256,
                    start_ms=h_start,
                    end_ms=h_end,
                    filter_graph=horizontal_filter,
                    geometry=horizontal_geometry,
                    track_fingerprint=horizontal_track_fingerprint,
                )
                horizontal_segment = (
                    output_dir
                    / "segments"
                    / render_variant
                    / "16x9"
                    / f"{index:02d}-{horizontal_segment_fingerprint[:12]}.mp4"
                )
                if not _segment_is_valid(
                    horizontal_segment,
                    expected_duration=(h_end - h_start) / 1000,
                    dimensions=(1920, 1080),
                ):
                    _render_source_segment(
                        source_path=Path(horizontal_clip.path),
                        start_ms=h_start,
                        end_ms=h_end,
                        overlay_path=(horizontal_overlay if brief.render_title_overlays else None),
                        base_filter=horizontal_filter,
                        output_path=horizontal_segment,
                        source_has_audio=horizontal_source_has_audio,
                    )
                vertical_filter = _vertical_fit_filter()
                vertical_geometry: dict[str, Any] = {
                    "applied_strategy": "fit_with_background",
                    "fallback_reason": None,
                }
                vertical_debug: Path | None = None
                vertical_track_fingerprint: str | None = None
                vertical_primary_override = brief_chapter.vertical_primary_target_description
                if selected.vertical_strategy == "tracked_crop" or vertical_primary_override:
                    target = vertical_primary_override or selected.vertical_target_description or ""
                    cache_key = (vertical_frame.frame_id, target, v_start, v_end)
                    track_root = output_dir / "geometry" / selected.feature_id / "vertical"
                    if vertical_primary_override:
                        target_key = hashlib.sha256(target.encode("utf-8")).hexdigest()[:10]
                        track_root = track_root / f"primary-{target_key}"
                    try:
                        if cache_key not in track_cache:
                            proposal, track = _build_track(
                                client=client,
                                clip=vertical_clip,
                                frame=vertical_frame,
                                start_ms=v_start,
                                end_ms=v_end,
                                feature_id=selected.feature_id,
                                event_description=(
                                    brief_chapter.title + "；" + selected.observed_visual_evidence
                                ),
                                target_description=target,
                                checkpoint_path=checkpoint_path,
                                grounding_prompt=grounding_prompt,
                                output_dir=track_root,
                                run_id=f"feature-v-{uuid.uuid4().hex[:8]}",
                                analysis_fps=sam_analysis_fps,
                                scdet_threshold=scdet_threshold,
                            )
                            track_cache[cache_key] = (proposal, track, track_root)
                        _, track, track_root = track_cache[cache_key]
                        vertical_track_fingerprint = _track_geometry_fingerprint(track)
                        vertical_filter, vertical_geometry = _vertical_filter_from_track(
                            track,
                            allow_subject_clipping=(
                                brief_chapter.vertical_crop_mode == "primary_center"
                            ),
                        )
                        vertical_debug = track_root / "grounding-debug.png"
                    except Exception as error:
                        vertical_geometry = {
                            "applied_strategy": "fit_with_background",
                            "fallback_reason": f"tracking_or_grounding_failed:{type(error).__name__}:{error}",
                        }
                vertical_segment_fingerprint = _segment_variant_fingerprint(
                    source_sha256=vertical_clip.sha256,
                    start_ms=v_start,
                    end_ms=v_end,
                    filter_graph=vertical_filter,
                    geometry=vertical_geometry,
                    track_fingerprint=vertical_track_fingerprint,
                )
                vertical_segment = (
                    output_dir
                    / "segments"
                    / render_variant
                    / "9x16"
                    / f"{index:02d}-{vertical_segment_fingerprint[:12]}.mp4"
                )
                if not _segment_is_valid(
                    vertical_segment,
                    expected_duration=(v_end - v_start) / 1000,
                    dimensions=(1080, 1920),
                ):
                    _render_source_segment(
                        source_path=Path(vertical_clip.path),
                        start_ms=v_start,
                        end_ms=v_end,
                        overlay_path=(vertical_overlay if brief.render_title_overlays else None),
                        base_filter=vertical_filter,
                        output_path=vertical_segment,
                        source_has_audio=vertical_source_has_audio,
                    )
                horizontal_entry = {
                    "feature_id": selected.feature_id,
                    "source_frame_id": horizontal_frame.frame_id,
                    "source_clip_id": horizontal_clip.clip_id,
                    "source_in_ms": h_start,
                    "source_out_ms": h_end,
                    "source_shot_id": h_shot,
                    "segment_render_fingerprint": horizontal_segment_fingerprint,
                    "track_geometry_fingerprint": horizontal_track_fingerprint,
                    "segment_path": str(horizontal_segment.resolve()),
                    "audio_origin": (
                        "source" if horizontal_source_has_audio else "synthetic_silence"
                    ),
                    "grounding_debug": str(horizontal_debug.resolve()) if horizontal_debug else None,
                    **horizontal_geometry,
                }
                vertical_entry = {
                    "feature_id": selected.feature_id,
                    "source_frame_id": vertical_frame.frame_id,
                    "source_clip_id": vertical_clip.clip_id,
                    "source_in_ms": v_start,
                    "source_out_ms": v_end,
                    "source_shot_id": v_shot,
                    "segment_render_fingerprint": vertical_segment_fingerprint,
                    "track_geometry_fingerprint": vertical_track_fingerprint,
                    "segment_path": str(vertical_segment.resolve()),
                    "audio_origin": (
                        "source" if vertical_source_has_audio else "synthetic_silence"
                    ),
                    "target_description": (
                        vertical_primary_override or selected.vertical_target_description
                    ),
                    "primary_target_override": vertical_primary_override is not None,
                    "grounding_debug": str(vertical_debug.resolve()) if vertical_debug else None,
                    **vertical_geometry,
                }
            horizontal_segments.append(horizontal_segment)
            vertical_segments.append(vertical_segment)
            manifest["horizontal"]["chapters"].append(horizontal_entry)
            manifest["vertical"]["chapters"].append(vertical_entry)
        timings["geometry_and_segment_render_seconds"] = round(monotonic() - stage, 3)
    finally:
        client.close()
    output_suffix = "" if brief.render_title_overlays else "-clean"
    horizontal_output = (
        output_dir / "renders" / f"feature-cut-16x9{output_suffix}.mp4"
    )
    vertical_output = (
        output_dir / "renders" / f"feature-cut-9x16{output_suffix}.mp4"
    )
    horizontal_output.parent.mkdir(parents=True, exist_ok=True)
    stage = monotonic()
    _concat_segments(horizontal_segments, horizontal_output)
    _concat_segments(vertical_segments, vertical_output)
    timings["concat_seconds"] = round(monotonic() - stage, 3)
    timings["total_seconds"] = round(monotonic() - started, 3)
    manifest["horizontal"]["output_path"] = str(horizontal_output.resolve())
    manifest["vertical"]["output_path"] = str(vertical_output.resolve())
    manifest["horizontal"]["media"] = _output_media_metadata(horizontal_output)
    manifest["vertical"]["media"] = _output_media_metadata(vertical_output)
    manifest["generated_at"] = utc_now()
    write_json(output_dir / "render-manifest.json", manifest)
    pricing = summarize_usage_and_list_price(output_dir)
    write_json(output_dir / "pricing.json", pricing)
    write_json(
        output_dir / "timing.json",
        {
            **timings,
            "file_api_reused": reused,
            "feature_plan_reused": plan_reused,
            "generated_at": utc_now(),
        },
    )
    _render_review_html(output_dir, brief, plan, manifest)
    result = {
        "horizontal_output": str(horizontal_output.resolve()),
        "vertical_output": str(vertical_output.resolve()),
        "review_path": str((output_dir / "index.html").resolve()),
        "plan_path": str((plan_dir / "feature_edit_plan.json").resolve()),
        "manifest_path": str((output_dir / "render-manifest.json").resolve()),
        "timing": timings,
        "pricing": pricing,
    }
    write_json(output_dir / "result.json", result)
    return result
