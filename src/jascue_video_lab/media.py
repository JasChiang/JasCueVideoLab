from __future__ import annotations

import hashlib
import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path

from PIL import Image

from .models import ExtractedFrame, MediaInfo, Rational, VideoStreamInfo


class MediaCommandError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise MediaCommandError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr.strip()}"
        )
    return completed


def _rational(value: str | None) -> Rational | None:
    if not value or value == "0/0":
        return None
    fraction = Fraction(value)
    return Rational(numerator=fraction.numerator, denominator=fraction.denominator)


def _rotation(stream: dict[str, object]) -> int:
    tags = stream.get("tags") or {}
    if isinstance(tags, dict) and "rotate" in tags:
        try:
            return int(float(str(tags["rotate"]))) % 360
        except ValueError:
            pass
    for side_data in stream.get("side_data_list") or []:
        if isinstance(side_data, dict) and "rotation" in side_data:
            return int(float(str(side_data["rotation"]))) % 360
    return 0


def probe_video(path: Path) -> MediaInfo:
    source = path.expanduser().resolve(strict=True)
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(source),
        ]
    )
    payload = json.loads(result.stdout)
    streams = [stream for stream in payload["streams"] if stream.get("codec_type") == "video"]
    if not streams:
        raise MediaCommandError(f"no video stream found: {source}")
    stream = streams[0]
    rotation = _rotation(stream)
    coded_width = int(stream["width"])
    coded_height = int(stream["height"])
    display_width, display_height = (
        (coded_height, coded_width) if rotation in {90, 270} else (coded_width, coded_height)
    )
    time_base = _rational(stream.get("time_base"))
    if time_base is None:
        raise MediaCommandError("video stream has no usable time_base")
    format_info = payload.get("format", {})
    duration_s = format_info.get("duration") or stream.get("duration")
    if duration_s is None:
        raise MediaCommandError("video has no duration")
    file_hash = sha256_file(source)
    return MediaInfo(
        path=str(source),
        sha256=file_hash,
        asset_id=f"sha256:{file_hash}",
        format_name=format_info.get("format_name"),
        duration_ms=round(float(duration_s) * 1000),
        size_bytes=int(format_info.get("size") or source.stat().st_size),
        format_metadata={str(k): str(v) for k, v in (format_info.get("tags") or {}).items()},
        video=VideoStreamInfo(
            index=int(stream["index"]),
            codec_name=stream.get("codec_name"),
            coded_width=coded_width,
            coded_height=coded_height,
            display_width=display_width,
            display_height=display_height,
            rotation_degrees=rotation,
            average_frame_rate=_rational(stream.get("avg_frame_rate")),
            real_frame_rate=_rational(stream.get("r_frame_rate")),
            time_base=time_base,
            start_pts=int(stream["start_pts"]) if stream.get("start_pts") is not None else None,
            duration_ts=int(stream["duration_ts"]) if stream.get("duration_ts") is not None else None,
            metadata={str(k): str(v) for k, v in (stream.get("tags") or {}).items()},
        ),
    )


_SHOWINFO_RE = re.compile(r"pts:\s*(?P<pts>-?\d+)\s+pts_time:(?P<time>-?[0-9.]+)")


def extract_frame(source: Path, requested_time_ms: int, output: Path) -> ExtractedFrame:
    if requested_time_ms < 0:
        raise ValueError("requested_time_ms must be non-negative")
    output.parent.mkdir(parents=True, exist_ok=True)
    seconds = requested_time_ms / 1000
    # select runs after FFmpeg's default orientation correction. showinfo records
    # the exact chosen source PTS rather than pretending the semantic time is exact.
    filter_graph = f"select=gte(t\\,{seconds:.6f}),showinfo"
    completed = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(source.expanduser().resolve(strict=True)),
            "-map",
            "0:v:0",
            "-vf",
            filter_graph,
            "-fps_mode",
            "vfr",
            "-frames:v",
            "1",
            "-y",
            str(output),
        ]
    )
    match = _SHOWINFO_RE.search(completed.stderr)
    if not match:
        raise MediaCommandError("could not parse selected frame PTS from ffmpeg showinfo")
    with Image.open(output) as image:
        width, height = image.size
    frame_hash = sha256_file(output)
    return ExtractedFrame(
        path=str(output.resolve()),
        requested_time_ms=requested_time_ms,
        frame_time_ms=round(float(match.group("time")) * 1000),
        frame_pts=int(match.group("pts")),
        frame_hash=frame_hash,
        width=width,
        height=height,
    )


def create_analysis_proxy(
    source: Path,
    output: Path,
    *,
    max_side: int = 1920,
    fps: int = 30,
    max_duration_delta_ms: int = 100,
) -> tuple[MediaInfo, dict[str, object]]:
    """Create a small orientation-corrected semantic-analysis proxy; geometry stays on source."""
    if max_side < 320 or fps < 1:
        raise ValueError("analysis proxy max_side and fps must be positive practical values")
    source_media = probe_video(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source.expanduser().resolve(strict=True)),
            "-map",
            "0:v:0",
            "-vf",
            f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-r",
            str(fps),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ]
    )
    proxy_media = probe_video(output)
    duration_delta_ms = abs(proxy_media.duration_ms - source_media.duration_ms)
    if duration_delta_ms > max_duration_delta_ms:
        raise MediaCommandError(
            f"analysis proxy duration differs by {duration_delta_ms} ms; "
            f"maximum is {max_duration_delta_ms} ms"
        )
    record = {
        "purpose": "Gemini semantic analysis only; original source remains geometry authority",
        "source_asset_id": source_media.asset_id,
        "proxy_asset_id": proxy_media.asset_id,
        "duration_delta_ms": duration_delta_ms,
        "max_side": max_side,
        "fps": fps,
        "original_bytes": source_media.size_bytes,
        "proxy_bytes": proxy_media.size_bytes,
        "byte_reduction_ratio": round(1 - proxy_media.size_bytes / source_media.size_bytes, 8),
    }
    return proxy_media, record
