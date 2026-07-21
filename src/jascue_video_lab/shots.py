from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import Field

from .models import Rational, StrictModel
from .storage import utc_now, write_json


class ShotBoundary(StrictModel):
    boundary_id: str
    frame_pts: int
    frame_time_ms: int = Field(ge=0)
    score: float = Field(ge=0.0, le=100.0)


class ShotSegment(StrictModel):
    shot_id: str
    start_time_ms: int = Field(ge=0)
    end_time_ms: int = Field(gt=0)
    start_frame_pts: int | None
    boundary_source: str
    boundary_score: float | None = Field(default=None, ge=0.0, le=100.0)


class ShotManifest(StrictModel):
    video_path: str
    duration_ms: int = Field(gt=0)
    detector: str
    threshold: float = Field(ge=0.0, le=100.0)
    generated_at: str
    timeline_basis: Literal[
        "local_ms_from_decoded_pts",
        "legacy_detector_time",
    ] = "legacy_detector_time"
    source_start_pts: int | None = None
    source_time_base: Rational | None = None
    boundaries: list[ShotBoundary]
    shots: list[ShotSegment]


def _lavfi_movie_path(path: Path) -> str:
    value = str(path.resolve())
    return value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _probe_timing(video_path: Path) -> tuple[int, int, Fraction]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=start_pts,time_base",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"video has no video stream: {video_path}")
    stream = streams[0]
    duration_ms = round(float(payload["format"]["duration"]) * 1000)
    source_start_pts = int(stream.get("start_pts") or 0)
    time_base = Fraction(stream["time_base"])
    return duration_ms, source_start_pts, time_base


def detect_shots_ffmpeg(
    video_path: Path,
    *,
    threshold: float = 4.0,
    output_path: Path | None = None,
) -> ShotManifest:
    """Detect decoded-frame shot boundaries with FFmpeg scdet and preserve PTS."""
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not 0 <= threshold <= 100:
        raise ValueError("FFmpeg scdet threshold must be within 0..100")
    filter_graph = f"movie='{_lavfi_movie_path(video_path)}',scdet=t={threshold}:s=1"
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-f",
            "lavfi",
            filter_graph,
            "-show_entries",
            "frame=pts,pts_time:frame_tags=lavfi.scd.time,lavfi.scd.score",
            "-of",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    duration_ms, source_start_pts, time_base = _probe_timing(video_path)
    by_time: dict[int, tuple[int, float]] = {}
    for frame in payload.get("frames", []):
        tags = frame.get("tags") or {}
        frame_pts = int(frame["pts"])
        local_time_ms = round(
            Fraction(frame_pts - source_start_pts) * time_base * 1000
        )
        if not 0 < local_time_ms < duration_ms:
            continue
        score = float(tags["lavfi.scd.score"])
        existing = by_time.get(local_time_ms)
        if existing is None or score > existing[1]:
            by_time[local_time_ms] = (frame_pts, score)
    boundaries = [
        ShotBoundary(
            boundary_id=f"boundary-{index:04d}",
            frame_pts=frame_pts,
            frame_time_ms=local_time_ms,
            score=score,
        )
        for index, (local_time_ms, (frame_pts, score)) in enumerate(
            sorted(by_time.items()),
            start=1,
        )
    ]
    starts = [(0, source_start_pts, None)] + [
        (boundary.frame_time_ms, boundary.frame_pts, boundary.score)
        for boundary in boundaries
        if 0 < boundary.frame_time_ms < duration_ms
    ]
    shots = [
        ShotSegment(
            shot_id=f"shot-{index + 1:04d}",
            start_time_ms=start_time,
            end_time_ms=(starts[index + 1][0] if index + 1 < len(starts) else duration_ms),
            start_frame_pts=start_pts,
            boundary_source="video_start" if index == 0 else "ffmpeg_scdet",
            boundary_score=score,
        )
        for index, (start_time, start_pts, score) in enumerate(starts)
    ]
    manifest = ShotManifest(
        video_path=str(video_path.resolve()),
        duration_ms=duration_ms,
        detector="ffprobe lavfi movie + scdet",
        threshold=threshold,
        generated_at=utc_now(),
        timeline_basis="local_ms_from_decoded_pts",
        source_start_pts=source_start_pts,
        source_time_base=Rational(
            numerator=time_base.numerator,
            denominator=time_base.denominator,
        ),
        boundaries=boundaries,
        shots=shots,
    )
    if output_path is not None:
        write_json(output_path, manifest)
    return manifest
