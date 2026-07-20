from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pydantic import Field

from .models import StrictModel
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
    boundaries: list[ShotBoundary]
    shots: list[ShotSegment]


def _lavfi_movie_path(path: Path) -> str:
    value = str(path.resolve())
    return value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _probe_duration_ms(video_path: Path) -> int:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(json.loads(completed.stdout)["format"]["duration"])
    return round(duration * 1000)


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
    boundaries: list[ShotBoundary] = []
    for index, frame in enumerate(payload.get("frames", []), start=1):
        tags = frame.get("tags") or {}
        seconds = float(tags.get("lavfi.scd.time", frame["pts_time"]))
        boundaries.append(
            ShotBoundary(
                boundary_id=f"boundary-{index:04d}",
                frame_pts=int(frame["pts"]),
                frame_time_ms=round(seconds * 1000),
                score=float(tags["lavfi.scd.score"]),
            )
        )
    duration_ms = _probe_duration_ms(video_path)
    starts = [(0, None, None)] + [
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
        boundaries=boundaries,
        shots=shots,
    )
    if output_path is not None:
        write_json(output_path, manifest)
    return manifest
