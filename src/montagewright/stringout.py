"""A logging/selects stringout: every source watched as one editorial reel.

This is an assistant-editor operation, not semantic retrieval.  Sources stay
in their supplied order, retain their complete proxy duration and carry a
burned-in stable id.  The JSON sidecar is the source/reel timecode map; Gemini
names source ids and local code keeps the clocks exact.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

from PIL import Image, ImageDraw, ImageFont

from montagewright.measure.media import has_audio_stream, sha256_file


STRINGOUT_VERSION = "montagewright-stringout-v1"
DEFAULT_SLATE_SECONDS = 1.0


class StringoutError(RuntimeError):
    """A local reel could not be built or did not preserve its source map."""


class StringoutSource(Protocol):
    @property
    def source_id(self) -> str: ...

    @property
    def duration_seconds(self) -> float: ...

    @property
    def proxy(self) -> Path | None: ...


@dataclass(frozen=True)
class StringoutEntry:
    source_id: str
    source_path: str
    source_sha256: str
    source_in_seconds: float
    source_out_seconds: float
    reel_in_seconds: float
    reel_out_seconds: float


@dataclass(frozen=True)
class StringoutManifest:
    version: str
    video_path: str
    width: int
    height: int
    fps: int
    slate_seconds: float
    entries: tuple[StringoutEntry, ...]

    @property
    def duration_seconds(self) -> float:
        return self.entries[-1].reel_out_seconds if self.entries else 0.0

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "entries": [asdict(entry) for entry in self.entries],
            "duration_seconds": round(self.duration_seconds, 6),
        }


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow before scalable default.
        return ImageFont.load_default()


def _label_images(
    label: str, directory: Path, *, width: int, height: int, duration: float,
    source_start: float = 0.0,
) -> tuple[Path, Path]:
    """Rasterize ids locally; packaged FFmpeg builds may omit drawtext."""

    slate = Image.new("RGB", (width, height), "black")
    slate_draw = ImageDraw.Draw(slate)
    slate_font = _font(52)
    box = slate_draw.textbbox((0, 0), label, font=slate_font)
    slate_draw.text(
        ((width - (box[2] - box[0])) / 2, (height - (box[3] - box[1])) / 2),
        label, fill="white", font=slate_font,
    )
    slate_path = directory / "slate.png"
    slate.save(slate_path)

    clock_dir = directory / "source-clock"
    clock_dir.mkdir(parents=True, exist_ok=True)
    for second in range(max(1, math.ceil(duration)) + 1):
        clock = Image.new("RGBA", (620, 64), (0, 0, 0, 155))
        clock_draw = ImageDraw.Draw(clock)
        mm, ss = divmod(int(source_start) + second, 60)
        clock_draw.text(
            (16, 12), f"SOURCE {label}   TC {mm}:{ss:02d}",
            fill="white", font=_font(28),
        )
        clock.save(clock_dir / f"{second:06d}.png")
    return slate_path, clock_dir / "%06d.png"


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise StringoutError(
            f"stringout command failed ({completed.returncode}): "
            f"{completed.stderr.strip()[-1200:]}"
        )


def _render_segment(
    source: StringoutSource,
    destination: Path,
    *,
    width: int,
    height: int,
    fps: int,
    slate_seconds: float,
    source_start: float = 0.0,
    source_end: float | None = None,
) -> None:
    if source.proxy is None or not source.proxy.exists():
        raise StringoutError(f"{source.source_id} has no readable planning proxy")
    duration = (
        float(source.duration_seconds)
        if source_end is None else float(source_end) - float(source_start)
    )
    if duration <= 0:
        raise StringoutError(f"{source.source_id} has no positive duration")

    slate_frames = max(1, round(slate_seconds * fps))
    slate_duration = slate_frames / fps
    clip_frames = max(1, math.ceil(duration * fps - 1e-9))
    clip_duration = clip_frames / fps
    slate_path, clock_pattern = _label_images(
        source.source_id, destination.parent, width=width, height=height,
        duration=duration, source_start=source_start,
    )
    common = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={fps},setsar=1,format=yuv420p"
    )
    filters = [
        f"[1:v]fps={fps},format=yuv420p,trim=duration={slate_duration},"
        "setpts=PTS-STARTPTS[slatev]",
        f"[0:v]{common},tpad=stop_mode=clone:stop_duration={1 / fps:.9f},"
        f"trim=duration={clip_duration:.9f},setpts=PTS-STARTPTS[basev]",
        "[basev][2:v]overlay=x=18:y=18:eof_action=repeat:shortest=1:format=auto,"
        "format=yuv420p[clipv]",
        "[slatev][clipv]concat=n=2:v=1:a=0[v]",
        f"anullsrc=r=48000:cl=stereo,atrim=duration={slate_duration},"
        "asetpts=PTS-STARTPTS[slatea]",
    ]
    if has_audio_stream(source.proxy):
        filters.append(
            f"[0:a]aresample=48000,aformat=sample_fmts=fltp:"
            f"channel_layouts=stereo,apad,atrim=duration={clip_duration:.9f},"
            "asetpts=PTS-STARTPTS[clipa]"
        )
    else:
        filters.append(
            f"anullsrc=r=48000:cl=stereo,atrim=duration={clip_duration:.9f},"
            "asetpts=PTS-STARTPTS[clipa]"
        )
    filters.append("[slatea][clipa]concat=n=2:v=0:a=1[a]")

    destination.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{source_start:.6f}", "-t", f"{duration:.6f}",
        "-i", str(source.proxy),
        "-loop", "1", "-framerate", str(fps), "-t", f"{slate_duration:.9f}",
        "-i", str(slate_path),
        "-framerate", "1", "-start_number", "0", "-i", str(clock_pattern),
        "-filter_complex", ";".join(filters),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
        "-c:a", "aac", "-b:a", "128k",
        "-r", str(fps), "-fps_mode", "cfr", "-movflags", "+faststart",
        str(destination),
    ])


def build_stringout(
    sources: Iterable[StringoutSource],
    destination: Path,
    *,
    manifest_path: Path | None = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    slate_seconds: float = DEFAULT_SLATE_SECONDS,
) -> StringoutManifest:
    """Render sources in order and write the exact source/reel clock sidecar."""

    ordered = tuple(sources)
    if not ordered:
        raise StringoutError("cannot build an empty stringout")
    if width < 320 or height < 180 or fps <= 0 or slate_seconds < 0:
        raise ValueError("invalid stringout canvas, fps or slate duration")

    destination = destination.expanduser().resolve()
    manifest_path = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else destination.with_suffix(".json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    entries: list[StringoutEntry] = []
    cursor_frames = 0
    slate_frames = max(1, round(slate_seconds * fps))
    effective_slate_seconds = slate_frames / fps
    with tempfile.TemporaryDirectory(
        prefix="montagewright-stringout-", dir=destination.parent,
    ) as raw_work:
        work = Path(raw_work)
        segments: list[Path] = []
        slice_index = 0
        for source in ordered:
            ranges = tuple(getattr(source, "planning_ranges", ()) or ())
            if not ranges:
                ranges = ((0.0, float(source.duration_seconds)),)
            for source_start, source_end in ranges:
                segment = work / f"{slice_index:05d}.mp4"
                slice_index += 1
                _render_segment(
                    source, segment, width=width, height=height, fps=fps,
                    slate_seconds=effective_slate_seconds,
                    source_start=float(source_start),
                    source_end=float(source_end),
                )
                duration = float(source_end) - float(source_start)
                clip_frames = max(1, math.ceil(duration * fps - 1e-9))
                reel_in_frame = cursor_frames + slate_frames
                reel_out_frame = reel_in_frame + clip_frames
                reel_in = reel_in_frame / fps
                reel_out = reel_out_frame / fps
                assert source.proxy is not None
                entries.append(StringoutEntry(
                    source_id=source.source_id,
                    source_path=str(source.proxy.resolve()),
                    source_sha256=sha256_file(source.proxy),
                    source_in_seconds=round(float(source_start), 6),
                    source_out_seconds=round(float(source_end), 6),
                    reel_in_seconds=round(reel_in, 6),
                    reel_out_seconds=round(reel_out, 6),
                ))
                cursor_frames = reel_out_frame
                segments.append(segment)

        concat_list = work / "concat.txt"
        concat_list.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in segments),
            encoding="utf-8",
        )
        joined = work / "joined.mp4"
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", "-movflags", "+faststart", str(joined),
        ])
        shutil.move(str(joined), destination)

    manifest = StringoutManifest(
        version=STRINGOUT_VERSION,
        video_path=str(destination), width=width, height=height, fps=fps,
        slate_seconds=effective_slate_seconds, entries=tuple(entries),
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_stringout_manifest(path: Path) -> StringoutManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != STRINGOUT_VERSION:
        raise StringoutError(f"unsupported stringout manifest {payload.get('version')!r}")
    entries = tuple(StringoutEntry(**one) for one in payload.get("entries") or [])
    if not entries:
        raise StringoutError("stringout manifest has no entries")
    return StringoutManifest(
        version=payload["version"], video_path=str(payload["video_path"]),
        width=int(payload["width"]), height=int(payload["height"]),
        fps=int(payload["fps"]), slate_seconds=float(payload["slate_seconds"]),
        entries=entries,
    )


def require_stringout_matches(
    manifest: StringoutManifest, sources: Iterable[StringoutSource],
) -> None:
    """Fail before upload when a reel is missing or stale for this material."""

    expected = tuple(sources)
    expected_slices = tuple(
        (source, float(start), float(end))
        for source in expected
        for start, end in (
            tuple(getattr(source, "planning_ranges", ()) or ())
            or ((0.0, float(source.duration_seconds)),)
        )
    )
    if tuple(one.source_id for one in manifest.entries) != tuple(
        source.source_id for source, _start, _end in expected_slices
    ):
        raise StringoutError("stringout source order does not match planning material")
    for entry, (source, start, end) in zip(manifest.entries, expected_slices):
        if source.proxy is None or not source.proxy.exists():
            raise StringoutError(f"{source.source_id} has no readable planning proxy")
        if entry.source_sha256 != sha256_file(source.proxy):
            raise StringoutError(f"stringout is stale for {source.source_id}")
        if (
            abs(entry.source_in_seconds - start) > 0.05
            or abs(entry.source_out_seconds - end) > 0.05
        ):
            raise StringoutError(f"stringout duration is stale for {source.source_id}")
