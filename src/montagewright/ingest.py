"""Freeze the rushes before an editor or a paid model sees them.

An assistant editor starts with an inventory: what arrived, which files are
the same, which camera/audio streams they contain, and whether a proxy still
belongs to its master.  This module is that ingest sheet.  It deliberately
precedes clip cards and editorial planning so a technically plausible edit
can never be made from stale or silently omitted material.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from montagewright.measure.media import sha256_file


VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".m4v", ".avi", ".mkv"})
AUDIO_SUFFIXES = frozenset({".wav", ".aif", ".aiff", ".m4a", ".mp3", ".flac", ".aac"})
MEDIA_SUFFIXES = VIDEO_SUFFIXES | AUDIO_SUFFIXES


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AudioStream(_Frozen):
    index: int
    codec: str | None = None
    channels: int = 0
    channel_layout: str | None = None
    sample_rate: int | None = None
    language: str | None = None


class VideoFacts(_Frozen):
    codec: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    average_frame_rate: str
    real_frame_rate: str
    variable_frame_rate: bool = False
    field_order: str = "progressive"
    pixel_format: str | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_space: str | None = None
    rotation_degrees: int = 0


class IngestAsset(_Frozen):
    source_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    asset_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    relative_path: str
    absolute_path: str
    kind: Literal["video", "audio"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    mtime_ns: int
    duration_seconds: float = Field(gt=0)
    video: VideoFacts | None = None
    audio_streams: tuple[AudioStream, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def picture_has_video_facts(self) -> "IngestAsset":
        if self.kind == "video" and self.video is None:
            raise ValueError("video asset requires video stream facts")
        if self.kind == "audio" and self.video is not None:
            raise ValueError("audio-only asset cannot carry video facts")
        return self


class RejectedAsset(_Frozen):
    relative_path: str
    reason: str


class IngestManifest(_Frozen):
    version: Literal["montagewright-ingest-v1"] = "montagewright-ingest-v1"
    root: str
    assets: tuple[IngestAsset, ...]
    rejected: tuple[RejectedAsset, ...] = ()
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimated_work_bytes: int = Field(ge=0)
    free_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def unique_sources(self) -> "IngestManifest":
        ids = [one.source_id for one in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("ingest source IDs are not unique")
        paths = [one.relative_path for one in self.assets]
        if len(paths) != len(set(paths)):
            raise ValueError("ingest paths are not unique")
        return self

    @property
    def videos(self) -> tuple[IngestAsset, ...]:
        return tuple(one for one in self.assets if one.kind == "video")

    @property
    def external_audio(self) -> tuple[IngestAsset, ...]:
        return tuple(one for one in self.assets if one.kind == "audio")


class IngestError(RuntimeError):
    pass


def technical_contract_faults(
    manifest: IngestManifest,
    *,
    delivery_color: str = "normalize_to_sdr",
    selected_audio_streams: dict[str, int] | None = None,
) -> tuple[str, ...]:
    """Return media facts the current frame-accurate path cannot honour.

    These are refusals, not quality warnings.  A proxy can make VFR, HDR or
    interlaced material pleasant for a model to watch while the final source
    still seeks on another clock or reaches the encoder with the wrong colour.
    Until a measured mezzanine/time-map or colour transform exists, paying for
    an edit would only defer the failure.
    """

    chosen = selected_audio_streams or {}
    faults: list[str] = []
    for asset in manifest.assets:
        video = asset.video
        if video is not None and video.variable_frame_rate:
            faults.append(
                f"{asset.relative_path}: variable-frame-rate source requires "
                "a CFR mezzanine with a source time-map"
            )
        if video is not None and video.field_order not in {"progressive", "unknown"}:
            faults.append(
                f"{asset.relative_path}: {video.field_order} video requires "
                "an explicit deinterlace ingest transform"
            )
        if video is not None and video.color_transfer in {"smpte2084", "arib-std-b67"}:
            action = (
                "verified HDR preservation"
                if delivery_color == "preserve_hdr"
                else "a verified HDR-to-Rec.709 transform"
            )
            faults.append(f"{asset.relative_path}: HDR source requires {action}")
        if len(asset.audio_streams) > 1:
            selected = chosen.get(asset.source_id)
            indices = {stream.index for stream in asset.audio_streams}
            if selected not in indices:
                faults.append(
                    f"{asset.relative_path}: has {len(asset.audio_streams)} audio "
                    "streams; select the dialogue stream before planning"
                )
    return tuple(faults)


def decode_preflight(manifest: IngestManifest) -> None:
    """Decode the beginning and end of every asset before paid analysis.

    ffprobe can read a truncated file whose final GOP is gone.  Two tiny
    decode probes catch the common incomplete-copy/container failures without
    transcoding the rushes or pretending to be a full checksum verification.
    """

    failures: list[str] = []
    for asset in manifest.assets:
        path = Path(asset.absolute_path)
        starts = (0.0,) if asset.duration_seconds < 2.0 else (
            0.0, max(0.0, asset.duration_seconds - 0.75),
        )
        for at in starts:
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss",
                f"{at:.6f}", "-i", str(path),
            ]
            if asset.kind == "video":
                command += ["-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"]
            else:
                command += ["-map", "0:a:0", "-t", "0.1", "-f", "null", "-"]
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False,
            )
            if completed.returncode:
                failures.append(
                    f"{asset.relative_path} at {at:.3f}s: "
                    f"{completed.stderr.strip() or 'decode failed'}"
                )
                break
        if any(one.startswith(f"{asset.relative_path} ") for one in failures):
            continue
        # Head/tail catches incomplete copies quickly; a complete decode is
        # the only honest way to catch a corrupt middle GOP before spending.
        stream = "0:v:0" if asset.kind == "video" else "0:a:0"
        completed = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-err_detect", "explode", "-i", str(path), "-map", stream,
                "-f", "null", "-",
            ],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode:
            failures.append(
                f"{asset.relative_path} full decode: "
                f"{completed.stderr.strip() or 'decode failed'}"
            )
    if failures:
        raise IngestError("media decode preflight failed:\n  " + "\n  ".join(failures))


def revalidate_manifest(manifest: IngestManifest) -> None:
    """Refuse a source-path swap after ingest froze the inventory."""

    failures: list[str] = []
    for asset in manifest.assets:
        path = Path(asset.absolute_path)
        try:
            stat = path.stat()
            if stat.st_size != asset.size_bytes or stat.st_mtime_ns != asset.mtime_ns:
                failures.append(f"{asset.relative_path}: size or modification time changed")
                continue
            if sha256_file(path) != asset.sha256:
                failures.append(f"{asset.relative_path}: content hash changed")
        except OSError as error:
            failures.append(f"{asset.relative_path}: {error}")
    if failures:
        raise IngestError(
            "source material changed after ingest; start a new revision:\n  "
            + "\n  ".join(failures)
        )


def _fraction(value: object) -> Fraction | None:
    try:
        found = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    return found if found > 0 else None


def _rotation(stream: dict[str, Any]) -> int:
    tags = stream.get("tags") or {}
    try:
        if "rotate" in tags:
            return int(float(tags["rotate"])) % 360
    except (TypeError, ValueError):
        pass
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                return int(float(side["rotation"])) % 360
            except (TypeError, ValueError):
                pass
    return 0


def _source_ids(rows: list[tuple[Path, str]]) -> dict[Path, str]:
    groups: dict[str, list[tuple[Path, str]]] = {}
    for path, digest in rows:
        groups.setdefault(path.stem.casefold(), []).append((path, digest))
    made: dict[Path, str] = {}
    for members in groups.values():
        for path, digest in members:
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.stem).strip("-.") or "source"
            if len(members) == 1:
                made[path] = stem
                continue
            path_key = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:6]
            made[path] = f"{stem}__{digest[:8]}__{path_key}"
    return made


def _probe(path: Path) -> tuple[dict[str, Any], str]:
    before = path.stat()
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_format", "-show_streams",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise IngestError(completed.stderr.strip() or "ffprobe could not read file")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise IngestError(f"ffprobe returned invalid JSON: {error}") from error
    digest = sha256_file(path)
    after = path.stat()
    before_fingerprint = (
        before.st_size, before.st_mtime_ns, getattr(before, "st_ino", None)
    )
    after_fingerprint = (
        after.st_size, after.st_mtime_ns, getattr(after, "st_ino", None)
    )
    if before_fingerprint != after_fingerprint:
        raise IngestError(
            f"media changed while it was being inventoried: {path.name}; "
            "finish copying it before planning"
        )
    return payload, digest


def _inspect(path: Path, relative: Path, source_id: str, payload: dict[str, Any], digest: str) -> IngestAsset:
    streams = list(payload.get("streams") or [])
    pictures = [one for one in streams if one.get("codec_type") == "video"]
    sounds = [one for one in streams if one.get("codec_type") == "audio"]
    format_row = payload.get("format") or {}
    raw_duration = format_row.get("duration") or next(
        (one.get("duration") for one in streams if one.get("duration")), None
    )
    try:
        duration = float(str(raw_duration))
    except (TypeError, ValueError) as error:
        raise IngestError("media has no usable duration") from error
    if duration <= 0:
        raise IngestError("media duration is not positive")
    audio = tuple(
        AudioStream(
            index=int(one.get("index", 0)),
            codec=one.get("codec_name"),
            channels=int(one.get("channels") or 0),
            channel_layout=one.get("channel_layout"),
            sample_rate=(int(one["sample_rate"]) if one.get("sample_rate") else None),
            language=(one.get("tags") or {}).get("language"),
        )
        for one in sounds
    )
    video = None
    warnings: list[str] = []
    kind: Literal["video", "audio"] = "audio"
    if pictures:
        kind = "video"
        one = pictures[0]
        average = str(one.get("avg_frame_rate") or "0/0")
        real = str(one.get("r_frame_rate") or "0/0")
        average_rate, real_rate = _fraction(average), _fraction(real)
        variable = bool(
            average_rate and real_rate
            and abs(float(average_rate - real_rate)) > 0.01
        )
        field_order = str(one.get("field_order") or "progressive")
        transfer = one.get("color_transfer")
        if variable:
            warnings.append("variable_frame_rate")
        if field_order not in {"progressive", "unknown"}:
            warnings.append(f"interlaced:{field_order}")
        if transfer in {"smpte2084", "arib-std-b67"}:
            warnings.append(f"hdr:{transfer}")
        video = VideoFacts(
            codec=one.get("codec_name"), width=int(one.get("width") or 0),
            height=int(one.get("height") or 0),
            average_frame_rate=average, real_frame_rate=real,
            variable_frame_rate=variable, field_order=field_order,
            pixel_format=one.get("pix_fmt"), color_primaries=one.get("color_primaries"),
            color_transfer=transfer, color_space=one.get("color_space"),
            rotation_degrees=_rotation(one),
        )
    if not pictures and not sounds:
        raise IngestError("media has neither a video nor an audio stream")
    stat = path.stat()
    return IngestAsset(
        source_id=source_id, asset_id=f"sha256:{digest}",
        relative_path=relative.as_posix(), absolute_path=str(path), kind=kind,
        sha256=digest, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
        duration_seconds=duration, video=video, audio_streams=audio,
        warnings=tuple(warnings),
    )


def build_manifest(root: Path, *, work_root: Path, previous: Path | None = None) -> IngestManifest:
    """Discover recursively, probe once, and freeze the exact material set."""

    source_root = root.expanduser().resolve(strict=True)
    work = work_root.expanduser().resolve()
    candidates: list[Path] = []
    rejected: list[RejectedAsset] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or work == path or work in path.parents:
            continue
        suffix = path.suffix.casefold()
        relative = path.relative_to(source_root)
        if suffix in MEDIA_SUFFIXES:
            candidates.append(path)
        elif suffix:
            rejected.append(RejectedAsset(
                relative_path=relative.as_posix(), reason="unsupported_extension",
            ))
    probed: dict[Path, tuple[dict[str, Any], str]] = {}
    failures: list[str] = []
    for path in candidates:
        try:
            probed[path] = _probe(path)
        except (OSError, IngestError) as error:
            failures.append(f"{path.relative_to(source_root)}: {error}")
    if failures:
        raise IngestError("unreadable media:\n  " + "\n  ".join(failures))
    ids = _source_ids([(path, digest) for path, (_, digest) in probed.items()])
    assets = tuple(
        _inspect(path, path.relative_to(source_root), ids[path], payload, digest)
        for path, (payload, digest) in probed.items()
    )
    if not any(one.kind == "video" for one in assets):
        raise IngestError(f"no video files in {source_root}")
    identity = [
        {"source_id": one.source_id, "asset_id": one.asset_id,
         "relative_path": one.relative_path, "size_bytes": one.size_bytes}
        for one in assets
    ]
    inventory_sha = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    # Proxies, decoded frames, tracking and a staging master. This is a
    # conservative local readiness estimate, not a promise of final bitrate.
    video_bytes = sum(one.size_bytes for one in assets if one.kind == "video")
    estimated = max(1024**3, int(video_bytes * 0.35) + 512 * 1024**2)
    free = shutil.disk_usage(work.parent if work.parent.exists() else source_root).free
    manifest = IngestManifest(
        root=str(source_root), assets=assets, rejected=tuple(rejected),
        inventory_sha256=inventory_sha, estimated_work_bytes=estimated,
        free_bytes=free,
    )
    if free < estimated:
        raise IngestError(
            f"not enough free disk for this edit: need about {estimated} bytes, "
            f"found {free}"
        )
    if previous is not None and previous.exists():
        old = IngestManifest.model_validate_json(previous.read_text(encoding="utf-8"))
        if old.inventory_sha256 != manifest.inventory_sha256:
            raise IngestError(
                "rushes changed since this output was ingested; start a new revision "
                "instead of combining old proxies with new masters"
            )
    return manifest


def write_manifest(path: Path, manifest: IngestManifest) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
