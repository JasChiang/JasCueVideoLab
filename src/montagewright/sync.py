"""Resolve multicam and double-system sound onto one source clock.

The map uses the convention ``group_time = source_time + offset_seconds``.
That makes an angle and a recorder interchangeable at a cut: convert the
audible source time to group time, then subtract the picture angle's offset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from montagewright.ingest import IngestManifest


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyncMember(_Frozen):
    source: str = Field(min_length=1)
    offset_seconds: float = 0.0
    role: Literal["picture_angle", "master_audio", "scratch_audio"]
    audio_stream_index: int | None = Field(default=None, ge=0)
    channel: int | None = Field(default=None, ge=0)


class SyncGroup(_Frozen):
    group_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    members: tuple[SyncMember, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def one_master(self) -> "SyncGroup":
        masters = [one for one in self.members if one.role == "master_audio"]
        if len(masters) > 1:
            raise ValueError(f"sync group {self.group_id} has more than one master audio")
        if masters and not any(one.role == "picture_angle" for one in self.members):
            raise ValueError(
                f"sync group {self.group_id} has master audio but no picture angle"
            )
        refs = [one.source for one in self.members]
        if len(refs) != len(set(refs)):
            raise ValueError(f"sync group {self.group_id} repeats a source")
        return self


class SyncMap(_Frozen):
    version: Literal["montagewright-sync-map-v1"] = "montagewright-sync-map-v1"
    authority: Literal["timecode", "audio_fingerprint", "slate", "manual"]
    groups: tuple[SyncGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def sources_belong_once(self) -> "SyncMap":
        refs = [member.source for group in self.groups for member in group.members]
        if len(refs) != len(set(refs)):
            raise ValueError("one source cannot belong to more than one sync group")
        return self


class ResolvedSyncMember(_Frozen):
    group_id: str
    source_id: str
    path: str
    offset_seconds: float
    role: Literal["picture_angle", "master_audio", "scratch_audio"]
    audio_stream_index: int | None = None
    channel: int | None = None


class ResolvedSyncMap(_Frozen):
    version: Literal["montagewright-resolved-sync-v1"] = "montagewright-resolved-sync-v1"
    authority: Literal["timecode", "audio_fingerprint", "slate", "manual"]
    members: tuple[ResolvedSyncMember, ...]

    def by_source(self) -> dict[str, ResolvedSyncMember]:
        return {one.source_id: one for one in self.members}


def load_sync_map(path: Path) -> SyncMap:
    source = path.expanduser().resolve(strict=True)
    try:
        return SyncMap.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid sync map {source}: {error}") from error


def resolve_sync(sync: SyncMap, ingest: IngestManifest) -> ResolvedSyncMap:
    lookup = {
        key: asset
        for asset in ingest.assets
        for key in (asset.source_id, asset.relative_path, asset.asset_id)
    }
    resolved: list[ResolvedSyncMember] = []
    for group in sync.groups:
        for member in group.members:
            asset = lookup.get(member.source)
            if asset is None:
                raise ValueError(
                    f"sync group {group.group_id} names unknown source {member.source!r}"
                )
            if member.role == "picture_angle" and asset.kind != "video":
                raise ValueError(f"{member.source} is not a picture source")
            if member.role == "master_audio" and not asset.audio_streams:
                raise ValueError(f"{member.source} has no audio stream")
            if (
                member.role == "master_audio"
                and len(asset.audio_streams) > 1
                and member.audio_stream_index is None
            ):
                raise ValueError(
                    f"{member.source} has {len(asset.audio_streams)} audio streams; "
                    "master_audio must name audio_stream_index explicitly"
                )
            selected_stream = None
            if member.audio_stream_index is not None:
                selected_stream = next(
                    (
                        stream for stream in asset.audio_streams
                        if stream.index == member.audio_stream_index
                    ),
                    None,
                )
                if selected_stream is None:
                    available = ", ".join(
                        str(stream.index) for stream in asset.audio_streams
                    ) or "none"
                    raise ValueError(
                        f"{member.source} has no audio stream "
                        f"{member.audio_stream_index}; available: {available}"
                    )
            elif member.channel is not None:
                if len(asset.audio_streams) != 1:
                    raise ValueError(
                        f"{member.source} channel selection requires an explicit "
                        "audio_stream_index when the asset has multiple streams"
                    )
                selected_stream = asset.audio_streams[0]
            if member.channel is not None:
                assert selected_stream is not None
                if member.channel >= selected_stream.channels:
                    raise ValueError(
                        f"{member.source} stream {selected_stream.index} has "
                        f"{selected_stream.channels} channels, not channel "
                        f"{member.channel}"
                    )
            resolved.append(ResolvedSyncMember(
                group_id=group.group_id, source_id=asset.source_id,
                path=asset.absolute_path, offset_seconds=member.offset_seconds,
                role=member.role, audio_stream_index=member.audio_stream_index,
                channel=member.channel,
            ))
    return ResolvedSyncMap(authority=sync.authority, members=tuple(resolved))


def write_resolved_sync(path: Path, sync: ResolvedSyncMap) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(sync.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
