"""An explicit, inspectable edit timeline compiled from the legacy EDL.

The v1 :class:`~montagewright.schema.Clip` deliberately preserved an old
convention: ``approx_in_seconds`` is a source in-point while
``approx_out_seconds - approx_in_seconds`` is screen duration.  That compact
shape kept the renderer stable, but it is too easy to confuse the source and
programme clocks once retiming and split edits coexist.

This module is the migration boundary.  It does not replace the proven
renderer yet; it projects every v1 decision into explicit source and timeline
ranges, then records the sound/picture links and edit points separately.  New
editing operations can target this contract while the existing executor keeps
accepting v1 through the adapter.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from montagewright.schema import EDL


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRange(_Strict):
    source_id: str = Field(min_length=1)
    in_seconds: float = Field(ge=0.0)
    out_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def ordered(self) -> "SourceRange":
        if self.out_seconds <= self.in_seconds:
            raise ValueError("source out must follow source in")
        return self


class TimelineRange(_Strict):
    in_seconds: float = Field(ge=0.0)
    out_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def ordered(self) -> "TimelineRange":
        if self.out_seconds <= self.in_seconds:
            raise ValueError("timeline out must follow timeline in")
        return self


class PictureItem(_Strict):
    item_id: str
    source: SourceRange
    timeline: TimelineRange
    speed: float = Field(gt=0.0)
    picture_role: str
    # Source seconds available on either side of the selected range.  These
    # are the legal room for slip/roll operations, not decorative metadata.
    head_handle_seconds: float = Field(default=0.0, ge=0.0)
    tail_handle_seconds: float = Field(default=0.0, ge=0.0)


class AudioItem(_Strict):
    item_id: str
    source: SourceRange
    timeline: TimelineRange
    role: str
    completion: str


class SyncLink(_Strict):
    link_id: str
    picture_item_id: str
    audio_item_id: str
    mode: Literal["lip_sync", "sync_action", "loose", "anchor"]
    allowed_drift_seconds: float = Field(default=0.0, ge=0.0)


class EditPoint(_Strict):
    edit_id: str
    at_seconds: float = Field(gt=0.0)
    outgoing_picture_id: str
    incoming_picture_id: str
    story_point: str = ""
    continuity_mode: str = "none"
    motivation: str = "content"
    source_event_ref: str = "none"
    # Sound edges are deliberately not assumed to coincide with picture.
    audio_edges: tuple[str, ...] = ()


class EditorialTimeline(_Strict):
    contract_version: Literal["montagewright-edit-timeline-v2"] = (
        "montagewright-edit-timeline-v2"
    )
    project_id: str
    duration_seconds: float = Field(gt=0.0)
    picture: tuple[PictureItem, ...]
    audio: tuple[AudioItem, ...] = ()
    sync_links: tuple[SyncLink, ...] = ()
    edit_points: tuple[EditPoint, ...] = ()

    @model_validator(mode="after")
    def references_and_tracks_are_coherent(self) -> "EditorialTimeline":
        picture_ids = {one.item_id for one in self.picture}
        audio_ids = {one.item_id for one in self.audio}
        for link in self.sync_links:
            if link.picture_item_id not in picture_ids:
                raise ValueError(f"sync link names unknown picture {link.picture_item_id}")
            if link.audio_item_id not in audio_ids:
                raise ValueError(f"sync link names unknown audio {link.audio_item_id}")
        for edit in self.edit_points:
            if edit.outgoing_picture_id not in picture_ids:
                raise ValueError("edit point names unknown outgoing picture")
            if edit.incoming_picture_id not in picture_ids:
                raise ValueError("edit point names unknown incoming picture")
        return self


def from_edl(edl: EDL) -> EditorialTimeline:
    """Project the hybrid v1 clock into explicit source/programme ranges."""

    picture: list[PictureItem] = []
    starts: dict[str, float] = {}
    cursor = 0.0
    for clip in edl.clips:
        duration = clip.approx_out_seconds - clip.approx_in_seconds
        speed = float(clip.speed or 1.0)
        source_out = clip.approx_in_seconds + duration * speed
        window = clip.usable_window
        head = max(0.0, clip.approx_in_seconds - window[0]) if window else 0.0
        tail = max(0.0, window[1] - source_out) if window else 0.0
        starts[clip.clip_id] = cursor
        picture.append(PictureItem(
            item_id=clip.clip_id,
            source=SourceRange(
                source_id=clip.source_id,
                in_seconds=clip.approx_in_seconds,
                out_seconds=source_out,
            ),
            timeline=TimelineRange(
                in_seconds=cursor,
                out_seconds=cursor + duration,
            ),
            speed=speed,
            picture_role=clip.picture_role,
            head_handle_seconds=head,
            tail_handle_seconds=tail,
        ))
        cursor += duration

    audio: list[AudioItem] = []
    links: list[SyncLink] = []
    picture_by_id = {one.item_id: one for one in picture}
    for one in edl.audio_clips:
        begins = starts[one.starts_at_clip_id] + one.offset_seconds
        duration = one.out_seconds - one.in_seconds
        audio_item = AudioItem(
            item_id=one.audio_id,
            source=SourceRange(
                source_id=one.source_id,
                in_seconds=one.in_seconds,
                out_seconds=one.out_seconds,
            ),
            timeline=TimelineRange(
                in_seconds=begins,
                out_seconds=begins + duration,
            ),
            role=one.role,
            completion=one.completion,
        )
        audio.append(audio_item)
        anchor = picture_by_id[one.starts_at_clip_id]
        if anchor.source.source_id == one.source_id and anchor.picture_role == "speaker":
            mode = "lip_sync"
            drift = 0.0
        elif one.role == "sync_action":
            mode = "sync_action"
            drift = 0.0
        else:
            mode = "loose"
            drift = 0.6
        links.append(SyncLink(
            link_id=f"sync:{one.audio_id}",
            picture_item_id=one.starts_at_clip_id,
            audio_item_id=one.audio_id,
            mode=mode,
            allowed_drift_seconds=drift,
        ))

    edits: list[EditPoint] = []
    for index, (outgoing, incoming) in enumerate(zip(picture, picture[1:])):
        at = outgoing.timeline.out_seconds
        audio_edges = tuple(
            item.item_id
            for item in audio
            if abs(item.timeline.in_seconds - at) <= 1e-6
            or abs(item.timeline.out_seconds - at) <= 1e-6
        )
        edits.append(EditPoint(
            edit_id=f"edit-{index:03d}",
            at_seconds=at,
            outgoing_picture_id=outgoing.item_id,
            incoming_picture_id=incoming.item_id,
            story_point=edl.clips[index + 1].story_point,
            continuity_mode=edl.clips[index + 1].continuity_mode,
            motivation=edl.clips[index + 1].cut_motivation,
            source_event_ref=edl.clips[index + 1].source_event_ref,
            audio_edges=audio_edges,
        ))

    return EditorialTimeline(
        project_id=edl.project_id,
        duration_seconds=cursor,
        picture=tuple(picture),
        audio=tuple(audio),
        sync_links=tuple(links),
        edit_points=tuple(edits),
    )
