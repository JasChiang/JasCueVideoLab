"""Turn an EDL into concrete render instructions.

This layer has no veto. Every clip in the plan produces a segment, always. When
something cannot be done as written it is done a rung lower and the reason is
recorded with the measurement that forced it -- but the segment still exists,
and the film is still deliverable.

That is a deliberate inversion. The previous system could decide a piece of
material was not worth using and abandon it, or abandon a whole aspect, and it
did: a run during verification dropped every 9:16 candidate because one
declared region wanted to be fully visible, and another dropped the 16:9
aspect because a frame held two similar handsets. Both times the semantic layer
had already said what it wanted and the execution layer said no. Only the
review loop gets to say no here, and it says it about a finished cut.

Degrading is not free either. Each rung down needs its own evidence that the
rung above was attempted and failed, because a layer permitted to skip to the
safe option takes it every time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from typing import TYPE_CHECKING

from montagewright.schema import EDL, Clip, DegradationStep

if TYPE_CHECKING:  # a runtime import would make the two modules circular
    from montagewright.reframe import CropPath

# One safety margin, applied once, at the end. The old code added a little
# padding at detection, a little more at tracking, and more again at
# smoothing, so the crop that reached ffmpeg was tighter than any single layer
# intended and no one layer looked wrong.
CROP_MARGIN = 0.05


def seconds_to_frames(
    seconds: float | Decimal, fps: int | float | Decimal,
) -> int:
    """One half-up conversion for every delivered/NLE frame address."""

    return max(0, int(
        (Decimal(str(seconds)) * Decimal(str(fps))).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    ))


def allocate_timeline_frames(
    durations: list[float], fps: int,
) -> list[tuple[int, int]]:
    """Quantise one timeline, not each shot independently.

    The renderer and both NLE writers must share these exact boundaries.
    Rounding every duration separately creates gaps or extra frames whenever
    several fractional-frame shots are placed next to one another.
    """

    elapsed = Decimal("0")
    start = 0
    allocated = []
    for duration in durations:
        elapsed += Decimal(str(duration))
        end = seconds_to_frames(elapsed, fps)
        allocated.append((start, end))
        start = end
    return allocated


@dataclass(frozen=True)
class Source:
    """A resolved input file and the facts needed to place cuts in it."""

    source_id: str
    path: Path
    duration_seconds: float
    width: int
    height: int
    # Rational source clock, kept as text so 30000/1001 never becomes a
    # lossy decimal before NLE export.
    native_fps: str = "30/1"

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


@dataclass(frozen=True)
class CropBox:
    """A crop in normalised coordinates. Pixels happen at the ffmpeg edge."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not (0.0 < self.width <= 1.0 and 0.0 < self.height <= 1.0):
            raise ValueError("crop extent must sit within the frame")
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            raise ValueError("crop origin must sit within the frame")

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Project onto a real frame.

        Chroma-subsampled encoders want even numbers. An origin of zero is a
        legitimate answer and stays zero; only the extent is held above zero,
        since a crop of no width is not a crop.
        """

        def even_origin(value: float) -> int:
            return max(0, int(value) // 2 * 2)

        def even_extent(value: float, limit: int) -> int:
            return max(2, min(limit, int(value) // 2 * 2))

        x = even_origin(self.x * width)
        y = even_origin(self.y * height)
        return (
            x,
            y,
            even_extent(self.width * width, width - x),
            even_extent(self.height * height, height - y),
        )


@dataclass
class Segment:
    """One rendered piece of the timeline."""

    clip_id: str
    source: Source
    in_seconds: float
    out_seconds: float
    crop: CropBox | None = None
    # Set when the camera follows a subject. `crop` stays populated with the
    # opening position so anything reading a single box still works.
    crop_path: "CropPath | None" = None
    # How much louder or quieter this shot is than it was recorded, in dB.
    # Levelling makes every speaker the same loudness, which is not the same
    # as every speaker being right: one of them stood next to a road.
    gain_db: float = 0.0
    # Sound and picture are separate editorial assignments.  The first
    # executable slice still uses one source window for both, but this role
    # already prevents irrelevant on-location speech from leaking into a
    # project merely because another shot contains an interview.
    audio_role: str = "auto"
    audio_completion: str = "none"
    picture_role: str = "primary_action"
    coverage_claim_seconds: float | None = None
    # The stretch of the source this segment may not leave, when the card
    # named one. The renderer writes handles either side of every cut so an
    # editor opening the timeline can pull a shot longer; those were bounded
    # by the file, so half a second before a take is half a second of the
    # camera still being aimed, and half a second after it is often somebody
    # saying "again". A handle nobody can use is worse than none, because it
    # is there to be trusted.
    usable_from_seconds: float = 0.0
    usable_to_seconds: float = 0.0
    # How many source seconds pass for one screen second. 1.0 plays the take
    # at recorded speed; 2.0 reads twice the source per screen second (fast);
    # 0.5 reads half (slow motion). The renderer retimes the trimmed stream to
    # honour it. Every stage that measures the delivered timeline reads
    # `screen_duration_seconds`; only the crop path, which is evaluated on the
    # source-time stream before that retime, still measures `duration_seconds`.
    # Nothing sets this away from 1.0 yet: the plumbing is in place so the two
    # clocks can never be silently fused again once speed is exposed.
    speed_ratio: float = 1.0

    @property
    def duration_seconds(self) -> float:
        """Source seconds spanned, in - out on the source clock."""

        return self.out_seconds - self.in_seconds

    @property
    def screen_duration_seconds(self) -> float:
        """Seconds this segment occupies on the delivered timeline.

        Equal to the source span at recorded speed, and shorter or longer
        once the segment is sped up or slowed down. Timeline allocation,
        concatenation and any graphic laid over the cut measure here.
        """

        ratio = self.speed_ratio if self.speed_ratio > 0.0 else 1.0
        return (self.out_seconds - self.in_seconds) / ratio


@dataclass(frozen=True)
class AudioAssignment:
    """Source audio laid independently of picture cuts."""

    audio_id: str
    source: Source
    in_seconds: float
    out_seconds: float
    timeline_in_seconds: float
    timeline_start_frame: int
    frame_count: int
    role: str
    timeline_fps: int = 30
    completion: str = "none"
    gain_db: float = 0.0
    why: str = ""

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.timeline_fps


# What a delivery actually measures, per shape. Every segment is scaled to
# this, and zoom_budget is calibrated against it -- the two have to be the
# same number or the budget is guarding an output that does not exist.
DELIVERY_SIZES: dict[str, tuple[int, int]] = {
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
}


def delivery_size(target_aspect: float) -> tuple[int, int]:
    """The nearest standard size for this shape."""

    best, gap = (1080, 1920), None
    for wide, tall in DELIVERY_SIZES.values():
        off = abs((wide / tall) - target_aspect)
        if gap is None or off < gap:
            best, gap = (wide, tall), off
    return best


@dataclass
class RenderPlan:
    """Everything the renderer needs, plus an honest account of the cost."""

    project_id: str
    segments: list[Segment]
    # Pixels, not "whatever the first crop happened to measure".
    output_size: tuple[int, int] = (1080, 1920)
    # One CFR timeline for every source. Source FPS is observation, not a
    # property that may leak across cuts into a concatenated deliverable.
    output_fps: int = 30
    # Where in the track the bed starts, carried from the EDL so the renderer
    # does not have to know about planning to lay music that is not the intro.
    music_from_seconds: float = 0.0
    music_spans: list[tuple[float, float]] = field(default_factory=list)
    audio_assignments: list[AudioAssignment] = field(default_factory=list)
    # True when sound has been intentionally separated from picture. An
    # empty assignment list then means deliberate silence, not legacy auto.
    audio_track_explicit: bool = False
    degradations: list[DegradationStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return sum(
            segment.screen_duration_seconds for segment in self.segments
        )

    @property
    def degraded_clip_ids(self) -> set[str]:
        return {step.clip_id for step in self.degradations}


class MissingSource(KeyError):
    """Raised when an EDL names a source nobody supplied.

    This is the one refusal the executor is allowed. It is not a judgement
    about whether the material is good enough -- it is the plan referring to a
    file that does not exist, which no amount of degrading can render.
    """


def _centred_crop(source_aspect: float, target_aspect: float) -> CropBox:
    """The widest crop of `source_aspect` that fills `target_aspect`."""

    if target_aspect < source_aspect:
        width = target_aspect / source_aspect
        return CropBox(x=(1.0 - width) / 2.0, y=0.0, width=width, height=1.0)
    height = source_aspect / target_aspect
    return CropBox(x=0.0, y=(1.0 - height) / 2.0, width=1.0, height=height)


def _subject_crop(
    source_aspect: float, target_aspect: float, clip: Clip
) -> tuple[CropBox, str | None]:
    """Centre, and say the subject was never located.

    This used to anchor the crop from the planner's nine-box name. That name
    answers "which of these things do you mean", and it was being read as
    "aim here": "mid_right" put the frame at 0.615 for a handset spanning
    0.475 to 0.825, so the shot arrived half out of frame with the backdrop
    filling the rest. The same mistake had already been found and fixed on
    the handoff path, where two handsets at 0.359 and 0.635 were panned
    between 0.192 and 0.808.

    Every position the pipeline acts on is measured now -- from the clip
    card's box, from SAM propagation, or from a grounding call. Reaching
    this function means none of those had an answer, and a centred frame
    that says so is more use than a confident guess.
    """

    base = _centred_crop(source_aspect, target_aspect)
    reframe = clip.reframe
    if reframe is None or reframe.subject is None:
        return base, "no subject stated; centred"
    return base, "the subject was never located; centred"


def plan_render(
    edl: EDL,
    sources: dict[str, Source],
    *,
    target_aspect: float | None = None,
    crop_paths: "dict[str, CropPath] | None" = None,
    output_size: "tuple[int, int] | None" = None,
    output_fps: int = 30,
) -> RenderPlan:
    """Compile an EDL into segments. Never returns fewer than it was given.

    `target_aspect` is width over height; leave it out to keep each source as
    shot. The EDL's origin is irrelevant here -- a hand-written plan and a
    generated one compile identically, which is what makes a hand-written one
    useful for telling an execution bug apart from a planning one.
    """

    if output_fps not in {24, 25, 30, 50, 60}:
        raise ValueError("output_fps must be one of 24, 25, 30, 50 or 60")
    segments: list[Segment] = []
    degradations: list[DegradationStep] = []
    notes: list[str] = []

    for clip in edl.clips:
        source = sources.get(clip.source_id)
        if source is None:
            raise MissingSource(
                f"clip {clip.clip_id} names source {clip.source_id!r}, which "
                f"was not supplied. Known: {sorted(sources)}"
            )

        path = (crop_paths or {}).get(clip.clip_id)
        speed = float(getattr(clip, "speed", 1.0) or 1.0)
        # A digital move is authored on the screen clock and the renderer
        # divides that clock by speed before the retime, so a pan or a push
        # stretches across the wider source window and comes back as the move
        # that was authored. Speed and a move compose; no shot is held back
        # from it here.
        in_seconds, out_seconds = _resolve_times(
            clip, source, degradations, notes, speed=speed
        )
        crop = None
        if path is not None:
            # A followed subject supersedes the coarse anchor: the path was
            # built from where the subject actually was, not from a nine-box
            # guess. `crop` keeps the opening position so a caller reading one
            # box still sees something sensible.
            crop = path.keyframes[0].crop
        elif target_aspect is not None:
            crop = _resolve_crop(
                clip, source, target_aspect, degradations
            )
        segments.append(
            Segment(
                clip_id=clip.clip_id,
                source=source,
                in_seconds=in_seconds,
                out_seconds=out_seconds,
                speed_ratio=speed,
                crop=crop,
                crop_path=path,
                audio_role=clip.audio_role,
                audio_completion=clip.audio_completion,
                picture_role=clip.picture_role,
                coverage_claim_seconds=clip.coverage_claim_seconds,
                usable_from_seconds=clip.usable_from_seconds,
                usable_to_seconds=clip.usable_to_seconds,
            )
        )

    assert len(segments) == len(edl.clips), "the executor never drops a clip"
    frame_spans = allocate_timeline_frames(
        [segment.screen_duration_seconds for segment in segments], output_fps
    )
    timeline_starts = {
        segment.clip_id: start
        for segment, (start, _) in zip(segments, frame_spans, strict=True)
    }
    audio_assignments: list[AudioAssignment] = []
    for audio in edl.audio_clips:
        source = sources.get(audio.source_id)
        if source is None:
            raise MissingSource(
                f"audio {audio.audio_id} names source {audio.source_id!r}, "
                "which was not supplied"
            )
        timeline_start_frame = (
            timeline_starts[audio.starts_at_clip_id]
            + seconds_to_frames(audio.offset_seconds, output_fps)
        )
        out_seconds = min(audio.out_seconds, source.duration_seconds)
        if out_seconds <= audio.in_seconds:
            raise MissingSource(
                f"audio {audio.audio_id} window {audio.in_seconds:.3f}-"
                f"{audio.out_seconds:.3f}s is outside {audio.source_id}"
            )
        audio_assignments.append(AudioAssignment(
            audio_id=audio.audio_id,
            source=source,
            in_seconds=audio.in_seconds,
            out_seconds=out_seconds,
            timeline_in_seconds=timeline_start_frame / output_fps,
            timeline_start_frame=timeline_start_frame,
            frame_count=seconds_to_frames(
                out_seconds - audio.in_seconds, output_fps
            ),
            role=audio.role,
            timeline_fps=output_fps,
            completion=audio.completion,
            gain_db=audio.gain_db,
            why=audio.why,
        ))
    # Once a project uses the explicit sound contract, every retained piece
    # of sync/ambient/narrative source audio joins that track. Otherwise the
    # renderer would mute it with the picture while laying only the detached
    # interview assignment.
    explicit_audio_ids = {one.audio_id for one in audio_assignments}
    has_detached_narrative = any(
        one.role == "narrative" for one in audio_assignments
    )
    for clip, segment, (start_frame, end_frame) in zip(
        edl.clips, segments, frame_spans, strict=True
    ):
        if clip.audio_role not in {
            "narrative", "sync_action", "ambient_texture"
        }:
            continue
        if has_detached_narrative and clip.audio_role == "narrative":
            raise ValueError(
                f"{clip.clip_id} duplicates narrative audio already assigned "
                "on the independent track"
            )
        audio_id = f"shot-{clip.clip_id}"
        if audio_id in explicit_audio_ids:
            # A committed v2 manifest already materialises retained shot
            # sound as an assignment. Rebuilding it must be idempotent.
            continue
        frame_count = end_frame - start_frame
        audio_assignments.append(AudioAssignment(
            audio_id=audio_id,
            source=segment.source,
            in_seconds=segment.in_seconds,
            out_seconds=segment.out_seconds,
            timeline_in_seconds=start_frame / output_fps,
            timeline_start_frame=start_frame,
            frame_count=frame_count,
            role=clip.audio_role,
            timeline_fps=output_fps,
            completion=clip.audio_completion,
            gain_db=segment.gain_db,
            why=f"picture shot {clip.clip_id} retained its explicit source audio",
        ))
    total_frames = frame_spans[-1][1] if frame_spans else 0
    narrative: list[tuple[int, int, str]] = []
    for audio in audio_assignments:
        end_frame = audio.timeline_start_frame + audio.frame_count
        if (
            audio.timeline_start_frame < 0
            or audio.frame_count <= 0
            or end_frame > total_frames
        ):
            raise ValueError(
                f"audio {audio.audio_id} falls outside the {total_frames}-frame "
                "picture timeline"
            )
        if audio.role == "narrative":
            narrative.append((
                audio.timeline_start_frame, end_frame, audio.audio_id
            ))
    narrative.sort()
    for previous, here in zip(narrative, narrative[1:]):
        if here[0] < previous[1]:
            raise ValueError(
                f"narrative audio {previous[2]} overlaps {here[2]}"
            )
    return RenderPlan(
        project_id=edl.project_id,
        segments=segments,
        degradations=degradations,
        notes=notes,
        output_size=output_size or delivery_size(target_aspect or 9 / 16),
        output_fps=output_fps,
        music_from_seconds=getattr(edl, "music_from_seconds", 0.0),
        music_spans=list(getattr(edl, "music_spans", []) or []),
        audio_assignments=audio_assignments,
        audio_track_explicit=(
            bool(edl.audio_clips)
            or any(clip.audio_role != "auto" for clip in edl.clips)
        ),
    )


def _resolve_times(
    clip: Clip,
    source: Source,
    degradations: list[DegradationStep],
    notes: list[str],
    *,
    speed: float = 1.0,
) -> tuple[float, float]:
    """Clamp a clip into its source without ever discarding it.

    The clip's in and out are screen time; the source read that fills them is
    that span times ``speed``. At recorded speed the two are equal and this is
    the window it always was, so every existing cut resolves unchanged.
    """

    in_seconds = max(0.0, clip.approx_in_seconds)
    screen_span = max(0.0, clip.approx_out_seconds - clip.approx_in_seconds)
    source_span = screen_span * (speed if speed > 0.0 else 1.0)
    out_seconds = min(source.duration_seconds, in_seconds + source_span)

    # Past the end of what the take is worth using is a different fault from
    # past the end of the file, and it was not being noticed at all -- the
    # only question asked here was whether the time existed. It is recorded
    # rather than repaired, because shortening a shot to stay inside the
    # window changes the length the rhythm pass chose, and which of those to
    # give up is a judgement. The shot reviewer sees this one and the film is
    # still delivered.
    window = clip.usable_window
    if window is not None and out_seconds > window[1] + 1e-6:
        degradations.append(
            DegradationStep(
                clip_id=clip.clip_id,
                ladder="other",
                ladder_other="ran_past_the_usable_take",
                trigger=(
                    "the cut runs past the point the card said this take "
                    "stops being usable, so the tail is whatever follows it "
                    "-- a reset, a repositioned camera, somebody walking in"
                ),
                measured={
                    "out_seconds": round(out_seconds, 3),
                    "usable_to": round(window[1], 3),
                    "overrun_seconds": round(out_seconds - window[1], 3),
                },
            )
        )

    if out_seconds <= in_seconds:
        # The window fell outside the material. Keep whatever tail exists
        # rather than dropping the beat: a short segment can be reviewed and
        # replanned, a missing one just leaves a hole nobody can see.
        in_seconds = max(0.0, min(in_seconds, source.duration_seconds - 0.1))
        out_seconds = source.duration_seconds
        degradations.append(
            DegradationStep(
                clip_id=clip.clip_id,
                ladder="other",
                ladder_other="trim_window_clamped_to_source",
                trigger=(
                    f"requested {clip.approx_in_seconds:.3f}.."
                    f"{clip.approx_out_seconds:.3f}s from a "
                    f"{source.duration_seconds:.3f}s source"
                ),
                measured={
                    "requested_in": clip.approx_in_seconds,
                    "requested_out": clip.approx_out_seconds,
                    "source_duration": source.duration_seconds,
                    "resolved_duration": out_seconds - in_seconds,
                },
            )
        )
    elif clip.approx_out_seconds > source.duration_seconds:
        notes.append(
            f"{clip.clip_id}: out-point trimmed to the end of "
            f"{clip.source_id} ({source.duration_seconds:.3f}s)"
        )
    return in_seconds, out_seconds


def _resolve_crop(
    clip: Clip,
    source: Source,
    target_aspect: float,
    degradations: list[DegradationStep],
) -> CropBox | None:
    """Frame for the target aspect, recording any fall back to centre."""

    if abs(source.aspect_ratio - target_aspect) < 1e-3:
        return None

    crop, fallback_reason = _subject_crop(
        source.aspect_ratio, target_aspect, clip
    )
    if fallback_reason is not None:
        degradations.append(
            DegradationStep(
                clip_id=clip.clip_id,
                ladder="center_crop",
                trigger=fallback_reason,
                measured={
                    "source_aspect": source.aspect_ratio,
                    "target_aspect": target_aspect,
                },
            )
        )
    return crop
