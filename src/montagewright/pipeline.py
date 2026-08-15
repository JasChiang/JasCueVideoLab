"""Run the stages in order and report what happened.

The order is fixed and each stage answers one question:

    rhythm    how long should each shot be, given the music and the material
    subject   where is the thing this shot is about, at a few sampled moments
    reframe   what crop follows that, inside this shot's energy budget
    ground    which musical event does each cut actually land on
    render    the file

Every semantic question goes to the model; every measurement stays local. The
report at the end says which cuts landed on music and which shots had to
settle for less than the plan asked, because a cut that quietly did neither is
indistinguishable from one that did both.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from montagewright.clipcard import find_subject, load_card
from montagewright.cost import Ledger, Spend
from montagewright.executor import (
    CropBox, RenderPlan, Source, delivery_size, plan_render,
)
from montagewright.grounding import BeatGrid, apply_to_edl, ground_timeline
from montagewright.planner import Usage, decide_rhythm, locate_subject
from montagewright.reframe import (
    CropPath,
    DEADBAND,
    Keyframe,
    achieved_upscale,
    zoom_budget,
    Observation,
    OutOfFrame,
    build_crop_path,
    build_sweep_path,
    build_look_path,
    build_tilt_path,
    build_zoom_path,
    observations_from_sam,
)
from montagewright.renderer import RenderResult, render
from montagewright.schema import EDL, DegradationStep

# Enough samples to see a subject change direction, few enough that one shot
# costs a fraction of a cent. Interpolation covers the gaps; SAM propagation
# replaces this when per-frame accuracy starts to matter.
SUBJECT_SAMPLES = 5

# Analysis rate for mask propagation. Four a second catches a gesture starting
# and finishing; the cost is roughly ten seconds of wall time per shot, which
# is worth it against interpolating between samples 0.6 seconds apart.
TRACK_FPS = 4.0

# How much of a shot the tracker has to hold before its trajectory is worth
# more than the sampled positions. Below this the samples are used and the
# shortfall is recorded, because a track that survived one frame in nine is
# not a measurement, it is a single guess wearing a measurement's name.
#
# A half was too much of a fraction and not enough of a floor. A subject a
# hand covers for a moment, or that leaves the frame and comes back, loses
# samples for reasons that are facts about the take -- and the crop
# interpolates between the observations it does have. Measured on an event
# shoot: six usable frames out of thirteen, identity confirmed on every
# frame it was asked about, refused at 46%. What actually distinguishes a
# measurement from a guess is having several observations spread across the
# shot, so that is what is required.
TRACK_QUORUM = 0.34
TRACK_MINIMUM_OBSERVATIONS = 3
_SAM_PREFLIGHTED: dict[Path, tuple[int, int]] = {}

# How far outside a cut a confirmed frame may sit and still be worth seeding
# from. Far enough to reach the close-up that opens a take; not so far that
# the tracker is asked to cross a scene.
CONFIRMED_REACH_SECONDS = 6.0

# One exact identity seed may replace repeated semantic checkpoints only over
# a short, uninterrupted local track. Longer shots keep the existing
# multi-anchor proof even when SAM's masks look geometrically smooth.
ADAPTIVE_TRACK_MAX_SECONDS = 5.0


def align_speaker_pictures_to_audio(edl: EDL) -> tuple[EDL, list[str]]:
    """Put every lip-synced picture on the independent audio source clock.

    One answer may continue while picture cuts to illustrative B-roll and
    later returns to the speaker.  The returning shot must resume at the
    source frame being heard then; aligning only the shot where the audio
    assignment starts makes that return visibly out of sync.

    B-roll and reactions are deliberately untouched.  A speaker shot is
    aligned only when its own source is the source of the unique narrative
    assignment active at that point (or when it is the assignment's anchor
    shot before a positive offset).  Ambiguous or ungrounded speaker pictures
    fail closed instead of pretending to be lip-synced.
    """

    starts: dict[str, float] = {}
    cursor = 0.0
    for clip in edl.clips:
        starts[clip.clip_id] = cursor
        cursor += clip.approx_out_seconds - clip.approx_in_seconds

    narrative = []
    for audio in edl.audio_clips:
        if audio.role != "narrative":
            continue
        anchored = starts[audio.starts_at_clip_id]
        at = anchored + audio.offset_seconds
        narrative.append((audio, at, at + audio.out_seconds - audio.in_seconds))

    aligned, notes = [], []
    for clip in edl.clips:
        if clip.picture_role != "speaker":
            aligned.append(clip)
            continue
        shot_at = starts[clip.clip_id]
        candidates = [
            (audio, audio_at)
            for audio, audio_at, audio_end in narrative
            if audio.source_id == clip.source_id
            and (
                audio.starts_at_clip_id == clip.clip_id
                or audio_at <= shot_at < audio_end
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"speaker picture {clip.clip_id} must match exactly one active "
                f"narrative assignment from {clip.source_id}; found "
                f"{len(candidates)}"
            )
        audio, audio_at = candidates[0]
        source_in = audio.in_seconds + shot_at - audio_at
        if source_in < 0:
            raise ValueError(
                f"speaker picture {clip.clip_id} cannot begin "
                f"{abs(source_in):.3f}s before its source"
            )
        duration = clip.approx_out_seconds - clip.approx_in_seconds
        aligned.append(clip.model_copy(update={
            "approx_in_seconds": source_in,
            "approx_out_seconds": source_in + duration,
        }))
        notes.append(
            f"{clip.clip_id}: speaker picture aligned to {audio.audio_id} "
            f"at source {source_in:.3f}s"
        )
    return edl.model_copy(update={"clips": aligned}), notes


class ReferenceShotUnusable(RuntimeError):
    """This shot cannot be delivered against the locked identity.

    Refusing to track a lookalike, and refusing to crop a reference-critical
    target on a box no local geometry ever confirmed, are both right and
    neither is negotiable. Ending the run over one shot is a separate
    decision, and it was being made by accident: the messages said "reselect
    the shot", which is advice to a person, and no caller could act on a
    bare RuntimeError anyway. Naming the shot and the identity lets the
    layer that owns the selection swap in the alternate its own planning
    already paid for.
    """

    def __init__(self, clip_id: str, entity_id: str, message: str) -> None:
        super().__init__(message)
        self.clip_id = clip_id
        self.entity_id = entity_id


class ReferenceIdentityUnconfirmed(ReferenceShotUnusable):
    """The frames could not show that this is the same instance."""


class ReferenceShotsUnusable(RuntimeError):
    """Every shot that could not be delivered, found in one pass.

    One at a time meant one repair per attempt with a full re-render
    between them, so a bounded three attempts covered three shots -- and
    material shot at an event with four similar handsets on the tables has
    more than three. The work spent on the other shots is discarded either
    way; doing it once and reporting all of them lets the layer holding the
    selection fix them together.
    """

    def __init__(self, faults: "list[ReferenceShotUnusable]") -> None:
        super().__init__("; ".join(str(one) for one in faults))
        self.faults = list(faults)


class ReferenceGeometryUnavailable(ReferenceShotUnusable):
    """It is the right instance, and nothing local can say where it is.

    A tracker that holds nothing is not a smaller amount of tracking: the
    crop would have to come from a model's box on a sampled frame, which is
    the substitution the whole reference path exists to refuse.
    """


@dataclass
class Report:
    """What the run did, in the terms someone would ask about it."""

    aligned_cuts: int = 0
    total_cuts: int = 0
    following_shots: int = 0
    static_shots: int = 0
    # Two layers a viewer sees at once.  A fixed digital crop over an authored
    # source pan is not a static shot, and a digital pan over a locked source
    # is not source camera work.  The old held/following totals collapsed them.
    source_motion: dict[str, str] = field(default_factory=dict)
    source_motion_details: dict[str, dict] = field(default_factory=dict)
    digital_motion: dict[str, str] = field(default_factory=dict)
    degradations: list[DegradationStep] = field(default_factory=list)
    subject_notes: dict[str, str] = field(default_factory=dict)
    # Lightweight, durable geometry from reliable SAM tracks. Full masks are
    # temporary; downstream layout only needs where the subject was in the
    # source frame at each moment.
    subject_tracks: dict[str, list[dict]] = field(default_factory=dict)
    # The semantic proof behind reference-conditioned tracks.  SAM boxes say
    # where pixels went; these records say which locked identity Gemini
    # confirmed on exact source PTS frames before those boxes were trusted.
    reference_grounding: dict[str, dict] = field(default_factory=dict)
    # Where a plan contradicted itself, kept rather than printed. These were
    # written to stdout and nowhere else, so the one that mattered -- a shot
    # naming a subject its own window never reaches -- was on screen while
    # the same shot was replanned twice into the same failure.
    plan_disagreements: list[str] = field(default_factory=list)
    # What the direction asked for against what the cut runs to. Three layers
    # each made a defensible call -- 45 seconds of material, ten shots, a
    # beat-led length -- and the result was a third of the intended film with
    # nobody reporting the gap.
    target_seconds: float | None = None
    coverage_seconds: float | None = None
    unsupported_seconds: float | None = None
    coverage_details: list[dict] = field(default_factory=list)
    moves_too_short: dict[str, str] = field(default_factory=dict)
    rhythm_decisions: dict[str, dict] = field(default_factory=dict)
    # Enlargement actually applied per shot. Reported as a number because
    # sharpness is measurable: nobody should have to tell a soft proxy apart
    # from an over-enlarged shot by eye, and at preview resolution they look
    # the same.
    upscales: dict[str, float] = field(default_factory=dict)
    delivered_seconds: float | None = None
    usages: list[Usage] = field(default_factory=list)
    # What each stage cost, in dollars. The totals were only ever tokens,
    # which answers "how much was sent" rather than "where did the money go"
    # -- and the two point at different stages entirely.
    ledger: Ledger | None = None

    @property
    def input_tokens(self) -> int:
        return sum(usage.input_tokens for usage in self.usages)

    @property
    def output_tokens(self) -> int:
        return sum(
            usage.output_tokens + usage.thought_tokens for usage in self.usages
        )

    @property
    def duration_shortfall(self) -> float | None:
        if self.target_seconds is None or self.delivered_seconds is None:
            return None
        return round(self.target_seconds - self.delivered_seconds, 2)

    def spend(self) -> "Spend":
        return self.ledger.summary() if self.ledger is not None else Spend(
            cap_usd=0.0, spent_usd=0.0, remaining_usd=0.0, calls=0, by_stage={}
        )

    def summary(self) -> str:
        gap = self.duration_shortfall
        spent = (
            f", ${self.ledger.spent_usd:.4f}"
            if self.ledger is not None and self.ledger.entries
            else ""
        )
        tail = (
            f", {abs(gap):.0f}s {'short of' if gap > 0 else 'over'} the "
            f"{self.target_seconds:.0f}s asked for"
            if gap is not None and abs(gap) >= 1.0
            else ""
        )
        # A film where nothing asked for a beat is not a film that missed
        # every beat. The fallback for "no rhythm pass ran" was reading a
        # speech-led cut -- thirteen shots, every one deliberately off the
        # grid so a sentence could finish -- as 0/13 aligned.
        if self.rhythm_decisions:
            wanted = sum(
                1
                for entry in self.rhythm_decisions.values()
                if entry.get("cut_on_beat")
            )
        else:
            wanted = self.total_cuts
        return (
            f"{self.aligned_cuts}/{wanted} cuts on a musical event "
            f"({self.total_cuts - wanted} content-led by choice), "
            f"{self.following_shots} digital frames moving, "
            f"{self.static_shots} digital frames held, "
            f"{len(self.degradations)} degradations, "
            f"{self.input_tokens} in / {self.output_tokens} out tokens"
            f"{spent}{tail}"
        )


def _digital_motion_of(path: CropPath | None) -> str:
    """Describe measured crop motion instead of repeating the plan's label."""

    if path is None or path.is_static:
        return "hold"
    crops = [frame.crop for frame in path.keyframes]
    first, last = crops[0], crops[-1]
    xs = [crop.x + crop.width / 2 for crop in crops]
    ys = [crop.y + crop.height / 2 for crop in crops]
    legs_x = [right - left for left, right in zip(xs, xs[1:])]
    legs_y = [right - left for left, right in zip(ys, ys[1:])]
    dx, dy = xs[-1] - xs[0], ys[-1] - ys[0]
    travel_x = sum(abs(delta) for delta in legs_x)
    travel_y = sum(abs(delta) for delta in legs_y)
    scale = first.width / max(last.width, 1e-9)
    parts: list[str] = []
    reverses = any(a * b < 0 for a, b in zip(legs_x, legs_x[1:])) or any(
        a * b < 0 for a, b in zip(legs_y, legs_y[1:])
    )
    if reverses:
        parts.append("multi_stop")
    if abs(scale - 1.0) >= 0.02:
        parts.append("push_in" if scale > 1.0 else "pull_out")
    if travel_x >= 0.002 and travel_y >= 0.002:
        parts.append("diagonal")
    elif travel_x >= 0.002:
        parts.append("pan_right" if dx >= 0 else "pan_left")
    elif travel_y >= 0.002:
        parts.append("tilt_down" if dy >= 0 else "tilt_up")
    return "+".join(parts) or "moving"


def _audit_static_holds(
    edl: EDL, max_static_seconds: float,
) -> tuple[EDL, list[str]]:
    """Audit long locked holds after rhythm has chosen the resolved windows.

    Cutting a shot here would silently throw away speech/action and shorten
    the film. The review loop treats these notes as mandatory replan input;
    runs without review still expose the violation instead of disguising it.
    """

    if max_static_seconds <= 0:
        return edl, []
    notes: list[str] = []
    for clip in edl.clips:
        reframe = clip.reframe
        seconds = clip.approx_out_seconds - clip.approx_in_seconds
        visually_static = (
            reframe is not None
            and reframe.camera_move == "hold"
            and reframe.source_motion_role == "locked"
        )
        exempt = bool(
            reframe is not None
            and reframe.pacing_exception
            and reframe.pacing_exception_reason.strip()
        )
        if visually_static and not exempt and seconds > max_static_seconds:
            notes.append(
                f"{clip.clip_id} static hold resolves to {seconds:.1f}s, over "
                f"the direction's {max_static_seconds:.1f}s maximum; replan "
                "or document a content-led exception"
            )
    return edl, notes


def _resolved_sequence_disagreements(edl: EDL) -> list[str]:
    notes: list[str] = []
    for left, right in zip(edl.clips, edl.clips[1:]):
        overlap = min(left.approx_out_seconds, right.approx_out_seconds) - max(
            left.approx_in_seconds, right.approx_in_seconds
        )
        if left.source_id == right.source_id and overlap > 0.25:
            notes.append(
                f"{left.clip_id} and {right.clip_id} repeat {overlap:.1f}s "
                "of the same resolved source window"
            )
        if left.source_id != right.source_id:
            continue
        gap = right.approx_in_seconds - left.approx_out_seconds
        if -0.25 <= gap <= 0.25:
            notes.append(
                f"{left.clip_id} and {right.clip_id} cut nearly continuously "
                f"inside {left.source_id}; the {gap:+.2f}s source-clock jump "
                "may read as an accidental cut"
            )
        left_reframe, right_reframe = left.reframe, right.reframe
        if left_reframe is None or right_reframe is None:
            continue
        left_labels = [look.at for look in left_reframe.looks]
        right_labels = [look.at for look in right_reframe.looks]
        if not left_labels or not right_labels or left_labels[-1] != right_labels[0]:
            continue
        if not left_reframe.look_boxes or not right_reframe.look_boxes:
            continue
        left_width = float(left_reframe.look_boxes[-1][2])
        right_width = float(right_reframe.look_boxes[0][2])
        scale = max(left_width, right_width) / max(
            min(left_width, right_width), 1e-9
        )
        if scale >= 1.35:
            notes.append(
                f"{left.clip_id} to {right.clip_id} keeps the same subject in "
                f"{left.source_id} but changes planned scale {scale:.2f}x; "
                "review as a possible punch-in jump cut"
            )
    return notes


def _may_ask(client: Any) -> bool:
    """Whether a grounding call is available at all.

    Rebuilding a plan for a timeline or a recut has no key and needs none:
    the subject positions are in the cards. A shot the cards cannot answer
    falls through to a centred frame, which the executor records -- the same
    answer it gives when nothing could be located, rather than an exception
    from a path that was only ever meant to draw a crop.
    """

    return client is not None


def _afford(report: "Report") -> None:
    """Stop before a paid call rather than after it.

    The subject pass runs once per shot that needs locating, so a plan with
    twenty shots is twenty calls; a cap consulted only between stages lets
    all twenty through after it has already been reached.
    """

    if report.ledger is not None:
        report.ledger.check()


def _charge(report: "Report", stage: str, usage: Usage) -> None:
    """Keep the report's token tally; ``ask`` settles the stage ledger."""

    report.usages.append(usage)


def _source_motion_measurement(
    intervals: Any, starts_seconds: float, ends_seconds: float
) -> dict[str, Any]:
    """Summarise local optical-flow facts for the selected source window."""

    selected = []
    travel = 0.0
    peak = 0.0
    for interval in intervals or ():
        overlap = max(
            0.0,
            min(ends_seconds, float(interval.ends_seconds))
            - max(starts_seconds, float(interval.starts_seconds)),
        )
        if overlap <= 0.0:
            continue
        seconds = max(
            1e-9, float(interval.ends_seconds) - float(interval.starts_seconds)
        )
        selected.append(interval)
        if str(interval.state) == "moving":
            travel += float(interval.travel_vw) * overlap / seconds
            peak = max(peak, float(interval.peak_vw_s))
    states = tuple(dict.fromkeys(str(one.state) for one in selected))
    return {
        "available": bool(selected),
        "states": list(states),
        "moving": "moving" in states,
        "travel_frame_widths": round(travel, 4),
        "peak_frame_widths_per_second": round(peak, 4),
        "settles": any(bool(one.settles) for one in selected),
        "event_ids": [str(one.event_id) for one in selected],
    }


def _locate_subject(frames, description, *, client, report):
    """Keep test/offline callers free of a keyword only live runs need."""

    if report.ledger is None:
        return locate_subject(frames, description, client=client)
    return locate_subject(
        frames, description, client=client, ledger=report.ledger
    )


# What a file is does not change while it sits there, and reading it costs
# an ffprobe -- a whole process, tens of milliseconds. Opening a finished cut
# asked for the same dozen files every time the timeline was drawn, which is
# most of the second and a half it took before anything appeared.
_PROBED: dict[tuple[str, int, int], "Source"] = {}


def probe(source_id: str, path: Path) -> Source:
    import json

    try:
        stat = path.stat()
        seen = (str(path), stat.st_size, int(stat.st_mtime))
    except OSError:
        seen = None
    if seen is not None and seen in _PROBED:
        kept = _PROBED[seen]
        # The same file may be known by more than one id across runs.
        return (
            kept if kept.source_id == source_id
            else Source(source_id=source_id, path=kept.path,
                        duration_seconds=kept.duration_seconds,
                        width=kept.width, height=kept.height,
                        native_fps=kept.native_fps)
        )

    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate:stream_tags=rotate:"
            "stream_side_data=rotation:format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    rotation = int(float(
        stream.get("tags", {}).get("rotate")
        or next(
            (
                side.get("rotation", 0)
                for side in stream.get("side_data_list", [])
                if "rotation" in side
            ),
            0,
        )
        or 0
    )) % 360
    width, height = int(stream["width"]), int(stream["height"])
    if rotation in {90, 270}:
        width, height = height, width
    native_fps = "30/1"
    for candidate in (
        stream.get("avg_frame_rate"), stream.get("r_frame_rate")
    ):
        try:
            rate = Fraction(str(candidate))
            if rate > 0:
                native_fps = f"{rate.numerator}/{rate.denominator}"
                break
        except (ValueError, ZeroDivisionError):
            continue
    found = Source(
        source_id=source_id,
        path=path,
        duration_seconds=float(payload["format"]["duration"]),
        width=width,
        height=height,
        native_fps=native_fps,
    )
    if seen is not None:
        _PROBED[seen] = found
    return found


def _sample_frames(
    source: Source, start: float, end: float, work: Path
) -> tuple[list[Path], list[float]]:
    """Pull evenly spaced stills across the shot's own window."""

    span = max(end - start, 0.1)
    times = [
        start + span * (index + 0.5) / SUBJECT_SAMPLES
        for index in range(SUBJECT_SAMPLES)
    ]
    frames = []
    for index, at in enumerate(times):
        destination = work / f"{source.source_id}-{index}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{at:.3f}", "-i", str(source.path),
                "-frames:v", "1", "-vf", "scale=960:-2", str(destination),
            ],
            check=True,
        )
        frames.append(destination)
    return frames, times


def _track_subject(
    source: Source,
    clip,
    subject_description: str,
    seed_box: list[int],
    checkpoint: Path,
    work: Path,
    *,
    seed_time_seconds: float | None = None,
    track_name: str | None = None,
    semantic_anchors: tuple[
        tuple[float, tuple[float, float, float, float]], ...
    ] = (),
    require_identity_validation: bool = False,
    seed_lineage: Any | None = None,
) -> tuple[list[Observation], dict[str, int]]:
    """Propagate a Gemini seed across every analysed frame of the shot."""

    from montagewright.measure.sam_tracking import track_bbox_sam21

    seed_time_ms = int(
        (seed_time_seconds
         if seed_time_seconds is not None
         else clip.approx_in_seconds + 0.1) * 1000
    )
    exact_seed: dict[str, Any] = {}
    if seed_lineage is not None:
        seed_time_ms = int(seed_lineage.frame_time_ms)
        if (
            seed_time_seconds is not None
            and round(seed_time_seconds * 1000) != seed_time_ms
        ):
            raise ValueError(
                "semantic seed time does not match exact-frame PTS lineage"
            )
        exact_seed = {
            "asset_id": seed_lineage.video_asset_id,
            "seed_frame_pts": seed_lineage.frame_pts,
            "seed_frame_sha256": seed_lineage.frame_sha256,
            "seed_source_width": seed_lineage.width,
            "seed_source_height": seed_lineage.height,
        }

    track = track_bbox_sam21(
        video_path=source.path,
        checkpoint_path=checkpoint,
        seed_time_ms=seed_time_ms,
        seed_box_2d=seed_box,
        target_description=subject_description,
        output_dir=work / (
            f"sam-{clip.clip_id}-{track_name}"
            if track_name else f"sam-{clip.clip_id}"
        ),
        seed_source=(
            "reference_exact_frame_grounding"
            if seed_lineage is not None
            else "gemini_frame_grounding"
        ),
        analysis_fps=TRACK_FPS,
        # A handset on a table in a wide event shot is a couple of hundred
        # pixels across at 960, and the masks came back sparse and broken --
        # three usable frames out of twelve on one shot, which then failed
        # the tracking quorum with its identity confirmed on every frame it
        # was asked about. The decode is local and the model is already
        # loaded; the pixels are the cheapest thing to give it.
        max_side=1440,
        # Prompted with a box drawn tight around the phone, SAM tends to cut
        # inside it -- the screen, or the lit half of a folded body. A little
        # room around the prompt lets the mask take the whole object, which
        # is what the exact-frame box it is being compared against describes.
        seed_box_padding_ratio=0.06,
        allowed_start_ms=int(clip.approx_in_seconds * 1000),
        allowed_end_ms=int(clip.approx_out_seconds * 1000),
        **exact_seed,
    )
    return observations_from_sam(
        track,
        clip_start_seconds=clip.approx_in_seconds,
        semantic_anchors=semantic_anchors,
        require_identity_validation=require_identity_validation,
    )


def _measure_looks(
    looks, source, clip, work: Path, report, client, target_aspect: float,
    checkpoint: Path | None = None,
    reference_samples: Mapping[
        str,
        tuple[
            list[dict[str, Any]],
            list[float],
            tuple[tuple[float, tuple[float, float, float, float]], ...],
        ],
    ] | None = None,
) -> tuple[
    list[tuple[float, float, float, float]],
    str,
    list[list[tuple[float, float, float]]],
]:
    """Turn each look into a place on the frame, measured rather than assumed.

    One grounding call per distinct subject, so two looks at the same thing
    -- which is how a push in is written -- are located once and paid for
    once.

    Returns where each look settles, anything that could not be found, and
    where each subject was at every moment it was sampled. The last was being
    averaged away: the frames are pulled across the shot and the boxes came
    back one per frame, and all of them were collapsed into a mean before
    anything downstream saw them. So a subject that walked across the frame
    was handed on as one point in the middle of its own path, and a shot
    planned to follow it held on a place it passed through.

    The crop width comes from framing: `fill` closes toward the subject
    within the source's own budget, and the others take the widest crop
    there is and place the subject inside it. The vertical centre is nudged
    so the subject lands on the line its framing asks for rather than in the
    middle of the crop, which is what makes a held frame read as composed.
    """

    from montagewright.reframe import PLACEMENT

    # Checked here as well as at the call site. Pulling frames runs ffmpeg
    # over the take, and a guard that lives only in the caller stops being a
    # guard the moment this is called from anywhere else.
    if not _may_ask(client):
        return [], "", []

    frames, times = _sample_frames(
        source, clip.approx_in_seconds, clip.approx_out_seconds, work
    )
    # Shot time, not source time: everything downstream of here counts from
    # the cut. These came back from the sampler and were being dropped on the
    # floor by the caller, which is why a box could not be tied to a moment.
    moments = [at - clip.approx_in_seconds for at in times]
    if target_aspect < source.aspect_ratio:
        base, base_height = target_aspect / source.aspect_ratio, 1.0
    else:
        base, base_height = 1.0, source.aspect_ratio / target_aspect
    seen: dict[str, tuple[float, float, float]] = {}
    walked: dict[str, list[tuple[float, float, float]]] = {}
    stops: list[tuple[float, float, float, float]] = []
    tracks: list[list[tuple[float, float, float]]] = []
    missing: list[str] = []
    for look_index, look in enumerate(looks):
        subject_key = look.entity_id or look.at
        if subject_key not in seen:
            reference = (
                (reference_samples or {}).get(look.entity_id)
                if look.entity_id else None
            )
            if look.entity_id and reference is None:
                missing.append(f"{look.at} ({look.entity_id}: identity unverified)")
                continue
            if reference is not None:
                boxes, look_times, semantic_anchors = reference
                look_moments = [
                    at - clip.approx_in_seconds for at in look_times
                ]
            else:
                _afford(report)
                boxes, usage = _locate_subject(
                    frames, look.at, client=client, report=report
                )
                _charge(report, "subject", usage)
                look_times = times
                look_moments = moments
                semantic_anchors = ()
            found = [
                one for one in boxes
                if one.get("present") and one.get("centre_x") is not None
            ]
            if not found:
                missing.append(look.at)
                continue
            # Kept as a path as well as a place. The mean is still what a
            # stop settles on -- it is the right answer for something that
            # is not going anywhere, and it is what the framing and the
            # travel between stops are computed from -- but it is no longer
            # all that is known.
            seen[subject_key] = (
                sum(float(one["centre_x"]) for one in found) / len(found),
                sum(float(one["centre_y"]) for one in found) / len(found),
                sum(float(one.get("height") or 0.0) for one in found) / len(found),
            )
            walked[subject_key] = sorted(
                (
                    look_moments[int(one["frame_index"])],
                    float(one["centre_x"]),
                    float(one["centre_y"]),
                )
                for one in found
                if 0 <= int(one.get("frame_index", -1)) < len(look_moments)
            )
            # Gemini identifies which object the edit means; SAM turns that
            # semantic seed into the dense trajectory used by the crop. This
            # used to happen only in the single-look branch, so a push or a
            # handoff bypassed SAM precisely when a tight crop made small
            # tracking errors most visible.
            if checkpoint is not None and reference is None:
                seed = found[0]
                frame_index = int(seed.get("frame_index", -1))
                seed_time = (
                    look_times[frame_index]
                    if 0 <= frame_index < len(look_times)
                    else clip.approx_in_seconds + 0.1
                )
                try:
                    tracked, states = _track_subject(
                        source,
                        clip,
                        look.at,
                        _seed_box(seed),
                        checkpoint,
                        work,
                        seed_time_seconds=seed_time,
                        track_name=str(look_index),
                        semantic_anchors=semantic_anchors,
                        require_identity_validation=bool(look.entity_id),
                    )
                except Exception as error:  # sampled positions remain valid
                    report.subject_notes[clip.clip_id] = (
                        "SAM unavailable for a multi-look subject; using "
                        f"sampled positions ({type(error).__name__})"
                    )
                else:
                    total = sum(states.values()) or 1
                    kept = states.get("tracked", 0)
                    report.subject_notes[clip.clip_id] = (
                        f"SAM tracked {look.at}: {states}"
                    )
                    if kept / total >= TRACK_QUORUM and tracked:
                        report.subject_tracks.setdefault(
                            clip.clip_id, []
                        ).extend({
                            "seconds": round(one.seconds, 4),
                            "centre_x": round(one.centre_x, 6),
                            "centre_y": round(one.centre_y, 6),
                            "width": round(one.width, 6),
                            "height": round(one.height, 6),
                            "subject": look.at,
                            "entity_id": look.entity_id,
                            "semantic_identity_status": (
                                "reference_validated"
                                if look.entity_id else "description_grounded"
                            ),
                            "source": "sam2.1",
                        } for one in tracked)
                        walked[subject_key] = [
                            (one.seconds, one.centre_x, one.centre_y)
                            for one in tracked
                        ]
                    else:
                        report.degradations.append(
                            DegradationStep(
                                clip_id=clip.clip_id,
                                ladder="other",
                                ladder_other="tracking_lost_most_frames",
                                trigger=(
                                    "SAM held the multi-look subject in "
                                    f"{kept} of {total} analysed frames; "
                                    "using sampled positions"
                                ),
                                measured={
                                    "tracked_frames": float(kept),
                                    "analysed_frames": float(total),
                                    "kept_fraction": round(kept / total, 3),
                                },
                            )
                        )
        if subject_key not in seen:
            continue
        centre_x, centre_y, tall = seen[subject_key]
        width = base
        if look.framing == "fill" and tall > 0.0:
            # The same reach a push used to compute, bounded the same way.
            width = base * max(0.2, min(0.9, tall / 0.66))
        height = min(1.0, base_height * (width / base) if base > 0 else base_height)
        share = PLACEMENT.get(look.framing, 0.5)
        lift = height * (0.5 - share)
        stops.append((
            -1.0 if look.presentation_intent == "transition_pass"
            else max(0.0, float(look.seconds)),
            centre_x,
            centre_y + lift,
            width,
        ))
        # Keep the measured path.  Whether it matters is a property of the
        # delivered crop, not of the full source: a tiny source-space drift
        # can become a visible reversal during a tight push.  The path builder
        # applies its deadband after it knows the crop width.
        path = walked.get(subject_key) or []
        tracks.append([(when, x, y + lift) for when, x, y in path])
    return stops, "、".join(missing), tracks


def _seed_box(box: dict[str, Any]) -> list[int]:
    """Gemini's centre-and-size answer as the tracker's x-first 0..1000 box."""

    half_w = float(box["width"]) / 2.0
    half_h = float(box["height"]) / 2.0
    centre_x = float(box["centre_x"])
    centre_y = float(box["centre_y"])
    return [
        int(max(0.0, centre_x - half_w) * 1000),
        int(max(0.0, centre_y - half_h) * 1000),
        int(min(1.0, centre_x + half_w) * 1000),
        int(min(1.0, centre_y + half_h) * 1000),
    ]


def write_crops(paths: dict[str, CropPath], destination: Path) -> None:
    """Keep the crop paths the render actually used.

    Rebuilding them later is cheap arithmetic only for a held frame whose
    subject a card can name. A follow came out of a mask propagation that
    needed a checkpoint and a grounding call, and neither is available after
    the fact -- so without this the interface could only redraw every move as
    a centred still and present that as what happened.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                clip_id: [
                    {
                        "at": round(frame.seconds, 3),
                        "x": round(frame.crop.x, 5),
                        "y": round(frame.crop.y, 5),
                        "w": round(frame.crop.width, 5),
                        "h": round(frame.crop.height, 5),
                    }
                    for frame in path.keyframes
                ]
                for clip_id, path in paths.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def read_crops(source: Path) -> dict[str, CropPath]:
    """The crop paths a render actually used, back off disk.

    The counterpart nobody wrote. `write_crops` exists because a follow came
    out of a mask propagation that cannot be repeated afterwards, and the
    interface reads it for exactly that reason -- but the timeline exports
    rebuilt the plan instead, with no client and no checkpoint, and wrote
    whatever that came to into the FCPXML.

    Which is the one output where being approximately right is worst. A
    report that disagrees with the film is a wrong number on a page; an
    edit list that disagrees with it opens in Final Cut as a different cut,
    and the person who opens it has no way to tell.
    """

    if not source.exists():
        return {}
    try:
        stored = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    paths: dict[str, CropPath] = {}
    for clip_id, frames in (stored or {}).items():
        keyframes = [
            Keyframe(
                seconds=float(frame["at"]),
                crop=CropBox(
                    x=float(frame["x"]), y=float(frame["y"]),
                    width=float(frame["w"]), height=float(frame["h"]),
                ),
            )
            for frame in frames or []
            if all(key in frame for key in ("at", "x", "y", "w", "h"))
        ]
        if keyframes:
            paths[str(clip_id)] = CropPath(keyframes)
    return paths


def _write_grounding_record(path: Path, value: Any) -> None:
    """Publish one replayable grounding record without exposing half JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _set_target_grounding(
    report: "Report", clip_id: str, target_id: str, value: dict[str, Any]
) -> dict[str, Any]:
    """Store one target's state without erasing its neighbours.

    The top-level projection remains for old reports and the existing
    single-target UI.  ``targets`` is the durable authority when a shot names
    more than one locked entity.
    """

    current = report.reference_grounding.get(clip_id)
    if not isinstance(current, dict):
        current = {}
    targets = current.get("targets")
    if not isinstance(targets, dict):
        targets = {}
        previous_target = str(current.get("target_id") or "")
        if previous_target:
            previous = {
                key: item for key, item in current.items() if key != "targets"
            }
            targets[previous_target] = previous
    record = {"target_id": target_id, **value}
    targets[target_id] = record
    # Compatibility projection: the latest target is still readable by old
    # single-target consumers, while target-aware consumers never lose data.
    current = {**record, "targets": targets}
    report.reference_grounding[clip_id] = current
    return record


def _target_grounding(
    report: "Report", clip_id: str, target_id: str
) -> dict[str, Any]:
    current = report.reference_grounding.get(clip_id, {})
    targets = current.get("targets") if isinstance(current, dict) else None
    if isinstance(targets, dict) and isinstance(targets.get(target_id), dict):
        return targets[target_id]
    return _set_target_grounding(report, clip_id, target_id, {})


def _preflight_sam_checkpoint(checkpoint: Path | None) -> Path:
    """Reject unavailable local geometry before any paid semantic call."""

    if checkpoint is None:
        raise FileNotFoundError("SAM checkpoint is not configured")
    resolved = Path(checkpoint).expanduser().resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file() or stat.st_size <= 0:
        raise FileNotFoundError(f"SAM checkpoint is not a non-empty file: {resolved}")
    # Importing dependencies is cheap and catches a broken local environment
    # before exact-frame grounding is purchased. Predictor construction stays
    # in the tracking call because loading the large checkpoint for every shot
    # would turn a preflight into the dominant local cost.
    signature = (stat.st_mtime_ns, stat.st_size)
    if _SAM_PREFLIGHTED.get(resolved) == signature:
        return resolved
    from montagewright.measure.sam_tracking import _require_segmentation_dependencies

    _np, torch, _builder = _require_segmentation_dependencies()
    try:
        payload = torch.load(
            str(resolved), map_location="cpu", weights_only=True, mmap=True
        )
    except Exception as error:  # noqa: BLE001 -- local health gate
        raise RuntimeError(
            f"SAM checkpoint cannot be read safely: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, Mapping) or not payload:
        raise RuntimeError("SAM checkpoint contains no model state")
    del payload
    _SAM_PREFLIGHTED[resolved] = signature
    return resolved


def _confirmed_target_frames(
    confirmed_identities: Mapping[str, Any] | None,
    source_id: str,
    target_id: str,
) -> tuple[Any, ...]:
    """Resolve target-keyed confirmations and refuse legacy cross-target use."""

    if not confirmed_identities:
        return ()
    source_confirmed = confirmed_identities.get(source_id)
    if isinstance(source_confirmed, Mapping):
        frames = source_confirmed.get(target_id, ())
    else:
        # Transitional support for source-only callers: newly written and
        # migrated cached frames carry target_id, so filtering makes the old
        # container fail closed instead of routing A's box into B's tracker.
        frames = source_confirmed or ()
    return tuple(
        one for one in frames
        if getattr(one, "target_id", None) == target_id
    )


FINAL_EXACT_LOCAL_VALIDATOR_VERSION = (
    "pipeline-final-window-exact-v2:lineage+two-anchors+sam-seeds"
)


def _final_exact_cache_key(
    spec: Any,
    target_id: str,
    prepared: list[tuple[Any, str]],
    *,
    discovery: Any | None = None,
    model_id: str | None = None,
    reference_resolution: str = "high",
    frame_resolution: str = "high",
    minimum_matched_anchors: int = 2,
    local_validator_version: str = FINAL_EXACT_LOCAL_VALIDATOR_VERSION,
) -> str:
    """Name a cached answer by the complete semantic request contract."""

    from montagewright.reference_grounding import (
        EXACT_OUTPUT_POLICY_VERSION,
        MODEL_ID,
        _exact_frame_batch_schema,
        _read_prompt,
    )

    candidate_ids = tuple(
        dict.fromkeys(candidate_id for _, candidate_id in prepared)
    )
    schema = _exact_frame_batch_schema(
        target_id, candidate_ids, len(prepared)
    )
    contract = {
        "cache_contract_version": "final-window-exact-cache-v2",
        "grounding_spec_sha256": spec.definition_sha256(),
        "target_id": target_id,
        "frame_requests": [
            {
                "lineage": (
                    frame.lineage.model_dump(mode="json", exclude_none=True)
                    if hasattr(frame.lineage, "model_dump")
                    else {
                        "frame_pts": frame.lineage.frame_pts,
                        "frame_sha256": frame.lineage.frame_sha256,
                    }
                ),
                "candidate": (
                    discovery.candidate(candidate_id).model_dump(
                        mode="json", exclude_none=True
                    )
                    if discovery is not None else {"candidate_id": candidate_id}
                ),
            }
            for frame, candidate_id in prepared
        ],
        "model_id": model_id or MODEL_ID,
        "prompt_sha256": hashlib.sha256(_read_prompt().encode("utf-8")).hexdigest(),
        "response_schema_sha256": hashlib.sha256(json.dumps(
            schema, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "local_validator_version": local_validator_version,
        "reference_resolution": reference_resolution,
        "frame_resolution": frame_resolution,
        "minimum_matched_anchors": minimum_matched_anchors,
        "max_frames_per_call": 6,
        "generation": {
            "thinking_level": "low",
            "max_output_policy": EXACT_OUTPUT_POLICY_VERSION,
        },
    }
    return hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _read_final_exact_cache(
    path: Path, contract_key: str, result_type: Any
) -> Any | None:
    """Read only the v2 wrapper; raw/legacy batches are intentionally stale."""

    try:
        cached = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(cached, dict) or set(cached) != {
            "cache_contract_sha256", "batch",
        }:
            return None
        if cached["cache_contract_sha256"] != contract_key:
            return None
        return result_type.model_validate(cached["batch"])
    except (OSError, TypeError, ValueError):
        return None


def _write_final_exact_cache(path: Path, contract_key: str, batch: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_grounding_record(destination, {
        "cache_contract_sha256": contract_key,
        "batch": batch.model_dump(mode="json"),
    })


def _reaches(one: Any, opens: float, closes: float) -> bool:
    """Whether a confirmed frame speaks for this cut.

    Within reach in time, and belonging to a sighting that the cut sits in:
    the two together are what make it safe to carry a box forward.
    """

    if not (
        opens - CONFIRMED_REACH_SECONDS
        <= one.at_seconds
        <= closes + CONFIRMED_REACH_SECONDS
    ):
        return False
    window = getattr(one, "sighting_window", None)
    if window is None:
        return True
    starts, ends = window
    return starts - 1e-6 <= opens and closes <= ends + 1e-6


def _single_seed_eligibility(
    one: Any, opens: float, closes: float,
) -> tuple[bool, tuple[str, ...]]:
    """Whether this exact confirmation may start the cheap continuity path."""

    risks: list[str] = list(
        tuple(getattr(one, "seed_risk_flags", ()) or ())
    )
    if not all(getattr(one, field, None) is not None for field in (
        "video_asset_id", "frame_time_ms", "width", "height",
    )):
        risks.append("exact_lineage_unavailable")
    track_span = max(closes, one.at_seconds + 0.1) - max(
        0.0, min(opens, one.at_seconds)
    )
    if track_span > ADAPTIVE_TRACK_MAX_SECONDS + 1e-6:
        risks.append("unchecked_span_too_long")
    return not risks, tuple(dict.fromkeys(risks))


def _geometry_from_confirmed(
    source: Source,
    clip: Any,
    target_id: str,
    confirmed: "list[Any]",
    inside: "list[Any]",
    *,
    report: Report,
    work: Path,
    checkpoint: Path,
    validation_mode: str = "multi_anchor",
) -> tuple[
    list[dict[str, Any]],
    list[float],
    tuple[tuple[float, tuple[float, float, float, float]], ...],
]:
    """Track from a frame whose identity is already settled.

    The seed is the confirmed frame closest to this cut -- preferring one
    inside it -- and the analysed range stretches to reach it, so a box
    proved during the close-up that opens a take can be carried into the
    seconds an edit actually wants. What the tracker produces is checked
    against every confirmed box as before; the coverage it has to earn is
    measured over the cut, not over the reach, or a shot could ride in on a
    pre-roll it does not use.
    """

    if validation_mode == "multi_anchor_fallback" and len(confirmed) < 2:
        raise ReferenceGeometryUnavailable(
            clip.clip_id, target_id,
            f"{clip.clip_id}: multi-anchor fallback requires two confirmed "
            f"frames; got {len(confirmed)}",
        )

    seed = min(
        inside or confirmed,
        key=lambda one: abs(
            one.at_seconds
            - (clip.approx_in_seconds + clip.approx_out_seconds) / 2
        ),
    )
    reach_in = max(0.0, min(clip.approx_in_seconds, seed.at_seconds))
    reach_out = max(clip.approx_out_seconds, seed.at_seconds + 0.1)
    widened = clip.model_copy(update={
        "approx_in_seconds": reach_in, "approx_out_seconds": reach_out,
    })
    exact_lineage = None
    lineage_values = (
        getattr(seed, "video_asset_id", None),
        getattr(seed, "frame_time_ms", None),
        getattr(seed, "width", None),
        getattr(seed, "height", None),
    )
    if all(value is not None for value in lineage_values):
        from types import SimpleNamespace

        exact_lineage = SimpleNamespace(
            video_asset_id=seed.video_asset_id,
            frame_pts=seed.frame_pts,
            frame_time_ms=seed.frame_time_ms,
            frame_sha256=seed.frame_sha256,
            width=seed.width,
            height=seed.height,
        )
    tracked, states = _track_subject(
        source, widened, f"the locked identity {target_id}",
        [int(value * 1000) for value in seed.box],
        checkpoint, work,
        seed_time_seconds=seed.at_seconds,
        track_name=f"confirmed-{target_id.replace(':', '_')}",
        semantic_anchors=tuple((one.at_seconds, one.box) for one in confirmed),
        require_identity_validation=True,
        seed_lineage=exact_lineage,
    )
    boxes: list[dict[str, Any]] = []
    times: list[float] = []
    for observation in tracked:
        at = reach_in + observation.seconds
        if not (
            clip.approx_in_seconds - 1e-6 <= at <= clip.approx_out_seconds + 1e-6
        ):
            # The reach exists to find a seed, not to lengthen the shot.
            continue
        # The index the crop builders look this observation's moment up by.
        # Without it every consumer but the held-frame branch drops the box
        # and reports that the subject was never located.
        boxes.append({
            "frame_index": len(times),
            "present": True,
            "centre_x": observation.centre_x,
            "centre_y": observation.centre_y,
            "width": observation.width,
            "height": observation.height,
            "disambiguation": (
                f"identity confirmed at {seed.at_seconds:.2f}s"
            ),
            "geometry_source": "sam2.1",
        })
        times.append(at)

    # Over the cut. `states` counts every sample across the reach, so a long
    # pre-roll the tracker held would otherwise pay for seconds it lost.
    analysed = max(
        1,
        round(
            (clip.approx_out_seconds - clip.approx_in_seconds) * TRACK_FPS
        ),
    )
    kept = len(boxes)
    continuity_risks = sorted(
        name.split(":", 1)[1]
        for name in states
        if name.startswith("_continuity_risk:")
    )
    segment_id = hashlib.sha256(
        json.dumps([
            source.source_id, target_id, getattr(seed, "sighting", ""),
            seed.frame_pts,
        ], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    grounding_record = _target_grounding(report, clip.clip_id, target_id)
    grounding_record.update({
        "matched_anchors": len(confirmed),
        "tracked_frames": kept,
        "analysed_frames": analysed,
        "sam_seed_seconds": round(seed.at_seconds, 3),
        "sam_seed_pts": seed.frame_pts,
        "sam_seed_sha256": seed.frame_sha256,
        "identity_by_continuity": bool(states.get("_identity_by_continuity")),
        "validation_mode": validation_mode,
        "continuity_segment_id": segment_id,
        "risk_flags": continuity_risks,
        "checkpoint_count": len(confirmed),
        "paid_checkpoint_count": 0,
    })
    if kept < TRACK_MINIMUM_OBSERVATIONS or kept / analysed < TRACK_QUORUM:
        grounding_record["status"] = (
            "local_geometry_unverified"
        )
        _set_target_grounding(report, clip.clip_id, target_id, grounding_record)
        raise ReferenceGeometryUnavailable(
            clip.clip_id, target_id,
            f"{clip.clip_id}: tracking from the confirmed frame at "
            f"{seed.at_seconds:.2f}s covered {kept}/{analysed} of the cut; "
            + (
                f"continuity risks: {', '.join(continuity_risks)}; "
                if continuity_risks else ""
            )
            + "refusing Gemini-box fallback",
        )
    grounding_record["status"] = "sam_geometry_validated"
    # Keep the compatibility projection in sync after mutating the durable
    # per-target record in place.
    _set_target_grounding(report, clip.clip_id, target_id, grounding_record)
    return (
        boxes,
        times,
        tuple((one.at_seconds, one.box) for one in confirmed),
    )


def _reference_subject_samples(
    source: Source,
    clip: Any,
    target_id: str,
    *,
    spec: Any,
    client: Any,
    upload_cache: Any | None,
    report: Report,
    work: Path,
    output: Path | None,
    discoveries: dict[str, Any],
    checkpoint: Path | None,
    memory: Path | None = None,
    confirmed: "tuple[Any, ...] | None" = None,
) -> tuple[
    list[dict[str, Any]],
    list[float],
    tuple[tuple[float, tuple[float, float, float, float]], ...],
]:
    """Reference-confirm one identity on exact source frames.

    Video grounding supplies coarse candidate intervals only.  Local decoding
    turns those intervals into immutable source PTS frames; a second,
    reference-conditioned decision must match the locked identity on at least
    two distinct PTS values before any box is exposed to SAM or reframing.
    """

    from montagewright.reference_grounding import (
        ReferenceGroundingError,
        CandidateDiscoveryResult,
        ExactFrameBBoxBatchResult,
        decide_exact_frame_bboxes,
        discover_reference_candidates,
        inspect_video_lineage,
        materialize_frame_at_time,
    )

    try:
        checkpoint = _preflight_sam_checkpoint(checkpoint)
    except (FileNotFoundError, ImportError, RuntimeError) as error:
        _set_target_grounding(report, clip.clip_id, target_id, {
            "status": "local_geometry_unavailable",
            "reason": str(error)[:200],
        })
        raise RuntimeError(
            f"{clip.clip_id}: reference-critical target {target_id} requires "
            "SAM/local geometry; Gemini boxes are semantic seeds, not crop geometry"
        ) from error
    clip_start_ms = round(float(clip.approx_in_seconds) * 1000)
    clip_end_ms = round(float(clip.approx_out_seconds) * 1000)
    # Identity was settled for this source, at the moments it was clearest,
    # before anything chose which seconds to cut. Use it: seed the tracker
    # from the confirmed frame nearest this cut and let SAM carry the box
    # in, rather than asking again inside seconds that may show nothing
    # identifiable -- an edge beside a coin, or three handsets at a distance.
    #
    # Only from inside the same sighting, though. A box proved while the
    # subject was in shot says nothing after it left and came back, and the
    # tracker cannot cross that gap either.
    confirmed = tuple(
        one for one in (confirmed or ())
        if getattr(one, "target_id", None) == target_id
    )
    if confirmed:
        # Only from inside a sighting that this cut is part of. A box proved
        # while the subject was in shot says nothing after it left and came
        # back -- and the tracker cannot cross that gap either, so reaching
        # across one would seed the cut from whatever the mask drifted onto.
        near = [
            one for one in confirmed
            if _reaches(one, clip_start_ms / 1000.0, clip_end_ms / 1000.0)
        ]
        inside = [
            one for one in near
            if clip_start_ms / 1000.0 <= one.at_seconds <= clip_end_ms / 1000.0
        ]
        # Seeding from outside the cut on a single confirmation is not a
        # check at all: SAM is prompted with that exact box at that exact
        # time, so it agrees with itself, and nothing then speaks for the
        # seconds the cut actually uses. Two confirmations, or one inside.
        usable = near if (inside or len(near) >= 2) else []
        if usable:
            # Try one exact, low-risk seed before requiring a second semantic
            # checkpoint. This is safe only while the seed and every second
            # being delivered fit inside one short continuous tracking span.
            # The strict SAM continuity audit is the second half of this
            # proof; on any doubt we fall through to the old multi-anchor
            # route rather than blessing the track.
            centre = (clip_start_ms + clip_end_ms) / 2000.0
            eligibility = {
                id(one): _single_seed_eligibility(
                    one, clip_start_ms / 1000.0, clip_end_ms / 1000.0
                )
                for one in usable
            }
            single_candidates = [
                one for one in usable if eligibility[id(one)][0]
            ]
            if single_candidates:
                single = min(
                    single_candidates, key=lambda one: abs(one.at_seconds - centre)
                )
                grounding_record = _set_target_grounding(
                    report, clip.clip_id, target_id, {
                    "status": "identity_from_source",
                    "confirmed_at": [round(single.at_seconds, 3)],
                    "seeded_inside_cut": single in inside,
                    "validation_mode": "single_seed_continuity",
                })
                try:
                    return _geometry_from_confirmed(
                        source, clip, target_id, [single],
                        [single] if single in inside else [],
                        report=report, work=work, checkpoint=checkpoint,
                        validation_mode="single_seed_continuity",
                    )
                except ReferenceShotUnusable as unusable:
                    grounding_record["fallback_reason"] = str(unusable)[:200]
                    _set_target_grounding(
                        report, clip.clip_id, target_id, grounding_record
                    )
                    report.subject_notes[clip.clip_id] = str(unusable)[:200]
                except Exception as error:  # noqa: BLE001 -- fallback is safe
                    reason = (
                        f"single-seed tracking failed: "
                        f"{type(error).__name__}: {error}"
                    )[:200]
                    grounding_record["fallback_reason"] = reason
                    _set_target_grounding(
                        report, clip.clip_id, target_id, grounding_record
                    )
                    report.subject_notes[clip.clip_id] = reason

            fallback_reason = report.subject_notes.get(clip.clip_id)
            if not single_candidates:
                risks = sorted({
                    risk for one in usable for risk in eligibility[id(one)][1]
                })
                fallback_reason = (
                    "single seed ineligible"
                    + (f": {', '.join(risks)}" if risks else "")
                )
            _set_target_grounding(report, clip.clip_id, target_id, {
                "status": "identity_from_source",
                "confirmed_at": [round(one.at_seconds, 3) for one in usable],
                "seeded_inside_cut": bool(inside),
                "validation_mode": "multi_anchor_fallback",
                "fallback_reason": fallback_reason,
            })
            try:
                return _geometry_from_confirmed(
                    source, clip, target_id, usable, inside,
                    report=report, work=work, checkpoint=checkpoint,
                    validation_mode="multi_anchor_fallback",
                )
            except ReferenceShotUnusable as unusable:
                # Fall through and ask inside the cut's own window, which is
                # what this did before there was anything to fall back from.
                report.subject_notes[clip.clip_id] = str(unusable)[:200]
            except Exception as error:  # noqa: BLE001 -- reported, not fatal
                report.subject_notes[clip.clip_id] = (
                    f"tracking from the confirmed frame failed: "
                    f"{type(error).__name__}: {error}"[:200]
                )

    # The window this shot uses IS the question, so ask it directly.
    #
    # This used to pay for a second discovery, on the master, over a source
    # the material screen had already judged on its proxy -- the same
    # question, the same model, two different encodings of the same seconds,
    # and no reconciliation between the answers. Measured on this material
    # the two disagreed on three of eight shots: the screen said the Fold8
    # was there, the master pass said absent, no exact frame was ever
    # decoded, and the shot died reporting that its identity "could not be
    # confirmed on two exact source frames" -- a sentence about a judgement
    # nobody had made.
    #
    # The contract forbids pairing a proxy discovery with master frames, and
    # rightly: candidate milliseconds belong to the video they were measured
    # on. So the interval is built locally from the cut instead. It claims
    # nothing about identity -- `uncertain` is exactly what is known before
    # the frames are judged -- and the judgement comes where it always came
    # from, the exact frames themselves.
    video = inspect_video_lineage(source.path)
    window_end = min(clip_end_ms, int(video.duration_ms))
    if window_end <= clip_start_ms:
        report.subject_notes[clip.clip_id] = (
            "the cut starts at or after the end of the take; nothing was judged"
        )
        return [], [], ()
    discovery = CandidateDiscoveryResult.model_validate({
        "contract_version": "reference-candidate-discovery-v1",
        "query_id": spec.identity_lock.query_id,
        "query_lock_sha256": spec.identity_lock.definition_sha256(),
        "grounding_spec_sha256": spec.definition_sha256(),
        "video_asset_id": video.asset_id,
        "video_sha256": video.content_sha256,
        "duration_ms": int(video.duration_ms),
        "candidates": [{
            "candidate_id": f"cut_{clip.clip_id}",
            "target_id": target_id,
            "start_ms": clip_start_ms,
            "end_ms": window_end,
            "recommended_seed_ms": clip_start_ms + (window_end - clip_start_ms) // 2,
            "identity_status": "uncertain",
            "confidence": 0.5,
            "visible_state": "unjudged; this interval is the cut, not a sighting",
            "visibility_state": "unknown",
            "occlusion_state": "unknown",
            "identity_evidence": [],
            "exclusion_evidence": [],
        }],
        "target_summaries": [{
            "target_id": target_id,
            "verdict": "uncertain",
            "reason": (
                "the material screen placed this identity in this source; "
                "these frames decide whether it is in this cut"
            ),
        }],
        "warnings": [],
    })
    discoveries[source.source_id] = discovery
    eligible = list(discovery.candidates)

    # Spread exact checkpoints over every overlapping candidate interval.
    # Reusing one decoded frame twice would satisfy a count while proving
    # nothing about drift, so frame PTS are deduplicated below.
    requested: list[tuple[int, str]] = []
    for candidate in eligible:
        start = max(candidate.start_ms, clip_start_ms)
        end = min(candidate.end_ms, clip_end_ms)
        if end <= start:
            continue
        span = end - start
        points = [
            max(start, min(end - 1, candidate.recommended_seed_ms)),
            start + max(0, span // 5),
            start + max(0, (span * 4) // 5),
        ]
        for at in points:
            requested.append((at, candidate.candidate_id))

    prepared: list[tuple[Any, str]] = []
    used_pts: set[int] = set()
    for requested_ms, candidate_id in sorted(requested):
        candidate = discovery.candidate(candidate_id)
        destination = work / (
            f"reference-{clip.clip_id}-{target_id.replace(':', '_')}-"
            f"{len(prepared):02d}.jpg"
        )
        frame = materialize_frame_at_time(
            source.path, requested_ms, destination, max_width=1440
        )
        if frame.lineage.frame_pts in used_pts:
            destination.unlink(missing_ok=True)
            continue
        if not candidate.start_ms <= frame.lineage.frame_time_ms < candidate.end_ms:
            continue
        if not clip_start_ms <= frame.lineage.frame_time_ms < clip_end_ms:
            continue
        used_pts.add(frame.lineage.frame_pts)
        prepared.append((frame, candidate_id))
        if len(prepared) >= 4:
            break
    if len(prepared) < 2:
        # Not the same thing as a refused identity, and it read as one for a
        # whole evening: the caller's message says the identity "could not be
        # confirmed on two exact source frames", which describes a judgement
        # that in this case was never asked for. Say which it was.
        report.subject_notes[clip.clip_id] = (
            f"only {len(prepared)} distinct frame(s) could be decoded inside "
            f"{clip_start_ms}-{window_end}ms; nothing was judged"
        )
        return [], [], ()

    # What this call answers is a fact about specific decoded frames under
    # one identity lock, and the frames are content-hashed on the way in --
    # so the question has a name. It was written down and never read back,
    # so every repair attempt re-paid for every shot it had already judged,
    # including the shots it was not repairing.
    batch = None
    remembered = None
    cache_contract_key = None
    if memory is not None:
        cache_contract_key = _final_exact_cache_key(
            spec, target_id, prepared,
            discovery=discovery,
            reference_resolution="high",
            frame_resolution="high",
            minimum_matched_anchors=2,
        )
        remembered = Path(memory) / f"exact-v2-{cache_contract_key[:24]}.json"
        if remembered.exists():
            batch = _read_final_exact_cache(
                remembered, cache_contract_key, ExactFrameBBoxBatchResult
            )

    if batch is None and not _may_ask(client):
        report.subject_notes[clip.clip_id] = (
            "no client was available and no exact-frame cache matched this window"
        )
        return [], [], ()
    if batch is None:
        _afford(report)
        try:
            decided = decide_exact_frame_bboxes(
                spec,
                discovery,
                target_id,
                [frame for frame, _ in prepared],
                candidate_ids=[candidate_id for _, candidate_id in prepared],
                client=client,
                cache=upload_cache,
                ledger=report.ledger,
                minimum_matched_anchors=2,
            )
        except ReferenceGroundingError as error:
            report.subject_notes[clip.clip_id] = (
                f"reference identity could not be verified: {error}"[:160]
            )
            return [], [], ()
        if decided is None:
            report.subject_notes[clip.clip_id] = (
                "no client was available to judge the exact frames"
            )
            return [], [], ()
        batch, usage = decided
        _charge(report, "reference_exact", usage)
        if remembered is not None:
            _write_final_exact_cache(
                remembered, str(cache_contract_key), batch
            )
    if output is not None:
        _write_grounding_record(
            output / (
                f"{clip.clip_id}-{target_id.replace(':', '_')}-exact.json"
            ),
            batch.model_dump(mode="json"),
        )
    try:
        matched = batch.sam_seed_evaluations()
    except ReferenceGroundingError as error:
        report.subject_notes[clip.clip_id] = str(error)[:160]
        _set_target_grounding(report, clip.clip_id, target_id, {
            "status": "identity_unverified",
            "matched_anchors": batch.matched_anchor_count,
        })
        return [], [], ()

    anchors: list[
        tuple[float, tuple[float, float, float, float]]
    ] = []
    for evaluation in matched:
        native = evaluation.decision.tracking_box_xyxy_1000
        if native is None:
            continue
        x0, y0, x1, y1 = (float(value) / 1000.0 for value in native)
        at = evaluation.lineage.frame_time_ms / 1000.0
        anchors.append((at, (x0, y0, x1, y1)))
    if len(anchors) < 2:
        raise ReferenceGeometryUnavailable(
            clip.clip_id, target_id,
            f"{clip.clip_id}: reference-critical target {target_id} has fewer "
            "than two usable exact semantic anchors",
        )

    seed = max(
        matched,
        key=lambda evaluation: float(evaluation.decision.confidence),
    )
    seed_box = seed.decision.tracking_box_xyxy_1000
    if seed_box is None:
        raise RuntimeError("matched exact-frame seed has no semantic bbox")
    try:
        identity = spec.identity_lock.identity.target(target_id)
        target_description = identity.target_description
    except (AttributeError, ValueError):
        target_description = target_id
    try:
        tracked, states = _track_subject(
            source,
            clip,
            target_description,
            list(seed_box),
            checkpoint,
            work,
            seed_time_seconds=seed.lineage.frame_time_ms / 1000.0,
            track_name=f"reference-{target_id.replace(':', '_')}",
            semantic_anchors=tuple(anchors),
            require_identity_validation=True,
            seed_lineage=seed.lineage,
        )
    except Exception as error:
        _set_target_grounding(report, clip.clip_id, target_id, {
            "status": "local_geometry_failed",
            "reason": type(error).__name__,
        })
        raise ReferenceGeometryUnavailable(
            clip.clip_id, target_id,
            f"{clip.clip_id}: SAM/local geometry failed for locked reference "
            f"identity {target_id}; refusing Gemini-box crop",
        ) from error
    total = sum(
        count for name, count in states.items() if not name.startswith("_")
    ) or 1
    # Only locally materialized observations can become crop geometry. A
    # sample labelled tracked but lacking a valid mask-derived box is not a
    # successful handoff for a reference-critical target.
    kept = len(tracked)
    if kept < TRACK_MINIMUM_OBSERVATIONS or kept / total < TRACK_QUORUM:
        _set_target_grounding(report, clip.clip_id, target_id, {
            "status": "local_geometry_unverified",
            "tracked_frames": kept,
            "analysed_frames": total,
            "anchors_agreed": states.get("_anchors_agreed"),
            "anchors_offered": states.get("_anchors_offered"),
            "best_agreement_pct": states.get("_best_agreement_pct"),
        })
        if states.get("identity_unverified"):
            raise ReferenceGeometryUnavailable(
                clip.clip_id, target_id,
                f"{clip.clip_id}: the track for {target_id} was never tied to "
                f"the exact frames that proved it -- "
                f"{states.get('_anchors_agreed', 0)} of "
                f"{states.get('_anchors_offered', 0)} anchors agreed with the "
                f"mask (best overlap "
                f"{states.get('_best_agreement_pct', 0)}%, needs 35% on two); "
                "refusing Gemini-box fallback",
            )
        raise ReferenceGeometryUnavailable(
            clip.clip_id, target_id,
            f"{clip.clip_id}: SAM/local geometry for locked reference identity "
            f"{target_id} passed only {kept}/{total} frames; refusing "
            "Gemini-box fallback",
        )

    boxes: list[dict[str, Any]] = []
    times: list[float] = []
    disambiguation = "; ".join(seed.decision.identity_evidence)
    for frame_index, observation in enumerate(tracked):
        times.append(clip.approx_in_seconds + observation.seconds)
        boxes.append({
            "frame_index": frame_index,
            "present": True,
            "centre_x": observation.centre_x,
            "centre_y": observation.centre_y,
            "width": observation.width,
            "height": observation.height,
            "disambiguation": disambiguation,
            "geometry_source": "sam2.1",
        })
    from montagewright.measure.geometry import native_yxyx_to_canonical_xyxy

    grounding_record = _set_target_grounding(
        report, clip.clip_id, target_id, {
        "status": "sam_geometry_validated",
        "query_lock_sha256": batch.query_lock_sha256,
        "grounding_spec_sha256": batch.grounding_spec_sha256,
        "matched_anchors": len(anchors),
        "source_pts": [item.lineage.frame_pts for item in matched],
        "sam_seed_pts": seed.lineage.frame_pts,
        "sam_seed_sha256": seed.lineage.frame_sha256,
        "sam_seed_width": seed.lineage.width,
        "sam_seed_height": seed.lineage.height,
        "tracked_frames": kept,
        "analysed_frames": total,
        # Where the lookalikes were, in the frames that proved the target.
        # A shot with both devices in it is not automatically unusable -- a
        # 9:16 crop out of 16:9 keeps about a third of the width and can
        # often leave the other one outside the frame -- but nothing could
        # even ask that question while their position was never reported.
        # A lookalike standing in the same frame is a fact the plan needs to
        # answer for: on the run this was written from, selection named "the
        # left unfolded smartphone" and the exact frames judged the left one
        # to be the excluded model, so the crop went right -- correctly, and
        # invisibly. The record existed and nothing read it.
        "excluded_instances": [
            {
                "at_seconds": round(
                    item.lineage.frame_time_ms / 1000.0, 3
                ),
                "box_xyxy_1000": list(
                    native_yxyx_to_canonical_xyxy(
                        instance.native_box_yxyx_1000
                    )
                ),
                "reason": instance.reason,
            }
            for item in matched
            for instance in item.decision.excluded_instances
        ],
    })
    lookalikes = grounding_record["excluded_instances"]
    if lookalikes:
        report.plan_disagreements.append(
            f"{clip.clip_id} shares the frame with "
            f"{len({one['reason'] for one in lookalikes})} instance(s) the "
            f"lock excludes ({lookalikes[0]['reason'][:90]}); the crop "
            "follows the confirmed target, which may not be the one the "
            "plan described"
        )
    return boxes, times, tuple(anchors)


def follow_subjects(
    edl: EDL,
    sources: dict[str, Source],
    *,
    target_aspect: float,
    report: Report,
    output_size: tuple[int, int] | None = None,
    cards: dict[str, Path] | None = None,
    checkpoint: Path | None = None,
    client: Any | None = None,
    grounding_spec: Any | None = None,
    grounding_output: Path | None = None,
    grounding_memory: Path | None = None,
    confirmed_identities: "dict[str, Any] | None" = None,
    upload_cache: Any | None = None,
) -> dict[str, CropPath]:
    """Build a crop path per shot that names a subject.

    A shot whose subject turns out to be still gets a held frame, which is the
    right answer rather than a failure -- the deadband decides that, not a
    threshold anyone has to tune.
    """

    # The size the film is actually delivered at, so the zoom budget guards
    # the enlargement that really happens. It used to assume 1080x1920 while
    # the renderer scaled each segment to whatever the opening crop measured
    # -- 1214x2160 off a 4K source -- so a push reported as 1.35x was 1.52x
    # on disk, and the limit was protecting an output that did not exist.
    output_size = output_size or delivery_size(target_aspect)

    paths: dict[str, CropPath] = {}
    discoveries: dict[str, Any] = {}
    unusable_shots: list[ReferenceShotUnusable] = []
    with tempfile.TemporaryDirectory() as raw_work:
        work = Path(raw_work)
        total = len(edl.clips)
        for index, clip in enumerate(edl.clips, start=1):
            reframe = clip.reframe
            if reframe is None:
                continue
            try:
                # This stage is a grounding call and sometimes a SAM propagation
                # per shot, and the propagation writes a progress bar with
                # carriage returns that never reaches a log. Minutes could pass
                # with the last line still being about the music.
                print(
                    f"  subject {index}/{total}  {clip.clip_id}  "
                    f"{reframe.camera_move}",
                    flush=True,
                )
                source = sources[clip.source_id]
                move = reframe.camera_move
                reference_samples: dict[
                    str,
                    tuple[
                        list[dict[str, Any]],
                        list[float],
                        tuple[
                            tuple[float, tuple[float, float, float, float]], ...
                        ],
                    ],
                ] = {}
                if grounding_spec is not None:
                    entity_faults: list[ReferenceShotUnusable] = []
                    for entity_id in dict.fromkeys(
                        look.entity_id
                        for look in reframe.looks
                        if look.entity_id
                    ):
                        try:
                            samples = _reference_subject_samples(
                                source,
                                clip,
                                entity_id,
                                spec=grounding_spec,
                                client=client,
                                upload_cache=upload_cache,
                                report=report,
                                work=work,
                                output=grounding_output,
                                discoveries=discoveries,
                                checkpoint=checkpoint,
                                memory=grounding_memory,
                                confirmed=_confirmed_target_frames(
                                    confirmed_identities,
                                    clip.source_id,
                                    entity_id,
                                ),
                            )
                        except ReferenceShotUnusable as unusable:
                            entity_faults.append(unusable)
                            continue
                        if samples[0]:
                            reference_samples[entity_id] = samples
                        else:
                            report.degradations.append(
                                DegradationStep(
                                    clip_id=clip.clip_id,
                                    ladder="center_crop",
                                    trigger=(
                                        "the locked reference identity "
                                        f"{entity_id} was not confirmed on two "
                                        "exact source frames; refusing text-only "
                                        "or SAM lookalike substitution"
                                    ),
                                    measured={"confirmed_anchors": 0.0},
                                )
                            )
                            entity_faults.append(ReferenceIdentityUnconfirmed(
                                clip.clip_id, entity_id,
                                f"{clip.clip_id}: locked reference identity "
                                f"{entity_id} could not be delivered from "
                                "this cut ("
                                + (
                                    report.subject_notes.get(clip.clip_id)
                                    or "the exact frames did not confirm it"
                                )
                                + "); reselect the shot instead of "
                                "substituting a lookalike",
                            ))
                    if entity_faults:
                        unusable_shots.extend(entity_faults)
                        continue
                card = (
                    load_card(cards[clip.source_id])
                    if cards and clip.source_id in cards
                    else None
                )

                duration = clip.approx_out_seconds - clip.approx_in_seconds

                # No card means no measured subject, and the reframe below will
                # quietly centre the crop. That is a defensible last resort and
                # an indefensible silence: a whole timeline was rebuilt this way,
                # every crop identical and dead centre, and nothing anywhere said
                # the subject had gone missing.
                if (
                    reframe.subject is not None
                    and card is None
                    and not reframe.subject.entity_id
                ):
                    report.degradations.append(
                        DegradationStep(
                            clip_id=clip.clip_id,
                            ladder="center_crop",
                            trigger=(
                                f"no card describes {clip.source_id}, so there is "
                                f"no measured position for "
                                f"\"{reframe.subject.description}\" and the crop "
                                f"can only be centred"
                            ),
                            measured={},
                        )
                    )

                # Whether a subject fits the delivery aspect is a fact about the
                # material, not a property of the move chosen for it. It used to
                # be checked inside the hold branch only, so the same wordmark
                # that a hold would have swept across was quietly cropped to
                # "Galaxy Unpac" the moment the planner asked for a push instead
                # -- and nothing recorded it, because only one of the five path
                # builders reports fit. The check belongs to the clip.
                known = (
                    find_subject(
                        card, reframe.subject.description,
                        entity_id=reframe.subject.entity_id,
                    )
                    if card is not None and reframe.subject is not None
                    else None
                )
                # Two moves stacked. The selection prompt has said since it was
                # written that a take which moves on its own does not want a
                # digital move on top -- the real one is real and the added one
                # is a crop sliding -- and until the span carried the role there
                # was nothing downstream holding the fact needed to check it.
                #
                # Reported rather than repaired: which of the two to give up is
                # editorial. Dropping the digital move would quietly change what
                # the shot shows, and dropping the shot would quietly change the
                # film.
                if (
                    reframe.planned_to_move
                    and reframe.source_motion_role in {"authored", "subject_follow"}
                ):
                    report.degradations.append(
                        DegradationStep(
                            clip_id=clip.clip_id,
                            ladder="other",
                            ladder_other="digital_move_on_a_moving_take",
                            trigger=(
                                "this take already moves on its own "
                                f"({reframe.source_motion_role}) and the plan "
                                "asks the frame to travel as well, so the two "
                                "movements are added together on screen"
                            ),
                            measured={"looks": float(len(reframe.looks))},
                        )
                    )

                # Every look, not only the first. `reframe.subject` is built from
                # `looks[0]`, so a shot that settles on a face and then on a
                # wordmark that has to be whole had the promise on the second one
                # read by nothing at all -- and `must_be_whole` moved onto the
                # look precisely because a shot can make different promises about
                # different parts of itself.
                if card is not None:
                    crop_width = target_aspect / source.aspect_ratio
                    for index, look in enumerate(reframe.looks[1:], start=1):
                        if not look.must_be_whole:
                            continue
                        box = find_subject(
                            card, look.at, entity_id=look.entity_id
                        )
                        if box is None or box.width <= crop_width:
                            continue
                        report.degradations.append(
                            DegradationStep(
                                clip_id=clip.clip_id,
                                ladder="other",
                                ladder_other="whole_subject_promised_without_a_move",
                                trigger=(
                                    f"look {index + 1} of this shot was declared "
                                    "whole and no crop of this source can hold "
                                    "it, so settling on it shows part of it"
                                ),
                                measured={
                                    "look": float(index + 1),
                                    "subject_width_vw": round(box.width, 4),
                                    "widest_crop_vw": round(crop_width, 4),
                                    "most_visible_fraction": round(
                                        crop_width / box.width, 4
                                    ),
                                },
                            )
                        )
                if reframe.subject is not None and card is not None:
                    crop_width = target_aspect / source.aspect_ratio
                    # A promise that the subject must be whole, on a subject no
                    # crop of this source can hold, with a move that does not
                    # travel across it. The three cannot all be true, and the
                    # planner was told the fraction when it made the promise --
                    # the wordmark it marked whole could only ever show 51%.
                    #
                    # Said here rather than left to be discovered in the output,
                    # because a contradiction between two fields of one plan is
                    # visible before anything is rendered, and the reviewer that
                    # would otherwise find it costs a paid call and a round.
                    # Not corrected: which of the three to give up is the
                    # planner's to choose.
                    if (
                        known is not None
                        and known.width > crop_width
                        and reframe.subject.min_visible >= 0.99
                        and reframe.camera_move not in {"pan", "tilt"}
                    ):
                        report.degradations.append(
                            DegradationStep(
                                clip_id=clip.clip_id,
                                ladder="other",
                                ladder_other="whole_subject_promised_without_a_move",
                                trigger=(
                                    "the subject was declared whole, no crop of "
                                    f"this source can hold it, and {reframe.camera_move} "
                                    "does not travel across it -- one of the three "
                                    "has to give"
                                ),
                                measured={
                                    "subject_width_vw": round(known.width, 4),
                                    "widest_crop_vw": round(crop_width, 4),
                                    "most_visible_fraction": round(
                                        crop_width / known.width, 4
                                    ),
                                },
                            )
                        )
                    if known is not None and known.width > crop_width:
                        report.degradations.append(
                            DegradationStep(
                                clip_id=clip.clip_id,
                                ladder="other",
                                ladder_other="subject_wider_than_delivery",
                                trigger=(
                                    f"the subject is wider than any crop of this "
                                    f"source at the delivery aspect, so a {move} "
                                    f"can only ever hold part of it"
                                ),
                                measured={
                                    "subject_width_vw": round(known.width, 4),
                                    "widest_crop_vw": round(crop_width, 4),
                                    "most_visible_fraction": round(
                                        crop_width / known.width, 4
                                    ),
                                    "requested_min_visible": (
                                        reframe.subject.min_visible
                                    ),
                                },
                            )
                        )

                # The two-subject pan used to be handled here, above the looks
                # branch below -- and `then_subject` is set exactly when there is
                # a second look, so it shadowed it completely. Every pan went to
                # the old builder and the third look of a row of three watches
                # was still being dropped, which is the whole thing the refactor
                # was for. Deleted rather than reordered: build_look_path does
                # what it did, for any number of stops.


                # Every shot that settles somewhere more than once, whatever the
                # move turned out to be called. One builder walks the list; the
                # older branches below stay for a single look, where following a
                # moving subject still needs the tracker.
                #
                # Three looks used to be silently truncated to two -- `pan` read
                # `subject` and `then_subject` and there was nowhere for a third
                # to go, so a row of three watches lost its middle stop with
                # nothing recorded.
                if len(reframe.looks) >= 2 and _may_ask(client):
                    stops, missing, tracks = _measure_looks(
                        reframe.looks, source, clip, work, report, client,
                        target_aspect, checkpoint, reference_samples,
                    )
                    if missing:
                        report.subject_notes[clip.clip_id] = (
                            f"could not find {missing} in any sampled frame"[:160]
                        )
                    if len(stops) >= 2:
                        out_w, out_h = output_size
                        paths[clip.clip_id] = build_look_path(
                            stops,
                            source_aspect=source.aspect_ratio,
                            target_aspect=target_aspect,
                            duration_seconds=duration,
                            energy=reframe.camera_energy,
                            clip_id=clip.clip_id,
                            degradations=report.degradations,
                            # Without these it can crop as tightly as a framing
                            # asks and the upscale is only discovered below, on
                            # a shot that has already been rendered soft.
                            source_width=source.width,
                            source_height=source.height,
                            output_width=out_w,
                            output_height=out_h,
                            # Where each subject went while the frame was looking
                            # at it. Without these a stop is a place, and a shot
                            # planned to follow somebody walking held on a point
                            # halfway along the walk.
                            tracks=tracks,
                        )
                        tightest = min(
                            paths[clip.clip_id].keyframes,
                            key=lambda one: one.crop.width,
                        ).crop
                        report.upscales[clip.clip_id] = achieved_upscale(
                            tightest,
                            source_width=source.width,
                            source_height=source.height,
                            output_width=out_w,
                            output_height=out_h,
                        )
                        report.following_shots += 1
                        continue

                if move in {"push_in", "pull_out"}:
                    # Push toward the subject, not toward the middle of the frame.
                    # Zooming on the geometric centre put a coin-against-a-hinge
                    # shot in the bottom third with 40% of the frame empty above
                    # it -- and a zoom is the only move that can change vertical
                    # framing at all when the crop is otherwise full height, so
                    # aiming it blindly wastes the one chance the shot has.
                    centre_x, centre_y = 0.5, 0.5
                    middles: list[tuple[float, float]] = []
                    boxes = []
                    sampled_at: list[float] = []
                    reference = (
                        reference_samples.get(reframe.subject.entity_id)
                        if reframe.subject is not None
                        and reframe.subject.entity_id else None
                    )
                    if reference is not None:
                        boxes, sampled_at, _ = reference
                    elif (
                        reframe.subject is not None
                        and reframe.subject.entity_id
                    ):
                        # A stable identity may never degrade to matching its
                        # prose description; that is exactly how a lookalike gets
                        # substituted confidently.
                        boxes, sampled_at = [], []
                    elif reframe.subject is not None and not _may_ask(client):
                        # A rebuild, with nothing to ask. Skipping the clip left
                        # no path at all, and the executor fell back to a centred
                        # still -- so a push read back as a hold on the middle of
                        # the frame, and the interface drew that as what had been
                        # rendered. A zoom needs one point to aim at, and the
                        # card has one.
                        if known is not None:
                            centre_x, centre_y = known.centre_x, known.centre_y
                            report.subject_notes[clip.clip_id] = (
                                f"{move} on ({centre_x:.3f}, {centre_y:.3f}) "
                                f"from the card"
                            )
                    elif reframe.subject is not None:
                        # The sample times were being discarded here. They are
                        # what turns five positions into a path: without them a
                        # push can only aim at their mean, and a subject that
                        # walks is squeezed out of the closing frame.
                        frames, sampled_at = _sample_frames(
                            source,
                            clip.approx_in_seconds,
                            clip.approx_out_seconds,
                            work,
                        )
                        _afford(report)
                        boxes, usage = _locate_subject(
                            frames, reframe.subject.description, client=client,
                            report=report,
                        )
                        _charge(report, "subject", usage)
                        middles = [
                            (float(b["centre_x"]), float(b["centre_y"]))
                            for b in boxes
                            if b.get("present") and b.get("centre_x") is not None
                        ]
                        if middles:
                            centre_x = sum(x for x, _ in middles) / len(middles)
                            centre_y = sum(y for _, y in middles) / len(middles)
                            report.subject_notes[clip.clip_id] = (
                                f"{move} on ({centre_x:.3f}, {centre_y:.3f})"
                            )
                    if boxes and not middles:
                        middles = [
                            (float(box["centre_x"]), float(box["centre_y"]))
                            for box in boxes
                            if box.get("present")
                            and box.get("centre_x") is not None
                        ]
                        if middles:
                            centre_x = sum(x for x, _ in middles) / len(middles)
                            centre_y = sum(y for _, y in middles) / len(middles)
                    # Read what the source can supply before choosing how far to
                    # push. A fixed percentage is blind to what it is cropping.
                    out_w, out_h = output_size
                    budget = zoom_budget(
                        source_width=source.width,
                        source_height=source.height,
                        source_aspect=source.aspect_ratio,
                        target_aspect=target_aspect,
                        output_width=out_w,
                        output_height=out_h,
                    )
                    subject_height = known.height if known is not None else None
                    if middles:
                        heights = [
                            float(b["height"])
                            for b in boxes
                            if b.get("present") and b.get("height") is not None
                        ]
                        if heights:
                            subject_height = sum(heights) / len(heights)
                    track = []
                    for box in boxes:
                        index = int(box.get("frame_index", -1))
                        if not box.get("present") or box.get("centre_x") is None:
                            continue
                        if not 0 <= index < len(sampled_at):
                            continue
                        track.append((
                            sampled_at[index] - clip.approx_in_seconds,
                            float(box["centre_x"]),
                            float(box["centre_y"]),
                        ))
                    path = build_zoom_path(
                        source_aspect=source.aspect_ratio,
                        target_aspect=target_aspect,
                        duration_seconds=duration,
                        direction=move,
                        centre_x=centre_x,
                        centre_y=centre_y,
                        track=track or None,
                        energy=reframe.camera_energy,
                        framing=reframe.framing,
                        budget=budget,
                        subject_height=subject_height,
                        clip_id=clip.clip_id,
                        degradations=report.degradations,
                    )
                    tightest = min(path.keyframes, key=lambda k: k.crop.width).crop
                    report.upscales[clip.clip_id] = achieved_upscale(
                        tightest,
                        source_width=source.width,
                        source_height=source.height,
                        output_width=out_w,
                        output_height=out_h,
                    )
                    paths[clip.clip_id] = path
                    report.following_shots += 1
                    continue

                if move in {"sweep_left", "sweep_right"}:  # legacy names
                    # A designed move across a still arrangement. Nothing is
                    # tracked because nothing is moving, so this costs no call.
                    paths[clip.clip_id] = build_sweep_path(
                        source_aspect=source.aspect_ratio,
                        target_aspect=target_aspect,
                        duration_seconds=duration,
                        direction=move,
                        energy=reframe.camera_energy,
                    )
                    report.following_shots += 1
                    continue

                if move == "hold" or reframe.subject is None:
                    # No substitution here. This branch used to notice that a
                    # subject too wide to sit in the crop could be read across
                    # instead, and swap the hold for a sweep. It looks like help
                    # and it is the execution layer deciding: a replan that had
                    # just chosen hold *because* travelling across the title was
                    # what cut it came back describing a sweep across the title,
                    # having been overruled by a layer it cannot see or argue
                    # with. The fit is recorded above; whether to answer it with
                    # a different move, a different take or a partial view is a
                    # planning question, and the shot reviewer now puts it to the
                    # planner in those terms.
                    #
                    # A held shot still has to be aimed. Every other move
                    # measures where its subject is; this one fell through to
                    # the executor's fallback, which reads the nine-box name as
                    # a coordinate -- "mid_right" became x=0.61 and a phone
                    # spanning 0.475 to 0.825 arrived half out of frame with the
                    # wall behind it filling the rest. The card already knows
                    # where the thing is, and that lesson is written down for
                    # the handoff path, which was fixed and left this one alone.
                    if reframe.subject is not None:
                        reference = (
                            reference_samples.get(reframe.subject.entity_id)
                            if reframe.subject.entity_id else None
                        )
                        box = (
                            find_subject(
                                card, reframe.subject.description,
                                entity_id=reframe.subject.entity_id,
                            )
                            if card is not None
                            else None
                        )
                        # A card box is one moment. That is the whole answer for
                        # a locked-off frame and a snapshot for anything else --
                        # the subject the card saw at 1.2s is somewhere else by
                        # the end of a take whose camera pans. The card says
                        # which of those this is, in a field nothing read.
                        settled = box is not None and not box.moves and not (
                            card or {}
                        ).get("camera_motion")
                        centre = (
                            (box.centre_x, box.centre_y, box.width, box.height)
                            if settled and box is not None
                            else None
                        )
                        if reference is not None:
                            ref_boxes, _, _ = reference
                            present = [
                                item for item in ref_boxes
                                if item.get("present")
                                and item.get("centre_x") is not None
                            ]
                            if present:
                                count = len(present)
                                centre = (
                                    sum(float(item["centre_x"]) for item in present)
                                    / count,
                                    sum(float(item["centre_y"]) for item in present)
                                    / count,
                                    sum(float(item.get("width") or 0.0) for item in present)
                                    / count,
                                    sum(float(item.get("height") or 0.0) for item in present)
                                    / count,
                                )
                        if centre is None and not reframe.subject.entity_id:
                            # Either the card had no box for what was named, or
                            # it had one that will not hold still. Measure across
                            # this shot's own window: for a held frame the mean
                            # of the trajectory is the placement that keeps the
                            # subject inside for all of it, rather than framing
                            # where it started and letting it walk out.
                            if not _may_ask(client):
                                continue
                            frames, _ = _sample_frames(
                                source,
                                clip.approx_in_seconds,
                                clip.approx_out_seconds,
                                work,
                            )
                            _afford(report)
                            boxes, usage = _locate_subject(
                                frames, reframe.subject.description, client=client,
                                report=report,
                            )
                            _charge(report, "subject", usage)
                            present = [
                                b for b in boxes
                                if b.get("present") and b.get("centre_x") is not None
                            ]
                            if present:
                                count = len(present)
                                centre = (
                                    sum(float(b["centre_x"]) for b in present) / count,
                                    sum(float(b["centre_y"]) for b in present) / count,
                                    sum(float(b.get("width") or 0.0) for b in present) / count,
                                    sum(float(b.get("height") or 0.0) for b in present) / count,
                                )
                        if centre is not None:
                            cx, cy, bw, bh = centre
                            paths[clip.clip_id] = build_crop_path(
                                [
                                    Observation(
                                        seconds=0.0,
                                        centre_x=cx,
                                        centre_y=cy,
                                        width=bw,
                                        height=bh,
                                    )
                                ],
                                source_aspect=source.aspect_ratio,
                                target_aspect=target_aspect,
                                energy=reframe.camera_energy,
                                # A shot that said it settles is not a follow
                                # that was downgraded. One look means "stay on
                                # this", which for something that walks is a
                                # follow and for something standing still is a
                                # held frame -- both are the plan being carried
                                # out, and only one of them used to say so.
                                planned_to_move=reframe.planned_to_move,
                                framing=reframe.framing,
                                clip_id=clip.clip_id,
                                min_visible=reframe.subject.min_visible,
                                degradations=report.degradations,
                            )
                            report.subject_notes[clip.clip_id] = (
                                f"hold on ({cx:.3f}, {cy:.3f})"
                            )
                            report.static_shots += 1
                            continue
                    report.static_shots += 1
                    continue

                # A follow needs to know where the subject is. The card answered
                # that when it was written and the answer has not changed since,
                # so ask it before paying for a fresh grounding on every rhythm
                # tweak, second aspect and review round.
                known = (
                    find_subject(
                        card, reframe.subject.description,
                        entity_id=reframe.subject.entity_id,
                    )
                    if card is not None
                    else None
                )
                reference = (
                    reference_samples.get(reframe.subject.entity_id)
                    if reframe.subject.entity_id else None
                )
                semantic_anchors = reference[2] if reference is not None else ()
                if reference is not None:
                    boxes, times, _ = reference
                    frames = []
                    report.subject_notes[clip.clip_id] = (
                        f"reference identity: {reframe.subject.entity_id}"
                    )
                elif reframe.subject.entity_id:
                    boxes, times, frames = [], [], []
                elif known is not None and move != "follow_subject":
                    boxes = [
                        {
                            "frame_index": 0,
                            "present": True,
                            "centre_x": known.centre_x,
                            "centre_y": known.centre_y,
                            "width": known.width,
                            "height": known.height,
                        }
                    ]
                    times = [clip.approx_in_seconds]
                    frames = []
                    report.subject_notes[clip.clip_id] = f"card: {known.label}"
                else:
                    if not _may_ask(client):
                        continue
                    # A card box is one observed place, not a trajectory.  It is
                    # enough to aim a hold and categorically insufficient for an
                    # explicit follow: without SAM, one observation made every
                    # follow static while the report claimed the chosen intent.
                    frames, times = _sample_frames(
                        source, clip.approx_in_seconds, clip.approx_out_seconds, work
                    )
                    _afford(report)
                    boxes, usage = _locate_subject(
                        frames, reframe.subject.description, client=client,
                        report=report,
                    )
                    _charge(report, "subject", usage)

                wants_tilt = move == "tilt"
                observations = []
                for box in boxes:
                    if not box.get("present"):
                        continue
                    index = int(box.get("frame_index", -1))
                    if not 0 <= index < len(times):
                        continue
                    try:
                        observations.append(
                            Observation(
                                seconds=times[index] - clip.approx_in_seconds,
                                centre_x=float(box["centre_x"]),
                                centre_y=float(box["centre_y"]),
                                width=float(box["width"]),
                                height=float(box["height"]),
                            )
                        )
                    except (OutOfFrame, KeyError, TypeError, ValueError) as error:
                        # One unusable observation is not a reason to abandon the
                        # shot; the remaining samples still describe the motion.
                        report.subject_notes[clip.clip_id] = str(error)[:160]

                if not observations:
                    report.degradations.append(
                        DegradationStep(
                            clip_id=clip.clip_id,
                            ladder="center_crop",
                            trigger=(
                                "the subject was not located in any sampled "
                                "frame, so the shot is framed centrally"
                            ),
                            measured={"samples": float(len(times))},
                        )
                    )
                    continue

                note = next(
                    (
                        str(box["disambiguation"])
                        for box in boxes
                        if box.get("disambiguation")
                    ),
                    "",
                )
                if note:
                    report.subject_notes[clip.clip_id] = note

                # Gemini said which subject; the tracker says where it goes. When
                # a checkpoint is available the trajectory is measured per frame
                # rather than interpolated between five samples.
                if checkpoint is not None and boxes and reference is None:
                    seed = next(
                        (
                            box
                            for box in boxes
                            if box.get("present")
                            and box.get("centre_x") is not None
                        ),
                        None,
                    )
                    if seed is not None:
                        try:
                            tracked, states = _track_subject(
                                source,
                                clip,
                                reframe.subject.description,
                                _seed_box(seed),
                                checkpoint,
                                work,
                            seed_time_seconds=(
                                    times[int(seed.get("frame_index", -1))]
                                    if 0 <= int(seed.get("frame_index", -1)) < len(times)
                                    else None
                                ),
                                semantic_anchors=semantic_anchors,
                                require_identity_validation=bool(
                                    reframe.subject.entity_id
                                ),
                            )
                        except Exception as error:  # tracking is an optimisation
                            report.subject_notes[clip.clip_id] = (
                                f"tracking unavailable, using sampled positions: "
                                f"{type(error).__name__}"
                            )
                        else:
                            report.subject_notes[clip.clip_id] = f"tracked {states}"
                            total = sum(states.values()) or 1
                            kept = states.get("tracked", 0)
                            if kept / total < TRACK_QUORUM:
                                # Most of the shot was not tracked. Falling back to
                                # the sampled positions is right, but doing it
                                # quietly is how a nine-frame trajectory becomes a
                                # single observation and the crop lands wherever
                                # that one frame happened to be -- on a hand rather
                                # than the phone it was holding, in one case.
                                report.degradations.append(
                                    DegradationStep(
                                        clip_id=clip.clip_id,
                                        ladder="other",
                                        ladder_other="tracking_lost_most_frames",
                                        trigger=(
                                            "the tracker held the subject in "
                                            f"{kept} of {total} analysed frames, "
                                            "so the framing rests on sampled "
                                            "positions instead"
                                        ),
                                        measured={
                                            "tracked_frames": float(kept),
                                            "analysed_frames": float(total),
                                            "kept_fraction": round(kept / total, 3),
                                        },
                                    )
                                )
                            elif tracked:
                                report.subject_tracks[clip.clip_id] = [{
                                    "seconds": round(one.seconds, 4),
                                    "centre_x": round(one.centre_x, 6),
                                    "centre_y": round(one.centre_y, 6),
                                    "width": round(one.width, 6),
                                    "height": round(one.height, 6),
                                    "subject": reframe.subject.description,
                                    "entity_id": reframe.subject.entity_id,
                                    "semantic_identity_status": (
                                        "reference_validated"
                                        if reframe.subject.entity_id
                                        else "description_grounded"
                                    ),
                                    "source": "sam2.1",
                                } for one in tracked]
                                observations = tracked

                if wants_tilt:
                    path = build_tilt_path(
                        observations,
                        source_aspect=source.aspect_ratio,
                        target_aspect=target_aspect,
                        energy=reframe.camera_energy,
                        clip_id=clip.clip_id,
                        degradations=report.degradations,
                    )
                else:
                    path = build_crop_path(
                        observations,
                        source_aspect=source.aspect_ratio,
                        target_aspect=target_aspect,
                        energy=reframe.camera_energy,
                        framing=reframe.framing,
                        clip_id=clip.clip_id,
                        min_visible=reframe.subject.min_visible,
                        degradations=report.degradations,
                    )
                paths[clip.clip_id] = path
                if path.is_static:
                    report.static_shots += 1
                else:
                    report.following_shots += 1
            except ReferenceShotUnusable as unusable:
                # Find every shot that cannot be delivered, not the
                # first. Raising here meant one swap per attempt and a
                # full re-render between them, so three attempts covered
                # three shots -- and this material had more than three.
                # The work already done on the others is thrown away
                # either way; doing it once and reporting all of them
                # lets the selection repair them in a single pass.
                unusable_shots.append(unusable)
                continue
    if unusable_shots:
        raise ReferenceShotsUnusable(unusable_shots)
    return paths


def split_handoffs(edl: EDL) -> EDL:
    """Turn a two-subject shot into two shots, each with one subject.

    A camera cannot follow one thing and then another inside a single move
    without losing both: the first subject is abandoned mid-gesture and the
    second is arrived at late. Splitting at the plan layer gives each half its
    own frame and its own follow, and the join between them is a cut, which is
    how an editor carries the eye from one thing to the next anyway.

    Each half keeps the parent's rhythm decision, halved, so a handoff costs
    the same screen time it was given.
    """

    rewritten: list[Any] = []
    for clip in edl.clips:
        reframe = clip.reframe
        second = getattr(reframe, "then_subject", None) if reframe else None
        if reframe is None or second is None:
            rewritten.append(clip)
            continue

        midpoint = (clip.approx_in_seconds + clip.approx_out_seconds) / 2.0
        first_half = clip.model_copy(
            update={
                "clip_id": f"{clip.clip_id}a",
                "approx_out_seconds": midpoint,
            }
        )
        second_half = clip.model_copy(
            update={
                "clip_id": f"{clip.clip_id}b",
                "approx_in_seconds": midpoint,
                "reframe": reframe.model_copy(update={"subject": second}),
            }
        )
        rewritten += [first_half, second_half]
    return edl.model_copy(update={"clips": rewritten})


def run(
    edl: EDL,
    sources: dict[str, Source],
    # A cut carried by what people say does not need a bed, and this has
    # handled its absence since the day it stopped refusing to run without
    # one. The signature was the last thing still claiming otherwise.
    grid: BeatGrid | None,
    output_dir: Path,
    *,
    target_aspect: float,
    intent: str,
    brief: str = "",
    rhythm_context: dict[str, dict] | None = None,
    music: Path | None = None,
    cards: dict[str, Path] | None = None,
    checkpoint: Path | None = None,
    ledger: Ledger | None = None,
    decide_rhythm_first: bool = True,
    target_seconds: float = 0.0,
    duration_mode: str = "exact",
    max_static_seconds: float = 0.0,
    keep_voice: bool = False,
    under_speech: str = "duck",
    client: Any | None = None,
    transcripts: Mapping[str, dict | None] | None = None,
    grounding_spec: Any | None = None,
    grounding_memory: Path | None = None,
    confirmed_identities: "dict[str, Any] | None" = None,
    rhythm_shots: "list[Any] | None" = None,
    upload_cache: Any | None = None,
    source_motion_measurements: Mapping[str, Any] | None = None,
) -> tuple[RenderResult, RenderPlan, Report, EDL]:
    """Take an EDL to a finished file.

    The resolved EDL comes back with it. The rhythm pass rewrites durations
    inside this call, and a caller that renders a second aspect from the EDL
    it passed in gets the placeholder lengths instead of the decided ones --
    which looks like a render that worked and sounds like one that ignored the
    music.
    """

    report = Report(ledger=ledger)

    # Everything needed to prove source-clock feasibility is already local at
    # this boundary.  Refuse an impossible Selection before paying Rhythm to
    # choose timings that no answer could make executable.
    from montagewright.planning_release import resolved_source_contract_faults

    preflight_faults = resolved_source_contract_faults(edl)
    if preflight_faults:
        raise ValueError(
            "selection has unresolved source-clock contracts before Rhythm: "
            + "; ".join(preflight_faults)
        )

    # Runs whether or not there is a track. It was gated on having one --
    # the reasoning being that with no music there is nothing to reconcile --
    # and that was wrong: what it reconciles is the sequence against itself.
    # Without it every length was whatever selection guessed for that shot
    # alone, and nothing ever asked whether eight of them in a row had any
    # shape. Speech-led cuts, which need shaping most, got none of it.
    if decide_rhythm_first:
        if ledger is not None:
            ledger.check()
        edl, usage = decide_rhythm(
            edl,
            grid,
            intent=intent,
            brief=brief,
            context=rhythm_context or {},
            music=music,
            shots=rhythm_shots,
            cache=upload_cache,
            target_seconds=target_seconds,
            duration_mode=duration_mode,
            client=client,
            ledger=ledger,
            artifact_dir=output_dir / "work",
        )
        _charge(report, "rhythm", usage)

    # Music grounding and dialogue boundaries both move cuts.  Neither may
    # silently invalidate the other, so converge them before rendering and
    # report only the final timeline. Speech has the final say inside each
    # round; a following grounding round proves the musical request still
    # lands. A cycle is a real planning conflict, not something to hide.
    dialogue_history: list[str] = []
    seen_windows: set[tuple[tuple[float, float], ...]] = set()
    timeline = ground_timeline(edl, grid)
    for attempt in range(4):
        edl = apply_to_edl(edl, timeline)
        if not transcripts:
            break
        from montagewright.transcript import (
            DialogueBoundaryError, snap_edl_to_dialogue,
        )

        snapped, dialogue_notes, dialogue_faults = snap_edl_to_dialogue(
            edl, transcripts
        )
        dialogue_history.extend(dialogue_notes)
        if dialogue_faults:
            raise DialogueBoundaryError(
                "final cut crosses unfinished dialogue; reselect or replan: "
                + "; ".join(dialogue_faults)
            )
        if not dialogue_notes:
            break
        signature = tuple(
            (round(clip.approx_in_seconds, 4), round(clip.approx_out_seconds, 4))
            for clip in snapped.clips
        )
        if signature in seen_windows or attempt == 3:
            raise DialogueBoundaryError(
                "music grounding and dialogue-safe boundaries do not "
                "converge; replan the named speech shots"
            )
        seen_windows.add(signature)
        edl = snapped
        timeline = ground_timeline(edl, grid)
    report.plan_disagreements.extend(dict.fromkeys(dialogue_history))
    report.aligned_cuts = timeline.aligned_count
    report.total_cuts = len(timeline.clips)
    report.delivered_seconds = round(timeline.duration_seconds, 2)
    # Whether a cut missed the grid or was never aimed at it are different
    # facts, and a bare "8/10" cannot tell them apart -- which is how a
    # deliberate content-led cut reads as a failure to align.
    report.rhythm_decisions = {
        entry.clip.clip_id: {
            "cut_on_beat": entry.clip.music_sync.cut_on_beat,
            "landed_on": entry.landed_on,
            # Without this the count could not be read back: a cue id is a
            # position in a list, not a description, so "nine of nine on the
            # music" hid six cuts running one beat ahead of the bar.
            "landed_kind": entry.landed_kind,
            "grounding_note": entry.note,
            "seconds": round(entry.duration_seconds, 3),
            "why": entry.clip.music_sync.rhythm_reason,
        }
        for entry in timeline.clips
    }
    report.moves_too_short = {
        entry.clip.clip_id: entry.move_too_short
        for entry in timeline.clips
        if entry.move_too_short
    }
    edl, pacing_notes = _audit_static_holds(edl, max_static_seconds)
    report.plan_disagreements.extend(pacing_notes)
    report.plan_disagreements.extend(_resolved_sequence_disagreements(edl))
    # Release gate: the sequence and rhythm passes already had a chance to
    # repair this.  At the resolved source clock we only verify; silently
    # trimming here would move music, subtitles and every downstream frame.
    from montagewright.coverage import (
        TimelineCoverageError, edl_coverage_audit,
    )

    coverage = edl_coverage_audit(
        edl, target_seconds, hard_target=duration_mode == "exact"
    )
    report.coverage_seconds = round(coverage.supported_seconds, 3)
    report.unsupported_seconds = round(coverage.unsupported_seconds, 3)
    report.coverage_details = [
        {
            "clip_id": entry.clip_id,
            "picture_role": entry.picture_role,
            "seconds": round(entry.seconds, 3),
            "audio_seconds": round(entry.audio_seconds, 3),
            "visual_only_seconds": round(entry.visual_only_seconds, 3),
            "supported_seconds": round(entry.supported_seconds, 3),
        }
        for entry in coverage.entries
    ]
    if target_seconds > 0 and coverage.faults:
        raise TimelineCoverageError(
            "final timeline contains duration without content evidence; "
            "selection/rhythm must be structurally replanned: "
            + "; ".join(coverage.faults)
        )
    # Rhythm now fixes the picture timeline, so this is the first point where
    # a return to the speaker after B-roll can be mapped to the exact progress
    # of the continuing audio assignment.  It must precede SAM/reframing.
    edl, speaker_notes = align_speaker_pictures_to_audio(edl)
    report.plan_disagreements.extend(speaker_notes)
    # Lip-sync owns the final source in-point for speaker pictures.  It runs
    # after musical/dialogue grounding, so it must not be allowed to move a
    # clip through an action, source-motion or usable-window boundary that was
    # proved on the earlier window.
    from montagewright.planning_release import (
        audio_timeline_faults, resolved_source_contract_faults,
    )

    resolved_faults = (
        *resolved_source_contract_faults(edl),
        *audio_timeline_faults(edl),
    )
    if resolved_faults:
        raise ValueError(
            "resolved timeline violates local source/audio contracts: "
            + "; ".join(dict.fromkeys(resolved_faults))
        )
    for clip in edl.clips:
        if clip.clip_id in report.rhythm_decisions:
            report.rhythm_decisions[clip.clip_id]["seconds"] = round(
                clip.approx_out_seconds - clip.approx_in_seconds, 3
            )

    paths = follow_subjects(
        edl,
        sources,
        target_aspect=target_aspect,
        report=report,
        cards=cards,
        checkpoint=checkpoint,
        client=client,
        grounding_spec=grounding_spec,
        grounding_output=output_dir / "work" / "reference-grounding",
        grounding_memory=grounding_memory,
        confirmed_identities=confirmed_identities,
        upload_cache=upload_cache,
    )

    for clip in edl.clips:
        reframe = clip.reframe
        if reframe is None:
            continue
        report.source_motion[clip.clip_id] = reframe.source_motion_role
        report.source_motion_details[clip.clip_id] = {
            "role": reframe.source_motion_role,
            "description": reframe.source_motion_description,
            "window": [
                round(clip.approx_in_seconds, 3),
                round(clip.approx_out_seconds, 3),
            ],
            "measurement": _source_motion_measurement(
                (source_motion_measurements or {}).get(clip.source_id, ()),
                clip.approx_in_seconds,
                clip.approx_out_seconds,
            ),
        }
        measured = report.source_motion_details[clip.clip_id]["measurement"]
        semantic_moves = reframe.source_motion_role != "locked"
        if measured["available"] and bool(measured["moving"]) != semantic_moves:
            report.plan_disagreements.append(
                f"{clip.clip_id} source motion differs: local measurement says "
                f"{'moving' if measured['moving'] else 'still'} but semantic role "
                f"is {reframe.source_motion_role}"
            )
        path = paths.get(clip.clip_id)
        report.digital_motion[clip.clip_id] = _digital_motion_of(path)

    write_crops(paths, output_dir / "work" / "crops.json")

    plan = plan_render(
        edl, sources, target_aspect=target_aspect, crop_paths=paths,
        output_size=delivery_size(target_aspect),
    )
    report.degradations.extend(plan.degradations)

    # Kept. A finished cut answers whether this is a film; it does not answer
    # whether any one shot came out the way it was planned -- six composition
    # faults in a row survived review of the whole thing and were found by
    # opening a single shot. The segments are what that question is asked of.
    result = render(
        plan, output_dir, music=music, keep_segments=True,
        keep_voice=(
            keep_voice
            or any(
                segment.audio_role
                in {"narrative", "sync_action", "ambient_texture"}
                for segment in plan.segments
            )
        ),
        under_speech=under_speech,
    )
    # The resolved plan is the only truthful source for a later Web/CLI
    # edit. Selection in-points precede action snapping and beat grounding;
    # reconstructing from them made a zero-change recut select different
    # source frames. Commit the exact rendered windows with the first film.
    from montagewright.measure.storage import write_json
    from montagewright.executor import allocate_timeline_frames

    frame_spans = allocate_timeline_frames(
        [segment.duration_seconds for segment in plan.segments],
        plan.output_fps,
    )

    write_json(output_dir / "work" / "current-timeline.json", {
        "version": "montagewright-current-timeline-v2",
        "revision": 0,
        "output_fps": plan.output_fps,
        "output_size": list(plan.output_size),
        "music_from_seconds": plan.music_from_seconds,
        "music_spans": plan.music_spans,
        "shots": [
            {
                "selection_index": index,
                "in_seconds": segment.in_seconds,
                "start_frame": start,
                "frame_count": end - start,
                "seconds": (end - start) / plan.output_fps,
                "gain_db": segment.gain_db,
                "audio_role": segment.audio_role,
                "audio_completion": segment.audio_completion,
                "picture_role": segment.picture_role,
                "coverage_claim_seconds": segment.coverage_claim_seconds,
            }
            for index, (segment, (start, end)) in enumerate(
                zip(plan.segments, frame_spans, strict=True)
            )
        ],
        "audio_assignments": [
            {
                "audio_id": audio.audio_id,
                "source_id": audio.source.source_id,
                "in_seconds": audio.in_seconds,
                "out_seconds": audio.out_seconds,
                "timeline_start_frame": audio.timeline_start_frame,
                "frame_count": audio.frame_count,
                "role": audio.role,
                "completion": audio.completion,
                "gain_db": audio.gain_db,
                "why": audio.why,
            }
            for audio in plan.audio_assignments
        ],
    })
    report.delivered_seconds = round(result.duration_seconds, 2)
    return result, plan, report, edl
