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

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
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
TRACK_QUORUM = 0.5


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
    from fractions import Fraction

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
) -> tuple[list[Observation], dict[str, int]]:
    """Propagate a Gemini seed across every analysed frame of the shot."""

    from montagewright.measure.sam_tracking import track_bbox_sam21

    track = track_bbox_sam21(
        video_path=source.path,
        checkpoint_path=checkpoint,
        seed_time_ms=int(
            (seed_time_seconds
             if seed_time_seconds is not None
             else clip.approx_in_seconds + 0.1) * 1000
        ),
        seed_box_2d=seed_box,
        target_description=subject_description,
        output_dir=work / (
            f"sam-{clip.clip_id}-{track_name}"
            if track_name else f"sam-{clip.clip_id}"
        ),
        seed_source="gemini_frame_grounding",
        analysis_fps=TRACK_FPS,
        max_side=960,
        allowed_start_ms=int(clip.approx_in_seconds * 1000),
        allowed_end_ms=int(clip.approx_out_seconds * 1000),
    )
    return observations_from_sam(
        track, clip_start_seconds=clip.approx_in_seconds
    )


def _measure_looks(
    looks, source, clip, work: Path, report, client, target_aspect: float,
    checkpoint: Path | None = None,
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
        if look.at not in seen:
            _afford(report)
            boxes, usage = _locate_subject(
                frames, look.at, client=client, report=report
            )
            _charge(report, "subject", usage)
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
            seen[look.at] = (
                sum(float(one["centre_x"]) for one in found) / len(found),
                sum(float(one["centre_y"]) for one in found) / len(found),
                sum(float(one.get("height") or 0.0) for one in found) / len(found),
            )
            walked[look.at] = sorted(
                (
                    moments[int(one["frame_index"])],
                    float(one["centre_x"]),
                    float(one["centre_y"]),
                )
                for one in found
                if 0 <= int(one.get("frame_index", -1)) < len(moments)
            )
            # Gemini identifies which object the edit means; SAM turns that
            # semantic seed into the dense trajectory used by the crop. This
            # used to happen only in the single-look branch, so a push or a
            # handoff bypassed SAM precisely when a tight crop made small
            # tracking errors most visible.
            if checkpoint is not None:
                seed = found[0]
                frame_index = int(seed.get("frame_index", -1))
                seed_time = (
                    times[frame_index]
                    if 0 <= frame_index < len(times)
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
                            "source": "sam2.1",
                        } for one in tracked)
                        walked[look.at] = [
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
        if look.at not in seen:
            continue
        centre_x, centre_y, tall = seen[look.at]
        width = base
        if look.framing == "fill" and tall > 0.0:
            # The same reach a push used to compute, bounded the same way.
            width = base * max(0.2, min(0.9, tall / 0.66))
        height = min(1.0, base_height * (width / base) if base > 0 else base_height)
        share = PLACEMENT.get(look.framing, 0.5)
        lift = height * (0.5 - share)
        stops.append((
            max(0.0, float(look.seconds)),
            centre_x,
            centre_y + lift,
            width,
        ))
        # Keep the measured path.  Whether it matters is a property of the
        # delivered crop, not of the full source: a tiny source-space drift
        # can become a visible reversal during a tight push.  The path builder
        # applies its deadband after it knows the crop width.
        path = walked.get(look.at) or []
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
    with tempfile.TemporaryDirectory() as raw_work:
        work = Path(raw_work)
        total = len(edl.clips)
        for index, clip in enumerate(edl.clips, start=1):
            reframe = clip.reframe
            if reframe is None:
                continue
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
            if reframe.subject is not None and card is None:
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
                find_subject(card, reframe.subject.description)
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
                    box = find_subject(card, look.at)
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
                    target_aspect, checkpoint,
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
                if reframe.subject is not None and not _may_ask(client):
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
                    box = (
                        find_subject(card, reframe.subject.description)
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
                    if centre is None:
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
                find_subject(card, reframe.subject.description)
                if card is not None
                else None
            )
            if known is not None and move != "follow_subject":
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
            if checkpoint is not None and boxes:
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
    max_static_seconds: float = 0.0,
    keep_voice: bool = False,
    under_speech: str = "duck",
    client: Any | None = None,
    transcripts: Mapping[str, dict | None] | None = None,
) -> tuple[RenderResult, RenderPlan, Report, EDL]:
    """Take an EDL to a finished file.

    The resolved EDL comes back with it. The rhythm pass rewrites durations
    inside this call, and a caller that renders a second aspect from the EDL
    it passed in gets the placeholder lengths instead of the decided ones --
    which looks like a render that worked and sounds like one that ignored the
    music.
    """

    report = Report(ledger=ledger)

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
            target_seconds=target_seconds,
            client=client,
            ledger=ledger,
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
    # Rhythm now fixes the picture timeline, so this is the first point where
    # a return to the speaker after B-roll can be mapped to the exact progress
    # of the continuing audio assignment.  It must precede SAM/reframing.
    edl, speaker_notes = align_speaker_pictures_to_audio(edl)
    report.plan_disagreements.extend(speaker_notes)
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
        }
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
