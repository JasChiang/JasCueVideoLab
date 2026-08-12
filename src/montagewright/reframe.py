"""Follow a subject with the crop instead of holding it still.

The semantic layer names the subject; grounding finds it at a handful of
sampled frames; this module turns those observations into a crop that moves.

Three choices here come straight from what the previous system got wrong.

Failure is judged on the raw track, before smoothing. Filtering a trajectory
and then testing the filtered result lets the filter manufacture the problem
it is then blamed for.

The follow uses the ninetieth percentile of where the subject actually went,
not its extremes. One frame of a hand crossing the lens should not define the
whole move; the previous system planned for the worst frame and so planned a
move nobody asked for.

Speed limits live in viewport widths per second, not pixels. A pixel limit is
only meaningful beside the resolution it was written against, which is how the
same constant meant two different things on 1080 and 4K sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from montagewright.executor import CROP_MARGIN, CropBox
from montagewright.schema import CameraEnergy, DegradationStep

# Per camera_energy, in viewport widths per second and per second squared.
# 720 px/s on a 1080-wide portrait viewport, the old constant, is 0.667 vw/s --
# which is roughly what "active" means here.
ENERGY_LIMITS: dict[CameraEnergy, dict[str, float]] = {
    "calm": {"max_speed": 0.25, "max_accel": 0.60, "lead": 0.10},
    "active": {"max_speed": 0.67, "max_accel": 1.67, "lead": 0.20},
    "dynamic": {"max_speed": 1.20, "max_accel": 3.00, "lead": 0.30},
}

# Movement below this is invisible and only adds jitter, so the camera holds.
DEADBAND = 0.02

# How much of a subject's travel has to end up as net displacement before a
# follow is worth executing. A subject that steps out and comes back inside
# one short shot has gone nowhere, and chasing each swing reads as a wobble.
#
# Crossing this threshold is not a veto. The planner asked for a follow and a
# follow is what it gets when there is motion to follow; below the threshold
# the shot is framed on where the subject spent its time AND the substitution
# is written into the degradation record. Silently returning a hold, which is
# what this did first, tells the planner its instruction was carried out.
MIN_DIRECTNESS = 0.6


class OutOfFrame(ValueError):
    """An observation that is not in normalised coordinates.

    Worth its own type because the failure it prevents is silent. A model that
    answers in pixels produces values like 381 where 0..1 was asked for, and
    clamping those into range yields a crop that sits at the edge for every
    frame -- which reads downstream as "the subject never moved" and renders
    as a considered hold. A wrong answer wearing the shape of a decision is
    worse than a loud one.
    """


@dataclass(frozen=True)
class Observation:
    """Where the subject was at one sampled moment, normalised."""

    seconds: float
    centre_x: float
    centre_y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name, value in (
            ("centre_x", self.centre_x),
            ("centre_y", self.centre_y),
            ("width", self.width),
            ("height", self.height),
        ):
            if not 0.0 <= value <= 1.0:
                raise OutOfFrame(
                    f"{name}={value:g} is outside 0..1 at t={self.seconds:g}s. "
                    "Observations are frame fractions; a value above 1 is "
                    "usually pixels, which has to be converted by whoever "
                    "knows the frame size rather than clamped here."
                )


@dataclass(frozen=True)
class Keyframe:
    seconds: float
    crop: CropBox


@dataclass
class CropPath:
    """A crop over time. One keyframe means a static crop."""

    keyframes: list[Keyframe] = field(default_factory=list)

    @property
    def is_static(self) -> bool:
        if len(self.keyframes) < 2:
            return True
        first = self.keyframes[0].crop
        return all(
            abs(frame.crop.x - first.x) < 1e-6
            and abs(frame.crop.y - first.y) < 1e-6
            and abs(frame.crop.width - first.width) < 1e-6
            and abs(frame.crop.height - first.height) < 1e-6
            for frame in self.keyframes
        )

    def travel(self) -> float:
        """Total movement across every axis, in viewport widths."""

        return sum(
            max(
                abs(later.crop.x - earlier.crop.x),
                abs(later.crop.y - earlier.crop.y),
                abs(later.crop.width - earlier.crop.width),
            )
            for earlier, later in zip(self.keyframes, self.keyframes[1:])
        )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _limit_speed(
    keyframes: list[Keyframe], limits: dict[str, float]
) -> tuple[list[Keyframe], float]:
    """Hold the camera inside its energy budget, reporting what it hit."""

    if len(keyframes) < 2:
        return keyframes, 0.0

    limited = [keyframes[0]]
    peak = 0.0
    for previous, current in zip(keyframes, keyframes[1:]):
        span = max(current.seconds - previous.seconds, 1e-6)
        anchor = limited[-1].crop
        # All three axes, and scaled together rather than one at a time.
        # This read only `x`, so a tilt arrived at any speed it liked and a
        # push changed size instantly -- the limiter was named for the
        # budget but only enforced it on the one move that existed when it
        # was written. Clamping each axis on its own would also bend the
        # path, since a diagonal whose across component is cut and whose
        # down component is not stops being a straight line.
        want = (
            current.crop.x - anchor.x,
            current.crop.y - anchor.y,
            current.crop.width - anchor.width,
        )
        fastest = max(abs(one) for one in want)
        speed = fastest / span
        peak = max(peak, speed)
        allowed = limits["max_speed"] * span
        share = allowed / fastest if fastest > allowed else 1.0
        moved_x, moved_y, moved_w = (one * share for one in want)
        width = min(1.0, max(1e-3, anchor.width + moved_w))
        height = min(1.0, current.crop.height * (width / max(current.crop.width, 1e-9)))
        limited.append(
            Keyframe(
                seconds=current.seconds,
                crop=CropBox(
                    x=min(max(anchor.x + moved_x, 0.0), max(0.0, 1.0 - width)),
                    y=min(max(anchor.y + moved_y, 0.0), max(0.0, 1.0 - height)),
                    width=width,
                    height=height,
                ),
            )
        )
    return limited, peak


def _smooth(keyframes: list[Keyframe], strength: float = 0.5) -> list[Keyframe]:
    """Take the corners off. Endpoints stay put so the shot starts where it starts."""

    if len(keyframes) < 3:
        return keyframes
    smoothed = [keyframes[0]]
    for previous, current, following in zip(
        keyframes, keyframes[1:], keyframes[2:]
    ):
        blended = (
            previous.crop.x + following.crop.x
        ) / 2.0 * strength + current.crop.x * (1.0 - strength)
        smoothed.append(
            Keyframe(
                seconds=current.seconds,
                crop=CropBox(
                    x=min(max(blended, 0.0), 1.0 - current.crop.width),
                    y=current.crop.y,
                    width=current.crop.width,
                    height=current.crop.height,
                ),
            )
        )
    smoothed.append(keyframes[-1])
    return smoothed


def build_sweep_path(
    *,
    source_aspect: float,
    target_aspect: float,
    duration_seconds: float,
    direction: str,
    energy: CameraEnergy = "calm",
) -> CropPath:
    """A designed move across a static arrangement.

    No subject is followed here because nothing is moving. A row of handsets
    wants the eye carried across it, and that is a decision about how to
    present the frame rather than a reaction to something in it. Treating this
    as a follow of the group's centre yields a hold, which is the answer to a
    question nobody asked.
    """

    if target_aspect < source_aspect:
        crop_width = target_aspect / source_aspect
        crop_height = 1.0
    else:
        crop_width = 1.0
        crop_height = source_aspect / target_aspect

    free_x = max(0.0, 1.0 - crop_width)
    if free_x <= 0.0:
        return CropPath(
            [Keyframe(0.0, CropBox(0.0, (1.0 - crop_height) / 2.0, crop_width, crop_height))]
        )

    # Sweep the width the energy allows in the time available, never more of
    # the frame than exists.
    limits = ENERGY_LIMITS[energy]
    reach = min(free_x, limits["max_speed"] * duration_seconds)
    inset = free_x * CROP_MARGIN
    if direction == "sweep_left":
        start, end = min(free_x - inset, inset + reach), inset
    else:
        start, end = inset, min(free_x - inset, inset + reach)

    y = (1.0 - crop_height) / 2.0
    return CropPath(
        [
            Keyframe(0.0, CropBox(start, y, crop_width, crop_height)),
            Keyframe(
                duration_seconds,
                CropBox(end, y, crop_width, crop_height),
            ),
        ]
    )


# The shortest pause that reads as the camera having arrived. Below this the
# eye registers a slowing rather than a stop, which is not the same thing --
# the point of resting is that the subject gets a moment of stillness to be
# looked at, not that the move decelerates politely.
SETTLE_SECONDS = 0.35

# And never more than this share of the shot at each end, so a short take is
# not all settling and no move.
SETTLE_SHARE = 0.25

# A movement smaller than this share of the delivered frame is not worth
# chasing.  Source-space deadbands are insufficient once a crop zooms in: a
# coin drifting 0.012 of a 16:9 source moves almost 0.06 of a 9:16 crop after
# a push.  The viewer sees the latter number.
SCREEN_DEADBAND = 0.02


def _with_rest(
    keyframes: list[Keyframe],
    duration_seconds: float,
    energy: CameraEnergy,
    *,
    clip_id: str = "",
    degradations: list[DegradationStep] | None = None,
) -> CropPath:
    """Let a designed move arrive somewhere and stay there for a moment.

    A path of two keyframes, one at zero and one at the end, is moving in
    every frame of the shot. That is not a pan: it is a pan with its first
    and last seconds cut off, and it reads as one -- the eye never gets a
    still frame to recognise where it started or where it ended up.

    So the move rests at both ends and travels in between. This is craft
    rather than intent, which is why it happens here and is not another field
    for the planner to fill: nobody is asked to choose the easing curve
    either. What is the planner's is the total length, and when settling plus
    travelling will not fit inside it at a speed this energy allows, that is
    said rather than absorbed.

    Absorbing it is what happened before. `_limit_speed` clamped the speed
    and the camera simply stopped partway: a 1.2s calm pan across 0.700 of
    frame arrived 43% of the way and no degradation was recorded. The
    destination -- usually the point of the shot -- never appeared at all.
    """

    if len(keyframes) != 2 or duration_seconds <= 0:
        return CropPath(keyframes)

    start, end = keyframes[0].crop, keyframes[-1].crop
    # How far the frame changes, whichever way it changes. A push keeps its
    # centre and alters its size, so measuring only translation reported a
    # zoom as motionless and left it resting nowhere.
    travelled = max(
        abs((end.x + end.width / 2) - (start.x + start.width / 2)),
        abs((end.y + end.height / 2) - (start.y + start.height / 2)),
        abs(end.width - start.width),
        abs(end.height - start.height),
    )
    if travelled < DEADBAND:
        return CropPath(keyframes)

    settle = min(SETTLE_SECONDS, duration_seconds * SETTLE_SHARE)
    moving = duration_seconds - settle * 2
    if moving <= 0:
        return CropPath(keyframes)

    ceiling = ENERGY_LIMITS[energy]["max_speed"]
    if degradations is not None and travelled / moving > ceiling + 1e-6:
        degradations.append(
            DegradationStep(
                clip_id=clip_id,
                ladder="other",
                ladder_other="move_does_not_fit_the_time",
                trigger=(
                    "the move cannot rest at both ends and still cross the "
                    "distance inside this shot at this energy, so it arrives "
                    "short of where it was aimed"
                ),
                measured={
                    "travel_vw": round(travelled, 4),
                    "seconds_moving": round(moving, 3),
                    "needed_speed_vw_s": round(travelled / moving, 4),
                    "max_speed_vw_s": ceiling,
                },
            )
        )

    rested = [
        Keyframe(0.0, start),
        Keyframe(settle, start),
        Keyframe(settle + moving, end),
        Keyframe(duration_seconds, end),
    ]
    limited, _ = _limit_speed(rested, ENERGY_LIMITS[energy])
    return CropPath(limited)


def seconds_needed_for(
    stops: list[tuple[float, float, float, float]],
    energy: CameraEnergy = "calm",
) -> float:
    """The least time this shot can do what it was asked to do.

    Measured from the shot rather than looked up per move. `min_seconds` said
    a pan needs 2.5 seconds, which is one number for every pan on every clip
    ever -- and the real floor is the rests plus the distance divided by the
    speed this energy allows, which for the same word ranges from about a
    second to over two depending on how far the frame has to go and how many
    places it stops. A constant is wrong in both directions: it forbids a
    short pan across a narrow gap and permits a long one that cannot arrive.

    Nothing in here is tuned to any particular footage. The distances come
    from where the subjects were measured to be, and the speed ceiling is a
    property of the energy the shot asked for.
    """

    if len(stops) < 2:
        # A held frame needs long enough to be seen and no longer, which is
        # not a geometric question -- the planner answers it.
        return stops[0][0] if stops else 0.0

    resting = sum(max(0.0, one[0]) or SETTLE_SECONDS for one in stops)
    ceiling = ENERGY_LIMITS[energy]["max_speed"]
    travelling = 0.0
    for before, after in zip(stops, stops[1:]):
        gap = max(
            abs(after[1] - before[1]),
            abs(after[2] - before[2]),
            abs(after[3] - before[3]),
        )
        travelling += gap / ceiling
    return round(resting + travelling, 3)


def build_look_path(
    stops: list[tuple[float, float, float, float]],
    *,
    source_aspect: float,
    target_aspect: float,
    duration_seconds: float,
    energy: CameraEnergy = "calm",
    clip_id: str = "",
    degradations: list[DegradationStep] | None = None,
    source_width: int = 0,
    source_height: int = 0,
    output_width: int = 0,
    output_height: int = 0,
    tracks: "list[list[tuple[float, float, float]]] | None" = None,
) -> CropPath:
    """Walk a shot through the places it looks, resting at each.

    `stops` is one entry per look, already measured: seconds to rest, the
    subject's centre across and down, and the crop width that framing asks
    for. Everything semantic was decided upstream; everything geometric was
    measured here. This only has to route between them.

    One stop is a hold. Two are a move. More stop on the way. A stop whose
    crop width differs from the one before it is a push or a pull, and one
    that also moves is both -- which needed a separate builder before, and
    is now what happens when two looks disagree about size.

    Time is shared out in order: every stop gets the rest it asked for, and
    what is left over is divided between the journeys in proportion to how
    far each has to go. When the rests alone exceed the shot, they are scaled
    down together rather than the last ones being dropped -- a shot that
    cannot afford to stop everywhere should look hurried everywhere, not
    complete and then truncated.
    """

    if not stops:
        return CropPath([])
    if duration_seconds <= 0:
        duration_seconds = 1e-3

    # The tightest this source may be cropped before the delivered frame is
    # being enlarged past what anyone will accept. The single-look push
    # builder has always asked; this one was only given two aspect ratios, so
    # it could crop as tightly as a framing asked for and the pipeline
    # recorded the upscale afterwards. Measuring a violation is not the same
    # as preventing one, and the shot has already been spent by then.
    floor = 0.0
    if source_width and source_height and output_width and output_height:
        widest = min(1.0, target_aspect / source_aspect) if (
            target_aspect < source_aspect
        ) else 1.0
        floor = widest * zoom_budget(
            source_width=source_width,
            source_height=source_height,
            source_aspect=source_aspect,
            target_aspect=target_aspect,
            output_width=output_width,
            output_height=output_height,
        )

    def box(centre_x: float, centre_y: float, width: float) -> CropBox:
        width = max(width, floor)
        height = min(1.0, width * source_aspect / target_aspect)
        width = min(1.0, width)
        x = min(max(centre_x - width / 2.0, 0.0), max(0.0, 1.0 - width))
        y = min(max(centre_y - height / 2.0, 0.0), max(0.0, 1.0 - height))
        return CropBox(x, y, width, height)

    if floor > 0.0 and degradations is not None:
        asked = min((w for _, _, _, w in stops), default=floor)
        if asked < floor - 1e-6:
            degradations.append(
                DegradationStep(
                    clip_id=clip_id,
                    ladder="reduced_zoom",
                    trigger=(
                        "the framing asked to crop tighter than this source "
                        "can supply at the delivery size, so the tightest "
                        "look was opened out to the resolution budget"
                    ),
                    measured={
                        "asked_crop_vw": round(asked, 4),
                        "allowed_crop_vw": round(floor, 4),
                        "max_upscale": MAX_UPSCALE,
                    },
                )
            )

    boxes = [box(cx, cy, w) for _, cx, cy, w in stops]
    def track_matters(
        track: list[tuple[float, float, float]] | None, width: float,
    ) -> bool:
        if not track or len(track) < 2:
            return False
        spread = max(
            max(one[axis] for one in track) - min(one[axis] for one in track)
            for axis in (1, 2)
        )
        return spread / max(width, 1e-9) >= SCREEN_DEADBAND

    walking = [
        one for one in (tracks or [])
        if track_matters(one, min(box.width for box in boxes))
    ]
    if len(boxes) == 1 and not walking:
        return CropPath([Keyframe(0.0, boxes[0])])
    if len(boxes) == 1:
        # One look at something that does not stay put. This used to return
        # here, which is the case a follow most obviously is: a single
        # subject, walking, for the whole shot. It came out as a held frame
        # on the middle of the walk.
        return CropPath(
            _dedupe(
                _limit_speed(
                    [
                        Keyframe(round(when, 4), box(cx, cy, boxes[0].width))
                        for when, cx, cy in _across(
                            walking[0], 0.0, duration_seconds
                        )
                    ],
                    ENERGY_LIMITS[energy],
                )[0]
            )
        )

    rests = [max(0.0, one[0]) or SETTLE_SECONDS for one in stops]
    # Never let resting eat the whole shot; leave at least as much for
    # travelling as for standing still.
    if sum(rests) > duration_seconds * (1.0 - SETTLE_SHARE * 2):
        room = duration_seconds * (1.0 - SETTLE_SHARE * 2)
        scale = room / sum(rests) if sum(rests) > 0 else 0.0
        rests = [one * scale for one in rests]

    spans = [
        max(
            abs((b.x + b.width / 2) - (a.x + a.width / 2)),
            abs((b.y + b.height / 2) - (a.y + a.height / 2)),
            abs(b.width - a.width),
        )
        for a, b in zip(boxes, boxes[1:])
    ]
    left = max(duration_seconds - sum(rests), 1e-6)
    total = sum(spans)
    legs = [
        left * (one / total) if total > 1e-9 else left / len(spans)
        for one in spans
    ]

    # Two stops that measured to the same place is a move that goes nowhere.
    # It reads as a plan being carried out -- there are two looks, the frame
    # travels, the report says so -- and on screen it is a hold. It happens
    # when both looks describe the same thing rather than its two ends: a
    # wordmark asked to be read across came back as a frame drifting from
    # "y Unpa" to "acked", which is a fifth of the title in shot throughout.
    #
    # Said here because this is where the distance is known. The planner was
    # told to describe both ends so they can be told apart, and whether they
    # were is not a fact about the words -- it is a fact about where they
    # turned out to be.
    if degradations is not None and sum(spans) < DEADBAND:
        degradations.append(
            DegradationStep(
                clip_id=clip_id,
                ladder="other",
                ladder_other="looks_landed_on_the_same_place",
                trigger=(
                    f"this shot asks for {len(stops)} looks and they measured "
                    "to within a hair of each other, so the frame travels "
                    "nowhere -- the two ends were probably described as the "
                    "same thing rather than as its edges"
                ),
                measured={
                    "looks": float(len(stops)),
                    "total_travel_vw": round(sum(spans), 4),
                    "deadband_vw": DEADBAND,
                },
            )
        )

    ceiling = ENERGY_LIMITS[energy]["max_speed"]
    hurried = [
        (index, span, leg)
        for index, (span, leg) in enumerate(zip(spans, legs))
        if leg > 1e-9 and span / leg > ceiling + 1e-6
    ]
    if hurried and degradations is not None:
        worst = max(hurried, key=lambda one: one[1] / one[2])
        degradations.append(
            DegradationStep(
                clip_id=clip_id,
                ladder="other",
                ladder_other="looks_do_not_fit_the_time",
                trigger=(
                    f"this shot looks at {len(stops)} things and cannot rest "
                    f"on each and still travel between them in the time it "
                    f"has, so the frame arrives short of the last one"
                ),
                measured={
                    "looks": float(len(stops)),
                    "seconds": round(duration_seconds, 3),
                    "resting_seconds": round(sum(rests), 3),
                    "needed_speed_vw_s": round(worst[1] / worst[2], 4),
                    "max_speed_vw_s": ceiling,
                },
            )
        )

    keyframes: list[Keyframe] = []
    at = 0.0
    for index, crop in enumerate(boxes):
        seen = tracks[index] if tracks and index < len(tracks) else None
        if not track_matters(seen, crop.width):
            seen = None
        # Resting on a subject that is walking is not the same as resting on
        # a place. Where a stop has a track, the frame stays on the subject
        # for as long as it is looking at it; where it has none -- a static
        # thing, or a card-derived box with one position -- the crop is held,
        # which is what this did for every stop before.
        if seen and len(seen) > 1:
            keyframes.extend(
                Keyframe(round(when, 4), box(cx, cy, crop.width))
                for when, cx, cy in _across(seen, at, at + rests[index])
            )
        else:
            keyframes.append(Keyframe(round(at, 4), crop))
            keyframes.append(Keyframe(round(at + rests[index], 4), crop))
        at += rests[index]
        if index < len(legs):
            leg_start = at
            leg_end = at + legs[index]
            next_crop = boxes[index + 1]
            from_track = (
                tracks[index]
                if tracks and index < len(tracks)
                and track_matters(tracks[index], min(crop.width, next_crop.width))
                else None
            )
            to_track = (
                tracks[index + 1]
                if tracks and index + 1 < len(tracks)
                and track_matters(
                    tracks[index + 1], min(crop.width, next_crop.width)
                )
                else None
            )
            # The old path jumped from two mean positions.  That follows a
            # track while resting, then ignores it during the one interval in
            # which a push makes every source-space error larger.  Route the
            # leg through the subjects' positions at the same moments.  For
            # two framings of one subject the tracks are identical, so this
            # becomes a zoom whose optical centre follows the subject.  For a
            # handoff it interpolates between two moving subjects.
            moments = {leg_start, leg_end}
            for track in (from_track, to_track):
                if track:
                    moments.update(
                        when for when, _, _ in track
                        if leg_start < when < leg_end
                    )
            from_static = (
                crop.x + crop.width / 2.0,
                crop.y + crop.height / 2.0,
            )
            to_static = (
                next_crop.x + next_crop.width / 2.0,
                next_crop.y + next_crop.height / 2.0,
            )
            for when in sorted(moments):
                share = (
                    (when - leg_start) / max(leg_end - leg_start, 1e-9)
                )
                from_x, from_y = (
                    _track_at(from_track, when) if from_track else from_static
                )
                to_x, to_y = (
                    _track_at(to_track, when) if to_track else to_static
                )
                centre_x = from_x + (to_x - from_x) * share
                centre_y = from_y + (to_y - from_y) * share
                width = crop.width + (next_crop.width - crop.width) * share
                keyframes.append(
                    Keyframe(round(when, 4), box(centre_x, centre_y, width))
                )
            at = leg_end
    limited, _ = _limit_speed(keyframes, ENERGY_LIMITS[energy])
    return CropPath(_dedupe(limited))


def _across(
    track: list[tuple[float, float, float]], start: float, end: float
) -> list[tuple[float, float, float]]:
    """Where the subject is, at each moment the frame is looking at it.

    The track is in shot time and so is the window, but they are not the same
    span: the frame arrives at a subject partway through a shot and leaves
    before it ends. Sampling the whole track into a shorter window would run
    the subject's movement at the wrong speed, so this reads the track where
    the window actually sits and pins the edges.
    """

    if end <= start:
        return [(start, track[0][1], track[0][2])]

    moments = sorted(
        {start, end}
        | {one[0] for one in track if start < one[0] < end}
    )
    return [(when, *_track_at(track, when)) for when in moments]


def _track_at(
    track: list[tuple[float, float, float]], when: float,
) -> tuple[float, float]:
    """Interpolate a measured subject track in shot time."""

    if when <= track[0][0]:
        return track[0][1], track[0][2]
    if when >= track[-1][0]:
        return track[-1][1], track[-1][2]
    for before, after in zip(track, track[1:]):
        if before[0] <= when <= after[0]:
            span = max(after[0] - before[0], 1e-9)
            share = (when - before[0]) / span
            return (
                before[1] + (after[1] - before[1]) * share,
                before[2] + (after[2] - before[2]) * share,
            )
    return track[-1][1], track[-1][2]


def _dedupe(keyframes: list[Keyframe]) -> list[Keyframe]:
    """Drop keyframes that land on the same moment as the one before."""

    kept = [keyframes[0]]
    for one in keyframes[1:]:
        if one.seconds - kept[-1].seconds > 1e-4:
            kept.append(one)
        else:
            kept[-1] = one
    return kept


def build_handoff_path(
    *,
    source_aspect: float,
    target_aspect: float,
    duration_seconds: float,
    from_centre: float,
    to_centre: float,
    from_width: float = 0.0,
    to_width: float = 0.0,
    energy: CameraEnergy = "calm",
    clip_id: str = "",
    degradations: list[DegradationStep] | None = None,
) -> CropPath:
    """Carry the eye from one subject to another inside one shot.

    Splitting the shot instead, which this did first, cuts a continuous take
    to itself: same background, same light, same moment, and the frame jumps
    sideways. That is a jump cut with nothing motivating it.

    The endpoints are measured subject centres, not nine-box guesses. Mapping
    "mid_left" and "mid_right" onto the edges of the frame sent the crop to
    0.192 and 0.808 for two handsets that actually sit at 0.359 and 0.635 --
    overshooting each subject by about a sixth of the frame, so the pan began
    with the first phone half out of frame on the right, crossed empty
    background in the middle, and ended with the second one clipped on the
    left. It also made the move twice as long as it needed to be, which is the
    same bug showing up as speed.
    """

    if target_aspect < source_aspect:
        crop_width = target_aspect / source_aspect
        crop_height = 1.0
    else:
        crop_width = 1.0
        crop_height = source_aspect / target_aspect

    free_x = max(0.0, 1.0 - crop_width)
    y = (1.0 - crop_height) / 2.0

    def centred_on(centre: float, subject_width: float = 0.0) -> float:
        """Frame the subject without travelling past it into background.

        Centring alone is not framing. A subject sitting at 0.72 of the frame
        is centred by pushing the crop's right edge to 0.88 -- and if the
        subject itself ends at 0.83, the last twentieth of the shot is wall.
        The pan stops where the subject's far edge does, with a margin, so it
        arrives on the subject rather than beside it.
        """

        wanted = centre - crop_width / 2.0
        if subject_width > 0.0:
            far = centre + subject_width / 2.0
            # Keep a little air past the subject, no more.
            limit = min(free_x, max(0.0, far + crop_width * CROP_MARGIN - crop_width))
            near = centre - subject_width / 2.0
            floor = max(0.0, near - crop_width * CROP_MARGIN)
            # Both bounds exist to stop the crop drifting off a subject that
            # fills it. A subject narrower than the crop cannot touch both
            # edges, so they contradict -- and resolving that by taking one of
            # them hugs the subject to that edge: a folded Flip ending a pan
            # sat at a tenth of the frame with a third of it wall, which reads
            # as the pan having gone too far. Slack means centred.
            if floor <= limit:
                wanted = min(max(wanted, floor), limit)
        return min(max(wanted, 0.0), free_x)

    start_x = centred_on(from_centre, from_width)
    end_x = centred_on(to_centre, to_width)

    return _with_rest(
        [
            Keyframe(0.0, CropBox(start_x, y, crop_width, crop_height)),
            Keyframe(duration_seconds, CropBox(end_x, y, crop_width, crop_height)),
        ],
        duration_seconds,
        energy,
        clip_id=clip_id,
        degradations=degradations,
    )


# How far the delivered frame may be enlarged past what the source supplies.
#
# These are the cost of each editorial answer, not a rule about which to pick.
# Holding every shot to the sharp figure is a local veto on a question only the
# edit can settle: a coin held against a hinge to show 4.1mm is worth showing
# soft, and a wide establisher is not. The planner says which this is; these
# numbers say what it costs.
# One ceiling, because softening the picture was never the thing worth
# trading. Empty frame is an acceptable outcome; a soft frame is not.
MAX_UPSCALE = 1.35

# Where the subject centre lands vertically within the crop, per intent. The
# thirds figure leaves the air above a subject that a centred frame throws
# away evenly on both sides -- which is why a coin dead-centre in a tall frame
# reads as untouched rather than composed.
PLACEMENT: dict[str, float] = {
    "thirds": 0.618,
    "centre": 0.5,
    "fill": 0.5,
}


def travel_room(
    *,
    source_width: int,
    source_height: int,
    target_aspect: float,
    output_width: int,
    output_height: int,
) -> tuple[float, float]:
    """How far a crop can move across this source, per axis, for free.

    Fractions of the source frame, horizontal first. Zero means a move along
    that axis has nowhere to go.

    The first version of this asked only about shape, and so assumed the crop
    is always the largest one that fits -- full height for a wide source at a
    tall target. That is only forced when the source has no resolution to
    spare. A 4K take delivering 1080x1920 needs a crop just 1920 pixels tall,
    not the 2160 it has, and the 240 left over are room to tilt: 0.111 of the
    frame at no cost at all, and 0.342 if the whole zoom budget goes on it.
    Reported as zero, the one move that take could carry looked impossible.

    What is returned is the free room -- the crop that needs no enlarging --
    because travel and tightening are drawn from the same spare pixels, and a
    number that has already spent the zoom budget is not one the planner can
    combine with `push_room`. A source with nothing spare falls back to the
    largest crop that fits, which is what it was doing before.
    """

    if min(source_width, source_height, output_width, output_height) <= 0:
        return 0.0, 0.0
    if target_aspect <= 0:
        return 0.0, 0.0

    # As small as the delivery, never larger than the frame, and never wider
    # than the frame is either.
    tall = min(float(source_height), float(output_height), source_width / target_aspect)
    wide = tall * target_aspect
    return (
        max(0.0, (source_width - wide) / source_width),
        max(0.0, (source_height - tall) / source_height),
    )


def zoom_budget(
    *,
    source_width: int,
    source_height: int,
    source_aspect: float,
    target_aspect: float,
    output_width: int,
    output_height: int,
    max_upscale: float = MAX_UPSCALE,
) -> float:
    """The tightest crop this source can supply at this output size.

    Read before choosing a zoom rather than after. A fixed percentage per
    energy level is blind to what it is cropping: the same 28% is nothing on
    a 4K source and unwatchable on a 1080 one.

    Returned as a scale factor -- 1.0 means no room to push at all, 0.4 means
    the crop may shrink to 40% of its base size before the delivered frame is
    being enlarged past the budget.
    """

    if target_aspect < source_aspect:
        base_w = (target_aspect / source_aspect) * source_width
        base_h = float(source_height)
    else:
        base_w = float(source_width)
        base_h = (source_aspect / target_aspect) * source_height

    # The crop may shrink until delivering it would enlarge past the budget.
    limit_w = (output_width / max_upscale) / max(base_w, 1e-6)
    limit_h = (output_height / max_upscale) / max(base_h, 1e-6)
    return min(1.0, max(limit_w, limit_h, 0.05))


def achieved_upscale(
    crop: CropBox,
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> float:
    """How much this crop has to be enlarged to fill the output.

    A number, reported alongside the cut. Sharpness is measurable, so nobody
    should be asked to judge it from a preview -- least of all to tell a soft
    proxy apart from a genuinely over-enlarged shot, which at preview
    resolution look identical.
    """

    pixels_w = max(crop.width * source_width, 1e-6)
    pixels_h = max(crop.height * source_height, 1e-6)
    return max(output_width / pixels_w, output_height / pixels_h)


def build_tilt_path(
    observations: list[Observation],
    *,
    source_aspect: float,
    target_aspect: float,
    energy: CameraEnergy = "calm",
    clip_id: str = "",
    degradations: list[DegradationStep] | None = None,
) -> CropPath:
    """Follow a subject up or down the frame.

    A watch lowered into a tank, a handset lifted off a table: the motion that
    carries the shot is vertical, and a crop that only ever moves sideways
    cannot follow it. Worse, it reports the subject as having no horizontal
    spread and holds -- the shot is described as needing no camera work when
    what actually happened is that its movement was on an axis nothing looked
    at.

    Converting a wide source to a tall one leaves the crop at full height with
    nowhere to go vertically. That is a real limit rather than a failure to
    try, so it is stated as one.
    """

    if not observations:
        raise ValueError("a tilt needs at least one observation")

    if target_aspect < source_aspect:
        crop_width = target_aspect / source_aspect
        crop_height = 1.0
    else:
        crop_width = 1.0
        crop_height = source_aspect / target_aspect

    free_y = max(0.0, 1.0 - crop_height)
    centres_y = [observation.centre_y for observation in observations]
    spread = _percentile(centres_y, 0.95) - _percentile(centres_y, 0.05)
    x = min(
        max(
            sum(o.centre_x for o in observations) / len(observations)
            - crop_width / 2.0,
            0.0,
        ),
        max(0.0, 1.0 - crop_width),
    )

    def at(centre_y: float) -> CropBox:
        y = min(max(centre_y - crop_height / 2.0, 0.0), free_y)
        return CropBox(x=x, y=y, width=crop_width, height=crop_height)

    if free_y <= 0.0 or spread < DEADBAND:
        if degradations is not None:
            degradations.append(
                DegradationStep(
                    clip_id=clip_id,
                    ladder="static_on_subject",
                    trigger=(
                        "a tilt was planned but the crop already fills the "
                        "frame vertically, leaving nowhere to move"
                        if free_y <= 0.0
                        else "a tilt was planned but the subject does not "
                        "move vertically in this shot"
                    ),
                    measured={
                        "vertical_spread_vw": round(spread, 4),
                        "free_travel_vw": round(free_y, 4),
                    },
                )
            )
        return CropPath(
            [Keyframe(observations[0].seconds, at(_percentile(centres_y, 0.5)))]
        )

    raw = [
        Keyframe(observation.seconds, at(observation.centre_y))
        for observation in observations
    ]
    limited, _ = _limit_speed(raw, ENERGY_LIMITS[energy])
    return CropPath(limited)


def build_zoom_path(
    *,
    source_aspect: float,
    target_aspect: float,
    duration_seconds: float,
    direction: str,
    centre_x: float = 0.5,
    centre_y: float = 0.5,
    # Where the subject is over time, when that was measured. A push is the
    # one move whose frame shrinks, so a subject that walks while it happens
    # is being squeezed out of a closing window -- aiming at the mean of five
    # samples puts it against one edge at the start and outside the frame at
    # the end. Given a track, the push follows.
    track: list[tuple[float, float, float]] | None = None,
    energy: CameraEnergy = "calm",
    framing: str = "thirds",
    budget: float = 0.0,
    subject_height: float | None = None,
    clip_id: str = "",
    degradations: list[DegradationStep] | None = None,
) -> CropPath:
    """Close in on something, or open out from it.

    The most-asked-for move on this material by some distance -- four of
    eleven shots in one selection -- because a product film is mostly about
    looking closer at things rather than chasing them.

    The crop shrinks toward the subject for a push in and grows away for a
    pull out. How far it travels is bounded by how much frame there is: a crop
    that is already most of the source has nowhere to go, and forcing one
    would soften the picture rather than move it.
    """

    if target_aspect < source_aspect:
        base_width = target_aspect / source_aspect
        base_height = 1.0
    else:
        base_width = 1.0
        base_height = source_aspect / target_aspect

    # What the shot wants: enough of a push that the subject reads, with room
    # around it. What the source allows: whatever keeps the delivered frame
    # inside its enlargement budget. Take the more conservative.
    wanted = {"calm": 0.90, "active": 0.82, "dynamic": 0.72}[energy]
    if subject_height is not None and subject_height > 0.0 and framing == "fill":
        # Only a fill intent chases the subject's size. The others accept the
        # frame the source gives and place the subject inside it.
        wanted = min(wanted, max(0.2, subject_height / 0.66))
    allowed = budget if budget > 0.0 else 0.35
    tight = max(wanted, allowed)

    if degradations is not None and wanted < allowed - 1e-6:
        degradations.append(
            DegradationStep(
                clip_id=clip_id,
                ladder="reduced_zoom",
                trigger=(
                    "the source cannot supply the push this shot wants without "
                    "enlarging the delivered frame past its budget"
                ),
                measured={
                    "wanted_scale": round(wanted, 4),
                    "allowed_scale": round(allowed, 4),
                    "subject_height_vw": round(subject_height or 0.0, 4),
                },
            )
        )

    wide = (base_width, base_height)
    close = (base_width * tight, base_height * tight)
    first, last = (wide, close) if direction == "push_in" else (close, wide)

    def box(size: tuple[float, float], at_x: float, at_y: float) -> CropBox:
        width, height = size
        # Put the subject on the intended line of the crop rather than in the
        # middle of it. A centred subject splits its negative space evenly
        # above and below, which reads as an untouched frame; placing it on
        # the lower third gathers that space into one piece of air above.
        share = PLACEMENT.get(framing, 0.5)
        x = min(max(at_x - width / 2.0, 0.0), max(0.0, 1.0 - width))
        y = min(max(at_y - height * share, 0.0), max(0.0, 1.0 - height))
        return CropBox(x, y, width, height)

    moving = [one for one in (track or []) if 0.0 <= one[0] <= duration_seconds]
    if len(moving) < 2:
        # A push that starts on the first frame and ends on the last reads as
        # cut out of a longer one, the same as a pan that never rests.
        return _with_rest(
            [
                Keyframe(0.0, box(first, centre_x, centre_y)),
                Keyframe(duration_seconds, box(last, centre_x, centre_y)),
            ],
            duration_seconds,
            energy,
            clip_id=clip_id,
            degradations=degradations,
        )

    moving.sort()
    spread = max(
        max(x for _, x, _ in moving) - min(x for _, x, _ in moving),
        max(y for _, _, y in moving) - min(y for _, _, y in moving),
    )
    if spread < DEADBAND:
        # Measured and barely moving. A track that only jitters would make the
        # push wander, which reads worse than aiming at one point does.
        return CropPath(
            [
                Keyframe(0.0, box(first, centre_x, centre_y)),
                Keyframe(duration_seconds, box(last, centre_x, centre_y)),
            ]
        )

    # One keyframe per measurement: the size ramps across the shot, the aim
    # comes from where the subject was at that moment.
    raw: list[Keyframe] = []
    for seconds, at_x, at_y in moving:
        share = seconds / duration_seconds if duration_seconds > 0 else 0.0
        size = (
            first[0] + (last[0] - first[0]) * share,
            first[1] + (last[1] - first[1]) * share,
        )
        raw.append(Keyframe(seconds, box(size, at_x, at_y)))
    if raw[0].seconds > 1e-6:
        raw.insert(0, Keyframe(0.0, box(first, moving[0][1], moving[0][2])))
    if raw[-1].seconds < duration_seconds - 1e-6:
        raw.append(Keyframe(duration_seconds, box(last, moving[-1][1], moving[-1][2])))

    limited, _ = _limit_speed(raw, ENERGY_LIMITS[energy])
    if degradations is not None:
        degradations.append(
            DegradationStep(
                clip_id=clip_id,
                ladder="other",
                ladder_other="zoom_followed_subject",
                trigger=(
                    "the subject moved while the frame was closing, so the "
                    "push follows it rather than aiming at where it averaged"
                ),
                measured={
                    "subject_spread_vw": round(spread, 4),
                    "samples": float(len(moving)),
                },
            )
        )
    return CropPath(_smooth(limited))


def visible_fraction(crop: CropBox, observation: Observation) -> float:
    """How much of the subject the crop actually contains.

    Centring on a subject says nothing about whether it fits. A subject 0.55
    of the frame wide inside a 0.316 crop is 57% visible however perfectly it
    is centred, and the path builder reported that shot as executed with no
    degradation -- the crop was exactly where it was asked to be, and half the
    product was outside it.
    """

    left = max(crop.x, observation.centre_x - observation.width / 2.0)
    right = min(crop.x + crop.width, observation.centre_x + observation.width / 2.0)
    across = max(0.0, right - left) / max(observation.width, 1e-9)

    top = max(crop.y, observation.centre_y - observation.height / 2.0)
    bottom = min(
        crop.y + crop.height, observation.centre_y + observation.height / 2.0
    )
    down = max(0.0, bottom - top) / max(observation.height, 1e-9)
    return across * down


def _report_fit(
    path: CropPath,
    observations: list[Observation],
    *,
    clip_id: str,
    min_visible: float,
    degradations: list[DegradationStep] | None,
) -> CropPath:
    """Record how much of the subject the finished path actually holds.

    Every exit from the path builder passes through here. Placing this only on
    the moving branch, as it was first, skipped exactly the shots that need it
    -- a subject too big for the crop usually is not moving, so the static
    early return carried it straight past the check.
    """

    if degradations is None or not observations:
        return path

    worst = min(
        visible_fraction(
            path.keyframes[
                min(index, len(path.keyframes) - 1)
            ].crop,
            observation,
        )
        for index, observation in enumerate(observations)
    )
    if worst < min_visible:
        degradations.append(
            DegradationStep(
                clip_id=clip_id,
                ladder="other",
                ladder_other="subject_larger_than_crop",
                trigger=(
                    "the subject does not fit the target aspect at any point "
                    "in this shot, so it is framed as fully as the crop allows"
                ),
                measured={
                    "worst_visible_fraction": round(worst, 4),
                    "requested_min_visible": min_visible,
                    "subject_width_vw": round(
                        max(o.width for o in observations), 4
                    ),
                    "crop_width_vw": round(path.keyframes[0].crop.width, 4),
                },
            )
        )
    return path


def build_crop_path(
    observations: list[Observation],
    *,
    source_aspect: float,
    target_aspect: float,
    energy: CameraEnergy = "calm",
    framing: str = "thirds",
    clip_id: str = "",
    min_visible: float = 0.85,
    degradations: list[DegradationStep] | None = None,
    planned_to_move: bool = True,
) -> CropPath:
    """Turn subject observations into a crop that follows them.

    Returns a single-keyframe path when the subject barely moves. That is a
    hold, and when a follow was asked for it is also a substitution, so it is
    recorded as one.
    """

    if not observations:
        raise ValueError("a crop path needs at least one observation")

    limits = ENERGY_LIMITS[energy]
    if target_aspect < source_aspect:
        crop_width = target_aspect / source_aspect
        crop_height = 1.0
    else:
        crop_width = 1.0
        crop_height = source_aspect / target_aspect

    free_x = max(0.0, 1.0 - crop_width)
    free_y = max(0.0, 1.0 - crop_height)

    # Judge the movement on the raw observations, before anything is filtered.
    centres_x = [observation.centre_x for observation in observations]
    centres_y = [observation.centre_y for observation in observations]
    spread_x = _percentile(centres_x, 0.95) - _percentile(centres_x, 0.05)
    spread_y = _percentile(centres_y, 0.95) - _percentile(centres_y, 0.05)
    spread = max(spread_x if free_x > 0 else 0.0,
                 spread_y if free_y > 0 else 0.0)

    def crop_at(
        centre_x: float, centre_y: float, lead_x: float = 0.0,
        lead_y: float = 0.0,
    ) -> CropBox:
        x = centre_x + lead_x - crop_width / 2.0
        if free_x > 0.0:
            x = min(max(x, free_x * CROP_MARGIN), free_x * (1.0 - CROP_MARGIN))
        else:
            x = 0.0
        share = PLACEMENT.get(framing, 0.5)
        y = min(
            max(centre_y + lead_y - crop_height * share, 0.0), free_y
        )
        return CropBox(x=x, y=y, width=crop_width, height=crop_height)

    def hold(trigger: str, measured: dict[str, float]) -> CropPath:
        """Frame where the subject spent its time, and say why not to follow.

        Both reasons to hold end here: the subject barely moved, or it moved
        and came back. Whether that is a substitution depends on what was
        asked for, and until the plan said so out loud there was no way to
        tell -- so every shot that settled on one thing came through here and
        was recorded as a follow that had been downgraded. Fourteen of them
        in a sixteen-shot film, each one a note saying the frame held still
        on a shot whose plan was to hold still.

        That is not free. Every degradation is a question the shot reviewer
        has to adjudicate, and the shot reviewer is a paid call.
        """

        if degradations is not None and planned_to_move:
            degradations.append(
                DegradationStep(
                    clip_id=clip_id,
                    ladder="static_on_subject",
                    trigger=trigger,
                    measured=measured,
                )
            )
        return _report_fit(
            CropPath(
                [
                    Keyframe(
                        observations[0].seconds,
                        crop_at(
                            _percentile(centres_x, 0.5),
                            _percentile(centres_y, 0.5),
                        ),
                    )
                ]
            ),
            observations,
            clip_id=clip_id,
            min_visible=min_visible,
            degradations=degradations,
        )

    if spread < DEADBAND or (free_x <= 0.0 and free_y <= 0.0):
        return hold(
            "a follow was planned but the subject does not move in this shot"
            if free_x > 0.0 or free_y > 0.0
            else "a follow was planned but the crop already fills the frame "
            "on both axes, leaving nowhere to move",
            {
                "subject_spread_x": round(spread_x, 4),
                "subject_spread_y": round(spread_y, 4),
                "deadband_vw": DEADBAND,
                "free_travel_x": round(free_x, 4),
                "free_travel_y": round(free_y, 4),
            },
        )

    def available_delta(earlier: Observation, later: Observation) -> tuple[float, float]:
        return (
            later.centre_x - earlier.centre_x if free_x > 0.0 else 0.0,
            later.centre_y - earlier.centre_y if free_y > 0.0 else 0.0,
        )

    wandered = sum(
        math.hypot(*available_delta(earlier, later))
        for earlier, later in zip(observations, observations[1:])
    )
    net = math.hypot(*available_delta(observations[0], observations[-1]))
    directness = net / wandered if wandered > 1e-9 else 0.0
    if directness < MIN_DIRECTNESS:
        return hold(
            "a follow was planned but the subject returned to where it "
            "started, so following it would read as a wobble rather than a "
            "move",
            {
                "directness": round(directness, 4),
                "threshold": MIN_DIRECTNESS,
                "net_displacement_vw": round(net, 4),
                "total_wander_vw": round(wandered, 4),
            },
        )

    raw: list[Keyframe] = []
    for index, observation in enumerate(observations):
        # Lead the subject in its direction of travel so it is not pinned to
        # the trailing edge of the frame. The size of that lead scales with
        # how fast the subject is actually going.
        #
        # Taking only the sign of the drift, as this did first, gives a
        # subject creeping by a thousandth of a frame the same full-magnitude
        # lead as one crossing it -- and then swings the crop by twice that
        # the moment the subject pauses or drifts back. A shot of a nearly
        # still product measured a subject travel of 0.100 and a crop travel
        # of 0.158, reversing once on the way. It reads as the camera
        # wobbling, because that is what it is.
        if index + 1 < len(observations):
            neighbour = observations[index + 1]
        else:
            neighbour = observations[index - 1]
        span = max(abs(neighbour.seconds - observation.seconds), 1e-6)
        drift_x, drift_y = available_delta(observation, neighbour)
        if index + 1 >= len(observations):
            drift_x, drift_y = -drift_x, -drift_y
        speed_x, speed_y = drift_x / span, drift_y / span
        speed = math.hypot(speed_x, speed_y)
        # Full lead at the energy's top speed, proportionally less below it,
        # and nothing at all when the subject is effectively parked.
        share = min(1.0, speed / limits["max_speed"])
        lead = limits["lead"] * min(crop_width, crop_height) * share
        lead_x = lead * speed_x / speed if speed > 1e-9 else 0.0
        lead_y = lead * speed_y / speed if speed > 1e-9 else 0.0
        raw.append(Keyframe(
            observation.seconds,
            crop_at(observation.centre_x, observation.centre_y, lead_x, lead_y),
        ))

    limited, peak_speed = _limit_speed(raw, limits)
    path = CropPath(_smooth(limited))

    if degradations is not None and peak_speed > limits["max_speed"]:
        degradations.append(
            DegradationStep(
                clip_id=clip_id,
                ladder="slower_follow",
                trigger=(
                    "the subject outran the camera budget for this energy, so "
                    "the follow was held to its limit"
                ),
                measured={
                    "observed_speed_vw_per_s": round(peak_speed, 4),
                    "limit_vw_per_s": limits["max_speed"],
                    "subject_spread_x": round(spread_x, 4),
                    "subject_spread_y": round(spread_y, 4),
                },
            )
        )
    return path


def _eased(
    start: float, end: float, at: float, span: float, *, clock: str = "t"
) -> str:
    """A smoothstep ramp between two values, as an ffmpeg expression.

    Linear interpolation, which this used first, gives constant velocity with
    an instant start and an instant stop. Nothing physical moves that way and
    nothing shot by hand looks that way, which is most of what reads as
    mechanical in a generated move. Smoothstep eases both ends, so the camera
    takes up the move and sets it down.

    This is also where the acceleration budget finally does something:
    ENERGY_LIMITS carried a max_accel that nothing referenced, because a
    constant-velocity ramp has no acceleration to bound.
    """

    delta = end - start
    # 3u^2 - 2u^3 over the segment's own normalised time.
    unit = f"clip(({clock}-{at:.3f})/{span:.6f},0,1)"
    return f"{start:.3f}+({delta:.3f})*({unit}*{unit}*(3-2*{unit}))"


def interpolate_crop_keyframes(
    keys: list[dict], seconds: float, *, ease: bool = True
) -> dict | None:
    """Numeric twin of the ffmpeg crop path used by evidence consumers."""

    if not keys:
        return None
    before, after = keys[0], keys[-1]
    for left, right in zip(keys, keys[1:]):
        if float(left["at"]) <= seconds <= float(right["at"]):
            before, after = left, right
            break
    span = float(after["at"]) - float(before["at"])
    share = 0.0 if span <= 0 else max(
        0.0, min(1.0, (seconds - float(before["at"])) / span)
    )
    if ease:
        share = share * share * (3.0 - 2.0 * share)
    return {
        key: float(before[key])
        + (float(after[key]) - float(before[key])) * share
        for key in ("x", "y", "w", "h")
    }


def retime_crop_path(
    path: CropPath, *, old_in_seconds: float, new_in_seconds: float,
    new_duration_seconds: float,
) -> CropPath:
    """Trim or extend a recorded path without inventing new tracking.

    A crop path is relative to its old shot, while a recut changes the source
    window. Shift the original interpolation domain through source time and
    retain its easing; keys outside the new window are deliberate and the
    evaluator holds the nearest endpoint. This preserves a follow for
    gain-only recuts and honest head or tail trims instead of silently
    rebuilding it as a static crop.
    """

    del new_duration_seconds  # the renderer naturally holds beyond last key
    shift = new_in_seconds - old_in_seconds
    # Keep the original interpolation domain. Cropping the old curve and
    # easing again over a shorter span changes its velocity and even its
    # midpoint. Negative or post-out key times are intentional: evaluating
    # the same curve at new-relative t is exactly old-relative t + shift;
    # ffmpeg holds the nearest endpoint outside the measured range.
    return CropPath([
        Keyframe(frame.seconds - shift, frame.crop)
        for frame in path.keyframes
    ])


def _axis_expression(
    path: CropPath, pick, scale: int, *, ease: bool = True,
    clock: str = "t",
) -> str:
    """Piecewise expression for one axis over the whole shot."""

    first = pick(path.keyframes[0].crop) * scale
    expression = f"{first:.3f}"
    for earlier, later in zip(path.keyframes, path.keyframes[1:]):
        start = pick(earlier.crop) * scale
        end = pick(later.crop) * scale
        span = max(later.seconds - earlier.seconds, 1e-6)
        ramp = (
            _eased(start, end, earlier.seconds, span, clock=clock)
            if ease
            else f"{start:.3f}+({end - start:.3f})*({clock}-{earlier.seconds:.3f})/{span:.6f}"
        )
        expression = (
            f"if(between({clock},{earlier.seconds:.3f},{later.seconds:.3f}),"
            f"{ramp},{expression})"
        )
    last = path.keyframes[-1]
    return (
        f"if(gte({clock},{last.seconds:.3f}),{pick(last.crop) * scale:.3f},"
        f"{expression})"
    )


def ffmpeg_crop_expression(
    path: CropPath, width: int, height: int, *, clock: str = "t"
) -> tuple[str, str, str, str]:
    """Render a path as a crop filter's four arguments.

    Returns expressions for width, height, x and y. A static path yields
    plain numbers. Anything that moves -- across, down, or in -- yields eased
    expressions in `t`, so one filter carries the whole move.
    """

    first = path.keyframes[0].crop
    if path.is_static:
        x, y, crop_w, crop_h = first.to_pixels(width, height)
        return str(crop_w), str(crop_h), str(x), str(y)

    # Crop extents must stay even for chroma subsampling, and must not run off
    # the frame at any point in the ramp.
    w_expr = f"floor(min({_axis_expression(path, lambda c: c.width, width, clock=clock)},{width})/2)*2"
    h_expr = f"floor(min({_axis_expression(path, lambda c: c.height, height, clock=clock)},{height})/2)*2"
    x_expr = f"floor(max(0,min({_axis_expression(path, lambda c: c.x, width, clock=clock)},{width}-out_w))/2)*2"
    y_expr = f"floor(max(0,min({_axis_expression(path, lambda c: c.y, height, clock=clock)},{height}-out_h))/2)*2"
    return w_expr, h_expr, x_expr, y_expr


def ffmpeg_crop_filters(
    path: CropPath, width: int, height: int,
    output_size: tuple[int, int], *, output_fps: int = 30,
    clock_offset_seconds: float = 0.0,
) -> list[str]:
    """Build filters that really execute pan, tilt *and* zoom per frame.

    FFmpeg's crop x/y expressions are evaluated per frame, but crop w/h are
    normally configured only once. Feeding a changing width into `crop`
    therefore rendered pans correctly while silently freezing push-ins and
    pull-outs at their opening size. For a zoom we map the changing authored
    rectangle to a fixed canvas with the perspective filter; all four
    authored coordinates are consequently evaluated on the same clock.
    """

    output_width, output_height = output_size
    first = path.keyframes[0].crop
    zooms = any(
        abs(frame.crop.width - first.width) > 1e-6
        or abs(frame.crop.height - first.height) > 1e-6
        for frame in path.keyframes[1:]
    )
    if not zooms:
        clock = (
            "t" if abs(clock_offset_seconds) < 1e-9
            else f"(t-{clock_offset_seconds:.6f})"
        )
        w_expr, h_expr, x_expr, y_expr = ffmpeg_crop_expression(
            path, width, height, clock=clock,
        )
        return [
            f"crop=w='{w_expr}':h='{h_expr}':x='{x_expr}':y='{y_expr}'",
            f"scale={output_width}:{output_height}",
        ]

    clock = f"(on/{output_fps}-{clock_offset_seconds:.6f})"
    left = _axis_expression(path, lambda crop: crop.x, 1, clock=clock)
    top = _axis_expression(path, lambda crop: crop.y, 1, clock=clock)
    right = _axis_expression(
        path, lambda crop: crop.x + crop.width, 1, clock=clock
    )
    bottom = _axis_expression(
        path, lambda crop: crop.y + crop.height, 1, clock=clock
    )
    # perspective(source) maps the authored rectangle to the four corners of
    # a fixed-size source canvas and supports per-frame corner expressions.
    # The following scale changes only that fixed canvas into delivery pixels,
    # so the encoder never sees a changing frame size.
    return [
        "perspective="
        f"x0='({left})*W':y0='({top})*H':"
        f"x1='({right})*W':y1='({top})*H':"
        f"x2='({left})*W':y2='({bottom})*H':"
        f"x3='({right})*W':y3='({bottom})*H':"
        "sense=source:eval=frame:interpolation=cubic",
        f"scale={output_width}:{output_height}",
    ]


def observations_from_sam(
    track,
    *,
    clip_start_seconds: float,
    accept_states: frozenset[str] = frozenset({"tracked"}),
    semantic_anchors: tuple[tuple[float, tuple[float, float, float, float]], ...] = (),
    require_identity_validation: bool = False,
) -> tuple[list[Observation], dict[str, int]]:
    """Read a propagated track as subject observations.

    Sampling five frames and interpolating, which this replaces, describes a
    three-second shot at one point every 0.6 seconds and guesses the rest. A
    subject that crosses the frame and returns inside one sample interval is
    invisible to that.

    Samples the tracker itself flagged as lost are dropped rather than
    averaged in. A mask that has drifted onto the background still reports a
    box, and feeding that to the crop is how a camera ends up following the
    wrong thing confidently. The counts come back so the caller can say how
    much of the shot was actually tracked.

    The seed still comes from Gemini: which of two similar handsets is meant
    is not a question a tracker can answer.
    """

    observations: list[Observation] = []
    states: dict[str, int] = {}
    samples = list(getattr(track, "samples", []) or [])

    def overlap(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> float:
        x0 = max(left[0], right[0])
        y0 = max(left[1], right[1])
        x1 = min(left[2], right[2])
        y1 = min(left[3], right[3])
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = (
            max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
            + max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
            - intersection
        )
        return intersection / union if union > 0 else 0.0

    validated_interval: tuple[float, float] | None = None
    if require_identity_validation:
        # SAM propagates geometry, not identity. A reference-critical track is
        # usable only where independent exact-frame grounding brackets it and
        # agrees with its geometry. This prevents a confident mask that has
        # switched to a lookalike from becoming an equally confident crop.
        matched_times: list[float] = []
        for anchor_time, anchor_box in semantic_anchors:
            candidates = [
                sample for sample in samples
                if sample.derived_tracking_box
                and abs(sample.analysis_sample_time_ms / 1000.0 - anchor_time)
                <= max(0.51, 1.0 / max(float(getattr(track, "analysis_fps", 1.0)), 0.1))
            ]
            if not candidates:
                continue
            nearest = min(
                candidates,
                key=lambda sample: abs(
                    sample.analysis_sample_time_ms / 1000.0 - anchor_time
                ),
            )
            raw_box = nearest.derived_tracking_box
            sam_box = (
                float(raw_box[0]) / 1000.0,
                float(raw_box[1]) / 1000.0,
                float(raw_box[2]) / 1000.0,
                float(raw_box[3]) / 1000.0,
            )
            if overlap(sam_box, anchor_box) >= 0.35:
                matched_times.append(anchor_time)
        if len(matched_times) < 2:
            states["identity_unverified"] = len(samples)
            return [], states
        validated_interval = (min(matched_times), max(matched_times))
    for sample in samples:
        state = getattr(sample.tracking_state, "value", str(sample.tracking_state))
        if state not in accept_states:
            states[state] = states.get(state, 0) + 1
            continue
        semantic = getattr(
            getattr(sample, "semantic_identity_status", ""),
            "value",
            str(getattr(sample, "semantic_identity_status", "")),
        )
        if require_identity_validation:
            if semantic in {"revalidation_required", "revalidation_failed"}:
                states["identity_rejected"] = states.get("identity_rejected", 0) + 1
                continue
            at_source = sample.analysis_sample_time_ms / 1000.0
            if validated_interval is None or not (
                validated_interval[0] <= at_source <= validated_interval[1]
            ):
                states["identity_unbracketed"] = states.get(
                    "identity_unbracketed", 0
                ) + 1
                continue
        # State counters are a partition of physical samples. Previously a
        # reference-critical sample counted once as ``tracked`` and again as
        # ``identity_unbracketed``/``identity_rejected``. The pipeline's
        # tracked/sum(states) quorum could therefore pass a long track with
        # only its first two frames semantically bracketed.
        states[state] = states.get(state, 0) + 1
        box = sample.derived_tracking_box
        if not box or len(box) != 4:
            continue
        # The old package reports boxes as x-first 0..1000.
        x0, y0, x1, y1 = (value / 1000.0 for value in box)
        width, height = abs(x1 - x0), abs(y1 - y0)
        if width <= 0 or height <= 0:
            continue
        at = sample.analysis_sample_time_ms / 1000.0 - clip_start_seconds
        if at < 0:
            continue
        try:
            observations.append(
                Observation(
                    seconds=at,
                    centre_x=(x0 + x1) / 2.0,
                    centre_y=(y0 + y1) / 2.0,
                    width=width,
                    height=height,
                )
            )
        except OutOfFrame:
            continue
    return observations, states
