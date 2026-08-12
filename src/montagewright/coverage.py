"""Prove that every second of an edit has an editorial reason to exist.

Duration is not coverage.  A planner can make nine individually plausible
talking-head shots add up to sixty seconds by leaving a second after every
answer; the arithmetic passes while the film repeatedly appears to stop.
This module is deliberately format-agnostic: narrative, synchronous sound,
visible action, reactions, transitions and end holds are all evidence, and
the same audit runs before rhythm and again on the resolved EDL.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from montagewright.schema import EDL


TARGET_TOLERANCE_SECONDS = 0.5
TARGET_TOLERANCE_FRACTION = 0.02
SPEAKER_POST_ROLL_SECONDS = 0.30

# Visual-only time permitted by a role.  `None` means the role itself carries
# the full visual beat; the material still has to pass the ordinary action,
# usable-window and static-hold audits elsewhere.
VISUAL_ONLY_LIMITS: dict[str, float | None] = {
    "speaker": SPEAKER_POST_ROLL_SECONDS,
    "illustrative_broll": 3.00,
    "reaction": 1.50,
    "transition": 0.50,
    "establishing": 3.00,
    "punchline_hold": 1.20,
    "end_hold": 1.50,
    "title_read": 5.00,
    # A role name is not proof. Longer action is authorised per shot from
    # measured action/motion windows and persisted as coverage_claim_seconds.
    "primary_action": 3.00,
    # A montage is made of shots; the label cannot turn one static minute
    # into sixty seconds of evidence. Slow visual action uses primary_action.
    "music_montage": 4.00,
}

# These roles describe bounded pauses rather than content whose minimum
# duration has to be inferred from speech, readable copy or measured action.
# A model overshooting one of these ceilings can therefore be repaired
# monotonically: removing the unsupported tail cannot cut a sentence or an
# action.  If the shorter sequence no longer reaches the delivery target, the
# ordinary coverage audit asks selection for more real content instead of
# silently stretching another shot.
REPAIRABLE_VISUAL_HOLD_ROLES = frozenset({
    "reaction", "transition", "punchline_hold", "end_hold",
})

_SPEECH = re.compile(
    r"^`(?P<id>[^`]+)`\s+"
    r"(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)s"
)
_ACTION = re.compile(
    r"(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)s$"
)


@dataclass(frozen=True)
class CoverageEntry:
    clip_id: str
    picture_role: str
    seconds: float
    audio_seconds: float
    visual_only_seconds: float
    supported_seconds: float


@dataclass(frozen=True)
class CoverageAudit:
    duration_seconds: float
    supported_seconds: float
    target_seconds: float
    entries: tuple[CoverageEntry, ...]
    faults: tuple[str, ...]

    @property
    def unsupported_seconds(self) -> float:
        return max(0.0, self.duration_seconds - self.supported_seconds)


class TimelineCoverageError(ValueError):
    """The edit has timeline seconds for which no content evidence exists."""


def repair_bounded_visual_holds(
    chosen: dict[str, Any], commitments: Any | None = None,
    material: Iterable[Any] | None = None,
) -> tuple[str, ...]:
    """Clamp unsupported pure-visual tails before asking Gemini to replan.

    The function deliberately excludes speaker, title/read, B-roll and action
    roles.  Those need source evidence and may require a structural choice;
    only semantically bounded holds are safe to shorten deterministically.
    """

    repairs: list[str] = []
    shots = chosen.get("shots") or []
    by_source = {
        str(item.source_id): item for item in (material or ())
    }
    # Without canonical span lengths this small selection-stage helper cannot
    # prove that a top-level narrative assignment has ended before a visual
    # hold.  Be conservative: any independent audio keeps duration decisions
    # in the structural planner, where the full clock is available.
    has_independent_audio = bool(chosen.get("audio_assignments"))
    for index, shot in enumerate(shots):
        role = str(shot.get("picture_role") or "")
        if role not in REPAIRABLE_VISUAL_HOLD_ROLES:
            continue
        if (
            has_independent_audio
            or str(shot.get("audio_role") or "discard") != "discard"
            or bool(shot.get("audio_completion"))
        ):
            continue
        limit = VISUAL_ONLY_LIMITS[role]
        requested = max(0.0, float(shot.get("seconds_needed") or 0.0))
        if limit is None or requested <= limit + 1e-9:
            continue
        from montagewright.candidate_commitments import (
            minimum_supported_seconds_for_shot,
        )

        minimum = minimum_supported_seconds_for_shot(shot, commitments)
        supported = float(limit)
        item = by_source.get(str(shot.get("source_id") or ""))
        if item is not None:
            supported = visual_supported_max(
                item,
                role=role,
                source_start=float(shot.get("start_seconds") or 0.0),
                available_seconds=requested,
                motion_role=str(shot.get("source_motion_role") or ""),
            )
        repaired = max(supported, minimum)
        if requested <= repaired + 1e-9:
            continue
        shot["seconds_needed"] = round(repaired, 3)
        # This field is derived by the coverage audit, but clearing a cached
        # value keeps the mutation honest until the audit recomputes it.
        shot.pop("coverage_claim_seconds", None)
        repairs.append(
            f"k{index:02d}: shortened {role} from {requested:.2f}s to "
            f"{repaired:.2f}s; the removed tail had no additional "
            "content evidence"
        )
    return tuple(repairs)


def repair_preferred_unsupported_time(
    chosen: dict[str, Any], audit: CoverageAudit,
    commitments: Any | None = None,
) -> tuple[str, ...]:
    """Remove unsupported visual tails when delivery length is a preference.

    This is deliberately monotonic: it only shortens a shot to time the same
    audit already proved, never below its selected commitment minimum. Speech
    and readable titles remain structural decisions and are not trimmed here.
    """

    repairable = {
        "illustrative_broll", "reaction", "transition", "establishing",
        "punchline_hold", "end_hold", "music_montage", "primary_action",
    }
    repairs: list[str] = []
    shots = chosen.get("shots") or []
    for index, (shot, entry) in enumerate(zip(shots, audit.entries, strict=True)):
        if entry.picture_role not in repairable:
            continue
        requested = float(shot.get("seconds_needed") or 0.0)
        if entry.supported_seconds >= requested - 0.05:
            continue
        from montagewright.candidate_commitments import (
            minimum_supported_seconds_for_shot,
        )

        minimum = minimum_supported_seconds_for_shot(shot, commitments)
        repaired = max(float(entry.supported_seconds), minimum)
        if repaired >= requested - 0.05 or repaired > entry.supported_seconds + 0.05:
            continue
        shot["seconds_needed"] = round(repaired, 3)
        shot.pop("coverage_claim_seconds", None)
        repairs.append(
            f"k{index:02d}: shortened unsupported {entry.picture_role} tail "
            f"from {requested:.2f}s to {repaired:.2f}s for preferred delivery"
        )
    return tuple(repairs)


def _overlap(
    start: float, end: float, intervals: Iterable[tuple[float, float, str]],
    *, source_id: str | None = None,
) -> float:
    pieces = []
    for left, right, source in intervals:
        if source_id is not None and source != source_id:
            continue
        left, right = max(start, left), min(end, right)
        if right > left:
            pieces.append((left, right))
    if not pieces:
        return 0.0
    pieces.sort()
    total, left, right = 0.0, *pieces[0]
    for here_left, here_right in pieces[1:]:
        if here_left <= right:
            right = max(right, here_right)
        else:
            total += right - left
            left, right = here_left, here_right
    return total + right - left


def visual_supported_max(
    item: Any,
    *,
    role: str,
    source_start: float,
    available_seconds: float,
    motion_role: str = "",
    presentation_intent: str = "",
    target_id: str = "none",
) -> float:
    """Return the locally supportable visual duration for one source window.

    Role ceilings are conservative defaults, not genre rules.  Canonical
    action intervals and locally established source motion may prove that a
    picture continues to develop for longer.  The result is always bounded by
    the actual source window, so it is safe to use both before selection and
    when auditing the selected shot.
    """

    available = max(0.0, float(available_seconds))
    base = VISUAL_ONLY_LIMITS.get(str(role), 3.0)
    if base is None:
        return available
    if (
        str(role) == "end_hold"
        and str(presentation_intent) in {"complete_hold", "centered_hold"}
        and str(target_id) not in {"", "none"}
    ):
        # A grounded, deliberately complete final identity is content rather
        # than an arbitrary frozen tail. Geometry and identity are still
        # proven later; this only permits a conservative three-second hold.
        base = max(float(base), 3.0)

    source_end = float(source_start) + available
    intervals: list[tuple[float, float, str]] = []
    # `illustrative_broll` is explicitly carried by another lane (usually
    # narrative). A source action can make the same picture independently
    # useful, but then the planner must call it `primary_action`; silently
    # upgrading the evidence here would make the role contract meaningless.
    if str(role) != "illustrative_broll":
        for described in getattr(item, "action", ()):
            found = _ACTION.search(str(described))
            if found:
                intervals.append((
                    float(found["start"]), float(found["end"]), "action",
                ))
    for measured in getattr(item, "motion", ()):
        if str(getattr(measured, "state", "still")) == "still":
            continue
        intervals.append((
            float(getattr(measured, "starts_seconds", 0.0)),
            float(getattr(measured, "ends_seconds", 0.0)),
            "motion",
        ))
    measured_seconds = _overlap(
        float(source_start), source_end, intervals
    )
    if measured_seconds <= 0.0 and (
        str(motion_role) in {"authored", "subject_follow"}
        or (
            str(role) == "primary_action"
            and bool(getattr(item, "camera_moves", False))
        )
    ):
        measured_seconds = available

    # A small comprehension tail is already part of the existing coverage
    # contract.  It cannot extend beyond source evidence or invent more time.
    evidenced = measured_seconds + 0.60 if measured_seconds > 0.0 else 0.0
    return min(available, max(float(base), evidenced))


def _entry(
    clip_id: str,
    role: str,
    source_id: str,
    start: float,
    seconds: float,
    narrative: list[tuple[float, float, str]],
    audible: list[tuple[float, float, str]] | None = None,
    *, retained_source_audio: bool = False,
    visual_limit_seconds: float | None = None,
    is_last: bool = False,
) -> tuple[CoverageEntry, list[str]]:
    end = start + seconds
    if role == "speaker":
        # Lip-sync is stricter than audible coverage: only narrative from the
        # source visible in this shot proves that a speaker frame is alive.
        audio = _overlap(start, end, narrative, source_id=source_id)
    else:
        audio = _overlap(start, end, audible or narrative)
    if retained_source_audio:
        # Legacy per-picture source audio has no independently measured event
        # window. It is real sound, but the role assertion alone may not
        # justify an arbitrarily long shot. Explicit AudioClips below do.
        audio = max(audio, min(seconds, 3.0))

    limit = (
        visual_limit_seconds
        if visual_limit_seconds is not None
        else VISUAL_ONLY_LIMITS.get(role, 3.0)
    )
    visual = max(0.0, seconds - audio)
    supported = seconds if limit is None else min(seconds, audio + limit)
    faults: list[str] = []
    unsupported = seconds - supported
    if role == "speaker" and audio <= 0.001:
        faults.append(
            f"{clip_id}: speaker picture has no same-source narrative audio "
            "at this timeline position"
        )
    if role == "end_hold" and not is_last:
        faults.append(f"{clip_id}: end_hold is only valid on the final shot")
    if unsupported > 0.05:
        if role == "speaker":
            faults.append(
                f"{clip_id}: speaker picture contains {visual:.2f}s without "
                f"same-source narrative; only {SPEAKER_POST_ROLL_SECONDS:.2f}s "
                "of natural lead/tail is allowed—split a measured reaction "
                "or select more content instead of stretching the speaker"
            )
        elif role == "illustrative_broll":
            faults.append(
                f"{clip_id}: illustrative B-roll outlasts the narrative it "
                f"illustrates by {visual:.2f}s; use primary_action only when "
                "the picture has an independent visual beat"
            )
        else:
            faults.append(
                f"{clip_id}: {role} claims {visual:.2f}s of visual-only time, "
                f"over its {float(limit or 0):.2f}s policy ceiling"
            )
    return CoverageEntry(
        clip_id=clip_id,
        picture_role=role,
        seconds=seconds,
        audio_seconds=audio,
        visual_only_seconds=visual,
        supported_seconds=supported,
    ), faults


def _target_faults(
    duration: float, supported: float, target: float, *, hard: bool = False,
) -> list[str]:
    if target <= 0:
        return []
    tolerance = max(
        TARGET_TOLERANCE_SECONDS, target * TARGET_TOLERANCE_FRACTION
    )
    faults = []
    if hard and duration < target - tolerance:
        faults.append(
            f"timeline is only {duration:.2f}s against the {target:.2f}s "
            f"target; structural selection is short by {target - duration:.2f}s"
        )
    if duration > target + tolerance:
        faults.append(
            f"timeline is {duration:.2f}s against the {target:.2f}s target; "
            f"it is over by {duration - target:.2f}s and must remove or "
            "shorten content rather than ignore the delivery length"
        )
    # Soft targets may be shorter, but a timeline that already reaches its
    # requested duration by adding unsupported seconds is still invalid.
    # Compare evidence with the actual edit here, not with the optional goal.
    if supported < duration - tolerance:
        faults.append(
            f"evidence covers {supported:.2f}s of the {duration:.2f}s edit; "
            f"{duration - supported:.2f}s needs additional narrative, visible "
            "action/reaction, readable graphics, montage or an explicit "
            "shorter delivery—not longer holds"
        )
    elif hard and supported < target - tolerance:
        faults.append(
            f"evidence covers {supported:.2f}s of the {target:.2f}s target; "
            f"{target - supported:.2f}s needs additional narrative, visible "
            "action/reaction, readable graphics, montage or an explicit "
            "shorter delivery—not longer holds"
        )
    return faults


def _visual_claim(item: Any, shot: dict[str, Any], role: str) -> float:
    """Bound a visual claim with locally measured source-time evidence."""

    seconds = max(0.0, float(shot.get("seconds_needed") or 0.0))
    source_start = float(shot.get("start_seconds") or 0.0)
    return visual_supported_max(
        item,
        role=role,
        source_start=source_start,
        available_seconds=seconds,
        motion_role=(
            str(shot.get("source_motion_role") or "")
            if str(shot.get("camera_intent") or "") == "use_source_motion"
            else ""
        ),
        presentation_intent=next((
            str(look.get("presentation_intent") or "")
            for look in shot.get("looks") or []
            if look.get("presentation_intent")
        ), ""),
        target_id=next((
            str(look.get("entity_id") or "none")
            for look in shot.get("looks") or []
            if str(look.get("entity_id") or "none") != "none"
        ), "none"),
    )


def selection_coverage_audit(
    chosen: dict[str, Any], material: list[Any], target_seconds: float,
    *, hard_target: bool = False,
) -> CoverageAudit:
    """Audit the model's nominal selection using canonical audio durations."""

    spans: dict[str, tuple[float, str]] = {}
    for item in material:
        for span_id, start, end in getattr(item, "audio_spans", ()):
            spans[str(span_id)] = (
                float(end) - float(start), str(item.source_id),
            )
        for line in getattr(item, "speech", ()):
            matched = _SPEECH.match(line)
            if not matched or matched["id"] in spans:
                continue
            spans[matched["id"]] = (
                float(matched["end"]) - float(matched["start"]),
                str(item.source_id),
            )

    shots = chosen.get("shots") or []
    durations = [float(shot.get("seconds_needed") or 0.0) for shot in shots]
    starts, cursor = [], 0.0
    for seconds in durations:
        starts.append(cursor)
        cursor += seconds

    narrative: list[tuple[float, float, str]] = []
    for assignment in chosen.get("audio_assignments") or []:
        span = spans.get(str(assignment.get("audio_span_id") or ""))
        index = int(assignment.get("starts_at_shot_index", -1))
        if span is None or not 0 <= index < len(starts):
            continue
        seconds, source_id = span
        at = starts[index] + float(assignment.get("offset_seconds") or 0.0)
        narrative.append((at, at + seconds, source_id))

    by_source = {str(item.source_id): item for item in material}
    entries, faults = [], []
    for index, (shot, start, seconds) in enumerate(
        zip(shots, starts, durations, strict=True)
    ):
        role = str(shot.get("picture_role") or "unknown")
        item = by_source.get(str(shot.get("source_id") or ""))
        claim = _visual_claim(item, shot, role) if item is not None else min(
            seconds, float(VISUAL_ONLY_LIMITS.get(role, 3.0) or seconds)
        )
        shot["coverage_claim_seconds"] = round(claim, 3)
        made, found = _entry(
            f"k{index:02d}",
            role,
            str(shot.get("source_id") or ""),
            start,
            seconds,
            narrative,
            narrative,
            retained_source_audio=str(shot.get("audio_role") or "") in {
                "sync_action", "ambient_texture", "narrative",
            },
            visual_limit_seconds=claim,
            is_last=index == len(shots) - 1,
        )
        entries.append(made)
        faults.extend(found)
    supported = sum(one.supported_seconds for one in entries)
    faults.extend(_target_faults(
        cursor, supported, float(target_seconds or 0), hard=hard_target
    ))
    return CoverageAudit(
        cursor, supported, float(target_seconds or 0), tuple(entries),
        tuple(dict.fromkeys(faults)),
    )


def edl_coverage_audit(
    edl: EDL, target_seconds: float, *, hard_target: bool = False,
) -> CoverageAudit:
    """Audit the resolved picture/audio clock shared by CLI, Web and render."""

    starts, cursor = {}, 0.0
    for clip in edl.clips:
        starts[clip.clip_id] = cursor
        cursor += clip.approx_out_seconds - clip.approx_in_seconds
    narrative = []
    audible = []
    for audio in edl.audio_clips:
        at = starts[audio.starts_at_clip_id] + audio.offset_seconds
        interval = (
            at, at + audio.out_seconds - audio.in_seconds, audio.source_id,
        )
        audible.append(interval)
        if audio.role == "narrative":
            narrative.append(interval)

    entries, faults = [], []
    for index, clip in enumerate(edl.clips):
        seconds = clip.approx_out_seconds - clip.approx_in_seconds
        made, found = _entry(
            clip.clip_id,
            clip.picture_role,
            clip.source_id,
            starts[clip.clip_id],
            seconds,
            narrative,
            audible,
            retained_source_audio=clip.audio_role in {
                "sync_action", "ambient_texture", "narrative",
            },
            visual_limit_seconds=clip.coverage_claim_seconds,
            is_last=index == len(edl.clips) - 1,
        )
        entries.append(made)
        faults.extend(found)
    supported = sum(one.supported_seconds for one in entries)
    faults.extend(_target_faults(
        cursor, supported, float(target_seconds or 0), hard=hard_target
    ))
    return CoverageAudit(
        cursor, supported, float(target_seconds or 0), tuple(entries),
        tuple(dict.fromkeys(faults)),
    )
