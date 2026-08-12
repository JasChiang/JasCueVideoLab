"""Local release proofs between editorial planning and expensive rendering."""

from __future__ import annotations

from montagewright.grounding import BeatGrid, ground_timeline
from montagewright.schema import EDL


def rhythm_motion_faults(
    authored: EDL, candidate: EDL, grid: BeatGrid | None
) -> tuple[str, ...]:
    """Prove that rhythm did not break a content or motion commitment.

    Gemini may choose a musical relationship.  Exact cue availability,
    usable source windows and minimum move duration are local facts, so they
    are checked here before tracking or rendering spends any more work.
    """

    original = {clip.clip_id: clip for clip in authored.clips}
    timeline = ground_timeline(candidate, grid)
    faults: list[str] = []
    for entry in timeline.clips:
        clip = entry.clip
        if entry.move_too_short:
            faults.append(f"{clip.clip_id}: {entry.move_too_short}")
        requested = clip.music_sync.sync_to
        if requested and entry.landed_on != requested:
            faults.append(
                f"{clip.clip_id}: requested music cue {requested!r} did not "
                f"land ({entry.note or 'no matching local cue'})"
            )
        before = original.get(clip.clip_id)
        before_move = (
            before.reframe.editorial_intent
            if before is not None and before.reframe is not None else "hold"
        )
        if before is None or before_move != "use_source_motion":
            continue
        required = before.approx_out_seconds - before.approx_in_seconds
        # Not past the end of the take. A span can outrun its own usable
        # window by a few milliseconds -- the two are measured by different
        # passes -- and then the feasibility clamp trims it, correctly, and
        # this refused the timeline over ten thousandths of a second. What
        # rhythm may not do is choose to shorten the move; what the material
        # cannot supply is not rhythm's doing and is already reported as a
        # note on the shot.
        window = before.usable_window
        if window is not None:
            required = min(
                required, max(0.0, window[1] - clip.approx_in_seconds)
            )
        if entry.duration_seconds < required - 1e-6:
            faults.append(
                f"{clip.clip_id}: native source motion needs its authored "
                f"{required:.3f}s span but rhythm left {entry.duration_seconds:.3f}s"
            )
    return tuple(dict.fromkeys(faults))
