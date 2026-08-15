"""Local release proofs between editorial planning and expensive rendering."""

from __future__ import annotations

from montagewright.grounding import BeatGrid, ground_timeline
from montagewright.schema import EDL


def resolved_source_contract_faults(edl: EDL) -> tuple[str, ...]:
    """Validate source-clock promises on the exact windows to be rendered.

    Several deterministic passes legitimately rewrite a clip's source window:
    action snapping, dialogue-safe snapping, musical anchors and speaker
    lip-sync alignment.  The promises attached to that clip do not move with
    the window.  Keeping this check independent of any one rewriter prevents a
    later pass from silently crossing a protected start or usable boundary.
    """

    from montagewright.grounding import _floor_for

    faults: list[str] = []
    for clip in edl.clips:
        source_in = float(clip.approx_in_seconds)
        source_out = float(clip.approx_out_seconds)
        window = clip.usable_window
        if window is not None:
            if source_in < window[0] - 1e-6:
                faults.append(
                    f"{clip.clip_id}: source in-point {source_in:.3f}s precedes "
                    f"usable window {window[0]:.3f}s"
                )
            if source_out > window[1] + 1e-6:
                faults.append(
                    f"{clip.clip_id}: source out-point {source_out:.3f}s exceeds "
                    f"usable window {window[1]:.3f}s"
                )
        for contract in clip.action_contracts:
            if contract.completion_policy in {"may_cut_on_action", "loopable"}:
                continue
            if source_in > contract.source_start_seconds + 1e-6:
                faults.append(
                    f"{clip.clip_id}: source in-point {source_in:.3f}s starts "
                    f"after protected action {contract.action_id} began at "
                    f"{contract.source_start_seconds:.3f}s"
                )
            if source_out < contract.safe_cut_after_seconds - 1e-6:
                faults.append(
                    f"{clip.clip_id}: protected action {contract.action_id} "
                    f"cannot safely cut before "
                    f"{contract.safe_cut_after_seconds:.3f}s source time; "
                    f"window ends at {source_out:.3f}s"
                )
        for contract in clip.source_motion_contracts:
            if source_in > contract.source_start_seconds + 1e-6:
                faults.append(
                    f"{clip.clip_id}: source in-point {source_in:.3f}s starts "
                    f"after protected {contract.motion_role} source motion began "
                    f"at {contract.source_start_seconds:.3f}s"
                )
            if source_out < contract.safe_cut_after_seconds - 1e-6:
                faults.append(
                    f"{clip.clip_id}: native {contract.motion_role} motion "
                    f"cannot safely cut before "
                    f"{contract.safe_cut_after_seconds:.3f}s source time; "
                    f"window ends at {source_out:.3f}s"
                )
        floor = _floor_for(clip)
        duration = source_out - source_in
        if floor > 0.0 and duration < floor - 1e-6:
            move = clip.reframe.camera_move if clip.reframe is not None else "hold"
            faults.append(
                f"{clip.clip_id}: {move} across this shot needs about "
                f"{floor:.1f}s and has {duration:.2f}s"
            )
    return tuple(dict.fromkeys(faults))


def audio_timeline_faults(edl: EDL) -> tuple[str, ...]:
    """Require every independent audio assignment to fit the picture clock."""

    starts: dict[str, float] = {}
    cursor = 0.0
    for clip in edl.clips:
        starts[clip.clip_id] = cursor
        cursor += clip.approx_out_seconds - clip.approx_in_seconds
    faults: list[str] = []
    for audio in edl.audio_clips:
        anchor = starts.get(audio.starts_at_clip_id)
        if anchor is None:
            faults.append(
                f"audio {audio.audio_id} starts at unknown clip "
                f"{audio.starts_at_clip_id!r}"
            )
            continue
        begins = anchor + audio.offset_seconds
        ends = begins + audio.out_seconds - audio.in_seconds
        if begins < -1e-6 or ends > cursor + 1e-6:
            faults.append(
                f"audio {audio.audio_id} falls outside the {cursor:.3f}s "
                f"picture timeline ({begins:.3f}-{ends:.3f}s)"
            )
    return tuple(dict.fromkeys(faults))


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
        if before is not None:
            source_out = clip.approx_in_seconds + entry.duration_seconds
            for contract in before.action_contracts:
                if contract.completion_policy in {"may_cut_on_action", "loopable"}:
                    continue
                if clip.approx_in_seconds > contract.source_start_seconds + 1e-6:
                    faults.append(
                        f"{clip.clip_id}: source in-point "
                        f"{clip.approx_in_seconds:.3f}s starts after protected "
                        f"action {contract.action_id} began at "
                        f"{contract.source_start_seconds:.3f}s"
                    )
                if source_out < contract.safe_cut_after_seconds - 1e-6:
                    faults.append(
                        f"{clip.clip_id}: protected action {contract.action_id} "
                        f"cannot safely cut before "
                        f"{contract.safe_cut_after_seconds:.3f}s source time; "
                        f"rhythm ends it at {source_out:.3f}s"
                    )
        if before is None:
            continue
        if (
            not before.source_motion_contracts
            and before.reframe is not None
            and before.reframe.editorial_intent == "use_source_motion"
            and before.reframe.source_motion_role == "locked"
        ):
            # Legacy EDLs had no semantic source role or measured contract.
            # Preserve their old conservative behaviour without turning the
            # nominal duration of a current authored/follow shot into proof.
            required = before.approx_out_seconds - before.approx_in_seconds
            window = before.usable_window
            if window is not None:
                required = min(
                    required, max(0.0, window[1] - clip.approx_in_seconds)
                )
            if entry.duration_seconds < required - 1e-6:
                faults.append(
                    f"{clip.clip_id}: legacy native source motion needs its "
                    f"{required:.3f}s span but rhythm left "
                    f"{entry.duration_seconds:.3f}s"
                )
        for contract in before.source_motion_contracts:
            if clip.approx_in_seconds > contract.source_start_seconds + 1e-6:
                faults.append(
                    f"{clip.clip_id}: source in-point "
                    f"{clip.approx_in_seconds:.3f}s starts after protected "
                    f"{contract.motion_role} source motion began at "
                    f"{contract.source_start_seconds:.3f}s"
                )
            source_out = clip.approx_in_seconds + entry.duration_seconds
            if source_out < contract.safe_cut_after_seconds - 1e-6:
                faults.append(
                    f"{clip.clip_id}: native {contract.motion_role} motion "
                    f"cannot safely cut before "
                    f"{contract.safe_cut_after_seconds:.3f}s source time; "
                    f"rhythm ends it at {source_out:.3f}s"
                )
    from montagewright.grounding import apply_to_edl

    executable = apply_to_edl(candidate, timeline)
    faults.extend(resolved_source_contract_faults(executable))
    faults.extend(audio_timeline_faults(executable))
    return tuple(dict.fromkeys(faults))
