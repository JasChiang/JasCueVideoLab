"""Local release proofs between editorial planning and expensive rendering."""

from __future__ import annotations

from montagewright.grounding import BeatGrid, ground_timeline
from montagewright.schema import EDL


def resolve_preferred_camera_durations(
    edl: EDL, *, duration_mode: str,
) -> tuple[EDL, tuple[str, ...]]:
    """Give an already-selected move enough time without changing its idea.

    This is the only automatic camera repair.  It may extend a preferred
    delivery inside the proven usable window when camera duration is the sole
    unresolved contract.  It never trims content, changes source/treatment,
    or moves an exact-duration delivery off its requested clock.
    """

    if duration_mode != "preferred":
        return edl, ()
    from montagewright.grounding import camera_floor_for

    rewritten = []
    notes: list[str] = []
    for clip in edl.clips:
        duration = clip.approx_out_seconds - clip.approx_in_seconds
        required = camera_floor_for(clip.reframe)
        if required <= duration + 1e-6:
            rewritten.append(clip)
            continue
        isolated = edl.model_copy(update={"clips": [clip], "audio_clips": []})
        faults = resolved_source_contract_faults(isolated)
        if faults:
            rewritten.append(clip)
            continue
        window = clip.usable_window
        desired_out = clip.approx_in_seconds + required
        if window is None or desired_out > window[1] + 1e-6:
            rewritten.append(clip)
            continue
        sync = clip.music_sync.model_copy(update={
            "cut_on_beat": False,
            "beats": None,
            "sync_to": None,
            "rhythm_reason": (
                clip.music_sync.rhythm_reason
                + "；本機保留完成運鏡所需時間，切點離開拍點"
            ).strip("；"),
        })
        rewritten.append(clip.model_copy(update={
            "approx_out_seconds": desired_out,
            "coverage_claim_seconds": max(
                float(clip.coverage_claim_seconds or 0.0), required
            ),
            "music_sync": sync,
        }))
        treatment = str(
            getattr(clip.reframe, "editorial_intent", None) or "hold"
        )
        notes.append(
            f"{clip.clip_id}: kept {treatment} and extended "
            f"{duration:.3f}s -> {required:.3f}s inside the proven usable window"
        )
    return edl.model_copy(update={"clips": rewritten}), tuple(notes)


def resolved_source_contract_faults(edl: EDL) -> tuple[str, ...]:
    """Validate source-clock promises on the exact windows to be rendered.

    Several deterministic passes legitimately rewrite a clip's source window:
    action snapping, dialogue-safe snapping, musical anchors and speaker
    lip-sync alignment.  The promises attached to that clip do not move with
    the window.  Keeping this check independent of any one rewriter prevents a
    later pass from silently crossing a protected start or usable boundary.
    """

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
        duration = source_out - source_in
        for contract in clip.content_contracts:
            if duration < contract.minimum_seconds - 1e-6:
                faults.append(
                    f"{clip.clip_id}: {contract.policy} for "
                    f"{contract.commitment_id} needs at least "
                    f"{contract.minimum_seconds:.3f}s to fulfil "
                    f"{contract.purpose!r}; window has {duration:.3f}s"
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
        # A short digital route is reviewable and must not reject the film.
        # ground_timeline already keeps the geometric floor when possible;
        # any remaining shortfall is surfaced by the delivery report.
        requested = clip.music_sync.sync_to
        if requested and entry.landed_on != requested:
            faults.append(
                f"{clip.clip_id}: requested music cue {requested!r} did not "
                f"land ({entry.note or 'no matching local cue'})"
            )
        before = original.get(clip.clip_id)
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
    from montagewright.grounding import apply_to_edl

    executable = apply_to_edl(candidate, timeline)
    faults.extend(resolved_source_contract_faults(executable))
    faults.extend(audio_timeline_faults(executable))
    return tuple(dict.fromkeys(faults))
