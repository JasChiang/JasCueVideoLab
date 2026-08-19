from montagewright.grounding import BeatGrid, Cue
from montagewright.planning_release import rhythm_motion_faults
from montagewright.schema import (
    ActionContract, Clip, ContentContract, EDL, Look, MusicSync, Reframe,
)


def _clip(
    clip_id="k00", *, seconds=3.0, move="hold", sync_to=None,
    usable_to=10.0,
):
    return Clip(
        clip_id=clip_id,
        source_id="C1",
        approx_in_seconds=0.0,
        approx_out_seconds=seconds,
        usable_from_seconds=0.0,
        usable_to_seconds=usable_to,
        reframe=Reframe(camera_move=("hold" if move == "use_source_motion" else move),
                        editorial_intent=move, intent=move),
        music_sync=MusicSync(cut_on_beat=False, sync_to=sync_to),
    )


def _grid():
    return BeatGrid(
        bpm=120.0, meter=4, duration_seconds=10.0,
        cues=(Cue("section_002", 4.0, "section_boundary", 1.0),),
    )


def test_preferred_camera_duration_extends_the_same_move_inside_usable_window():
    from montagewright.planning_release import resolve_preferred_camera_durations

    # usable_to 7.0, not 6.0: the peak-aware move floor for this reveal is now
    # ~6.6s, so a 6.0s window can no longer hold it and nothing would extend.
    # The test is about extending INTO the usable window, so the window must
    # be able to hold the honestly-priced move.
    clip = _clip(seconds=2.0, move="reveal", usable_to=7.0).model_copy(update={
        "reframe": Reframe(
            looks=[
                Look(at="left", seconds=1.5, presentation_intent="complete_hold"),
                Look(at="right", seconds=1.5, presentation_intent="reveal_endpoint"),
            ],
            look_boxes=[(0.2, 0.5, 0.3), (0.8, 0.5, 0.3)],
            camera_move="pan",
            editorial_intent="reveal",
            intent="read left to right",
        )
    })
    resolved, notes = resolve_preferred_camera_durations(
        EDL(project_id="p", clips=[clip]), duration_mode="preferred"
    )

    assert notes
    assert resolved.clips[0].reframe.editorial_intent == "reveal"
    assert resolved.clips[0].approx_out_seconds > 2.0
    assert not resolved.clips[0].music_sync.cut_on_beat


def test_exact_camera_duration_is_never_locally_extended():
    from montagewright.planning_release import resolve_preferred_camera_durations

    edl = EDL(project_id="p", clips=[_clip(seconds=1.0, move="push_in")])
    assert resolve_preferred_camera_durations(
        edl, duration_mode="exact"
    ) == (edl, ())


def test_named_music_cue_must_really_land():
    authored = EDL(project_id="p", clips=[_clip(sync_to="missing")])
    faults = rhythm_motion_faults(authored, authored, _grid())
    assert "did not land" in faults[0]


def test_named_music_cue_and_feasible_virtual_move_pass():
    authored = EDL(project_id="p", clips=[_clip(sync_to="section_002")])
    assert rhythm_motion_faults(authored, authored, _grid()) == ()


def test_music_anchor_may_not_move_past_a_protected_action_start():
    """Content completion outranks a sub-second beat alignment."""

    clip = Clip(
        clip_id="k08",
        source_id="C8361",
        approx_in_seconds=2.0,
        approx_out_seconds=6.0,
        usable_from_seconds=0.0,
        usable_to_seconds=9.0,
        moments={"a01": 2.0},
        action_contracts=[ActionContract(
            action_id="a01",
            what="models turn their phones around",
            source_start_seconds=2.0,
            source_complete_seconds=6.0,
            safe_cut_after_seconds=6.0,
            timing_basis="coarse_mmss",
        )],
        music_sync=MusicSync(
            cut_on_beat=False,
            anchor="a01",
            anchor_lands_on="downbeat",
        ),
    )
    grid = BeatGrid(
        bpm=120.0,
        meter=4,
        duration_seconds=10.0,
        cues=(Cue("d0", 1.918, "downbeat", 1.0),),
    )

    from montagewright.grounding import ground_timeline

    lead = Clip(
        clip_id="k07",
        source_id="C0",
        approx_in_seconds=0.0,
        approx_out_seconds=2.0,
        music_sync=MusicSync(cut_on_beat=False),
    )
    grounded = ground_timeline(
        EDL(project_id="p", clips=[lead, clip]), grid
    ).clips[1]
    assert grounded.clip.approx_in_seconds == 2.0
    assert "kept the complete source action" in (grounded.note or "")
    assert rhythm_motion_faults(
        EDL(project_id="p", clips=[lead, clip]),
        EDL(project_id="p", clips=[lead, clip]),
        grid,
    ) == ()


def test_music_may_not_shorten_a_native_reveal_below_its_authored_span():
    authored = EDL(project_id="p", clips=[
        _clip(seconds=4.0, move="use_source_motion")
    ])
    shortened = EDL(project_id="p", clips=[
        _clip(seconds=2.0, move="use_source_motion")
    ])
    faults = rhythm_motion_faults(authored, shortened, None)
    assert "native source motion" in faults[0]


def test_a_move_is_not_shortened_by_rhythm_when_the_take_simply_ends():
    """Ten thousandths of a second refused a whole timeline.

    A span and its usable window are measured by different passes, so a
    4.000s authored span can sit inside a 3.990s window. The feasibility
    clamp trims the shot to what the take can supply -- correctly, and with
    a note saying so -- and this check then read the result as rhythm having
    cut the move short. Rhythm may not choose to shorten an authored move;
    it cannot be blamed for material that runs out.
    """

    from montagewright.grounding import BeatGrid, Cue
    from montagewright.planning_release import rhythm_motion_faults
    from montagewright.schema import EDL, Clip, MusicSync, Reframe

    def film(out_seconds, usable_to):
        return EDL(project_id="p", clips=[Clip(
            clip_id="k00", source_id="C1",
            approx_in_seconds=0.0, approx_out_seconds=out_seconds,
            usable_from_seconds=0.0, usable_to_seconds=usable_to,
            audio_role="discard",
            reframe=Reframe(
                editorial_intent="use_source_motion", intent="let it play",
            ),
            music_sync=MusicSync(cut_on_beat=False),
        )])

    grid = BeatGrid(bpm=120.0, meter=4, duration_seconds=60.0, cues=(
        Cue(cue_id="d0", time_seconds=0.0, kind="downbeat"),
    ))

    # The take ends 10ms before the span does: the shortfall is the
    # material's, and the shot already carries a note about it.
    assert rhythm_motion_faults(
        film(4.0, 3.99), film(4.0, 3.99), grid
    ) == ()

    # Rhythm choosing to leave half a second less is still refused.
    assert rhythm_motion_faults(
        film(4.0, 10.0), film(3.5, 10.0), grid
    ) != ()


def test_speaker_alignment_is_revalidated_against_protected_source_contracts():
    from montagewright.pipeline import align_speaker_pictures_to_audio
    from montagewright.planning_release import resolved_source_contract_faults
    from montagewright.schema import AudioClip

    clip = Clip(
        clip_id="k00", source_id="C1",
        approx_in_seconds=5.0, approx_out_seconds=9.0,
        picture_role="speaker",
        action_contracts=[ActionContract(
            action_id="a01", what="phone unfolds",
            source_start_seconds=5.0,
            source_complete_seconds=9.0,
            safe_cut_after_seconds=9.0,
        )],
    )
    edl = EDL(project_id="p", clips=[clip], audio_clips=[AudioClip(
        audio_id="voice", source_id="C1",
        in_seconds=7.0, out_seconds=9.0,
        starts_at_clip_id="k00", role="narrative",
        completion="complete_thought",
    )])

    aligned, _ = align_speaker_pictures_to_audio(edl)
    assert aligned.clips[0].approx_in_seconds == 7.0
    faults = resolved_source_contract_faults(aligned)
    assert any("starts after protected action a01" in fault for fault in faults)


def test_independent_audio_must_fit_inside_the_resolved_picture_timeline():
    from montagewright.planning_release import audio_timeline_faults
    from montagewright.schema import AudioClip

    edl = EDL(project_id="p", clips=[_clip(seconds=2.0)], audio_clips=[
        AudioClip(
            audio_id="voice", source_id="C1",
            in_seconds=0.0, out_seconds=3.0,
            starts_at_clip_id="k00", role="narrative",
            completion="complete_thought",
        )
    ])

    faults = audio_timeline_faults(edl)
    assert faults == (
        "audio voice falls outside the 2.000s picture timeline (0.000-3.000s)",
    )


def test_release_refuses_rhythm_that_shortens_a_content_dwell_contract():
    from montagewright.planning_release import resolved_source_contract_faults

    clip = _clip(seconds=2.0)
    clip = clip.model_copy(update={"content_contracts": [ContentContract(
        commitment_id="result",
        purpose="read the generated result",
        policy="result_hold",
        minimum_seconds=3.0,
    )]})

    faults = resolved_source_contract_faults(EDL(project_id="p", clips=[clip]))

    assert len(faults) == 1
    assert "result_hold" in faults[0]
    assert "needs at least 3.000s" in faults[0]
