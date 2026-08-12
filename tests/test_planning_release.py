from montagewright.grounding import BeatGrid, Cue
from montagewright.planning_release import rhythm_motion_faults
from montagewright.schema import Clip, EDL, MusicSync, Reframe


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


def test_named_music_cue_must_really_land():
    authored = EDL(project_id="p", clips=[_clip(sync_to="missing")])
    faults = rhythm_motion_faults(authored, authored, _grid())
    assert "did not land" in faults[0]


def test_named_music_cue_and_feasible_virtual_move_pass():
    authored = EDL(project_id="p", clips=[_clip(sync_to="section_002")])
    assert rhythm_motion_faults(authored, authored, _grid()) == ()


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
