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
