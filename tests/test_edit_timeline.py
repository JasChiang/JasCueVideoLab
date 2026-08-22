"""The explicit v2 timeline separates source, programme and sound clocks."""

from montagewright.edit_timeline import from_edl
from montagewright.schema import AudioClip, Clip, EDL


def test_v1_hybrid_clock_projects_to_explicit_ranges_and_handles():
    edl = EDL(
        project_id="fold-launch",
        clips=[
            Clip(
                clip_id="k00", source_id="C1",
                approx_in_seconds=4.0, approx_out_seconds=6.0,
                speed=2.0, usable_from_seconds=2.0, usable_to_seconds=10.0,
            ),
            Clip(
                clip_id="k01", source_id="C2",
                approx_in_seconds=1.0, approx_out_seconds=4.0,
                story_point="show the hinge",
                continuity_mode="associative_montage",
                cut_motivation="match_shape",
                source_event_ref="span_start:C2:s00",
            ),
        ],
    )

    timeline = from_edl(edl)
    first = timeline.picture[0]
    assert (first.source.in_seconds, first.source.out_seconds) == (4.0, 8.0)
    assert (first.timeline.in_seconds, first.timeline.out_seconds) == (0.0, 2.0)
    assert first.head_handle_seconds == 2.0
    assert first.tail_handle_seconds == 2.0
    assert timeline.picture[1].timeline.in_seconds == 2.0
    assert timeline.duration_seconds == 5.0
    assert timeline.edit_points[0].at_seconds == 2.0
    assert timeline.edit_points[0].story_point == "show the hinge"
    assert timeline.edit_points[0].motivation == "match_shape"


def test_independent_audio_becomes_its_own_range_and_sync_link():
    edl = EDL(
        project_id="interview",
        clips=[
            Clip(
                clip_id="k00", source_id="speaker",
                approx_in_seconds=10.0, approx_out_seconds=12.0,
                picture_role="speaker",
            ),
            Clip(
                clip_id="k01", source_id="broll",
                approx_in_seconds=0.0, approx_out_seconds=3.0,
                picture_role="illustrative_broll",
            ),
        ],
        audio_clips=[AudioClip(
            audio_id="a00", source_id="speaker",
            in_seconds=10.0, out_seconds=14.0,
            starts_at_clip_id="k00", offset_seconds=0.0,
            role="narrative", completion="complete_thought",
        )],
    )

    timeline = from_edl(edl)
    assert timeline.audio[0].timeline.out_seconds == 4.0
    assert timeline.sync_links[0].mode == "lip_sync"
    # The voice crosses the picture edit at 2s; there is no audio edge there.
    assert timeline.edit_points[0].audio_edges == ()


def test_source_clock_completion_floors_are_converted_through_speed():
    from montagewright.grounding import _action_floor_for
    from montagewright.schema import ActionContract

    contract = ActionContract(
        action_id="fold", source_start_seconds=1.0,
        source_complete_seconds=5.0, safe_cut_after_seconds=5.0,
    )
    fast = Clip(
        clip_id="fast", source_id="C1",
        approx_in_seconds=1.0, approx_out_seconds=3.0,
        speed=2.0, action_contracts=[contract],
    )
    slow = fast.model_copy(update={"clip_id": "slow", "speed": 0.5})
    assert _action_floor_for(fast) == 2.0
    assert _action_floor_for(slow) == 8.0


def test_visual_coverage_returns_programme_seconds_after_retime():
    from types import SimpleNamespace
    from montagewright.coverage import visual_supported_max

    item = SimpleNamespace(
        action=("`fold` phone folds 0.0-8.0s",), motion=(), camera_moves=False,
    )
    supported = visual_supported_max(
        item, role="primary_action", source_start=0.0,
        available_seconds=5.0, speed=2.0,
    )
    # Eight measured source seconds occupy four programme seconds at 2x; the
    # existing 0.6s comprehension tail is also a programme-clock allowance.
    assert supported == 4.6
