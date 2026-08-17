import json
from types import SimpleNamespace

import pytest


def test_none_audio_completion_does_not_disable_local_hold_repair():
    from montagewright.coverage import repair_bounded_visual_holds

    shot = {
        "source_id": "C1",
        "seconds_needed": 2.0,
        "picture_role": "end_hold",
        "audio_role": "discard",
        "audio_completion": "none",
    }

    repairs = repair_bounded_visual_holds({"shots": [shot]})

    assert shot["seconds_needed"] == 1.5
    assert repairs


def test_action_contract_requires_an_explicit_selected_action():
    from montagewright.clipcard import snap_to_action_contract

    card = {"action": [{
        "id": "a01", "what": "phone unfolds",
        "from": "0:05", "to": "0:09",
    }]}

    unchanged = snap_to_action_contract(
        card, 5.2, 2.0, action_id="none", within=(4.0, 12.0)
    )
    selected = snap_to_action_contract(
        card, 5.2, 2.0, action_id="a01", within=(4.0, 12.0)
    )

    assert unchanged == (5.2, None, None)
    assert selected[0] == 5.0
    assert selected[1].action_id == "a01"
    assert selected[1].minimum_duration_from(selected[0]) == 4.0


def test_raw_motion_only_extends_coverage_for_semantic_source_motion():
    from montagewright.coverage import visual_supported_max
    from montagewright.motion import MotionInterval

    item = SimpleNamespace(
        action=(),
        camera_moves=False,
        motion=(MotionInterval(
            "m00", 0.0, 6.0, "moving", 0.3, 0.4, settles=True,
        ),),
    )

    locked = visual_supported_max(
        item, role="primary_action", source_start=0.0,
        available_seconds=6.0, motion_role="",
    )
    authored = visual_supported_max(
        item, role="primary_action", source_start=0.0,
        available_seconds=6.0, motion_role="authored",
    )

    assert locked == 3.0
    assert authored == 6.0


def test_edl_uses_explicit_action_and_recomputes_coverage_after_snap(tmp_path):
    from montagewright.cli import _edl_from_selection
    from montagewright.clipcard import CARD_VERSION
    from montagewright.planner import MaterialItem

    card_path = tmp_path / "C1.json"
    card_path.write_text(json.dumps({
        "version": CARD_VERSION,
        "action": [{
            "id": "a01", "what": "phone unfolds",
            "from": "0:05", "to": "0:09",
        }]
    }), encoding="utf-8")
    shot = {
        "source_id": "C1", "span_id": "C1:s00",
        "start_seconds": 5.2, "seconds_needed": 4.0,
        "usable_from_seconds": 4.0, "usable_to_seconds": 12.0,
        "action_id": "a01", "action_treatment": "complete_here",
        "camera_intent": "hold",
        "source_motion_role": "locked", "frame": "settles",
        "energy": "medium", "why": "show the complete unfold",
        "audio_role": "discard", "audio_completion": "none",
        "picture_role": "primary_action", "coverage_claim_seconds": 1.0,
        "looks": [{"at": "phone", "seconds": 0.0, "framing": "thirds"}],
    }
    material = [MaterialItem(
        source_id="C1", duration_seconds=12.0, summary="unfold",
        action=("`a01` phone unfolds 5.0-9.0s",),
        action_ids=("a01",),
        action_windows=(("a01", 5.0, 9.0),),
    )]

    edl, _ = _edl_from_selection(
        {"shots": [shot]}, tmp_path, {"C1": card_path}, material=material,
    )
    clip = edl.clips[0]

    assert clip.approx_in_seconds == 5.0
    assert clip.approx_out_seconds == 9.0
    assert clip.coverage_claim_seconds == 4.0
    assert [one.action_id for one in clip.action_contracts] == ["a01"]


def test_invalid_or_zero_selection_duration_never_becomes_four_seconds(tmp_path):
    from montagewright.cli import _edl_from_selection

    selection = {"shots": [{
        "source_id": "C1", "start_seconds": 0.0, "seconds_needed": 0.0,
        "action_id": "none", "camera_intent": "hold",
        "looks": [{"at": "centre", "seconds": 0.0, "framing": "thirds"}],
    }]}

    with pytest.raises(ValueError, match="not changed into 4s"):
        _edl_from_selection(selection, tmp_path, cards={})


def test_source_motion_floor_comes_from_local_interval_not_nominal_duration():
    from montagewright.coverage import source_motion_contract_for
    from montagewright.grounding import ground_timeline
    from montagewright.motion import MotionInterval
    from montagewright.schema import Clip, EDL, Reframe

    item = SimpleNamespace(motion=(MotionInterval(
        "m00", 0.5, 2.0, "moving", 0.3, 0.4, settles=True,
    ),))
    contract = source_motion_contract_for(
        item, source_start=0.0, source_end=4.0, motion_role="authored"
    )
    clip = Clip(
        clip_id="k00", source_id="C1",
        approx_in_seconds=0.0, approx_out_seconds=1.0,
        reframe=Reframe(
            camera_move="hold", editorial_intent="use_source_motion",
            source_motion_role="authored", intent="keep authored move",
        ),
        source_motion_contracts=[contract],
    )

    grounded = ground_timeline(EDL(project_id="p", clips=[clip]), None)

    assert contract.source_start_seconds == 0.5
    assert contract.safe_cut_after_seconds == 2.0
    assert grounded.clips[0].duration_seconds == 2.0


def test_content_floor_survives_rhythm_even_without_an_action_or_music():
    from montagewright.grounding import ground_timeline
    from montagewright.schema import Clip, ContentContract, EDL

    clip = Clip(
        clip_id="k00", source_id="C1",
        approx_in_seconds=10.0, approx_out_seconds=12.0,
        content_contracts=[ContentContract(
            commitment_id="calendar_result",
            purpose="let the viewer read the generated calendar event",
            policy="result_hold",
            minimum_seconds=3.5,
        )],
    )

    grounded = ground_timeline(EDL(project_id="p", clips=[clip]), None)

    assert grounded.clips[0].duration_seconds == 3.5
    assert "result_hold content needs 3.50s" in (grounded.clips[0].note or "")


def test_edl_persists_locally_bound_content_contract(tmp_path):
    from montagewright.cli import _edl_from_selection

    selection = {"shots": [{
        "source_id": "C1", "start_seconds": 0.0, "seconds_needed": 2.5,
        "action_id": "none", "action_treatment": "none",
        "camera_intent": "hold", "picture_role": "title_read",
        "content_policy": "result_hold", "content_min_seconds": 2.5,
        "content_purpose": "read the result", "commitment_id": "result",
        "looks": [{"at": "screen", "seconds": 0.0, "framing": "thirds"}],
    }]}

    edl, _ = _edl_from_selection(selection, tmp_path, cards={})

    assert edl.clips[0].content_contracts[0].policy == "result_hold"
    assert edl.clips[0].content_contracts[0].minimum_seconds == 2.5


def test_selection_schema_requires_explicit_action_and_strict_mmss():
    from montagewright.planner import _selection_schema

    shot = _selection_schema(
        ["C1:s00"], action_ids=["a01"]
    )["properties"]["shots"]["items"]

    assert "action_id" in shot["required"]
    assert "action_treatment" in shot["required"]
    assert shot["properties"]["action_id"]["enum"] == ["none", "a01"]
    assert shot["properties"]["action_treatment"]["enum"] == [
        "none", "complete_here", "after_completion", "intentional_cut",
    ]
    assert shot["properties"]["seconds_needed"]["pattern"]


def test_action_id_is_validated_against_the_selected_source():
    from montagewright.planner import (
        MaterialItem, action_contract_disagreements,
    )

    material = [MaterialItem(
        source_id="C1", duration_seconds=4.0, summary="one action",
        action_ids=("a01",),
    )]

    faults = action_contract_disagreements([{
        "source_id": "C1", "action_id": "a02",
        "action_treatment": "complete_here",
    }], material)

    assert "not an action offered inside span C1" in faults[0]


def test_primary_action_cannot_silently_drop_an_offered_action_contract():
    from montagewright.planner import (
        MaterialItem, action_contract_disagreements,
    )

    material = [MaterialItem(
        source_id="C1", duration_seconds=6.0, summary="tap then result",
        action_ids=("a01",), action_windows=(("a01", 1.0, 4.0),),
    )]
    shot = {
        "source_id": "C1", "picture_role": "primary_action",
        "action_id": "none", "action_treatment": "none",
    }

    faults = action_contract_disagreements([shot], material)

    assert len(faults) == 1
    assert "must choose complete_here" in faults[0]


def test_primary_action_does_not_inherit_actions_from_another_span():
    from montagewright.planner import MaterialItem, action_contract_disagreements
    from montagewright.spans import Span

    material = [MaterialItem(
        source_id="C1", duration_seconds=60.0, summary="action then later result",
        action_ids=("a01",), action_windows=(("a01", 2.0, 6.0),),
        spans=(
            Span("C1:s00", "C1", 0.0, 8.0),
            Span("C1:s02", "C1", 40.0, 50.0),
        ),
    )]
    later_result = {
        "source_id": "C1", "span_id": "C1:s02",
        "picture_role": "primary_action",
        "action_id": "none", "action_treatment": "none",
    }

    assert action_contract_disagreements([later_result], material) == []


def test_intentional_action_cut_must_really_end_before_completion():
    from montagewright.planner import (
        MaterialItem, action_contract_disagreements,
    )

    material = [MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="long gesture",
        action_ids=("a01",), action_windows=(("a01", 1.0, 6.0),),
    )]
    common = {
        "source_id": "C1", "picture_role": "primary_action",
        "action_id": "a01", "action_treatment": "intentional_cut",
        "start_seconds": 1.0, "usable_from_seconds": 0.0,
        "usable_to_seconds": 8.0, "why": "cut on the gesture for momentum",
    }

    assert action_contract_disagreements([
        {**common, "seconds_needed": 2.0}
    ], material) == []
    faults = action_contract_disagreements([
        {**common, "seconds_needed": 5.0}
    ], material)
    assert "already reaches completion" in faults[0]


def test_compiled_camera_move_that_must_run_too_fast_requires_replan():
    from montagewright.reframe import build_look_path

    degradations = []
    path = build_look_path(
        [(0.35, 0.2, 0.5, 0.3164), (0.35, 0.8, 0.5, 0.3164)],
        source_aspect=16 / 9,
        target_aspect=9 / 16,
        duration_seconds=0.6,
        energy="calm",
        clip_id="k00",
        degradations=degradations,
    )

    hurried = [
        step for step in degradations
        if step.ladder_other == "looks_do_not_fit_the_time"
    ]
    assert path.keyframes
    assert path.keyframes[-1].crop.x > path.keyframes[0].crop.x
    assert len(hurried) == 1
    assert hurried[0].adjudication == "replan"


def test_real_camera_stop_never_shrinks_below_readable_settle():
    from montagewright.capabilities import SETTLE_SECONDS
    from montagewright.reframe import seconds_needed_for

    needed = seconds_needed_for([
        (0.05, 0.25, 0.5, 0.3164),
        (0.10, 0.75, 0.5, 0.3164),
    ], "calm")
    travelling_only = seconds_needed_for([
        (-1.0, 0.25, 0.5, 0.3164),
        (-1.0, 0.75, 0.5, 0.3164),
    ], "calm")

    assert needed >= travelling_only + SETTLE_SECONDS * 2


def test_short_complete_action_is_rejected_before_rhythm():
    from montagewright.planner import (
        MaterialItem, action_contract_disagreements,
    )

    material = [MaterialItem(
        source_id="C1", duration_seconds=20.0, summary="long action",
        action_ids=("a01",), action_windows=(("a01", 0.0, 14.0),),
    )]
    faults = action_contract_disagreements([{
        "source_id": "C1", "action_id": "a01",
        "action_treatment": "complete_here", "seconds_needed": 4.0,
        "usable_from_seconds": 0.0, "usable_to_seconds": 20.0,
    }], material)

    assert "needs at least 14.00s" in faults[0]


def test_after_completion_is_a_safe_short_treatment(tmp_path):
    from montagewright.cli import _edl_from_selection
    from montagewright.clipcard import CARD_VERSION
    from montagewright.planner import MaterialItem, action_contract_disagreements

    card_path = tmp_path / "C1.json"
    card_path.write_text(json.dumps({
        "version": CARD_VERSION,
        "action": [{
            "id": "a01", "what": "phone unfolds",
            # Deliberately stale/conflicting. The EDL must execute the exact
            # MaterialItem contract that Selection validated, not reparse a
            # second authority after the paid decision has passed.
            "from": "0:00", "to": "0:18",
        }],
    }), encoding="utf-8")
    material = [MaterialItem(
        source_id="C1", duration_seconds=20.0, summary="result hold",
        action_ids=("a01",), action_windows=(("a01", 0.0, 14.0),),
    )]
    shot = {
        "source_id": "C1", "span_id": "C1:s00",
        "start_seconds": 0.0, "seconds_needed": 3.0,
        "usable_from_seconds": 0.0, "usable_to_seconds": 20.0,
        "action_id": "a01", "action_treatment": "after_completion",
        "camera_intent": "hold", "source_motion_role": "locked",
        "frame": "settles", "energy": "low", "why": "show the result",
        "audio_role": "discard", "audio_completion": "none",
        "picture_role": "end_hold",
        "looks": [{"at": "phone", "seconds": 0.0, "framing": "thirds"}],
    }

    assert action_contract_disagreements([shot], material) == []
    edl, _ = _edl_from_selection(
        {"shots": [shot]}, tmp_path, {"C1": card_path}, material=material,
    )
    clip = edl.clips[0]
    assert clip.approx_in_seconds == 14.0
    assert clip.approx_out_seconds == 17.0
    assert clip.action_contracts == []
    assert clip.moments == {}, "a finished action cannot be reintroduced as a Rhythm anchor"


def test_action_validation_uses_the_named_span_not_selection_echoes():
    from montagewright.planner import MaterialItem, action_contract_disagreements
    from montagewright.spans import Span

    material = [MaterialItem(
        source_id="C1", duration_seconds=30.0, summary="result hold",
        action_ids=("a01",), action_windows=(("a01", 0.0, 14.0),),
        spans=(Span("C1:s02", "C1", 0.0, 16.0),),
    )]
    shot = {
        "source_id": "C1", "span_id": "C1:s02",
        "action_id": "a01", "action_treatment": "after_completion",
        "seconds_needed": 3.0,
        # A stale/model-derived echo says there is room to 20s. The named
        # local span ends at 16s and is the only boundary execution may use.
        "usable_from_seconds": 0.0, "usable_to_seconds": 20.0,
    }

    faults = action_contract_disagreements([shot], material)

    assert len(faults) == 1
    assert "cannot hold 3.00s" in faults[0]


def test_after_completion_must_begin_inside_the_selected_span():
    from montagewright.planner import MaterialItem, action_contract_disagreements
    from montagewright.spans import Span

    material = [MaterialItem(
        source_id="C1", duration_seconds=70.0, summary="later result island",
        action_ids=("a03",), action_windows=(("a03", 20.0, 30.0),),
        spans=(Span("C1:s02", "C1", 41.0, 61.0),),
    )]
    shot = {
        "source_id": "C1", "span_id": "C1:s02",
        "action_id": "a03", "action_treatment": "after_completion",
        "seconds_needed": 3.0,
        "usable_from_seconds": 41.0, "usable_to_seconds": 61.0,
    }

    faults = action_contract_disagreements([shot], material)

    assert len(faults) == 1
    assert "not an action offered inside span C1:s02" in faults[0]


def test_normalized_selection_cannot_keep_stale_span_clock_echoes():
    from montagewright.planner import MaterialItem, span_contract_disagreements
    from montagewright.spans import Span

    material = [MaterialItem(
        source_id="C1", duration_seconds=30.0, summary="one usable island",
        spans=(Span("C1:s02", "C1", 4.0, 12.0),),
    )]

    faults = span_contract_disagreements([{
        "source_id": "C1", "span_id": "C1:s02",
        "usable_from_seconds": 0.0, "usable_to_seconds": 20.0,
    }], material)

    assert len(faults) == 1
    assert "does not match named span C1:s02" in faults[0]


def test_edl_uses_the_same_material_look_geometry_as_selection(tmp_path):
    from montagewright.cli import _edl_from_selection
    from montagewright.clipcard import CARD_VERSION
    from montagewright.planner import MaterialItem

    card_path = tmp_path / "C1.json"
    card_path.write_text(json.dumps({
        "version": CARD_VERSION,
        "subjects": [{
            "label": "phone", "centre_x": 0.9, "centre_y": 0.5,
            "width": 0.2, "height": 0.4, "moves": False,
        }],
    }), encoding="utf-8")
    material = [MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="phone",
        subject_geometry=(("phone", None, 0.2, 0.4, 0.2, 0.5),),
    )]
    shot = {
        "source_id": "C1", "span_id": "C1:s00",
        "start_seconds": 0.0, "seconds_needed": 3.0,
        "usable_from_seconds": 0.0, "usable_to_seconds": 8.0,
        "action_id": "none", "action_treatment": "none",
        "camera_intent": "push_in", "source_motion_role": "locked",
        "frame": "travels", "energy": "medium", "why": "show phone",
        "audio_role": "discard", "audio_completion": "none",
        "picture_role": "illustrative_broll",
        "looks": [
            {"at": "phone", "seconds": 1.0, "framing": "thirds"},
            {"at": "phone", "seconds": 1.0, "framing": "fill"},
        ],
    }

    edl, _ = _edl_from_selection(
        {"shots": [shot]}, tmp_path, {"C1": card_path}, material=material,
    )

    assert edl.clips[0].reframe.look_boxes[0][:2] == (0.2, 0.4)


def test_intentional_cut_is_explicit_but_does_not_create_completion_floor(tmp_path):
    from montagewright.cli import _edl_from_selection
    from montagewright.clipcard import CARD_VERSION
    from montagewright.planner import MaterialItem

    card_path = tmp_path / "C1.json"
    card_path.write_text(json.dumps({
        "version": CARD_VERSION,
        "action": [{
            "id": "a01", "what": "hand crosses the display",
            "from": "0:01", "to": "0:06",
        }],
    }), encoding="utf-8")
    material = [MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="gesture",
        action_ids=("a01",), action_windows=(("a01", 1.0, 6.0),),
    )]
    shot = {
        "source_id": "C1", "span_id": "C1:s00",
        "start_seconds": 1.0, "seconds_needed": 2.0,
        "usable_from_seconds": 0.0, "usable_to_seconds": 8.0,
        "action_id": "a01", "action_treatment": "intentional_cut",
        "camera_intent": "hold", "source_motion_role": "locked",
        "frame": "settles", "energy": "medium",
        "why": "cut on the hand crossing frame to carry momentum",
        "audio_role": "discard", "audio_completion": "none",
        "picture_role": "primary_action",
        "looks": [{"at": "display", "seconds": 0.0, "framing": "thirds"}],
    }

    edl, snaps = _edl_from_selection(
        {"shots": [shot]}, tmp_path, {"C1": card_path}, material=material,
    )

    assert edl.clips[0].approx_in_seconds == 1.0
    assert edl.clips[0].approx_out_seconds == 3.0
    assert edl.clips[0].action_contracts == []
    assert "intentionally cuts" in snaps["k00"]


def test_local_clock_gate_keeps_invalid_mmss_visible_for_repair():
    from montagewright.planner import selection_clock_disagreements

    faults = selection_clock_disagreements([{
        "start_offset_seconds": "1.5",
        "seconds_needed": "0:00",
        "looks": [{"seconds": "0:00"}],
    }])

    assert any("start_offset_seconds" in fault for fault in faults)
    assert any("zero MM:SS seconds_needed" in fault for fault in faults)


def test_unrenderable_selection_preserves_a_reviewable_paid_draft() -> None:
    from montagewright.planner import SelectionUnrenderable

    answer = {"shots": [{"source_id": "C8342", "camera_intent": "pan"}]}
    error = SelectionUnrenderable(
        "cannot execute", draft=answer, faults=["k00 cannot reach its look"]
    )
    answer["shots"][0]["camera_intent"] = "hold"

    assert error.draft["shots"][0]["camera_intent"] == "pan"
    assert error.faults == ("k00 cannot reach its look",)


def test_single_complete_hold_keeps_only_a_settle_floor_for_rhythm() -> None:
    from montagewright.grounding import _floor_for
    from montagewright.schema import Clip, Look, Reframe

    clip = Clip(
        clip_id="k00", source_id="C1",
        approx_in_seconds=0.0, approx_out_seconds=4.0,
        reframe=Reframe(looks=[Look(
            at="phone detail", seconds=2.0,
            presentation_intent="complete_hold", must_be_whole=False,
        )]),
    )

    assert _floor_for(clip) == 0.35


def test_selection_rejects_look_rests_longer_than_the_shot() -> None:
    from montagewright.planner import frame_disagreements

    faults = frame_disagreements([{
        "camera_intent": "use_source_motion",
        "source_motion_role": "authored",
        "frame": "settles",
        "seconds_needed": 3.0,
        "looks": [{
            "at": "the product",
            "seconds": 3.5,
            "presentation_intent": "complete_hold",
        }],
    }])

    assert faults == ["k00 promises 3.50s of look holds inside a 3.00s shot"]


def test_one_look_overflow_is_repaired_without_reselecting_the_edit() -> None:
    from montagewright.planner import (
        frame_disagreements,
        repair_single_look_hold_overflow,
    )

    chosen = {"shots": [{
        "commitment_id": "cmt_product",
        "camera_intent": "use_source_motion",
        "source_motion_role": "authored",
        "frame": "settles",
        "seconds_needed": 3.0,
        "looks": [{
            "at": "the product",
            "seconds": 3.5,
            "presentation_intent": "complete_hold",
        }],
    }]}

    repairs = repair_single_look_hold_overflow(chosen)

    assert chosen["shots"][0]["looks"][0]["seconds"] == 3.0
    assert chosen["shots"][0]["commitment_id"] == "cmt_product"
    assert frame_disagreements(chosen["shots"]) == []
    assert "source, commitment and total edit length are unchanged" in repairs[0]


def test_transition_pass_is_not_a_stop_or_a_default_settle() -> None:
    from montagewright.planner import frame_disagreements
    from montagewright.reframe import _rest_for_stop

    assert _rest_for_stop(-1.0) == 0.0
    faults = frame_disagreements([{
        "camera_intent": "multi_stop", "frame": "travels",
        "looks": [
            {"at": "A", "presentation_intent": "complete_hold"},
            {"at": "passing B", "presentation_intent": "transition_pass"},
            {"at": "C", "presentation_intent": "complete_hold"},
        ],
    }])
    assert any("gave 2 looks" in fault for fault in faults)


def test_selected_action_already_inside_window_keeps_earlier_context() -> None:
    from montagewright.clipcard import snap_to_action_contract

    card = {"action": [{
        "id": "a01", "what": "camera pulls back",
        "from": 1.0, "to": 3.0,
    }]}

    start, contract, note = snap_to_action_contract(
        card, 0.0, 3.0, action_id="a01", within=(0.0, 3.0)
    )

    assert start == 0.0
    assert contract is not None
    assert contract.safe_cut_after_seconds == 3.0
    assert note is None


def test_material_action_window_preserves_contract_when_card_path_is_missing(
    tmp_path,
) -> None:
    from montagewright.cli import _edl_from_selection
    from montagewright.planner import MaterialItem

    shot = {
        "source_id": "C1", "span_id": "C1:s00",
        "start_seconds": 5.0, "seconds_needed": 4.0,
        "usable_from_seconds": 4.0, "usable_to_seconds": 12.0,
        "action_id": "a01", "action_treatment": "complete_here",
        "camera_intent": "hold", "source_motion_role": "locked",
        "energy": "medium", "why": "complete it",
        "audio_role": "discard", "audio_completion": "none",
        "picture_role": "primary_action",
        "looks": [{"at": "phone", "seconds": 0.0, "framing": "thirds"}],
    }
    material = [MaterialItem(
        source_id="C1", duration_seconds=12.0, summary="unfold",
        action_ids=("a01",), action_windows=(("a01", 5.0, 9.0),),
    )]

    edl, _ = _edl_from_selection(
        {"shots": [shot]}, tmp_path, cards={}, material=material,
    )

    assert edl.clips[0].approx_in_seconds == 5.0
    assert edl.clips[0].approx_out_seconds == 9.0
    assert edl.clips[0].action_contracts[0].safe_cut_after_seconds == 9.0


def test_partial_native_motion_is_an_excerpt_not_a_whole_move_contract() -> None:
    from montagewright.coverage import source_motion_contract_for
    from montagewright.motion import MotionInterval
    from montagewright.planning_release import resolved_source_contract_faults
    from montagewright.schema import Clip, EDL, Reframe

    item = SimpleNamespace(motion=(MotionInterval(
        "m00", 0.5, 3.0, "moving", 0.3, 0.4, settles=True,
    ),))
    contract = source_motion_contract_for(
        item, source_start=1.0, source_end=2.0, motion_role="authored",
    )
    assert contract is None


def test_whole_native_motion_keeps_its_measured_contract_boundaries() -> None:
    from montagewright.coverage import source_motion_contract_for
    from montagewright.motion import MotionInterval
    from montagewright.planning_release import resolved_source_contract_faults
    from montagewright.schema import Clip, EDL, Reframe

    item = SimpleNamespace(motion=(MotionInterval(
        "m00", 0.5, 3.0, "moving", 0.3, 0.4, settles=True,
    ),))
    contract = source_motion_contract_for(
        item, source_start=0.0, source_end=4.0, motion_role="authored",
    )
    assert contract.source_start_seconds == 0.5
    assert contract.safe_cut_after_seconds == 3.0

    clip = Clip(
        clip_id="k00", source_id="C1",
        # Simulate Rhythm subsequently cutting the selected complete move.
        approx_in_seconds=1.0, approx_out_seconds=2.0,
        reframe=Reframe(
            camera_move="hold", editorial_intent="use_source_motion",
            source_motion_role="authored", intent="keep the native move",
        ),
        source_motion_contracts=[contract],
    )
    faults = resolved_source_contract_faults(
        EDL(project_id="p", clips=[clip])
    )
    assert any("starts after protected authored" in fault for fault in faults)
    assert any("cannot safely cut before 3.000s" in fault for fault in faults)


def test_rhythm_prompt_uses_the_same_single_look_settle_floor_as_release() -> None:
    from montagewright.grounding import _floor_for
    from montagewright.planner import _needs_at_least
    from montagewright.schema import Clip, Look, Reframe

    clip = Clip(
        clip_id="k00", source_id="C1",
        approx_in_seconds=0.0, approx_out_seconds=3.0,
        reframe=Reframe(looks=[Look(
            at="readout", seconds=2.0, presentation_intent="complete_hold",
        )]),
    )
    assert _needs_at_least(clip) == _floor_for(clip) == 0.35


def test_multi_look_dwell_remains_a_real_camera_floor() -> None:
    from montagewright.grounding import _floor_for
    from montagewright.schema import Clip, Look, Reframe

    clip = Clip(
        clip_id="k00", source_id="C1",
        approx_in_seconds=0.0, approx_out_seconds=4.0,
        reframe=Reframe(
            camera_move="pan",
            looks=[
                Look(at="left", seconds=1.0),
                Look(at="right", seconds=1.0),
            ],
            look_boxes=[(0.25, 0.5, 0.2), (0.75, 0.5, 0.2)],
        ),
    )

    assert _floor_for(clip) > 2.0


def test_selection_prices_camera_time_from_legacy_card_geometry() -> None:
    from montagewright.planner import (
        MaterialItem, camera_duration_disagreements,
    )

    material = [MaterialItem(
        "C8377", 2.0, "three foldables",
        subject_geometry=(
            ("the silver foldable phone on the left", None,
             0.23, 0.52, 0.22, 0.49),
            ("the purple foldable phone in the middle", None,
             0.44, 0.58, 0.17, 0.39),
            ("the flip phone on the right", None,
             0.65, 0.61, 0.18, 0.34),
        ),
    )]
    shot = {
        "source_id": "C8377", "camera_intent": "compare",
        "energy": "low", "seconds_needed": 2.0,
        "looks": [
            {
                "entity_id": "device.foldable", "at": "左側銀色摺疊手機",
                "seconds": 1.0, "framing": "thirds",
                "presentation_intent": "complete_hold",
            },
            {
                "entity_id": "device.foldable", "at": "中間紫色摺疊手機",
                "seconds": 1.0, "framing": "thirds",
                "presentation_intent": "complete_hold",
            },
        ],
    }

    faults = camera_duration_disagreements([shot], material)

    assert len(faults) == 1
    assert "2.840s" in faults[0]
    assert "measured card positions" in faults[0]


def test_grounding_id_does_not_hide_geometry_on_pre_identity_cards() -> None:
    from montagewright.clipcard import find_subject

    card = {"subjects": [
        {"label": "the silver phone on the left", "centre_x": 0.2,
         "centre_y": 0.5, "width": 0.2, "height": 0.4},
        {"label": "the purple phone in the middle", "centre_x": 0.5,
         "centre_y": 0.5, "width": 0.2, "height": 0.4},
        {"label": "the dark phone on the right", "centre_x": 0.8,
         "centre_y": 0.5, "width": 0.2, "height": 0.4},
    ]}

    found = find_subject(
        card, "中間紫色摺疊手機", entity_id="device.foldable"
    )

    assert found is not None
    assert found.centre_x == 0.5


def test_same_subject_push_uses_measured_zoom_distance_not_move_constant() -> None:
    from montagewright.planner import (
        MaterialItem, camera_duration_disagreements,
    )

    material = [MaterialItem(
        "C8343", 11.0, "lineup",
        subject_geometry=(
            ("左側白色半開摺疊手機", None, 0.23, 0.58, 0.23, 0.53),
            ("中間展開狀態的手機", None, 0.46, 0.59, 0.17, 0.57),
            ("右側雙折裝置", None, 0.72, 0.59, 0.29, 0.56),
        ),
    )]
    shot = {
        "source_id": "C8343", "camera_intent": "push_in",
        "energy": "low", "seconds_needed": 1.5,
        "looks": [
            {
                "entity_id": "device.foldable",
                "at": "中間展開狀態的折疊手機（彩色螢幕亮起）",
                "seconds": 0.5, "framing": "centre",
                "presentation_intent": "centered_hold",
            },
            {
                "entity_id": "device.foldable",
                "at": "中間展開狀態的折疊手機（彩色螢幕亮起）",
                "seconds": 1.0, "framing": "fill",
                "presentation_intent": "centered_hold",
            },
        ],
    }

    faults = camera_duration_disagreements([shot], material)

    assert len(faults) == 1
    assert "2.045s" in faults[0]
    assert "2.500s" not in faults[0]


def test_selection_keeps_shot_and_rhythm_when_only_preferred_rests_overflow() -> None:
    from montagewright.planner import (
        MaterialItem,
        camera_duration_disagreements,
        repair_camera_rests_to_duration,
    )

    material = [MaterialItem(
        "C8343", 11.0, "lineup",
        subject_geometry=(
            ("中間展開狀態的手機", None, 0.46, 0.59, 0.17, 0.57),
        ),
    )]
    chosen = {"shots": [{
        "source_id": "C8343", "commitment_id": "cmt_model",
        "camera_intent": "push_in", "energy": "low",
        "seconds_needed": 1.5,
        "looks": [
            {
                "entity_id": "device.foldable",
                "at": "中間展開狀態的折疊手機", "seconds": 0.5,
                "framing": "centre", "presentation_intent": "centered_hold",
            },
            {
                "entity_id": "device.foldable",
                "at": "中間展開狀態的折疊手機", "seconds": 1.0,
                "framing": "fill", "presentation_intent": "centered_hold",
            },
        ],
    }]}

    repairs = repair_camera_rests_to_duration(chosen, material)

    assert repairs and "kept push_in" in repairs[0]
    shot = chosen["shots"][0]
    assert shot["source_id"] == "C8343"
    assert shot["commitment_id"] == "cmt_model"
    assert shot["seconds_needed"] == 1.5
    assert shot["camera_intent"] == "push_in"
    assert camera_duration_disagreements([shot], material) == []


def test_single_sequential_read_fits_its_dwell_after_local_travel_is_priced() -> None:
    from montagewright.planner import (
        MaterialItem,
        camera_duration_disagreements,
        repair_camera_rests_to_duration,
    )

    material = [MaterialItem(
        "C1", 8.0, "wide sign read vertically",
        crop_width=0.316,
        subject_geometry=((
            "wide sign", None, 0.5, 0.5, 0.82, 0.25,
        ),),
    )]
    chosen = {"shots": [{
        "source_id": "C1", "commitment_id": "c01",
        "camera_intent": "reveal", "energy": "medium",
        "seconds_needed": 3.0,
        "looks": [{
            "entity_id": "none", "at": "wide sign", "seconds": 2.0,
            "framing": "centre", "must_be_whole": False,
            "presentation_intent": "sequential_read", "includes": ["v01"],
        }],
    }]}

    before = camera_duration_disagreements(chosen["shots"], material)
    repairs = repair_camera_rests_to_duration(chosen, material)

    assert before
    assert repairs
    assert chosen["shots"][0]["seconds_needed"] == 3.0
    assert chosen["shots"][0]["camera_intent"] == "reveal"
    assert chosen["shots"][0]["looks"][0]["seconds"] < 2.0
    assert camera_duration_disagreements(chosen["shots"], material) == []


def test_bound_action_is_stronger_than_one_recurring_subject_sighting() -> None:
    from montagewright.motion import MotionInterval
    from montagewright.planner import MaterialItem, frame_disagreements
    from montagewright.spans import Span

    item = MaterialItem(
        "C1", 30.0, "tutorial on the same display",
        sightings=(("interactive display", 7.0),),
        spans=(Span("C1:s00", "C1", 0.0, 30.0),),
        motion=(
            MotionInterval("m00", 0.0, 13.0, "still", 0.0, 0.0, False),
            MotionInterval("m01", 13.0, 16.0, "not_a_shift", 0.0, 0.0, False),
            MotionInterval("m02", 16.0, 30.0, "still", 0.0, 0.0, False),
        ),
    )
    shot = {
        "source_id": "C1", "span_id": "C1:s00",
        "start_seconds": 20.0, "seconds_needed": 4.0,
        "camera_intent": "hold", "frame": "settles",
        "content_required_visuals": ["v01"],
        "content_action_start_seconds": 20.0,
        "content_action_complete_seconds": 26.0,
        "action_id": "a03", "action_treatment": "intentional_cut",
        "looks": [{
            "entity_id": "none", "at": "interactive display",
            "seconds": 1.0, "framing": "centre",
            "must_be_whole": False,
            "presentation_intent": "centered_hold", "includes": ["v01"],
        }],
    }

    assert frame_disagreements([shot], [item]) == []

    without_action = dict(shot, action_treatment="none")
    assert "picture changes" in frame_disagreements([without_action], [item])[0]


def test_selection_patch_schema_cannot_return_a_complete_timeline() -> None:
    from montagewright.planner import _selection_patch_schema

    schema = _selection_patch_schema(
        ["C1:s00"], [16], commitment_ids=["c17"], action_ids=["a01"]
    )

    assert "shots" not in schema["properties"]
    choices = schema["properties"]["choices"]
    assert choices["minItems"] == choices["maxItems"] == 1
    assert choices["items"]["properties"]["shot_index"]["enum"] == [16]
    assert set(choices["items"]["properties"]) == {
        "shot_index", "option_id", "camera_treatment", "why",
    }


def test_selection_patch_schema_keeps_large_authorisation_enums_local() -> None:
    from montagewright.planner import _selection_patch_schema

    schema = _selection_patch_schema(
        [f"C{i}:s00" for i in range(40)], [2, 7],
        grounding_target_ids=[f"target-{i}" for i in range(12)],
        commitment_ids=[f"cmt-{i}" for i in range(20)],
        action_ids=[f"a{i:02d}" for i in range(30)],
    )
    shot = schema["properties"]["choices"]["items"]

    assert shot["properties"]["shot_index"]["enum"] == [2, 7]
    assert shot["properties"]["option_id"]["enum"] == [
        f"C{i}:s00" for i in range(40)
    ]
    assert "seconds_needed" not in shot["properties"]
    assert "start_offset_seconds" not in shot["properties"]
    assert "looks" not in shot["properties"]


def test_motion_contract_normalizes_authored_reveal_without_inventing_a_move() -> None:
    from montagewright.planner import repair_selection_motion_contracts

    chosen = {"shots": [{
        "source_id": "C1", "seconds_needed": 3.0,
        "camera_intent": "hold", "source_motion_role": "authored",
        "content_visual_relationship": "ordered",
        "content_required_visuals": ["v01", "v02"],
        "direction_motion_advice": {
            "treatment": "use_source_motion",
            "locally_feasible": ["use_source_motion", "hold"],
        },
        "looks": [{
            "at": "the endpoint", "seconds": 1.0,
            "presentation_intent": "reveal_endpoint",
        }],
    }]}

    repairs = repair_selection_motion_contracts(chosen, [])

    assert repairs
    assert chosen["shots"][0]["camera_intent"] == "use_source_motion"


def test_motion_contract_translates_two_stop_multistop_to_reveal() -> None:
    from montagewright.planner import repair_selection_motion_contracts

    chosen = {"shots": [{
        "source_id": "C1", "seconds_needed": 3.0,
        "camera_intent": "multi_stop",
        "content_visual_relationship": "ordered",
        "looks": [
            {"at": "left", "seconds": 1.0,
             "presentation_intent": "sequential_read"},
            {"at": "right", "seconds": 1.0,
             "presentation_intent": "reveal_endpoint"},
        ],
    }]}

    repair_selection_motion_contracts(chosen, [])

    assert chosen["shots"][0]["camera_intent"] == "reveal"
    assert chosen["shots"][0]["frame"] == "travels"


def test_source_motion_participants_are_not_replayed_as_digital_stops() -> None:
    from montagewright.planner import repair_selection_motion_contracts

    chosen = {"shots": [{
        "source_id": "C1", "seconds_needed": 2.0,
        "camera_intent": "use_source_motion",
        "source_motion_role": "authored",
        "content_visual_relationship": "ordered",
        "content_required_visuals": ["v01", "v02"],
        "looks": [
            {"at": "left", "seconds": 0.7,
             "presentation_intent": "reveal_endpoint"},
            {"at": "right", "seconds": 0.7,
             "presentation_intent": "reveal_endpoint"},
        ],
    }]}

    repair_selection_motion_contracts(chosen, [])

    assert len(chosen["shots"][0]["looks"]) == 1
    assert chosen["shots"][0]["looks"][0]["at"] == "right"
    assert chosen["shots"][0]["looks"][0]["includes"] == ["v01", "v02"]


def test_source_motion_endpoint_carries_the_ordered_visual_evidence() -> None:
    from montagewright.planner import repair_selection_motion_contracts

    chosen = {"shots": [{
        "source_id": "C1", "seconds_needed": 3.0,
        "camera_intent": "hold", "source_motion_role": "authored",
        "content_visual_relationship": "ordered",
        "content_required_visuals": ["v01", "v02"],
        "direction_motion_advice": {
            "treatment": "use_source_motion",
            "locally_feasible": ["use_source_motion", "hold"],
        },
        "looks": [{
            "at": "the reveal endpoint", "seconds": 1.0,
            "presentation_intent": "reveal_endpoint",
        }],
    }]}

    repair_selection_motion_contracts(chosen, [])

    shot = chosen["shots"][0]
    assert shot["camera_intent"] == "use_source_motion"
    assert shot["looks"][0]["includes"] == ["v01", "v02"]


def test_source_window_solver_places_reveal_sighting_near_the_shot_end() -> None:
    from montagewright.planner import MaterialItem, repair_selection_source_windows
    from montagewright.spans import Span

    item = MaterialItem(
        source_id="C1", duration_seconds=7.0, summary="source pan",
        sightings=(("screen", 6.0),),
        spans=(Span("C1:s01", "C1", 1.0, 7.0),),
    )
    shot = {
        "source_id": "C1", "span_id": "C1:s01",
        "usable_from_seconds": 1.0, "usable_to_seconds": 7.0,
        "start_seconds": 1.0, "start_offset_seconds": 0.0,
        "seconds_needed": 3.0, "action_id": "none",
        "action_treatment": "none",
        "looks": [{
            "at": "screen", "seconds": 1.6,
            "presentation_intent": "reveal_endpoint",
        }],
    }

    repairs = repair_selection_source_windows({"shots": [shot]}, [item])

    assert repairs
    assert 3.7 < shot["start_seconds"] < 3.9
    assert shot["start_offset_seconds"] == shot["start_seconds"] - 1.0


def test_source_window_solver_contains_a_direction_bound_complete_action() -> None:
    from montagewright.planner import MaterialItem, repair_selection_source_windows
    from montagewright.spans import Span

    item = MaterialItem(
        source_id="C1", duration_seconds=16.0, summary="receipt capture",
        spans=(Span("C1:s00", "C1", 0.0, 16.0),),
    )
    shot = {
        "source_id": "C1", "span_id": "C1:s00",
        "start_seconds": 0.5, "start_offset_seconds": 0.5,
        "seconds_needed": 5.0, "action_id": "a01",
        "action_treatment": "complete_here",
        "content_action_start_seconds": 1.0,
        "content_action_complete_seconds": 6.0,
        "looks": [],
    }

    repairs = repair_selection_source_windows({"shots": [shot]}, [item])

    assert repairs
    assert shot["start_seconds"] == 1.0
    assert shot["seconds_needed"] == 5.0


def test_a_sighting_outside_the_named_span_is_not_absence_evidence() -> None:
    from montagewright.planner import MaterialItem, frame_disagreements
    from montagewright.spans import Span

    item = MaterialItem(
        source_id="C1", duration_seconds=30.0, summary="recurring phone",
        sightings=(("phone", 7.0),), motion=(), crop_width=0.316,
        spans=(Span("C1:s02", "C1", 20.0, 30.0),),
    )
    shot = {
        "source_id": "C1", "span_id": "C1:s02",
        "start_seconds": 21.0, "seconds_needed": 3.0,
        "camera_intent": "hold", "frame": "settles",
        "looks": [{
            "at": "phone", "seconds": 3.0, "framing": "centre",
            "entity_id": "none", "must_be_whole": False,
            "presentation_intent": "centered_hold",
        }],
    }

    assert frame_disagreements([shot], [item]) == []


def test_partial_simultaneous_composition_is_advisory_not_a_hard_fault() -> None:
    from montagewright.planner import MaterialItem, camera_duration_disagreements

    item = MaterialItem(
        source_id="C1", duration_seconds=4.0, summary="wide interaction",
        crop_width=0.316,
        subject_geometry=(
            ("phone", None, 0.25, 0.5, 0.30, 0.5),
            ("person", None, 0.75, 0.5, 0.30, 0.8),
        ),
    )
    shot = {
        "source_id": "C1", "seconds_needed": 3.0,
        "camera_intent": "hold", "energy": "medium",
        "content_required_visuals": ["v01", "v02"],
        "content_visual_relationship": "simultaneous",
        "looks": [{
            "at": "phone and person", "seconds": 3.0,
            "includes": ["v01", "v02"], "must_be_whole": False,
            "presentation_intent": "centered_hold",
        }],
    }

    assert camera_duration_disagreements([shot], [item]) == []
    assert shot["visual_fit_advisories"]


def test_whole_simultaneous_composition_still_fails_when_it_cannot_fit() -> None:
    from montagewright.planner import MaterialItem, camera_duration_disagreements

    item = MaterialItem(
        source_id="C1", duration_seconds=4.0, summary="wide interaction",
        crop_width=0.316,
        subject_geometry=(
            ("phone", None, 0.25, 0.5, 0.30, 0.5),
            ("person", None, 0.75, 0.5, 0.30, 0.8),
        ),
    )
    shot = {
        "source_id": "C1", "seconds_needed": 3.0,
        "camera_intent": "hold", "energy": "medium",
        "content_required_visuals": ["v01", "v02"],
        "content_visual_relationship": "simultaneous",
        "looks": [{
            "at": "phone and person", "seconds": 3.0,
            "includes": ["v01", "v02"], "must_be_whole": True,
            "presentation_intent": "complete_hold",
        }],
    }

    faults = camera_duration_disagreements([shot], [item])

    assert len(faults) == 1
    assert "target aspect shows simultaneous participants partially" in faults[0]


def test_selection_patch_merge_keeps_every_unauthorized_shot_byte_for_byte() -> None:
    import copy

    from montagewright.planner import _merge_selection_patch
    from montagewright.spans import Span

    base = {
        "shots": [
            {"source_id": "C0", "commitment_id": "c00", "nested": {"x": 1}},
            {"source_id": "C1", "span_id": "C1:s00",
             "commitment_id": "c17", "nested": {"x": 2},
             "seconds_needed": 2.0, "start_offset_seconds": 0.0,
             "looks": [{"at": "old", "framing": "centre"}]},
            {"source_id": "C2", "commitment_id": "c02", "nested": {"x": 3}},
        ],
        "audio_assignments": [{"audio_span_id": "voice:01"}],
        "covered": ["c00", "c17", "c02"],
        "uncovered": [],
    }
    before = copy.deepcopy(base)
    option = SimpleNamespace(
        commitment_id="c17", span_id="C9:s00",
        feasible_treatments=("hold",), min_supported_seconds=1.0,
        content_action_start_seconds=None, content_action_complete_seconds=None,
        content_policy="static_display", content_action_id="none",
        picture_role="illustrative_broll", required_visuals=("v01",),
        target_id="none", purpose="show the alternate",
        presentation_intent="centered_hold",
    )

    merged = _merge_selection_patch(
        base, {"choices": [{
            "shot_index": 1, "option_id": "C9:s00",
            "camera_treatment": "hold", "why": "cleaner alternate",
        }]}, allowed_indices={1},
        offered=[Span("C9:s00", "C9", 0.0, 3.0, "alternate", "locked")],
        source_motion={"C9": "locked"},
        commitments=SimpleNamespace(options=(option,)),
        material=[SimpleNamespace(
            source_id="C9", subject_geometry=((
                "alternate product", None, .5, .5, .2, .4,
            ),),
        )],
    )

    assert merged["shots"][0] == before["shots"][0]
    assert merged["shots"][2] == before["shots"][2]
    assert merged["audio_assignments"] == before["audio_assignments"]
    assert merged["covered"] == before["covered"]
    assert merged["shots"][1]["source_id"] == "C9"
    assert merged["shots"][1]["commitment_id"] == "c17"
    assert merged["shots"][1]["seconds_needed"] == 2.0


def test_selection_choice_cannot_select_an_option_from_another_commitment() -> None:
    import pytest

    from montagewright.planner import PlannerError, _merge_selection_patch
    from montagewright.spans import Span

    option = SimpleNamespace(commitment_id="c99", span_id="C9:s00")
    with pytest.raises(PlannerError, match="not an option for commitment 'c17'"):
        _merge_selection_patch(
            {"shots": [{"commitment_id": "c17", "seconds_needed": 2.0}]},
            {"choices": [{
                "shot_index": 0, "option_id": "C9:s00",
                "camera_treatment": "hold", "why": "wrong promise",
            }]},
            allowed_indices={0},
            offered=[Span("C9:s00", "C9", 0.0, 3.0, "alt", "locked")],
            source_motion={},
            commitments=SimpleNamespace(options=(option,)),
        )


def test_selection_patch_span_must_belong_to_the_original_commitment() -> None:
    import pytest

    from montagewright.planner import PlannerError, _merge_selection_patch
    from montagewright.spans import Span

    option = SimpleNamespace(commitment_id="c17", span_id="C9:s00")
    with pytest.raises(PlannerError, match="outside commitment 'c17'"):
        _merge_selection_patch(
            {"shots": [{"commitment_id": "c17", "seconds_needed": 2.0}]},
            {"choices": [{
                "shot_index": 0, "option_id": "C9:s00",
                "camera_treatment": "hold", "why": "unauthorized",
            }]},
            allowed_indices={0},
            offered=[Span(
                "C9:s00", "C9", 0.0, 3.0, "alternate", "locked"
            )],
            source_motion={"C9": "locked"},
            commitment_spans={"c17": {"C17:s00"}},
            commitments=SimpleNamespace(options=(option,)),
        )


def test_fresh_selection_final_repair_uses_patch_not_full_timeline() -> None:
    import inspect

    from montagewright import planner

    source = inspect.getsource(planner.select_shots)

    assert "pending_patch_base = copy.deepcopy(chosen)" in source
    assert "response 只選既有 option_id 與 camera_treatment" in source
    assert "本次不重新附影片" in source
    assert "attempt_schema = schema" in source  # global-fault fallback only


def test_multi_look_rests_fit_locally_when_the_move_itself_still_fits() -> None:
    from montagewright.cli import _fit_camera_rests_to_shot
    from montagewright.grounding import _floor_for
    from montagewright.schema import Clip, EDL, Look, Reframe

    selection = {"shots": [{
        "camera_intent": "push_in",
        "looks": [
            {"at": "logo", "seconds": 1.0, "framing": "thirds",
             "presentation_intent": "centered_hold"},
            {"at": "logo", "seconds": 1.5, "framing": "fill",
             "presentation_intent": "centered_hold"},
        ],
    }]}
    clip = Clip(
        clip_id="k00", source_id="C1",
        approx_in_seconds=0.0, approx_out_seconds=3.0,
        reframe=Reframe(
            camera_move="push_in",
            looks=[
                Look(at="logo", seconds=1.0, framing="thirds"),
                Look(at="logo", seconds=1.5, framing="fill"),
            ],
            look_boxes=[(0.5, 0.5, 1.0), (0.5, 0.5, 0.8)],
        ),
    )
    edl = EDL(project_id="p", clips=[clip])
    assert _floor_for(clip) == 3.3

    repairs = _fit_camera_rests_to_shot(selection, edl)
    repaired_reframe = clip.reframe.model_copy(update={
        "looks": [Look.model_validate(one) for one in selection["shots"][0]["looks"]]
    })

    assert repairs and "keeps its travel" in repairs[0]
    assert sum(one["seconds"] for one in selection["shots"][0]["looks"]) == 2.2
    assert _floor_for(clip.model_copy(update={"reframe": repaired_reframe})) == 3.0


def test_sequential_read_turns_one_wide_subject_into_readable_landings() -> None:
    from montagewright.reframe import (
        build_look_path, sequential_read_centres, seconds_needed_for,
    )

    centres = sequential_read_centres(
        centre_x=0.5, subject_width=0.65, crop_width=0.3164,
    )
    assert len(centres) == 3
    assert centres[0] < 0.35 < centres[-1]

    stops = [(0.0, x, 0.5, 0.3164) for x in centres]
    needed = seconds_needed_for(stops, "active")
    path = build_look_path(
        stops, source_aspect=16 / 9, target_aspect=9 / 16,
        duration_seconds=needed, energy="active",
    )

    assert path.keyframes[0].crop.x < path.keyframes[-1].crop.x
    # Each synthesized landing has a repeated crop, which is a real pause
    # rather than a continuous unreadable pass.
    assert len({round(one.seconds, 3) for one in path.keyframes}) >= 5


def test_continuous_read_suppresses_a_redundant_tail_rebound() -> None:
    from montagewright.reframe import build_look_path

    degradations = []
    path = build_look_path(
        [
            (0.5, 0.16, 0.5, 0.3164),
            (0.5, 0.42, 0.5, 0.3164),
            (0.5, 0.69, 0.5, 0.3164),
            (0.5, 0.37, 0.5, 0.3164),
        ],
        source_aspect=16 / 9,
        target_aspect=9 / 16,
        duration_seconds=3.1,
        energy="active",
        clip_id="k00",
        degradations=degradations,
        continuous_read=True,
    )

    xs = [one.crop.x for one in path.keyframes]
    assert xs == sorted(xs)
    assert path.keyframes[-1].crop.x > 0.5
    assert any(
        one.ladder_other == "redundant_readback_suppressed"
        for one in degradations
    )


def test_continuous_read_passes_collinear_landings_without_hard_stops() -> None:
    from montagewright.reframe import build_look_path

    path = build_look_path(
        [
            (0.7, 0.16, 0.5, 0.3164),
            (0.7, 0.53, 0.5, 0.3164),
            (1.0, 0.64, 0.5, 0.3164),
        ],
        source_aspect=16 / 9,
        target_aspect=9 / 16,
        duration_seconds=3.0,
        energy="active",
        continuous_read=True,
    )

    xs = [round(one.crop.x, 4) for one in path.keyframes]
    # Collinear internal waypoints need no authored keyframe. The single
    # eased leg passes through them without treating each as a new stop.
    assert len(xs) == 4
    assert xs[0] == xs[1]
    assert xs[-1] == xs[-2]
    assert xs[1] < xs[2]


def test_sequential_read_adapts_to_each_delivery_aspect() -> None:
    from montagewright.reframe import sequential_read_centres

    subject = dict(centre_x=0.5, subject_width=0.65)
    vertical = sequential_read_centres(crop_width=9 / 16 / (16 / 9), **subject)
    square = sequential_read_centres(crop_width=1 / (16 / 9), **subject)
    landscape = sequential_read_centres(crop_width=1.0, **subject)

    assert len(vertical) == 3
    assert len(square) == 2
    assert landscape == (0.5,)


def test_sequential_read_uses_delivery_crop_for_selection_timing() -> None:
    from montagewright.grounding import camera_floor_for
    from montagewright.planner import MaterialItem, material_look_boxes
    from montagewright.schema import Look, Reframe

    item = MaterialItem(
        source_id="C1", duration_seconds=5.0, summary="wide product row",
        crop_width=0.3164,
        subject_geometry=((
            "the complete row", None, 0.5, 0.5, 0.65, 0.2,
        ),),
    )
    reframe = Reframe(
        camera_move="pan", planned_to_move=True,
        looks=[Look(
            at="the complete row", seconds=0.0,
            presentation_intent="sequential_read", must_be_whole=False,
        )],
    )
    boxes = material_look_boxes(item, reframe)
    priced = reframe.model_copy(update={"look_boxes": boxes})

    assert len(boxes) == 3
    assert all(one[2] == pytest.approx(0.3164) for one in boxes)
    assert camera_floor_for(priced) > 1.0


def test_push_does_not_expand_its_first_look_into_a_sequential_pan() -> None:
    """A presentation label on one landing cannot rewrite the treatment."""

    from montagewright.planner import MaterialItem, material_look_boxes
    from montagewright.reframe import camera_route_policy
    from montagewright.schema import Look, Reframe

    item = MaterialItem(
        source_id="C1", duration_seconds=5.0, summary="phone LED detail",
        crop_width=0.3164,
        subject_geometry=((
            "the phone", None, 0.55, 0.52, 0.7, 0.6,
        ),),
    )
    reframe = Reframe(
        camera_move="push_in", editorial_intent="push_in",
        looks=[
            Look(
                at="the phone", framing="centre",
                presentation_intent="sequential_read",
            ),
            Look(
                at="the phone", framing="fill",
                presentation_intent="complete_hold",
            ),
        ],
    )

    boxes = material_look_boxes(item, reframe)

    assert not camera_route_policy(reframe).expand_sequential_read
    assert len(boxes) == 2
    assert boxes[0][0] == pytest.approx(0.55)
    assert boxes[1][0] == pytest.approx(0.55)


def test_sequential_read_is_a_move_not_a_whole_frame_promise() -> None:
    from pydantic import ValidationError

    from montagewright.planner import frame_disagreements
    from montagewright.schema import Look

    shot = {
        "source_id": "C1", "start_seconds": 0.0, "seconds_needed": 3.0,
        "camera_intent": "reveal", "source_motion_role": "locked",
        "looks": [{
            "entity_id": "none", "at": "the wide wordmark", "seconds": 0.0,
            "framing": "centre", "must_be_whole": False,
            "presentation_intent": "sequential_read",
        }],
    }
    assert frame_disagreements([shot]) == []

    with pytest.raises(ValidationError, match="sequential_read"):
        Look(
            at="the wide wordmark", presentation_intent="sequential_read",
            must_be_whole=True,
        )

    shot["camera_intent"] = "hold"
    assert "sequential_read needs reveal" in frame_disagreements([shot])[0]
