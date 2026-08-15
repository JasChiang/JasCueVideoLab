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


def test_selection_schema_requires_explicit_action_and_strict_mmss():
    from montagewright.planner import _selection_schema

    shot = _selection_schema(
        ["C1:s00"], action_ids=["a01"]
    )["properties"]["shots"]["items"]

    assert "action_id" in shot["required"]
    assert "action_treatment" in shot["required"]
    assert shot["properties"]["action_id"]["enum"] == ["none", "a01"]
    assert shot["properties"]["action_treatment"]["enum"] == [
        "none", "complete_here", "after_completion",
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

    assert "not an action offered by source C1" in faults[0]


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
            "from": "0:00", "to": "0:14",
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


def test_selection_patch_schema_cannot_return_a_complete_timeline() -> None:
    from montagewright.planner import _selection_patch_schema

    schema = _selection_patch_schema(
        ["C1:s00"], [16], commitment_ids=["c17"], action_ids=["a01"]
    )

    assert "shots" not in schema["properties"]
    replacements = schema["properties"]["replacements"]
    assert replacements["minItems"] == replacements["maxItems"] == 1
    assert replacements["items"]["properties"]["shot_index"]["enum"] == [16]


def test_selection_patch_merge_keeps_every_unauthorized_shot_byte_for_byte() -> None:
    import copy

    from montagewright.planner import _merge_selection_patch
    from montagewright.spans import Span

    base = {
        "shots": [
            {"source_id": "C0", "commitment_id": "c00", "nested": {"x": 1}},
            {"source_id": "C1", "commitment_id": "c17", "nested": {"x": 2}},
            {"source_id": "C2", "commitment_id": "c02", "nested": {"x": 3}},
        ],
        "audio_assignments": [{"audio_span_id": "voice:01"}],
        "covered": ["c00", "c17", "c02"],
        "uncovered": [],
    }
    before = copy.deepcopy(base)
    replacement = {
        "shot_index": 1,
        "span_id": "C9:s00", "start_offset_seconds": "0:00",
        "seconds_needed": "0:02", "commitment_id": "c17",
        "camera_intent": "hold", "looks": [{"seconds": "0:00.5"}],
    }

    merged = _merge_selection_patch(
        base, {"replacements": [replacement]}, allowed_indices={1},
        offered=[Span("C9:s00", "C9", 0.0, 3.0, "alternate", "locked")],
        source_motion={"C9": "locked"},
    )

    assert merged["shots"][0] == before["shots"][0]
    assert merged["shots"][2] == before["shots"][2]
    assert merged["audio_assignments"] == before["audio_assignments"]
    assert merged["covered"] == before["covered"]
    assert merged["shots"][1]["source_id"] == "C9"
    assert merged["shots"][1]["commitment_id"] == "c17"


def test_selection_patch_cannot_change_the_shot_commitment() -> None:
    import pytest

    from montagewright.planner import PlannerError, _merge_selection_patch
    from montagewright.spans import Span

    with pytest.raises(PlannerError, match="changed commitment"):
        _merge_selection_patch(
            {"shots": [{"commitment_id": "c17"}]},
            {"replacements": [{
                "shot_index": 0, "span_id": "C9:s00",
                "start_offset_seconds": "0:00", "seconds_needed": "0:02",
                "commitment_id": "c99", "camera_intent": "hold", "looks": [],
            }]},
            allowed_indices={0},
            offered=[Span("C9:s00", "C9", 0.0, 3.0, "alt", "locked")],
            source_motion={},
        )


def test_selection_patch_span_must_belong_to_the_original_commitment() -> None:
    import pytest

    from montagewright.planner import PlannerError, _merge_selection_patch
    from montagewright.spans import Span

    with pytest.raises(PlannerError, match="outside commitment 'c17'"):
        _merge_selection_patch(
            {"shots": [{"commitment_id": "c17"}]},
            {"replacements": [{
                "shot_index": 0, "span_id": "C9:s00",
                "start_offset_seconds": "0:00", "seconds_needed": "0:02",
                "commitment_id": "c17", "camera_intent": "hold",
                "looks": [],
            }]},
            allowed_indices={0},
            offered=[Span(
                "C9:s00", "C9", 0.0, 3.0, "alternate", "locked"
            )],
            source_motion={"C9": "locked"},
            commitment_spans={"c17": {"C17:s00"}},
        )


def test_fresh_selection_final_repair_uses_patch_not_full_timeline() -> None:
    import inspect

    from montagewright import planner

    source = inspect.getsource(planner.select_shots)

    assert "pending_patch_base = copy.deepcopy(chosen)" in source
    assert "response schema 只允許回傳指定鏡頭的 replacements" in source
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
