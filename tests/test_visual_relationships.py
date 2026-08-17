from montagewright.candidate_commitments import (
    CandidateCommitments,
    CandidateOption,
    bind_selection_content_contracts,
    validate_selection_commitments,
    resolve_candidate_commitments,
)
from montagewright.planner import (
    MaterialItem,
    camera_duration_disagreements,
    material_look_boxes,
)
from montagewright.schema import Look, Reframe, looks_of
from montagewright.spans import Span


def _commitment(*, relationship="simultaneous") -> CandidateCommitments:
    option = CandidateOption(
        commitment_id="cmt_interaction",
        purpose="show the tool acting on its target",
        required=True,
        picture_role="primary_action",
        span_id="C1:span",
        tier="primary",
        min_supported_seconds=2.0,
        presentation_intent="complete_hold",
        content_policy="complete_action",
        content_action_id="a01",
        content_action_start_seconds=1.0,
        content_action_complete_seconds=3.0,
        required_visuals=("the tool", "the target"),
        visual_relationship=relationship,
        motion_preference="virtual_allowed",
        target_id="none",
        why="the interaction, not either object alone, is the content",
        feasible_treatments=("hold", "reveal"),
    )
    return CandidateCommitments(
        contract_version="candidate-commitment-v5-visual-relationships",
        material_digest="0" * 64,
        direction_sha256="1" * 64,
        target_aspect="9:16",
        target_seconds=10.0,
        options=(option,),
    )


def _shot() -> dict:
    return {
        "commitment_id": "cmt_interaction",
        "span_id": "C1:span",
        "seconds_needed": 2.0,
        "camera_intent": "hold",
        "action_id": "a01",
        "action_treatment": "complete_here",
        "looks": [{
            "at": "the target",
            "entity_id": "none",
            "seconds": 1.0,
            "framing": "centre",
            "must_be_whole": False,
            "presentation_intent": "complete_hold",
        }],
    }


def test_simultaneous_visual_contract_is_projected_onto_one_landing():
    shot = _shot()
    commitments = _commitment()

    bind_selection_content_contracts([shot], commitments)

    assert shot["looks"][0]["includes"] == ["the tool", "the target"]
    assert validate_selection_commitments([shot], commitments) == []


def test_ordered_visual_contract_rejects_a_dropped_participant():
    shot = _shot()
    commitments = _commitment(relationship="ordered")

    bind_selection_content_contracts([shot], commitments)

    faults = validate_selection_commitments([shot], commitments)
    assert any("drops required visual participants" in one for one in faults)


def test_group_landing_measures_the_union_not_only_the_named_target():
    item = MaterialItem(
        source_id="C1",
        duration_seconds=5.0,
        summary="tool operates on target",
        crop_width=0.3164,
        subjects=("the tool", "the target"),
        subject_geometry=(
            ("the tool", None, 0.40, 0.50, 0.20, 0.40),
            ("the target", None, 0.62, 0.55, 0.16, 0.18),
        ),
    )
    reframe = Reframe(looks=[Look(
        at="the target",
        includes=["the tool", "the target"],
        framing="centre",
        presentation_intent="complete_hold",
    )])

    boxes = material_look_boxes(item, reframe)

    assert len(boxes) == 1
    centre_x, _, crop_width = boxes[0]
    assert 0.48 < centre_x < 0.52
    assert crop_width == 0.3164


def test_narrow_delivery_records_partial_interaction_without_overriding_it():
    item = MaterialItem(
        source_id="C1",
        duration_seconds=5.0,
        summary="tool operates on target",
        crop_width=0.3164,
        subjects=("the tool", "the target"),
        subject_geometry=(
            ("the tool", None, 0.45, 0.50, 0.35, 0.40),
            ("the target", None, 0.58, 0.60, 0.15, 0.18),
        ),
    )
    shot = _shot()
    shot.update(
        source_id="C1",
        content_required_visuals=["v01", "v02"],
        content_visual_relationship="simultaneous",
    )
    shot["looks"][0]["includes"] = ["v01", "v02"]

    faults = camera_duration_disagreements([shot], [item])

    assert faults == []
    assert shot["visual_fit_advisories"]


def test_cached_label_contract_migrates_to_source_scoped_visual_ids():
    item = MaterialItem(
        source_id="C1", duration_seconds=5.0, summary="interaction",
        subjects=("the tool", "the target"),
        subject_geometry=(
            ("the tool", None, 0.4, 0.5, 0.2, 0.3),
            ("the target", None, 0.6, 0.5, 0.2, 0.3),
        ),
        spans=(Span("C1:s00", "C1", 0.0, 5.0),),
    )
    direction = {"candidate_options": [{
        "commitment_id": "c1", "purpose": "show interaction",
        "required": True, "picture_role": "primary_action",
        "span_id": "C1:s00", "tier": "primary",
        "min_supported_seconds": "0:02",
        "presentation_intent": "complete_hold",
        "content_policy": "static_display", "content_action_id": "none",
        "required_visuals": ["the tool", "the target", "result is visible"],
        # No required_evidence means this is the paid label-era contract.
        "visual_relationship": "simultaneous", "target_id": "none",
        "why": "both participants carry the meaning",
        "recommended_treatment": "hold", "suggested_move": "hold",
        "camera_route": "keep both visible", "motion_reason": "readability",
        "fallback_treatment": "hold",
    }]}

    resolved = resolve_candidate_commitments(
        direction, [item], material_digest="a" * 64,
        aspect="9:16", target_seconds=5.0,
    )

    option = resolved.options[0]
    assert option.required_visuals == ("v01", "v02")
    assert option.required_evidence == ("result is visible",)


def test_alternate_can_fulfil_the_same_purpose_with_a_different_relationship():
    primary = _commitment().options[0]
    alternate = primary.model_copy(update={
        "span_id": "C2:span", "tier": "alternate",
        "required_visuals": ("v01",), "visual_relationship": "single",
    })

    commitments = _commitment().model_copy(update={
        "options": (primary, alternate),
    })

    assert commitments.options[0].visual_relationship == "simultaneous"
    assert commitments.options[1].visual_relationship == "single"


def test_action_outcome_is_bound_to_final_landing_without_replacing_identity():
    option = _commitment().options[0].model_copy(update={
        "required_visuals": ("v01",),
        "required_evidence": ("the status lamp is visibly lit",),
        "outcome_evidence": "the status lamp is visibly lit",
        "visual_relationship": "action_sequence",
        "target_id": "pixel_phone",
    })
    commitments = _commitment().model_copy(update={"options": (option,)})
    shot = _shot()
    shot["looks"][0]["entity_id"] = "pixel_phone"

    bind_selection_content_contracts([shot], commitments)

    endpoint = shot["looks"][-1]
    assert endpoint["geometry_query"] == "the status lamp is visibly lit"
    assert endpoint["geometry_after_source_seconds"] == 3.0
    assert endpoint["entity_id"] == "pixel_phone"
    assert shot["outcome_visual_contract"]["carrier_entity_id"] == "pixel_phone"
    assert validate_selection_commitments([shot], commitments) == []

    executable = looks_of(shot)[0]
    assert executable.includes == ["v01"]
    assert executable.geometry_query == "the status lamp is visibly lit"
    assert executable.entity_id == "pixel_phone"


def test_wide_carrier_with_small_action_outcome_is_not_promoted_to_sequential_read():
    item = MaterialItem(
        source_id="C1", duration_seconds=5.0, summary="tap lights a lamp",
        crop_width=0.3164,
        subjects=("the phone",),
        subject_geometry=(("the phone", None, 0.55, 0.5, 0.68, 0.7),),
        action_windows=(("a01", 1.0, 3.0),),
        spans=(Span("C1:s00", "C1", 0.0, 5.0),),
    )
    direction = {"candidate_options": [{
        "commitment_id": "c1", "purpose": "show the lamp turn on",
        "required": True, "picture_role": "primary_action",
        "span_id": "C1:s00", "tier": "primary",
        "min_supported_seconds": "0:02",
        "presentation_intent": "centered_hold",
        "content_policy": "complete_action", "content_action_id": "a01",
        "required_visuals": ["v01"],
        "required_evidence": ["the status lamp is visibly lit"],
        "visual_relationship": "action_sequence", "target_id": "none",
        "why": "the visible result is the story beat",
        "recommended_treatment": "push_in", "suggested_move": "push_in",
        "camera_route": "settle on the lit lamp", "motion_reason": "show result",
        "fallback_treatment": "hold",
    }]}

    resolved = resolve_candidate_commitments(
        direction, [item], material_digest="a" * 64,
        aspect="9:16", target_seconds=5.0,
    )

    assert resolved.options[0].presentation_intent == "centered_hold"
    assert resolved.options[0].required_visuals == ("v01",)
    assert resolved.options[0].required_evidence == (
        "the status lamp is visibly lit",
    )
    assert resolved.options[0].outcome_evidence == "the status lamp is visibly lit"
