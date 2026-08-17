from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from montagewright.candidate_commitments import (
    _camera_treatments,
    bind_selection_content_contracts,
    CandidateCommitments,
    CandidateOption,
    CommitmentError,
    describe_commitments,
    provider_commitment_schema,
    readable_extent,
    resolve_candidate_commitments,
    validate_replacement_commitments,
    validate_selection_commitments,
)
from montagewright.spans import Span
from montagewright.planner import MaterialItem, correct_candidate_options
from montagewright.coverage import (
    CoverageAudit, CoverageEntry, repair_preferred_unsupported_time,
)


@dataclass
class Material:
    source_id: str
    spans: tuple[Span, ...]
    action: tuple[str, ...] = ()
    motion: tuple[object, ...] = ()
    camera_moves: bool = False
    action_windows: tuple[tuple[str, float, float], ...] = ()


def _material():
    return [Material("C1", (
        Span("C1:s00", "C1", 0.0, 4.0, "reveal", "authored"),
        Span("C1:s01", "C1", 4.0, 8.0, "detail", "locked"),
    ), action_windows=(("a01", 0.0, 3.5),))]


class _Interactions:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **request):
        self.calls.append(request)
        return SimpleNamespace(
            status="completed", output_text=json.dumps(self.payload),
            usage={"total_input_tokens": 100, "total_output_tokens": 20},
        )


class _Client:
    def __init__(self, payload):
        self.interactions = _Interactions(payload)


def _direction(**change):
    option = {
        "commitment_id": "hero", "purpose": "complete product reveal",
        "required": True, "picture_role": "primary_action",
        "span_id": "C1:s00", "tier": "primary",
        "min_supported_seconds": "0:03.5",
        "presentation_intent": "reveal_endpoint",
        "content_action_id": "none",
        "motion_preference": "native_first", "target_id": "device.fold",
        "recommended_treatment": "use_source_motion",
        "suggested_move": "source_motion",
        "camera_route": "preserve the authored reveal and settle on the product",
        "motion_reason": "the source already performs the reveal cleanly",
        "fallback_treatment": "reveal",
        "why": "authored movement reaches the product",
    }
    option.update(change)
    return {"candidate_options": [option], "direction": "d"}


def _resolved(direction=None):
    return resolve_candidate_commitments(
        direction or _direction(), _material(), material_digest="a" * 64,
        aspect="9:16", target_seconds=20.0,
        grounding_target_ids=("device.fold",), grounding_sha256="b" * 64,
    )


def test_local_resolver_keeps_the_full_pool_and_marks_unoffered_spans_deferred():
    resolved = _resolved()
    assert resolved.options[0].min_supported_seconds == 3.5
    assert resolved.deferred_span_ids == ("C1:s01",)
    assert "single point of failure" in resolved.warnings[0]
    option = resolved.options[0]
    assert option.preferred_treatment == "use_source_motion"
    assert option.feasible_treatments[:2] == (
        "use_source_motion", "follow_subject"
    )
    assert "Selection 在本機量測的可行清單內選 camera_intent" in describe_commitments(
        resolved
    )


def test_only_identity_primary_and_alternate_sources_are_exact_confirmed():
    from montagewright.cli import _identity_commitment_sources

    direction = _direction()
    alternate = dict(direction["candidate_options"][0])
    alternate.update({
        "span_id": "C1:s01", "tier": "alternate",
        "min_supported_seconds": "0:03",
    })
    direction["candidate_options"].append(alternate)
    resolved = _resolved(direction)
    assert _identity_commitment_sources(resolved) == {"C1"}

    context = resolved.model_copy(update={
        "options": tuple(
            option.model_copy(update={"target_id": "none"})
            for option in resolved.options
        )
    })
    assert _identity_commitment_sources(context) == set()


def test_local_camera_catalog_exposes_measured_virtual_room_before_hold():
    material = [MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="foldable detail",
        spans=_material()[0].spans, pan_room=0.24, tilt_room=0.0,
        push_room=1.4,
    )]
    resolved = resolve_candidate_commitments(
        _direction(
            span_id="C1:s01", motion_preference="hold",
            min_supported_seconds="0:03"
        ), material,
        material_digest="a" * 64, aspect="9:16", target_seconds=20.0,
        grounding_target_ids=("device.fold",), grounding_sha256="b" * 64,
    )
    option = resolved.options[0]
    # A named object is trackable, but this locked product shot already has
    # measured crop room for a deliberate reveal. Ranking follow first would
    # ask the tracker to follow something static and collapse back to hold.
    assert option.preferred_treatment == "reveal"
    assert {"reveal", "compare", "push_in", "pull_out", "multi_stop"} <= set(
        option.feasible_treatments
    )
    assert option.feasible_treatments[-1] == "hold"
    assert option.minimum_camera_seconds == 1.8

    faults = validate_selection_commitments([{
        "commitment_id": "hero", "span_id": "C1:s01",
        "seconds_needed": 1.0, "camera_intent": "multi_stop",
        "looks": [{
            "presentation_intent": "reveal_endpoint",
            "entity_id": "device.fold",
        }],
        }], resolved)
    # Membership/capability lives here; actual duration is priced once from
    # Selection's looks plus measured card coordinates in
    # planner.camera_duration_disagreements.  A second simplified floor table
    # here was the source of planner/executor drift.
    assert not any("camera geometry needs" in fault for fault in faults)
    assert "local camera menu=" in describe_commitments(resolved)
    assert "local preferred=reveal" in describe_commitments(resolved)


def test_direction_hold_does_not_veto_the_camera_intent_selection_watched():
    """A pre-selection preference must not turn a capable clip hold-only."""

    resolved = _resolved(_direction(
        motion_preference="hold", min_supported_seconds="0:03",
        presentation_intent="centered_hold",
    ))
    assert resolved.options[0].motion_preference == "native_first"
    faults = validate_selection_commitments([{
        "commitment_id": "hero", "span_id": "C1:s00",
        "seconds_needed": 3.0, "camera_intent": "push_in",
        "looks": [{
            "presentation_intent": "centered_hold",
            "entity_id": "device.fold",
        }],
    }], resolved)
    assert any("camera treatment" in fault for fault in faults)
    assert "偏好，不是限制" in describe_commitments(resolved)
    provider_fields = provider_commitment_schema(
        ["C1:s00"], ["device.fold"]
    )["items"]["properties"]
    assert "motion_preference" not in provider_fields
    assert {
        "recommended_treatment", "suggested_move", "camera_route",
        "motion_reason", "fallback_treatment",
    } <= set(provider_fields)


def test_direction_motion_advice_ranks_a_locally_feasible_treatment_only():
    material = [Material("C1", (
        Span("C1:s00", "C1", 0.0, 4.0, "wide logo", "locked"),
    ))]
    material[0].pan_room = 0.45
    direction = _direction(
        motion_preference="hold",
        recommended_treatment="reveal",
        suggested_move="pan",
        camera_route="read the wide logo from left to right, then settle",
        motion_reason="the 9:16 crop cannot show the full wide mark at once",
        fallback_treatment="compare",
        presentation_intent="partial_reveal",
        min_supported_seconds="0:03",
    )
    resolved = resolve_candidate_commitments(
        direction, material, material_digest="a" * 64,
        aspect="9:16", target_seconds=20.0,
        grounding_target_ids=("device.fold",), grounding_sha256="b" * 64,
    )
    option = resolved.options[0]
    assert option.direction_suggested_move == "pan"
    assert option.direction_treatment == "reveal"
    assert option.preferred_treatment == "reveal"
    assert "reveal" in option.feasible_treatments
    described = describe_commitments(resolved)
    assert "move=pan" in described
    assert "read the wide logo" in described


def test_sequential_read_can_preserve_verified_authored_source_motion():
    material = [Material("C1", (
        Span("C1:s00", "C1", 0.0, 4.0, "wide moving wordmark", "authored"),
    ))]
    direction = _direction(
        presentation_intent="sequential_read",
        recommended_treatment="use_source_motion",
        suggested_move="source_motion",
        camera_route="read the wide wordmark from left to right",
        motion_reason="the authored pan already performs the sequential read",
        fallback_treatment="reveal",
        min_supported_seconds="0:03",
    )

    resolved = resolve_candidate_commitments(
        direction, material, material_digest="a" * 64,
        aspect="9:16", target_seconds=20.0,
        grounding_target_ids=("device.fold",), grounding_sha256="b" * 64,
    )

    option = resolved.options[0]
    assert option.presentation_intent == "sequential_read"
    assert option.direction_treatment == "use_source_motion"
    assert option.preferred_treatment == "use_source_motion"
    assert "use_source_motion" in option.feasible_treatments


def test_commitment_presentation_is_projected_into_selection_locally():
    material = [Material("C1", (
        Span("C1:s00", "C1", 0.0, 4.0, "wide moving wordmark", "authored"),
    ))]
    commitments = resolve_candidate_commitments(
        _direction(
            presentation_intent="sequential_read",
            recommended_treatment="use_source_motion",
            min_supported_seconds="0:03",
        ),
        material, material_digest="a" * 64, aspect="9:16",
        target_seconds=20.0, grounding_target_ids=("device.fold",),
        grounding_sha256="b" * 64,
    )
    shots = [{
        "commitment_id": "hero", "span_id": "C1:s00",
        "seconds_needed": 3.0,
        "seconds_needed": 3.0, "camera_intent": "hold",
        "looks": [{
            "at": "the wordmark", "entity_id": "device.fold",
            "presentation_intent": "centered_hold", "must_be_whole": True,
        }],
    }]

    bind_selection_content_contracts(shots, commitments)

    assert shots[0]["looks"][0]["presentation_intent"] == "sequential_read"
    assert shots[0]["looks"][0]["must_be_whole"] is False
    assert validate_selection_commitments(shots, commitments) == []


def test_stable_visual_ids_are_projected_onto_readable_ordered_looks():
    option = CandidateOption(
        commitment_id="hero", purpose="show the interaction",
        required=True, picture_role="primary_action", target_id="none",
        span_id="C1:s00", min_supported_seconds=2.0,
        tier="primary", presentation_intent="sequential_read",
        motion_preference="virtual_allowed", why="read both in order",
        required_visuals=("v01", "v02"), visual_relationship="ordered",
    )
    commitments = CandidateCommitments(
        contract_version="candidate-commitment-v5-visual-relationships",
        material_digest="a" * 64, direction_sha256="b" * 64,
        grounding_sha256=None, target_aspect="9:16", target_seconds=20.0,
        options=(option,),
    )
    shots = [{
        "commitment_id": "hero", "span_id": "C1:s00",
        "seconds_needed": 3.0,
        "looks": [
            {"at": "phone screen", "includes": []},
            {"at": "LED result", "includes": []},
        ],
    }]

    bind_selection_content_contracts(shots, commitments)

    assert shots[0]["looks"][0]["includes"] == ["v01"]
    assert shots[0]["looks"][1]["includes"] == ["v02"]
    assert validate_selection_commitments(shots, commitments) == []


def test_native_motion_preserves_the_direction_sequential_read_contract():
    material = [Material("C1", (
        Span("C1:s00", "C1", 0.0, 4.0, "moving wordmark", "authored"),
    ))]
    commitments = resolve_candidate_commitments(
        _direction(
            presentation_intent="sequential_read",
            recommended_treatment="use_source_motion",
            min_supported_seconds="0:03",
        ),
        material, material_digest="a" * 64, aspect="9:16",
        target_seconds=20.0, grounding_target_ids=("device.fold",),
        grounding_sha256="b" * 64,
    )
    shots = [{
        "commitment_id": "hero", "span_id": "C1:s00",
        "seconds_needed": 3.0, "camera_intent": "use_source_motion",
        "looks": [{
            "at": "the wordmark", "entity_id": "device.fold",
            "presentation_intent": "sequential_read", "must_be_whole": False,
        }],
    }]

    bind_selection_content_contracts(shots, commitments)

    assert shots[0]["looks"][0]["presentation_intent"] == "sequential_read"
    assert validate_selection_commitments(shots, commitments) == []


def test_direction_schema_requires_a_generic_content_policy():
    fields = provider_commitment_schema(
        ["C1:s00"], ["device.fold"], ["a01"]
    )["items"]

    assert "content_policy" in fields["required"]
    assert "content_action_id" in fields["required"]
    assert fields["properties"]["content_action_id"]["enum"] == ["none", "a01"]
    assert set(fields["properties"]["content_policy"]["enum"]) == {
        "complete_action", "representative_excerpt", "result_hold",
        "continuous_process", "static_display",
    }
    assert "兩個相鄰 commitment" in fields["properties"]["content_policy"][
        "description"
    ]


def test_complete_action_policy_cannot_be_downgraded_to_an_intentional_cut():
    resolved = _resolved(_direction(
        content_policy="complete_action", content_action_id="a01",
    ))
    shot = {
        "commitment_id": "hero", "span_id": "C1:s00",
        "seconds_needed": 3.5, "camera_intent": "use_source_motion",
        "action_treatment": "intentional_cut",
        "looks": [{
            "presentation_intent": "reveal_endpoint",
            "entity_id": "device.fold",
        }],
    }

    faults = validate_selection_commitments([shot], resolved)

    assert any("complete_action" in fault and "complete_here" in fault for fault in faults)


def test_representative_excerpt_policy_allows_a_deliberate_early_cut():
    resolved = _resolved(_direction(
        content_policy="representative_excerpt", content_action_id="a01",
    ))
    shot = {
        "commitment_id": "hero", "span_id": "C1:s00",
        "seconds_needed": 3.5, "camera_intent": "use_source_motion",
        "action_id": "a01",
        "action_treatment": "intentional_cut",
        "looks": [{
            "presentation_intent": "reveal_endpoint",
            "entity_id": "device.fold",
        }],
    }

    faults = validate_selection_commitments([shot], resolved)

    assert not any("content policy" in fault for fault in faults)


def test_selected_candidate_projects_its_content_floor_without_model_repeating_it():
    from montagewright.candidate_commitments import bind_selection_content_contracts

    resolved = _resolved(_direction(
        content_policy="result_hold",
        min_supported_seconds="0:02.5",
    ))
    shot = {"commitment_id": "hero", "span_id": "C1:s00"}

    bind_selection_content_contracts([shot], resolved)

    assert shot["content_policy"] == "result_hold"
    assert shot["content_min_seconds"] == 2.5
    assert shot["content_purpose"] == "complete product reveal"


def test_complete_action_binds_local_action_duration_before_selection():
    from montagewright.candidate_commitments import bind_selection_content_contracts

    resolved = _resolved(_direction(
        content_policy="complete_action", content_action_id="a01",
        min_supported_seconds="0:01",
    ))
    option = resolved.options[0]
    shot = {"commitment_id": "hero", "span_id": "C1:s00"}

    bind_selection_content_contracts([shot], resolved)

    assert option.min_supported_seconds == 3.5
    assert option.content_action_start_seconds == 0.0
    assert option.content_action_complete_seconds == 3.5
    assert shot["action_id"] == "a01"
    assert shot["action_treatment"] == "complete_here"
    assert shot["content_min_seconds"] == 3.5


def test_complete_action_without_a_direction_bound_action_is_rejected():
    with pytest.raises(CommitmentError, match="content_action_id"):
        _resolved(_direction(
            content_policy="complete_action", content_action_id="none",
        ))


def test_impossible_direction_advice_does_not_expand_local_capability():
    direction = _direction(
        recommended_treatment="push_in",
        suggested_move="push_in",
    )
    resolved = _resolved(direction)
    option = resolved.options[0]
    assert option.direction_treatment == "push_in"
    assert "push_in" not in option.feasible_treatments
    assert option.preferred_treatment == "use_source_motion"


def test_direction_motion_advice_is_carried_to_the_reviewable_shot():
    from montagewright.cli import _annotate_selection_direction_motion

    commitments = _resolved(_direction(
        recommended_treatment="reveal",
        suggested_move="pan",
        camera_route="left logo to right product",
        motion_reason="the vertical crop should read the wide composition",
        fallback_treatment="compare",
    ))
    selection = {"shots": [{
        "commitment_id": "hero", "span_id": "C1:s00",
        "camera_intent": "use_source_motion",
    }]}

    _annotate_selection_direction_motion(selection, commitments)

    advice = selection["shots"][0]["direction_motion_advice"]
    assert advice["move"] == "pan"
    assert advice["route"] == "left logo to right product"
    assert "use_source_motion" in advice["locally_feasible"]
    assert selection["shots"][0]["agrees_with_direction"] is False
    assert selection["plan_disagreements"][0].startswith("k00:")


@pytest.mark.parametrize("change, message", [
    ({"span_id": "missing"}, "unknown span"),
    ({"min_supported_seconds": "0:05"}, "has 4.000s"),
    ({"target_id": "device.other"}, "unknown target"),
])
def test_local_facts_reject_provider_claims_the_material_cannot_execute(
    change, message
):
    with pytest.raises(CommitmentError, match=message):
        _resolved(_direction(**change))


def test_source_motion_fact_is_derived_locally_not_chosen_by_direction():
    resolved = _resolved(_direction(
        span_id="C1:s01", min_supported_seconds="0:03",
    ))
    assert resolved.options[0].motion_preference == "virtual_allowed"
    faults = validate_selection_commitments([{
        "commitment_id": "hero", "span_id": "C1:s01",
        "seconds_needed": 3.0, "camera_intent": "hold",
        "looks": [{
            "presentation_intent": "reveal_endpoint",
            "entity_id": "device.fold",
        }],
    }], resolved)
    assert not any("source motion" in fault for fault in faults)


def test_grounded_complete_end_hold_can_support_three_seconds():
    resolved = _resolved(_direction(
        span_id="C1:s01", picture_role="end_hold",
        min_supported_seconds="0:03", motion_preference="hold",
        presentation_intent="complete_hold",
    ))
    assert resolved.options[0].min_supported_seconds == 3.0


def test_candidate_correction_is_text_only_and_cannot_change_direction():
    replacement = _direction()["candidate_options"]
    client = _Client({
        "candidate_options": replacement,
        "repair_summary": "made the group coherent",
    })
    base = {
        **_direction(), "target_seconds": 29.0,
        "music_suggestion": "keep the existing musical arc",
    }
    material = [MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="foldable detail",
        spans=_material()[0].spans,
    )]
    corrected, _ = correct_candidate_options(
        base, material, fault="hero has conflicting purposes",
        grounding_target_ids=("device.fold",), client=client,
    )
    request = client.interactions.calls[0]
    assert [part["type"] for part in request["input"]] == ["text"]
    assert "video" not in json.dumps(request["input"])
    assert corrected["candidate_options"] == replacement
    assert {
        key: value for key, value in corrected.items()
        if key != "candidate_options"
    } == {
        key: value for key, value in base.items()
        if key != "candidate_options"
    }


def test_preferred_delivery_removes_only_proven_unsupported_tail():
    commitments = _resolved(_direction(min_supported_seconds="0:02"))
    chosen = {"shots": [{
        "commitment_id": "hero", "span_id": "C1:s00",
        "seconds_needed": 3.0, "picture_role": "primary_action",
    }]}
    audit = CoverageAudit(
        3.0, 2.0, 3.0,
        (CoverageEntry("k00", "primary_action", 3.0, 0.0, 3.0, 2.0),),
        ("one unsupported second",),
    )
    repairs = repair_preferred_unsupported_time(chosen, audit, commitments)
    assert chosen["shots"][0]["seconds_needed"] == 2.0
    assert repairs


def test_preferred_delivery_never_trims_below_commitment_minimum():
    commitments = _resolved(_direction(min_supported_seconds="0:03"))
    chosen = {"shots": [{
        "commitment_id": "hero", "span_id": "C1:s00",
        "seconds_needed": 3.0, "picture_role": "primary_action",
    }]}
    audit = CoverageAudit(
        3.0, 2.0, 3.0,
        (CoverageEntry("k00", "primary_action", 3.0, 0.0, 3.0, 2.0),),
        ("one unsupported second",),
    )
    assert not repair_preferred_unsupported_time(chosen, audit, commitments)
    assert chosen["shots"][0]["seconds_needed"] == 3.0


def test_each_commitment_requires_exactly_one_primary():
    direction = _direction()
    direction["candidate_options"][0]["tier"] = "alternate"
    with pytest.raises(ValueError, match="exactly one primary"):
        _resolved(direction)


def test_conflicting_group_flags_are_a_retryable_commitment_error():
    direction = _direction()
    alternate = dict(direction["candidate_options"][0])
    alternate.update({
        "span_id": "C1:s01",
        "tier": "alternate",
        "required": False,
        "motion_preference": "hold",
        "min_supported_seconds": "0:03",
    })
    direction["candidate_options"].append(alternate)
    with pytest.raises(CommitmentError, match="conflicting required flags"):
        _resolved(direction)


def test_selection_must_use_an_option_and_fulfil_every_required_commitment():
    commitments = _resolved()
    assert validate_selection_commitments(
        [{
            "commitment_id": "hero", "span_id": "C1:s00",
            "seconds_needed": 3.5, "camera_intent": "use_source_motion",
            "looks": [{
                "presentation_intent": "reveal_endpoint",
                "entity_id": "device.fold",
            }],
        }], commitments
    ) == []
    faults = validate_selection_commitments(
        [{"commitment_id": "hero", "span_id": "C1:s01"}], commitments
    )
    assert "outside commitment hero" in faults[0]
    assert "appears 0 times" in validate_selection_commitments([], commitments)[0]


def test_direction_cannot_commit_to_a_source_it_also_ruled_broken():
    with pytest.raises(CommitmentError, match="ruled broken"):
        resolve_candidate_commitments(
            _direction(), _material(), material_digest="a" * 64,
            aspect="9:16", target_seconds=20.0,
            grounding_target_ids=("device.fold",),
            excluded_source_ids=("C1",),
        )


def test_review_replacement_may_change_span_but_not_the_story_promise():
    direction = _direction()
    alternate = dict(direction["candidate_options"][0])
    alternate.update({
        "span_id": "C1:s01", "tier": "alternate",
        "motion_preference": "hold", "presentation_intent": "centered_hold",
        "min_supported_seconds": "0:02",
    })
    direction["candidate_options"].append(alternate)
    commitments = _resolved(direction)
    failing = [(2, {"commitment_id": "hero", "span_id": "C1:s00"}, "bad")]
    assert validate_replacement_commitments(
        failing,
        [{
            "replace_clip_id": "k02", "commitment_id": "hero",
            "span_id": "C1:s01",
        }],
        commitments,
    ) == []
    faults = validate_replacement_commitments(
        failing,
        [{
            "replace_clip_id": "k02", "commitment_id": "other",
            "span_id": "C1:s01",
        }],
        commitments,
    )
    assert "changed commitment" in faults[0]


def test_target_and_presentation_must_belong_to_the_same_look():
    commitments = _resolved()
    shot = {
        "commitment_id": "hero", "span_id": "C1:s00",
        "seconds_needed": 3.5, "camera_intent": "use_source_motion",
        "looks": [{
            "entity_id": "device.fold",
            "presentation_intent": "partial_reveal",
        }, {
            "entity_id": "none",
            "presentation_intent": "reveal_endpoint",
        }],
    }

    faults = validate_selection_commitments([shot], commitments)

    assert any(
        "reveal_endpoint on target device.fold" in fault for fault in faults
    )


def test_review_replacement_cannot_bypass_local_camera_capability():
    commitments = _resolved()
    failing = [(0, {"commitment_id": "hero", "span_id": "C1:s00"}, "bad")]
    faults = validate_replacement_commitments(
        failing,
        [{
            "replace_clip_id": "k00",
            "commitment_id": "hero",
            "span_id": "C1:s00",
            "seconds_needed": 3.5,
            "camera_intent": "push_in",
        }],
        commitments,
    )
    assert any("camera treatment" in fault for fault in faults)


def test_selection_must_execute_duration_motion_presentation_and_target():
    commitments = _resolved()
    shot = {
        "commitment_id": "hero", "span_id": "C1:s00",
        "seconds_needed": 1.0, "camera_intent": "hold",
        "looks": [{"presentation_intent": "partial_reveal", "entity_id": None}],
    }
    faults = validate_selection_commitments([shot], commitments)
    assert any("content needs 3.500s" in fault for fault in faults)
    assert not any("source motion" in fault for fault in faults)
    assert any("presentation intent reveal_endpoint" in fault for fault in faults)
    assert any("bind target device.fold" in fault for fault in faults)


@pytest.mark.parametrize("role", ["reaction", "end_hold"])
def test_role_default_rejects_a_commitment_minimum_above_local_evidence(role):
    direction = _direction(
        span_id="C1:s01",
        picture_role=role,
        min_supported_seconds="0:03",
        motion_preference="hold",
        presentation_intent="complete_hold",
        target_id="none",
    )
    with pytest.raises(CommitmentError, match="locally supported"):
        _resolved(direction)


def test_measured_four_second_visual_action_overrides_the_role_default():
    material = [Material(
        "C1",
        (Span("C1:s00", "C1", 0.0, 4.0, "reaction", "locked"),),
        action=("visible reaction continues 0.0-4.0s",),
    )]
    direction = _direction(
        picture_role="reaction",
        min_supported_seconds="0:04",
        motion_preference="hold",
        presentation_intent="complete_hold",
        target_id="none",
    )
    resolved = resolve_candidate_commitments(
        direction, material, material_digest="a" * 64,
        aspect="9:16", target_seconds=4.0,
    )
    assert resolved.options[0].min_supported_seconds == 4.0


def test_local_hold_repair_never_shortens_below_candidate_minimum():
    from montagewright.coverage import repair_bounded_visual_holds

    material = [Material(
        "C1",
        (Span("C1:s00", "C1", 0.0, 4.0, "reaction", "locked"),),
        action=("visible reaction continues 0.0-4.0s",),
    )]
    direction = _direction(
        picture_role="reaction",
        min_supported_seconds="0:02.5",
        motion_preference="hold",
        presentation_intent="complete_hold",
        target_id="none",
    )
    commitments = resolve_candidate_commitments(
        direction, material, material_digest="a" * 64,
        aspect="9:16", target_seconds=4.0,
    )
    shot = {
        "commitment_id": "hero", "span_id": "C1:s00",
        "source_id": "C1", "seconds_needed": 4.0,
        "picture_role": "reaction", "audio_role": "discard",
    }
    repairs = repair_bounded_visual_holds({"shots": [shot]}, commitments)
    assert shot["seconds_needed"] == 2.5
    assert "to 2.50s" in repairs[0]


def test_local_hold_repair_preserves_a_longer_measured_visual_action():
    from montagewright.coverage import repair_bounded_visual_holds

    material = Material(
        "C1",
        (Span("C1:s00", "C1", 0.0, 4.0, "reaction", "locked"),),
        action=("visible reaction continues 0.0-4.0s",),
    )
    shot = {
        "source_id": "C1", "start_seconds": 0.0, "seconds_needed": 4.0,
        "picture_role": "reaction", "audio_role": "discard",
    }
    repairs = repair_bounded_visual_holds(
        {"shots": [shot]}, material=[material]
    )
    assert repairs == ()
    assert shot["seconds_needed"] == 4.0


def test_the_planner_is_told_the_budget_it_will_be_judged_against():
    """Three paid corrections in a row asked for a three-second transition.

    Which is an ordinary connective shot in a music cut and a contract
    violation here: `transition` carries 0.50s of visual-only time, and with
    no narration every second of a shot is visual-only. The schema asked for
    a role and a length while saying nothing about how long each role can
    hold, so the model could not have known -- and the sentence is generated
    from the table that does the refusing, so they cannot drift apart.
    """

    from montagewright.candidate_commitments import provider_commitment_schema
    from montagewright.coverage import VISUAL_ONLY_LIMITS

    schema = provider_commitment_schema(["C1:s00"], ["device.fold"])
    role = schema["items"]["properties"]["picture_role"]["description"]
    length = schema["items"]["properties"]["min_supported_seconds"]["description"]

    for name, limit in VISUAL_ONLY_LIMITS.items():
        assert name in role, f"{name} has a ceiling the planner never sees"
        if limit is not None:
            assert f"{limit:.2f}" in role, f"{name}'s ceiling is not stated"
    assert "transition" in length, "name the trap that was actually fallen into"


def test_exact_hard_negative_primary_promotes_the_surviving_alternate():
    from montagewright.cli import _commitments_without_exact_hard_negatives

    resolved = _resolved()
    primary = resolved.options[0]
    context = resolved.model_copy(update={
        "options": (
            primary,
            primary.model_copy(update={
                "span_id": "C2:s00", "tier": "alternate",
            }),
        ),
    })
    pruned = _commitments_without_exact_hard_negatives(
        context,
        {(primary.span_id.split(":", 1)[0], primary.target_id): {
            "status": "hard_negative", "reason": "wrong instance",
        }},
    )

    assert len(pruned.options) == 1
    assert pruned.options[0].span_id == "C2:s00"
    assert pruned.options[0].tier == "primary"


def test_screen_negative_promotion_requires_exact_confirmation():
    from montagewright.cli import _commitments_without_exact_hard_negatives

    resolved = _resolved()
    option = resolved.options[0]
    pair = ("C1", option.target_id)

    degraded = _commitments_without_exact_hard_negatives(
        resolved,
        {pair: {"status": "uncertain", "reason": "blurred exact seed"}},
        require_confirmation={pair},
    )
    assert degraded.options[0].target_id == "none"
    assert degraded.options[0].identity_status == "needs_review"
    assert degraded.options[0].identity_issue
    assert any("ungrounded" in warning for warning in degraded.warnings)

    confirmed = _commitments_without_exact_hard_negatives(
        resolved,
        {pair: {"status": "confirmed", "reason": "master frame matched"}},
        require_confirmation={pair},
    )
    assert confirmed.options == resolved.options


# The delivery crop for 16:9 material at 9:16, which is what both runs used
# and the width every travel treatment has to be measured against.
VERTICAL_CROP = 0.3164


def _roomy_source(**change):
    """A take with plenty of frame to move in, which is the normal case.

    A 16:9 source delivered 9:16 has about 68% travel room on every clip
    ever shot, so frame room alone can never be the reason a reveal is
    withheld.
    """

    item = SimpleNamespace(
        pan_room=0.6836, tilt_room=0.0, push_room=1.6,
        crop_width=VERTICAL_CROP,
    )
    item.__dict__.update(change)
    return item


def _travel_span(seconds: float = 4.0):
    return Span("C1:s00", "C1", 0.0, seconds, "detail", "locked")


def test_a_subject_the_crop_already_contains_is_offered_no_travel():
    """Frame room and something to cross it for are different facts.

    A phone measuring 0.19 of the frame sits whole inside a 0.32 crop. A
    reveal across it has nowhere to go, so the compiler holds -- correctly --
    and the film carries a review note for a move that was never possible.
    Withhold it from the menu instead, before Selection is paid to choose it.
    """

    treatments, _, _, reason = _camera_treatments(
        _roomy_source(), _travel_span(),
        preference="native_first", target_id="none",
        content_extent=0.19,
    )

    assert "reveal" not in treatments
    assert "compare" not in treatments
    assert "multi_stop" not in treatments
    assert "nothing to travel across" in reason
    # A push is about resolution, not width, and a contained subject is
    # exactly what a push in is for.
    assert "push_in" in treatments
    assert "hold" in treatments


def test_content_wider_than_the_crop_keeps_every_travel_treatment():
    treatments, _, _, reason = _camera_treatments(
        _roomy_source(), _travel_span(),
        preference="native_first", target_id="none",
        content_extent=0.40,
    )

    assert {"reveal", "compare", "multi_stop"} <= set(treatments)
    assert "measured travel room" in reason


def test_unmeasured_content_never_loses_a_treatment():
    """Unknown geometry is not evidence that a move is impossible.

    Cards do not always place every named visual. Treating that silence as
    "too narrow" would be a new way to lose moves the footage supports, and
    the failure would look exactly like the one being fixed.
    """

    treatments, _, _, _ = _camera_treatments(
        _roomy_source(), _travel_span(),
        preference="native_first", target_id="none",
        content_extent=None,
    )

    assert {"reveal", "compare", "multi_stop"} <= set(treatments)


def test_readable_extent_spans_every_named_visual_or_declines():
    geometry = (
        ("left phone", None, 0.20, 0.5, 0.10, 0.4),
        ("right phone", None, 0.80, 0.5, 0.10, 0.4),
    )

    # Edge to edge across both, not the width of either.
    assert readable_extent(
        object(), geometry, ["v01", "v02"]
    ) == pytest.approx(0.70)
    assert readable_extent(object(), geometry, ["v01"]) == pytest.approx(0.10)
    # A visual the card cannot place makes the whole span unmeasurable.
    assert readable_extent(object(), geometry, ["v01", "v09"]) is None
    assert readable_extent(object(), (), ["v01"]) is None
    assert readable_extent(object(), geometry, []) is None


def test_a_read_across_contained_content_becomes_a_hold_not_a_lost_option():
    """Direction may promise a read the delivery crop cannot need.

    Withholding travel for contained content has a second effect: an option
    Direction marked `sequential_read` then has no readable treatment left.
    Dropping it would shrink the pool over a framing detail and can cost a
    commitment its only take, so the option survives as the composition it
    actually is.
    """

    narrow = [MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="one handset, centred",
        spans=_material()[0].spans, pan_room=0.6836, tilt_room=0.0,
        push_room=1.4, crop_width=VERTICAL_CROP,
        subject_geometry=(("the handset", None, 0.5, 0.5, 0.19, 0.55),),
    )]
    resolved = resolve_candidate_commitments(
        _direction(
            span_id="C1:s01", min_supported_seconds="0:03",
            presentation_intent="sequential_read",
            recommended_treatment="reveal",
            required_visuals=["v01"],
            visual_relationship="single",
        ), narrow,
        material_digest="a" * 64, aspect="9:16", target_seconds=20.0,
        grounding_target_ids=("device.fold",), grounding_sha256="b" * 64,
    )

    assert len(resolved.options) == 1
    option = resolved.options[0]
    assert option.presentation_intent == "centered_hold"
    assert not {"reveal", "compare", "multi_stop"} & set(
        option.feasible_treatments
    )
    assert "nothing to travel across" in option.feasibility_reason
    assert "held composition" in option.feasibility_reason


def test_a_read_across_wide_content_keeps_its_reader():
    wide = [MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="a row of handsets",
        spans=_material()[0].spans, pan_room=0.6836, tilt_room=0.0,
        push_room=1.4, crop_width=VERTICAL_CROP,
        subject_geometry=(
            ("left handset", None, 0.20, 0.5, 0.12, 0.5),
            ("right handset", None, 0.80, 0.5, 0.12, 0.5),
        ),
    )]
    resolved = resolve_candidate_commitments(
        _direction(
            span_id="C1:s01", min_supported_seconds="0:03",
            presentation_intent="sequential_read",
            recommended_treatment="reveal",
            required_visuals=["v01", "v02"],
            visual_relationship="ordered",
        ), wide,
        material_digest="a" * 64, aspect="9:16", target_seconds=20.0,
        grounding_target_ids=("device.fold",), grounding_sha256="b" * 64,
    )

    option = resolved.options[0]
    assert option.presentation_intent == "sequential_read"
    assert option.preferred_treatment == "reveal"
    assert {"reveal", "compare", "multi_stop"} <= set(
        option.feasible_treatments
    )
