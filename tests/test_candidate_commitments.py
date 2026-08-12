from __future__ import annotations

from dataclasses import dataclass

import pytest

from montagewright.candidate_commitments import (
    CommitmentError,
    resolve_candidate_commitments,
    validate_replacement_commitments,
    validate_selection_commitments,
)
from montagewright.spans import Span


@dataclass
class Material:
    source_id: str
    spans: tuple[Span, ...]
    action: tuple[str, ...] = ()
    motion: tuple[object, ...] = ()
    camera_moves: bool = False


def _material():
    return [Material("C1", (
        Span("C1:s00", "C1", 0.0, 4.0, "reveal", "authored"),
        Span("C1:s01", "C1", 4.0, 8.0, "detail", "locked"),
    ))]


def _direction(**change):
    option = {
        "commitment_id": "hero", "purpose": "complete product reveal",
        "required": True, "picture_role": "primary_action",
        "span_id": "C1:s00", "tier": "primary",
        "min_supported_seconds": "0:03.5",
        "presentation_intent": "reveal_endpoint",
        "motion_preference": "native_first", "target_id": "device.fold",
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


@pytest.mark.parametrize("change, message", [
    ({"span_id": "missing"}, "unknown span"),
    ({"min_supported_seconds": "0:05"}, "has 4.000s"),
    ({"target_id": "device.other"}, "unknown target"),
    ({"span_id": "C1:s01"}, "requests native motion"),
])
def test_local_facts_reject_provider_claims_the_material_cannot_execute(
    change, message
):
    with pytest.raises(CommitmentError, match=message):
        _resolved(_direction(**change))


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


def test_selection_must_execute_duration_motion_presentation_and_target():
    commitments = _resolved()
    shot = {
        "commitment_id": "hero", "span_id": "C1:s00",
        "seconds_needed": 1.0, "camera_intent": "hold",
        "looks": [{"presentation_intent": "partial_reveal", "entity_id": None}],
    }
    faults = validate_selection_commitments([shot], commitments)
    assert any("content needs 3.500s" in fault for fault in faults)
    assert any("authored source motion" in fault for fault in faults)
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
