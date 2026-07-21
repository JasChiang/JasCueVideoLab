from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jascue_video_lab.models import (
    AspectConstraint,
    ContentMap,
    DirectVideoGroundingProposal,
    DirectMomentMap,
    EvidenceClaimSource,
    EvidenceQueryLock,
    EvidenceQueryProvenance,
    EvidenceQueryTargetRef,
    EntityKind,
    FeatureEditPlan,
    GeminiNativeDirectVideoGroundingProposal,
    GeminiNativeGroundingProposal,
    GeminiNativeSegmentationCandidate,
    GeminiNativeSegmentationProposal,
    GroundingCandidate,
    GroundingProposal,
    MatchStatus,
    Occlusion,
    PredicateStatus,
    RushesEditPlan,
    TargetCandidateMap,
    TemporalMap,
)
from jascue_video_lab.schema import gemini_response_schema


def test_content_map_round_trip(content_map: ContentMap) -> None:
    reparsed = ContentMap.model_validate_json(content_map.model_dump_json())
    assert reparsed == content_map
    assert reparsed.events[0].boundary_precision.value == "coarse"


def test_frame_accurate_boundary_is_rejected(content_map: ContentMap) -> None:
    payload = content_map.model_dump(mode="json")
    payload["events"][0]["boundary_precision"] = "frame_accurate"
    with pytest.raises(ValidationError):
        ContentMap.model_validate(payload)


def test_keyframe_must_be_inside_half_open_interval(content_map: ContentMap) -> None:
    payload = content_map.model_dump(mode="json")
    payload["events"][0]["recommended_keyframe_ms"] = 10_000
    with pytest.raises(ValidationError):
        ContentMap.model_validate(payload)


def test_unknown_entity_reference_is_rejected(content_map: ContentMap) -> None:
    payload = content_map.model_dump(mode="json")
    payload["events"][0]["entity_ids"] = ["not-real"]
    with pytest.raises(ValidationError):
        ContentMap.model_validate(payload)


def test_invisible_grounding_requires_empty_candidates(provenance) -> None:
    with pytest.raises(ValidationError):
        GroundingProposal(
            asset_id="sha256:" + "a" * 64,
            event_id="event-1",
            entity_id="phone-1",
            frame_pts=100,
            frame_time_ms=4000,
            frame_hash="b" * 64,
            source_width=1280,
            source_height=720,
            visible=False,
            occlusion=Occlusion.UNKNOWN,
            visibility_reason="not visible",
            candidates=[
                GroundingCandidate(
                    box_2d=(100, 100, 200, 200),
                    label="guessed phone",
                    confidence=0.1,
                    disambiguation_reason="guess",
                )
            ],
            model_provenance=provenance,
        )


def _generic_grounding_payload(provenance) -> dict:
    return {
        "asset_id": "sha256:" + "a" * 64,
        "event_id": "event-1",
        "entity_id": "subject-1",
        "frame_pts": 100,
        "frame_time_ms": 4000,
        "frame_hash": "b" * 64,
        "source_width": 1280,
        "source_height": 720,
        "visible": True,
        "occlusion": "none",
        "visibility_reason": "the selected subject is directly visible",
        "candidates": [
            {
                "box_2d": [100, 100, 200, 200],
                "label": "selected subject",
                "confidence": 0.8,
                "disambiguation_reason": "matches the locked visual attributes",
            }
        ],
        "model_provenance": provenance.model_dump(mode="json"),
    }


def test_legacy_grounding_derives_orthogonal_statuses(provenance) -> None:
    proposal = GroundingProposal.model_validate(_generic_grounding_payload(provenance))
    assert proposal.match_status == MatchStatus.MATCHED
    assert proposal.predicate_status == PredicateStatus.NOT_APPLICABLE


def test_legacy_multi_candidate_grounding_is_fail_closed_as_ambiguous(provenance) -> None:
    payload = _generic_grounding_payload(provenance)
    payload["candidates"].append(
        {
            "box_2d": [300, 100, 400, 200],
            "label": "similar subject",
            "confidence": 0.7,
            "disambiguation_reason": "also satisfies the available visual attributes",
        }
    )
    proposal = GroundingProposal.model_validate(payload)
    assert proposal.match_status == MatchStatus.AMBIGUOUS

    payload["match_status"] = "matched"
    with pytest.raises(ValidationError, match="exactly one candidate"):
        GroundingProposal.model_validate(payload)


def test_ambiguous_grounding_is_not_promoted_to_matched(provenance) -> None:
    payload = _generic_grounding_payload(provenance)
    payload["match_status"] = "ambiguous"
    payload["predicate_status"] = "indeterminate"
    proposal = GroundingProposal.model_validate(payload)
    assert proposal.match_status == MatchStatus.AMBIGUOUS
    assert proposal.predicate_status == PredicateStatus.INDETERMINATE


@pytest.mark.parametrize(
    "match_status",
    ["not_visible", "target_mismatch", "insufficient_evidence"],
)
def test_nonmatched_status_cannot_keep_visible_geometry(
    provenance, match_status: str
) -> None:
    payload = _generic_grounding_payload(provenance)
    payload["match_status"] = match_status
    with pytest.raises(ValidationError, match="requires visible=false and no candidates"):
        GroundingProposal.model_validate(payload)


def test_predicate_result_does_not_change_target_match(provenance) -> None:
    payload = _generic_grounding_payload(provenance)
    payload["predicate_status"] = "not_satisfied"
    proposal = GroundingProposal.model_validate(payload)
    assert proposal.match_status == MatchStatus.MATCHED
    assert proposal.predicate_status == PredicateStatus.NOT_SATISFIED


@pytest.mark.parametrize(
    "model",
    [
        GroundingProposal,
        GeminiNativeGroundingProposal,
        GeminiNativeSegmentationProposal,
        DirectVideoGroundingProposal,
        GeminiNativeDirectVideoGroundingProposal,
    ],
)
def test_all_grounding_contracts_expose_match_and_predicate_status(model) -> None:
    assert "match_status" in model.model_fields
    assert "predicate_status" in model.model_fields


def _evidence_query_lock() -> EvidenceQueryLock:
    return EvidenceQueryLock(
        query_id="query:001",
        revision=2,
        editorial_goal="Show a complete observable interaction and its result.",
        targets=[
            EvidenceQueryTargetRef(
                target_id="subject.primary",
                target_description="the selected foreground subject",
                positive_attributes=["foreground", "selected by reviewer"],
                negative_attributes=["background depiction", "reflection"],
                reference_frame_ids=["DF000123"],
                reference_crop_hashes=["c" * 64],
            )
        ],
        observable_predicate="the interaction reaches a directly visible result state",
        required_evidence=["the selected subject is visible", "the result state is visible"],
        negative_constraints=["do not substitute a visually similar instance"],
        editing_uses=["demonstration", "portrait_reframe"],
        aspect_constraints=[
            AspectConstraint(
                aspect_ratio="9:16",
                required_target_ids=["subject.primary"],
                constraint="keep the selected subject inside the visible frame",
            )
        ],
        claim_source=EvidenceClaimSource.USER_BRIEF,
        provenance=EvidenceQueryProvenance(
            created_at="2026-07-21T00:00:00Z",
            created_by="reviewer",
            source_reference="brief:001",
        ),
    )


def test_evidence_query_lock_is_domain_neutral_and_hash_stable() -> None:
    lock = _evidence_query_lock()
    reparsed = EvidenceQueryLock.model_validate_json(lock.model_dump_json())
    assert reparsed == lock
    assert reparsed.definition_sha256() == lock.definition_sha256()
    assert len(lock.definition_sha256()) == 64


def test_checked_in_query_lock_example_matches_contract() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "evidence-query-lock.json"
    lock = EvidenceQueryLock.model_validate_json(path.read_text(encoding="utf-8"))
    assert lock.targets[0].target_id == "subject.primary"
    assert len(lock.definition_sha256()) == 64


def test_entity_kinds_cover_generic_non_product_targets() -> None:
    assert EntityKind("animal") is EntityKind.ANIMAL
    assert EntityKind("object") is EntityKind.OBJECT
    assert EntityKind("device") is EntityKind.DEVICE
    assert EntityKind("vehicle") is EntityKind.VEHICLE


def test_evidence_query_lock_rejects_unknown_aspect_target() -> None:
    payload = _evidence_query_lock().model_dump(mode="json")
    payload["aspect_constraints"][0]["required_target_ids"] = ["subject.unknown"]
    with pytest.raises(ValidationError, match="unknown targets"):
        EvidenceQueryLock.model_validate(payload)


def test_evidence_query_lock_rejects_conflicting_target_attributes() -> None:
    payload = _evidence_query_lock().model_dump(mode="json")
    payload["targets"][0]["negative_attributes"].append("Foreground")
    with pytest.raises(ValidationError, match="must not overlap"):
        EvidenceQueryLock.model_validate(payload)


def test_predicate_phases_require_a_locked_observable_predicate() -> None:
    payload = _evidence_query_lock().model_dump(mode="json")
    payload["observable_predicate"] = None
    payload["predicate_phases"] = {
        "precondition": "the state is absent",
        "apex": "the requested change is directly visible",
        "postcondition": "the resulting state remains visible",
    }
    with pytest.raises(ValidationError, match="require observable_predicate"):
        EvidenceQueryLock.model_validate(payload)


@pytest.mark.parametrize(
    "box",
    [(-1, 0, 100, 100), (0, 0, 1001, 100), (100, 100, 100, 200), (100, 300, 200, 200)],
)
def test_invalid_normalized_boxes_are_rejected(box) -> None:
    with pytest.raises(ValidationError):
        GroundingCandidate(
            box_2d=box,
            label="phone",
            confidence=0.8,
            disambiguation_reason="central phone",
        )


def test_api_schema_uses_only_supported_constraint_keywords() -> None:
    schema_text = str(gemini_response_schema(GroundingProposal))
    rushes_schema_text = str(gemini_response_schema(RushesEditPlan))
    feature_schema_text = str(gemini_response_schema(FeatureEditPlan))
    for unsupported in (
        "const",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "pattern",
        "default",
        "minItems",
        "maxItems",
    ):
        assert unsupported not in schema_text
        assert unsupported not in rushes_schema_text
        assert unsupported not in feature_schema_text
    assert "prefixItems" in schema_text


def test_native_grounding_schema_names_y_first_field_explicitly() -> None:
    schema_text = str(gemini_response_schema(GeminiNativeGroundingProposal))
    assert "box_2d_yxyx" in schema_text
    assert "box_2d'" not in schema_text


def test_native_segmentation_schema_preserves_bbox_and_polygon_orders() -> None:
    schema_text = str(gemini_response_schema(GeminiNativeSegmentationProposal))
    assert "box_2d_yxyx" in schema_text
    assert "mask" in schema_text
    candidate = GeminiNativeSegmentationCandidate(
        box_2d_yxyx=(100, 200, 800, 900),
        mask=[(200, 100), (900, 100), (900, 800), (200, 800)],
        label="phone",
        confidence=0.9,
        disambiguation_reason="requested instance",
    )
    assert candidate.mask[0] == (200, 100)


def test_native_segmentation_rejects_degenerate_polygon() -> None:
    with pytest.raises(ValidationError, match="non-zero area"):
        GeminiNativeSegmentationCandidate(
            box_2d_yxyx=(100, 200, 800, 900),
            mask=[(200, 100), (300, 200), (400, 300)],
            label="phone",
            confidence=0.9,
            disambiguation_reason="collinear points",
        )


def test_native_segmentation_rejects_polygon_outside_bbox() -> None:
    with pytest.raises(ValidationError, match="inside its bounding box"):
        GeminiNativeSegmentationCandidate(
            box_2d_yxyx=(100, 200, 800, 900),
            mask=[(100, 100), (900, 100), (900, 800), (100, 800)],
            label="phone",
            confidence=0.9,
            disambiguation_reason="polygon includes another object",
        )


def test_native_segmentation_rejects_self_intersection() -> None:
    with pytest.raises(ValidationError, match="must not self-intersect"):
        GeminiNativeSegmentationCandidate(
            box_2d_yxyx=(100, 100, 900, 900),
            mask=[(200, 200), (800, 800), (800, 200), (200, 800)],
            label="phone",
            confidence=0.9,
            disambiguation_reason="bow-tie polygon",
        )


def test_temporal_map_rejects_event_past_duration(content_map: ContentMap) -> None:
    event = content_map.events[0]
    payload = {
        "asset_id": content_map.asset_id,
        "duration_ms": 10_000,
        "summary": "test",
        "events": [
            {
                "event_id": event.event_id,
                "start_ms": 9_000,
                "end_ms": 11_000,
                "label": event.label,
                "observable_evidence": event.description,
                "recommended_keyframe_ms": 9_500,
                "keyframe_reason": event.keyframe_reason,
                "confidence": event.confidence,
                "boundary_precision": event.boundary_precision,
            }
        ],
        "uncertainties": [],
        "model_provenance": content_map.model_provenance,
    }
    with pytest.raises(ValidationError):
        TemporalMap.model_validate(payload)


def _direct_moment_payload(content_map: ContentMap) -> dict:
    return {
        "asset_id": content_map.asset_id,
        "duration_ms": 61_862,
        "summary": "salient screenshot anchors",
        "moments": [
            {
                "moment_id": "moment-01",
                "timestamp_mmss": "00:04",
                "label": "MacBook screen",
                "observable_evidence": "The laptop display is clearly visible.",
                "grounding_target_id": "macbook-screen",
                "grounding_target_description": "the visible MacBook screen",
                "confidence": 0.9,
            },
            {
                "moment_id": "moment-02",
                "timestamp_mmss": "00:19",
                "label": "AirDrop on iPhone",
                "observable_evidence": "The yellow iPhone screen is visible.",
                "grounding_target_id": "yellow-iphone-screen",
                "grounding_target_description": "the yellow iPhone screen",
                "confidence": 0.9,
            },
        ],
        "uncertainties": [],
        "model_provenance": content_map.model_provenance.model_dump(mode="json"),
    }


def test_direct_moments_accept_exact_mmss_below_duration(content_map: ContentMap) -> None:
    parsed = DirectMomentMap.model_validate(_direct_moment_payload(content_map))
    assert [moment.timestamp_mmss for moment in parsed.moments] == ["00:04", "00:19"]


@pytest.mark.parametrize("timestamp", ["00:62", "0:04", "01:02"])
def test_direct_moments_reject_bad_or_out_of_range_mmss(
    content_map: ContentMap, timestamp: str
) -> None:
    payload = _direct_moment_payload(content_map)
    payload["moments"] = [payload["moments"][0] | {"timestamp_mmss": timestamp}]
    with pytest.raises(ValidationError):
        DirectMomentMap.model_validate(payload)


def _target_candidate_payload(content_map: ContentMap) -> dict:
    return {
        "asset_id": content_map.asset_id,
        "duration_ms": 22_022,
        "summary": "Selectable visible objects",
        "candidates": [
            {
                "candidate_id": "center-purple-phone",
                "label": "中央紫色手機",
                "entity_kind": "phone",
                "target_description": "中央偏左、背面朝向鏡頭的紫色手機；排除背板與白色手機。",
                "distinguishing_features": "紫色、中央偏左、實體手機",
                "representative_timestamp_mmss": "00:10",
                "selection_reason": "跨鏡頭可見且邊界清楚",
                "confidence": 0.95,
            }
        ],
        "uncertainties": [],
        "model_provenance": content_map.model_provenance.model_dump(mode="json"),
    }


def test_target_candidate_map_accepts_selectable_target(content_map: ContentMap) -> None:
    parsed = TargetCandidateMap.model_validate(_target_candidate_payload(content_map))
    assert parsed.candidates[0].candidate_id == "center-purple-phone"
    assert parsed.candidates[0].entity_kind.value == "phone"


def test_target_candidate_map_rejects_duplicate_ids(content_map: ContentMap) -> None:
    payload = _target_candidate_payload(content_map)
    payload["candidates"].append(dict(payload["candidates"][0]))
    with pytest.raises(ValidationError):
        TargetCandidateMap.model_validate(payload)


@pytest.mark.parametrize("timestamp", ["00:23", "0:10", "00:61"])
def test_target_candidate_map_rejects_invalid_representative_time(
    content_map: ContentMap, timestamp: str
) -> None:
    payload = _target_candidate_payload(content_map)
    payload["candidates"][0]["representative_timestamp_mmss"] = timestamp
    with pytest.raises(ValidationError):
        TargetCandidateMap.model_validate(payload)


def test_direct_video_grounding_cannot_claim_exact_frame_identity(provenance) -> None:
    payload = {
        "asset_id": "sha256:" + "a" * 64,
        "event_id": "moment-01",
        "entity_id": "phone-1",
        "requested_timestamp_mmss": "00:08",
        "reference_frame_status": "exact_ffmpeg_frame",
        "reference_frame_description": "unknown sample",
        "visible": False,
        "occlusion": "unknown",
        "visibility_reason": "not observed",
        "candidates": [],
        "model_provenance": provenance.model_dump(mode="json"),
    }
    with pytest.raises(ValidationError):
        DirectVideoGroundingProposal.model_validate(payload)
