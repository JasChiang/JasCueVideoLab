from __future__ import annotations

import pytest
from pydantic import ValidationError

from jascue_video_lab.models import (
    ContentMap,
    DirectMomentMap,
    GeminiNativeGroundingProposal,
    GroundingCandidate,
    GroundingProposal,
    Occlusion,
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
    for unsupported in ("const", "exclusiveMinimum", "exclusiveMaximum", "pattern", "default"):
        assert unsupported not in schema_text
    assert "prefixItems" in schema_text


def test_native_grounding_schema_names_y_first_field_explicitly() -> None:
    schema_text = str(gemini_response_schema(GeminiNativeGroundingProposal))
    assert "box_2d_yxyx" in schema_text
    assert "box_2d'" not in schema_text


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
