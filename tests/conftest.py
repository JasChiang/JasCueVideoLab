from __future__ import annotations

import pytest

from jascue_video_lab.models import (
    BoundaryPrecision,
    ContentMap,
    Entity,
    EntityKind,
    Event,
    EvidenceModality,
    ModelProvenance,
)


@pytest.fixture
def provenance() -> ModelProvenance:
    return ModelProvenance(
        model_id="gemini-3.5-flash",
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version="2.3.0",
        interaction_id="interaction-1",
        run_id="run-01",
        generated_at="2026-07-20T00:00:00+00:00",
    )


@pytest.fixture
def content_map(provenance: ModelProvenance) -> ContentMap:
    return ContentMap(
        asset_id="sha256:" + "a" * 64,
        duration_ms=10_000,
        summary="A phone is operated.",
        content_type="silent phone demonstration",
        entities=[
            Entity(
                entity_id="phone-1",
                kind=EntityKind.PHONE,
                label="operated phone",
                distinguishing_features="central dark phone",
                evidence="visible in the center",
            )
        ],
        events=[
            Event(
                event_id="event-1",
                start_ms=0,
                end_ms=10_000,
                label="operate phone",
                description="A hand operates the central phone.",
                evidence_modalities=EvidenceModality.VISUAL,
                entity_ids=["phone-1"],
                recommended_keyframe_ms=5_000,
                keyframe_reason="phone is unobstructed",
                confidence=0.9,
                boundary_precision=BoundaryPrecision.COARSE,
                primary_entity_ids=["phone-1"],
                required_entity_ids=["phone-1"],
                optional_entity_ids=[],
                avoid_overlay_entity_ids=["phone-1"],
                framing_intent="keep the entire phone visible",
                card_opportunities=[],
            )
        ],
        uncertainties=["default video sampling can miss fast states"],
        model_provenance=provenance,
    )

