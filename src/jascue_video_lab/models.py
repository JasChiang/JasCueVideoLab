from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


NormalizedCoordinate = Annotated[int, Field(ge=0, le=1000)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundaryPrecision(StrEnum):
    COARSE = "coarse"
    SECOND_LEVEL = "second_level"
    UNCERTAIN = "uncertain"


class EvidenceModality(StrEnum):
    VISUAL = "visual"
    AUDIO = "audio"
    VISUAL_AND_AUDIO = "visual_and_audio"


class EntityKind(StrEnum):
    PERSON = "person"
    FACE = "face"
    HAND = "hand"
    PRODUCT = "product"
    PHONE = "phone"
    PHONE_SCREEN = "phone_screen"
    SCREEN = "screen"
    DOCUMENT = "document"
    LOGO = "logo"
    TEXT_REGION = "text_region"
    UI_ELEMENT = "ui_element"
    OTHER = "other"


class Occlusion(StrEnum):
    NONE = "none"
    PARTIAL = "partial"
    HEAVY = "heavy"
    UNKNOWN = "unknown"


class ModelProvenance(StrictModel):
    model_id: str
    api: Literal["gemini_interactions"]
    sdk: Literal["google-genai"]
    sdk_version: str
    interaction_id: str | None = None
    run_id: str
    generated_at: str


class CardOpportunity(StrictModel):
    kind: Literal["feature_card", "step_card", "object_callout"]
    rationale: str
    entity_ids: list[str] = Field(default_factory=list)


class Entity(StrictModel):
    entity_id: str = Field(min_length=1)
    kind: EntityKind
    label: str
    distinguishing_features: str
    evidence: str


class Event(StrictModel):
    event_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    label: str
    description: str
    evidence_modalities: EvidenceModality
    entity_ids: list[str]
    recommended_keyframe_ms: int | None = Field(default=None, ge=0)
    keyframe_reason: str
    confidence: Confidence
    boundary_precision: BoundaryPrecision
    primary_entity_ids: list[str]
    required_entity_ids: list[str]
    optional_entity_ids: list[str]
    avoid_overlay_entity_ids: list[str]
    framing_intent: str
    card_opportunities: list[CardOpportunity]

    @model_validator(mode="after")
    def validate_interval_and_keyframe(self) -> "Event":
        if self.end_ms <= self.start_ms:
            raise ValueError("event interval must be non-empty and half-open")
        if self.recommended_keyframe_ms is not None and not (
            self.start_ms <= self.recommended_keyframe_ms < self.end_ms
        ):
            raise ValueError("recommended_keyframe_ms must be inside [start_ms, end_ms)")
        return self


class ContentMap(StrictModel):
    asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    content_type: str
    events: list[Event]
    entities: list[Entity]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_references(self) -> "ContentMap":
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity_id values must be unique")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique")
        known = set(entity_ids)
        for event in self.events:
            if event.end_ms > self.duration_ms:
                raise ValueError(f"event {event.event_id} exceeds duration_ms")
            refs = (
                event.entity_ids
                + event.primary_entity_ids
                + event.required_entity_ids
                + event.optional_entity_ids
                + event.avoid_overlay_entity_ids
            )
            unknown = set(refs) - known
            if unknown:
                raise ValueError(f"event {event.event_id} references unknown entities: {sorted(unknown)}")
        return self


class TemporalEvent(StrictModel):
    """Small first-pass event contract; intentionally excludes entities and layout advice."""

    event_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    label: str
    observable_evidence: str
    recommended_keyframe_ms: int = Field(ge=0)
    keyframe_reason: str
    confidence: Confidence
    boundary_precision: BoundaryPrecision

    @model_validator(mode="after")
    def validate_interval_and_keyframe(self) -> "TemporalEvent":
        if self.end_ms <= self.start_ms:
            raise ValueError("event interval must be non-empty and half-open")
        if not self.start_ms <= self.recommended_keyframe_ms < self.end_ms:
            raise ValueError("recommended_keyframe_ms must be inside [start_ms, end_ms)")
        return self


class TemporalMap(StrictModel):
    asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    events: list[TemporalEvent]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_timeline(self) -> "TemporalMap":
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("event_id values must be unique")
        previous_end = 0
        for event in self.events:
            if event.end_ms > self.duration_ms:
                raise ValueError(f"event {event.event_id} exceeds duration_ms")
            if event.start_ms < previous_end:
                raise ValueError(f"event {event.event_id} overlaps or is out of order")
            previous_end = event.end_ms
        return self


class IndexedFrameEvent(StrictModel):
    event_id: str = Field(min_length=1)
    first_frame_id: str
    last_frame_id: str
    recommended_frame_id: str
    label: str
    observable_evidence: str
    grounding_target_id: str = Field(min_length=1)
    grounding_target_description: str
    confidence: Confidence
    boundary_precision: BoundaryPrecision


class IndexedStoryboardMap(StrictModel):
    """Model selects supplied IDs; local code owns all timestamp arithmetic."""

    asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    events: list[IndexedFrameEvent]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_ids(self) -> "IndexedStoryboardMap":
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("event_id values must be unique")
        return self


class DirectMoment(StrictModel):
    """A salient screenshot request using Gemini's documented MM:SS notation."""

    moment_id: str = Field(min_length=1)
    timestamp_mmss: str = Field(pattern=r"^\d{2,}:[0-5]\d$")
    label: str
    observable_evidence: str
    grounding_target_id: str = Field(min_length=1)
    grounding_target_description: str
    confidence: Confidence


class DirectMomentMap(StrictModel):
    asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    moments: list[DirectMoment]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_moments(self) -> "DirectMomentMap":
        ids = [moment.moment_id for moment in self.moments]
        if len(ids) != len(set(ids)):
            raise ValueError("moment_id values must be unique")
        previous_ms = -1
        for moment in self.moments:
            minutes, seconds = (int(part) for part in moment.timestamp_mmss.split(":"))
            timestamp_ms = (minutes * 60 + seconds) * 1000
            if timestamp_ms >= self.duration_ms:
                raise ValueError(
                    f"moment {moment.moment_id} timestamp {moment.timestamp_mmss} exceeds duration"
                )
            if timestamp_ms <= previous_ms:
                raise ValueError("moment timestamps must be strictly increasing")
            previous_ms = timestamp_ms
        return self


class TargetCandidate(StrictModel):
    """A user-selectable object proposal; this stage deliberately has no bbox."""

    candidate_id: str = Field(min_length=1)
    label: str
    entity_kind: EntityKind
    target_description: str
    distinguishing_features: str
    representative_timestamp_mmss: str = Field(pattern=r"^\d{2,}:[0-5]\d$")
    selection_reason: str
    confidence: Confidence


class TargetCandidateMap(StrictModel):
    asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    candidates: list[TargetCandidate]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_candidates(self) -> "TargetCandidateMap":
        ids = [candidate.candidate_id for candidate in self.candidates]
        if not ids:
            raise ValueError("at least one target candidate is required")
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique")
        for candidate in self.candidates:
            minutes, seconds = (
                int(part) for part in candidate.representative_timestamp_mmss.split(":")
            )
            timestamp_ms = (minutes * 60 + seconds) * 1000
            if timestamp_ms >= self.duration_ms:
                raise ValueError(
                    f"candidate {candidate.candidate_id} representative timestamp exceeds duration"
                )
        return self


class GroundingCandidate(StrictModel):
    box_2d: tuple[
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
    ]
    label: str
    confidence: Confidence
    disambiguation_reason: str

    @model_validator(mode="after")
    def validate_box(self) -> "GroundingCandidate":
        x_min, y_min, x_max, y_max = self.box_2d
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("box_2d must satisfy x_min < x_max and y_min < y_max")
        return self


class GeminiNativeGroundingCandidate(StrictModel):
    """API-boundary bbox using Gemini's documented y-first coordinate order."""

    box_2d_yxyx: tuple[
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
    ]
    label: str
    confidence: Confidence
    disambiguation_reason: str

    @model_validator(mode="after")
    def validate_box(self) -> "GeminiNativeGroundingCandidate":
        y_min, x_min, y_max, x_max = self.box_2d_yxyx
        if y_min >= y_max or x_min >= x_max:
            raise ValueError("box_2d_yxyx must satisfy ymin < ymax and xmin < xmax")
        return self


class GroundingProposal(StrictModel):
    asset_id: str
    event_id: str
    entity_id: str
    frame_pts: int
    frame_time_ms: int = Field(ge=0)
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    visible: bool
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GroundingProposal":
        if not self.visible and self.candidates:
            raise ValueError("invisible targets must have an empty candidates array")
        if self.visible and not self.candidates:
            raise ValueError("visible targets must have at least one candidate")
        return self


class GeminiNativeGroundingProposal(StrictModel):
    """Structured API response converted locally into GroundingProposal."""

    asset_id: str
    event_id: str
    entity_id: str
    frame_pts: int
    frame_time_ms: int = Field(ge=0)
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    visible: bool
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GeminiNativeGroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GeminiNativeGroundingProposal":
        if not self.visible and self.candidates:
            raise ValueError("invisible targets must have an empty candidates array")
        if self.visible and not self.candidates:
            raise ValueError("visible targets must have at least one candidate")
        return self


class Rational(StrictModel):
    numerator: int
    denominator: int = Field(gt=0)


class VideoStreamInfo(StrictModel):
    index: int
    codec_name: str | None
    coded_width: int
    coded_height: int
    display_width: int
    display_height: int
    rotation_degrees: int
    average_frame_rate: Rational | None
    real_frame_rate: Rational | None
    time_base: Rational
    start_pts: int | None
    duration_ts: int | None
    metadata: dict[str, str]


class MediaInfo(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_id: str
    format_name: str | None
    duration_ms: int
    size_bytes: int
    format_metadata: dict[str, str]
    video: VideoStreamInfo


class ExtractedFrame(StrictModel):
    path: str
    requested_time_ms: int
    frame_time_ms: int
    frame_pts: int
    frame_hash: str
    width: int
    height: int


class RunStatus(StrictModel):
    run_id: str
    stage: str
    ok: bool
    errors: list[dict[str, object]] = Field(default_factory=list)
