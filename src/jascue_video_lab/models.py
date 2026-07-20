from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


NormalizedCoordinate = Annotated[int, Field(ge=0, le=1000)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _proper_segments_intersect(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
    d: tuple[int, int],
) -> bool:
    def orientation(
        first: tuple[int, int], second: tuple[int, int], third: tuple[int, int]
    ) -> int:
        return (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])

    return (
        orientation(a, b, c) * orientation(a, b, d) < 0
        and orientation(c, d, a) * orientation(c, d, b) < 0
    )


def _polygon_has_proper_self_intersection(points: list[tuple[int, int]]) -> bool:
    compact = [
        point
        for index, point in enumerate(points)
        if index == 0 or point != points[index - 1]
    ]
    if len(compact) > 1 and compact[0] == compact[-1]:
        compact.pop()
    count = len(compact)
    for left_index in range(count):
        left_start = compact[left_index]
        left_end = compact[(left_index + 1) % count]
        for right_index in range(left_index + 1, count):
            if right_index in {left_index, (left_index + 1) % count}:
                continue
            if left_index == (right_index + 1) % count:
                continue
            right_start = compact[right_index]
            right_end = compact[(right_index + 1) % count]
            if _proper_segments_intersect(left_start, left_end, right_start, right_end):
                return True
    return False


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


class GeminiNativeSegmentationCandidate(StrictModel):
    """Gemini single-image bbox plus polygon mask in documented native orders."""

    box_2d_yxyx: tuple[
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
    ]
    mask: list[tuple[NormalizedCoordinate, NormalizedCoordinate]]
    label: str
    confidence: Confidence
    disambiguation_reason: str

    @model_validator(mode="after")
    def validate_geometry(self) -> "GeminiNativeSegmentationCandidate":
        y_min, x_min, y_max, x_max = self.box_2d_yxyx
        if y_min >= y_max or x_min >= x_max:
            raise ValueError("box_2d_yxyx must satisfy ymin < ymax and xmin < xmax")
        if len(self.mask) < 3:
            raise ValueError("segmentation mask polygon must contain at least three points")
        twice_area = abs(
            sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(self.mask, self.mask[1:] + self.mask[:1])
            )
        )
        if _polygon_has_proper_self_intersection(self.mask):
            raise ValueError("segmentation mask polygon must not self-intersect")
        if twice_area == 0:
            raise ValueError("segmentation mask polygon must have non-zero area")
        xs = [point[0] for point in self.mask]
        ys = [point[1] for point in self.mask]
        tolerance = 5
        if (
            min(xs) < x_min - tolerance
            or max(xs) > x_max + tolerance
            or min(ys) < y_min - tolerance
            or max(ys) > y_max + tolerance
        ):
            raise ValueError("segmentation mask polygon must remain inside its bounding box")
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


class GeminiNativeSegmentationProposal(StrictModel):
    """Structured Gemini single-frame segmentation response; polygons remain x/y ordered."""

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
    candidates: list[GeminiNativeSegmentationCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GeminiNativeSegmentationProposal":
        if not self.visible and self.candidates:
            raise ValueError("invisible targets must have an empty candidates array")
        if self.visible and not self.candidates:
            raise ValueError("visible targets must have at least one candidate")
        return self


class DirectVideoGroundingProposal(StrictModel):
    """Experimental video-input bbox whose exact sampled source frame is unknowable locally."""

    asset_id: str
    event_id: str
    entity_id: str
    requested_timestamp_mmss: str = Field(pattern=r"^\d{2,}:[0-5]\d$")
    reference_frame_status: Literal["unknown_gemini_video_sample"]
    reference_frame_description: str
    visible: bool
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "DirectVideoGroundingProposal":
        if not self.visible and self.candidates:
            raise ValueError("invisible targets must have an empty candidates array")
        if self.visible and not self.candidates:
            raise ValueError("visible targets must have at least one candidate")
        return self


class GeminiNativeDirectVideoGroundingProposal(StrictModel):
    asset_id: str
    event_id: str
    entity_id: str
    requested_timestamp_mmss: str = Field(pattern=r"^\d{2,}:[0-5]\d$")
    reference_frame_status: Literal["unknown_gemini_video_sample"]
    reference_frame_description: str
    visible: bool
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GeminiNativeGroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GeminiNativeDirectVideoGroundingProposal":
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


class TrackingState(StrEnum):
    """Geometry state; it must not be mistaken for semantic identity confidence."""

    TRACKED = "tracked"
    REACQUIRED = "reacquired"
    OCCLUDED = "occluded"
    LOW_CONFIDENCE = "low_confidence"
    DRIFT_SUSPECTED = "drift_suspected"
    LOST = "lost"


class SemanticIdentityStatus(StrEnum):
    SEED_GROUNDED = "seed_grounded"
    NOT_REVALIDATED = "not_revalidated"
    REVALIDATION_REQUIRED = "revalidation_required"
    REVALIDATION_FAILED = "revalidation_failed"


class SegmentationModelProvenance(StrictModel):
    model_id: str
    implementation: str
    implementation_revision: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device: str
    torch_version: str
    generated_at: str


class SegmentationSample(StrictModel):
    sample_index: int = Field(ge=0)
    analysis_sample_time_ms: int = Field(ge=0)
    source_pts: int | None = None
    timing_basis: Literal["uniform_ffmpeg_analysis_sample"]
    mask_path: str | None
    mask_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    mask_area_pixels: int = Field(ge=0)
    mask_area_ratio: float = Field(ge=0.0, le=1.0)
    connected_components: int = Field(ge=0)
    derived_tracking_box: list[NormalizedCoordinate] | None
    center_2d: list[float] | None
    mean_positive_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    scene_cut_score: float | None = Field(default=None, ge=0.0, le=100.0)
    shot_boundary: bool
    tracking_state: TrackingState
    state_reasons: list[str]
    semantic_identity_status: SemanticIdentityStatus

    @model_validator(mode="after")
    def validate_mask_geometry(self) -> "SegmentationSample":
        if self.derived_tracking_box is not None:
            if len(self.derived_tracking_box) != 4:
                raise ValueError("derived_tracking_box must contain four coordinates")
            x_min, y_min, x_max, y_max = self.derived_tracking_box
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("derived_tracking_box must satisfy xmin < xmax and ymin < ymax")
        if self.mask_area_pixels == 0:
            if self.mask_path is not None or self.mask_sha256 is not None:
                raise ValueError("empty masks cannot reference a mask artifact")
            if self.derived_tracking_box is not None or self.center_2d is not None:
                raise ValueError("empty masks cannot contain geometry")
            if self.tracking_state != TrackingState.LOST:
                raise ValueError("empty masks must use tracking_state=lost")
        elif self.mask_path is None or self.mask_sha256 is None:
            raise ValueError("non-empty masks must reference a hashed mask artifact")
        return self


class SegmentationTrack(StrictModel):
    method: Literal[
        "gemini_bbox_seed_sam2_video_mask_propagation",
        "gemini_polygon_seed_sam2_video_mask_propagation",
    ]
    asset_id: str
    video_path: str
    target_description: str
    seed_source: str
    seed_time_ms: int = Field(ge=0)
    seed_sample_index: int = Field(ge=0)
    semantic_seed_box: list[NormalizedCoordinate]
    seed_prompt_type: Literal["box", "mask_polygon"] = "box"
    sam_prompt_box: list[NormalizedCoordinate] | None
    sam_prompt_mask_polygon_xy: list[
        tuple[NormalizedCoordinate, NormalizedCoordinate]
    ] | None = None
    seed_box_padding_ratio: float = Field(ge=0.0, le=1.0)
    refined_seed_mask_path: str
    analysis_fps: float = Field(gt=0, le=60)
    analysis_width: int = Field(gt=0)
    analysis_height: int = Field(gt=0)
    timing_warning: str
    semantic_warning: str
    total_samples: int = Field(gt=0)
    state_counts: dict[TrackingState, int]
    elapsed_seconds: float = Field(ge=0)
    effective_fps: float = Field(ge=0)
    model_provenance: SegmentationModelProvenance
    samples: list[SegmentationSample]

    @model_validator(mode="after")
    def validate_track(self) -> "SegmentationTrack":
        if len(self.semantic_seed_box) != 4:
            raise ValueError("semantic_seed_box must contain four coordinates")
        x_min, y_min, x_max, y_max = self.semantic_seed_box
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("semantic_seed_box must satisfy xmin < xmax and ymin < ymax")
        if self.seed_prompt_type == "box":
            if self.sam_prompt_box is None or len(self.sam_prompt_box) != 4:
                raise ValueError("box prompts require four sam_prompt_box coordinates")
            prompt_x_min, prompt_y_min, prompt_x_max, prompt_y_max = self.sam_prompt_box
            if prompt_x_min >= prompt_x_max or prompt_y_min >= prompt_y_max:
                raise ValueError("sam_prompt_box must satisfy xmin < xmax and ymin < ymax")
            if self.sam_prompt_mask_polygon_xy is not None:
                raise ValueError("box prompts cannot contain a mask polygon")
        else:
            if self.sam_prompt_box is not None:
                raise ValueError("mask polygon prompts cannot contain sam_prompt_box")
            if (
                self.sam_prompt_mask_polygon_xy is None
                or len(self.sam_prompt_mask_polygon_xy) < 3
            ):
                raise ValueError("mask polygon prompts require at least three points")
        if self.total_samples != len(self.samples):
            raise ValueError("total_samples must equal len(samples)")
        if sum(self.state_counts.values()) != self.total_samples:
            raise ValueError("state_counts must cover every sample")
        if self.seed_sample_index >= self.total_samples:
            raise ValueError("seed_sample_index is outside sampled frames")
        return self


class TrackerAgreementSample(StrictModel):
    analysis_sample_time_ms: int = Field(ge=0)
    reference_time_ms: float = Field(ge=0)
    segmentation_box: list[NormalizedCoordinate]
    reference_box: list[NormalizedCoordinate]
    bbox_iou: float = Field(ge=0.0, le=1.0)
    center_distance_normalized: float = Field(ge=0.0)


class TrackerAgreementReport(StrictModel):
    interpretation: Literal["tracker_agreement_not_accuracy"]
    segmentation_path: str
    reference_path: str
    reference_method: str
    aligned_samples: int = Field(gt=0)
    mean_bbox_iou: float = Field(ge=0.0, le=1.0)
    min_bbox_iou: float = Field(ge=0.0, le=1.0)
    mean_center_distance_normalized: float = Field(ge=0.0)
    max_center_distance_normalized: float = Field(ge=0.0)
    warning: str
    samples: list[TrackerAgreementSample]


class RushClip(StrictModel):
    clip_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_ms: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: str
    size_bytes: int = Field(gt=0)


class RushFrame(StrictModel):
    frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    clip_id: str
    requested_time_ms: int = Field(ge=0)
    image_path: str


class RushesCatalog(StrictModel):
    catalog_id: str
    source_directory: str
    sample_interval_ms: int = Field(ge=500)
    total_duration_ms: int = Field(gt=0)
    clips: list[RushClip]
    frames: list[RushFrame]
    analysis_reel_path: str
    generated_at: str


class RushesSelectShot(StrictModel):
    select_id: str = Field(min_length=1)
    representative_frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    suggested_duration_seconds: float = Field(ge=1.5, le=6.0)
    role: Literal["opening", "establishing", "product", "detail", "movement", "transition", "closing"]
    visual_description: str
    selection_reason: str
    quality_risks: list[str]
    vertical_focus: Literal["left", "center", "right"]
    confidence: Confidence


class RushesTimelinePlan(StrictModel):
    aspect_ratio: Literal["16:9", "9:16"]
    title: str
    editorial_intent: str
    shots: list[RushesSelectShot] = Field(min_length=1, max_length=16)


class RushesEditPlan(StrictModel):
    project_id: str
    catalog_id: str
    summary: str
    timelines: list[RushesTimelinePlan]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_timelines(self) -> "RushesEditPlan":
        aspects = [timeline.aspect_ratio for timeline in self.timelines]
        if sorted(aspects) != ["16:9", "9:16"]:
            raise ValueError("timelines must contain exactly one 16:9 and one 9:16 plan")
        for timeline in self.timelines:
            ids = [shot.select_id for shot in timeline.shots]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate select_id in {timeline.aspect_ratio} timeline")
        return self


class FeatureChapterBrief(StrictModel):
    feature_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    title: str
    detail_lines: list[str]
    target_duration_seconds: float = Field(ge=3.0, le=10.0)
    vertical_primary_target_description: str | None = None
    vertical_crop_mode: Literal["strict", "primary_center"] = "strict"


class FeatureEditBrief(StrictModel):
    project_id: str
    title: str
    target_duration_seconds: float = Field(ge=60.0, le=90.0)
    render_title_overlays: bool = True
    chapters: list[FeatureChapterBrief] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_chapters(self) -> "FeatureEditBrief":
        ids = [chapter.feature_id for chapter in self.chapters]
        if len(ids) != len(set(ids)):
            raise ValueError("feature brief chapter IDs must be unique")
        return self


class FeatureChapterSelect(StrictModel):
    feature_id: str
    evidence_status: Literal["supported", "partial", "not_found"]
    horizontal_frame_id: str | None = Field(default=None, pattern=r"^RF[0-9]{6}$")
    vertical_frame_id: str | None = Field(default=None, pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str
    selection_reason: str
    horizontal_strategy: Literal["original", "tracked_reframe"]
    horizontal_zoom_intent: Literal["none", "subtle", "detail"]
    horizontal_target_description: str | None
    vertical_strategy: Literal["tracked_crop", "fit_with_background"]
    vertical_target_description: str | None
    quality_risks: list[str]
    confidence: Confidence

    @model_validator(mode="after")
    def validate_evidence(self) -> "FeatureChapterSelect":
        if self.evidence_status == "not_found":
            if self.horizontal_frame_id is not None or self.vertical_frame_id is not None:
                raise ValueError("not_found feature chapters cannot reference catalog frames")
        elif self.horizontal_frame_id is None or self.vertical_frame_id is None:
            raise ValueError("supported/partial feature chapters require both aspect frame IDs")
        if self.horizontal_strategy == "tracked_reframe":
            if self.horizontal_zoom_intent == "none" or not self.horizontal_target_description:
                raise ValueError(
                    "tracked_reframe requires a zoom intent and precise horizontal target"
                )
        elif self.horizontal_zoom_intent != "none":
            raise ValueError("original horizontal strategy must use zoom intent none")
        if self.vertical_strategy == "tracked_crop" and not self.vertical_target_description:
            raise ValueError("tracked_crop requires a precise vertical_target_description")
        return self


class FeatureEditPlan(StrictModel):
    project_id: str
    catalog_id: str
    title: str
    chapters: list[FeatureChapterSelect]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_chapters(self) -> "FeatureEditPlan":
        ids = [chapter.feature_id for chapter in self.chapters]
        if len(ids) != len(set(ids)):
            raise ValueError("feature plan chapter IDs must be unique")
        return self
