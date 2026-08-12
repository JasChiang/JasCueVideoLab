"""Only the shapes the measurements still need.

Carried over from a package that had a hundred and forty-seven of these. The
tracker, the shot detector and the music analysis touch a fraction of them;
the rest described a decision layer that no longer exists.
"""

from __future__ import annotations
import hashlib
import json
import re
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
NormalizedCoordinate = Annotated[int, Field(ge=0, le=1000)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
MmSs = Annotated[str, Field(pattern=r"^\d{2,}:[0-5]\d$")]
FeatureEvidenceProvenance = Literal[
    "direct_physical_action",
    "direct_ui_interaction",
    "direct_result",
    "prerecorded_screen_playback",
    "promotional_graphic",
    "textual_claim_only",
    "context_only",
    "unknown",
]
IllustrativeCoveragePolicy = Literal[
    "related_product_or_environment_when_direct_absent",
]
EvidenceRelation = Literal[
    "direct_source_event",
    "mediated_depiction",
    "graphic_or_text_claim",
    "context_only",
    "unknown",
]
_LEGACY_EVIDENCE_RELATIONS: dict[FeatureEvidenceProvenance, EvidenceRelation] = {
    "direct_physical_action": "direct_source_event",
    "direct_ui_interaction": "direct_source_event",
    "direct_result": "direct_source_event",
    "prerecorded_screen_playback": "mediated_depiction",
    "promotional_graphic": "graphic_or_text_claim",
    "textual_claim_only": "graphic_or_text_claim",
    "context_only": "context_only",
    "unknown": "unknown",
}
def evidence_relation_from_legacy(
    provenance: FeatureEvidenceProvenance,
) -> EvidenceRelation:
    """Project the legacy mixed enum onto its subject-neutral origin relation."""

    return _LEGACY_EVIDENCE_RELATIONS[provenance]
def legacy_evidence_mirror_is_compatible(
    relation: EvidenceRelation,
    provenance: FeatureEvidenceProvenance,
) -> bool:
    """Accept only exact mirrors or one conservative legacy demotion.

    The legacy enum has no value for a directly observed static state or
    spatial relation.  Older readers can safely receive ``context_only`` for
    that case while the generic origin remains authoritative.  The reverse
    promotion, mediated content marked direct, and every other disagreement
    remain invalid.
    """

    if provenance == "unknown":
        return True
    projected = evidence_relation_from_legacy(provenance)
    return projected == relation or (
        relation == "direct_source_event"
        and provenance == "context_only"
    )
def _mmss_to_ms(value: str) -> int:
    minutes, seconds = (int(part) for part in value.split(":"))
    return (minutes * 60 + seconds) * 1000
def _local_ms_from_pts(pts: int, start_pts: int, time_base: "Rational") -> int:
    return round(
        Fraction(
            (pts - start_pts) * time_base.numerator * 1000,
            time_base.denominator,
        )
    )
def _half_open_ms_matches_pts(
    actual_ms: int,
    expected_rounded_ms: int,
    end_ms: int | None,
) -> bool:
    return actual_ms == expected_rounded_ms or (
        end_ms is not None
        and expected_rounded_ms == end_ms
        and actual_ms == end_ms - 1
    )
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
AspectRatio = Annotated[str, Field(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")]
def _canonical_contract_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=True)
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
def _contract_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_contract_json(value).encode("utf-8")).hexdigest()
def _validate_unique_non_empty_strings(
    values: tuple[str, ...], field_name: str
) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} values must be non-empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")
def _validate_query_cross_references(
    *,
    identity: EvidenceIdentityContract,
    predicate: EvidencePredicateContract | None,
    framing: EvidenceFramingObligations,
) -> None:
    known_targets = {target.target_id for target in identity.targets}
    if predicate is not None:
        unknown = set(predicate.participant_target_ids) - known_targets
        if unknown:
            raise ValueError(
                f"predicate references unknown participant targets: {sorted(unknown)}"
            )
    for field_name in (
        "required_target_ids",
        "preferred_target_ids",
        "sacrificable_target_ids",
        "overlay_keepout_target_ids",
    ):
        unknown = set(getattr(framing, field_name)) - known_targets
        if unknown:
            raise ValueError(
                f"framing {field_name} references unknown targets: {sorted(unknown)}"
            )
    for aspect in framing.aspect_constraints:
        if len(aspect.required_target_ids) != len(set(aspect.required_target_ids)):
            raise ValueError("aspect required_target_ids must be unique")
        unknown = set(aspect.required_target_ids) - known_targets
        if unknown:
            raise ValueError(
                f"aspect constraint references unknown targets: {sorted(unknown)}"
            )
        visibility_unknown = {
            item.target_id for item in aspect.target_visibility_constraints
        } - known_targets
        if visibility_unknown:
            raise ValueError(
                "aspect visibility constraints reference unknown targets: "
                f"{sorted(visibility_unknown)}"
            )
def _query_component_hashes(
    *,
    identity: EvidenceIdentityContract,
    predicate: EvidencePredicateContract | None,
    framing: EvidenceFramingObligations,
) -> dict[str, str]:
    return {
        "identity_sha256": identity.definition_sha256(),
        "predicate_sha256": _contract_sha256(predicate),
        "framing_sha256": framing.definition_sha256(),
    }
def _query_composite_sha256(
    *,
    editorial_goal: str,
    identity: EvidenceIdentityContract,
    predicate: EvidencePredicateContract | None,
    framing: EvidenceFramingObligations,
) -> str:
    return _contract_sha256(
        {
            "contract_version": "grounding-query-v1",
            "editorial_goal": editorial_goal,
            **_query_component_hashes(
                identity=identity,
                predicate=predicate,
                framing=framing,
            ),
        }
    )
def approve_evidence_query_proposal(
    proposal: EvidenceQueryProposal,
    *,
    query_id: str,
    approval: EvidenceQueryApprovalProvenance,
) -> EvidenceQueryLock:
    """Create a new immutable lock without mutating or relabeling the proposal."""

    return EvidenceQueryLock(
        query_id=query_id,
        revision=proposal.revision,
        editorial_goal=proposal.editorial_goal,
        identity=proposal.identity,
        predicate=proposal.predicate,
        framing=proposal.framing,
        claim_source=proposal.claim_source,
        provenance=proposal.provenance,
        approval=approval,
    )
QualityRiskReason = Literal[
    "black",
    "white_clip",
    "freeze",
    "focus_loss",
    "motion_blur",
    "camera_shake",
    "occlusion",
    "target_not_visible",
    "duplicate_pts",
    "decoder_gap",
    "compression_artifact",
]
QualityRiskSeverity = Literal[
    "hard_block",
    "trim_candidate",
    "review",
    "note",
]
QualityRiskIntent = Literal[
    "accidental",
    "intentional",
    "unknown",
]
TrimTailIntent = Literal[
    "none",
    "natural_pause",
    "intentional_hold",
    "title_safe_hold",
    "clean_plate",
    "reset_or_false_end",
    "uncertain",
]
TrimPhase = Literal[
    "setup_start",
    "action_start",
    "result_start",
    "hold_start",
    "hold_end",
    "reset_start",
    "recommended_in",
    "recommended_out",
]
VirtualCameraIntent = Literal[
    "hold",
    "follow",
    "punch_in_cut",
    "push_in",
    "pull_out",
    "pan_reveal",
    "recenter",
]
CandidateVisualEventType = Literal[
    "camera_gesture_apex",
    "generation_result_stable_start",
    "watch_ui_state_change",
    "underwater_lift_apex",
    "group_laugh_reaction_peak",
    "freeze_start",
    "action_onset",
    "action_apex",
    "state_change",
    "result_stable_start",
    "reaction_peak",
    "clean_out",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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
    ANIMAL = "animal"
    OBJECT = "object"
    PRODUCT = "product"
    DEVICE = "device"
    PHONE = "phone"
    PHONE_SCREEN = "phone_screen"
    SCREEN = "screen"
    DOCUMENT = "document"
    LOGO = "logo"
    TEXT_REGION = "text_region"
    UI_ELEMENT = "ui_element"
    VEHICLE = "vehicle"
    OTHER = "other"


class EvidenceClaimSource(StrEnum):
    USER_BRIEF = "user_brief"
    HUMAN_REVIEW = "human_review"
    IMPORTED_METADATA = "imported_metadata"
    MODEL_PROPOSAL = "model_proposal"


class TargetIdentityScope(StrEnum):
    """The geometric level at which a persistent target is identified."""

    WHOLE_INSTANCE = "whole_instance"
    SUBPART = "subpart"
    VISIBLE_REGION = "visible_region"


class PredicateRequiredAt(StrEnum):
    """The stage or interval at which an observable predicate must hold."""

    CANDIDATE = "candidate"
    SEED = "seed"
    TRANSITION = "transition"
    INTERVAL = "interval"


class EvidenceApprovalSource(StrEnum):
    """An approval authority. Models are deliberately not an authority here."""

    USER_BRIEF = "user_brief"
    HUMAN_REVIEW = "human_review"
    AUTO_POLICY = "auto_policy"


class EvidenceAnchor(FrozenStrictModel):
    """A content-addressed crop from one immutable evidence frame."""

    frame_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    crop_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidenceTargetIdentity(FrozenStrictModel):
    """Persistent instance identity, separated from temporary event state."""

    target_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    target_description: str = Field(min_length=1)
    scope: TargetIdentityScope = TargetIdentityScope.WHOLE_INSTANCE
    parent_target_id: str | None = Field(
        default=None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    identity_cues: tuple[str, ...] = ()
    context_cues: tuple[str, ...] = ()
    positive_anchors: tuple[EvidenceAnchor, ...] = ()
    stable_exclusions: tuple[str, ...] = ()
    negative_anchors: tuple[EvidenceAnchor, ...] = ()

    @model_validator(mode="after")
    def validate_identity(self) -> "EvidenceTargetIdentity":
        for field_name in (
            "identity_cues",
            "context_cues",
            "stable_exclusions",
        ):
            _validate_unique_non_empty_strings(getattr(self, field_name), field_name)
        if self.scope == TargetIdentityScope.WHOLE_INSTANCE:
            if self.parent_target_id is not None:
                raise ValueError("whole_instance targets cannot have parent_target_id")
        elif self.parent_target_id is None:
            raise ValueError("subpart and visible_region targets require parent_target_id")
        if self.parent_target_id == self.target_id:
            raise ValueError("target cannot be its own parent")
        if not self.identity_cues and not self.positive_anchors:
            raise ValueError("identity requires identity_cues or positive_anchors")
        identity = {value.casefold() for value in self.identity_cues}
        exclusions = {value.casefold() for value in self.stable_exclusions}
        if identity & exclusions:
            raise ValueError("identity cues and stable exclusions must not overlap")
        positive = {
            (anchor.frame_id, anchor.crop_sha256) for anchor in self.positive_anchors
        }
        negative = {
            (anchor.frame_id, anchor.crop_sha256) for anchor in self.negative_anchors
        }
        if len(positive) != len(self.positive_anchors):
            raise ValueError("positive_anchors must be unique")
        if len(negative) != len(self.negative_anchors):
            raise ValueError("negative_anchors must be unique")
        if positive & negative:
            raise ValueError("positive and negative anchors must not overlap")
        positive_crop_hashes = {
            anchor.crop_sha256 for anchor in self.positive_anchors
        }
        negative_crop_hashes = {
            anchor.crop_sha256 for anchor in self.negative_anchors
        }
        if positive_crop_hashes & negative_crop_hashes:
            raise ValueError(
                "the same crop bytes cannot be both a positive and negative anchor"
            )
        return self


class EvidenceIdentityContract(FrozenStrictModel):
    targets: tuple[EvidenceTargetIdentity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target_graph(self) -> "EvidenceIdentityContract":
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("identity target_id values must be unique")
        known = set(target_ids)
        parents = {
            target.target_id: target.parent_target_id
            for target in self.targets
            if target.parent_target_id is not None
        }
        unknown = set(parents.values()) - known
        if unknown:
            raise ValueError(f"identity parents reference unknown targets: {sorted(unknown)}")
        for target_id in target_ids:
            visited: set[str] = set()
            cursor: str | None = target_id
            while cursor is not None:
                if cursor in visited:
                    raise ValueError("identity parent links must not contain cycles")
                visited.add(cursor)
                cursor = parents.get(cursor)
        return self

    def canonical_definition_json(self) -> str:
        return _canonical_contract_json(self)

    def definition_sha256(self) -> str:
        return _contract_sha256(self)

    def target(self, target_id: str) -> EvidenceTargetIdentity:
        try:
            return next(target for target in self.targets if target.target_id == target_id)
        except StopIteration as error:
            raise ValueError(f"unknown identity target: {target_id}") from error

    def ancestors(self, target_id: str) -> tuple[EvidenceTargetIdentity, ...]:
        """Return nearest-to-farthest parent identities for subpart disambiguation."""

        ancestors: list[EvidenceTargetIdentity] = []
        cursor = self.target(target_id)
        while cursor.parent_target_id is not None:
            cursor = self.target(cursor.parent_target_id)
            ancestors.append(cursor)
        return tuple(ancestors)


class EvidencePredicatePhases(FrozenStrictModel):
    precondition: str = Field(min_length=1)
    apex: str = Field(min_length=1)
    postcondition: str = Field(min_length=1)


class EvidencePredicateContract(FrozenStrictModel):
    """A media-observable eligibility condition, not a persistent identity cue."""

    predicate_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    statement: str = Field(min_length=1)
    participant_target_ids: tuple[str, ...] = Field(min_length=1)
    required_at: PredicateRequiredAt
    phases: EvidencePredicatePhases | None = None
    required_evidence: tuple[str, ...] = ()
    disqualifying_conditions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_predicate(self) -> "EvidencePredicateContract":
        for field_name in (
            "participant_target_ids",
            "required_evidence",
            "disqualifying_conditions",
        ):
            _validate_unique_non_empty_strings(getattr(self, field_name), field_name)
        if self.required_at == PredicateRequiredAt.TRANSITION and self.phases is None:
            raise ValueError("transition predicates require pre/apex/post phases")
        return self

    def canonical_definition_json(self) -> str:
        return _canonical_contract_json(self)

    def definition_sha256(self) -> str:
        return _contract_sha256(self)


class EvidenceTargetVisibilityConstraint(FrozenStrictModel):
    """A domain-neutral visibility floor for one target in one layout."""

    target_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    minimum_visible_fraction: float = Field(gt=0.0, le=1.0)
    atomic: bool = False


class EvidenceAspectConstraint(FrozenStrictModel):
    aspect_ratio: AspectRatio
    required_target_ids: tuple[str, ...] = ()
    constraint: str = Field(min_length=1)
    target_visibility_constraints: tuple[
        EvidenceTargetVisibilityConstraint, ...
    ] = ()
    required_target_clipping_policy: Literal[
        "forbid", "allow_controlled"
    ] = "forbid"

    @model_validator(mode="after")
    def validate_aspect(self) -> "EvidenceAspectConstraint":
        _validate_unique_non_empty_strings(
            self.required_target_ids, "required_target_ids"
        )
        visibility_ids = [
            item.target_id for item in self.target_visibility_constraints
        ]
        if len(visibility_ids) != len(set(visibility_ids)):
            raise ValueError("target visibility constraints must be unique")
        if any(
            item.atomic and item.minimum_visible_fraction != 1.0
            for item in self.target_visibility_constraints
        ):
            raise ValueError("atomic target visibility must be 1.0")
        return self


class EvidenceFramingObligations(FrozenStrictModel):
    """Semantic framing priorities; contains no generated crop coordinates."""

    required_target_ids: tuple[str, ...] = ()
    preferred_target_ids: tuple[str, ...] = ()
    sacrificable_target_ids: tuple[str, ...] = ()
    overlay_keepout_target_ids: tuple[str, ...] = ()
    framing_intent: str = Field(min_length=1)
    editing_uses: tuple[str, ...] = ()
    aspect_constraints: tuple[EvidenceAspectConstraint, ...] = ()

    @model_validator(mode="after")
    def validate_obligations(self) -> "EvidenceFramingObligations":
        for field_name in (
            "required_target_ids",
            "preferred_target_ids",
            "sacrificable_target_ids",
            "overlay_keepout_target_ids",
            "editing_uses",
        ):
            _validate_unique_non_empty_strings(getattr(self, field_name), field_name)
        required = set(self.required_target_ids)
        preferred = set(self.preferred_target_ids)
        sacrificable = set(self.sacrificable_target_ids)
        if required & preferred or required & sacrificable or preferred & sacrificable:
            raise ValueError(
                "required, preferred, and sacrificable target roles must be disjoint"
            )
        return self

    def canonical_definition_json(self) -> str:
        return _canonical_contract_json(self)

    def definition_sha256(self) -> str:
        return _contract_sha256(self)


class EvidenceQueryApprovalProvenance(FrozenStrictModel):
    approved_at: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    approval_source: EvidenceApprovalSource
    source_reference: str | None = None
    policy_reference: str | None = None

    @model_validator(mode="after")
    def validate_approval(self) -> "EvidenceQueryApprovalProvenance":
        if self.approval_source == EvidenceApprovalSource.AUTO_POLICY:
            if self.policy_reference is None or not self.policy_reference.strip():
                raise ValueError("auto_policy approval requires policy_reference")
        elif self.policy_reference is not None:
            raise ValueError("policy_reference is only valid for auto_policy approval")
        return self


class EvidenceQueryProvenance(FrozenStrictModel):
    created_at: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    source_reference: str | None = None
    parent_query_id: str | None = None


class EvidenceQueryProposal(StrictModel):
    """Unapproved three-layer query definition suitable for review."""

    contract_version: Literal["grounding-query-proposal-v1"] = (
        "grounding-query-proposal-v1"
    )
    proposal_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    revision: int = Field(ge=1)
    editorial_goal: str = Field(min_length=1)
    identity: EvidenceIdentityContract
    predicate: EvidencePredicateContract | None = None
    framing: EvidenceFramingObligations
    claim_source: EvidenceClaimSource
    provenance: EvidenceQueryProvenance

    @model_validator(mode="after")
    def validate_proposal(self) -> "EvidenceQueryProposal":
        _validate_query_cross_references(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )
        return self

    def component_hashes(self) -> dict[str, str]:
        return _query_component_hashes(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )

    def composite_sha256(self) -> str:
        return _query_composite_sha256(
            editorial_goal=self.editorial_goal,
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )


class EvidenceQueryLock(StrictModel):
    """Approved, frozen query definition with separate claim and approval origins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["grounding-query-lock-v1"] = "grounding-query-lock-v1"
    query_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    revision: int = Field(ge=1)
    editorial_goal: str = Field(min_length=1)
    identity: EvidenceIdentityContract
    predicate: EvidencePredicateContract | None = None
    framing: EvidenceFramingObligations
    claim_source: EvidenceClaimSource
    provenance: EvidenceQueryProvenance
    approval: EvidenceQueryApprovalProvenance

    @model_validator(mode="after")
    def validate_lock(self) -> "EvidenceQueryLock":
        _validate_query_cross_references(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )
        return self

    def component_hashes(self) -> dict[str, str]:
        return _query_component_hashes(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )

    def composite_sha256(self) -> str:
        return _query_composite_sha256(
            editorial_goal=self.editorial_goal,
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )

    def canonical_definition_json(self) -> str:
        return _canonical_contract_json(self)

    def definition_sha256(self) -> str:
        return _contract_sha256(self)


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
    sample_aspect_ratio: Rational = Field(
        default_factory=lambda: Rational(numerator=1, denominator=1)
    )
    display_sample_aspect_ratio: Rational = Field(
        default_factory=lambda: Rational(numerator=1, denominator=1)
    )
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
    timing_basis: Literal[
        "decoded_source_pts",
        "uniform_ffmpeg_analysis_sample",
    ]
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
        "bbox_seed_sam2_video_mask_propagation",
        "gemini_bbox_seed_sam2_video_mask_propagation",
        "gemini_polygon_seed_sam2_video_mask_propagation",
    ]
    asset_id: str
    video_path: str
    target_description: str
    seed_source: str
    seed_time_ms: int = Field(ge=0)
    seed_sample_index: int = Field(ge=0)
    seed_frame_pts: int | None = None
    seed_frame_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    seed_source_width: int | None = Field(default=None, gt=0)
    seed_source_height: int | None = Field(default=None, gt=0)
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
    analysis_start_ms: int = Field(default=0, ge=0)
    analysis_end_ms: int | None = Field(default=None, gt=0)
    source_start_pts: int | None = None
    source_time_base: Rational | None = None
    timing_warning: str
    semantic_warning: str
    total_samples: int = Field(gt=0)
    state_counts: dict[TrackingState, int]
    elapsed_seconds: float = Field(ge=0)
    effective_fps: float = Field(ge=0)
    model_provenance: SegmentationModelProvenance
    samples: list[SegmentationSample]
    target_id: str | None = Field(
        default=None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    shared_session_id: str | None = Field(default=None, min_length=1)
    analysis_frames_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )

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
        observed_state_counts: dict[TrackingState, int] = {}
        for sample in self.samples:
            observed_state_counts[sample.tracking_state] = (
                observed_state_counts.get(sample.tracking_state, 0) + 1
            )
        declared_state_counts = {
            state: count for state, count in self.state_counts.items() if count != 0
        }
        if declared_state_counts != observed_state_counts:
            raise ValueError("state_counts must match sample tracking_state values")
        if self.seed_sample_index >= self.total_samples:
            raise ValueError("seed_sample_index is outside sampled frames")
        if [sample.sample_index for sample in self.samples] != list(
            range(len(self.samples))
        ):
            raise ValueError("sample indexes must be contiguous from zero")
        sample_times = [sample.analysis_sample_time_ms for sample in self.samples]
        if sample_times != sorted(set(sample_times)):
            raise ValueError("sample times must be strictly increasing")
        timing_bases = {sample.timing_basis for sample in self.samples}
        if len(timing_bases) != 1:
            raise ValueError("all samples in one track must use the same timing basis")
        decoded_pts_timing = timing_bases == {"decoded_source_pts"}
        if decoded_pts_timing:
            if any(sample.source_pts is None for sample in self.samples):
                raise ValueError("decoded-source-PTS samples require source_pts")
            sample_pts = [sample.source_pts for sample in self.samples]
            if sample_pts != sorted(set(sample_pts)):
                raise ValueError("sample source PTS values must be strictly increasing")
        seed_lineage = (
            self.seed_frame_pts,
            self.seed_frame_sha256,
            self.seed_source_width,
            self.seed_source_height,
        )
        if any(value is not None for value in seed_lineage) and not all(
            value is not None for value in seed_lineage
        ):
            raise ValueError("seed frame lineage fields must be provided together")
        shared_identity = (
            self.target_id,
            self.shared_session_id,
            self.analysis_frames_manifest_sha256,
        )
        if any(value is not None for value in shared_identity) and not all(
            value is not None for value in shared_identity
        ):
            raise ValueError("shared track identity fields must be provided together")
        if self.seed_frame_pts is not None:
            seed_sample = self.samples[self.seed_sample_index]
            if seed_sample.source_pts != self.seed_frame_pts:
                raise ValueError("seed frame source PTS must match the seed sample")
        timing_lineage = (self.source_start_pts, self.source_time_base)
        if any(value is not None for value in timing_lineage) and not all(
            value is not None for value in timing_lineage
        ):
            raise ValueError("source timing lineage fields must be provided together")
        if (
            decoded_pts_timing
            and self.source_start_pts is not None
            and self.source_time_base is not None
        ):
            if any(
                not _half_open_ms_matches_pts(
                    sample.analysis_sample_time_ms,
                    _local_ms_from_pts(
                        sample.source_pts,  # type: ignore[arg-type]
                        self.source_start_pts,
                        self.source_time_base,
                    ),
                    self.analysis_end_ms,
                )
                for sample in self.samples
            ):
                raise ValueError("sample times must be derived from source PTS lineage")
        if self.analysis_end_ms is not None:
            if self.analysis_end_ms <= self.analysis_start_ms:
                raise ValueError("analysis interval must be non-empty and half-open")
            if not self.analysis_start_ms <= self.seed_time_ms < self.analysis_end_ms:
                raise ValueError("seed_time_ms must be inside the analysis interval")
            if any(
                not self.analysis_start_ms
                <= sample.analysis_sample_time_ms
                < self.analysis_end_ms
                for sample in self.samples
            ):
                raise ValueError("tracking samples must remain inside the analysis interval")
        return self


class SharedSam21BBoxSeed(StrictModel):
    """One semantic instance seed for a shared SAM 2.1 video session."""

    target_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    target_description: str = Field(min_length=1)
    seed_source: str = Field(min_length=1)
    seed_time_ms: int = Field(ge=0)
    seed_frame_pts: int
    seed_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_source_width: int = Field(gt=0)
    seed_source_height: int = Field(gt=0)
    seed_box_2d: list[NormalizedCoordinate]

    @model_validator(mode="after")
    def validate_seed_box(self) -> "SharedSam21BBoxSeed":
        if len(self.seed_box_2d) != 4:
            raise ValueError("seed_box_2d must contain four coordinates")
        x_min, y_min, x_max, y_max = self.seed_box_2d
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("seed_box_2d must satisfy xmin < xmax and ymin < ymax")
        return self


class SharedSam21TrackingRequest(StrictModel):
    """BBox-only request; every target must resolve to one shared shot interval."""

    asset_id: str = Field(min_length=1)
    targets: list[SharedSam21BBoxSeed] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_targets(self) -> "SharedSam21TrackingRequest":
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("shared SAM target_id values must be unique")
        return self


class SharedSam21AnalysisFrame(StrictModel):
    sample_index: int = Field(ge=0)
    analysis_sample_time_ms: int = Field(ge=0)
    source_pts: int
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SharedSam21SessionTiming(StrictModel):
    shot_detection_seconds: float = Field(ge=0)
    analysis_frame_extraction_seconds: float = Field(ge=0)
    predictor_initialization_seconds: float = Field(ge=0)
    prompt_seconds: float = Field(ge=0)
    forward_propagation_seconds: float = Field(ge=0)
    reverse_propagation_seconds: float = Field(ge=0)
    target_artifact_seconds: float = Field(ge=0)
    total_seconds: float = Field(ge=0)


class SharedSam21SessionTarget(StrictModel):
    target_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    target_description: str = Field(min_length=1)
    seed_time_ms: int = Field(ge=0)
    seed_sample_index: int = Field(ge=0)
    seed_frame_pts: int
    seed_frame_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_source_width: int = Field(gt=0)
    seed_source_height: int = Field(gt=0)
    track_path: str = Field(min_length=1)
    track_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_counts: dict[TrackingState, int]


class SharedSam21SessionManifest(StrictModel):
    """Auditable batch record for tracks sharing decode, backbone, and state."""

    artifact_type: Literal["shared_sam21_multi_object_tracking_session"]
    method: Literal["bbox_seed_shared_sam2_video_mask_propagation"]
    session_id: str = Field(min_length=1)
    asset_id: str
    video_path: str
    shot_id: str = Field(min_length=1)
    analysis_fps: float = Field(gt=0, le=60)
    analysis_width: int = Field(gt=0)
    analysis_height: int = Field(gt=0)
    analysis_start_ms: int = Field(ge=0)
    analysis_end_ms: int = Field(gt=0)
    source_start_pts: int
    source_time_base: Rational
    analysis_frames_path: str = Field(min_length=1)
    analysis_frames_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_frames: list[SharedSam21AnalysisFrame] = Field(min_length=1)
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool
    target_count: int = Field(ge=2)
    targets: list[SharedSam21SessionTarget] = Field(min_length=2)
    model_provenance: SegmentationModelProvenance
    timing: SharedSam21SessionTiming
    warning: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_session(self) -> "SharedSam21SessionManifest":
        if self.analysis_start_ms >= self.analysis_end_ms:
            raise ValueError("analysis interval must be non-empty and half-open")
        if self.target_count != len(self.targets):
            raise ValueError("target_count must equal len(targets)")
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("shared session target_id values must be unique")
        expected_indexes = list(range(len(self.analysis_frames)))
        if [frame.sample_index for frame in self.analysis_frames] != expected_indexes:
            raise ValueError("analysis frame sample indexes must be contiguous from zero")
        times = [frame.analysis_sample_time_ms for frame in self.analysis_frames]
        if times != sorted(set(times)):
            raise ValueError("analysis frame times must be strictly increasing")
        source_pts = [frame.source_pts for frame in self.analysis_frames]
        if source_pts != sorted(set(source_pts)):
            raise ValueError("analysis frame source PTS values must be strictly increasing")
        if any(
            not _half_open_ms_matches_pts(
                frame.analysis_sample_time_ms,
                _local_ms_from_pts(
                    frame.source_pts,
                    self.source_start_pts,
                    self.source_time_base,
                ),
                self.analysis_end_ms,
            )
            for frame in self.analysis_frames
        ):
            raise ValueError("analysis frame times must be derived from source PTS lineage")
        if any(
            not self.analysis_start_ms <= time_ms < self.analysis_end_ms
            for time_ms in times
        ):
            raise ValueError("analysis frames must remain inside the shared interval")
        if any(target.seed_sample_index >= len(self.analysis_frames) for target in self.targets):
            raise ValueError("target seed_sample_index is outside analysis frames")
        for target in self.targets:
            seed_frame = self.analysis_frames[target.seed_sample_index]
            if target.seed_frame_pts != seed_frame.source_pts:
                raise ValueError("target seed frame PTS must match analysis frame lineage")
            if target.seed_time_ms != seed_frame.analysis_sample_time_ms:
                raise ValueError("target seed time must match analysis frame lineage")
            if any(count < 0 for count in target.state_counts.values()):
                raise ValueError("target state counts cannot be negative")
            if sum(target.state_counts.values()) != len(self.analysis_frames):
                raise ValueError("target state counts must cover every analysis frame")
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
    frame_id: str = Field(
        pattern=r"^RF[0-9]{6}$",
        min_length=8,
        max_length=8,
    )
    clip_id: str
    requested_time_ms: int = Field(ge=0)
    image_path: str
    source_image_path: str | None = None
    frame_time_ms: int | None = Field(default=None, ge=0)
    frame_pts: int | None = None
    frame_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_exact_frame_lineage(self) -> "RushFrame":
        exact = (
            self.source_image_path,
            self.frame_time_ms,
            self.frame_pts,
            self.frame_hash,
        )
        if any(value is not None for value in exact) and not all(
            value is not None for value in exact
        ):
            raise ValueError(
                "rush frame exact lineage requires time, PTS and hash together"
            )
        return self


class RushesCatalog(StrictModel):
    catalog_id: str
    source_directory: str
    sample_interval_ms: int = Field(ge=500)
    total_duration_ms: int = Field(gt=0)
    clips: list[RushClip]
    frames: list[RushFrame]
    analysis_reel_path: str
    generated_at: str
