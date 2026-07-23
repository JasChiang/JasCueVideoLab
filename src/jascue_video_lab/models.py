from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from fractions import Fraction
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


NormalizedCoordinate = Annotated[int, Field(ge=0, le=1000)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
MmSs = Annotated[str, Field(pattern=r"^\d{2,}:[0-5]\d$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrozenStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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


class Occlusion(StrEnum):
    NONE = "none"
    PARTIAL = "partial"
    HEAVY = "heavy"
    UNKNOWN = "unknown"


class MatchStatus(StrEnum):
    """Semantic target match result; intentionally separate from visibility."""

    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    NOT_VISIBLE = "not_visible"
    TARGET_MISMATCH = "target_mismatch"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class PredicateStatus(StrEnum):
    """Whether an optional observable event predicate is supported by evidence."""

    SATISFIED = "satisfied"
    NOT_SATISFIED = "not_satisfied"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class EvidenceClaimSource(StrEnum):
    USER_BRIEF = "user_brief"
    HUMAN_REVIEW = "human_review"
    IMPORTED_METADATA = "imported_metadata"
    MODEL_PROPOSAL = "model_proposal"


class EvidenceQueryTargetRef(StrictModel):
    """Stable, domain-neutral reference to a selected target instance."""

    target_id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    target_description: str = Field(min_length=1)
    positive_attributes: list[str] = Field(default_factory=list)
    negative_attributes: list[str] = Field(default_factory=list)
    reference_frame_ids: list[str] = Field(default_factory=list)
    reference_crop_hashes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_target_reference(self) -> "EvidenceQueryTargetRef":
        for field_name in (
            "positive_attributes",
            "negative_attributes",
            "reference_frame_ids",
            "reference_crop_hashes",
        ):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} values must be non-empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} values must be unique")
        for digest in self.reference_crop_hashes:
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("reference_crop_hashes must be lowercase SHA-256 digests")
        positive = {value.casefold() for value in self.positive_attributes}
        negative = {value.casefold() for value in self.negative_attributes}
        if positive & negative:
            raise ValueError("positive and negative target attributes must not overlap")
        return self


AspectRatio = Annotated[str, Field(pattern=r"^[1-9][0-9]*:[1-9][0-9]*$")]


class AspectConstraint(StrictModel):
    aspect_ratio: AspectRatio
    required_target_ids: list[str] = Field(default_factory=list)
    constraint: str = Field(min_length=1)


class EvidenceQueryProvenance(StrictModel):
    created_at: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    source_reference: str | None = None
    parent_query_id: str | None = None


class PredicatePhaseConditions(StrictModel):
    """Observable before/apex/after evidence for a locked temporal predicate."""

    precondition: str = Field(min_length=1)
    apex: str = Field(min_length=1)
    postcondition: str = Field(min_length=1)


class EvidenceQueryLock(StrictModel):
    """Immutable-by-convention editorial/evidence contract for downstream stages.

    It intentionally describes neither a media domain nor a tracker. Consumers may
    persist ``definition_sha256()`` with derived artifacts to prove which revision
    governed a result.
    """

    query_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    revision: int = Field(ge=1)
    editorial_goal: str = Field(min_length=1)
    targets: list[EvidenceQueryTargetRef] = Field(default_factory=list)
    observable_predicate: str | None = None
    predicate_phases: PredicatePhaseConditions | None = None
    required_evidence: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    editing_uses: list[str] = Field(default_factory=list)
    aspect_constraints: list[AspectConstraint] = Field(default_factory=list)
    claim_source: EvidenceClaimSource
    provenance: EvidenceQueryProvenance

    @model_validator(mode="after")
    def validate_query_lock(self) -> "EvidenceQueryLock":
        target_ids = [target.target_id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("query lock target_id values must be unique")
        known_targets = set(target_ids)
        for aspect in self.aspect_constraints:
            if len(aspect.required_target_ids) != len(set(aspect.required_target_ids)):
                raise ValueError("aspect required_target_ids must be unique")
            unknown = set(aspect.required_target_ids) - known_targets
            if unknown:
                raise ValueError(
                    f"aspect constraint references unknown targets: {sorted(unknown)}"
                )
        for field_name in (
            "required_evidence",
            "negative_constraints",
            "editing_uses",
        ):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} values must be non-empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} values must be unique")
        if self.observable_predicate is not None and not self.observable_predicate.strip():
            raise ValueError("observable_predicate must be non-empty when supplied")
        if self.predicate_phases is not None and self.observable_predicate is None:
            raise ValueError("predicate_phases require observable_predicate")
        return self

    def canonical_definition_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=True),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def definition_sha256(self) -> str:
        return hashlib.sha256(self.canonical_definition_json().encode("utf-8")).hexdigest()


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


class EvidenceAnchor(FrozenStrictModel):
    """A content-addressed crop from one immutable evidence frame."""

    frame_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    crop_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidenceTargetIdentityV2(FrozenStrictModel):
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
    def validate_identity(self) -> "EvidenceTargetIdentityV2":
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


class EvidenceIdentityContractV2(FrozenStrictModel):
    targets: tuple[EvidenceTargetIdentityV2, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target_graph(self) -> "EvidenceIdentityContractV2":
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

    def target(self, target_id: str) -> EvidenceTargetIdentityV2:
        try:
            return next(target for target in self.targets if target.target_id == target_id)
        except StopIteration as error:
            raise ValueError(f"unknown identity target: {target_id}") from error

    def ancestors(self, target_id: str) -> tuple[EvidenceTargetIdentityV2, ...]:
        """Return nearest-to-farthest parent identities for subpart disambiguation."""

        ancestors: list[EvidenceTargetIdentityV2] = []
        cursor = self.target(target_id)
        while cursor.parent_target_id is not None:
            cursor = self.target(cursor.parent_target_id)
            ancestors.append(cursor)
        return tuple(ancestors)


class EvidencePredicatePhasesV2(FrozenStrictModel):
    precondition: str = Field(min_length=1)
    apex: str = Field(min_length=1)
    postcondition: str = Field(min_length=1)


class EvidencePredicateContractV2(FrozenStrictModel):
    """A media-observable eligibility condition, not a persistent identity cue."""

    predicate_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    statement: str = Field(min_length=1)
    participant_target_ids: tuple[str, ...] = Field(min_length=1)
    required_at: PredicateRequiredAt
    phases: EvidencePredicatePhasesV2 | None = None
    required_evidence: tuple[str, ...] = ()
    disqualifying_conditions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_predicate(self) -> "EvidencePredicateContractV2":
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


class EvidenceTargetVisibilityConstraintV2(FrozenStrictModel):
    """A domain-neutral visibility floor for one target in one layout."""

    target_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    minimum_visible_fraction: float = Field(gt=0.0, le=1.0)
    atomic: bool = False


class EvidenceAspectConstraintV2(FrozenStrictModel):
    aspect_ratio: AspectRatio
    required_target_ids: tuple[str, ...] = ()
    constraint: str = Field(min_length=1)
    target_visibility_constraints: tuple[
        EvidenceTargetVisibilityConstraintV2, ...
    ] = ()
    required_target_clipping_policy: Literal[
        "forbid", "allow_controlled"
    ] = "forbid"

    @model_validator(mode="after")
    def validate_aspect(self) -> "EvidenceAspectConstraintV2":
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


class EvidenceFramingObligationsV2(FrozenStrictModel):
    """Semantic framing priorities; contains no generated crop coordinates."""

    required_target_ids: tuple[str, ...] = ()
    preferred_target_ids: tuple[str, ...] = ()
    sacrificable_target_ids: tuple[str, ...] = ()
    overlay_keepout_target_ids: tuple[str, ...] = ()
    framing_intent: str = Field(min_length=1)
    editing_uses: tuple[str, ...] = ()
    aspect_constraints: tuple[EvidenceAspectConstraintV2, ...] = ()

    @model_validator(mode="after")
    def validate_obligations(self) -> "EvidenceFramingObligationsV2":
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


class EvidenceQueryProvenanceV2(FrozenStrictModel):
    created_at: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    source_reference: str | None = None
    parent_query_id: str | None = None


def _validate_query_v2_cross_references(
    *,
    identity: EvidenceIdentityContractV2,
    predicate: EvidencePredicateContractV2 | None,
    framing: EvidenceFramingObligationsV2,
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


def _query_v2_component_hashes(
    *,
    identity: EvidenceIdentityContractV2,
    predicate: EvidencePredicateContractV2 | None,
    framing: EvidenceFramingObligationsV2,
) -> dict[str, str]:
    return {
        "identity_sha256": identity.definition_sha256(),
        "predicate_sha256": _contract_sha256(predicate),
        "framing_sha256": framing.definition_sha256(),
    }


def _query_v2_composite_sha256(
    *,
    editorial_goal: str,
    identity: EvidenceIdentityContractV2,
    predicate: EvidencePredicateContractV2 | None,
    framing: EvidenceFramingObligationsV2,
) -> str:
    return _contract_sha256(
        {
            "contract_version": "evidence-query-v2",
            "editorial_goal": editorial_goal,
            **_query_v2_component_hashes(
                identity=identity,
                predicate=predicate,
                framing=framing,
            ),
        }
    )


class EvidenceQueryProposalV2(StrictModel):
    """Unapproved three-layer query definition suitable for review."""

    contract_version: Literal["evidence-query-proposal-v2"] = (
        "evidence-query-proposal-v2"
    )
    proposal_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    revision: int = Field(ge=1)
    editorial_goal: str = Field(min_length=1)
    identity: EvidenceIdentityContractV2
    predicate: EvidencePredicateContractV2 | None = None
    framing: EvidenceFramingObligationsV2
    claim_source: EvidenceClaimSource
    provenance: EvidenceQueryProvenanceV2

    @model_validator(mode="after")
    def validate_proposal(self) -> "EvidenceQueryProposalV2":
        _validate_query_v2_cross_references(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )
        return self

    def component_hashes(self) -> dict[str, str]:
        return _query_v2_component_hashes(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )

    def composite_sha256(self) -> str:
        return _query_v2_composite_sha256(
            editorial_goal=self.editorial_goal,
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )


class EvidenceQueryLockV2(StrictModel):
    """Approved, frozen query definition with separate claim and approval origins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["evidence-query-lock-v2"] = "evidence-query-lock-v2"
    query_id: str = Field(
        min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$"
    )
    revision: int = Field(ge=1)
    editorial_goal: str = Field(min_length=1)
    identity: EvidenceIdentityContractV2
    predicate: EvidencePredicateContractV2 | None = None
    framing: EvidenceFramingObligationsV2
    claim_source: EvidenceClaimSource
    provenance: EvidenceQueryProvenanceV2
    approval: EvidenceQueryApprovalProvenance

    @model_validator(mode="after")
    def validate_lock(self) -> "EvidenceQueryLockV2":
        _validate_query_v2_cross_references(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )
        return self

    def component_hashes(self) -> dict[str, str]:
        return _query_v2_component_hashes(
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )

    def composite_sha256(self) -> str:
        return _query_v2_composite_sha256(
            editorial_goal=self.editorial_goal,
            identity=self.identity,
            predicate=self.predicate,
            framing=self.framing,
        )

    def canonical_definition_json(self) -> str:
        return _canonical_contract_json(self)

    def definition_sha256(self) -> str:
        return _contract_sha256(self)


def approve_evidence_query_proposal_v2(
    proposal: EvidenceQueryProposalV2,
    *,
    query_id: str,
    approval: EvidenceQueryApprovalProvenance,
) -> EvidenceQueryLockV2:
    """Create a new immutable lock without mutating or relabeling the proposal."""

    return EvidenceQueryLockV2(
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


def migrate_evidence_query_lock_v1_to_proposal_v2(
    lock: EvidenceQueryLock,
) -> EvidenceQueryProposalV2:
    """Losslessly move a v1 lock definition into the reviewable v2 layers."""

    targets: list[EvidenceTargetIdentityV2] = []
    for target in lock.targets:
        if len(target.reference_frame_ids) != len(target.reference_crop_hashes):
            raise ValueError(
                "v1 reference frame IDs and crop hashes must have equal lengths "
                "for lossless v2 migration"
            )
        targets.append(
            EvidenceTargetIdentityV2(
                target_id=target.target_id,
                target_description=target.target_description,
                scope=TargetIdentityScope.WHOLE_INSTANCE,
                identity_cues=(
                    tuple(target.positive_attributes)
                    or (target.target_description,)
                ),
                positive_anchors=tuple(
                    EvidenceAnchor(frame_id=frame_id, crop_sha256=crop_hash)
                    for frame_id, crop_hash in zip(
                        target.reference_frame_ids,
                        target.reference_crop_hashes,
                        strict=True,
                    )
                ),
                stable_exclusions=tuple(target.negative_attributes),
            )
        )
    if not targets:
        raise ValueError("v1 lock must contain a target for v2 identity migration")

    if (
        (lock.required_evidence or lock.negative_constraints)
        and not lock.observable_predicate
        and lock.predicate_phases is None
    ):
        raise ValueError(
            "v1 evidence constraints without an observable predicate cannot be "
            "losslessly migrated to QueryLock v2"
        )
    has_predicate_evidence = bool(
        lock.observable_predicate or lock.predicate_phases
    )
    predicate = (
        EvidencePredicateContractV2(
            predicate_id=f"{lock.query_id}:predicate",
            statement=(
                lock.observable_predicate
                or (
                    "Observable transition: "
                    f"{lock.predicate_phases.precondition}; "
                    f"{lock.predicate_phases.apex}; "
                    f"{lock.predicate_phases.postcondition}"
                )
            ),
            participant_target_ids=tuple(target.target_id for target in lock.targets),
            required_at=(
                PredicateRequiredAt.TRANSITION
                if lock.predicate_phases is not None
                else PredicateRequiredAt.SEED
            ),
            phases=(
                EvidencePredicatePhasesV2(
                    precondition=lock.predicate_phases.precondition,
                    apex=lock.predicate_phases.apex,
                    postcondition=lock.predicate_phases.postcondition,
                )
                if lock.predicate_phases is not None
                else None
            ),
            required_evidence=tuple(lock.required_evidence),
            disqualifying_conditions=tuple(lock.negative_constraints),
        )
        if has_predicate_evidence
        else None
    )
    required_targets = tuple(
        dict.fromkeys(
            target_id
            for aspect in lock.aspect_constraints
            for target_id in aspect.required_target_ids
        )
    )
    return EvidenceQueryProposalV2(
        proposal_id=f"{lock.query_id}:migrated-v2",
        revision=lock.revision,
        editorial_goal=lock.editorial_goal,
        identity=EvidenceIdentityContractV2(targets=tuple(targets)),
        predicate=predicate,
        framing=EvidenceFramingObligationsV2(
            required_target_ids=required_targets,
            preferred_target_ids=tuple(
                target.target_id
                for target in lock.targets
                if target.target_id not in required_targets
            ),
            framing_intent=(
                "; ".join(aspect.constraint for aspect in lock.aspect_constraints)
                or "Preserve the selected evidence targets for the intended edit."
            ),
            editing_uses=tuple(lock.editing_uses),
            aspect_constraints=tuple(
                EvidenceAspectConstraintV2(
                    aspect_ratio=aspect.aspect_ratio,
                    required_target_ids=tuple(aspect.required_target_ids),
                    constraint=aspect.constraint,
                )
                for aspect in lock.aspect_constraints
            ),
        ),
        claim_source=lock.claim_source,
        provenance=EvidenceQueryProvenanceV2(
            created_at=lock.provenance.created_at,
            created_by=lock.provenance.created_by,
            source_reference=lock.provenance.source_reference,
            parent_query_id=lock.provenance.parent_query_id,
        ),
    )


def migrate_evidence_query_lock_v1_to_v2(
    lock: EvidenceQueryLock,
    *,
    approval: EvidenceQueryApprovalProvenance,
) -> EvidenceQueryLockV2:
    """Migrate v1 through an explicit proposal and truthful approval record."""

    proposal = migrate_evidence_query_lock_v1_to_proposal_v2(lock)
    return approve_evidence_query_proposal_v2(
        proposal,
        query_id=lock.query_id,
        approval=approval,
    )


def _validated_grounding_match_status(
    *, visible: bool, candidate_count: int, match_status: MatchStatus | None
) -> MatchStatus:
    status = match_status
    if status is None:
        if not visible:
            status = MatchStatus.NOT_VISIBLE
        elif candidate_count == 1:
            status = MatchStatus.MATCHED
        else:
            status = MatchStatus.AMBIGUOUS
    if status == MatchStatus.MATCHED:
        if not visible or candidate_count != 1:
            raise ValueError(
                "match_status=matched requires visible=true and exactly one candidate"
            )
    elif status == MatchStatus.AMBIGUOUS:
        if not visible or candidate_count == 0:
            raise ValueError(
                f"match_status={status.value} requires visible=true and candidates"
            )
    elif visible or candidate_count:
        raise ValueError(
            f"match_status={status.value} requires visible=false and no candidates"
        )
    return status


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


class FullClipGroundingTarget(StrictModel):
    entity_id: str = Field(min_length=1)
    target_kind: EntityKind
    target_description: str = Field(min_length=1)
    purpose: Literal["reframe", "callout", "isolation", "identity_check"]


class FullClipEvent(StrictModel):
    """Gemini semantic event with second-level MM:SS anchors, never model milliseconds."""

    event_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    start_mmss: MmSs
    end_mmss: MmSs
    recommended_keyframe_mmss: MmSs | None
    label: str
    description: str
    observable_evidence: str
    evidence_modalities: EvidenceModality
    entity_ids: list[str]
    primary_entity_ids: list[str]
    required_entity_ids: list[str]
    optional_entity_ids: list[str]
    avoid_overlay_entity_ids: list[str]
    keyframe_reason: str
    boundary_precision: BoundaryPrecision
    confidence: Confidence
    action_completeness: Literal["complete", "partial", "uncertain"]
    editing_uses: list[
        Literal[
            "opening",
            "establishing",
            "hero",
            "detail",
            "demo",
            "reaction",
            "transition",
            "ending",
        ]
    ]
    quality_risks: list[str]
    framing_intent: str
    card_opportunities: list[CardOpportunity]
    dense_refinement: Literal["required", "recommended", "not_needed"]
    dense_refinement_reasons: list[str]
    grounding_targets: list[FullClipGroundingTarget]

    @model_validator(mode="after")
    def validate_mmss_interval(self) -> "FullClipEvent":
        start_ms = _mmss_to_ms(self.start_mmss)
        end_ms = _mmss_to_ms(self.end_mmss)
        if end_ms <= start_ms:
            raise ValueError("event MM:SS interval must be non-empty and half-open")
        if self.recommended_keyframe_mmss is not None:
            keyframe_ms = _mmss_to_ms(self.recommended_keyframe_mmss)
            if not start_ms <= keyframe_ms < end_ms:
                raise ValueError("recommended MM:SS keyframe must be inside [start, end)")
        return self

    def resolved_end_ms(self, duration_ms: int) -> int:
        """Resolve the only MM:SS interval that can represent a sub-second clip."""
        labeled_end_ms = _mmss_to_ms(self.end_mmss)
        if (
            duration_ms < 1000
            and _mmss_to_ms(self.start_mmss) == 0
            and labeled_end_ms == 1000
        ):
            return duration_ms
        return labeled_end_ms


class FullClipCard(StrictModel):
    """Complete per-clip semantic record produced from a full analysis proxy."""

    source_asset_id: str = Field(min_length=1)
    proxy_asset_id: str = Field(min_length=1)
    duration_ms: int = Field(gt=0)
    summary: str
    content_type: str
    entities: list[Entity]
    events: list[FullClipEvent]
    clip_uses: list[str]
    portrait_reframe_feasibility: Literal["good", "conditional", "poor", "uncertain"]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_timeline_and_references(self) -> "FullClipCard":
        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity_id values must be unique")
        event_ids = [event.event_id for event in self.events]
        if not event_ids:
            raise ValueError("a full Clip Card must contain at least one event")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event_id values must be unique")
        known_entities = set(entity_ids)
        entity_kinds = {entity.entity_id: entity.kind for entity in self.entities}
        previous_end = 0
        for event in self.events:
            start_ms = _mmss_to_ms(event.start_mmss)
            end_ms = event.resolved_end_ms(self.duration_ms)
            if end_ms > self.duration_ms:
                raise ValueError(f"event {event.event_id} MM:SS exceeds duration")
            if end_ms <= start_ms:
                raise ValueError(
                    f"event {event.event_id} resolved interval must be non-empty"
                )
            if event.recommended_keyframe_mmss is not None:
                keyframe_ms = _mmss_to_ms(event.recommended_keyframe_mmss)
                if not start_ms <= keyframe_ms < end_ms:
                    raise ValueError(
                        f"event {event.event_id} keyframe exceeds resolved interval"
                    )
            if start_ms < previous_end:
                raise ValueError(f"event {event.event_id} overlaps or is out of order")
            previous_end = end_ms
            references = (
                event.entity_ids
                + event.primary_entity_ids
                + event.required_entity_ids
                + event.optional_entity_ids
                + event.avoid_overlay_entity_ids
                + [target.entity_id for target in event.grounding_targets]
            )
            unknown = sorted(set(references) - known_entities)
            if unknown:
                raise ValueError(
                    f"event {event.event_id} references unknown entities: {unknown}"
                )
            for target in event.grounding_targets:
                if entity_kinds[target.entity_id] != target.target_kind:
                    raise ValueError(
                        f"event {event.event_id} Grounding target kind differs from Entity kind"
                    )
            for opportunity in event.card_opportunities:
                unknown_card_entities = sorted(
                    set(opportunity.entity_ids) - known_entities
                )
                if unknown_card_entities:
                    raise ValueError(
                        f"event {event.event_id} card references unknown entities: "
                        f"{unknown_card_entities}"
                    )
        return self


class DerivedClipEvent(StrictModel):
    """Local conversion of model MM:SS plus FFmpeg shot membership."""

    event_id: str
    start_mmss: MmSs
    end_mmss: MmSs
    recommended_keyframe_mmss: MmSs | None
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    recommended_keyframe_ms: int | None = Field(default=None, ge=0)
    shot_ids: list[str]
    boundary_source: Literal[
        "gemini_mmss_local_conversion",
        "gemini_mmss_subsecond_clip_end_conversion",
    ]
    exact_frame_required: bool

    @model_validator(mode="after")
    def validate_derived_interval(self) -> "DerivedClipEvent":
        if self.start_ms != _mmss_to_ms(self.start_mmss):
            raise ValueError("start_ms must be locally derived from start_mmss")
        labeled_end_ms = _mmss_to_ms(self.end_mmss)
        if self.boundary_source == "gemini_mmss_local_conversion":
            if self.end_ms != labeled_end_ms:
                raise ValueError("end_ms must be locally derived from end_mmss")
        elif not (
            self.start_ms == 0
            and labeled_end_ms == 1000
            and 0 < self.end_ms < 1000
        ):
            raise ValueError(
                "sub-second clip-end conversion requires 00:00–00:01 display "
                "labels and an authoritative end below 1000 ms"
            )
        expected_keyframe = (
            _mmss_to_ms(self.recommended_keyframe_mmss)
            if self.recommended_keyframe_mmss is not None
            else None
        )
        if self.recommended_keyframe_ms != expected_keyframe:
            raise ValueError("recommended_keyframe_ms must be locally derived from MM:SS")
        if self.end_ms <= self.start_ms:
            raise ValueError("derived event interval must be non-empty")
        return self


class DerivedClipTimeline(StrictModel):
    source_asset_id: str
    duration_ms: int = Field(gt=0)
    events: list[DerivedClipEvent]
    generated_at: str


class ShotRepresentativeFrame(StrictModel):
    frame_id: str = Field(pattern=r"^CF[0-9]{6}$")
    shot_id: str
    role: Literal["start", "middle", "end"]
    requested_time_ms: int = Field(ge=0)
    frame_time_ms: int = Field(ge=0)
    frame_pts: int
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_path: str


class ClipShotCatalog(StrictModel):
    source_asset_id: str
    duration_ms: int = Field(gt=0)
    frames: list[ShotRepresentativeFrame]
    generated_at: str


class DenseFrame(StrictModel):
    frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    event_id: str
    requested_time_ms: int = Field(ge=0)
    frame_time_ms: int = Field(ge=0)
    frame_pts: int
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    image_path: str
    transport_image_path: str
    transport_image_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DenseFrameCatalog(StrictModel):
    source_asset_id: str
    event_id: str
    sampling_fps: float = Field(gt=0, le=8)
    source_start_ms: int = Field(ge=0)
    source_end_ms: int = Field(gt=0)
    frames: list[DenseFrame] = Field(min_length=1, max_length=3600)
    contact_sheet_paths: list[str] = Field(min_length=1)
    contact_sheet_hashes: list[str] = Field(min_length=1)
    generated_at: str

    @model_validator(mode="after")
    def validate_contact_sheets(self) -> "DenseFrameCatalog":
        if self.source_end_ms <= self.source_start_ms:
            raise ValueError("dense source window must be non-empty and half-open")
        if len(self.contact_sheet_paths) != len(self.contact_sheet_hashes):
            raise ValueError("contact sheet paths and hashes must align")
        frame_ids = [frame.frame_id for frame in self.frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("dense frame IDs must be unique")
        if frame_ids != sorted(frame_ids):
            raise ValueError("dense frame IDs must be ordered")
        requested_times = [frame.requested_time_ms for frame in self.frames]
        if any(
            current >= following
            for current, following in zip(requested_times, requested_times[1:])
        ):
            raise ValueError("dense requested times must be strictly increasing")
        frame_times = [frame.frame_time_ms for frame in self.frames]
        if any(
            current > following
            for current, following in zip(frame_times, frame_times[1:])
        ):
            raise ValueError("dense source frame times must be ordered")
        for frame in self.frames:
            if frame.event_id != self.event_id:
                raise ValueError("dense frame event_id must match its catalog")
            if not self.source_start_ms <= frame.requested_time_ms < self.source_end_ms:
                raise ValueError("dense requested time must be inside the source window")
            if not self.source_start_ms <= frame.frame_time_ms < self.source_end_ms:
                raise ValueError("dense source frame time must be inside the source window")
        return self


class DenseEventSelection(StrictModel):
    source_asset_id: str
    event_id: str
    visible: bool
    first_frame_id: str | None = Field(default=None, pattern=r"^DF[0-9]{6}$")
    recommended_frame_id: str | None = Field(default=None, pattern=r"^DF[0-9]{6}$")
    last_frame_id: str | None = Field(default=None, pattern=r"^DF[0-9]{6}$")
    target_entity_id: str | None = None
    target_description: str | None = None
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    observable_evidence: str
    selection_reason: str
    uncertainties: list[str]
    confidence: Confidence
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "DenseEventSelection":
        frame_ids = [self.first_frame_id, self.recommended_frame_id, self.last_frame_id]
        target_fields = [self.target_entity_id, self.target_description]
        self.match_status = self.match_status or (
            MatchStatus.MATCHED if self.visible else MatchStatus.NOT_VISIBLE
        )
        if self.visible:
            if any(frame_id is None for frame_id in frame_ids):
                raise ValueError("visible dense selections require first/recommended/last IDs")
            if self.match_status not in {MatchStatus.MATCHED, MatchStatus.AMBIGUOUS}:
                raise ValueError("visible dense selections require matched or ambiguous status")
            assert all(frame_id is not None for frame_id in frame_ids)
            if not (
                self.first_frame_id
                <= self.recommended_frame_id
                <= self.last_frame_id
            ):
                raise ValueError("dense selection frame IDs must be ordered")
        else:
            if any(value is not None for value in frame_ids + target_fields):
                raise ValueError(
                    "invisible dense selections cannot reference frame or target fields"
                )
            if self.match_status not in {
                MatchStatus.NOT_VISIBLE,
                MatchStatus.TARGET_MISMATCH,
                MatchStatus.INSUFFICIENT_EVIDENCE,
            }:
                raise ValueError("invisible dense selection has incompatible match_status")
        if bool(self.target_entity_id) != bool(self.target_description):
            raise ValueError("dense selection target ID and description must appear together")
        return self


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


class TrimPhaseSelection(StrictModel):
    """One observable trim phase tied to one supplied dense-frame ID."""

    phase: TrimPhase
    frame_id: str = Field(min_length=8, max_length=8)


class TrimIntentProposal(StrictModel):
    """Evidence-bound trim phases selected from immutable dense frame IDs."""

    source_asset_id: str
    event_id: str
    usable: bool
    selections: list[TrimPhaseSelection] = Field(max_length=8)
    tail_intent: TrimTailIntent
    observed_phase_evidence: str = Field(max_length=800)
    hold_evidence: str = Field(max_length=500)
    trim_reason: str = Field(max_length=500)
    quality_risks: list[str] = Field(max_length=8)
    uncertainties: list[str] = Field(max_length=8)
    requires_human_review: bool
    confidence: Confidence
    model_provenance: ModelProvenance

    def frame_id_for(self, phase: TrimPhase) -> str | None:
        selection = next((item for item in self.selections if item.phase == phase), None)
        return selection.frame_id if selection is not None else None

    @model_validator(mode="after")
    def validate_usable_fields(self) -> "TrimIntentProposal":
        if not self.requires_human_review:
            raise ValueError("Gemini trim proposals always require human review")
        phases = [selection.phase for selection in self.selections]
        if len(phases) != len(set(phases)):
            raise ValueError("trim phases must be unique")
        required = [self.frame_id_for("recommended_in"), self.frame_id_for("recommended_out")]
        if self.usable:
            if any(frame_id is None for frame_id in required):
                raise ValueError("usable trim proposals require recommended in/out frame IDs")
            if required[0] == required[1]:
                raise ValueError("trim proposal must include at least two sampled frames")
        elif self.selections:
            raise ValueError("unusable trim proposals cannot reference frame IDs")
        if ("hold_start" in phases) != ("hold_end" in phases):
            raise ValueError("hold start/end frame IDs must appear together")
        return self


class VideoTrimIntentProposal(StrictModel):
    """Second-level direct-video trim proposal; local code resolves exact frame PTS."""

    source_asset_id: str
    event_id: str
    usable: bool
    recommended_in_mmss: MmSs | None
    recommended_out_mmss: MmSs | None
    hold_start_mmss: MmSs | None = None
    hold_end_mmss: MmSs | None = None
    reset_start_mmss: MmSs | None = None
    tail_intent: TrimTailIntent
    observed_phase_evidence: str = Field(max_length=800)
    hold_evidence: str = Field(max_length=500)
    trim_reason: str = Field(max_length=500)
    quality_risks: list[str] = Field(max_length=8)
    uncertainties: list[str] = Field(max_length=8)
    requires_human_review: bool
    confidence: Confidence
    model_provenance: ModelProvenance

    @model_validator(mode="before")
    @classmethod
    def preserve_incomplete_hold_as_uncertainty(cls, value: object) -> object:
        """Omit an unusable half-interval explicitly instead of inventing its endpoint."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        start = normalized.get("hold_start_mmss")
        end = normalized.get("hold_end_mmss")
        if bool(start) == bool(end):
            return normalized
        warning = (
            "contract_normalization: Gemini returned only one hold endpoint; "
            "the incomplete hold interval was omitted without inferring a missing time"
        )
        uncertainties = list(normalized.get("uncertainties") or [])
        if len(uncertainties) < 8:
            uncertainties.append(warning)
        else:
            uncertainties[-1] = f"{uncertainties[-1]}; {warning}"
        normalized["uncertainties"] = uncertainties
        normalized["hold_start_mmss"] = None
        normalized["hold_end_mmss"] = None
        return normalized

    @model_validator(mode="after")
    def validate_video_trim_fields(self) -> "VideoTrimIntentProposal":
        if not self.requires_human_review:
            raise ValueError("Gemini video trim proposals always require human review")
        boundaries = [self.recommended_in_mmss, self.recommended_out_mmss]
        if self.usable:
            if any(value is None for value in boundaries):
                raise ValueError("usable video trim proposals require in/out MM:SS")
            assert self.recommended_in_mmss is not None
            assert self.recommended_out_mmss is not None
            if _mmss_to_ms(self.recommended_out_mmss) <= _mmss_to_ms(
                self.recommended_in_mmss
            ):
                raise ValueError("video trim proposal must have in < exclusive out")
        elif any(
            value is not None
            for value in [
                *boundaries,
                self.hold_start_mmss,
                self.hold_end_mmss,
                self.reset_start_mmss,
            ]
        ):
            raise ValueError("unusable video trim proposals cannot reference MM:SS")
        if bool(self.hold_start_mmss) != bool(self.hold_end_mmss):
            raise ValueError("video hold start/end MM:SS must appear together")
        if self.hold_start_mmss is not None and self.hold_end_mmss is not None:
            if _mmss_to_ms(self.hold_end_mmss) < _mmss_to_ms(self.hold_start_mmss):
                raise ValueError("video hold interval must be chronological")
        return self


class TrimFrameEvidence(StrictModel):
    frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    requested_time_ms: int = Field(ge=0)
    frame_time_ms: int = Field(ge=0)
    frame_pts: int
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrimHumanReview(StrictModel):
    reviewer: str = Field(min_length=1)
    reviewed_at: str
    decision: Literal["approved", "rejected"]
    notes: str = ""


class TrimIntentDecision(StrictModel):
    """Local PTS derivation from a model proposal; never a human-approved cut by default."""

    source_asset_id: str
    event_id: str
    shot_id: str
    usable: bool
    first_included_frame: TrimFrameEvidence | None
    last_included_frame: TrimFrameEvidence | None
    exclusive_out_frame: TrimFrameEvidence | None
    hold_start_frame: TrimFrameEvidence | None
    hold_end_frame: TrimFrameEvidence | None
    source_in_ms: int | None = Field(default=None, ge=0)
    source_out_ms: int | None = Field(default=None, gt=0)
    source_in_pts: int | None = None
    source_out_pts: int | None = None
    handle_in_ms: int | None = Field(default=None, ge=0)
    handle_out_ms: int | None = Field(default=None, gt=0)
    tail_intent: TrimTailIntent
    approval_status: Literal["proposed", "approved", "rejected"] = "proposed"
    requires_human_review: bool = True
    human_review: TrimHumanReview | None = None
    proposal_path: str
    catalog_path: str

    @model_validator(mode="after")
    def validate_derived_bounds(self) -> "TrimIntentDecision":
        required = [
            self.first_included_frame,
            self.source_in_ms,
            self.source_out_ms,
            self.source_in_pts,
            self.source_out_pts,
        ]
        if self.usable:
            if any(value is None for value in required):
                raise ValueError("usable trim decisions require derived in/out evidence")
            assert self.source_in_ms is not None and self.source_out_ms is not None
            if self.source_out_ms <= self.source_in_ms:
                raise ValueError("trim decision must be a non-empty half-open interval")
            if self.handle_in_ms is None or self.handle_out_ms is None:
                raise ValueError("usable trim decisions require adjacent handles")
            if not self.handle_in_ms <= self.source_in_ms < self.source_out_ms <= self.handle_out_ms:
                raise ValueError("trim bounds must remain inside saved handles")
        elif any(value is not None for value in required):
            raise ValueError("unusable trim decisions cannot contain derived bounds")
        if self.approval_status == "proposed":
            if not self.requires_human_review or self.human_review is not None:
                raise ValueError("proposed trim decisions must remain unreviewed")
        else:
            if self.requires_human_review or self.human_review is None:
                raise ValueError("reviewed trim decisions require a human review record")
            if self.human_review.decision != self.approval_status:
                raise ValueError("human review decision must match approval status")
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
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GroundingProposal":
        self.match_status = _validated_grounding_match_status(
            visible=self.visible,
            candidate_count=len(self.candidates),
            match_status=self.match_status,
        )
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
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GeminiNativeGroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GeminiNativeGroundingProposal":
        self.match_status = _validated_grounding_match_status(
            visible=self.visible,
            candidate_count=len(self.candidates),
            match_status=self.match_status,
        )
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
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GeminiNativeSegmentationCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GeminiNativeSegmentationProposal":
        self.match_status = _validated_grounding_match_status(
            visible=self.visible,
            candidate_count=len(self.candidates),
            match_status=self.match_status,
        )
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
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "DirectVideoGroundingProposal":
        self.match_status = _validated_grounding_match_status(
            visible=self.visible,
            candidate_count=len(self.candidates),
            match_status=self.match_status,
        )
        return self


class GeminiNativeDirectVideoGroundingProposal(StrictModel):
    asset_id: str
    event_id: str
    entity_id: str
    requested_timestamp_mmss: str = Field(pattern=r"^\d{2,}:[0-5]\d$")
    reference_frame_status: Literal["unknown_gemini_video_sample"]
    reference_frame_description: str
    visible: bool
    match_status: MatchStatus | None = None
    predicate_status: PredicateStatus = PredicateStatus.NOT_APPLICABLE
    occlusion: Occlusion
    visibility_reason: str
    candidates: list[GeminiNativeGroundingCandidate]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_visibility(self) -> "GeminiNativeDirectVideoGroundingProposal":
        self.match_status = _validated_grounding_match_status(
            visible=self.visible,
            candidate_count=len(self.candidates),
            match_status=self.match_status,
        )
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


class SharedSam21AnalysisFramesManifest(StrictModel):
    """Immutable decoded-frame lineage shared by every track in one session."""

    timing_basis: Literal["decoded_source_pts"]
    frames: list[SharedSam21AnalysisFrame] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_frames(self) -> "SharedSam21AnalysisFramesManifest":
        expected_indexes = list(range(len(self.frames)))
        if [frame.sample_index for frame in self.frames] != expected_indexes:
            raise ValueError("analysis frame sample indexes must be contiguous from zero")
        times = [frame.analysis_sample_time_ms for frame in self.frames]
        if times != sorted(set(times)):
            raise ValueError("analysis frame times must be strictly increasing")
        source_pts = [frame.source_pts for frame in self.frames]
        if source_pts != sorted(set(source_pts)):
            raise ValueError("analysis frame source PTS values must be strictly increasing")
        return self


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


class MultiSegmentationReviewMember(StrictModel):
    """One per-target track shown in a synchronized review video."""

    label: str = Field(min_length=1)
    color_rgb: tuple[
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
    ]
    target_description: str = Field(min_length=1)
    track_json_path: str
    track_json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_time_ms: int = Field(ge=0)


class MultiSegmentationReviewManifest(StrictModel):
    """Provenance for a synchronized multi-track manual-review visualization."""

    artifact_type: Literal["multi_segmentation_track_review"]
    interpretation: Literal["manual_review_visualization_not_accuracy"]
    asset_id: str
    source_video_path: str
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_fps: float = Field(gt=0, le=60)
    display_fps: float = Field(gt=0, le=60)
    analysis_width: int = Field(gt=0)
    analysis_height: int = Field(gt=0)
    analysis_start_ms: int = Field(ge=0)
    analysis_end_ms: int = Field(gt=0)
    total_samples: int = Field(gt=0)
    analysis_frames_dir: str
    analysis_frames_manifest_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    audio_muxed: bool
    output_video_path: str
    output_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_duration_ms: int = Field(gt=0)
    output_video_duration_ms: int = Field(gt=0)
    output_frame_count: int = Field(gt=0)
    output_codec_name: Literal["h264"]
    output_pixel_format: Literal["yuv420p"]
    output_frame_rate: Rational
    warning: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)
    members: list[MultiSegmentationReviewMember] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_review_manifest(self) -> "MultiSegmentationReviewManifest":
        if self.analysis_start_ms >= self.analysis_end_ms:
            raise ValueError("analysis interval must be non-empty and half-open")
        labels = [member.label for member in self.members]
        if len(labels) != len(set(labels)):
            raise ValueError("multi-track review labels must be unique")
        colors = [member.color_rgb for member in self.members]
        if len(colors) != len(set(colors)):
            raise ValueError("multi-track review colors must be unique")
        return self


class SegmentationTrackAgreementSample(StrictModel):
    sample_index: int = Field(ge=0)
    analysis_sample_time_ms: int = Field(ge=0)
    source_pts: int
    tracking_state_a: TrackingState
    tracking_state_b: TrackingState
    state_agreement: bool
    mask_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    center_distance_normalized: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_agreement(self) -> "SegmentationTrackAgreementSample":
        if self.state_agreement != (self.tracking_state_a == self.tracking_state_b):
            raise ValueError("state_agreement must reflect the two tracking states")
        if (self.bbox_iou is None) != (self.center_distance_normalized is None):
            raise ValueError("bbox IoU and center distance must have identical coverage")
        return self


class SegmentationTrackAgreementReport(StrictModel):
    """Symmetric agreement metrics for two exactly aligned segmentation tracks."""

    artifact_type: Literal["segmentation_track_agreement_report"]
    interpretation: Literal["peer_agreement_not_accuracy"]
    asset_id: str
    track_a_path: str
    track_a_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    track_a_target_description: str = Field(min_length=1)
    track_b_path: str
    track_b_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    track_b_target_description: str = Field(min_length=1)
    total_samples: int = Field(gt=0)
    mask_iou_samples: int = Field(ge=0)
    mean_mask_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_iou_samples: int = Field(ge=0)
    mean_bbox_iou: float | None = Field(default=None, ge=0.0, le=1.0)
    center_distance_samples: int = Field(ge=0)
    mean_center_distance_normalized: float | None = Field(default=None, ge=0.0)
    state_agreement_samples: int = Field(ge=0)
    state_agreement_rate: float = Field(ge=0.0, le=1.0)
    warning: str = Field(min_length=1)
    generated_at: str = Field(min_length=1)
    samples: list[SegmentationTrackAgreementSample] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_summary(self) -> "SegmentationTrackAgreementReport":
        if self.track_a_path == self.track_b_path:
            raise ValueError("track A and track B paths must be different")
        if self.total_samples != len(self.samples):
            raise ValueError("total_samples must equal len(samples)")
        if [sample.sample_index for sample in self.samples] != list(
            range(len(self.samples))
        ):
            raise ValueError("agreement sample indexes must be contiguous from zero")
        sample_times = [sample.analysis_sample_time_ms for sample in self.samples]
        if sample_times != sorted(set(sample_times)):
            raise ValueError("agreement sample times must be strictly increasing")
        sample_pts = [sample.source_pts for sample in self.samples]
        if sample_pts != sorted(set(sample_pts)):
            raise ValueError("agreement sample source PTS values must be strictly increasing")
        mask_count = sum(sample.mask_iou is not None for sample in self.samples)
        bbox_count = sum(sample.bbox_iou is not None for sample in self.samples)
        center_count = sum(
            sample.center_distance_normalized is not None for sample in self.samples
        )
        state_count = sum(sample.state_agreement for sample in self.samples)
        if self.mask_iou_samples != mask_count:
            raise ValueError("mask_iou_samples does not match sample coverage")
        if self.bbox_iou_samples != bbox_count:
            raise ValueError("bbox_iou_samples does not match sample coverage")
        if self.center_distance_samples != center_count:
            raise ValueError("center_distance_samples does not match sample coverage")
        if self.state_agreement_samples != state_count:
            raise ValueError("state_agreement_samples does not match samples")
        if (self.mean_mask_iou is None) != (mask_count == 0):
            raise ValueError("mean_mask_iou must reflect mask metric coverage")
        if (self.mean_bbox_iou is None) != (bbox_count == 0):
            raise ValueError("mean_bbox_iou must reflect bbox metric coverage")
        if (self.mean_center_distance_normalized is None) != (center_count == 0):
            raise ValueError("mean center distance must reflect metric coverage")
        expected_mask_mean = (
            round(
                sum(
                    sample.mask_iou
                    for sample in self.samples
                    if sample.mask_iou is not None
                )
                / mask_count,
                6,
            )
            if mask_count
            else None
        )
        expected_bbox_mean = (
            round(
                sum(
                    sample.bbox_iou
                    for sample in self.samples
                    if sample.bbox_iou is not None
                )
                / bbox_count,
                6,
            )
            if bbox_count
            else None
        )
        expected_center_mean = (
            round(
                sum(
                    sample.center_distance_normalized
                    for sample in self.samples
                    if sample.center_distance_normalized is not None
                )
                / center_count,
                6,
            )
            if center_count
            else None
        )
        expected_state_rate = round(state_count / len(self.samples), 6)
        if self.mean_mask_iou != expected_mask_mean:
            raise ValueError("mean_mask_iou does not match samples")
        if self.mean_bbox_iou != expected_bbox_mean:
            raise ValueError("mean_bbox_iou does not match samples")
        if self.mean_center_distance_normalized != expected_center_mean:
            raise ValueError("mean center distance does not match samples")
        if self.state_agreement_rate != expected_state_rate:
            raise ValueError("state_agreement_rate does not match samples")
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


class FramingRegionIntent(StrictModel):
    """One domain-neutral visual region used to guide a reframe.

    A region may describe a person, animal, product, document, sign, UI area,
    or any other directly visible subject.  The vocabulary intentionally does
    not encode fixture-specific brands or object classes.
    """

    region_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    entity_id: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$",
        description=(
            "Immutable Clip Card entity reference when this region was resolved "
            "from catalog evidence. Legacy/manual regions may omit it."
        ),
    )
    target_description: str = Field(min_length=1)
    kind: Literal["subject", "text_region", "ui_region", "graphic", "other"] = (
        "subject"
    )
    role: Literal["required", "preferred", "avoid_overlay"] = "required"
    atomic: bool = Field(
        default=False,
        description=(
            "True when partial clipping changes the meaning of the region, for "
            "example a text or UI state. Atomic regions are treated as hard cores."
        ),
    )
    minimum_visible_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    observable_relations: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def upgrade_relation_field(cls, value: Any) -> Any:
        if isinstance(value, dict) and "relation_constraints" in value:
            if "observable_relations" in value:
                raise ValueError("region cannot define both relation field versions")
            value = dict(value)
            value["observable_relations"] = value.pop("relation_constraints")
        return value

    @model_validator(mode="after")
    def validate_region_policy(self) -> "FramingRegionIntent":
        for field_name in ("observable_relations", "exclusions"):
            values = getattr(self, field_name)
            if any(not value.strip() for value in values):
                raise ValueError(f"{field_name} values must be non-empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} values must be unique")
        if self.role == "required" and self.minimum_visible_fraction not in (None, 1.0):
            raise ValueError("required regions must be fully visible")
        if self.atomic and self.minimum_visible_fraction not in (None, 1.0):
            raise ValueError("atomic regions must be fully visible")
        if self.role == "avoid_overlay" and self.minimum_visible_fraction is not None:
            raise ValueError("avoid_overlay regions do not use a crop visible fraction")
        return self

    @property
    def execution_role(self) -> Literal["hard_core", "soft_extent", "overlay_keepout"]:
        if self.role == "avoid_overlay":
            return "overlay_keepout"
        if self.role == "required" or self.atomic:
            return "hard_core"
        return "soft_extent"

    @property
    def effective_minimum_visible_fraction(self) -> float:
        if self.execution_role == "hard_core":
            return 1.0
        if self.execution_role == "overlay_keepout":
            return 0.0
        return self.minimum_visible_fraction if self.minimum_visible_fraction is not None else 0.72


class FeatureHorizontalCandidate(StrictModel):
    """One evidence-bound 16:9 option retained for local automatic routing."""

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64)
    rank: int = Field(ge=1, le=4)
    source_asset_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    strategy: Literal["original", "tracked_reframe"]
    zoom_intent: Literal["none", "subtle", "detail"]
    target_description: str | None = None
    quality_risks: list[str] = Field(default_factory=list)
    confidence: Confidence

    @model_validator(mode="after")
    def validate_geometry_intent(self) -> "FeatureHorizontalCandidate":
        if self.strategy == "tracked_reframe":
            if self.zoom_intent == "none" or not self.target_description:
                raise ValueError("tracked_reframe candidate requires zoom intent and target")
        elif self.zoom_intent != "none":
            raise ValueError("original candidate must use zoom intent none")
        return self


class FeatureVerticalCandidate(StrictModel):
    """One evidence-bound 9:16 option retained for geometry-first selection."""

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64)
    rank: int = Field(ge=1, le=4)
    source_asset_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str = Field(min_length=1)
    selection_reason: str = Field(min_length=1)
    strategy: Literal["tracked_crop", "fit_with_background"]
    crop_mode: Literal["strict", "primary_center"] = "strict"
    target_description: str | None = None
    regions: list[FramingRegionIntent] = Field(default_factory=list, max_length=8)
    quality_risks: list[str] = Field(default_factory=list)
    confidence: Confidence

    @model_validator(mode="after")
    def validate_geometry_intent(self) -> "FeatureVerticalCandidate":
        hard_regions = [
            region for region in self.regions if region.execution_role == "hard_core"
        ]
        if self.strategy == "tracked_crop" and not (
            self.target_description or hard_regions
        ):
            raise ValueError("tracked_crop candidate requires a target or hard-core region")
        if self.regions and self.strategy != "tracked_crop":
            raise ValueError("region constraints require tracked_crop")
        ids = [region.region_id for region in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate region IDs must be unique")
        entity_ids = [region.entity_id for region in self.regions if region.entity_id]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("candidate entity references must be unique")
        return self


class ReframePolicyBinding(StrictModel):
    """Immutable human-policy provenance embedded in a revised edit brief.

    The sidecar is content addressed and binds the policy decision to the
    exact source brief, catalog, saved feature plan, and plan binding.  It is
    intentionally domain-neutral: the chapter overrides carry the visible
    region descriptions, while this record only establishes provenance.
    """

    binding_version: Literal["human-reframe-policy-binding-v1"]
    policy_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    sidecar_path: str = Field(min_length=1)
    sidecar_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_brief_path: str = Field(min_length=1)
    source_brief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_feature_plan_path: str = Field(min_length=1)
    source_feature_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_plan_binding_path: str = Field(min_length=1)
    source_plan_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_path: str = Field(min_length=1)
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class FeatureChapterBrief(StrictModel):
    feature_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    title: str
    detail_lines: list[str]
    target_duration_seconds: float = Field(ge=3.0, le=10.0)
    vertical_primary_target_description: str | None = None
    vertical_crop_mode: Literal["strict", "primary_center"] = "strict"
    vertical_regions: list[FramingRegionIntent] = Field(default_factory=list, max_length=4)
    vertical_overflow_policy: Literal["preserve_all", "controlled_clip"] = (
        "preserve_all"
    )
    vertical_edge_priority: Literal[
        "balanced", "preserve_start", "preserve_end"
    ] = "balanced"

    @model_validator(mode="after")
    def validate_vertical_regions(self) -> "FeatureChapterBrief":
        ids = [region.region_id for region in self.vertical_regions]
        if len(ids) != len(set(ids)):
            raise ValueError("vertical region IDs must be unique within a chapter")
        if self.vertical_regions and not any(
            region.role == "required" for region in self.vertical_regions
        ):
            raise ValueError("vertical regions must include at least one required region")
        if (
            self.vertical_edge_priority != "balanced"
            and self.vertical_overflow_policy != "controlled_clip"
        ):
            raise ValueError(
                "edge priority only applies when vertical_overflow_policy is controlled_clip"
            )
        return self


class FeatureEditBrief(StrictModel):
    project_id: str
    title: str
    target_duration_seconds: float = Field(ge=60.0, le=90.0)
    render_title_overlays: bool = True
    vertical_fallback_strategy: Literal["fit_with_background", "center_crop"] = (
        "fit_with_background"
    )
    reframe_policy_binding: ReframePolicyBinding | None = None
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
    horizontal_candidates: list[FeatureHorizontalCandidate] = Field(
        default_factory=list, max_length=4
    )
    vertical_candidates: list[FeatureVerticalCandidate] = Field(
        default_factory=list, max_length=4
    )

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
            primary_candidate = next(
                (candidate for candidate in self.vertical_candidates if candidate.rank == 1),
                None,
            )
            if primary_candidate is None or not primary_candidate.regions:
                raise ValueError(
                    "tracked_crop requires a precise target or rank-1 region contract"
                )
        for field_name in ("horizontal_candidates", "vertical_candidates"):
            candidates = getattr(self, field_name)
            if candidates and not 2 <= len(candidates) <= 4:
                raise ValueError(f"{field_name} must preserve 2-4 options when present")
            ids = [candidate.candidate_id for candidate in candidates]
            ranks = [candidate.rank for candidate in candidates]
            if len(ids) != len(set(ids)) or len(ranks) != len(set(ranks)):
                raise ValueError(f"{field_name} candidate IDs and ranks must be unique")
            references = [
                (candidate.source_asset_id, candidate.event_id, candidate.frame_id)
                for candidate in candidates
            ]
            if len(references) != len(set(references)):
                raise ValueError(
                    f"{field_name} candidates must reference distinct evidence frames"
                )
            if ranks and sorted(ranks) != list(range(1, len(ranks) + 1)):
                raise ValueError(f"{field_name} ranks must be contiguous from 1")
        if self.evidence_status == "not_found" and (
            self.horizontal_candidates or self.vertical_candidates
        ):
            raise ValueError("not_found chapters cannot retain execution candidates")
        if self.horizontal_candidates:
            primary = min(self.horizontal_candidates, key=lambda item: item.rank)
            if (
                self.horizontal_frame_id != primary.frame_id
                or self.horizontal_strategy != primary.strategy
                or self.horizontal_zoom_intent != primary.zoom_intent
                or self.horizontal_target_description != primary.target_description
            ):
                raise ValueError("rank-1 horizontal candidate must match legacy projection")
        if self.vertical_candidates:
            primary = min(self.vertical_candidates, key=lambda item: item.rank)
            if (
                self.vertical_frame_id != primary.frame_id
                or self.vertical_strategy != primary.strategy
                or self.vertical_target_description != primary.target_description
            ):
                raise ValueError("rank-1 vertical candidate must match legacy projection")
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
