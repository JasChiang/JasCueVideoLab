"""Auditable, domain-neutral policy for unattended portrait reframing.

The policy never authorizes clipping a hard-core or atomic region.  It only
accepts a crop that was already solved with ``preserve_all`` and may label the
result ``auto_bounded_clip_v1`` when optional context is clipped within an
explicit visible-fraction floor.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FailureCode(StrEnum):
    SOURCE_LINEAGE_INVALID = "source_lineage_invalid"
    SHOT_BOUNDARY_CROSSING = "shot_boundary_crossing"
    EVIDENCE_CONFIDENCE_BELOW_MINIMUM = "evidence_confidence_below_minimum"
    SEMANTIC_MATCH_BELOW_MINIMUM = "semantic_match_below_minimum"
    TARGET_AMBIGUITY_ABOVE_MAXIMUM = "target_ambiguity_above_maximum"
    TRACK_CONFIDENCE_BELOW_MINIMUM = "track_confidence_below_minimum"
    TRACK_COVERAGE_BELOW_MINIMUM = "track_coverage_below_minimum"
    IDENTITY_SWITCH_DETECTED = "identity_switch_detected"
    HARD_CORE_NOT_FULLY_RETAINED = "hard_core_not_fully_retained"
    ATOMIC_REGION_CLIPPED = "atomic_region_clipped"
    SOFT_EXTENT_BELOW_MINIMUM = "soft_extent_below_minimum"
    OVERLAY_KEEPOUT_VIOLATED = "overlay_keepout_violated"
    CROP_SPEED_ABOVE_MAXIMUM = "crop_speed_above_maximum"
    CROP_ACCELERATION_ABOVE_MAXIMUM = "crop_acceleration_above_maximum"
    CROP_JERK_ABOVE_MAXIMUM = "crop_jerk_above_maximum"
    GEOMETRY_FINGERPRINT_MISMATCH = "geometry_fingerprint_mismatch"
    NO_FEASIBLE_PRESENTATION = "no_feasible_presentation"


class RecoveryAction(StrEnum):
    SPLIT_AT_SHOT_BOUNDARY = "split_at_shot_boundary"
    REPOSITION_OVERLAY = "reposition_overlay"
    TRY_SIMPLER_MOTION = "try_simpler_motion"
    TRY_ALTERNATE_SEED = "try_alternate_seed"
    TRY_NEXT_CANDIDATE = "try_next_candidate"
    FIT_WITH_BACKGROUND = "fit_with_background"
    REVIEW_REQUIRED = "review_required"


class RegionAssessment(StrictModel):
    region_id: str = Field(min_length=1)
    role: Literal["hard_core", "soft_extent", "overlay_keepout"]
    atomic: bool = False
    assessed: bool = True
    minimum_visible_fraction: float = Field(ge=0.0, le=1.0)
    required_visible_fraction: float = Field(ge=0.0, le=1.0)
    clipped_edges: list[Literal["left", "top", "right", "bottom"]] = Field(
        default_factory=list
    )
    overlay_overlap_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_region(self) -> "RegionAssessment":
        if self.role == "hard_core" or self.atomic:
            if self.required_visible_fraction != 1.0:
                raise ValueError("hard-core and atomic regions require full visibility")
        if self.role == "overlay_keepout" and self.required_visible_fraction != 0.0:
            raise ValueError("overlay keepout regions use overlap, not visible fraction")
        return self


class CandidatePreflight(StrictModel):
    candidate_id: str = Field(min_length=1)
    rank: int = Field(ge=1, le=4)
    presentation: Literal[
        "tracked_crop", "center_crop", "fit_with_background", "static_anchor"
    ]
    source_lineage_valid: bool
    within_single_shot: bool
    evidence_confidence: float = Field(ge=0.0, le=1.0)
    semantic_status: Literal[
        "matched",
        "ambiguous",
        "not_visible",
        "target_mismatch",
        "insufficient_evidence",
        "not_revalidated",
    ]
    tracking_confidence_gate_passed: bool
    tracking_coverage_passed: bool
    semantic_checkpoints_passed: bool | None = None
    regions: list[RegionAssessment] = Field(default_factory=list)
    max_crop_speed_pixels_per_second: float = Field(default=0.0, ge=0.0)
    max_crop_acceleration_pixels_per_second_squared: float = Field(
        default=0.0, ge=0.0
    )
    max_crop_jerk_pixels_per_second_cubed: float = Field(default=0.0, ge=0.0)
    geometry_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    track_fingerprints: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_fingerprints(self) -> "CandidatePreflight":
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.track_fingerprints
        ):
            raise ValueError("track fingerprints must be lowercase SHA-256")
        region_ids = [region.region_id for region in self.regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("preflight region IDs must be unique")
        return self


class AutoReframePolicy(StrictModel):
    policy_id: Literal["auto_bounded_clip_v1"] = "auto_bounded_clip_v1"
    max_candidates: int = Field(default=4, ge=1, le=4)
    max_alternate_seeds_per_candidate: int = Field(default=1, ge=0, le=2)
    max_repairs: int = Field(default=1, ge=0, le=2)
    minimum_evidence_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    max_crop_speed_pixels_per_second: float = Field(default=720.0, gt=0.0)
    max_crop_acceleration_pixels_per_second_squared: float = Field(
        default=1800.0, gt=0.0
    )
    max_crop_jerk_pixels_per_second_cubed: float = Field(
        default=7200.0, gt=0.0
    )
    require_semantic_checkpoints: bool = False
    # A center crop is not safe merely because it is deterministic.  Automatic
    # routing may only use it when a caller explicitly opts in *and* supplies
    # the same region/track preflight as any other presentation.
    allow_safe_center_crop: bool = False

    def definition_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class AutoBoundedClipAudit(StrictModel):
    policy_id: Literal["auto_bounded_clip_v1"]
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    candidate_rank: int
    approved: bool
    auto_bounded_clip_applied: bool
    failure_codes: list[FailureCode]
    geometry_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    track_fingerprints: list[str]
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def failure_codes_for_preflight(
    preflight: CandidatePreflight,
    policy: AutoReframePolicy,
    *,
    expected_geometry_fingerprint: str | None = None,
) -> list[FailureCode]:
    failures: list[FailureCode] = []
    if not preflight.source_lineage_valid:
        failures.append(FailureCode.SOURCE_LINEAGE_INVALID)
    if not preflight.within_single_shot:
        failures.append(FailureCode.SHOT_BOUNDARY_CROSSING)
    if preflight.evidence_confidence < policy.minimum_evidence_confidence:
        failures.append(FailureCode.EVIDENCE_CONFIDENCE_BELOW_MINIMUM)
    if preflight.semantic_status == "ambiguous":
        failures.append(FailureCode.TARGET_AMBIGUITY_ABOVE_MAXIMUM)
    elif preflight.semantic_status in {
        "not_visible",
        "target_mismatch",
        "insufficient_evidence",
    }:
        failures.append(FailureCode.SEMANTIC_MATCH_BELOW_MINIMUM)
    if not preflight.tracking_confidence_gate_passed:
        failures.append(FailureCode.TRACK_CONFIDENCE_BELOW_MINIMUM)
    if not preflight.tracking_coverage_passed:
        failures.append(FailureCode.TRACK_COVERAGE_BELOW_MINIMUM)
    if preflight.semantic_checkpoints_passed is False:
        failures.append(FailureCode.IDENTITY_SWITCH_DETECTED)
    elif (
        policy.require_semantic_checkpoints
        and preflight.semantic_checkpoints_passed is not True
    ):
        failures.append(FailureCode.IDENTITY_SWITCH_DETECTED)
    for region in preflight.regions:
        if not region.assessed:
            if region.role == "hard_core" or region.atomic:
                failures.append(FailureCode.HARD_CORE_NOT_FULLY_RETAINED)
            continue
        if (region.role == "hard_core" or region.atomic) and (
            region.minimum_visible_fraction < 1.0
        ):
            failures.append(
                FailureCode.ATOMIC_REGION_CLIPPED
                if region.atomic
                else FailureCode.HARD_CORE_NOT_FULLY_RETAINED
            )
        if region.role == "soft_extent" and (
            region.minimum_visible_fraction + 1e-6
            < region.required_visible_fraction
        ):
            failures.append(FailureCode.SOFT_EXTENT_BELOW_MINIMUM)
        if region.role == "overlay_keepout" and region.overlay_overlap_fraction > 0:
            failures.append(FailureCode.OVERLAY_KEEPOUT_VIOLATED)
    if (
        preflight.max_crop_speed_pixels_per_second
        > policy.max_crop_speed_pixels_per_second
    ):
        failures.append(FailureCode.CROP_SPEED_ABOVE_MAXIMUM)
    if (
        preflight.max_crop_acceleration_pixels_per_second_squared
        > policy.max_crop_acceleration_pixels_per_second_squared
    ):
        failures.append(FailureCode.CROP_ACCELERATION_ABOVE_MAXIMUM)
    if (
        preflight.max_crop_jerk_pixels_per_second_cubed
        > policy.max_crop_jerk_pixels_per_second_cubed
    ):
        failures.append(FailureCode.CROP_JERK_ABOVE_MAXIMUM)
    if (
        expected_geometry_fingerprint is not None
        and preflight.geometry_fingerprint != expected_geometry_fingerprint
    ):
        failures.append(FailureCode.GEOMETRY_FINGERPRINT_MISMATCH)
    if preflight.presentation == "center_crop" and not policy.allow_safe_center_crop:
        failures.append(FailureCode.NO_FEASIBLE_PRESENTATION)
    return list(dict.fromkeys(failures))


def audit_auto_bounded_clip(
    preflight: CandidatePreflight,
    policy: AutoReframePolicy,
    *,
    expected_geometry_fingerprint: str,
) -> AutoBoundedClipAudit:
    failures = failure_codes_for_preflight(
        preflight,
        policy,
        expected_geometry_fingerprint=expected_geometry_fingerprint,
    )
    clipped_soft_extent = any(
        region.role == "soft_extent" and region.minimum_visible_fraction < 1.0
        for region in preflight.regions
    )
    body = {
        "policy_id": policy.policy_id,
        "policy_sha256": policy.definition_sha256(),
        "candidate_id": preflight.candidate_id,
        "candidate_rank": preflight.rank,
        "approved": not failures,
        "auto_bounded_clip_applied": not failures and clipped_soft_extent,
        "failure_codes": [failure.value for failure in failures],
        "geometry_fingerprint": preflight.geometry_fingerprint,
        "source_fingerprint": preflight.source_fingerprint,
        "track_fingerprints": preflight.track_fingerprints,
    }
    return AutoBoundedClipAudit(
        **body,
        audit_sha256=_canonical_sha256(body),
    )


def choose_recovery(
    failures: list[FailureCode],
    *,
    candidates_remaining: bool,
    alternate_seed_remaining: bool = False,
) -> RecoveryAction:
    failure_set = set(failures)
    if FailureCode.SHOT_BOUNDARY_CROSSING in failure_set:
        return RecoveryAction.SPLIT_AT_SHOT_BOUNDARY
    if failure_set and failure_set <= {FailureCode.OVERLAY_KEEPOUT_VIOLATED}:
        return RecoveryAction.REPOSITION_OVERLAY
    if failure_set & {
        FailureCode.CROP_SPEED_ABOVE_MAXIMUM,
        FailureCode.CROP_ACCELERATION_ABOVE_MAXIMUM,
        FailureCode.CROP_JERK_ABOVE_MAXIMUM,
    }:
        return RecoveryAction.TRY_SIMPLER_MOTION
    if alternate_seed_remaining and failure_set & {
        FailureCode.SEMANTIC_MATCH_BELOW_MINIMUM,
        FailureCode.TARGET_AMBIGUITY_ABOVE_MAXIMUM,
        FailureCode.TRACK_CONFIDENCE_BELOW_MINIMUM,
        FailureCode.TRACK_COVERAGE_BELOW_MINIMUM,
    }:
        return RecoveryAction.TRY_ALTERNATE_SEED
    if candidates_remaining:
        return RecoveryAction.TRY_NEXT_CANDIDATE
    if failure_set & {
        FailureCode.SEMANTIC_MATCH_BELOW_MINIMUM,
        FailureCode.TARGET_AMBIGUITY_ABOVE_MAXIMUM,
        FailureCode.IDENTITY_SWITCH_DETECTED,
        FailureCode.SOURCE_LINEAGE_INVALID,
    }:
        return RecoveryAction.REVIEW_REQUIRED
    return RecoveryAction.FIT_WITH_BACKGROUND


def rank_preflights(
    preflights: list[CandidatePreflight], policy: AutoReframePolicy
) -> list[CandidatePreflight]:
    """Put fully accepted candidates first, then preserve editorial rank."""

    return sorted(
        preflights,
        key=lambda item: (
            bool(failure_codes_for_preflight(item, policy)),
            item.rank,
            item.candidate_id,
        ),
    )
