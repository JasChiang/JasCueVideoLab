from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from .models import (
    SegmentationSample,
    SemanticIdentityStatus,
    StrictModel,
    TrackingState,
)


CheckpointReason = Literal[
    "shot_boundary",
    "semantic_revalidation_required",
    "semantic_revalidation_failed",
    "drift_suspected",
    "low_confidence",
    "post_occlusion_reappearance",
    "geometry_area_jump",
    "geometry_center_jump",
    "track_start",
    "track_midpoint",
    "track_end",
]

IDENTITY_CHECKPOINT_PLANNER_VERSION = "identity-checkpoint-planner-v2"


def _request_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class IdentityCheckpointCandidate(StrictModel):
    sample_index: int = Field(ge=0)
    analysis_sample_time_ms: int = Field(ge=0)
    source_pts: int | None
    priority: int = Field(ge=0, le=100)
    reasons: tuple[CheckpointReason, ...] = Field(min_length=1)
    selected_for_verification: bool


class IdentityCheckpointPlan(StrictModel):
    artifact_type: Literal["identity_checkpoint_plan_v2"] = (
        "identity_checkpoint_plan_v2"
    )
    planner_version: Literal["identity-checkpoint-planner-v2"] = (
        IDENTITY_CHECKPOINT_PLANNER_VERSION
    )
    asset_id: str = Field(min_length=1)
    track_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    max_model_checks: int = Field(ge=0, le=8)
    seed_sample_index: int | None = Field(default=None, ge=0)
    area_relative_jump: float = Field(gt=0)
    center_distance_jump: float = Field(gt=0)
    planning_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_count: int = Field(ge=0)
    deferred_count: int = Field(ge=0)
    candidates: tuple[IdentityCheckpointCandidate, ...]
    model_calls_made: Literal[0] = 0
    execution_status: Literal["planned_not_executed"] = "planned_not_executed"
    warning: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> "IdentityCheckpointPlan":
        indexes = [candidate.sample_index for candidate in self.candidates]
        if len(indexes) != len(set(indexes)):
            raise ValueError("identity checkpoint sample indexes must be unique")
        selected = sum(
            candidate.selected_for_verification for candidate in self.candidates
        )
        if selected != self.selected_count:
            raise ValueError("selected_count does not match checkpoint candidates")
        if selected > self.max_model_checks:
            raise ValueError("selected checkpoints exceed the explicit model-call budget")
        if self.deferred_count != len(self.candidates) - selected:
            raise ValueError("deferred_count does not match checkpoint candidates")
        expected_request_sha256 = _request_sha256(
            {
                "planner_version": self.planner_version,
                "asset_id": self.asset_id,
                "track_fingerprint": self.track_fingerprint,
                "identity_sha256": self.identity_sha256,
                "max_model_checks": self.max_model_checks,
                "seed_sample_index": self.seed_sample_index,
                "area_relative_jump": self.area_relative_jump,
                "center_distance_jump": self.center_distance_jump,
            }
        )
        if self.planning_request_sha256 != expected_request_sha256:
            raise ValueError("identity checkpoint planning request hash mismatch")
        return self


def _add_reason(
    candidates: dict[int, tuple[int, set[CheckpointReason]]],
    sample_index: int,
    priority: int,
    reason: CheckpointReason,
) -> None:
    current_priority, reasons = candidates.get(sample_index, (0, set()))
    reasons.add(reason)
    candidates[sample_index] = (max(current_priority, priority), reasons)


def plan_identity_checkpoints(
    samples: Sequence[SegmentationSample],
    *,
    asset_id: str,
    track_fingerprint: str,
    identity_sha256: str | None = None,
    max_model_checks: int = 2,
    seed_sample_index: int | None = None,
    area_relative_jump: float = 0.6,
    center_distance_jump: float = 0.18,
) -> IdentityCheckpointPlan:
    """Choose a bounded set of semantic revalidation points without calling a model.

    SAM geometry cannot confirm identity.  This scheduler prioritizes exact source
    samples after risky state changes, then fills any remaining budget with
    start/mid/end coverage.  A separate explicit executor may later send only the
    selected exact frames to an identity verifier.
    """

    if not samples:
        raise ValueError("identity checkpoint planning requires at least one sample")
    if max_model_checks < 0 or max_model_checks > 8:
        raise ValueError("max_model_checks must be within 0..8")
    if area_relative_jump <= 0:
        raise ValueError("area_relative_jump must be positive")
    if center_distance_jump <= 0:
        raise ValueError("center_distance_jump must be positive")

    indexes = [sample.sample_index for sample in samples]
    if indexes != sorted(set(indexes)):
        raise ValueError("segmentation samples must have unique ordered indexes")
    candidates: dict[int, tuple[int, set[CheckpointReason]]] = {}

    coverage = (
        (samples[0], 25, "track_start"),
        (samples[len(samples) // 2], 20, "track_midpoint"),
        (samples[-1], 30, "track_end"),
    )
    for sample, priority, reason in coverage:
        _add_reason(candidates, sample.sample_index, priority, reason)

    for previous, sample in zip(samples, samples[1:]):
        if sample.shot_boundary:
            _add_reason(candidates, sample.sample_index, 100, "shot_boundary")
        if sample.semantic_identity_status == SemanticIdentityStatus.REVALIDATION_FAILED:
            _add_reason(
                candidates,
                sample.sample_index,
                98,
                "semantic_revalidation_failed",
            )
        elif (
            sample.semantic_identity_status
            == SemanticIdentityStatus.REVALIDATION_REQUIRED
        ):
            _add_reason(
                candidates,
                sample.sample_index,
                95,
                "semantic_revalidation_required",
            )
        if sample.tracking_state == TrackingState.DRIFT_SUSPECTED:
            _add_reason(candidates, sample.sample_index, 92, "drift_suspected")
        elif sample.tracking_state == TrackingState.LOW_CONFIDENCE:
            _add_reason(candidates, sample.sample_index, 80, "low_confidence")
        if (
            previous.tracking_state in {TrackingState.OCCLUDED, TrackingState.LOST}
            and sample.tracking_state
            not in {TrackingState.OCCLUDED, TrackingState.LOST}
        ):
            _add_reason(
                candidates,
                sample.sample_index,
                90,
                "post_occlusion_reappearance",
            )

        if previous.mask_area_ratio > 0 and sample.mask_area_ratio > 0:
            relative_change = abs(sample.mask_area_ratio - previous.mask_area_ratio) / max(
                previous.mask_area_ratio,
                1e-9,
            )
            if relative_change >= area_relative_jump:
                _add_reason(
                    candidates,
                    sample.sample_index,
                    75,
                    "geometry_area_jump",
                )
        if previous.center_2d is not None and sample.center_2d is not None:
            distance = math.dist(previous.center_2d, sample.center_2d) / 1000.0
            if distance >= center_distance_jump:
                _add_reason(
                    candidates,
                    sample.sample_index,
                    75,
                    "geometry_center_jump",
                )

    if seed_sample_index is not None:
        candidates.pop(seed_sample_index, None)

    ranked = sorted(
        candidates.items(),
        key=lambda item: (-item[1][0], item[0]),
    )
    selected_indexes = {
        sample_index for sample_index, _ in ranked[:max_model_checks]
    }
    sample_by_index = {sample.sample_index: sample for sample in samples}
    planned = tuple(
        IdentityCheckpointCandidate(
            sample_index=sample_index,
            analysis_sample_time_ms=sample_by_index[
                sample_index
            ].analysis_sample_time_ms,
            source_pts=sample_by_index[sample_index].source_pts,
            priority=priority,
            reasons=tuple(sorted(reasons)),
            selected_for_verification=sample_index in selected_indexes,
        )
        for sample_index, (priority, reasons) in ranked
    )
    planning_request = {
        "planner_version": IDENTITY_CHECKPOINT_PLANNER_VERSION,
        "asset_id": asset_id,
        "track_fingerprint": track_fingerprint,
        "identity_sha256": identity_sha256,
        "max_model_checks": max_model_checks,
        "seed_sample_index": seed_sample_index,
        "area_relative_jump": area_relative_jump,
        "center_distance_jump": center_distance_jump,
    }
    return IdentityCheckpointPlan(
        asset_id=asset_id,
        track_fingerprint=track_fingerprint,
        identity_sha256=identity_sha256,
        max_model_checks=max_model_checks,
        seed_sample_index=seed_sample_index,
        area_relative_jump=area_relative_jump,
        center_distance_jump=center_distance_jump,
        planning_request_sha256=_request_sha256(planning_request),
        selected_count=len(selected_indexes),
        deferred_count=len(planned) - len(selected_indexes),
        candidates=planned,
        warning=(
            "This artifact only schedules exact-frame semantic identity checks. "
            "No model was called and no candidate is verified until an explicit "
            "executor records a response."
        ),
    )
