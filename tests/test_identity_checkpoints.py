from __future__ import annotations

import pytest

from jascue_video_lab.auto_reframe import SemanticCheckpointStatus
from jascue_video_lab.identity_checkpoints import (
    IdentityCheckpointModelDecision,
    IdentityCheckpointVerdict,
    execute_identity_checkpoints,
    plan_identity_checkpoints,
)
from jascue_video_lab.models import (
    ExtractedFrame,
    SegmentationSample,
    SemanticIdentityStatus,
    TrackingState,
)


def _sample(
    index: int,
    *,
    state: TrackingState = TrackingState.TRACKED,
    semantic: SemanticIdentityStatus = SemanticIdentityStatus.NOT_REVALIDATED,
    area: float = 0.1,
    center: tuple[float, float] = (500.0, 500.0),
    shot_boundary: bool = False,
) -> SegmentationSample:
    return SegmentationSample(
        sample_index=index,
        analysis_sample_time_ms=index * 500,
        source_pts=index * 15,
        timing_basis="decoded_source_pts",
        mask_path=f"mask-{index}.png",
        mask_sha256=f"{index + 1:064x}",
        mask_area_pixels=100,
        mask_area_ratio=area,
        connected_components=1,
        derived_tracking_box=[400, 400, 600, 600],
        center_2d=list(center),
        mean_positive_probability=0.9,
        scene_cut_score=None,
        shot_boundary=shot_boundary,
        tracking_state=state,
        state_reasons=[],
        semantic_identity_status=semantic,
    )


def test_checkpoint_plan_prioritizes_identity_and_reappearance_risks() -> None:
    samples = [
        _sample(0, semantic=SemanticIdentityStatus.SEED_GROUNDED),
        _sample(1, state=TrackingState.OCCLUDED),
        _sample(2, state=TrackingState.REACQUIRED),
        _sample(3, semantic=SemanticIdentityStatus.REVALIDATION_REQUIRED),
        _sample(4),
    ]
    plan = plan_identity_checkpoints(
        samples,
        asset_id="sha256:" + "a" * 64,
        track_fingerprint="b" * 64,
        identity_sha256="c" * 64,
        max_model_checks=2,
        seed_sample_index=0,
    )
    selected = {
        candidate.sample_index: set(candidate.reasons)
        for candidate in plan.candidates
        if candidate.selected_for_verification
    }
    assert set(selected) == {2, 3}
    assert "post_occlusion_reappearance" in selected[2]
    assert "semantic_revalidation_required" in selected[3]
    assert plan.model_calls_made == 0
    assert plan.execution_status == "planned_not_executed"


def test_checkpoint_plan_detects_geometry_jumps_but_obeys_zero_budget() -> None:
    samples = [
        _sample(0),
        _sample(1, area=0.25, center=(800.0, 800.0)),
        _sample(2, area=0.25, center=(810.0, 810.0)),
    ]
    plan = plan_identity_checkpoints(
        samples,
        asset_id="asset-generic",
        track_fingerprint="d" * 64,
        max_model_checks=0,
    )
    risky = next(candidate for candidate in plan.candidates if candidate.sample_index == 1)
    assert "geometry_area_jump" in risky.reasons
    assert "geometry_center_jump" in risky.reasons
    assert plan.selected_count == 0
    assert all(not candidate.selected_for_verification for candidate in plan.candidates)


def test_seed_sample_is_not_charged_again_as_a_checkpoint() -> None:
    samples = [_sample(0), _sample(1)]
    plan = plan_identity_checkpoints(
        samples,
        asset_id="asset-generic",
        track_fingerprint="e" * 64,
        max_model_checks=2,
        seed_sample_index=0,
    )
    assert all(candidate.sample_index != 0 for candidate in plan.candidates)


def test_checkpoint_plan_hash_binds_scheduler_inputs() -> None:
    samples = [_sample(0), _sample(1), _sample(2)]
    baseline = plan_identity_checkpoints(
        samples,
        asset_id="asset-generic",
        track_fingerprint="f" * 64,
        max_model_checks=2,
        seed_sample_index=0,
        area_relative_jump=0.6,
        center_distance_jump=0.18,
    )
    changed = plan_identity_checkpoints(
        samples,
        asset_id="asset-generic",
        track_fingerprint="f" * 64,
        max_model_checks=2,
        seed_sample_index=1,
        area_relative_jump=0.5,
        center_distance_jump=0.18,
    )

    assert baseline.planner_version == "identity-checkpoint-planner-v2"
    assert baseline.planning_request_sha256 != changed.planning_request_sha256
    assert baseline.seed_sample_index == 0
    assert baseline.area_relative_jump == 0.6


def _frame(index: int) -> ExtractedFrame:
    return ExtractedFrame(
        path=f"/tmp/frame-{index}.png",
        requested_time_ms=index * 500,
        frame_time_ms=index * 500,
        frame_pts=index * 15,
        frame_hash=f"{index + 10:064x}",
        width=1920,
        height=1080,
    )


def test_checkpoint_executor_passes_only_when_all_selected_frames_match() -> None:
    samples = [
        _sample(0, semantic=SemanticIdentityStatus.SEED_GROUNDED),
        _sample(1),
        _sample(2),
    ]
    plan = plan_identity_checkpoints(
        samples,
        asset_id="asset-generic",
        track_fingerprint="1" * 64,
        max_model_checks=2,
        seed_sample_index=0,
    )

    execution = execute_identity_checkpoints(
        plan,
        frames_by_sample_index={1: _frame(1), 2: _frame(2)},
        verifier=lambda _candidate, _frame: IdentityCheckpointModelDecision(
            verdict=IdentityCheckpointVerdict.MATCHED,
            confidence=0.8,
            evidence="The locked instance remains visibly consistent.",
        ),
        verifier_id="test-verifier-v1",
    )

    assert execution.semantic_checkpoint_status == SemanticCheckpointStatus.PASSED
    assert execution.model_calls_made == plan.selected_count == 2
    assert len(execution.execution_request_sha256) == 64


def test_checkpoint_executor_preserves_mismatch_and_verifier_error() -> None:
    samples = [_sample(0), _sample(1), _sample(2)]
    plan = plan_identity_checkpoints(
        samples,
        asset_id="asset-generic",
        track_fingerprint="2" * 64,
        max_model_checks=2,
        seed_sample_index=0,
    )

    def verifier(candidate, _frame):
        if candidate.sample_index == 1:
            return IdentityCheckpointModelDecision(
                verdict=IdentityCheckpointVerdict.TARGET_MISMATCH,
                confidence=0.9,
                evidence="The visible instance conflicts with the locked identity.",
            )
        raise RuntimeError("simulated verifier outage")

    execution = execute_identity_checkpoints(
        plan,
        frames_by_sample_index={1: _frame(1), 2: _frame(2)},
        verifier=verifier,
        verifier_id="test-verifier-v1",
    )

    assert execution.semantic_checkpoint_status == SemanticCheckpointStatus.FAILED
    assert {result.verdict for result in execution.results} == {
        IdentityCheckpointVerdict.TARGET_MISMATCH,
        IdentityCheckpointVerdict.VERIFIER_ERROR,
    }
    error = next(
        result
        for result in execution.results
        if result.verdict == IdentityCheckpointVerdict.VERIFIER_ERROR
    )
    assert error.error_type == "RuntimeError"
    assert error.error_message == "simulated verifier outage"


def test_checkpoint_executor_rejects_wrong_exact_frame_pts() -> None:
    plan = plan_identity_checkpoints(
        [_sample(0), _sample(1)],
        asset_id="asset-generic",
        track_fingerprint="3" * 64,
        max_model_checks=1,
        seed_sample_index=0,
    )
    wrong = _frame(1).model_copy(update={"frame_pts": 999})

    with pytest.raises(ValueError, match="PTS mismatch"):
        execute_identity_checkpoints(
            plan,
            frames_by_sample_index={1: wrong},
            verifier=lambda _candidate, _frame: IdentityCheckpointModelDecision(
                verdict=IdentityCheckpointVerdict.MATCHED,
                evidence="match",
            ),
            verifier_id="test-verifier-v1",
        )


def test_zero_budget_with_risky_candidates_remains_pending() -> None:
    plan = plan_identity_checkpoints(
        [_sample(0), _sample(1)],
        asset_id="asset-generic",
        track_fingerprint="4" * 64,
        max_model_checks=0,
    )

    execution = execute_identity_checkpoints(
        plan,
        frames_by_sample_index={},
        verifier=lambda _candidate, _frame: IdentityCheckpointModelDecision(
            verdict=IdentityCheckpointVerdict.MATCHED,
            evidence="unreachable",
        ),
        verifier_id="test-verifier-v1",
    )

    assert (
        execution.semantic_checkpoint_status
        == SemanticCheckpointStatus.REQUIRED_PENDING
    )
    assert execution.model_calls_made == 0


def test_checkpoint_executor_aborts_instead_of_repeating_fatal_quota_error() -> None:
    plan = plan_identity_checkpoints(
        [_sample(0), _sample(1), _sample(2)],
        asset_id="asset-generic",
        track_fingerprint="5" * 64,
        max_model_checks=2,
        seed_sample_index=0,
    )
    attempts = 0

    def verifier(_candidate, _frame):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(RuntimeError, match="429"):
        execute_identity_checkpoints(
            plan,
            frames_by_sample_index={1: _frame(1), 2: _frame(2)},
            verifier=verifier,
            verifier_id="test-verifier-v1",
            abort_on_error=lambda error: "429" in str(error),
        )

    assert attempts == 1
