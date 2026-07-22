from __future__ import annotations

from jascue_video_lab.auto_reframe import (
    AutoReframePolicy,
    CandidatePreflight,
    FailureCode,
    RecoveryAction,
    RegionAssessment,
    audit_auto_bounded_clip,
    choose_recovery,
    failure_codes_for_preflight,
    rank_preflights,
)


def preflight(*, rank: int = 1, soft_visible: float = 0.8) -> CandidatePreflight:
    return CandidatePreflight(
        candidate_id=f"candidate-{rank}",
        rank=rank,
        presentation="tracked_crop",
        source_lineage_valid=True,
        within_single_shot=True,
        evidence_confidence=0.8,
        semantic_status="matched",
        tracking_confidence_gate_passed=True,
        tracking_coverage_passed=True,
        semantic_checkpoints_passed=True,
        regions=[
            RegionAssessment(
                region_id="core",
                role="hard_core",
                minimum_visible_fraction=1.0,
                required_visible_fraction=1.0,
            ),
            RegionAssessment(
                region_id="context",
                role="soft_extent",
                minimum_visible_fraction=soft_visible,
                required_visible_fraction=0.72,
                clipped_edges=["right"] if soft_visible < 1 else [],
            ),
        ],
        geometry_fingerprint="a" * 64,
        source_fingerprint="b" * 64,
        track_fingerprints=["c" * 64],
    )


def test_auto_bounded_clip_only_clips_soft_context() -> None:
    policy = AutoReframePolicy()
    audit = audit_auto_bounded_clip(
        preflight(), policy, expected_geometry_fingerprint="a" * 64
    )

    assert audit.approved is True
    assert audit.auto_bounded_clip_applied is True
    assert audit.failure_codes == []
    assert len(audit.audit_sha256) == 64


def test_hard_core_or_atomic_clipping_fails_closed() -> None:
    candidate = preflight()
    candidate.regions[0].minimum_visible_fraction = 0.99

    failures = failure_codes_for_preflight(candidate, AutoReframePolicy())

    assert FailureCode.HARD_CORE_NOT_FULLY_RETAINED in failures


def test_geometry_fingerprint_mismatch_is_not_approved() -> None:
    audit = audit_auto_bounded_clip(
        preflight(),
        AutoReframePolicy(),
        expected_geometry_fingerprint="d" * 64,
    )

    assert audit.approved is False
    assert FailureCode.GEOMETRY_FINGERPRINT_MISMATCH in audit.failure_codes


def test_center_crop_is_rejected_unless_policy_explicitly_opts_in() -> None:
    candidate = preflight().model_copy(update={"presentation": "center_crop"})

    assert FailureCode.NO_FEASIBLE_PRESENTATION in failure_codes_for_preflight(
        candidate, AutoReframePolicy()
    )
    assert FailureCode.NO_FEASIBLE_PRESENTATION not in failure_codes_for_preflight(
        candidate, AutoReframePolicy(allow_safe_center_crop=True)
    )


def test_reason_aware_recovery_prefers_candidate_switch_then_review() -> None:
    failures = [FailureCode.SEMANTIC_MATCH_BELOW_MINIMUM]

    assert choose_recovery(failures, candidates_remaining=True) == (
        RecoveryAction.TRY_NEXT_CANDIDATE
    )
    assert choose_recovery(failures, candidates_remaining=False) == (
        RecoveryAction.REVIEW_REQUIRED
    )


def test_ranked_candidates_put_geometry_pass_before_model_rank() -> None:
    first = preflight(rank=1, soft_visible=0.4)
    second = preflight(rank=2, soft_visible=0.9)

    ranked = rank_preflights([first, second], AutoReframePolicy())

    assert [item.candidate_id for item in ranked] == ["candidate-2", "candidate-1"]
