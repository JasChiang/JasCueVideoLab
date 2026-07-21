from __future__ import annotations

from dataclasses import dataclass

import pytest

from jascue_video_lab.grounding_selection import (
    GroundingSelectionError,
    require_grounding_request_match,
    require_tracking_seed_candidate,
)


@dataclass
class _Candidate:
    label: str
    confidence: float


@dataclass
class _Proposal:
    visible: bool
    candidates: list[_Candidate]
    match_status: str
    predicate_status: str = "not_applicable"


@dataclass(frozen=True)
class _Provenance:
    model_id: str


@dataclass
class _CachedProposal:
    asset_id: str = "sha256:source"
    event_id: str = "event-1"
    entity_id: str = "subject-1"
    frame_pts: int = 123
    frame_time_ms: int = 1000
    frame_hash: str = "a" * 64
    source_width: int = 1920
    source_height: int = 1080
    model_provenance: _Provenance = _Provenance("gemini-3.5-flash")


def test_single_matched_candidate_is_the_only_automatic_seed() -> None:
    candidate = _Candidate(label="generic subject", confidence=0.1)
    selected = require_tracking_seed_candidate(  # type: ignore[arg-type]
        _Proposal(visible=True, candidates=[candidate], match_status="matched")
    )
    assert selected.candidate is candidate
    assert selected.selection_source == "single_matched_candidate"


def test_confidence_cannot_resolve_multiple_candidates() -> None:
    proposal = _Proposal(
        visible=True,
        candidates=[
            _Candidate(label="candidate A", confidence=1.0),
            _Candidate(label="candidate B", confidence=0.1),
        ],
        match_status="ambiguous",
    )
    with pytest.raises(GroundingSelectionError, match="explicit review"):
        require_tracking_seed_candidate(proposal)  # type: ignore[arg-type]

    selected = require_tracking_seed_candidate(  # type: ignore[arg-type]
        proposal, candidate_number=2
    )
    assert selected.candidate.label == "candidate B"
    assert selected.candidate_number == 2
    assert selected.candidate_index == 1
    assert selected.selection_source == "explicit_candidate_number"


@pytest.mark.parametrize(
    "status", ["not_visible", "target_mismatch", "insufficient_evidence"]
)
def test_explicit_index_cannot_override_hard_semantic_rejection(status: str) -> None:
    proposal = _Proposal(
        visible=True,
        candidates=[_Candidate(label="candidate", confidence=1.0)],
        match_status=status,
    )
    with pytest.raises(GroundingSelectionError, match="semantic match status"):
        require_tracking_seed_candidate(proposal, candidate_number=1)  # type: ignore[arg-type]


def test_candidate_number_matches_one_based_debug_labels() -> None:
    proposal = _Proposal(
        visible=True,
        candidates=[
            _Candidate(label="candidate A", confidence=0.2),
            _Candidate(label="candidate B", confidence=0.1),
        ],
        match_status="ambiguous",
    )
    with pytest.raises(GroundingSelectionError, match="outside 1..2"):
        require_tracking_seed_candidate(proposal, candidate_number=0)  # type: ignore[arg-type]


def test_locked_predicate_must_be_satisfied_before_tracking() -> None:
    proposal = _Proposal(
        visible=True,
        candidates=[_Candidate(label="selected subject", confidence=0.9)],
        match_status="matched",
        predicate_status="indeterminate",
    )
    with pytest.raises(GroundingSelectionError, match="observable predicate"):
        require_tracking_seed_candidate(  # type: ignore[arg-type]
            proposal,
            require_predicate_satisfied=True,
        )


def test_cached_grounding_must_match_exact_frame_and_model() -> None:
    proposal = _CachedProposal()
    require_grounding_request_match(  # type: ignore[arg-type]
        proposal,
        asset_id=proposal.asset_id,
        event_id=proposal.event_id,
        entity_id=proposal.entity_id,
        frame_pts=proposal.frame_pts,
        frame_time_ms=proposal.frame_time_ms,
        frame_hash=proposal.frame_hash,
        source_width=proposal.source_width,
        source_height=proposal.source_height,
        model_id=proposal.model_provenance.model_id,
    )
    stale = _CachedProposal(frame_hash="b" * 64)
    with pytest.raises(GroundingSelectionError, match="exact request"):
        require_grounding_request_match(  # type: ignore[arg-type]
            stale,
            asset_id=proposal.asset_id,
            event_id=proposal.event_id,
            entity_id=proposal.entity_id,
            frame_pts=proposal.frame_pts,
            frame_time_ms=proposal.frame_time_ms,
            frame_hash=proposal.frame_hash,
            source_width=proposal.source_width,
            source_height=proposal.source_height,
            model_id=proposal.model_provenance.model_id,
        )
