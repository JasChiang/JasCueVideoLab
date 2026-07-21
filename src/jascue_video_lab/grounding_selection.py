from __future__ import annotations

from dataclasses import dataclass

from .models import GroundingCandidate, GroundingProposal


class GroundingSelectionError(ValueError):
    """Raised when a model proposal is unsafe to use as a tracking seed."""


@dataclass(frozen=True)
class SelectedGroundingCandidate:
    candidate: GroundingCandidate
    candidate_number: int
    candidate_index: int
    selection_source: str


def _status_value(proposal: GroundingProposal) -> str:
    status = getattr(proposal, "match_status", None)
    if status is None:
        return "matched" if proposal.visible and len(proposal.candidates) == 1 else "ambiguous"
    return str(status)


def require_grounding_request_match(
    proposal: GroundingProposal,
    *,
    asset_id: str,
    event_id: str,
    entity_id: str,
    frame_pts: int,
    frame_time_ms: int,
    frame_hash: str,
    source_width: int,
    source_height: int,
    model_id: str,
) -> None:
    """Fail closed when a cached proposal does not belong to this exact request."""
    expected = {
        "asset_id": asset_id,
        "event_id": event_id,
        "entity_id": entity_id,
        "frame_pts": frame_pts,
        "frame_time_ms": frame_time_ms,
        "frame_hash": frame_hash,
        "source_width": source_width,
        "source_height": source_height,
    }
    mismatches = {
        field: {"expected": value, "actual": getattr(proposal, field)}
        for field, value in expected.items()
        if getattr(proposal, field) != value
    }
    if proposal.model_provenance.model_id != model_id:
        mismatches["model_id"] = {
            "expected": model_id,
            "actual": proposal.model_provenance.model_id,
        }
    if mismatches:
        raise GroundingSelectionError(
            f"cached Grounding proposal does not match the exact request: {mismatches}"
        )


def require_tracking_seed_candidate(
    proposal: GroundingProposal,
    *,
    candidate_number: int | None = None,
    require_predicate_satisfied: bool = False,
) -> SelectedGroundingCandidate:
    """Choose a bbox seed without treating model confidence as identity evidence.

    An unreviewed proposal is accepted only when the target is visible, the model
    reports a semantic match, and exactly one candidate exists.  An explicit
    candidate number is a human/operator override for a multi-candidate proposal;
    it never overrides an absent, mismatched, or insufficient-evidence target.

    Candidate numbers are deliberately 1-based because the debug overlay presents
    candidates as ``1.``, ``2.``, and so on.  ``candidate_index`` remains in the
    returned audit record as the corresponding zero-based array index.
    """

    if not proposal.visible or not proposal.candidates:
        raise GroundingSelectionError("tracking seed target is not visibly grounded")

    status = _status_value(proposal)
    hard_rejections = {"not_visible", "target_mismatch", "insufficient_evidence"}
    if status in hard_rejections:
        raise GroundingSelectionError(
            f"tracking seed rejected by semantic match status: {status}"
        )

    if require_predicate_satisfied:
        predicate_status = str(getattr(proposal, "predicate_status", "indeterminate"))
        if predicate_status != "satisfied":
            raise GroundingSelectionError(
                "tracking seed does not satisfy the locked observable predicate: "
                f"{predicate_status}"
            )

    if candidate_number is None:
        if status != "matched":
            raise GroundingSelectionError(
                f"tracking seed requires explicit review for match status: {status}"
            )
        if len(proposal.candidates) != 1:
            raise GroundingSelectionError(
                "multiple Grounding candidates require an explicit candidate number"
            )
        return SelectedGroundingCandidate(
            candidate=proposal.candidates[0],
            candidate_number=1,
            candidate_index=0,
            selection_source="single_matched_candidate",
        )

    if not 1 <= candidate_number <= len(proposal.candidates):
        raise GroundingSelectionError(
            f"candidate number {candidate_number} is outside 1..{len(proposal.candidates)}"
        )
    candidate_index = candidate_number - 1
    return SelectedGroundingCandidate(
        candidate=proposal.candidates[candidate_index],
        candidate_number=candidate_number,
        candidate_index=candidate_index,
        selection_source="explicit_candidate_number",
    )
