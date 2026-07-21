from __future__ import annotations

from jascue_video_lab.models import (
    DenseFrame,
    DenseFrameCatalog,
    ModelProvenance,
    TrimIntentProposal,
)
from jascue_video_lab.storage import write_json
from jascue_video_lab.trim_intent import derive_trim_decision, review_trim_decision


def _provenance() -> ModelProvenance:
    return ModelProvenance(
        model_id="gemini-3.5-flash",
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version="2.3.0",
        interaction_id="interaction-1",
        run_id="trim-test",
        generated_at="2026-07-21T00:00:00Z",
    )


def _catalog() -> DenseFrameCatalog:
    frames = [
        DenseFrame(
            frame_id=f"DF{index:06d}",
            event_id="event-1",
            requested_time_ms=index * 250,
            frame_time_ms=index * 250,
            frame_pts=index * 30,
            frame_hash=f"{index:064x}",
            width=960,
            height=540,
            image_path=f"/tmp/frame-{index}.jpg",
            transport_image_path=f"/tmp/transport-{index}.jpg",
            transport_image_hash=f"{index + 10:064x}",
        )
        for index in range(1, 7)
    ]
    return DenseFrameCatalog(
        source_asset_id="sha256:source",
        event_id="event-1",
        sampling_fps=4,
        source_start_ms=250,
        source_end_ms=2000,
        frames=frames,
        contact_sheet_paths=["/tmp/page.jpg"],
        contact_sheet_hashes=["a" * 64],
        generated_at="2026-07-21T00:00:00Z",
    )


def _proposal() -> TrimIntentProposal:
    return TrimIntentProposal(
        source_asset_id="sha256:source",
        event_id="event-1",
        usable=True,
        selections=[
            {"phase": "setup_start", "frame_id": "DF000001"},
            {"phase": "action_start", "frame_id": "DF000002"},
            {"phase": "result_start", "frame_id": "DF000003"},
            {"phase": "hold_start", "frame_id": "DF000004"},
            {"phase": "hold_end", "frame_id": "DF000005"},
            {"phase": "reset_start", "frame_id": "DF000006"},
            {"phase": "recommended_in", "frame_id": "DF000002"},
            {"phase": "recommended_out", "frame_id": "DF000005"},
        ],
        tail_intent="title_safe_hold",
        observed_phase_evidence="action, result, and stable hold are visible",
        hold_evidence="subject remains stable with negative space",
        trim_reason="preserve the complete action and proposed hold",
        quality_risks=[],
        uncertainties=["director intent requires review"],
        requires_human_review=True,
        confidence=0.8,
        model_provenance=_provenance(),
    )


def test_trim_decision_maps_last_included_id_to_exclusive_pts(tmp_path) -> None:
    proposal_path = tmp_path / "proposal.json"
    catalog_path = tmp_path / "catalog.json"
    decision = derive_trim_decision(
        _proposal(),
        _catalog(),
        shot_id="shot-1",
        shot_start_ms=0,
        shot_end_ms=3000,
        proposal_path=proposal_path,
        catalog_path=catalog_path,
    )

    assert decision.source_in_ms == 500
    assert decision.source_out_ms == 1250
    assert decision.source_in_pts == 60
    assert decision.source_out_pts == 150
    assert decision.exclusive_out_frame is not None
    assert decision.exclusive_out_frame.frame_id == "DF000005"
    assert decision.last_included_frame is not None
    assert decision.last_included_frame.frame_id == "DF000004"
    assert decision.approval_status == "proposed"
    assert decision.requires_human_review is True


def test_human_review_is_required_before_trim_approval(tmp_path) -> None:
    decision_path = tmp_path / "trim-decision.json"
    output_path = tmp_path / "trim-decision.reviewed.json"
    decision = derive_trim_decision(
        _proposal(),
        _catalog(),
        shot_id="shot-1",
        shot_start_ms=0,
        shot_end_ms=3000,
        proposal_path=tmp_path / "proposal.json",
        catalog_path=tmp_path / "catalog.json",
    )
    write_json(decision_path, decision)

    reviewed = review_trim_decision(
        decision_path,
        output_path,
        reviewer="human-reviewer",
        decision="approved",
        notes="hold is intentional",
    )

    assert reviewed.approval_status == "approved"
    assert reviewed.requires_human_review is False
    assert reviewed.human_review is not None
    assert reviewed.human_review.reviewer == "human-reviewer"
    assert output_path.exists()
