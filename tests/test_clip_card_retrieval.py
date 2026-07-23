from __future__ import annotations

from jascue_video_lab.clip_card_retrieval import FeatureChapterShortlist


def test_partial_shortlist_allows_one_real_candidate() -> None:
    chapter = FeatureChapterShortlist.model_validate(
        {
            "feature_id": "feature",
            "evidence_status": "partial",
            "candidates": [
                {
                    "source_asset_id": "sha256:" + "1" * 64,
                    "event_id": "event-1",
                    "retrieval_reason": "Only directly observed partial evidence.",
                }
            ],
        }
    )
    assert len(chapter.candidates) == 1


def test_not_found_shortlist_cannot_carry_candidates() -> None:
    try:
        FeatureChapterShortlist.model_validate(
            {
                "feature_id": "feature",
                "evidence_status": "not_found",
                "candidates": [
                    {
                        "source_asset_id": "sha256:" + "1" * 64,
                        "event_id": "event-1",
                        "retrieval_reason": "No evidence.",
                    }
                ],
            }
        )
    except ValueError as error:
        assert "not_found" in str(error)
    else:
        raise AssertionError("not_found shortlist accepted a candidate")
