from __future__ import annotations

import pytest

from scripts.render_music_cue_preview import build_preview_timing


def _visual_map() -> dict[str, object]:
    return {
        "points": [
            {
                "visual_event_id": "start",
                "feature_id": "a",
                "project_time_ms": 0,
            },
            {
                "visual_event_id": "middle",
                "feature_id": "b",
                "project_time_ms": 1_000,
            },
            {
                "visual_event_id": "end",
                "feature_id": "end",
                "project_time_ms": 2_000,
            },
        ]
    }


def _cue_plan(middle_time_ms: int = 1_120) -> dict[str, object]:
    return {
        "alignments": [
            {
                "visual_event_id": "start",
                "status": "aligned",
                "within_authorized_window": True,
                "proposed_project_time_ms": 40,
                "music_cue_id": "cue-start",
                "music_cue_kind": "beat",
            },
            {
                "visual_event_id": "middle",
                "status": "aligned",
                "within_authorized_window": True,
                "proposed_project_time_ms": middle_time_ms,
                "music_cue_id": "cue-middle",
                "music_cue_kind": "downbeat",
            },
            {
                "visual_event_id": "end",
                "status": "aligned",
                "within_authorized_window": True,
                "proposed_project_time_ms": 1_980,
                "music_cue_id": "cue-end",
                "music_cue_kind": "beat",
            },
        ]
    }


def test_preview_applies_only_internal_authorized_boundaries() -> None:
    original, proposed, audit = build_preview_timing(_visual_map(), _cue_plan())
    assert original == [0, 1_000, 2_000]
    assert proposed == [0, 1_120, 2_000]
    assert audit[0]["application_status"] == "kept_original_boundary"
    assert audit[1]["application_status"] == "applied_inside_authorized_window"
    assert audit[1]["music_cue_id"] == "cue-middle"
    assert audit[2]["application_status"] == "kept_original_boundary"


def test_preview_rejects_non_monotonic_boundaries() -> None:
    with pytest.raises(ValueError, match="strictly ordered"):
        build_preview_timing(_visual_map(), _cue_plan(middle_time_ms=2_000))
