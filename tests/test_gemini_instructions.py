from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from jascue_video_lab.gemini import (
    EDITORIAL_SYSTEM_INSTRUCTION,
    VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
    GeminiLabClient,
)


class _StopRequest(RuntimeError):
    pass


class _RejectingInteractions:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    def create(self, **request: Any) -> None:
        self.request = request
        raise _StopRequest("request captured")


def _client() -> tuple[GeminiLabClient, _RejectingInteractions]:
    interactions = _RejectingInteractions()
    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=interactions)
    client.temperature = 0.2
    return client, interactions


def _capture_request(
    tmp_path: Path,
    name: str,
    request_filename: str,
    invoke: Callable[[GeminiLabClient, Path], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = tmp_path / name
    run_dir.mkdir()
    client, interactions = _client()
    with pytest.raises(_StopRequest, match="request captured"):
        invoke(client, run_dir)
    assert interactions.request is not None
    saved = json.loads((run_dir / request_filename).read_text(encoding="utf-8"))
    return interactions.request, saved


def test_all_candidate_and_frame_observation_calls_use_visual_evidence_instruction(
    tmp_path: Path,
) -> None:
    media = SimpleNamespace(asset_id="sha256:" + "a" * 64, duration_ms=10_000)
    uploaded = SimpleNamespace(uri="https://example.invalid/video", mime_type="video/mp4")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"test image payload")

    event = SimpleNamespace(
        event_id="event-generic",
        grounding_targets=[],
        model_dump_json=lambda indent=None: "{}",
    )
    dense_catalog = SimpleNamespace(
        source_asset_id=media.asset_id,
        frames=[SimpleNamespace(frame_id="DF000001")],
        contact_sheet_paths=[str(image)],
        contact_sheet_hashes=["b" * 64],
    )

    calls: list[tuple[str, str, Callable[[GeminiLabClient, Path], Any]]] = [
        (
            "targets",
            "target_candidates.request.json",
            lambda client, run_dir: client.suggest_targets(
                media=media,
                uploaded=uploaded,
                prompt_template="Find observable candidate targets.",
                run_id="run-targets",
                run_dir=run_dir,
            ),
        ),
        (
            "storyboard",
            "indexed_storyboard.request.json",
            lambda client, run_dir: client.analyze_indexed_storyboard(
                media=media,
                frames=[
                    {
                        "frame_id": "F000001",
                        "frame_pts": 0,
                        "frame_time_ms": 0,
                        "image_path": str(image),
                        "image_hash": "c" * 64,
                    }
                ],
                prompt_template="Describe only the supplied frames.",
                run_id="run-storyboard",
                run_dir=run_dir,
            ),
        ),
        (
            "moments",
            "direct_moments.request.json",
            lambda client, run_dir: client.analyze_direct_moments(
                media=media,
                uploaded=uploaded,
                prompt_template="Suggest observable moments.",
                run_id="run-moments",
                run_dir=run_dir,
                locked_target_id="entity-generic",
                locked_target_description="the user-selected physical object",
            ),
        ),
        (
            "dense",
            "dense_selection.request.json",
            lambda client, run_dir: client.select_dense_event_frames(
                event=event,
                catalog=dense_catalog,
                prompt_template="Select only supplied frame IDs.",
                run_id="run-dense",
                run_dir=run_dir,
            ),
        ),
    ]

    for name, filename, invoke in calls:
        api_request, saved_request = _capture_request(
            tmp_path, name, filename, invoke
        )
        assert api_request["system_instruction"] == VISUAL_EVIDENCE_SYSTEM_INSTRUCTION
        assert saved_request["system_instruction"] == VISUAL_EVIDENCE_SYSTEM_INSTRUCTION


def test_edit_planning_calls_separate_intent_from_media_evidence(tmp_path: Path) -> None:
    uploaded = SimpleNamespace(uri="https://example.invalid/reel", mime_type="video/mp4")
    catalog = SimpleNamespace(
        catalog_id="catalog-generic",
        frames=[SimpleNamespace(frame_id="RF000001")],
    )
    brief = SimpleNamespace(
        project_id="project-generic",
        model_dump_json=lambda indent=None: '{"chapters": []}',
    )

    calls: list[tuple[str, str, Callable[[GeminiLabClient, Path], Any]]] = [
        (
            "rushes",
            "rushes_edit_plan.request.json",
            lambda client, run_dir: client.plan_rushes_edit(
                catalog=catalog,
                uploaded=uploaded,
                prompt_template="Plan an evidence-backed edit.",
                project_id="project-generic",
                run_id="run-rushes",
                run_dir=run_dir,
            ),
        ),
        (
            "feature",
            "feature_edit_plan.request.json",
            lambda client, run_dir: client.plan_feature_edit(
                catalog=catalog,
                brief=brief,
                uploaded=uploaded,
                prompt_template="Match the brief to supported footage.",
                run_id="run-feature",
                run_dir=run_dir,
            ),
        ),
    ]

    for name, filename, invoke in calls:
        api_request, saved_request = _capture_request(
            tmp_path, name, filename, invoke
        )
        assert api_request["system_instruction"] == EDITORIAL_SYSTEM_INSTRUCTION
        assert saved_request["system_instruction"] == EDITORIAL_SYSTEM_INSTRUCTION

    assert "不證明素材中存在相符畫面" in EDITORIAL_SYSTEM_INSTRUCTION
    assert "不得選擇不相符素材補位" in EDITORIAL_SYSTEM_INSTRUCTION
    assert "OPPO" not in EDITORIAL_SYSTEM_INSTRUCTION
    assert "Reno" not in EDITORIAL_SYSTEM_INSTRUCTION
