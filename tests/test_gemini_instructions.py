from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import jascue_video_lab.gemini as gemini_module
from jascue_video_lab.billing import summarize_usage_and_list_price

from jascue_video_lab.gemini import (
    EDITORIAL_SYSTEM_INSTRUCTION,
    GroundingIdentityReference,
    MODEL_ID,
    SEMANTIC_IDENTITY_GENERATION_CONFIG,
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
    client.model_id = MODEL_ID
    return client, interactions


def test_live_client_disables_hidden_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def close(self) -> None:
            return None

    monkeypatch.setattr(gemini_module.genai, "Client", _Client)
    client = GeminiLabClient(api_key="test-key")
    try:
        retry_options = captured["http_options"].retry_options
        assert retry_options is not None
        assert retry_options.attempts == 1
    finally:
        client.close()


def test_paid_responses_are_preserved_as_immutable_attempts(tmp_path: Path) -> None:
    class _PaidInteraction:
        id = "interaction-reused-by-test-double"

        def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "id": self.id,
                "model": "gemini-3.6-flash",
                "output_text": "{}",
                "usage": {
                    "total_input_tokens": 100,
                    "total_output_tokens": 10,
                    "total_thought_tokens": 0,
                },
            }

    interaction = _PaidInteraction()
    for _ in range(2):
        gemini_module._record_interaction_attempt(
            run_dir=tmp_path,
            operation="contract_test",
            canonical_filename="contract_test.raw_interaction.json",
            interaction=interaction,
        )

    assert len(list((tmp_path / "attempts").glob("*.raw_interaction.json"))) == 2
    assert (tmp_path / "contract_test.raw_interaction.json").is_file()
    summary = summarize_usage_and_list_price(tmp_path)
    assert summary["request_count"] == 2
    assert summary["duplicate_artifact_count"] == 1


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
        assert api_request["model"] == MODEL_ID
        assert not {"temperature", "top_p", "top_k"}.intersection(
            api_request["generation_config"]
        )
        assert api_request["generation_config"]["thinking_level"] in {"low", "high"}


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
        assert api_request["model"] == MODEL_ID
        assert not {"temperature", "top_p", "top_k"}.intersection(
            api_request["generation_config"]
        )
        assert api_request["generation_config"]["thinking_level"] in {"low", "high"}

    assert "不證明素材中存在相符畫面" in EDITORIAL_SYSTEM_INSTRUCTION
    assert "不得選擇不相符素材補位" in EDITORIAL_SYSTEM_INSTRUCTION
    assert "OPPO" not in EDITORIAL_SYSTEM_INSTRUCTION
    assert "Reno" not in EDITORIAL_SYSTEM_INSTRUCTION


def test_live_request_sources_do_not_use_deprecated_sampling_parameters() -> None:
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "src", root / "scripts"):
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert '"temperature"' not in source, path
            assert '"top_p"' not in source, path
            assert '"top_k"' not in source, path


def test_ground_frame_interleaves_content_addressed_identity_references(
    tmp_path: Path,
) -> None:
    target_frame = tmp_path / "target-frame.png"
    target_frame.write_bytes(b"target frame bytes")
    target_hash = hashlib.sha256(target_frame.read_bytes()).hexdigest()
    positive = tmp_path / "positive.png"
    positive.write_bytes(b"same locked instance")
    negative = tmp_path / "negative.png"
    negative.write_bytes(b"explicit confuser")

    references = (
        GroundingIdentityReference(
            reference_id="anchor-positive",
            role="positive",
            target_id="subject.primary",
            description="same reviewer-selected instance",
            path=positive,
            sha256=hashlib.sha256(positive.read_bytes()).hexdigest(),
        ),
        GroundingIdentityReference(
            reference_id="anchor-negative",
            role="negative",
            target_id="subject.primary",
            description="similar instance that must be excluded",
            path=negative,
            sha256=hashlib.sha256(negative.read_bytes()).hexdigest(),
        ),
    )
    media = SimpleNamespace(asset_id="sha256:" + "a" * 64)
    frame = SimpleNamespace(
        path=str(target_frame),
        frame_time_ms=1250,
        frame_pts=30,
        frame_hash=target_hash,
        width=1920,
        height=1080,
    )

    api_request, saved_request = _capture_request(
        tmp_path,
        "ground-references",
        "grounding.request.json",
        lambda client, run_dir: client.ground_frame(
            media=media,
            frame=frame,
            event_id="event-generic",
            event_description="identity-only exact-frame grounding",
            entity_id="subject.primary",
            target_description="the locked foreground instance",
            prompt_template=(
                "Target {{target_description}} in event {{event_description}} "
                "for {{entity_id}} at {{frame_time_ms}}."
            ),
            run_id="run-ground-references",
            output_dir=run_dir,
            identity_references=references,
        ),
    )

    api_input = api_request["input"]
    labels = [item["text"] for item in api_input if item["type"] == "text"]
    assert any("anchor-positive" in label and "role=positive" in label for label in labels)
    assert any("anchor-negative" in label and "role=negative" in label for label in labels)
    assert labels[-1].startswith("FRAME_TO_GROUND")
    assert sum(item["type"] == "image" for item in api_input) == 3

    recorded_images = [
        item for item in saved_request["input"] if item["type"] == "image"
    ]
    assert [item.get("reference_role") for item in recorded_images[:-1]] == [
        "positive",
        "negative",
    ]
    assert recorded_images[-1]["image_role"] == "frame_to_ground"
    assert all("data" not in item for item in recorded_images)
    saved_text = json.dumps(saved_request, ensure_ascii=False)
    assert str(positive) not in saved_text
    assert str(negative) not in saved_text


def test_ground_frame_rejects_tampered_identity_reference_before_network(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"actual bytes")
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"frame bytes")
    client, interactions = _client()

    with pytest.raises(ValueError, match="hash mismatch"):
        client.ground_frame(
            media=SimpleNamespace(asset_id="sha256:" + "a" * 64),
            frame=SimpleNamespace(
                path=str(frame_path),
                frame_time_ms=0,
                frame_pts=0,
                frame_hash=hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                width=640,
                height=360,
            ),
            event_id="event-generic",
            event_description="identity-only",
            entity_id="subject.primary",
            target_description="the locked instance",
            prompt_template="Ground {{target_description}}.",
            run_id="run-tampered-reference",
            output_dir=tmp_path / "tampered",
            identity_references=(
                GroundingIdentityReference(
                    reference_id="tampered",
                    role="positive",
                    target_id="subject.primary",
                    description="same instance",
                    path=reference_path,
                    sha256="0" * 64,
                ),
            ),
        )
    assert interactions.request is None


def test_ground_frame_rejects_reference_for_another_requested_target(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(b"valid reference bytes")
    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"frame bytes")
    client, interactions = _client()

    with pytest.raises(ValueError, match="requested entity_id"):
        client.ground_frame(
            media=SimpleNamespace(asset_id="sha256:" + "a" * 64),
            frame=SimpleNamespace(
                path=str(frame_path),
                frame_time_ms=0,
                frame_pts=0,
                frame_hash=hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                width=640,
                height=360,
            ),
            event_id="event-generic",
            event_description="identity-only",
            entity_id="subject.requested",
            target_description="the requested instance",
            prompt_template="Ground {{target_description}}.",
            run_id="run-wrong-target-reference",
            output_dir=tmp_path / "wrong-target",
            identity_references=(
                GroundingIdentityReference(
                    reference_id="wrong-target",
                    role="positive",
                    target_id="subject.other",
                    description="another instance",
                    path=reference_path,
                    sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                ),
            ),
        )
    assert interactions.request is None


def test_identity_checkpoint_is_exact_frame_verify_only_request(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "verify-frame.png"
    frame_path.write_bytes(b"frame to verify")
    reference_path = tmp_path / "verify-reference.png"
    reference_path.write_bytes(b"locked reference")
    frame_hash = hashlib.sha256(frame_path.read_bytes()).hexdigest()
    reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()

    api_request, saved_request = _capture_request(
        tmp_path,
        "identity-checkpoint",
        "identity_checkpoint.request.json",
        lambda client, run_dir: client.verify_identity_checkpoint(
            frame=SimpleNamespace(
                path=str(frame_path),
                frame_time_ms=2250,
                frame_pts=54,
                frame_hash=frame_hash,
            ),
            target_id="subject.primary",
            target_description="the reviewer-locked foreground instance",
            run_id="identity-checkpoint-run",
            output_dir=run_dir,
            identity_references=(
                GroundingIdentityReference(
                    reference_id="positive-anchor",
                    role="positive",
                    target_id="subject.primary",
                    description="same locked instance",
                    path=reference_path,
                    sha256=reference_hash,
                ),
            ),
        ),
    )

    assert api_request["system_instruction"] == VISUAL_EVIDENCE_SYSTEM_INSTRUCTION
    assert api_request["generation_config"] == SEMANTIC_IDENTITY_GENERATION_CONFIG
    assert api_request["response_format"]["schema"]["properties"].keys() >= {
        "verdict",
        "evidence",
    }
    texts = [
        item["text"] for item in api_request["input"] if item["type"] == "text"
    ]
    assert texts[0].startswith("## Mode: VERIFY_IDENTITY")
    assert "不得輸出或修改 bounding box" in texts[0]
    assert texts[-1].startswith("FRAME_TO_VERIFY")
    assert all(
        "data" not in item
        for item in saved_request["input"]
        if item["type"] == "image"
    )
