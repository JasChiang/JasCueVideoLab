from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from jascue_video_lab.full_v1 import (
    _matching_dense_seed_frame,
    _query_lock_event_description,
    _query_lock_target_description,
    _render_dense_contact_sheets,
    _revalidate_saved_clip_card,
    _select_query_lock_target,
    _shared_upload_dir,
    create_dense_event_catalog,
    create_shot_catalog,
    dense_window_for_event,
    dense_sampling_fps,
    derive_clip_timeline,
    mmss_to_ms,
    run_full_clip,
    run_full_library,
    run_selected_full_clips,
    selected_clip_ids_from_feature_plan,
)
from jascue_video_lab.gemini import MODEL_ID, VISUAL_EVIDENCE_SYSTEM_INSTRUCTION
from jascue_video_lab.media import create_analysis_proxy, has_audio_stream, probe_video
from jascue_video_lab.models import (
    DenseEventSelection,
    DenseFrame,
    DenseFrameCatalog,
    EvidenceClaimSource,
    EvidenceQueryLock,
    EvidenceQueryProvenance,
    EvidenceQueryTargetRef,
    Entity,
    EntityKind,
    FullClipCard,
    FullClipEvent,
    ModelProvenance,
    FeatureEditPlan,
    RushesCatalog,
)
from jascue_video_lab.shots import ShotManifest, ShotSegment


def _provenance(model_id: str = "gemini-3.5-flash") -> ModelProvenance:
    return ModelProvenance(
        model_id=model_id,
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version="test",
        interaction_id=None,
        run_id="test",
        generated_at="now",
    )


def _query_lock(*targets: EvidenceQueryTargetRef) -> EvidenceQueryLock:
    return EvidenceQueryLock(
        query_id="query-demo",
        revision=1,
        editorial_goal="Show directly observable evidence for the selected subject.",
        targets=list(targets),
        observable_predicate="the selected subject reaches the requested visible state",
        required_evidence=["the selected subject is directly visible"],
        negative_constraints=["do not substitute another similar instance"],
        editing_uses=["review"],
        claim_source=EvidenceClaimSource.HUMAN_REVIEW,
        provenance=EvidenceQueryProvenance(
            created_at="2026-07-21T00:00:00Z",
            created_by="reviewer",
        ),
    )


def test_query_lock_requires_explicit_target_when_multiple() -> None:
    first = EvidenceQueryTargetRef(
        target_id="subject-a",
        target_description="the foreground subject selected by the reviewer",
        positive_attributes=["marked reference instance"],
        negative_attributes=["background depiction"],
    )
    second = EvidenceQueryTargetRef(
        target_id="subject-b",
        target_description="the second selected subject",
    )
    lock = _query_lock(first, second)
    with pytest.raises(ValueError, match="multiple targets"):
        _select_query_lock_target(lock, None)
    assert _select_query_lock_target(lock, "subject-b") == second


def test_query_lock_prompt_material_is_generic_and_keeps_exclusions() -> None:
    target = EvidenceQueryTargetRef(
        target_id="subject-a",
        target_description="the foreground subject selected by the reviewer",
        positive_attributes=["marked reference instance"],
        negative_attributes=["background depiction", "reflection"],
        reference_frame_ids=["DF000010"],
    )
    lock = _query_lock(target)
    target_text = _query_lock_target_description(target)
    event_text = _query_lock_event_description(_event(), lock)
    assert "marked reference instance" in target_text
    assert "background depiction" in target_text
    assert lock.observable_predicate in event_text
    assert lock.negative_constraints[0] in event_text
    assert not any(term in (target_text + event_text).lower() for term in ("oppo", "reno"))


def _event(**updates) -> FullClipEvent:
    payload = {
        "event_id": "event-demo",
        "start_mmss": "00:00",
        "end_mmss": "00:02",
        "recommended_keyframe_mmss": "00:01",
        "label": "快速 UI 狀態",
        "description": "按鈕狀態短暫改變 0.3 秒",
        "observable_evidence": "button changes color",
        "evidence_modalities": "visual",
        "entity_ids": ["phone-screen"],
        "primary_entity_ids": ["phone-screen"],
        "required_entity_ids": ["phone-screen"],
        "optional_entity_ids": [],
        "avoid_overlay_entity_ids": ["phone-screen"],
        "keyframe_reason": "UI is visible",
        "boundary_precision": "uncertain",
        "confidence": 0.8,
        "action_completeness": "complete",
        "editing_uses": ["demo"],
        "quality_risks": ["transient state"],
        "framing_intent": "preserve the complete phone screen",
        "card_opportunities": [],
        "dense_refinement": "required",
        "dense_refinement_reasons": ["0.3 秒 UI 可能被 1 FPS 漏掉"],
        "grounding_targets": [
            {
                "entity_id": "phone-screen",
                "target_kind": "phone_screen",
                "target_description": "完整手機螢幕，不含手機外框",
                "purpose": "reframe",
            }
        ],
    }
    payload.update(updates)
    return FullClipEvent.model_validate(payload)


def _card(**updates) -> FullClipCard:
    payload = {
        "source_asset_id": "sha256:" + "1" * 64,
        "proxy_asset_id": "sha256:" + "2" * 64,
        "duration_ms": 2000,
        "summary": "phone UI demo",
        "content_type": "product_demo",
        "entities": [
            Entity(
                entity_id="phone-screen",
                kind=EntityKind.PHONE_SCREEN,
                label="phone screen",
                distinguishing_features="only screen in frame",
                evidence="visible UI",
            )
        ],
        "events": [_event()],
        "clip_uses": ["demo"],
        "portrait_reframe_feasibility": "good",
        "uncertainties": [],
        "model_provenance": _provenance(),
    }
    payload.update(updates)
    return FullClipCard.model_validate(payload)


def _make_av_video(path: Path, duration: float = 2) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s=320x180:r=30:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def _make_video_only(path: Path, duration: float = 2) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s=320x180:r=30:d={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_mmss_is_locally_converted_and_not_model_milliseconds() -> None:
    assert mmss_to_ms("01:12") == 72_000
    assert mmss_to_ms("61:05") == 3_665_000
    with pytest.raises(ValueError, match="00..59"):
        mmss_to_ms("01:60")


def test_clip_card_rejects_out_of_duration_mmss() -> None:
    with pytest.raises(ValidationError, match="exceeds duration"):
        _card(events=[_event(end_mmss="00:03")])


def test_clip_card_rejects_grounding_target_kind_mismatch() -> None:
    event = _event()
    payload = event.model_dump(mode="json")
    payload["grounding_targets"][0]["target_kind"] = "phone"
    with pytest.raises(ValidationError, match="target kind differs"):
        _card(events=[FullClipEvent.model_validate(payload)])


def test_derived_timeline_maps_mmss_to_local_ms_and_shots() -> None:
    card = _card()
    shots = ShotManifest(
        video_path="private.mp4",
        duration_ms=2000,
        detector="test",
        threshold=4,
        generated_at="now",
        boundaries=[],
        shots=[
            ShotSegment(
                shot_id="shot-0001",
                start_time_ms=0,
                end_time_ms=1000,
                start_frame_pts=None,
                boundary_source="video_start",
                boundary_score=None,
            ),
            ShotSegment(
                shot_id="shot-0002",
                start_time_ms=1000,
                end_time_ms=2000,
                start_frame_pts=30,
                boundary_source="ffmpeg_scdet",
                boundary_score=12,
            ),
        ],
    )
    timeline = derive_clip_timeline(card, shots)
    event = timeline.events[0]
    assert (event.start_ms, event.end_ms, event.recommended_keyframe_ms) == (0, 2000, 1000)
    assert event.shot_ids == ["shot-0001", "shot-0002"]
    assert event.boundary_source == "gemini_mmss_local_conversion"


def test_fast_transient_event_selects_8fps() -> None:
    assert dense_sampling_fps(_event()) == 8
    assert dense_sampling_fps(
        _event(
            label="產品展示",
            description="slow static detail",
            dense_refinement="recommended",
            dense_refinement_reasons=["confirm focus"],
        )
    ) == 4


def test_dense_window_is_local_and_cannot_cross_shot() -> None:
    event = _event(
        start_mmss="00:00",
        end_mmss="00:10",
        recommended_keyframe_mmss="00:05",
    )
    shots = ShotManifest(
        video_path="private.mp4",
        duration_ms=10_000,
        detector="test",
        threshold=4,
        generated_at="now",
        boundaries=[],
        shots=[
            ShotSegment(
                shot_id="shot-0001",
                start_time_ms=0,
                end_time_ms=6000,
                start_frame_pts=None,
                boundary_source="video_start",
                boundary_score=None,
            ),
            ShotSegment(
                shot_id="shot-0002",
                start_time_ms=6000,
                end_time_ms=10_000,
                start_frame_pts=360_000,
                boundary_source="ffmpeg_scdet",
                boundary_score=10,
            ),
        ],
    )
    assert dense_window_for_event(event, shots, window_ms=4000) == (3000, 6000, "shot-0001")


def test_dense_catalog_preserves_exact_pts_and_separate_transport(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    _make_av_video(video)
    media = probe_video(video)
    catalog = create_dense_event_catalog(
        video,
        media.asset_id,
        _event(),
        tmp_path / "dense",
        sampling_fps=8,
        max_width=160,
    )
    assert catalog.sampling_fps == 8
    assert len(catalog.frames) >= 16
    assert all(Path(frame.image_path).exists() for frame in catalog.frames)
    assert all(Path(frame.transport_image_path).exists() for frame in catalog.frames)
    assert all(frame.frame_pts >= 0 for frame in catalog.frames)
    assert all(frame.frame_hash != frame.transport_image_hash for frame in catalog.frames)


def test_dense_contact_sheet_letterboxes_without_stretching(tmp_path: Path) -> None:
    transport = tmp_path / "portrait.jpg"
    from PIL import Image

    Image.new("RGB", (100, 200), "#ff0000").save(transport)
    frame = DenseFrame(
        frame_id="DF000001",
        event_id="event-demo",
        requested_time_ms=0,
        frame_time_ms=0,
        frame_pts=0,
        frame_hash="1" * 64,
        width=100,
        height=200,
        image_path=str(transport),
        transport_image_path=str(transport),
        transport_image_hash="2" * 64,
    )
    paths, _ = _render_dense_contact_sheets([frame], tmp_path / "sheets")
    with Image.open(paths[0]).convert("RGB") as sheet:
        assert sheet.getpixel((10, 100))[0] < 80
        assert sheet.getpixel((160, 100))[0] > 180


def test_shot_catalog_keeps_one_lightweight_middle_jpeg(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    _make_av_video(video)
    media = probe_video(video)
    manifest = ShotManifest(
        video_path=str(video),
        duration_ms=media.duration_ms,
        detector="test",
        threshold=4,
        generated_at="now",
        boundaries=[],
        shots=[
            ShotSegment(
                shot_id="shot-0001",
                start_time_ms=0,
                end_time_ms=media.duration_ms,
                start_frame_pts=None,
                boundary_source="video_start",
                boundary_score=None,
            )
        ],
    )
    catalog = create_shot_catalog(
        video,
        media.asset_id,
        manifest,
        tmp_path / "shots",
    )
    assert [frame.role for frame in catalog.frames] == ["middle"]
    assert Path(catalog.frames[0].image_path).suffix == ".jpg"
    assert catalog.frames[0].frame_time_ms < media.duration_ms


def test_analysis_proxy_can_preserve_audio(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    proxy = tmp_path / "proxy.mp4"
    _make_av_video(source)
    _, record = create_analysis_proxy(
        source,
        proxy,
        max_side=640,
        fps=15,
        preserve_audio=True,
    )
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(proxy),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = {stream["codec_type"] for stream in json.loads(probe.stdout)["streams"]}
    assert stream_types == {"video", "audio"}
    assert record["preserve_audio"] is True
    assert record["source_has_audio"] is True
    assert record["proxy_has_audio"] is True


def test_full_clip_audio_auto_accepts_video_only_source(tmp_path: Path) -> None:
    source = tmp_path / "video-only.mp4"
    output = tmp_path / "prepared"
    _make_video_only(source, duration=1)

    result = run_full_clip(
        source,
        output,
        clip_card_prompt="unused",
        dense_prompt="unused",
        audio_mode="auto",
        prepare_only=True,
    )

    assert result["source_has_audio"] is False
    assert result["proxy_has_audio"] is False
    assert has_audio_stream(output / "analysis-proxy.mp4") is False
    assert json.loads((output / "analysis-proxy.json").read_text())["audio_mode"] == "auto"


def test_full_clip_audio_off_strips_existing_audio(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "prepared"
    _make_av_video(source, duration=1)

    result = run_full_clip(
        source,
        output,
        clip_card_prompt="unused",
        dense_prompt="unused",
        audio_mode="off",
        prepare_only=True,
    )

    assert result["source_has_audio"] is True
    assert result["proxy_has_audio"] is False
    assert has_audio_stream(output / "analysis-proxy.mp4") is False


def test_full_clip_audio_required_rejects_video_only_source(tmp_path: Path) -> None:
    source = tmp_path / "video-only.mp4"
    _make_video_only(source, duration=1)

    with pytest.raises(ValueError, match="source has no audio"):
        run_full_clip(
            source,
            tmp_path / "prepared",
            clip_card_prompt="unused",
            dense_prompt="unused",
            audio_mode="required",
            prepare_only=True,
        )


def test_invisible_dense_selection_has_no_ids_or_target() -> None:
    selection = DenseEventSelection(
        source_asset_id="sha256:" + "1" * 64,
        event_id="event-demo",
        visible=False,
        first_frame_id=None,
        recommended_frame_id=None,
        last_frame_id=None,
        target_entity_id=None,
        target_description=None,
        observable_evidence="not visible",
        selection_reason="all frames miss the state",
        uncertainties=["transient may fall between samples"],
        confidence=0.2,
        model_provenance=_provenance(),
    )
    assert selection.visible is False


def test_invisible_dense_selection_rejects_stale_target_identity() -> None:
    with pytest.raises(ValidationError, match="frame or target fields"):
        DenseEventSelection(
            source_asset_id="sha256:" + "1" * 64,
            event_id="event-demo",
            visible=False,
            target_entity_id="subject-primary",
            target_description="selected foreground subject",
            observable_evidence="the selected subject cannot be confirmed",
            selection_reason="all supplied evidence frames are inconclusive",
            uncertainties=["the subject may be outside the sampled window"],
            confidence=0.2,
            model_provenance=_provenance(),
        )


def _dense_catalog_payload() -> dict:
    def frame(index: int, requested_time_ms: int, frame_time_ms: int) -> dict:
        return {
            "frame_id": f"DF{index:06d}",
            "event_id": "event-demo",
            "requested_time_ms": requested_time_ms,
            "frame_time_ms": frame_time_ms,
            "frame_pts": index * 100,
            "frame_hash": f"{index:x}" * 64,
            "width": 640,
            "height": 360,
            "image_path": f"/evidence/source-{index}.jpg",
            "transport_image_path": f"/evidence/transport-{index}.jpg",
            "transport_image_hash": f"{index + 4:x}" * 64,
        }

    return {
        "source_asset_id": "sha256:" + "1" * 64,
        "event_id": "event-demo",
        "sampling_fps": 4,
        "source_start_ms": 1000,
        "source_end_ms": 2000,
        "frames": [frame(1, 1000, 1000), frame(2, 1250, 1267)],
        "contact_sheet_paths": ["/evidence/contact-sheet.jpg"],
        "contact_sheet_hashes": ["f" * 64],
        "generated_at": "now",
    }


def test_dense_catalog_accepts_ordered_event_local_frames() -> None:
    catalog = DenseFrameCatalog.model_validate(_dense_catalog_payload())
    assert [frame.frame_id for frame in catalog.frames] == ["DF000001", "DF000002"]


def test_dense_seed_is_reused_only_for_the_same_locked_target() -> None:
    catalog = DenseFrameCatalog.model_validate(_dense_catalog_payload())
    selection = DenseEventSelection(
        source_asset_id=catalog.source_asset_id,
        event_id=catalog.event_id,
        visible=True,
        first_frame_id="DF000001",
        recommended_frame_id="DF000002",
        last_frame_id="DF000002",
        target_entity_id="subject-a",
        target_description="the selected foreground subject",
        match_status="matched",
        observable_evidence="the selected subject is visible",
        selection_reason="clear boundary and low blur",
        uncertainties=[],
        confidence=0.8,
        model_provenance=_provenance(),
    )
    matched = _matching_dense_seed_frame(
        selection,
        catalog,
        target_entity_id="subject-a",
        target_description="the selected foreground subject",
    )
    assert matched is not None
    assert matched.frame_pts == catalog.frames[1].frame_pts
    assert (
        _matching_dense_seed_frame(
            selection,
            catalog,
            target_entity_id="subject-b",
            target_description="the selected foreground subject",
        )
        is None
    )
    assert (
        _matching_dense_seed_frame(
            selection,
            catalog,
            target_entity_id="subject-a",
            target_description="the selected foreground subject with a changed constraint",
        )
        is None
    )


def test_ambiguous_dense_selection_is_not_a_tracking_seed() -> None:
    catalog = DenseFrameCatalog.model_validate(_dense_catalog_payload())
    selection = DenseEventSelection(
        source_asset_id=catalog.source_asset_id,
        event_id=catalog.event_id,
        visible=True,
        first_frame_id="DF000001",
        recommended_frame_id="DF000002",
        last_frame_id="DF000002",
        target_entity_id="subject-a",
        target_description="one of two similar foreground subjects",
        match_status="ambiguous",
        observable_evidence="two plausible instances are visible",
        selection_reason="identity is not distinguishable",
        uncertainties=["operator selection required"],
        confidence=0.4,
        model_provenance=_provenance(),
    )
    assert (
        _matching_dense_seed_frame(
            selection,
            catalog,
            target_entity_id="subject-a",
            target_description="one of two similar foreground subjects",
        )
        is None
    )


def test_dense_catalog_rejects_mixed_event_frames() -> None:
    payload = _dense_catalog_payload()
    payload["frames"][1]["event_id"] = "event-other"
    with pytest.raises(ValidationError, match="event_id must match"):
        DenseFrameCatalog.model_validate(payload)


def test_dense_catalog_rejects_duplicate_or_out_of_order_ids() -> None:
    duplicate = _dense_catalog_payload()
    duplicate["frames"][1]["frame_id"] = "DF000001"
    with pytest.raises(ValidationError, match="IDs must be unique"):
        DenseFrameCatalog.model_validate(duplicate)

    out_of_order = _dense_catalog_payload()
    out_of_order["frames"].reverse()
    with pytest.raises(ValidationError, match="IDs must be ordered"):
        DenseFrameCatalog.model_validate(out_of_order)


@pytest.mark.parametrize("time_field", ["requested_time_ms", "frame_time_ms"])
def test_dense_catalog_rejects_times_outside_window(time_field: str) -> None:
    payload = _dense_catalog_payload()
    payload["frames"][1][time_field] = 2000
    with pytest.raises(ValidationError, match="inside the source window"):
        DenseFrameCatalog.model_validate(payload)


def test_saved_raw_clip_card_can_be_revalidated_without_another_api_call(
    tmp_path: Path,
) -> None:
    card = _card(model_provenance=_provenance(MODEL_ID))
    prompt = "clip card prompt"
    run_dir = tmp_path / "gemini"
    run_dir.mkdir()
    (run_dir / "clip_card.raw_output.json").write_text(
        json.dumps({"output_text": card.model_dump_json()}),
        encoding="utf-8",
    )
    (run_dir / "clip_card.raw_interaction.json").write_text(
        json.dumps({"id": "interaction-1", "model": MODEL_ID}),
        encoding="utf-8",
    )
    (run_dir / "clip_card.request.json").write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "system_instruction": VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
                "input": [{"type": "text", "text": prompt + "\nmetadata"}],
            }
        ),
        encoding="utf-8",
    )
    recovered = _revalidate_saved_clip_card(
        run_dir,
        card.source_asset_id,
        card.proxy_asset_id,
        card.duration_ms,
        prompt,
    )
    assert recovered is not None
    assert recovered.model_provenance.interaction_id == "interaction-1"
    assert json.loads((run_dir / "clip_card.schema_validation.json").read_text())["ok"]


def test_saved_raw_clip_card_is_not_reused_after_prompt_change(tmp_path: Path) -> None:
    card = _card(model_provenance=_provenance(MODEL_ID))
    run_dir = tmp_path / "gemini"
    run_dir.mkdir()
    (run_dir / "clip_card.raw_output.json").write_text(
        json.dumps({"output_text": card.model_dump_json()}), encoding="utf-8"
    )
    (run_dir / "clip_card.request.json").write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "system_instruction": VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
                "input": [{"type": "text", "text": "old prompt\nmetadata"}],
            }
        ),
        encoding="utf-8",
    )
    recovered = _revalidate_saved_clip_card(
        run_dir,
        card.source_asset_id,
        card.proxy_asset_id,
        card.duration_ms,
        "new prompt",
    )
    assert recovered is None


def test_saved_raw_clip_card_is_not_reused_without_current_system_instruction(
    tmp_path: Path,
) -> None:
    card = _card(model_provenance=_provenance(MODEL_ID))
    prompt = "clip card prompt"
    run_dir = tmp_path / "gemini"
    run_dir.mkdir()
    (run_dir / "clip_card.raw_output.json").write_text(
        json.dumps({"output_text": card.model_dump_json()}), encoding="utf-8"
    )
    (run_dir / "clip_card.request.json").write_text(
        json.dumps(
            {
                "model": MODEL_ID,
                "input": [{"type": "text", "text": prompt + "\nmetadata"}],
            }
        ),
        encoding="utf-8",
    )
    recovered = _revalidate_saved_clip_card(
        run_dir,
        card.source_asset_id,
        card.proxy_asset_id,
        card.duration_ms,
        prompt,
    )
    assert recovered is None


def test_saved_raw_clip_card_from_previous_model_is_not_reused(tmp_path: Path) -> None:
    card = _card(model_provenance=_provenance("gemini-3.5-flash"))
    prompt = "clip card prompt"
    run_dir = tmp_path / "gemini"
    run_dir.mkdir()
    (run_dir / "clip_card.raw_output.json").write_text(
        json.dumps({"output_text": card.model_dump_json()}), encoding="utf-8"
    )
    (run_dir / "clip_card.raw_interaction.json").write_text(
        json.dumps({"id": "old-interaction", "model": "gemini-3.5-flash"}),
        encoding="utf-8",
    )
    (run_dir / "clip_card.request.json").write_text(
        json.dumps(
            {
                "model": "gemini-3.5-flash",
                "system_instruction": VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
                "input": [{"type": "text", "text": prompt + "\nmetadata"}],
            }
        ),
        encoding="utf-8",
    )

    assert (
        _revalidate_saved_clip_card(
            run_dir,
            card.source_asset_id,
            card.proxy_asset_id,
            card.duration_ms,
            prompt,
        )
        is None
    )


def test_full_clip_prompt_evidence_rule_is_domain_neutral() -> None:
    prompt = Path("prompts/full_clip_card_mmss_zh-TW.txt").read_text(encoding="utf-8")
    assert "關鍵字元／音節不清楚" in prompt
    assert "最具體類別泛稱" in prompt
    assert "Reno" not in prompt


def test_full_library_public_index_omits_source_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "private-rushes"
    source_dir.mkdir()
    video = source_dir / "identifying-name.mp4"
    _make_av_video(video, duration=1)

    def fake_run_full_clip(path: Path, output_dir: Path, **kwargs):
        media = probe_video(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "pricing": {"estimated_total_cost_usd": 0.01},
            "execution_pricing": {"estimated_total_cost_usd": 0.01},
            "event_count": 1,
            "shot_count": 1,
        }

    monkeypatch.setattr("jascue_video_lab.full_v1.run_full_clip", fake_run_full_clip)
    output = tmp_path / "library"
    result = run_full_library(
        source_dir,
        output,
        clip_card_prompt="clip",
        dense_prompt="dense",
    )
    public_text = (output / "library-index.json").read_text()
    private_text = (output / "private-library.json").read_text()
    assert result["succeeded"] == 1
    assert "identifying-name" not in public_text
    assert "identifying-name" in private_text


def test_prepare_only_never_constructs_gemini_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    _make_av_video(source, duration=1)

    def forbidden_client(*args, **kwargs):
        raise AssertionError("prepare-only must not construct GeminiLabClient")

    monkeypatch.setattr("jascue_video_lab.full_v1.GeminiLabClient", forbidden_client)
    result = run_full_clip(
        source,
        tmp_path / "prepared",
        clip_card_prompt="unused",
        dense_prompt="unused",
        prepare_only=True,
    )
    assert result["prepare_only"] is True
    assert result["execution_pricing"]["request_count"] == 0
    assert (tmp_path / "prepared" / "analysis-proxy.mp4").exists()
    assert (tmp_path / "prepared" / "shots.json").exists()
    assert not (tmp_path / "prepared" / "gemini").exists()


def test_shared_upload_cache_migrates_only_exact_proxy_hash(tmp_path: Path) -> None:
    proxy_hash = "a" * 64
    legacy = (
        tmp_path
        / "legacy-run"
        / "file-cache"
        / proxy_hash
        / "upload"
    )
    legacy.mkdir(parents=True)
    (legacy / "file_cache.json").write_text(
        json.dumps({"upload_asset_sha256": proxy_hash, "file_name": "files/existing"}),
        encoding="utf-8",
    )
    shared_root = tmp_path / "shared"
    resolved = _shared_upload_dir(
        proxy_hash,
        file_cache_root=shared_root,
        legacy_search_root=tmp_path,
    )
    assert resolved == shared_root / proxy_hash / "upload"
    assert json.loads((resolved / "file_cache.json").read_text())["file_name"] == "files/existing"
    assert json.loads((resolved / "migration.json").read_text())["proxy_sha256"] == proxy_hash


def _rushes_catalog(tmp_path: Path) -> RushesCatalog:
    clips = []
    frames = []
    for index in range(3):
        clip_id = f"clip-{index + 1}"
        source = tmp_path / f"source-{index + 1}.mp4"
        source.touch()
        clips.append(
            {
                "clip_id": clip_id,
                "path": str(source),
                "sha256": str(index + 1) * 64,
                "duration_ms": 1000,
                "width": 1920,
                "height": 1080,
                "frame_rate": "30/1",
                "size_bytes": 1,
            }
        )
        frames.append(
            {
                "frame_id": f"RF{index + 1:06d}",
                "clip_id": clip_id,
                "requested_time_ms": 0,
                "image_path": str(tmp_path / f"frame-{index + 1}.jpg"),
            }
        )
    return RushesCatalog.model_validate(
        {
            "catalog_id": "catalog-test",
            "source_directory": str(tmp_path),
            "sample_interval_ms": 2000,
            "total_duration_ms": 3000,
            "clips": clips,
            "frames": frames,
            "analysis_reel_path": str(tmp_path / "reel.mp4"),
            "generated_at": "now",
        }
    )


def _feature_plan() -> FeatureEditPlan:
    chapter_defaults = {
        "evidence_status": "supported",
        "observed_visual_evidence": "visible product",
        "selection_reason": "clear shot",
        "horizontal_strategy": "original",
        "horizontal_zoom_intent": "none",
        "horizontal_target_description": None,
        "vertical_strategy": "fit_with_background",
        "vertical_target_description": None,
        "quality_risks": [],
        "confidence": 0.9,
    }
    return FeatureEditPlan.model_validate(
        {
            "project_id": "project-test",
            "catalog_id": "catalog-test",
            "title": "test",
            "chapters": [
                {
                    **chapter_defaults,
                    "feature_id": "feature-a",
                    "horizontal_frame_id": "RF000001",
                    "vertical_frame_id": "RF000002",
                },
                {
                    **chapter_defaults,
                    "feature_id": "feature-b",
                    "horizontal_frame_id": "RF000001",
                    "vertical_frame_id": "RF000001",
                },
            ],
            "uncertainties": [],
            "model_provenance": _provenance().model_dump(mode="json"),
        }
    )


def test_selected_clip_ids_are_unique_and_preserve_plan_order(tmp_path: Path) -> None:
    assert selected_clip_ids_from_feature_plan(_rushes_catalog(tmp_path), _feature_plan()) == [
        "clip-1",
        "clip-2",
    ]


def test_selected_full_clips_touch_only_plan_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = _rushes_catalog(tmp_path)
    plan = _feature_plan()
    catalog_path = tmp_path / "catalog.json"
    plan_path = tmp_path / "plan.json"
    catalog_path.write_text(catalog.model_dump_json(), encoding="utf-8")
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    prepared = tmp_path / "prepared"
    for clip in catalog.clips[:2]:
        clip_dir = prepared / "clips" / clip.sha256[:16]
        clip_dir.mkdir(parents=True)
        (clip_dir / "analysis-proxy.mp4").touch()

    def fake_run_full_clip(path: Path, output_dir: Path, **kwargs):
        raise AssertionError("prepare-only selected validation must make no API/run calls")

    monkeypatch.setattr("jascue_video_lab.full_v1.run_full_clip", fake_run_full_clip)
    result = run_selected_full_clips(
        catalog_path=catalog_path,
        plan_path=plan_path,
        prepared_library_dir=prepared,
        output_dir=tmp_path / "selected",
        clip_card_prompt="clip",
        dense_prompt="dense",
        prepare_only=True,
    )

    assert result["selected_clip_count"] == 2
    assert result["failed"] == 0
    assert "source-3.mp4" not in json.dumps(
        json.loads((tmp_path / "selected" / "selected-clip-cards.json").read_text())
    )
