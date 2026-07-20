from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from jascue_video_lab.full_v1 import (
    _revalidate_saved_clip_card,
    _shared_upload_dir,
    create_dense_event_catalog,
    create_shot_catalog,
    dense_window_for_event,
    dense_sampling_fps,
    derive_clip_timeline,
    mmss_to_ms,
    run_full_clip,
    run_full_library,
)
from jascue_video_lab.media import create_analysis_proxy, probe_video
from jascue_video_lab.models import (
    DenseEventSelection,
    Entity,
    EntityKind,
    FullClipCard,
    FullClipEvent,
    ModelProvenance,
)
from jascue_video_lab.shots import ShotManifest, ShotSegment


def _provenance() -> ModelProvenance:
    return ModelProvenance(
        model_id="gemini-3.5-flash",
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version="test",
        interaction_id=None,
        run_id="test",
        generated_at="now",
    )


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


def test_saved_raw_clip_card_can_be_revalidated_without_another_api_call(
    tmp_path: Path,
) -> None:
    card = _card()
    prompt = "clip card prompt"
    run_dir = tmp_path / "gemini"
    run_dir.mkdir()
    (run_dir / "clip_card.raw_output.json").write_text(
        json.dumps({"output_text": card.model_dump_json()}),
        encoding="utf-8",
    )
    (run_dir / "clip_card.raw_interaction.json").write_text(
        json.dumps({"id": "interaction-1"}),
        encoding="utf-8",
    )
    (run_dir / "clip_card.request.json").write_text(
        json.dumps(
            {
                "model": "gemini-3.5-flash",
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
    card = _card()
    run_dir = tmp_path / "gemini"
    run_dir.mkdir()
    (run_dir / "clip_card.raw_output.json").write_text(
        json.dumps({"output_text": card.model_dump_json()}), encoding="utf-8"
    )
    (run_dir / "clip_card.request.json").write_text(
        json.dumps(
            {
                "model": "gemini-3.5-flash",
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
