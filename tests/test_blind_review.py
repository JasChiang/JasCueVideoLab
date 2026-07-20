from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from jascue_video_lab.blind_review import BlindReviewService, ReviewVerdict
from jascue_video_lab.models import (
    DirectMoment,
    DirectMomentMap,
    DirectVideoGroundingProposal,
    EntityKind,
    GroundingCandidate,
    GroundingProposal,
    ModelProvenance,
    Occlusion,
    TargetCandidate,
    TargetCandidateMap,
)
from jascue_video_lab.storage import write_json
from jascue_video_lab.webapp import create_app


def _provenance(run_id: str) -> ModelProvenance:
    return ModelProvenance(
        model_id="gemini-3.5-flash",
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version="2.3.0",
        interaction_id="fake-interaction",
        run_id=run_id,
        generated_at="2026-07-20T00:00:00+00:00",
    )


class FakeGeminiClient:
    def __init__(self, *, temperature: float = 0.2) -> None:
        self.temperature = temperature

    def close(self) -> None:
        pass

    def ensure_video_upload(self, path: Path, artifact_dir: Path):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        write_json(artifact_dir / "file_upload_initial.json", {"name": "files/fake"})
        return SimpleNamespace(uri="file://fake", mime_type="video/mp4"), True

    def suggest_targets(self, *, media, run_id, run_dir, **_kwargs):
        result = TargetCandidateMap(
            asset_id=media.asset_id,
            duration_ms=media.duration_ms,
            summary="One visible phone",
            candidates=[
                TargetCandidate(
                    candidate_id="phone-blue-center",
                    label="中央藍色手機",
                    entity_kind=EntityKind.PHONE,
                    target_description="中央藍色實體手機；排除背景圖案。",
                    distinguishing_features="藍色、中央",
                    representative_timestamp_mmss="00:00",
                    selection_reason="清楚可見",
                    confidence=0.9,
                )
            ],
            uncertainties=[],
            model_provenance=_provenance(run_id),
        )
        write_json(run_dir / "target_candidates.json", result)
        return result

    def analyze_direct_moments(
        self, *, media, run_id, run_dir, locked_target_id, locked_target_description, **_kwargs
    ):
        result = DirectMomentMap(
            asset_id=media.asset_id,
            duration_ms=media.duration_ms,
            summary="Phone moment",
            moments=[
                DirectMoment(
                    moment_id="moment-01",
                    timestamp_mmss="00:01",
                    label="手機清楚可見",
                    observable_evidence="藍色手機位於中央。",
                    grounding_target_id=locked_target_id,
                    grounding_target_description=locked_target_description,
                    confidence=0.9,
                )
            ],
            uncertainties=[],
            model_provenance=_provenance(run_id),
        )
        write_json(run_dir / "direct_moments.json", result)
        return result

    def ground_frame(self, *, media, frame, event_id, entity_id, run_id, output_dir, **_kwargs):
        result = GroundingProposal(
            asset_id=media.asset_id,
            event_id=event_id,
            entity_id=entity_id,
            frame_pts=frame.frame_pts,
            frame_time_ms=frame.frame_time_ms,
            frame_hash=frame.frame_hash,
            source_width=frame.width,
            source_height=frame.height,
            visible=True,
            occlusion=Occlusion.NONE,
            visibility_reason="target visible",
            candidates=[
                GroundingCandidate(
                    box_2d=(250, 200, 750, 850),
                    label="blue phone",
                    confidence=0.94,
                    disambiguation_reason="central blue object",
                )
            ],
            model_provenance=_provenance(run_id),
        )
        write_json(output_dir / "grounding.json", result)
        return result

    def ground_video_at_moment(
        self,
        *,
        media,
        requested_timestamp_mmss,
        event_id,
        entity_id,
        run_id,
        output_dir,
        **_kwargs,
    ):
        result = DirectVideoGroundingProposal(
            asset_id=media.asset_id,
            event_id=event_id,
            entity_id=entity_id,
            requested_timestamp_mmss=requested_timestamp_mmss,
            reference_frame_status="unknown_gemini_video_sample",
            reference_frame_description="A sampled frame near the requested second.",
            visible=True,
            occlusion=Occlusion.NONE,
            visibility_reason="target visible in sampled video frame",
            candidates=[
                GroundingCandidate(
                    box_2d=(260, 210, 740, 840),
                    label="blue phone",
                    confidence=0.88,
                    disambiguation_reason="central blue object",
                )
            ],
            model_provenance=_provenance(run_id),
        )
        write_json(output_dir / "direct_video_grounding.json", result)
        return result


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    video = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x180:r=10:d=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    return video


def _service(tmp_path: Path) -> BlindReviewService:
    return BlindReviewService(
        data_root=tmp_path / "sessions",
        file_cache_root=tmp_path / "file-cache",
        client_factory=FakeGeminiClient,
    )


def test_blind_review_requires_annotation_before_reveal(
    tmp_path: Path, sample_video: Path
) -> None:
    service = _service(tmp_path)
    session = service.create_session_from_file(
        sample_video,
        original_filename="sample.mp4",
        use_analysis_proxy=False,
    )
    candidates = service.suggest_targets(session["session_id"])
    assert candidates["file_api_object_reused"] is True
    service.select_target(session["session_id"], candidate_id="phone-blue-center")
    service.analyze_moments(session["session_id"])
    review = service.ground_moment(session["session_id"], moment_id="moment-01")

    with pytest.raises(PermissionError):
        service.reveal_review(session["session_id"], review["review_id"])

    reveal = service.submit_review(
        session["session_id"],
        review["review_id"],
        verdict=ReviewVerdict.CORRECT,
        reviewer_name="independent-reviewer",
        corrected_box_2d=(260, 210, 740, 840),
    )
    assert reveal["annotation"]["model_details_revealed_before_annotation"] is False
    assert reveal["proposal"]["candidates"][0]["confidence"] == 0.94
    direct_review = service.ground_moment(
        session["session_id"], moment_id="moment-01", mode="direct_video"
    )
    assert direct_review["bbox_reference_frame"] == "unknown_gemini_video_sample"
    direct_reveal = service.submit_review(
        session["session_id"],
        direct_review["review_id"],
        verdict=ReviewVerdict.SAMPLE_FRAME_MISMATCH,
        notes="diagnostic A/B only",
    )
    assert direct_reveal["method_comparison"]["comparable"] is True
    assert 0 < direct_reveal["method_comparison"]["bbox_iou"] <= 1
    exported = service.export_session(session["session_id"])
    assert len(exported["human_annotations"]) == 2
    assert exported["pending_review_ids"] == []


def test_http_app_runs_complete_blind_review_workflow(
    tmp_path: Path, sample_video: Path
) -> None:
    service = _service(tmp_path)
    client = TestClient(create_app(service))
    assert client.get("/api/health").json()["local_only"] is True
    assert "先選對物件" in client.get("/").text

    with sample_video.open("rb") as handle:
        created = client.post(
            "/api/sessions",
            files={"file": ("sample.mp4", handle, "video/mp4")},
            data={"use_analysis_proxy": "false"},
        )
    assert created.status_code == 200
    session_id = created.json()["session_id"]
    assert client.post(f"/api/sessions/{session_id}/candidates", json={"runs": 1}).status_code == 200
    assert client.post(
        f"/api/sessions/{session_id}/target",
        json={"candidate_id": "phone-blue-center"},
    ).status_code == 200
    assert client.post(f"/api/sessions/{session_id}/moments", json={"runs": 1}).status_code == 200
    grounded = client.post(
        f"/api/sessions/{session_id}/ground", json={"moment_id": "moment-01"}
    )
    assert grounded.status_code == 200
    assert "proposal" not in grounded.json()
    assert "confidence" not in grounded.text
    review_id = grounded.json()["review_id"]
    assert client.get(
        f"/api/sessions/{session_id}/reviews/{review_id}/reveal"
    ).status_code == 403
    assert client.get(
        f"/api/sessions/{session_id}/reviews/{review_id}/blind-image"
    ).headers["content-type"] == "image/png"

    submitted = client.post(
        f"/api/sessions/{session_id}/reviews/{review_id}",
        json={"verdict": "correct", "notes": "independent visual check"},
    )
    assert submitted.status_code == 200
    assert submitted.json()["annotation"]["reviewer_type"] == "human"
    assert client.get(
        f"/api/sessions/{session_id}/reviews/{review_id}/revealed-image"
    ).status_code == 200
    exported = client.get(f"/api/sessions/{session_id}/export")
    assert exported.status_code == 200
    assert "attachment" in exported.headers["content-disposition"]
