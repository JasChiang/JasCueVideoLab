from __future__ import annotations

from pathlib import Path

from jascue_video_lab.models import DirectMomentMap
from jascue_video_lab.timeline import render_direct_moment_timeline, render_timeline


def test_every_event_is_clickable(content_map, tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    output = render_timeline(
        content_map=content_map,
        video_path=video,
        proposals=[],
        output_path=tmp_path / "index.html",
    )
    html = output.read_text(encoding="utf-8")
    assert 'onclick="seekTo(0)"' in html
    assert "loadedmetadata" in html
    assert "coarse semantic time" in html
    assert "尚無 Grounding 結果" in html


def test_video_symlink_remains_artifact_local(content_map, tmp_path: Path) -> None:
    source = tmp_path / "source-original.mp4"
    source.write_bytes(b"placeholder")
    artifact = tmp_path / "artifact"
    run_dir = artifact / "run-01"
    run_dir.mkdir(parents=True)
    link = artifact / "source.mp4"
    link.symlink_to(source)
    output = render_timeline(
        content_map=content_map,
        video_path=link,
        proposals=[],
        output_path=run_dir / "index.html",
    )
    html = output.read_text(encoding="utf-8")
    assert 'src="../source.mp4"' in html
    assert "source-original.mp4" not in html


def test_direct_timeline_allows_intentionally_ungrounded_moments(content_map, tmp_path: Path) -> None:
    moment_map = DirectMomentMap.model_validate(
        {
            "asset_id": content_map.asset_id,
            "duration_ms": 10_000,
            "summary": "target-first moments",
            "moments": [
                {
                    "moment_id": "m1",
                    "timestamp_mmss": "00:02",
                    "label": "purple phone",
                    "observable_evidence": "visible",
                    "grounding_target_id": "purple-phone",
                    "grounding_target_description": "the center purple phone",
                    "confidence": 0.9,
                }
            ],
            "uncertainties": [],
            "model_provenance": content_map.model_provenance.model_dump(mode="json"),
        }
    )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    output = render_direct_moment_timeline(
        moment_map=moment_map,
        video_path=video,
        results=[],
        output_path=tmp_path / "direct.html",
    )
    html = output.read_text(encoding="utf-8")
    assert 'onclick="seekTo(2000)"' in html
    assert "未送出單幀 Grounding" in html
