from __future__ import annotations

from pathlib import Path

from jascue_video_lab.timeline import render_timeline


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
