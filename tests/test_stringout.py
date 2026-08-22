from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from montagewright.stringout import (
    StringoutError,
    build_stringout,
    load_stringout_manifest,
    require_stringout_matches,
)
from montagewright.measure.media import probe_video


def _video(path: Path, *, colour: str, seconds: float, audio: bool) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color={colour}:s=320x180:r=30:d={seconds}",
    ]
    if audio:
        command += [
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-shortest", "-c:a", "aac",
        ]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_stringout_preserves_every_source_and_clock(tmp_path):
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    _video(first, colour="red", seconds=1.0, audio=True)
    _video(second, colour="blue", seconds=1.5, audio=False)
    sources = [
        SimpleNamespace(source_id="SAM-001", duration_seconds=1.0, proxy=first),
        SimpleNamespace(source_id="SAM-002", duration_seconds=1.5, proxy=second),
    ]

    output = tmp_path / "all-material.mp4"
    manifest = build_stringout(sources, output, width=640, height=360)

    assert output.exists() and output.stat().st_size > 0
    assert [one.source_id for one in manifest.entries] == ["SAM-001", "SAM-002"]
    assert manifest.entries[0].reel_in_seconds == pytest.approx(1.0)
    assert manifest.entries[1].reel_in_seconds == pytest.approx(3.0)
    loaded = load_stringout_manifest(output.with_suffix(".json"))
    require_stringout_matches(loaded, sources)
    payload = json.loads(output.with_suffix(".json").read_text())
    assert payload["duration_seconds"] == pytest.approx(4.5)
    assert probe_video(output).duration_ms == pytest.approx(4500, abs=40)


def test_stringout_audit_rejects_stale_or_missing_sources(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"changed")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "version": "montagewright-stringout-v1",
        "video_path": str(tmp_path / "reel.mp4"),
        "width": 1280, "height": 720, "fps": 30, "slate_seconds": 0.75,
        "entries": [{
            "source_id": "C1", "source_path": str(source),
            "source_sha256": "not-the-file", "source_in_seconds": 0.0,
            "source_out_seconds": 1.0, "reel_in_seconds": 0.75,
            "reel_out_seconds": 1.75,
        }],
    }), encoding="utf-8")
    manifest = load_stringout_manifest(manifest_path)
    with pytest.raises(StringoutError, match="stale"):
        require_stringout_matches(manifest, [
            SimpleNamespace(source_id="C1", duration_seconds=1.0, proxy=source)
        ])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg missing")
def test_logged_span_ranges_make_a_short_reel_with_original_source_clock(tmp_path):
    source = tmp_path / "interview.mp4"
    _video(source, colour="green", seconds=4.0, audio=True)
    selected = SimpleNamespace(
        source_id="INT", duration_seconds=4.0, proxy=source,
        planning_ranges=((0.5, 1.5), (2.5, 3.5)),
    )

    output = tmp_path / "selects.mp4"
    manifest = build_stringout([selected], output, width=640, height=360)

    assert [(one.source_id, one.source_in_seconds, one.source_out_seconds)
            for one in manifest.entries] == [
        ("INT", 0.5, 1.5), ("INT", 2.5, 3.5),
    ]
    require_stringout_matches(manifest, [selected])
    assert probe_video(output).duration_ms == pytest.approx(4000, abs=60)
