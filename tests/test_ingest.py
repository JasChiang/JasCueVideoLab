from __future__ import annotations

import json
from pathlib import Path

import pytest

from montagewright.ingest import (
    IngestError, build_manifest, decode_preflight, technical_contract_faults,
    write_manifest,
)


def _probe_payload(*, audio: bool = True) -> dict:
    streams = [{
        "index": 0, "codec_type": "video", "codec_name": "h264",
        "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001",
        "r_frame_rate": "30000/1001", "field_order": "progressive",
        "pix_fmt": "yuv420p", "time_base": "1/30000",
    }]
    if audio:
        streams.append({
            "index": 1, "codec_type": "audio", "codec_name": "aac",
            "channels": 2, "channel_layout": "stereo", "sample_rate": "48000",
        })
    return {"streams": streams, "format": {"duration": "12.5"}}


def test_ingest_recurses_and_disambiguates_same_camera_names(tmp_path, monkeypatch):
    root = tmp_path / "rushes"
    first = root / "card-a" / "C0001.MOV"
    second = root / "card-b" / "C0001.MOV"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    def fake_probe(path: Path):
        import hashlib
        return _probe_payload(), hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr("montagewright.ingest._probe", fake_probe)
    manifest = build_manifest(root, work_root=tmp_path / "out" / "work")

    assert len(manifest.videos) == 2
    assert len({one.source_id for one in manifest.videos}) == 2
    assert all(one.source_id.startswith("C0001__") for one in manifest.videos)
    assert {one.relative_path for one in manifest.videos} == {
        "card-a/C0001.MOV", "card-b/C0001.MOV",
    }


def test_ingest_refuses_to_resume_when_master_bytes_changed(tmp_path, monkeypatch):
    root = tmp_path / "rushes"
    root.mkdir()
    source = root / "take.mp4"
    source.write_bytes(b"old")

    def fake_probe(path: Path):
        import hashlib
        return _probe_payload(), hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr("montagewright.ingest._probe", fake_probe)
    path = tmp_path / "out" / "work" / "ingest-manifest.json"
    write_manifest(path, build_manifest(root, work_root=path.parent))
    source.write_bytes(b"new")

    with pytest.raises(IngestError, match="start a new revision"):
        build_manifest(root, work_root=path.parent, previous=path)


def test_ingest_records_external_audio_and_technical_warnings(tmp_path, monkeypatch):
    root = tmp_path / "rushes"
    root.mkdir()
    video = root / "phone.mkv"
    wav = root / "boom.wav"
    video.write_bytes(b"video")
    wav.write_bytes(b"audio")

    def fake_probe(path: Path):
        import hashlib
        payload = _probe_payload()
        if path.suffix == ".wav":
            payload = {
                "streams": [{
                    "index": 0, "codec_type": "audio", "codec_name": "pcm_s24le",
                    "channels": 1, "channel_layout": "mono", "sample_rate": "48000",
                }],
                "format": {"duration": "20"},
            }
        else:
            payload["streams"][0]["avg_frame_rate"] = "2997/100"
            payload["streams"][0]["r_frame_rate"] = "30/1"
            payload["streams"][0]["color_transfer"] = "smpte2084"
        return payload, hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr("montagewright.ingest._probe", fake_probe)
    manifest = build_manifest(root, work_root=tmp_path / "out" / "work")

    assert len(manifest.external_audio) == 1
    assert manifest.external_audio[0].audio_streams[0].codec == "pcm_s24le"
    assert "variable_frame_rate" in manifest.videos[0].warnings
    assert "hdr:smpte2084" in manifest.videos[0].warnings

    faults = technical_contract_faults(manifest)
    assert any("variable-frame-rate" in one for one in faults)
    assert any("HDR-to-Rec.709" in one for one in faults)


def test_decode_preflight_checks_head_and_tail(tmp_path, monkeypatch):
    root = tmp_path / "rushes"
    root.mkdir()
    source = root / "take.mp4"
    source.write_bytes(b"video")

    def fake_probe(path: Path):
        import hashlib
        return _probe_payload(), hashlib.sha256(path.read_bytes()).hexdigest()

    calls = []
    monkeypatch.setattr("montagewright.ingest._probe", fake_probe)

    class Completed:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(
        "montagewright.ingest.subprocess.run",
        lambda command, **kwargs: calls.append(command) or Completed(),
    )
    manifest = build_manifest(root, work_root=tmp_path / "out" / "work")
    decode_preflight(manifest)

    assert len(calls) == 3
    assert calls[0][calls[0].index("-ss") + 1] == "0.000000"
    assert calls[1][calls[1].index("-ss") + 1] == "11.750000"
    assert "-err_detect" in calls[2]
