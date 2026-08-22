from montagewright.ingest import (
    AudioStream, IngestAsset, IngestManifest, VideoFacts,
)
from montagewright.sync import SyncMap, resolve_sync
import pytest


def _manifest(tmp_path):
    video = tmp_path / "A001.mov"
    audio = tmp_path / "boom.wav"
    video.touch(); audio.touch()
    picture = IngestAsset(
        source_id="A001", asset_id="sha256:" + "a" * 64,
        relative_path="A001.mov", absolute_path=str(video), kind="video",
        sha256="a" * 64, size_bytes=1, mtime_ns=1, duration_seconds=20,
        video=VideoFacts(
            width=1920, height=1080, average_frame_rate="25/1",
            real_frame_rate="25/1",
        ), audio_streams=(AudioStream(index=1, channels=2),),
    )
    sound = IngestAsset(
        source_id="boom", asset_id="sha256:" + "b" * 64,
        relative_path="boom.wav", absolute_path=str(audio), kind="audio",
        sha256="b" * 64, size_bytes=1, mtime_ns=1, duration_seconds=30,
        audio_streams=(AudioStream(index=0, channels=1),),
    )
    return IngestManifest(
        root=str(tmp_path), assets=(picture, sound),
        inventory_sha256="c" * 64, estimated_work_bytes=1, free_bytes=2,
    )


def test_sync_map_resolves_relative_paths_to_stable_ingest_ids(tmp_path):
    sync = SyncMap.model_validate({
        "authority": "manual",
        "groups": [{"group_id": "interview", "members": [
            {"source": "A001.mov", "role": "picture_angle", "offset_seconds": 0},
            {"source": "boom.wav", "role": "master_audio", "offset_seconds": 1.25},
        ]}],
    })

    resolved = resolve_sync(sync, _manifest(tmp_path))

    assert resolved.by_source()["boom"].offset_seconds == 1.25
    assert resolved.by_source()["A001"].role == "picture_angle"


@pytest.mark.parametrize(
    "audio_stream_index,channel,expected",
    [(99, None, "no audio stream 99"), (0, 4, "not channel 4")],
)
def test_sync_rejects_nonexistent_stream_or_channel(
    tmp_path, audio_stream_index, channel, expected,
):
    sync = SyncMap.model_validate({
        "authority": "manual",
        "groups": [{"group_id": "interview", "members": [
            {"source": "A001.mov", "role": "picture_angle"},
            {
                "source": "boom.wav", "role": "master_audio",
                "audio_stream_index": audio_stream_index, "channel": channel,
            },
        ]}],
    })
    with pytest.raises(ValueError, match=expected):
        resolve_sync(sync, _manifest(tmp_path))


def test_sync_rejects_master_audio_without_picture_angle():
    with pytest.raises(ValueError, match="master audio but no picture angle"):
        SyncMap.model_validate({
            "authority": "manual",
            "groups": [{"group_id": "interview", "members": [
                {"source": "boom.wav", "role": "master_audio"},
                {"source": "scratch.wav", "role": "scratch_audio"},
            ]}],
        })


def test_sync_requires_stream_for_multistream_master(tmp_path):
    manifest = _manifest(tmp_path)
    sound = manifest.assets[1].model_copy(update={
        "audio_streams": (
            AudioStream(index=0, channels=1),
            AudioStream(index=2, channels=2),
        ),
    })
    manifest = manifest.model_copy(update={"assets": (manifest.assets[0], sound)})
    sync = SyncMap.model_validate({
        "authority": "manual",
        "groups": [{"group_id": "interview", "members": [
            {"source": "A001.mov", "role": "picture_angle"},
            {"source": "boom.wav", "role": "master_audio"},
        ]}],
    })

    with pytest.raises(ValueError, match="must name audio_stream_index"):
        resolve_sync(sync, manifest)
