from pathlib import Path
from types import SimpleNamespace

from jascue_video_lab.gemini import GeminiLabClient
from jascue_video_lab.storage import write_json


def _client_without_network() -> GeminiLabClient:
    return object.__new__(GeminiLabClient)


def test_ensure_upload_reuses_active_saved_file(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    write_json(upload_dir / "file_upload_initial.json", {"name": "files/active"})
    client = _client_without_network()
    uploaded = SimpleNamespace(name="files/active")
    calls = {"upload": 0}
    client.resume_video_upload = lambda *_args, **_kwargs: uploaded  # type: ignore[method-assign]

    def unexpected_upload(*_args, **_kwargs):
        calls["upload"] += 1
        raise AssertionError("ACTIVE file must not be uploaded again")

    client.upload_video = unexpected_upload  # type: ignore[method-assign]
    result, reused = client.ensure_video_upload(tmp_path / "video.mp4", upload_dir)
    assert result is uploaded
    assert reused is True
    assert calls["upload"] == 0


def test_ensure_upload_reuploads_only_after_confirmed_404(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    write_json(upload_dir / "file_upload_initial.json", {"name": "files/expired"})
    client = _client_without_network()
    uploaded = SimpleNamespace(name="files/new")

    class NotFoundError(RuntimeError):
        code = 404

    def expired(*_args, **_kwargs):
        raise NotFoundError("file expired")

    client.resume_video_upload = expired  # type: ignore[method-assign]
    client.upload_video = lambda *_args, **_kwargs: uploaded  # type: ignore[method-assign]
    result, reused = client.ensure_video_upload(tmp_path / "video.mp4", upload_dir)
    assert result is uploaded
    assert reused is False
    assert list((upload_dir / "history").glob("*/file_upload_initial.json"))


def test_ensure_upload_does_not_duplicate_on_transient_error(tmp_path: Path) -> None:
    upload_dir = tmp_path / "upload"
    write_json(upload_dir / "file_upload_initial.json", {"name": "files/unknown"})
    client = _client_without_network()

    def transient(*_args, **_kwargs):
        raise RuntimeError("temporary network failure")

    client.resume_video_upload = transient  # type: ignore[method-assign]
    client.upload_video = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("must not upload after an ambiguous error")
    )

    try:
        client.ensure_video_upload(tmp_path / "video.mp4", upload_dir)
    except RuntimeError as error:
        assert "temporary network failure" in str(error)
    else:
        raise AssertionError("transient error should be preserved")
