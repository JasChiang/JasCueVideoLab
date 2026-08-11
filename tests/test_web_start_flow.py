from pathlib import Path

from fastapi.testclient import TestClient

import montagewright.webapp as web


def test_new_run_click_replaces_completed_progress_and_recovers_from_failure():
    page = (
        Path(__file__).parents[1] / "src" / "montagewright" / "web" / "index.html"
    ).read_text(encoding="utf-8")

    assert "$('state').textContent = '準備中'" in page
    assert "$('steps').innerHTML = ''" in page
    assert "$('progress').scrollIntoView" in page
    assert "catch (error)" in page
    assert page.count("$('state').textContent = '啟動失敗'") >= 2
    assert page.count("$('go').disabled = false") >= 3


def test_process_launch_failure_is_visible_and_records_a_failed_run(
    tmp_path, monkeypatch
):
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    runs = tmp_path / "runs"
    monkeypatch.setattr(web, "RUNS_ROOT", runs)
    web.RUNS.clear()

    def cannot_launch(*args, **kwargs):
        raise OSError(35, "resource temporarily unavailable")

    monkeypatch.setattr(web.subprocess, "Popen", cannot_launch)
    client = TestClient(web.create_app(), raise_server_exceptions=False)
    response = client.post(
        "/api/runs",
        data={
            "source_path": str(rushes),
            "aspect": "9:16",
            "budget": "6",
            "review": "false",
        },
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error_code"] == "process_launch_failed"
    assert "resource temporarily unavailable" in detail["message"]
    run = web.RUNS[detail["run_id"]]
    assert run.state == "failed"
    assert run.returncode == -1
    saved = (run.root / "run.json").read_text(encoding="utf-8")
    assert '"state": "failed"' in saved


def test_writable_run_root_can_also_discover_read_only_legacy_runs(
    tmp_path, monkeypatch
):
    current = tmp_path / "current"
    legacy = tmp_path / "legacy"
    old = legacy / "old-cut"
    old.mkdir(parents=True)
    (old / "run.json").write_text(
        '{"run_id":"old-cut","state":"done","source":"/old/rushes"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(web, "RUNS_ROOT", current)
    monkeypatch.setattr(web, "LEGACY_RUNS_ROOTS", (legacy,))
    web.RUNS.clear()

    web.recall()

    assert web.RUNS["old-cut"].root == old
    assert web.RUNS["old-cut"].source == "/old/rushes"
