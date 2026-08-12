import io
from pathlib import Path

from fastapi.testclient import TestClient

import montagewright.webapp as web


class _FinishedProcess:
    def __init__(self, command, **_):
        self.command = command
        self.stdout = io.StringIO("")

    def wait(self):
        return 0

    def poll(self):
        return 0


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


def test_web_duration_contract_is_explicit_and_reaches_the_cli(
    tmp_path, monkeypatch
):
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(web.subprocess, "Popen", _FinishedProcess)
    web.RUNS.clear()

    response = TestClient(web.create_app()).post(
        "/api/runs",
        data={
            "source_path": str(rushes),
            "seconds": "30",
            "duration_mode": "preferred",
            "review": "false",
        },
    )

    assert response.status_code == 200
    run = web.RUNS[response.json()["run_id"]]
    at = run.command.index("--duration-mode")
    assert run.command[at + 1] == "preferred"
    page = (Path(__file__).parents[1] / "src/montagewright/web/index.html").read_text()
    assert 'id="duration-mode"' in page
    assert "偏好長度，可自然縮短" in page


def test_new_round_inherits_parent_brief_on_the_server(
    tmp_path, monkeypatch
):
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    roots = tmp_path / "runs"
    monkeypatch.setattr(web, "RUNS_ROOT", roots)
    monkeypatch.setattr(web.subprocess, "Popen", _FinishedProcess)
    web.RUNS.clear()
    parent_root = roots / "parent"
    parent_brief = parent_root / "brief.md"
    parent_brief.parent.mkdir(parents=True)
    parent_brief.write_text("Only the approved foldable; exclude watches.")
    web.RUNS["parent"] = web.Run(
        "parent", parent_root, source=str(rushes),
        command=["render", str(rushes), "--brief", str(parent_brief)],
    )

    response = TestClient(web.create_app()).post(
        "/api/runs", data={
            "source_path": str(rushes), "base_run_id": "parent",
            "inherit_brief": "true", "brief": "", "review": "false",
        },
    )

    assert response.status_code == 200
    child = web.RUNS[response.json()["run_id"]]
    at = child.command.index("--brief")
    child_brief = Path(child.command[at + 1])
    assert child_brief.parent == child.root
    assert child_brief.read_text() == parent_brief.read_text()


def test_web_can_start_from_a_brief_file_path(tmp_path, monkeypatch):
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    brief = tmp_path / "fold8.md"
    brief.write_text("Z Fold8 only")
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(web.subprocess, "Popen", _FinishedProcess)
    web.RUNS.clear()

    response = TestClient(web.create_app()).post(
        "/api/runs", data={
            "source_path": str(rushes), "brief_path": str(brief),
            "brief": "experience event context", "review": "false",
        },
    )

    assert response.status_code == 200
    run = web.RUNS[response.json()["run_id"]]
    at = run.command.index("--brief")
    saved = Path(run.command[at + 1]).read_text()
    assert "Z Fold8 only" in saved
    assert "experience event context" in saved


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


def test_opening_a_cut_is_addressable_and_survives_a_reload():
    """Every cut lived at the same URL, so none of them could be returned to.

    Opening a past run left the address bar at the root: reloading threw the
    cut away, the back button left the application, and a link to one cut
    could not be sent or bookmarked. The page decides what to show from the
    path, so the path has to survive a reload rather than 404.
    """

    client = TestClient(web.create_app())
    page = (
        Path(__file__).parents[1] / "src" / "montagewright" / "web" / "index.html"
    ).read_text(encoding="utf-8")

    served = client.get("/run/231d62b566e7")
    assert served.status_code == 200
    assert served.text == client.get("/").text, (
        "one page; which cut it opens is read from the path"
    )
    assert "history.pushState" in page and "popstate" in page
    assert "function runIdInUrl" in page


def test_opening_a_running_cut_names_it_and_starts_the_clock():
    """A cut still being made showed a frozen, anonymous workspace.

    The header only ever learned a cut's name from its report, so a run
    without one said "nothing is open" over a workspace that plainly had
    something in it. Worse, polling started only in the tab that pressed
    start -- so a cut opened from its own URL, by reload, link or back
    button, showed one snapshot of a working run and kept showing it.
    """

    page = (
        Path(__file__).parents[1] / "src" / "montagewright" / "web" / "index.html"
    ).read_text(encoding="utf-8")

    opening = page.split("async function openRun(")[1].split("\nfunction ")[0]
    assert "crumb-what" in opening, "name the cut before its report exists"
    assert "timer = setInterval(poll" in opening, "and start the clock"
    assert opening.count("clearInterval(timer)") >= 1, (
        "without leaving the previous cut's timer running"
    )
