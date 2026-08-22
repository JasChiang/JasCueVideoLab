import io
import hashlib
import json
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


def test_web_children_get_a_writable_shared_cache_home(tmp_path, monkeypatch):
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    runs = tmp_path / "runs"
    launched = []

    class CapturedProcess(_FinishedProcess):
        def __init__(self, command, **kwargs):
            launched.append(kwargs["env"])
            super().__init__(command, **kwargs)

    monkeypatch.setattr(web, "RUNS_ROOT", runs)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(web.subprocess, "Popen", CapturedProcess)
    web.RUNS.clear()

    response = TestClient(web.create_app()).post(
        "/api/runs",
        data={"source_path": str(rushes), "review": "false"},
    )

    assert response.status_code == 200
    assert launched[0]["XDG_CACHE_HOME"] == str((tmp_path / "cache").resolve())
    assert launched[0]["MONTAGEWRIGHT_LIBRARY"] == str(
        (tmp_path / "cache" / "montagewright" / "library").resolve()
    )
    assert (tmp_path / "cache").is_dir()
    assert Path(launched[0]["MONTAGEWRIGHT_LIBRARY"]).is_dir()

    run_id = response.json()["run_id"]
    resumed = TestClient(web.create_app()).post(f"/api/runs/{run_id}/resume")
    assert resumed.status_code == 200
    assert launched[1]["XDG_CACHE_HOME"] == launched[0]["XDG_CACHE_HOME"]


def test_web_children_preserve_an_explicit_cache_home(tmp_path, monkeypatch):
    configured = tmp_path / "configured-cache"
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setenv("XDG_CACHE_HOME", str(configured))

    environment = web._child_environment()

    assert environment["XDG_CACHE_HOME"] == str(configured)
    assert environment["MONTAGEWRIGHT_LIBRARY"] == str(
        configured / "montagewright" / "library"
    )
    assert environment["PYTHONUNBUFFERED"] == "1"


def test_web_image_picker_lists_product_reference_images(tmp_path):
    (tmp_path / "fold8.jpg").write_bytes(b"jpg")
    (tmp_path / "brief.md").write_text("brief", encoding="utf-8")

    response = TestClient(web.create_app()).get(
        "/api/browse", params={"path": str(tmp_path), "kind": "image"},
    )

    assert response.status_code == 200
    assert [one["name"] for one in response.json()["videos"]] == ["fold8.jpg"]


def test_web_can_load_a_saved_job_and_resolve_its_relative_paths(tmp_path):
    from montagewright.job import EditJob, Subject, write_job

    rushes = tmp_path / "rushes"
    rushes.mkdir()
    reference = tmp_path / "flip.png"
    reference.touch()
    job_path = write_job(tmp_path / "edit-job.json", EditJob(
        rushes="rushes",
        output="out",
        subject=Subject(
            description="the specified foldable phone",
            references=("flip.png",),
            presence="target_only",
        ),
    ))

    response = TestClient(web.create_app()).get(
        "/api/jobs/inspect", params={"path": str(job_path)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["rushes"] == str(rushes)
    assert payload["output"] == str(tmp_path / "out")
    assert payload["subject"]["references"] == [str(reference)]
    assert payload["subject"]["presence"] == "target_only"


def test_default_material_library_follows_xdg_cache_home(tmp_path, monkeypatch):
    from montagewright.uploads import default_library

    monkeypatch.delenv("MONTAGEWRIGHT_LIBRARY", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert default_library() == tmp_path / "xdg" / "montagewright" / "library"


def test_web_cache_migration_preserves_paid_material_evidence(tmp_path):
    legacy_root = tmp_path / "legacy"
    old = legacy_root / "library"
    new_cache = tmp_path / "cache"
    new_library = new_cache / "montagewright" / "library"
    (old / "cards").mkdir(parents=True)
    (old / "cards" / "paid.json").write_text("{}", encoding="utf-8")
    new_library.mkdir(parents=True)

    web._seed_writable_web_cache(
        new_cache, new_library, legacy_root=legacy_root
    )

    assert (new_library / "cards" / "paid.json").is_file()


def test_web_refuses_to_launch_before_spend_when_cache_is_not_writable(
    tmp_path, monkeypatch
):
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    web.RUNS.clear()
    launched = []
    monkeypatch.setattr(
        web.subprocess, "Popen", lambda *args, **kwargs: launched.append(args)
    )

    def denied(path, *, purpose):
        raise PermissionError(f"{purpose} denied at {path}")

    monkeypatch.setattr(web, "_prove_writable_directory", denied)
    response = TestClient(
        web.create_app(), raise_server_exceptions=False
    ).post(
        "/api/runs",
        data={"source_path": str(rushes), "review": "false"},
    )

    assert response.status_code == 503
    assert launched == []
    assert "cache" in response.json()["detail"]["message"].lower()


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
    from montagewright.job import load_job

    job = load_job(Path(response.json()["job"]))
    assert job.delivery.seconds == 30
    assert job.delivery.duration_mode == "preferred"
    assert run.command[-2:] == ["--job", response.json()["job"]]
    page = (Path(__file__).parents[1] / "src/montagewright/web/index.html").read_text()
    assert 'id="duration-mode"' in page
    # The wording is the interface's to choose; what this test defends is that
    # both modes are offered and named for what they do, so the difference is
    # a decision somebody makes rather than a flag they inherit.
    assert 'value="preferred"' in page and 'value="exact"' in page
    assert 'value="range"' in page
    assert "不會用空停留硬補滿" in page


def test_web_builds_campaign_variants_without_handwritten_job(tmp_path, monkeypatch):
    from montagewright.job import load_job

    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(web.subprocess, "Popen", _FinishedProcess)
    web.RUNS.clear()

    response = TestClient(web.create_app()).post("/api/runs", data={
        "source_path": str(rushes), "review": "false",
        "aspect": "9:16", "seconds": "15", "duration_mode": "exact",
        "delivery_variants_json": json.dumps([{
            "variant_id": "wide-30",
            "delivery": {"aspect": "16:9", "seconds": 30, "duration_mode": "exact"},
        }]),
    })

    assert response.status_code == 200
    job = load_job(Path(response.json()["job"]))
    assert [(one.variant_id, one.delivery.aspect, one.delivery.seconds) for one in job.variants] == [
        ("primary", "9:16", 15), ("wide-30", "16:9", 30),
    ]


def test_starting_a_loaded_job_preserves_advanced_contracts(tmp_path, monkeypatch):
    from montagewright.job import EditJob, TimelineObligation, write_job, load_job

    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    saved = write_job(tmp_path / "advanced.json", EditJob.model_validate({
        "rushes": str(rushes),
        "dialogue": {"edit_mode": "phrase_edit", "remove_fillers": True},
        "music_policy": {"allowed_ranges": [
            {"start_seconds": 4, "end_seconds": 20},
        ]},
        "obligations": [{
            "obligation_id": "cta", "kind": "minimum_read",
            "track": "graphic", "refs": ["cta.copy"], "minimum_seconds": 2,
        }, {
            "obligation_id": "web.forbidden.old", "kind": "forbidden_presence",
            "track": "picture", "refs": ["device.old"],
        }],
        "rights": {"allowed_platforms": ["YouTube"]},
        "picture_composition": {"mode": "none"},
        "variants": [{
            "variant_id": "vertical", "delivery": {"aspect": "9:16"},
        }],
    }))
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(web.subprocess, "Popen", _FinishedProcess)
    web.RUNS.clear()

    response = TestClient(web.create_app()).post("/api/runs", data={
        "source_path": str(rushes), "loaded_job_path": str(saved),
        "review": "false",
        "picture_obligations_json": json.dumps([{
            "obligation_id": "web.forbidden.1",
            "kind": "forbidden_presence", "track": "picture",
            "refs": ["device.fold"],
        }]),
    })

    assert response.status_code == 200
    actual = load_job(Path(response.json()["job"]))
    assert actual.dialogue.edit_mode == "phrase_edit"
    assert actual.dialogue.remove_fillers is True
    assert actual.music_policy.allowed_ranges[0].start_seconds == 4
    assert [one.obligation_id for one in actual.obligations] == [
        "cta", "web.forbidden.1",
    ]
    assert actual.rights.allowed_platforms == ("YouTube",)
    assert actual.variants[0].variant_id == "vertical"

    cleared = TestClient(web.create_app()).post("/api/runs", data={
        "source_path": str(rushes), "loaded_job_path": str(saved),
        "review": "false", "picture_obligations_json": "[]",
    })
    assert cleared.status_code == 200
    cleared_job = load_job(Path(cleared.json()["job"]))
    assert [one.obligation_id for one in cleared_job.obligations] == ["cta"]


def test_web_builds_multi_sku_grounding_without_handwritten_json(
    tmp_path, monkeypatch,
):
    from montagewright.job import load_job
    from montagewright.reference_grounding import load_grounding_spec

    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    flip = tmp_path / "flip.jpg"
    fold = tmp_path / "fold.jpg"
    flip.write_bytes(b"flip")
    fold.write_bytes(b"fold")
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(web.subprocess, "Popen", _FinishedProcess)
    web.RUNS.clear()

    response = TestClient(web.create_app()).post("/api/runs", data={
        "source_path": str(rushes), "review": "false",
        "grounding_target_id": "sku.flip8",
        "grounding_target_description": "Z Flip8",
        "grounding_identity_semantics": "sku",
        "reference_image_paths": str(flip),
        "grounding_additional_targets_json": json.dumps([{
            "target_id": "sku.fold8", "description": "Fold8",
            "identity_semantics": "sku", "references": [str(fold)],
        }]),
    })

    assert response.status_code == 200
    job = load_job(Path(response.json()["job"]))
    assert job.subject is not None and job.subject.grounding_spec
    spec = load_grounding_spec(Path(job.subject.grounding_spec))
    assert [one.target_id for one in spec.identity_lock.identity.targets] == [
        "sku.flip8", "sku.fold8",
    ]


def test_web_can_author_group_and_timed_forbidden_picture_rules(
    tmp_path, monkeypatch,
):
    from montagewright.job import load_job

    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(web.subprocess, "Popen", _FinishedProcess)
    web.RUNS.clear()

    rules = [{
        "obligation_id": "web.required-cooccurrence",
        "kind": "required_cooccurrence", "track": "picture",
        "refs": ["sku.flip8", "sku.fold8"],
    }, {
        "obligation_id": "web.forbidden.1",
        "kind": "forbidden_presence", "track": "picture",
        "refs": ["sku.fold8"],
        "window": {"start_seconds": 0, "end_seconds": 3},
    }]
    response = TestClient(web.create_app()).post("/api/runs", data={
        "source_path": str(rushes), "review": "false",
        "picture_obligations_json": json.dumps(rules),
    })

    assert response.status_code == 200
    job = load_job(Path(response.json()["job"]))
    assert [one.kind for one in job.obligations] == [
        "required_cooccurrence", "forbidden_presence",
    ]
    assert job.obligations[1].window.end_seconds == 3


def test_web_release_approval_is_bound_to_the_watched_draft_hash(
    tmp_path, monkeypatch,
):
    from montagewright.job import EditJob, load_job, write_job
    import montagewright.release as release

    root = tmp_path / "run"
    output = root / "out"
    (output / "work").mkdir(parents=True)
    draft = output / "draft-preview.mp4"
    draft.write_bytes(b"the watched draft")
    digest = hashlib.sha256(draft.read_bytes()).hexdigest()
    write_job(output / "work" / "resolved-job.json", EditJob())
    (output / "report.json").write_text(json.dumps({
        "delivery_status": "ready", "plan_disagreements": [],
    }), encoding="utf-8")
    (output / "work" / "ingest-manifest.json").write_text(json.dumps({
        "inventory_sha256": "a" * 64,
    }), encoding="utf-8")
    monkeypatch.setattr(release, "technical_qc_faults", lambda *_a, **_k: ())
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path)
    web.RUNS.clear()
    web.RUNS["approved"] = web.Run(
        "approved", root, source="", command=[], state="done",
    )

    response = TestClient(web.create_app()).post(
        "/api/runs/approved/release",
        data={
            "approver": "Jas", "approval_note": "checked final crop",
            "expected_artifact_sha256": digest,
            "acknowledge_rights": "true",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "released"
    approved = load_job(output / "work" / "approved-job.json")
    assert approved.release.approved_artifact_sha256 == digest
    assert (output / "deliverable.mp4").read_bytes() == b"the watched draft"


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
    from montagewright.job import load_job

    job = load_job(Path(response.json()["job"]))
    child_brief = Path(str(job.brief))
    assert child_brief.parent == child.root
    assert child_brief.read_text() == parent_brief.read_text()


def test_new_round_inherits_parent_grounding_on_the_server(
    tmp_path, monkeypatch
):
    import shutil
    import montagewright.cli as cli

    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    roots = tmp_path / "runs"
    grounding = tmp_path / "approved-grounding.json"
    grounding.write_text('{"approved": true}', encoding="utf-8")
    monkeypatch.setattr(web, "RUNS_ROOT", roots)
    monkeypatch.setattr(web.subprocess, "Popen", _FinishedProcess)

    def prepare(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination, "f" * 64

    monkeypatch.setattr(cli, "prepare_grounding_spec_artifact", prepare)
    web.RUNS.clear()
    parent_root = roots / "parent"
    parent_root.mkdir(parents=True)
    web.RUNS["parent"] = web.Run(
        "parent", parent_root, source=str(rushes),
        command=[
            "render", str(rushes), "--grounding-spec", str(grounding),
        ],
    )

    response = TestClient(web.create_app()).post(
        "/api/runs", data={
            "source_path": str(rushes), "base_run_id": "parent",
            "review": "false",
        },
    )

    assert response.status_code == 200
    child = web.RUNS[response.json()["run_id"]]
    from montagewright.job import load_job

    job = load_job(Path(response.json()["job"]))
    assert job.subject is not None
    inherited = Path(str(job.subject.grounding_spec))
    assert inherited.parent == child.root / "out" / "work"
    assert inherited.read_text(encoding="utf-8") == grounding.read_text(
        encoding="utf-8"
    )
    page = (
        Path(__file__).parents[1] / "src" / "montagewright" / "web" / "index.html"
    ).read_text(encoding="utf-8")
    assert "已沿用上一輪 grounding spec" in page
    assert "grounding_reference_count" in page


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
    from montagewright.job import load_job

    job = load_job(Path(response.json()["job"]))
    saved = Path(str(job.brief)).read_text()
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


def test_the_empty_home_screen_keeps_watching_for_cli_runs():
    """A terminal run should appear without reloading an empty Web page."""

    page = (
        Path(__file__).parents[1] / "src" / "montagewright" / "web" / "index.html"
    ).read_text(encoding="utf-8")
    empty = page.split("if (!data.runs.length)", 1)[1].split("return;", 1)[0]
    assert "setTimeout(loadPast, 5000)" in empty
    assert "if (!runId)" in empty


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


def test_a_cut_made_from_the_command_line_reads_as_running(tmp_path, monkeypatch):
    """A folder with no report in it meant "died with the last server".

    Which is right for a run that did, and wrong for one that is busy
    cutting -- so a command-line cut showed as interrupted, under a page
    that never refreshed it, for as long as it took to finish. A pid is
    enough to tell them apart, and is only believed while it is alive.
    """

    import json
    import os

    from montagewright.webapp import _state_of_a_foreign_run

    out = tmp_path / "out"
    out.mkdir()
    assert _state_of_a_foreign_run(out) == "interrupted", "no report, no claim"

    (out / "run-state.json").write_text(
        json.dumps({"state": "running", "pid": os.getpid()}), encoding="utf-8"
    )
    assert _state_of_a_foreign_run(out) == "running", "this process is alive"

    # A sandboxed Web server can see the state file while macOS refuses the
    # harmless signal-zero probe. EPERM means the pid exists, not that it
    # died, so a live CLI run must keep polling in the editor.
    with monkeypatch.context() as guarded:
        def cannot_signal(pid, signal):
            del pid, signal
            raise PermissionError("sandbox")

        guarded.setattr(os, "kill", cannot_signal)
        assert _state_of_a_foreign_run(out) == "running"

    # A pid nobody is using: the claim outlived its process.
    (out / "run-state.json").write_text(
        json.dumps({"state": "running", "pid": 2 ** 22}), encoding="utf-8"
    )
    assert _state_of_a_foreign_run(out) == "interrupted"

    (out / "run-state.json").write_text(
        json.dumps({"state": "failed", "pid": 2 ** 22}), encoding="utf-8"
    )
    assert _state_of_a_foreign_run(out) == "failed", "a finished claim stands"

    (out / "run-state.json").unlink()
    (out / "report.json").write_text("{}", encoding="utf-8")
    assert _state_of_a_foreign_run(out) == "done", "older runs still read right"


def test_a_live_pid_survives_the_line_that_distrusts_running(tmp_path, monkeypatch):
    """The check was made and overruled one line later.

    "Anything found on disk is finished as far as this process is
    concerned" was right while the only way to be running was to have been
    started by this server. A cut from the command line leaves a pid, so
    the claim can be checked -- and the very next line still rewrote a
    verified "running" to "interrupted".
    """

    import json
    import os

    import montagewright.webapp as webapp

    runs = tmp_path / "runs"
    folder = runs / "from-the-command-line"
    (folder / "out").mkdir(parents=True)
    (folder / "out" / "command.json").write_text(
        json.dumps({"command": ["montagewright", "render"], "state": "running"}),
        encoding="utf-8",
    )
    (folder / "out" / "run-state.json").write_text(
        json.dumps({"state": "running", "pid": os.getpid()}), encoding="utf-8"
    )
    monkeypatch.setattr(webapp, "RUNS_ROOT", runs)
    monkeypatch.setattr(webapp, "RUNS", {})
    webapp.recall()

    assert webapp.RUNS["from-the-command-line"].state == "running"


def test_web_subtitle_writers_respect_the_output_lease(tmp_path):
    from montagewright.release import acquire_output_lease

    run = web.Run("subtitle-locked", tmp_path / "run")
    work = run.output / "work"
    work.mkdir(parents=True)
    (run.output / "deliverable.mp4").write_bytes(b"picture")
    (work / "subtitles.json").write_text(json.dumps([{
        "at": 0.0, "until": 1.0, "text": "hello",
    }]), encoding="utf-8")
    web.RUNS[run.run_id] = run
    lease = acquire_output_lease(run.output)
    client = TestClient(web.create_app())
    try:
        edited = client.put(
            f"/api/runs/{run.run_id}/subtitle-track",
            json={"lines": []},
        )
        burned = client.post(f"/api/runs/{run.run_id}/burn-subtitles")
        assert edited.status_code == 409
        assert burned.status_code == 409
    finally:
        lease.release()
        web.RUNS.pop(run.run_id, None)
