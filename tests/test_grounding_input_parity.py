from __future__ import annotations

import hashlib
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import montagewright.cli as cli
import montagewright.webapp as web


def _transport_loader_module(monkeypatch) -> None:
    """A narrow schema stand-in: these tests own transport, not validation."""

    class Loaded:
        def __init__(self, path: Path):
            self.path = path.resolve()
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.reference_images = tuple(
                SimpleNamespace(**reference)
                for reference in self.payload["reference_images"]
            )

        def canonical_definition_json(self) -> str:
            return json.dumps(
                self.payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )

        def resolve_reference_path(self, reference) -> Path:
            return (self.path.parent / reference.path).resolve()

    module = types.ModuleType("montagewright.reference_grounding")
    module.ReferenceGroundingError = RuntimeError
    module.load_grounding_spec = lambda path: Loaded(Path(path))
    monkeypatch.setitem(sys.modules, module.__name__, module)


class _FinishedProcess:
    def __init__(self, command, **_):
        self.command = command
        self.stdout = io.StringIO("")

    def wait(self) -> int:
        return 0

    def poll(self) -> int:
        return 0


def _source_spec(folder: Path) -> tuple[Path, bytes]:
    folder.mkdir(parents=True)
    image_bytes = b"reference-image-bytes"
    (folder / "chosen.png").write_bytes(image_bytes)
    payload = {
        "contract_version": "reference-grounding-spec-v1",
        "identity_lock": {"transport_fixture": True},
        "reference_images": [{
            "target_id": "subject.primary",
            "polarity": "positive",
            "frame_id": "RF000120",
            "path": "chosen.png",
            "content_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "mime_type": "image/png",
            "presentation": "raw",
        }],
    }
    source = folder / "grounding-spec.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    return source, image_bytes


def _real_source_spec(folder: Path) -> tuple[Path, bytes]:
    from montagewright.measure.models import (
        EvidenceAnchor,
        EvidenceApprovalSource,
        EvidenceClaimSource,
        EvidenceFramingObligations,
        EvidenceIdentityContract,
        EvidenceQueryApprovalProvenance,
        EvidenceQueryLock,
        EvidenceQueryProvenance,
        EvidenceTargetIdentity,
    )

    folder.mkdir(parents=True)
    image_bytes = b"approved-reference-crop"
    image = folder / "approved.jpg"
    image.write_bytes(image_bytes)
    digest = hashlib.sha256(image_bytes).hexdigest()
    lock = EvidenceQueryLock(
        query_id="query:web-cli-parity",
        revision=1,
        editorial_goal="Keep the approved instance in frame.",
        identity=EvidenceIdentityContract(targets=(
            EvidenceTargetIdentity(
                target_id="target.primary",
                target_description="the reviewer-approved instance",
                identity_cues=("distinctive corner mark",),
                positive_anchors=(EvidenceAnchor(
                    frame_id="reference.frame.001",
                    crop_sha256=digest,
                ),),
            ),
        )),
        framing=EvidenceFramingObligations(
            required_target_ids=("target.primary",),
            framing_intent="Keep the approved instance recognizable.",
        ),
        claim_source=EvidenceClaimSource.HUMAN_REVIEW,
        provenance=EvidenceQueryProvenance(
            created_at="2026-08-12T00:00:00Z",
            created_by="reviewer:parity",
        ),
        approval=EvidenceQueryApprovalProvenance(
            approved_at="2026-08-12T00:01:00Z",
            approved_by="reviewer:parity",
            approval_source=EvidenceApprovalSource.HUMAN_REVIEW,
        ),
    )
    payload = {
        "contract_version": "reference-grounding-spec-v1",
        "identity_lock": lock.model_dump(mode="json"),
        "reference_images": [{
            "target_id": "target.primary",
            "polarity": "positive",
            "frame_id": "reference.frame.001",
            "path": image.name,
            "anchor_crop_sha256": digest,
            "content_sha256": digest,
            "mime_type": "image/jpeg",
            "presentation": "raw",
        }],
    }
    source = folder / "grounding-spec.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    return source, image_bytes


def test_cli_parser_exposes_one_grounding_spec_input(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli, "command_render", lambda args: captured.setdefault("args", args) or 0
    )
    source = tmp_path / "spec.json"

    cli.main([
        "render", str(tmp_path), "--grounding-spec", str(source),
        "--output", str(tmp_path / "out"),
    ])

    assert captured["args"].grounding_spec == source
    assert captured["args"]._argv.count("--grounding-spec") == 1


def test_simple_cli_grounding_flags_are_replaced_by_the_canonical_spec():
    canonical = Path("/run/work/grounding-spec.json")
    recorded = cli._command_with_canonical_grounding([
        "render", "/rushes",
        "--grounding-target-id", "foldable.hero",
        "--grounding-target-description=the approved foldable",
        "--grounding-reference", "/refs/front.jpg",
        "--grounding-reference=/refs/open.jpg",
        "--grounding-identity-cue", "hinge and camera layout",
        "--grounding-exclusion=not a tablet",
        "--output", "/run",
    ], canonical)

    assert recorded == [
        "render", "/rushes", "--output", "/run",
        "--grounding-spec", str(canonical),
    ]


def test_public_helper_returns_path_and_real_validated_canonical_spec(tmp_path):
    from montagewright.reference_grounding import load_grounding_spec

    source, image_bytes = _real_source_spec(tmp_path / "source")
    destination = tmp_path / "out" / "work" / "grounding-spec.json"

    artifact, loaded = cli.prepare_grounding_spec_artifact(source, destination)

    assert artifact == destination.resolve()
    assert loaded.source_path == str(artifact)
    assert artifact.read_text(encoding="utf-8") == loaded.canonical_definition_json()
    assert loaded.definition_sha256() == load_grounding_spec(
        artifact
    ).definition_sha256()
    reference = loaded.reference_images[0]
    assert reference.path == (
        f"reference-images/{reference.content_sha256}.jpg"
    )
    assert loaded.resolve_reference_path(reference).read_bytes() == image_bytes


def test_cli_command_records_only_the_canonical_grounding_artifact(
    tmp_path, monkeypatch
):
    source, _ = _real_source_spec(tmp_path / "source")
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    output = tmp_path / "out"

    class StopAfterCommand(Exception):
        pass

    monkeypatch.setattr(cli, "_tee_output", lambda _path: None)
    monkeypatch.setattr(cli, "_sam_checkpoint_for", lambda _args: None)
    monkeypatch.setattr(cli, "_make_findable", lambda _path: None)
    # Ingest now deliberately precedes client construction. Stop at that
    # boundary: this test only needs the command record written immediately
    # before it and must not depend on a valid media fixture.
    monkeypatch.setattr(
        "montagewright.ingest.build_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(StopAfterCommand),
    )

    with pytest.raises(StopAfterCommand):
        cli.main([
            "render", str(rushes), "--grounding-spec", str(source),
            "--output", str(output),
        ])

    record = json.loads((output / "command.json").read_text(encoding="utf-8"))
    canonical = output / "work" / "grounding-spec.json"
    assert record["grounding_spec"] == str(canonical)
    assert record["grounding_spec_sha256"] == hashlib.sha256(
        canonical.read_bytes()
    ).hexdigest()
    at = record["original_command"].index("--grounding-spec")
    assert record["original_command"][at + 1] == str(canonical)
    assert record["command"][-2:] == [
        "--job", str(output / "work" / "resolved-job.json"),
    ]
    assert str(source) not in record["command"]


def test_web_uses_real_loader_and_rejects_changed_reference_bytes(
    tmp_path, monkeypatch
):
    source, _ = _real_source_spec(tmp_path / "source")
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(
        web.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid reference bytes must not start the CLI")
        ),
    )
    web.RUNS.clear()

    response = TestClient(web.create_app()).post(
        "/api/runs",
        data={
            "source_path": str(rushes),
            "grounding_spec_json": source.read_text(encoding="utf-8"),
        },
        files=[
            (
                "reference_images",
                ("approved.jpg", b"changed bytes", "image/jpeg"),
            )
        ],
    )

    assert response.status_code == 400
    assert "content hash mismatch" in response.json()["detail"]


def test_pasted_spec_requires_all_of_its_reference_uploads(tmp_path, monkeypatch):
    source, _ = _real_source_spec(tmp_path / "source")
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(
        web.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an incomplete portable spec must not start the CLI")
        ),
    )
    web.RUNS.clear()

    response = TestClient(web.create_app()).post(
        "/api/runs",
        data={
            "source_path": str(rushes),
            "grounding_spec_json": source.read_text(encoding="utf-8"),
        },
    )

    assert response.status_code == 400
    assert "require every reference image" in response.json()["detail"]


def test_web_path_and_upload_make_identical_cli_grounding_artifacts(
    tmp_path, monkeypatch
):
    _transport_loader_module(monkeypatch)
    source, image_bytes = _source_spec(tmp_path / "input")
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()

    runs = tmp_path / "runs"
    monkeypatch.setattr(web, "RUNS_ROOT", runs)
    monkeypatch.setattr(web.subprocess, "Popen", _FinishedProcess)
    web.RUNS.clear()
    client = TestClient(web.create_app())

    by_path = client.post(
        "/api/runs",
        data={
            "source_path": str(rushes),
            "grounding_spec_path": str(source),
            "review": "false",
        },
    )
    assert by_path.status_code == 200, by_path.text

    raw_json = source.read_text(encoding="utf-8")
    by_upload = client.post(
        "/api/runs",
        data={
            "source_path": str(rushes),
            "grounding_spec_json": raw_json,
            "review": "false",
        },
        files=[("reference_images", ("chosen.png", image_bytes, "image/png"))],
    )
    assert by_upload.status_code == 200, by_upload.text

    by_spec_upload = client.post(
        "/api/runs",
        data={"source_path": str(rushes), "review": "false"},
        files=[
            (
                "grounding_spec_file",
                ("grounding-spec.json", raw_json.encode("utf-8"), "application/json"),
            ),
            ("reference_images", ("chosen.png", image_bytes, "image/png")),
        ],
    )
    assert by_spec_upload.status_code == 200, by_spec_upload.text

    by_reference_path = client.post(
        "/api/runs",
        data={
            "source_path": str(rushes),
            "grounding_spec_json": raw_json,
            "reference_image_paths": str(source.parent / "chosen.png"),
            "review": "false",
        },
    )
    assert by_reference_path.status_code == 200, by_reference_path.text

    path_artifact = Path(by_path.json()["grounding_spec"])
    upload_artifact = Path(by_upload.json()["grounding_spec"])
    spec_upload_artifact = Path(by_spec_upload.json()["grounding_spec"])
    reference_path_artifact = Path(by_reference_path.json()["grounding_spec"])
    assert path_artifact.name == upload_artifact.name == "grounding-spec.json"
    assert path_artifact.read_bytes() == upload_artifact.read_bytes()
    assert path_artifact.read_bytes() == spec_upload_artifact.read_bytes()
    assert path_artifact.read_bytes() == reference_path_artifact.read_bytes()

    for response, artifact in (
        (by_path, path_artifact),
        (by_upload, upload_artifact),
        (by_spec_upload, spec_upload_artifact),
        (by_reference_path, reference_path_artifact),
    ):
        from montagewright.job import load_job

        run = web.RUNS[response.json()["run_id"]]
        assert run.command[-2:] == ["--job", response.json()["job"]]
        job = load_job(Path(response.json()["job"]))
        assert job.subject is not None
        assert job.subject.grounding_spec == str(artifact)
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        stored = artifact.parent / payload["reference_images"][0]["path"]
        assert stored.read_bytes() == image_bytes


def test_web_rejects_an_uploaded_reference_not_named_by_the_spec(
    tmp_path, monkeypatch
):
    _transport_loader_module(monkeypatch)
    source, _ = _source_spec(tmp_path / "input")
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()

    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(
        web.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an invalid spec must not start the CLI")
        ),
    )
    web.RUNS.clear()
    response = TestClient(web.create_app()).post(
        "/api/runs",
        data={
            "source_path": str(rushes),
            "grounding_spec_json": source.read_text(encoding="utf-8"),
        },
        files=[("reference_images", ("someone-else.png", b"x", "image/png"))],
    )

    assert response.status_code == 400
    assert "not named by the spec" in response.json()["detail"]
    assert not (tmp_path / "runs").exists() or not any(
        (tmp_path / "runs").iterdir()
    )


def test_web_simple_reference_fields_build_the_same_strict_contract(
    tmp_path, monkeypatch
):
    from montagewright.reference_grounding import load_grounding_spec

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
            "grounding_target_id": "device.fold",
            "grounding_target_description": "the exact foldable in the photo",
            "grounding_presence_policy": "target_only",
            "grounding_identity_cues": "distinct hinge\nvertical cameras",
            "grounding_exclusions": "ordinary tablet",
            "review": "false",
        },
        files=[
            ("reference_images", ("fold.jpg", b"reference bytes", "image/jpeg"))
        ],
    )

    assert response.status_code == 200, response.text
    artifact = Path(response.json()["grounding_spec"])
    spec = load_grounding_spec(artifact)
    target = spec.identity_lock.identity.target("device.fold")
    assert target.identity_cues == ("distinct hinge", "vertical cameras")
    assert target.stable_exclusions == ("ordinary tablet",)
    assert spec.identity_lock.framing.editorial_presence_policy == "target_only"
    assert spec.resolve_reference_path(spec.reference_images[0]).read_bytes() == (
        b"reference bytes"
    )


def test_web_caps_reference_uploads_before_starting_the_run(
    tmp_path, monkeypatch
):
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    monkeypatch.setattr(web, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(web, "MAX_GROUNDING_UPLOAD_BYTES", 4)
    monkeypatch.setattr(
        web.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized references must not start the CLI")
        ),
    )
    web.RUNS.clear()

    response = TestClient(web.create_app()).post(
        "/api/runs",
        data={
            "source_path": str(rushes),
            "grounding_target_description": "the exact approved object",
        },
        files=[("reference_images", ("large.jpg", b"12345", "image/jpeg"))],
    )

    assert response.status_code == 413
    assert "grounding spec and reference uploads exceed" in response.json()["detail"]
    assert not (tmp_path / "runs").exists() or not any(
        (tmp_path / "runs").iterdir()
    )


def test_both_entry_points_can_show_what_the_target_is_not(tmp_path):
    """The spec has carried negative anchors since it was written.

    Neither the CLI nor the form offered them, so telling two similar
    things apart rested entirely on prose -- and prose is where the
    mistakes were: a ratio written in the wrong orientation sent the
    grounding to the other model, confidently, with the reference images
    for the right one attached the whole time.
    """

    import inspect

    from montagewright import cli, webapp
    from montagewright.reference_grounding import build_reference_grounding_spec

    page = (
        Path(__file__).parents[1] / "src" / "montagewright" / "web" / "index.html"
    ).read_text(encoding="utf-8")
    assert "--grounding-negative" in inspect.getsource(cli.main)
    assert "negative_images=" in inspect.getsource(cli.command_render)
    assert "grounding_negatives" in inspect.getsource(webapp.create_app)
    assert 'id="grounding-negatives"' in page
    assert "grounding_negatives" in page, "and the form actually sends them"

    positive = tmp_path / "target.jpg"
    lookalike = tmp_path / "other.jpg"
    positive.write_bytes(b"the one we mean")
    lookalike.write_bytes(b"the one we do not")
    spec = build_reference_grounding_spec(
        tmp_path / "spec.json",
        target_id="device.fold",
        target_description="the exact foldable in the reference images",
        positive_images=(positive,),
        negative_images=(lookalike,),
    )
    target = spec.identity_lock.identity.target("device.fold")
    assert len(target.negative_anchors) == 1
    assert {one.polarity for one in spec.reference_images} == {
        "positive", "negative",
    }
