from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from montagewright.job import EditJob
from montagewright.release import OutputBusy, acquire_output_lease, finalize_release


def test_output_lease_refuses_a_second_writer(tmp_path):
    first = acquire_output_lease(tmp_path / "out")
    try:
        with pytest.raises(OutputBusy, match="already being written"):
            acquire_output_lease(tmp_path / "out")
    finally:
        first.release()


def test_unapproved_render_is_named_draft_and_bound_to_ingest(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    master = output / "deliverable.mp4"
    master.write_bytes(b"movie")
    report = output / "report.json"
    report.write_text(json.dumps({
        "delivery_status": "ready", "plan_disagreements": [],
    }), encoding="utf-8")

    manifest = finalize_release(
        output, master, EditJob(), ingest_inventory_sha256="a" * 64,
        report_path=report,
    )

    assert manifest.status == "draft"
    assert not master.exists()
    assert (output / "draft-preview.mp4").read_bytes() == b"movie"
    assert json.loads(report.read_text())["delivery_status"] == "release_blocked"


def test_approved_rights_checked_render_is_published_atomically(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    master = output / "render.mp4"
    master.write_bytes(b"movie")
    report = output / "report.json"
    report.write_text(json.dumps({
        "delivery_status": "ready", "plan_disagreements": [],
    }), encoding="utf-8")
    job = EditJob.model_validate({
        "rights": {"acknowledged": True},
        "release": {
            "producer_approval_required": True, "approver": "Jas",
            "approved_artifact_sha256": hashlib.sha256(b"movie").hexdigest(),
        },
    })

    manifest = finalize_release(
        output, master, job, ingest_inventory_sha256="b" * 64,
        report_path=report,
    )

    assert manifest.status == "released"
    assert (output / "deliverable.mp4").exists()
    assert not (output / ".deliverable.mp4.staging").exists()
