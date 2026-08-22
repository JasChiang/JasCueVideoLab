from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from montagewright import cli
from montagewright.job import EditJob, RunPolicy, load_job, write_job


def test_yaml_job_resolves_relative_paths_and_cli_flags_override_it(
    tmp_path, monkeypatch,
):
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    (rushes / "take.mp4").touch()
    (tmp_path / "brief.md").write_text("show the product", encoding="utf-8")
    job_path = tmp_path / "edit.yaml"
    job_path.write_text(
        """\
version: montagewright-job-v1
rushes: rushes
output: out
brief: brief.md
delivery:
  aspect: 9:16
  seconds: 60
  duration_mode: preferred
  subtitles: none
sound:
  speech: never
run:
  budget_usd: 3
""",
        encoding="utf-8",
    )
    captured: dict[str, argparse.Namespace] = {}
    monkeypatch.setattr(
        cli, "command_render",
        lambda args: captured.setdefault("args", args) and 0,
    )

    assert cli.main([
        "render", "--job", str(job_path), "--aspect", "16:9",
    ]) == 0

    args = captured["args"]
    assert args.rushes == rushes
    assert args.output == tmp_path / "out"
    assert args.brief == tmp_path / "brief.md"
    assert args.aspect == "16:9"
    assert args.speech == "never"
    assert args.subtitles == "none"
    assert "--job" not in args._argv
    assert args._job_path == job_path


def test_job_rejects_sound_and_delivery_that_cannot_both_be_executed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        """{
          "version": "montagewright-job-v1",
          "delivery": {"subtitles": "burn"},
          "sound": {"speech": "never"}
        }""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot produce transcript subtitles"):
        load_job(path)


def test_job_json_round_trip_is_strict_and_stable(tmp_path):
    path = write_job(tmp_path / "edit-job.json", EditJob(
        rushes="rushes", output="out",
    ))

    loaded = load_job(path)
    assert loaded.version == "montagewright-job-v1"
    assert loaded.rushes == "rushes"
    assert loaded.output == "out"


def test_explicit_cli_can_turn_off_review_requested_by_job(tmp_path, monkeypatch):
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    path = write_job(tmp_path / "edit-job.json", EditJob(
        rushes=str(rushes), output=str(tmp_path / "out"),
        run=RunPolicy(review=True),
    ))
    captured: dict[str, argparse.Namespace] = {}
    monkeypatch.setattr(
        cli, "command_render",
        lambda args: captured.setdefault("args", args) and 0,
    )

    assert cli.main(["render", "--job", str(path), "--no-review"]) == 0
    assert captured["args"].review is False


def test_execution_contract_refuses_unimplemented_delivery_without_hiding_it():
    job = EditJob.model_validate({
        "delivery": {"frame_rate": "23.976", "codec": "prores"},
    })

    faults = job.execution_contract_faults()

    assert any("23.976" in one for one in faults)
    assert any("prores" in one for one in faults)

    composite = EditJob.model_validate({
        "picture_composition": {
            "mode": "split_screen", "description": "two live sources side by side",
        },
    })
    assert any("split_screen" in one for one in composite.execution_contract_faults())


def test_campaign_variants_are_separate_edits_with_one_total_budget(tmp_path, monkeypatch):
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps({
        "version": "montagewright-job-v1",
        "rushes": str(rushes),
        "output": str(tmp_path / "campaign"),
        "run": {"budget_usd": 6},
        "variants": [
            {"variant_id": "wide", "delivery": {"aspect": "16:9", "seconds": 30}},
            {"variant_id": "vertical", "delivery": {"aspect": "9:16", "seconds": 15}},
        ],
    }), encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli, "command_render", lambda args: calls.append(args) or 0)

    assert cli.main(["render", "--job", str(path)]) == 0

    assert [(one.aspect, one.seconds) for one in calls] == [
        ("16:9", 30), ("9:16", 15), ("16:9", 30), ("9:16", 15),
    ]
    assert [one.preflight_only for one in calls] == [True, True, False, False]
    assert [one.budget for one in calls] == [3.0, 3.0, 3.0, 3.0]
    assert calls[0].output == tmp_path / "campaign" / "wide"
    assert calls[1].output == tmp_path / "campaign" / "vertical"
    manifest = json.loads((tmp_path / "campaign" / "campaign-manifest.json").read_text())
    assert [one["status"] for one in manifest["variants"]] == [
        "rendered_draft", "rendered_draft",
    ]


def test_campaign_preflight_only_never_enters_paid_variant_render(tmp_path, monkeypatch):
    rushes = tmp_path / "rushes"
    rushes.mkdir()
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps({
        "version": "montagewright-job-v1",
        "rushes": str(rushes),
        "output": str(tmp_path / "campaign"),
        "variants": [
            {"variant_id": "wide", "delivery": {"aspect": "16:9", "seconds": 30}},
            {"variant_id": "vertical", "delivery": {"aspect": "9:16", "seconds": 15}},
        ],
    }), encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli, "command_render", lambda args: calls.append(args) or 0)

    assert cli.main(["render", "--job", str(path), "--preflight-only"]) == 0

    assert [one.preflight_only for one in calls] == [True, True]
    manifest = json.loads((tmp_path / "campaign" / "campaign-manifest.json").read_text())
    assert [one["status"] for one in manifest["variants"]] == [
        "preflight_ready", "preflight_ready",
    ]


def test_duration_range_is_a_distinct_delivery_contract():
    from montagewright.job import Delivery

    delivery = Delivery(
        duration_mode="range", seconds=30,
        minimum_seconds=27, maximum_seconds=33,
    )
    assert delivery.minimum_seconds == 27
    with pytest.raises(ValueError, match="must fall inside"):
        Delivery(
            duration_mode="range", seconds=40,
            minimum_seconds=27, maximum_seconds=33,
        )


def test_range_without_centre_keeps_its_mode_when_compiled(tmp_path):
    from montagewright.job import EditJob, job_to_argv

    job = EditJob.model_validate({
        "rushes": str(tmp_path / "rushes"),
        "delivery": {
            "duration_mode": "range",
            "seconds": 0,
            "minimum_seconds": 27,
            "maximum_seconds": 33,
        },
    })
    _, argv = job_to_argv(job, tmp_path / "edit-job.yaml")

    assert argv[argv.index("--duration-mode") + 1] == "range"


def test_unverified_dialogue_promises_fail_before_paid_planning():
    from montagewright.job import EditJob

    job = EditJob.model_validate({
        "dialogue": {
            "edit_mode": "phrase_edit",
            "remove_fillers": True,
            "preserve_question": True,
            "allow_translation": True,
        },
    })

    faults = job.execution_contract_faults()
    assert any("remove_fillers" in one for one in faults)
    assert any("preserve_question" in one for one in faults)
    assert any("allow_translation" in one for one in faults)
