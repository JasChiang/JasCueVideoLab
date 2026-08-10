from pathlib import Path

import pytest
from pydantic import ValidationError

from montagewright.graphics import (
    BrandKit,
    CopyFact,
    DrawnGraphic,
    GraphicCue,
    GraphicsPlan,
    LayoutEvidence,
    burn_graphics,
    minimum_read_seconds,
    validate_for_render,
    validate_brief_authority,
    resolve_auto_position,
)


def fact(fact_id="name", text="Galaxy Z Fold8", *, approved=True):
    return CopyFact(
        fact_id=fact_id,
        exact_text=text,
        source_kind="user",
        approved=approved,
        approved_by="human_review" if approved else None,
    )


def cue(**changes):
    values = {
        "graphic_id": "g00",
        "kind": "product_name",
        "primary_fact_id": "name",
        "at_seconds": 0.5,
        "duration_seconds": 3.0,
        "template": "product_plate",
        "position": "upper_left",
        "motion": "rise",
        "status": "approved",
    }
    values.update(changes)
    return GraphicCue(**values)


def test_brief_copy_must_be_verbatim_before_it_is_auto_approved():
    from montagewright.brief import parse_brief_markdown

    brief = '''## 創意方向
展開新篇章。

```montagewright-approved-copy
{"version": 1, "items": [{"copy_id": "hero", "text": "Galaxy Z Fold8", "allowed_kinds": ["product_name"]}]}
```
'''
    parsed = parse_brief_markdown(brief)
    approved = parsed.approved_copy[0]

    assert approved.approved and approved.source_kind == "brief_exact"
    assert approved.approved_by == "user_brief"
    assert "approved-copy" not in parsed.creative_brief
    assert parsed.creative_brief == "## 創意方向\n展開新篇章。"


def test_brief_prose_is_not_approved_screen_copy():
    from montagewright.brief import parse_brief_markdown

    parsed = parse_brief_markdown("請介紹 Galaxy Z Fold8 的全新規格")

    assert parsed.approved_copy == ()


def test_model_copy_cannot_approve_itself():
    with pytest.raises(ValidationError, match="cannot be approved in place"):
        CopyFact(
            fact_id="draft", exact_text="史上最強",
            source_kind="model_draft", approved=True,
            approved_by="human_review",
        )


def test_an_approved_cue_cannot_reference_unapproved_copy():
    with pytest.raises(ValidationError, match="unapproved copy"):
        GraphicsPlan(facts=[fact(approved=False)], cues=[cue()])


def test_reading_floor_and_subtitle_collision_are_local_faults():
    plan = GraphicsPlan(
        facts=[fact(text="Galaxy Z Fold8 全新展開摺疊篇章")],
        cues=[cue(duration_seconds=0.5, position="lower_left")],
    )

    faults = validate_for_render(
        plan, duration_seconds=10.0, subtitle_windows=[(0.0, 3.0)]
    )

    assert any("reading floor" in fault for fault in faults)
    assert any("overlaps subtitles" in fault for fault in faults)
    assert minimum_read_seconds("一段需要閱讀的繁中文字卡", "feature") > 1.2


def test_explicit_lower_position_collides_with_subtitles():
    plan = GraphicsPlan(
        facts=[fact()],
        cues=[cue(position="lower_left", template="product_plate")],
    )

    faults = validate_for_render(
        plan, duration_seconds=10.0, subtitle_windows=[(0.0, 4.0)]
    )

    assert any("overlaps subtitles" in fault for fault in faults)


def test_auto_layout_chooses_quiet_side_across_frames(tmp_path: Path):
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", (300, 200), "black")
    pen = ImageDraw.Draw(frame)
    for x in range(0, 150, 2):
        pen.line((x, 0, x, 199), fill="white")
    card = DrawnGraphic(tmp_path / "card.png", 0, 0, 90, 35)

    resolved, _ = resolve_auto_position(
        cue(position="auto", composition="negative_space"),
        card, [frame, frame.copy()], frame_width=300, frame_height=200,
    )

    assert resolved.left > 150


def test_overlap_subject_auto_layout_requires_tracker_evidence(tmp_path: Path):
    from PIL import Image

    frame = Image.new("RGB", (300, 200), "black")
    card = DrawnGraphic(tmp_path / "card.png", 0, 0, 90, 35)

    with pytest.raises(ValueError, match="needs subject evidence"):
        resolve_auto_position(
            cue(position="auto", composition="overlap_subject"),
            card, [frame], frame_width=300, frame_height=200,
            evidence=LayoutEvidence(),
        )


def test_client_cannot_claim_copy_was_approved_by_brief():
    claimed = CopyFact(
        fact_id="hero", exact_text="Fold8 Ultra",
        source_kind="brief_exact", approved=True, approved_by="user_brief",
    )
    plan = GraphicsPlan(facts=[claimed])

    with pytest.raises(ValueError, match="server-side"):
        validate_brief_authority(plan, [])


def test_graphics_burn_to_a_separate_deliverable(tmp_path: Path):
    import subprocess

    source = tmp_path / "clean.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=0x283040:s=360x640:d=3:r=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
        ],
        check=True,
    )
    plan = GraphicsPlan(
        brand=BrandKit(),
        facts=[fact(text="Galaxy Z Fold8")],
        cues=[cue(
            at_seconds=0.25, duration_seconds=2.5,
            position="auto", composition="negative_space",
        )],
    )
    destination = tmp_path / "graphics.mp4"

    made = burn_graphics(
        source, plan, destination, work=tmp_path / "work"
    )

    assert made == destination and made.stat().st_size > 1000
    assert source.exists() and source != made
    assert (tmp_path / "work" / "layout.json").exists()


def test_web_graphics_track_round_trips_approved_copy(tmp_path: Path):
    import json
    import montagewright.webapp as web
    from fastapi.testclient import TestClient

    was = web.RUNS_ROOT
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        here = web.RUNS_ROOT / "r1"
        (here / "out" / "work").mkdir(parents=True)
        (here / "run.json").write_text(
            json.dumps({"state": "done", "started_at": 0.0}),
            encoding="utf-8",
        )
        client = TestClient(web.create_app())
        payload = GraphicsPlan(
            facts=[fact()], cues=[cue()]
        ).model_dump(mode="json")

        saved = client.put("/api/runs/r1/graphics-track", json=payload)
        loaded = client.get("/api/runs/r1/graphics-track")

        assert saved.status_code == 200
        assert loaded.status_code == 200
        assert loaded.json()["resolved"][0]["primary_text"] == "Galaxy Z Fold8"
        assert any(
            template["template_id"] == "product_plate"
            for template in loaded.json()["templates"]
        )
    finally:
        web.RUNS_ROOT = was
        web.RUNS.pop("r1", None)


def test_web_graphics_track_starts_with_copy_approved_by_brief(tmp_path: Path):
    import json
    import montagewright.webapp as web
    from fastapi.testclient import TestClient

    was = web.RUNS_ROOT
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        here = web.RUNS_ROOT / "r1"
        work = here / "out" / "work"
        work.mkdir(parents=True)
        (here / "run.json").write_text(
            json.dumps({"state": "done", "started_at": 0.0}),
            encoding="utf-8",
        )
        brief_fact = CopyFact(
            fact_id="hero", exact_text="Galaxy Z Fold8",
            source_kind="brief_exact", source_reference="/items/0/text",
            approved=True, approved_by="user_brief",
        )
        (work / "approved-copy.json").write_text(json.dumps({
            "brief_sha256": "abc",
            "facts": [brief_fact.model_dump(mode="json")],
        }), encoding="utf-8")

        loaded = TestClient(web.create_app()).get(
            "/api/runs/r1/graphics-track"
        )

        assert loaded.status_code == 200
        assert loaded.json()["facts"][0]["exact_text"] == "Galaxy Z Fold8"
        assert loaded.json()["facts"][0]["approved_by"] == "user_brief"
    finally:
        web.RUNS_ROOT = was
        web.RUNS.pop("r1", None)


def test_durable_sam_track_becomes_cue_window_layout_evidence(tmp_path: Path):
    import json
    import montagewright.webapp as web

    run = web.Run("r1", tmp_path / "r1")
    work = run.output / "work"
    work.mkdir(parents=True)
    (work / "crops.json").write_text(json.dumps({
        "k00": [{"at": 0.0, "x": 0.25, "y": 0.0, "w": 0.5, "h": 1.0}]
    }), encoding="utf-8")
    (run.output / "report.json").write_text(json.dumps({
        "subject_tracks": {"k00": [
            {"seconds": 0.5, "centre_x": 0.5, "centre_y": 0.5,
             "width": 0.2, "height": 0.6, "source": "sam2.1"},
            {"seconds": 4.0, "centre_x": 0.1, "centre_y": 0.1,
             "width": 0.1, "height": 0.1, "source": "sam2.1"},
        ]},
    }), encoding="utf-8")
    plan = GraphicsPlan(
        facts=[fact()], cues=[cue(anchor_clip_id="k00")]
    )

    found = web._graphics_layout_evidence(run, plan)

    assert found["g00"].source == "sam2.1_report_track"
    assert len(found["g00"].subject_boxes) == 1
    assert found["g00"].subject_boxes[0] == pytest.approx(
        (0.3, 0.2, 0.4, 0.6)
    )
