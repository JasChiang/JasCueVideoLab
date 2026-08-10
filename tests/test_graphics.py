from pathlib import Path

import pytest
from pydantic import ValidationError

from montagewright.graphics import (
    BrandKit,
    CopyFact,
    DrawnGraphic,
    GraphicCue,
    GraphicStyle,
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


def test_plain_markdown_becomes_unapproved_cards_and_layout_notes():
    from montagewright.brief import parse_brief_markdown

    parsed = parse_brief_markdown('''# 發表會
Galaxy Z Fold 系列
星成員登場

—
Galaxy Z Fold8 Ultra
極致進化

串場：兩款手機比較畫面

全系列支援
Galaxy AI
Gemini Intelligence
（可以的話置中三行）

Galaxy Watch Ultra2
EN13319 國際潛水標準認證
最深 40 公尺水下環境使用
（兩行塞不下就用 40m 那個好了）
''')

    assert parsed.approved_copy == ()
    assert parsed.candidates[0].kind == "opening_title"
    assert parsed.candidates[1].primary_text == "Galaxy Z Fold8 Ultra"
    assert parsed.candidates[2].template == "center_stack"
    assert parsed.candidates[2].instruction == "可以的話置中三行"
    assert parsed.candidates[3].variants[0].secondary_text == "40m"
    assert parsed.instructions[0].kind == "editorial"
    assert all(candidate.primary_text != "—" for candidate in parsed.candidates)


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


def test_multiline_center_and_end_templates_render(tmp_path: Path):
    from montagewright.graphics import draw_graphic

    plan = GraphicsPlan(
        facts=[
            fact("hero", "全系列支援"),
            fact("lines", "Galaxy AI\nGemini Intelligence"),
        ],
        cues=[cue(
            kind="feature", primary_fact_id="hero",
            secondary_fact_id="lines", template="center_stack",
            position="center",
        )],
    )

    made = draw_graphic(
        plan.cues[0], plan, width=1080, height=1920,
        into=tmp_path / "three-lines.png",
    )

    assert made.path.exists() and made.height > 150


def test_old_graphics_json_without_style_gets_safe_defaults():
    old = {
        "graphic_id": "legacy", "kind": "product_name",
        "primary_fact_id": "name", "status": "draft",
    }

    loaded = GraphicCue.model_validate(old)

    assert loaded.style == GraphicStyle()
    assert loaded.style.stroke_width == 0
    assert loaded.style.shadow_opacity == 0


def test_adjustable_outline_shadow_border_and_emphasis_render(tmp_path: Path):
    from PIL import Image
    from montagewright.graphics import draw_graphic

    plan = GraphicsPlan(
        facts=[fact(text="4.1mm 纖薄機身")],
        cues=[cue(
            background="plate",
            style=GraphicStyle(
                preset="tech_frame", primary_scale=1.2,
                primary_color="#FFFFFF", emphasis_text="4.1mm",
                emphasis_color="#00FF66", stroke_width=5,
                stroke_color="#000000", shadow_opacity=190,
                shadow_blur=6, shadow_offset_x=4, shadow_offset_y=5,
                plate_color="#101828", plate_alpha=220,
                plate_border_width=3, plate_border_color="#4B7BFF",
                corner_radius=18, padding_x=36, padding_y=24,
            ),
        )],
    )

    made = draw_graphic(
        plan.cues[0], plan, width=1080, height=1920,
        into=tmp_path / "styled.png",
    )
    colours = Image.open(made.path).convert("RGBA").getdata()

    assert any(r < 20 and g > 230 and b < 130 and a > 200
               for r, g, b, a in colours)
    assert any(b > 220 and 50 < r < 120 and 70 < g < 160 and a > 200
               for r, g, b, a in colours)
    assert made.height > 100


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
            facts=[fact(approved=False)], cues=[cue(status="draft")]
        ).model_dump(mode="json")

        saved = client.put("/api/runs/r1/graphics-track", json=payload)
        approved = client.post(
            "/api/runs/r1/approve-graphic/g00", json={"revision": 1}
        )
        loaded = client.get("/api/runs/r1/graphics-track")

        assert saved.status_code == 200
        assert saved.json()["revision"] == 1
        assert approved.status_code == 200
        assert approved.json()["revision"] == 2
        assert loaded.status_code == 200
        assert loaded.json()["revision"] == 2
        assert loaded.json()["resolved"][0]["primary_text"] == "Galaxy Z Fold8"
        assert loaded.json()["resolved"][0]["status"] == "approved"
        assert any(
            template["template_id"] == "product_plate"
            for template in loaded.json()["templates"]
        )
        assert client.put(
            "/api/runs/r1/graphics-track", json=payload
        ).status_code == 409
        edited = approved.json()
        edited["facts"][0].update({
            "exact_text": "Galaxy Z Fold8 Ultra", "approved": False,
            "approved_by": None, "text_sha256": "",
        })
        edited["cues"][0]["status"] = "draft"
        edited_save = client.put(
            "/api/runs/r1/graphics-track", json=edited
        )
        reapproved = client.post(
            "/api/runs/r1/approve-graphic/g00", json={"revision": 3}
        )
        assert edited_save.json()["revision"] == 3
        assert reapproved.status_code == 200
        assert reapproved.json()["revision"] == 4
        assert reapproved.json()["facts"][0]["text_sha256"]
    finally:
        web.RUNS_ROOT = was
        web.RUNS.pop("r1", None)


def test_generic_put_cannot_forge_human_review(tmp_path: Path):
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
        forged = GraphicsPlan(
            facts=[fact()], cues=[cue()]
        ).model_dump(mode="json")

        denied = TestClient(web.create_app()).put(
            "/api/runs/r1/graphics-track", json=forged
        )

        assert denied.status_code == 422
        assert "明確的核准動作" in denied.json()["detail"]
    finally:
        web.RUNS_ROOT = was
        web.RUNS.pop("r1", None)


def test_web_preview_uses_the_production_card_compiler(tmp_path: Path):
    import json
    import subprocess
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
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=360x640:d=1:r=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(here / "out" / "deliverable.mp4"),
        ], check=True)
        payload = GraphicsPlan(
            facts=[fact()], cues=[cue(
                status="draft", style=GraphicStyle(
                    preset="outlined", stroke_width=6,
                ),
            )],
        ).model_dump(mode="json")
        client = TestClient(web.create_app())

        compiled = client.post(
            "/api/runs/r1/graphics-preview/g00", json=payload
        )

        assert compiled.status_code == 200
        assert compiled.json()["card_width"] > 0
        assert client.get(compiled.json()["url"]).headers["content-type"] == "image/png"
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


def test_web_graphics_track_exposes_plain_brief_candidates(tmp_path: Path):
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
        (work / "brief-candidates.json").write_text(json.dumps({
            "brief_sha256": "abc",
            "candidates": [{
                "candidate_id": "brief.p01", "primary_text": "Galaxy AI",
                "secondary_text": "Gemini Intelligence", "kind": "feature",
                "template": "center_stack", "position": "center",
                "instruction": "置中", "source_reference": "/paragraphs/1",
                "variants": [],
            }],
            "instructions": [],
        }), encoding="utf-8")

        loaded = TestClient(web.create_app()).get(
            "/api/runs/r1/graphics-track"
        )

        assert loaded.status_code == 200
        assert loaded.json()["brief_candidates"][0]["template"] == "center_stack"
        assert loaded.json()["brief_sha256"] == "abc"
    finally:
        web.RUNS_ROOT = was
        web.RUNS.pop("r1", None)


def test_old_web_run_backfills_candidates_from_its_brief_without_gemini(
    tmp_path: Path,
):
    import json
    import montagewright.webapp as web
    from fastapi.testclient import TestClient

    was = web.RUNS_ROOT
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        brief = tmp_path / "brief.md"
        brief.write_text(
            "Galaxy Z Fold8\n全新比例\n\n串場：比較畫面",
            encoding="utf-8",
        )
        here = web.RUNS_ROOT / "r1"
        (here / "out" / "work").mkdir(parents=True)
        (here / "run.json").write_text(json.dumps({
            "state": "done", "started_at": 0.0,
            "command": ["montagewright", "--brief", str(brief)],
        }), encoding="utf-8")

        loaded = TestClient(web.create_app()).get(
            "/api/runs/r1/graphics-track"
        )

        assert loaded.status_code == 200
        assert loaded.json()["brief_candidates"][0]["primary_text"] == "Galaxy Z Fold8"
        assert loaded.json()["brief_instructions"][0]["kind"] == "editorial"
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
