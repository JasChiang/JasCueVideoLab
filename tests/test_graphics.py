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
    compile_graphic,
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


def test_concurrency_validation_uses_real_active_sets_not_pairwise_overlap():
    plan = GraphicsPlan(
        facts=[fact("a", "A"), fact("b", "B"), fact("c", "C")],
        cues=[
            cue(graphic_id="a", primary_fact_id="a", at_seconds=0, duration_seconds=10),
            cue(graphic_id="b", primary_fact_id="b", at_seconds=0, duration_seconds=1),
            cue(graphic_id="c", primary_fact_id="c", at_seconds=9, duration_seconds=1),
        ],
    )

    faults = validate_for_render(plan, duration_seconds=10)

    assert not any("more than two" in fault for fault in faults)


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


def test_single_pass_composites_two_cards_subtitles_and_audio(tmp_path: Path):
    import json
    import subprocess
    from montagewright.subtitles import prepare_overlays
    from montagewright.transcript import Line

    source = tmp_path / "clean-with-audio.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=0x283040:s=360x640:d=3:r=30000/1001",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(source),
    ], check=True)
    track = prepare_overlays([
        Line("first line", 0.3, 1.0),
        Line("second line", 1.4, 2.2),
    ], aspect="9:16", width=360, height=640, work=tmp_path / "subs")
    plan = GraphicsPlan(
        facts=[fact("one", "First"), fact("two", "Second")],
        cues=[
                cue(
                    graphic_id="g00", primary_fact_id="one",
                    kind="callout", template="stat_badge", position="auto",
                    at_seconds=.25, duration_seconds=2.5, z_index=5,
                ),
                cue(
                    graphic_id="g01", primary_fact_id="two",
                    kind="callout", template="stat_badge", position="auto",
                    at_seconds=.25, duration_seconds=2.5, z_index=-1,
                ),
        ],
    )
    destination = tmp_path / "combined.mp4"

    burn_graphics(
        source, plan, destination, work=tmp_path / "graphics",
        subtitle_windows=track.windows, subtitle_boxes=track.boxes,
        subtitle_overlays=list(track.overlays),
    )

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,avg_frame_rate", "-of", "json",
        str(destination),
    ], check=True, capture_output=True, text=True)
    media = json.loads(probe.stdout)
    assert {stream["codec_type"] for stream in media["streams"]} == {"video", "audio"}
    assert float(media["format"]["duration"]) == pytest.approx(3.0, abs=0.05)
    video = next(stream for stream in media["streams"] if stream["codec_type"] == "video")
    assert video["avg_frame_rate"] == "30000/1001"
    layout = json.loads((tmp_path / "graphics" / "layout.json").read_text())
    assert layout["g00"]["z_index"] == 5
    assert layout["g01"]["z_index"] == -1
    assert layout["g01"]["avoids_graphic_ids"] == ["g00"]
    assert len(track.overlays) == 2
    assert track.boxes[0][2:] == (
        track.overlays[0].left, track.overlays[0].top,
        track.overlays[0].width, track.overlays[0].height,
    )


def test_failed_compositor_keeps_previous_delivery_atomic(
    tmp_path: Path, monkeypatch
):
    import subprocess
    import montagewright.graphics as graphics

    source = tmp_path / "clean.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=360x640:d=3:r=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], check=True)
    destination = tmp_path / "final.mp4"
    destination.write_bytes(b"previous-good-delivery")
    original = graphics.subprocess.run

    def fail_only_the_compositor(command, *args, **kwargs):
        if "-filter_complex" in command:
            return subprocess.CompletedProcess(command, 1, "", "")
        return original(command, *args, **kwargs)

    monkeypatch.setattr(graphics.subprocess, "run", fail_only_the_compositor)
    plan = GraphicsPlan(facts=[fact()], cues=[cue(duration_seconds=2.0)])

    with pytest.raises(RuntimeError, match="compositor failed"):
        burn_graphics(source, plan, destination, work=tmp_path / "work")

    assert destination.read_bytes() == b"previous-good-delivery"
    assert not list(tmp_path.glob(".final-*.mp4"))


def test_auto_contrast_repairs_white_text_on_white_picture(tmp_path: Path):
    from PIL import Image

    plan = GraphicsPlan(
        facts=[fact(text="看得見的白字")],
        cues=[cue(
            template="editorial_rule", background="none",
            style=GraphicStyle(contrast_mode="auto"),
        )],
    )
    _, report = compile_graphic(
        plan.cues[0], plan, width=360, height=640,
        into=tmp_path / "card.png",
        frames=[Image.new("RGB", (360, 640), "white") for _ in range(3)],
    )

    assert report["fallback_plate"] is True
    assert report["contrast_ratio"] >= 4.5
    assert "dark_plate" in report["contrast_adjustments"]


def test_strict_unreadable_colours_fail_closed(tmp_path: Path):
    from PIL import Image

    plan = GraphicsPlan(
        facts=[fact(text="不能偷偷更改")],
        cues=[cue(
            template="editorial_rule", background="none",
            style=GraphicStyle(
                primary_color="#FFFFFF", contrast_mode="strict"
            ),
        )],
    )
    with pytest.raises(ValueError, match="contrast"):
        compile_graphic(
            plan.cues[0], plan, width=360, height=640,
            into=tmp_path / "locked.png",
            frames=[Image.new("RGB", (360, 640), "white") for _ in range(3)],
        )


def test_high_contrast_outline_can_remain_plate_free(tmp_path: Path):
    from PIL import Image

    plan = GraphicsPlan(
        facts=[fact(text="描邊字")],
        cues=[cue(
            template="editorial_rule", background="none",
            style=GraphicStyle(
                primary_color="#FFFFFF", stroke_color="#000000",
                stroke_width=5,
            ),
        )],
    )
    _, report = compile_graphic(
        plan.cues[0], plan, width=360, height=640,
        into=tmp_path / "outlined.png",
        frames=[Image.new("RGB", (360, 640), "white") for _ in range(3)],
    )

    assert report["fallback_plate"] is False
    assert report["contrast_ratio"] >= 4.5


def test_enlarged_card_stays_inside_safe_frame(tmp_path: Path):
    from montagewright.graphics import draw_graphic

    plan = GraphicsPlan(
        facts=[fact(text="大型標題")],
        cues=[cue(
            template="hero_center", position="upper_left",
            style=GraphicStyle(max_width_scale=1.25),
        )],
    )
    card = draw_graphic(
        plan.cues[0], plan, width=1080, height=1920,
        into=tmp_path / "wide.png",
    )

    assert card.left >= 0 and card.top >= 0
    assert card.left + card.width <= 1080
    assert card.top + card.height <= 1920


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


def test_old_graphics_json_gets_resolution_independent_transform_defaults():
    loaded = GraphicCue.model_validate({
        "graphic_id": "legacy", "kind": "product_name",
        "primary_fact_id": "name", "status": "draft",
    })

    assert loaded.transform.x == loaded.transform.y == 0.5
    assert loaded.transform.scale == 1.0
    assert loaded.transform.rotation_degrees == 0.0
    assert loaded.transform.locked is False


def test_manual_transform_places_scaled_rotated_card_by_its_centre(tmp_path: Path):
    from PIL import Image

    transformed = cue(
        position="manual", motion="none",
        transform={"x": 0.5, "y": 0.4, "scale": 0.75,
                   "rotation_degrees": 5},
        background="none",
    )
    plan = GraphicsPlan(facts=[fact(text="自由拖曳")], cues=[transformed])
    card, report = compile_graphic(
        transformed, plan, width=1080, height=1920,
        into=tmp_path / "manual.png",
        frames=[Image.new("RGB", (1080, 1920), "black") for _ in range(3)],
    )

    assert card.left + card.width / 2 == pytest.approx(1080 * 0.5, abs=1)
    assert card.top + card.height / 2 == pytest.approx(1920 * 0.4, abs=1)
    assert report["authored_transform"]["scale"] == 0.75
    assert report["authored_transform"]["rotation_degrees"] == 5
    assert Image.open(card.path).size == Image.open(card.text_mask_path).size
    assert Image.open(card.path).size == Image.open(card.backing_path).size


def test_locked_manual_transform_fails_instead_of_silently_moving(tmp_path: Path):
    from PIL import Image

    locked = cue(
        position="manual", motion="none",
        transform={"x": 0.0, "y": 0.0, "locked": True},
    )
    plan = GraphicsPlan(facts=[fact()], cues=[locked])

    with pytest.raises(ValueError, match="manual position"):
        compile_graphic(
            locked, plan, width=1080, height=1920,
            into=tmp_path / "locked-position.png",
            frames=[Image.new("RGB", (1080, 1920), "black") for _ in range(3)],
        )


def test_fixed_anchor_is_recomputed_after_scale_and_rotation(tmp_path: Path):
    from PIL import Image

    transformed = cue(
        position="upper_right", motion="none",
        transform={"scale": 0.7, "rotation_degrees": 12},
    )
    plan = GraphicsPlan(facts=[fact()], cues=[transformed])
    card, _ = compile_graphic(
        transformed, plan, width=1080, height=1920,
        into=tmp_path / "anchored.png",
        frames=[Image.new("RGB", (1080, 1920), "black") for _ in range(3)],
    )

    assert card.left + card.width == 1080 - round(1080 * 0.075)
    assert card.top == round(1920 * 0.09)


def test_tiny_rotated_glyph_never_skips_contrast_audit(tmp_path: Path):
    from PIL import Image

    tiny = cue(
        template="stat_badge", kind="callout", motion="none",
        background="none", transform={"scale": 0.25, "rotation_degrees": 15},
        style=GraphicStyle(
            primary_scale=0.55, primary_color="#FFFFFF",
            contrast_mode="strict",
        ),
    )
    plan = GraphicsPlan(facts=[fact(text="I")], cues=[tiny])

    with pytest.raises(ValueError, match="contrast"):
        compile_graphic(
            tiny, plan, width=360, height=640,
            into=tmp_path / "tiny.png",
            frames=[Image.new("RGB", (360, 640), "white") for _ in range(3)],
        )


def test_scaled_away_outline_cannot_claim_contrast_it_did_not_render(
    tmp_path: Path,
):
    from PIL import Image

    tiny = cue(
        template="stat_badge", kind="callout", motion="none",
        background="none", transform={"scale": 0.25},
        style=GraphicStyle(
            primary_scale=0.55, primary_color="#FFFFFF",
            stroke_color="#000000", stroke_width=2,
            contrast_mode="strict",
        ),
    )
    plan = GraphicsPlan(facts=[fact(text="I")], cues=[tiny])

    with pytest.raises(ValueError, match="contrast"):
        compile_graphic(
            tiny, plan, width=360, height=640,
            into=tmp_path / "no-physical-outline.png",
            frames=[Image.new("RGB", (360, 640), "white") for _ in range(3)],
        )


def test_uniform_scale_fails_instead_of_squashing_width_only(tmp_path: Path):
    from PIL import Image

    huge = cue(transform={"scale": 3.0}, motion="none")
    plan = GraphicsPlan(facts=[fact()], cues=[huge])

    with pytest.raises(ValueError, match="uniform scale"):
        compile_graphic(
            huge, plan, width=1080, height=1920,
            into=tmp_path / "huge.png",
            frames=[Image.new("RGB", (1080, 1920), "black") for _ in range(3)],
        )


def test_collision_overlap_is_symmetric_for_large_and_small_cards():
    from montagewright.graphics import _rect_overlap

    small = (100, 100, 20, 20)
    large = (0, 0, 240, 240)

    assert _rect_overlap(small, large) == 1.0
    assert _rect_overlap(large, small) == 1.0


def test_long_cue_contrast_samples_inside_actual_entrance():
    from montagewright.graphics import _layout_sample_shares

    long_title = cue(duration_seconds=30, motion="slide_left")
    first, middle, last = _layout_sample_shares(long_title)

    assert first * long_title.duration_seconds == pytest.approx(0.225)
    assert (middle, last) == (0.5, 0.97)


def test_v1_graphics_preserve_legacy_overlap_order():
    loaded = GraphicsPlan.model_validate({
        "version": "montagewright-graphics-v1",
        "facts": [fact().model_dump(mode="json")],
        "cues": [cue().model_dump(mode="json", exclude={"collision_policy"})],
    })

    assert loaded.version == "montagewright-graphics-v2"
    assert loaded.cues[0].collision_policy == "allow"


def test_legacy_explicit_colours_do_not_gain_silent_auto_fallback():
    loaded = GraphicStyle.model_validate({
        "primary_color": "#FFFFFF", "stroke_width": 4,
    })

    assert loaded.contrast_mode == "strict"


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
            "-f", "lavfi", "-i", "color=c=black:s=360x640:d=4:r=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(here / "out" / "deliverable.mp4"),
        ], check=True)
        payload = GraphicsPlan(
            facts=[fact(), fact("second", "A much longer second card")],
            cues=[
                cue(
                    status="draft", style=GraphicStyle(
                        preset="outlined", stroke_width=6,
                    ),
                ),
                cue(
                    graphic_id="g01", primary_fact_id="second",
                    status="draft", position="upper_right",
                ),
            ],
        ).model_dump(mode="json")
        client = TestClient(web.create_app())

        compiled = client.post(
            "/api/runs/r1/graphics-preview/g00", json=payload
        )
        second = client.post(
            "/api/runs/r1/graphics-preview/g01", json=payload
        )

        assert compiled.status_code == 200
        assert second.status_code == 200
        assert compiled.json()["url"] != second.json()["url"]
        assert client.get(compiled.json()["url"]).content != client.get(
            second.json()["url"]
        ).content
        assert compiled.json()["card_width"] > 0
        assert client.get(compiled.json()["url"]).headers["content-type"] == "image/png"

        joint = GraphicsPlan(
            facts=[fact(), fact("second", "Second")],
            cues=[
                cue(status="approved", position="auto"),
                cue(
                    graphic_id="g01", primary_fact_id="second",
                    status="draft", position="upper_right",
                ),
            ],
        )
        before = client.post(
            "/api/runs/r1/graphics-preview/g01",
            json=joint.model_dump(mode="json"),
        )
        joint = joint.model_copy(update={
            "cues": [joint.cues[0], joint.cues[1].model_copy(
                update={"status": "approved"}
            )],
        })
        after = client.post(
            "/api/runs/r1/graphics-preview/g01",
            json=joint.model_dump(mode="json"),
        )
        assert before.status_code == after.status_code == 200
        for graphic_id in ("g00", "g01"):
            assert before.json()["joint_layout"][graphic_id]["animation"] == (
                after.json()["joint_layout"][graphic_id]["animation"]
            )

        strict = GraphicsPlan(
            facts=[fact()], cues=[cue(
                status="draft", template="editorial_rule",
                background="none", style=GraphicStyle(
                    primary_color="#000000", contrast_mode="strict",
                ),
            )],
        )
        blocked = client.post(
            "/api/runs/r1/graphics-preview/g00",
            json=strict.model_dump(mode="json"),
        )
        assert blocked.status_code == 422
        assert "contrast" in blocked.json()["detail"]
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


def test_layout_evidence_uses_the_same_smoothstep_as_rendered_crop():
    from montagewright.reframe import interpolate_crop_keyframes

    crop = interpolate_crop_keyframes([
        {"at": 0.0, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"at": 1.0, "x": 1.0, "y": 0.0, "w": 1.0, "h": 1.0},
    ], 0.25)

    assert crop is not None
    assert crop["x"] == pytest.approx(0.15625)


def test_canvas_card_click_enters_text_edit_and_compare_reflows_overlay():
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "function editGraphicFromCanvas(id)" in page
    assert "editor.focus({preventScroll: true})" in page
    assert "editor.select()" in page
    assert "if (!moved) {\n      editGraphicFromCanvas(id);" in page
    assert "requestAnimationFrame(() => {\n    forgetPlacement(); drawCrop(); showBurnt(); showGraphicPreview();" in page
    assert "new ResizeObserver(() =>" in page


def test_graphics_track_click_seeks_to_the_point_that_was_clicked():
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "function graphicTrackTargetAtClientX(cue, clientX)" in page
    assert "const pointed = (clientX - reel.left) / scale;" in page
    assert "selectGraphic(index, graphicTrackTargetAtClientX(cue, next.clientX));" in page
    assert "if (Math.abs(next.clientX - startX) < 2) return;" in page
    assert "function selectGraphic(i, seekSeconds = null)" in page
    assert "const target = seekSeconds == null" in page


def test_web_graphics_show_copy_provenance_and_create_stable_anchors():
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    for label in (
        "Brief 原文", "Brief 候選", "畫面辨識", "語音逐字稿",
        "Gemini 草稿", "手動輸入",
    ):
        assert label in page
    assert page.count("anchor_selection_index: anchor?.index ?? null") == 2


def test_current_timeline_uses_the_renderers_cumulative_frame_clock(
    tmp_path: Path,
):
    import json
    import montagewright.webapp as web

    run = web.Run("r1", tmp_path / "r1")
    work = run.output / "work"
    work.mkdir(parents=True)
    (run.output / "report.json").write_text(json.dumps({
        "selection": {"shots": [{}, {}]},
    }), encoding="utf-8")
    (work / "current-timeline.json").write_text(json.dumps({
        "version": "montagewright-current-timeline-v1",
        "revision": 0,
        "output_fps": 30,
        "shots": [
            {"selection_index": 0, "in_seconds": 0, "seconds": .515},
            {"selection_index": 1, "in_seconds": 0, "seconds": .515},
        ],
    }), encoding="utf-8")

    current = web._current_timeline(run)

    assert current["shots"][0]["start_frame"] == 0
    assert current["shots"][0]["frame_count"] == 15
    assert current["shots"][0]["seconds"] == .5
    assert current["shots"][1]["start_frame"] == 15
    assert current["shots"][1]["frame_count"] == 16
    assert current["shots"][1]["seconds"] == pytest.approx(16 / 30)
    import inspect

    timeline_source = inspect.getsource(web.create_app)
    assert '"at": round(cursor, 3)' not in timeline_source
    assert '"seconds": round(seconds, 3)' not in timeline_source


def test_graphics_filter_timing_keeps_sub_frame_precision():
    import inspect
    import montagewright.graphics as graphics

    source = inspect.getsource(graphics.burn_graphics)
    assert "since:.9f" in source
    assert "until:.9f" in source
    assert "overlay.starts_seconds:.9f" in source
    assert "overlay.ends_seconds:.9f" in source


def test_saving_graphics_invalidates_every_previous_graphics_delivery(
    tmp_path: Path,
):
    import montagewright.webapp as web

    run = web.Run("r1", tmp_path / "r1")
    layout = run.output / "work" / "graphics-render" / "layout.json"
    layout.parent.mkdir(parents=True)
    layout.write_text("{}", encoding="utf-8")
    made = [
        run.output / "deliverable-graphics.mp4",
        run.output / "deliverable-graphics-subtitled.mp4",
    ]
    for path in made:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stale")

    web._invalidate_graphics_delivery(run)

    assert not layout.exists()
    assert all(not path.exists() for path in made)
