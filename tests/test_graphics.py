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
    draw_graphic,
    graphic_preset_defaults,
    minimum_read_seconds,
    validate_for_render,
    validate_brief_authority,
    resolve_auto_position,
    resolve_graphic_animation,
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


def test_new_surface_payload_uses_documented_auto_contrast_default():
    assert GraphicStyle(surface="glass").contrast_mode == "auto"


@pytest.mark.parametrize("text", [
    "AAAA\nI", "gypq\nHI", "中文測試\nAg", "I\ngypq\n中文",
])
def test_per_line_contrast_masks_exactly_cover_rendered_text(
    tmp_path: Path, text: str,
):
    from PIL import Image, ImageChops

    plan = GraphicsPlan(facts=[fact(text=text)], cues=[cue()])
    card = draw_graphic(
        plan.cues[0], plan, width=540, height=960,
        into=tmp_path / "multiline.png",
    )
    combined = Image.open(card.text_mask_path).convert("L")
    rebuilt = Image.new("L", combined.size, 0)
    for path in card.text_run_mask_paths:
        rebuilt = ImageChops.lighter(rebuilt, Image.open(path).convert("L"))

    assert ImageChops.difference(combined, rebuilt).getbbox() is None


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


def test_non_product_brief_uses_the_same_structural_card_rules():
    from montagewright.brief import parse_brief_markdown

    parsed = parse_brief_markdown('''午夜餐桌
一碗慢慢完成的湯

72°C
保持清澈

立即預約''')

    assert parsed.candidates[0].kind == "opening_title"
    assert parsed.candidates[1].template == "spec_stack"
    assert parsed.candidates[2].kind == "end_card"


def test_initial_selection_can_place_plain_brief_copy_only_as_a_draft():
    from montagewright.brief import initial_graphics_plan, parse_brief_markdown

    parsed = parse_brief_markdown("Galaxy Z Fold8 Ultra\n極致進化")
    candidate = parsed.candidates[0]
    plan = initial_graphics_plan(
        parsed,
        {
            "covered": [{
                "goal": "介紹產品",
                "shot_indexes": [1],
                "show_as_graphic": True,
                "graphic_candidate_id": candidate.candidate_id,
                "graphic_reason": "觀眾需要看到名稱",
            }],
        },
        shot_durations=[2.0, 3.0],
    )

    assert len(plan.cues) == 1
    assert plan.cues[0].status == "draft"
    assert plan.cues[0].anchor_clip_id == "k01"
    assert plan.cues[0].at_seconds == pytest.approx(2.2)
    assert all(f.source_kind == "brief_candidate" for f in plan.facts)
    assert all(not f.approved and f.approved_by is None for f in plan.facts)
    assert all(f.fact_id.startswith("brief_candidate.") for f in plan.facts)


def test_gemini_design_family_remains_freely_composable():
    from montagewright.brief import initial_graphics_plan, parse_brief_markdown

    parsed = parse_brief_markdown("Galaxy Z Fold8 Ultra\n極致進化")
    candidate = parsed.candidates[0]
    plan = initial_graphics_plan(
        parsed,
        {"covered": [{
            "goal": "介紹產品", "shot_indexes": [0],
            "show_as_graphic": True,
            "graphic_candidate_id": candidate.candidate_id,
            "graphic_design_family": "broadcast_info",
            "graphic_surface": "sticker",
            "graphic_motion": "fade",
            "graphic_composition": "overlap_subject",
            "graphic_shot_index": 0,
            "graphic_reason": "用資訊條語言，但改成貼紙並刻意疊產品",
        }]},
        shot_durations=[3.0],
    )

    made = plan.cues[0]
    assert made.style.preset == "broadcast_info"
    assert made.style.surface == "sticker"
    assert made.style.primary_color == "#FFFFFF"
    assert made.motion == "fade"
    assert made.composition == "overlap_subject"
    assert made.position == "auto"


def test_selection_contract_exposes_family_and_independent_overrides():
    from montagewright.planner import _selection_schema

    schema = _selection_schema(["s00"], graphic_candidate_ids=["brief.p00"])
    covered = schema["properties"]["covered"]["items"]
    required = set(covered["required"])
    properties = covered["properties"]

    assert {
        "graphic_design_family", "graphic_surface", "graphic_motion",
        "graphic_composition", "graphic_shot_index",
    } <= required
    assert "broadcast_info" in properties["graphic_design_family"]["enum"]
    assert "sticker" in properties["graphic_surface"]["enum"]
    assert "inherit" in properties["graphic_composition"]["enum"]
    assert "auto" in properties["graphic_composition"]["enum"]
    assert properties["graphic_candidate_id"]["enum"] == ["none", "brief.p00"]


def test_legacy_sparse_metadata_preset_keeps_old_pixels():
    made = GraphicCue(
        graphic_id="legacy", kind="callout", primary_fact_id="name",
        style={"preset": "tech_frame"},
    )

    assert made.background == "auto"
    assert made.style.stroke_width == 0
    assert made.style.preset == "tech_frame"


def test_preset_only_api_payload_is_materialised_by_backend():
    made = GraphicCue(
        graphic_id="family", kind="callout", primary_fact_id="name",
        style={"preset": "social_sticker", "surface": "outline"},
    )

    # Families style the authored content structure; they do not silently
    # replace a one-line callout with a different template.
    assert made.template == "editorial_rule"
    assert made.motion == "rise"
    assert made.style.surface == "outline"
    assert made.style.plate_color == "#FFFFFF"
    assert made.style.contrast_mode == "auto"


def test_approved_copy_cannot_use_the_server_candidate_namespace():
    from montagewright.brief import parse_brief_markdown

    with pytest.raises(ValueError, match="reserved copy_id prefix"):
        parse_brief_markdown('''```montagewright-approved-copy
{"version": 1, "items": [{"copy_id": "brief_candidate.fake", "text": "No"}]}
```''')


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

    assert report["fallback_plate"] is False
    assert report["contrast_ratio"] >= 4.5
    assert report["contrast_adjustments"] == ["white_text", "outline"]


def test_family_contrast_repair_never_erases_an_explicit_surface_override(
    tmp_path: Path,
):
    from PIL import Image

    plan = GraphicsPlan(
        facts=[fact(text="保留貼紙")],
        cues=[cue(
            template="stat_badge", background="plate",
            style=GraphicStyle(
                preset="cinematic_title", surface="sticker",
                primary_color="#FFFFFF", plate_color="#FFFFFF",
                plate_border_color="#FFFFFF", contrast_mode="auto",
            ),
        )],
    )
    _, report = compile_graphic(
        plan.cues[0], plan, width=360, height=640,
        into=tmp_path / "family-surface-override.png",
        frames=[Image.new("RGB", (360, 640), "white") for _ in range(3)],
    )

    assert "cinematic_scrim" not in report["contrast_adjustments"]
    assert report["authored_style"]["surface"] == "sticker"


def test_editorial_family_flips_palette_before_adding_an_outline(tmp_path: Path):
    from PIL import Image

    plan = GraphicsPlan(
        facts=[fact(text="安靜的章節")],
        cues=[cue(style=GraphicStyle(preset="editorial_minimal"))],
    )
    _, report = compile_graphic(
        plan.cues[0], plan, width=360, height=640,
        into=tmp_path / "editorial-dark.png",
        frames=[Image.new("RGB", (360, 640), "black") for _ in range(3)],
    )

    assert report["contrast_adjustments"] == ["light_editorial_palette"]


def test_cinematic_motion_override_keeps_a_real_travel_distance(tmp_path: Path):
    style_defaults, _ = graphic_preset_defaults("cinematic_title")
    moving = cue(
        motion="rise", style=GraphicStyle.model_validate({
            **style_defaults, "preset": "cinematic_title",
        }),
    )
    card = DrawnGraphic(tmp_path / "card.png", 100, 200, 240, 90)

    animation = resolve_graphic_animation(moving, card, 1080, 1920)

    assert animation["from_top"] > animation["settled_top"]


def test_sticker_shadow_stays_outside_two_line_content(tmp_path: Path):
    from PIL import Image

    primary = fact("primary", "TEST DECK")
    secondary = fact("secondary", "下一段內容")
    plan = GraphicsPlan(
        facts=[primary, secondary],
        cues=[cue(
            primary_fact_id="primary", secondary_fact_id="secondary",
            style=GraphicStyle(preset="social_sticker"),
        )],
    )
    _, report = compile_graphic(
        plan.cues[0], plan, width=360, height=640,
        into=tmp_path / "two-line-sticker.png",
        frames=[Image.new("RGB", (360, 640), "#AAAAAA") for _ in range(3)],
    )

    assert report["contrast_adjustments"] == []


def test_auto_contrast_fallback_does_not_reuse_destructive_outline(
    tmp_path: Path,
):
    from PIL import Image

    plan = GraphicsPlan(
        facts=[fact(text="4.1mm 纖薄機身")],
        cues=[cue(
            template="spec_stack", background="none",
            style=GraphicStyle(
                surface="outline", primary_color="#FFFFFF",
                stroke_color="#FFFFFF", stroke_width=12,
                contrast_mode="auto",
            ),
        )],
    )
    _, report = compile_graphic(
        plan.cues[0], plan, width=360, height=640,
        into=tmp_path / "fallback-outline.png",
        frames=[Image.new("RGB", (360, 640), "white") for _ in range(3)],
    )

    assert report["fallback_plate"] is False
    assert report["contrast_ratio"] >= 4.5
    assert report["contrast_adjustments"] == ["white_text", "outline"]
    assert report["authored_style"]["stroke_width"] == 12


def test_each_text_run_must_pass_contrast_independently(tmp_path: Path):
    from PIL import Image

    primary = fact("primary", "A" * 20)
    secondary = fact("secondary", "I")
    plan = GraphicsPlan(
        facts=[primary, secondary],
        cues=[cue(
            primary_fact_id="primary", secondary_fact_id="secondary",
            template="product_plate", background="plate",
            style=GraphicStyle(
                surface="split", primary_color="#FFFFFF",
                secondary_color="#FFE000", plate_color="#000000",
                surface_secondary_color="#FFE000", plate_alpha=255,
                contrast_mode="strict",
            ),
        )],
    )

    with pytest.raises(ValueError, match="contrast"):
        compile_graphic(
            plan.cues[0], plan, width=540, height=960,
            into=tmp_path / "split-low-secondary.png",
            frames=[Image.new("RGB", (540, 960), "white") for _ in range(3)],
        )


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


@pytest.mark.parametrize(
    "surface",
    [
        "solid", "pill", "split", "ribbon", "sticker", "highlight",
        "outline", "glass", "editorial",
    ],
)
def test_surface_treatments_render_with_production_pixels(
    tmp_path: Path, surface: str,
):
    from PIL import Image
    from montagewright.graphics import draw_graphic

    plan = GraphicsPlan(
        brand=BrandKit(plate="#171A22", accent="#F4B942"),
        facts=[fact("name", "人物專訪"), fact("deck", "關於選擇與改變")],
        cues=[],
    )
    card = draw_graphic(
        cue(
            secondary_fact_id="deck", template="editorial_rule",
            position="upper_left", background="none",
            style=GraphicStyle(
                surface=surface, surface_secondary_color="#F4B942",
                plate_color="#171A22", plate_alpha=220,
                plate_border_width=2, corner_radius=18,
            ),
        ),
        plan, width=360, height=640, into=tmp_path / f"{surface}.png",
    )

    pixels = Image.open(card.path).convert("RGBA")
    assert pixels.getbbox() is not None
    assert card.backing_path is not None and card.backing_path.exists()


def test_split_surface_accepts_large_border_and_zero_padding(tmp_path: Path):
    """Every schema-valid family/surface combination must remain renderable."""
    from montagewright.graphics import draw_graphic

    plan = GraphicsPlan(
        facts=[fact("name", "自由組合"), fact("deck", "粗框與零內距")],
        cues=[],
    )
    card = draw_graphic(
        cue(
            secondary_fact_id="deck", template="stat_badge",
            position="center", background="plate",
            style=GraphicStyle(
                surface="split", corner_radius=18,
                padding_x=0, padding_y=0, plate_border_width=24,
            ),
        ),
        plan, width=360, height=640, into=tmp_path / "split-boundary.png",
    )

    assert card.path.exists()
    assert card.width > 0 and card.height > 0


def test_legacy_template_surface_keeps_the_old_plate_pixels(tmp_path: Path):
    from montagewright.graphics import draw_graphic

    plan = GraphicsPlan(facts=[fact()], cues=[])
    legacy = draw_graphic(
        cue(background="plate", style=GraphicStyle()), plan,
        width=360, height=640, into=tmp_path / "legacy.png",
    )
    explicit = draw_graphic(
        cue(background="plate", style=GraphicStyle(surface="template")), plan,
        width=360, height=640, into=tmp_path / "explicit.png",
    )

    assert legacy.path.read_bytes() == explicit.path.read_bytes()


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

        future = dict(payload)
        future["revision"] = 9
        assert client.put(
            "/api/runs/r1/graphics-track", json=future
        ).status_code == 409
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


def test_web_preview_uses_the_production_card_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
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
        preview_file = client.get(compiled.json()["url"])
        assert preview_file.headers["content-type"] == "image/png"
        assert "immutable" in preview_file.headers["cache-control"]

        # Cached pixels are durable, but the run alias in the original URL
        # is not. Reopening the same output after a server restart must
        # project the current id instead of returning a broken old route.
        metadata_file = next(
            path for path in
            (here / "out" / "work" / "graphics-preview").glob("*.json")
            if "g00" in json.loads(path.read_text(encoding="utf-8"))
            .get("joint_layout", {})
        )
        cached = json.loads(metadata_file.read_text(encoding="utf-8"))
        if cached.get("url"):
            cached["url"] = cached["url"].replace(
                "/runs/r1/", "/runs/stale/"
            )
        for item in cached["joint_layout"].values():
            item["url"] = item["url"].replace("/runs/r1/", "/runs/stale/")
        metadata_file.write_text(json.dumps(cached), encoding="utf-8")
        reopened = client.post(
            "/api/runs/r1/graphics-preview/g00", json=payload
        )
        assert reopened.status_code == 200
        assert reopened.json()["url"].startswith("/api/runs/r1/")
        assert client.get(reopened.json()["url"]).status_code == 200

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
        problem = blocked.json()["detail"]
        assert problem["error_code"] == "GRAPHIC_CONTRAST_FAILED"
        assert problem["field"] == "cues.g00.style.contrast_mode"
        assert problem["suggested_patch"] == {"contrast_mode": "auto"}

        # A complete metadata + PNG pair is a direct cache hit. The production
        # compiler is not invoked again for identical content.
        def should_not_compile(*args, **kwargs):
            raise AssertionError("a complete preview cache should be reused")

        monkeypatch.setattr(
            "montagewright.graphics.compile_graphic", should_not_compile
        )
        cache_hit = client.post(
            "/api/runs/r1/graphics-preview/g00", json=payload
        )
        assert cache_hit.status_code == 200

        # A direct immutable URL must never bless a truncated file as a PNG.
        cached_png = (
            here / "out" / "work" / "graphics-preview"
            / Path(cache_hit.json()["url"]).name
        )
        cached_png.write_bytes(b"partial")
        corrupt = client.get(cache_hit.json()["url"])
        assert corrupt.status_code == 503
        assert corrupt.json()["detail"]["error_code"] == (
            "PREVIEW_CACHE_CORRUPT"
        )

        monkeypatch.setattr(
            web, "_video_display_size",
            lambda _picture: (_ for _ in ()).throw(OSError("busy disk")),
        )
        unavailable = client.post(
            "/api/runs/r1/graphics-preview/g00", json=payload
        )
        assert unavailable.status_code == 503
        assert unavailable.json()["detail"]["error_code"] == (
            "PREVIEW_SOURCE_UNAVAILABLE"
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

        client = TestClient(web.create_app())
        loaded = client.get("/api/runs/r1/graphics-track")

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
        candidate_fact = CopyFact(
            fact_id="brief.p01.primary", exact_text="Galaxy AI",
            source_kind="brief_candidate", source_reference="/paragraphs/1",
            source_sha256="abc", allowed_kinds=["feature"], approved=False,
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
            "facts": [candidate_fact.model_dump(mode="json")],
            "instructions": [],
        }), encoding="utf-8")

        client = TestClient(web.create_app())
        loaded = client.get("/api/runs/r1/graphics-track")

        assert loaded.status_code == 200
        assert loaded.json()["brief_candidates"][0]["template"] == "center_stack"
        assert loaded.json()["brief_sha256"] == "abc"
        assert loaded.json()["facts"][0]["source_kind"] == "brief_candidate"
    finally:
        web.RUNS_ROOT = was
        web.RUNS.pop("r1", None)


def test_generic_web_save_cannot_forge_an_evidence_source(tmp_path: Path):
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
        forged = GraphicsPlan(facts=[CopyFact(
            fact_id="fake-ocr", exact_text="Not really on screen",
            source_kind="onscreen", source_reference="client-claim",
        )])

        saved = TestClient(web.create_app()).put(
            "/api/runs/r1/graphics-track",
            json=forged.model_dump(mode="json"),
        )

        assert saved.status_code == 422
        assert "不能由一般存檔宣稱" in saved.json()["detail"]
    finally:
        web.RUNS_ROOT = was
        web.RUNS.pop("r1", None)


def test_legacy_unverified_provenance_is_downgraded_on_read(tmp_path: Path):
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
        legacy = GraphicsPlan(facts=[CopyFact(
            fact_id="old-ocr", exact_text="Unverified legacy label",
            source_kind="onscreen", source_reference="old-client",
        )])
        (work / "graphics.json").write_text(
            legacy.model_dump_json(), encoding="utf-8"
        )

        client = TestClient(web.create_app())
        loaded = client.get("/api/runs/r1/graphics-track")

        assert loaded.status_code == 200
        assert loaded.json()["facts"][0]["source_kind"] == "user"
        assert loaded.json()["facts"][0]["source_reference"] == "legacy-unverified"
        assert not loaded.json()["facts"][0]["approved"]
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

        client = TestClient(web.create_app())
        loaded = client.get("/api/runs/r1/graphics-track")

        assert loaded.status_code == 200
        assert loaded.json()["brief_candidates"][0]["primary_text"] == "Galaxy Z Fold8"
        assert loaded.json()["brief_instructions"][0]["kind"] == "editorial"
        payload = loaded.json()
        saved = client.put("/api/runs/r1/graphics-track", json={
            key: payload[key]
            for key in ("version", "revision", "brand", "facts", "cues")
        })
        assert saved.status_code == 200
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


def test_canvas_card_selects_then_double_click_enters_inline_text_edit():
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "function beginGraphicInlineEdit(event, id)" in page
    assert "editor.className = 'graphic-inline-editor'" in page
    assert "inspectorEditor.value = editor.value;" in page
    assert "now - graphicLastCanvasClick.at < 420" in page
    assert "beginGraphicInlineEdit(next, id);" in page
    assert "requestAnimationFrame(() => {\n    forgetPlacement(); drawCrop(); showBurnt(); showGraphicPreview();" in page
    assert "new ResizeObserver(() =>" in page


def test_web_prewarms_graphics_before_the_playhead_reaches_them():
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "scheduleGraphicPreviewPrewarm();" in page
    assert "const approved = graphicsPlan.cues.filter(one => one.status === 'approved')" in page
    assert "await requestGraphicPreview(cue);" in page
    assert "await image.decode();" in page
    assert "graphicPreviewPixelPromises" in page
    assert "error_placeholder: one.status === 'approved'" in page
    assert "字卡需要調整，無法產生精準預覽" in page


def test_layout_evidence_frames_are_reused_across_style_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from io import BytesIO
    from types import SimpleNamespace
    from PIL import Image
    from montagewright.graphics import _layout_frames

    picture = tmp_path / "picture.mp4"
    picture.write_bytes(b"stable picture fingerprint")
    encoded = BytesIO()
    Image.new("RGB", (32, 18), "#556677").save(encoded, "PNG")
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=encoded.getvalue())

    monkeypatch.setattr("montagewright.graphics.subprocess.run", fake_run)
    cache = tmp_path / "frames"
    first = _layout_frames(picture, cue(), cache_dir=cache)
    second = _layout_frames(picture, cue(), cache_dir=cache)

    assert len(first) == len(second) == 3
    assert len(calls) == 3
    assert len(list(cache.glob("*.png"))) == 3


def test_system_font_catalog_is_not_queried_for_every_fitting_size(
    monkeypatch: pytest.MonkeyPatch,
):
    from types import SimpleNamespace
    from montagewright import subtitles

    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="/fonts/one.ttf\n", returncode=0)

    subtitles._asked_of_the_system.cache_clear()
    monkeypatch.setattr(subtitles.subprocess, "run", fake_run)
    try:
        assert subtitles._asked_of_the_system("zh-tw") == ["/fonts/one.ttf"]
        assert subtitles._asked_of_the_system("zh-tw") == ["/fonts/one.ttf"]
        assert len(calls) == 1
    finally:
        subtitles._asked_of_the_system.cache_clear()


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


def test_missing_graphics_track_does_not_leave_output_locked(tmp_path: Path):
    from fastapi.testclient import TestClient
    import montagewright.webapp as web
    from montagewright.release import acquire_output_lease

    run = web.Run("missing-graphics", tmp_path / "run")
    run.output.mkdir(parents=True)
    web.RUNS[run.run_id] = run
    try:
        response = TestClient(web.create_app()).post(
            f"/api/runs/{run.run_id}/burn-graphics"
        )
        assert response.status_code == 404
        lease = acquire_output_lease(run.output)
        lease.release()
    finally:
        web.RUNS.pop(run.run_id, None)


@pytest.mark.parametrize(
    "family",
    [
        "clean", "outlined", "soft_shadow", "colour_label", "tech_frame",
        "bold_pop", "editorial_minimal", "youtube_pop", "magazine_story",
        "social_sticker", "broadcast_info", "cinematic_title",
        "sports_energy", "soft_lifestyle",
    ],
)
@pytest.mark.parametrize("width,height", [(360, 640), (640, 360)])
def test_every_family_compiles_a_scaled_multiline_spec_card(
    tmp_path: Path, family: str, width: int, height: int,
):
    """A legal family switch must not trip on resampled outline evidence."""

    from PIL import Image

    style_defaults, cue_defaults = graphic_preset_defaults(family)
    style = GraphicStyle.model_validate({
        **style_defaults,
        "preset": family,
        "contrast_mode": "auto",
    })
    made = cue(
        kind="feature",
        secondary_fact_id="details",
        template="spec_stack",
        position=str(cue_defaults.get("position") or "auto"),
        composition=str(cue_defaults.get("composition") or "auto"),
        background=str(cue_defaults.get("background") or "auto"),
        motion=str(cue_defaults.get("motion") or "rise"),
        transform={"scale": 0.9},
        style=style,
    )
    plan = GraphicsPlan(
        facts=[
            fact("name", "Galaxy Watch Ultra2"),
            fact("details", "EN13319 國際潛水標準認證\n40m"),
        ],
        cues=[made],
    )
    card, report = compile_graphic(
        made, plan, width=width, height=height,
        into=tmp_path / f"{family}-{width}x{height}.png",
        frames=[Image.new("RGB", (width, height), "#78909C") for _ in range(3)],
    )

    assert report["contrast_ratio"] >= 4.5
    assert card.foreground_path is not None and card.foreground_path.exists()


@pytest.mark.parametrize("family", ["sports_energy", "broadcast_info"])
def test_automatic_wide_family_uses_fade_when_slide_has_no_safe_travel(
    tmp_path: Path, family: str,
):
    from PIL import Image

    style_defaults, cue_defaults = graphic_preset_defaults(family)
    style = GraphicStyle.model_validate({
        **style_defaults,
        "preset": family,
        "contrast_mode": "auto",
    })
    made = cue(
        kind="feature", secondary_fact_id="details",
        template="center_stack", position="auto",
        composition=str(cue_defaults.get("composition") or "auto"),
        background=str(cue_defaults.get("background") or "auto"),
        motion=str(cue_defaults["motion"]), style=style,
    )
    plan = GraphicsPlan(
        facts=[fact("name", "全系列支援"), fact("details", "Galaxy AI")],
        cues=[made],
    )
    _, report = compile_graphic(
        made, plan, width=360, height=640,
        into=tmp_path / f"{family}.png",
        frames=[Image.new("RGB", (360, 640), "#4D5560") for _ in range(3)],
    )

    animation = report["animation"]
    assert animation["requested_motion"] in {"slide_left", "slide_right"}
    assert animation["resolved_motion"] == "fade"
    assert animation["resolved_motion_distance"] == 0


@pytest.mark.parametrize("guard", ["strict", "locked"])
def test_authored_motion_guard_is_not_silently_repaired(
    tmp_path: Path, guard: str,
):
    from PIL import Image

    style_defaults, cue_defaults = graphic_preset_defaults("sports_energy")
    style = GraphicStyle.model_validate({
        **style_defaults,
        "preset": "sports_energy",
        "contrast_mode": "strict" if guard == "strict" else "auto",
    })
    made = cue(
        kind="feature", secondary_fact_id="details",
        template="center_stack", position="auto",
        composition=str(cue_defaults.get("composition") or "auto"),
        background=str(cue_defaults.get("background") or "auto"),
        motion=str(cue_defaults["motion"]),
        transform={"locked": guard == "locked"}, style=style,
    )
    plan = GraphicsPlan(
        facts=[fact("name", "全系列支援"), fact("details", "Galaxy AI")],
        cues=[made],
    )

    with pytest.raises(ValueError, match="entrance motion cannot fit"):
        compile_graphic(
            made, plan, width=360, height=640,
            into=tmp_path / f"guard-{guard}.png",
            frames=[Image.new("RGB", (360, 640), "#4D5560") for _ in range(3)],
        )


def test_atomic_png_publish_keeps_previous_file_after_encoder_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    from PIL import Image
    from montagewright.graphics import _save_png_atomic

    destination = tmp_path / "card.png"
    destination.write_bytes(b"previous complete png")

    def fail_after_partial_write(self, target, *args, **kwargs):
        Path(target).write_bytes(b"partial")
        raise OSError("encoder stopped")

    monkeypatch.setattr(Image.Image, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="encoder stopped"):
        _save_png_atomic(Image.new("RGBA", (4, 4)), destination)

    assert destination.read_bytes() == b"previous complete png"
    assert not list(tmp_path.glob(".card-*.png"))


def test_web_graphics_preview_distinguishes_transient_validation_and_stale_pixels():
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "const kind = got.status === 422" in page
    assert "? 'validation' : 'transient'" in page
    assert "graphicPreviewRetryAttempts" in page
    assert "graphic-preview-stale-badge" in page
    assert "顯示上一次有效預覽" in page
    assert "預覽暫時無法更新" in page
    assert "預覽需要調整" in page


def test_web_graphics_inspector_progressively_discloses_advanced_controls():
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert 'id="graphic-readiness"' in page
    assert "graphicFamilyCards(style.preset, 4)" in page
    assert "<summary>進階內容與來源</summary>" in page
    assert "<summary>進階設計</summary>" in page
    assert "<summary>進階動畫設定</summary>" in page
    assert 'data-graphic-placement="auto"' in page
    assert 'data-graphic-placement="manual"' in page
    assert 'id="export-graphics"' in page
    assert "產生成片＋字卡" in page
    assert "畫面上單擊選取、雙擊可直接修改主文字。" in page


def test_web_graphics_family_and_auto_restore_reset_safe_geometry():
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "function resetGraphicPlacementControls()" in page
    assert "$('graphic-transform-x').value = '.5';" in page
    assert "$('graphic-transform-y').value = '.5';" in page
    assert "$('graphic-transform-scale').value = '1';" in page
    assert "$('graphic-transform-rotation').value = '0';" in page
    assert "if (cue.transform?.locked)" in page
    assert "Families have different intrinsic dimensions" in page
