import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from montagewright.executor import CropBox, RenderPlan, Segment, Source
from montagewright.graphics import (
    BrandKit,
    CopyFact,
    DrawnGraphic,
    GraphicCue,
    GraphicsPlan,
    render_graphics_overlay,
    resolve_graphic_animation,
)
from montagewright.grounding import BeatGrid, Cue
from montagewright.timeline import to_fcpxml, to_xmeml


def _plan() -> GraphicsPlan:
    return GraphicsPlan(
        brand=BrandKit(),
        facts=[CopyFact(
            fact_id="name", exact_text="Chapter One", source_kind="user",
            approved=True, approved_by="human_review",
        )],
        cues=[GraphicCue(
            graphic_id="g00", kind="chapter", primary_fact_id="name",
            at_seconds=.5, duration_seconds=2.0, template="center_stack",
            position="center", motion="fade", status="approved",
        )],
    )


def test_music_sync_lands_the_end_of_the_entrance_on_a_local_accent(tmp_path: Path):
    cue = _plan().cues[0].model_copy(update={
        "at_seconds": 1.0, "duration_seconds": 3.0,
        "music_sync": "accent",
    })
    card = DrawnGraphic(tmp_path / "card.png", 20, 30, 100, 50)
    grid = BeatGrid(
        bpm=120, meter=4, duration_seconds=10,
        cues=(Cue("accent-1", 1.5, "accent", .9),),
    )

    made = resolve_graphic_animation(
        cue, card, 360, 640, beat_grid=grid, output_fps=30,
        timeline_duration=10,
    )

    assert made["sync_applied"] is True
    assert made["sync_cue_id"] == "accent-1"
    assert made["start_seconds"] + made["enter_seconds"] == pytest.approx(1.5)
    assert made["end_seconds"] - made["start_seconds"] == pytest.approx(3.0)


def test_music_sync_does_not_jump_to_a_distant_event(tmp_path: Path):
    cue = _plan().cues[0].model_copy(update={"music_sync": "downbeat"})
    card = DrawnGraphic(tmp_path / "card.png", 20, 30, 100, 50)
    grid = BeatGrid(
        bpm=120, meter=4, duration_seconds=10,
        cues=(Cue("far", 4.0, "downbeat", 1.0),),
    )

    made = resolve_graphic_animation(cue, card, 360, 640, beat_grid=grid)

    assert made["sync_applied"] is False
    assert made["start_seconds"] == pytest.approx(.5)


def test_transparent_graphics_overlay_is_a_real_alpha_movie(tmp_path: Path):
    picture = tmp_path / "picture.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=0x243040:s=360x640:d=3:r=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(picture),
    ], check=True)

    overlay = render_graphics_overlay(
        picture, _plan(), tmp_path / "graphics-overlay.mov",
        work=tmp_path / "work",
    )
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,duration",
        "-of", "json", str(overlay),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]

    assert stream["codec_name"] == "prores"
    assert stream["pix_fmt"].startswith("yuva444p")
    assert float(stream["duration"]) == pytest.approx(3.0, abs=.04)


def test_both_nle_formats_place_the_alpha_track_above_picture(tmp_path: Path):
    original = tmp_path / "A.mp4"
    overlay = tmp_path / "graphics-overlay.mov"
    original.touch()
    overlay.touch()
    source = Source(
        source_id="A", path=original, duration_seconds=4,
        width=1920, height=1080,
    )
    plan = RenderPlan(project_id="p", segments=[Segment(
        clip_id="k00", source=source, in_seconds=0, out_seconds=3,
        crop=CropBox(x=0, y=0, width=1, height=1),
    )])

    xmeml = ET.fromstring(to_xmeml(
        plan, {}, name="p", width=1080, height=1920, graphics=overlay,
    ))
    fcpxml = ET.fromstring(to_fcpxml(
        plan, {}, name="p", width=1080, height=1920, graphics=overlay,
    ))

    video_tracks = xmeml.find("./sequence/media/video").findall("track")
    assert len(video_tracks) == 2
    assert video_tracks[1].findtext("clipitem/name") == "graphics-overlay"
    layer = next(
        item for item in fcpxml.iter("asset-clip")
        if item.get("name") == "graphics-overlay"
    )
    assert layer.get("lane") == "1"
    assert layer.get("videoRole") == "titles"


def test_gemini_graphic_intent_materialises_music_sync_without_owning_pixels():
    from montagewright.brief import initial_graphics_plan, parse_brief_markdown

    document = parse_brief_markdown("Galaxy Z Fold8\n極致輕薄")
    selection = {"covered": [{
        "goal": "introduce product", "shot_indexes": [0],
        "show_as_graphic": True, "graphic_candidate_id": "brief.p00",
        "graphic_design_family": "cinematic_title",
        "graphic_surface": "inherit", "graphic_motion": "fade",
        "graphic_music_sync": "downbeat",
        "graphic_composition": "negative_space", "graphic_shot_index": 0,
        "graphic_reason": "land the reveal on the bar",
    }]}

    made = initial_graphics_plan(document, selection, shot_durations=[3.0])

    assert made.cues[0].music_sync == "downbeat"
    assert made.cues[0].position == "auto"
    assert made.cues[0].status == "draft"


def test_web_inspector_exposes_the_same_music_sync_choices():
    page = (
        Path(__file__).parents[1] / "src" / "montagewright" / "web" / "index.html"
    ).read_text(encoding="utf-8")

    assert 'id="graphic-music-sync"' in page
    assert "['none','不吸附']" in page
    assert "['accent','進場完成對齊重音']" in page
    assert "['downbeat','進場完成對齊小節第一拍']" in page
