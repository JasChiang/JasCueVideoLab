"""Render a small, model-free catalogue with the production graphics path."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from montagewright.graphics import (
    BrandKit,
    CopyFact,
    GraphicCue,
    GraphicStyle,
    GraphicsPlan,
    burn_graphics,
    graphic_preset_defaults,
)


SAMPLES = [
    {
        "name": "極簡編輯 · Editorial minimal",
        "at": 0.15,
        "source_at": 34.0,
        "kind": "opening_title",
        "template": "hero_center",
        "family": "editorial_minimal",
        "position": "top",
        "primary": "山徑上的午後",
        "secondary": "TRAVEL DIARY · 2026",
    },
    {
        "name": "YouTube 強調 · 可讀重點",
        "at": 4.65,
        "source_at": 34.0,
        "kind": "callout",
        "template": "editorial_rule",
        "family": "youtube_pop",
        "primary": "週末去哪玩？",
        "secondary": "3 個不踩雷的私房景點",
    },
    {
        "name": "雜誌專題 · Magazine story",
        "at": 9.15,
        "source_at": 45.0,
        "kind": "chapter",
        "template": "center_stack",
        "family": "magazine_story",
        "primary": "城市觀察",
        "secondary": "人、空間與日常選擇",
    },
    {
        "name": "社群貼紙 · Social sticker",
        "at": 13.65,
        "source_at": 45.0,
        "kind": "callout",
        "template": "editorial_rule",
        "family": "social_sticker",
        "primary": "今天就出發！",
        "secondary": "別等完美時機",
    },
    {
        "name": "資訊下標 · Information lower third",
        "at": 18.15,
        "source_at": 51.0,
        "kind": "product_name",
        "template": "product_plate",
        "family": "broadcast_info",
        "primary": "Galaxy Z Fold8",
        "secondary": "4.1mm 纖薄機身 · IP68",
    },
    {
        "name": "電影標題 · Cinematic title",
        "at": 22.65,
        "source_at": 55.0,
        "kind": "opening_title",
        "template": "hero_center",
        "family": "cinematic_title",
        "primary": "重啟之後",
        "secondary": "A NEW BEGINNING",
    },
    {
        "name": "運動娛樂 · Sports energy",
        "at": 27.15,
        "source_at": 65.0,
        "kind": "callout",
        "template": "editorial_rule",
        "family": "sports_energy",
        "primary": "最後 10 分鐘",
        "secondary": "逆轉的關鍵",
    },
    {
        "name": "柔和生活 · Soft lifestyle",
        "at": 31.65,
        "source_at": 85.0,
        "kind": "chapter",
        "template": "center_stack",
        "family": "soft_lifestyle",
        "primary": "慢下來生活",
        "secondary": "留一點時間給自己",
    },
]


def run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "command failed")


def fact(fact_id: str, text: str) -> CopyFact:
    return CopyFact(
        fact_id=fact_id,
        exact_text=text,
        source_kind="user",
        approved=True,
        approved_by="human_review",
    )


def make_base(picture: Path, destination: Path) -> None:
    trims = []
    labels = []
    for index, sample in enumerate(SAMPLES):
        start = sample["source_at"]
        trims.append(
            f"[0:v]trim=start={start}:duration=4.5,setpts=PTS-STARTPTS," 
            f"scale=540:960:force_original_aspect_ratio=increase," 
            f"crop=540:960,fps=30[v{index}]"
        )
        labels.append(f"[v{index}]")
    filter_graph = ";".join(trims) + ";" + "".join(labels) + (
        f"concat=n={len(SAMPLES)}:v=1:a=0,format=yuv420p[out]"
    )
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(picture), "-filter_complex", filter_graph,
        "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "20",
        "-preset", "veryfast", "-movflags", "+faststart", str(destination),
    ])


def make_plan() -> GraphicsPlan:
    facts: list[CopyFact] = []
    cues: list[GraphicCue] = []
    for index, sample in enumerate(SAMPLES):
        primary_id = f"showcase.{index}.primary"
        secondary_id = f"showcase.{index}.secondary"
        facts += [fact(primary_id, sample["primary"]), fact(secondary_id, sample["secondary"])]
        style_defaults, cue_defaults = graphic_preset_defaults(sample["family"])
        cues.append(GraphicCue(
            graphic_id=f"showcase.{index}",
            kind=sample["kind"],
            primary_fact_id=primary_id,
            secondary_fact_id=secondary_id,
            at_seconds=sample["at"],
            duration_seconds=4.15,
            template=sample["template"],
            position=sample.get("position", cue_defaults.get("position", "auto")),
            composition=cue_defaults.get("composition", "auto"),
            background=cue_defaults.get("background", "auto"),
            motion=cue_defaults.get("motion", "rise"),
            transform={"rotation_degrees": sample.get("rotation", 0.0)},
            style=GraphicStyle.model_validate({
                **style_defaults, "preset": sample["family"],
                "contrast_mode": "auto",
            }),
            status="approved",
        ))
    return GraphicsPlan(
        brand=BrandKit(
            foreground="#FFFFFF", secondary="#E7E8EC", accent="#F4B942",
            plate="#12151B", plate_alpha=215, corner_radius=0.18,
        ),
        facts=facts,
        cues=cues,
    )


def extract_frame(video: Path, at: float, into: Path) -> None:
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1", str(into),
    ])


def make_sheet(video: Path, destination: Path, frames: Path) -> None:
    frames.mkdir(parents=True, exist_ok=True)
    panels = []
    for index, sample in enumerate(SAMPLES):
        frame = frames / f"sheet-{index}.png"
        extract_frame(video, sample["at"] + 2.0, frame)
        panel = Image.open(frame).convert("RGB").resize((270, 480), Image.Resampling.LANCZOS)
        panels.append((panel, sample["name"]))
    rows = (len(panels) + 2) // 3
    sheet = Image.new("RGB", (870, 58 + rows * 500), "#171717")
    draw = ImageDraw.Draw(sheet)
    showcase_font = "/System/Library/Fonts/STHeiti Medium.ttc"
    face = (
        ImageFont.truetype(showcase_font, size=15)
        if Path(showcase_font).exists() else ImageFont.load_default(size=15)
    )
    title = (
        ImageFont.truetype(showcase_font, size=22)
        if Path(showcase_font).exists() else ImageFont.load_default(size=22)
    )
    draw.text((30, 18), "MontageWright - production graphics showcase", fill="white", font=title)
    for index, (panel, label) in enumerate(panels):
        x = 30 + (index % 3) * 280
        y = 58 + (index // 3) * 500
        sheet.paste(panel, (x, y))
        draw.text((x + 3, y + 484), label, fill="#E5E5E5", font=face)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("picture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = args.output / "showcase-base.mp4"
    video = args.output / "graphics-showcase.mp4"
    make_base(args.picture, base)
    burn_graphics(base, make_plan(), video, work=args.output / "work")
    make_sheet(video, args.output / "graphics-showcase.png", args.output / "frames")
    print(video)
    print(args.output / "graphics-showcase.png")


if __name__ == "__main__":
    main()
