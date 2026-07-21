#!/usr/bin/env python3
"""Extract exact local seed frames for the EfficientTAM fixture set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from jascue_video_lab.media import extract_frame
from jascue_video_lab.storage import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        required=True,
        help="JSON with fixtures containing fixture_id, clip_run, and event_id.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resolve_local(path: str) -> Path:
    candidate = Path(path).expanduser()
    return (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    selection_path = args.selection_manifest.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    fixtures = read_json(selection_path)["fixtures"]
    rows = []
    for fixture in fixtures:
        fixture_id = fixture["fixture_id"]
        event_id = fixture["event_id"]
        clip_run = resolve_local(fixture["clip_run"])
        card = read_json(clip_run / "gemini/clip-card/clip_card.json")
        timeline = read_json(clip_run / "derived-timeline.json")
        source = Path(read_json(clip_run / "private-source.json")["path"])
        card_event = next(item for item in card["events"] if item["event_id"] == event_id)
        derived_event = next(item for item in timeline["events"] if item["event_id"] == event_id)
        requested_ms = derived_event.get("recommended_keyframe_ms")
        if requested_ms is None:
            requested_ms = (derived_event["start_ms"] + derived_event["end_ms"]) // 2
        frame_path = output / "seed-frames" / f"{fixture_id}.png"
        frame = extract_frame(source, int(requested_ms), frame_path)
        rows.append(
            {
                "fixture_id": fixture_id,
                "clip_run": str(clip_run.resolve()),
                "event_id": event_id,
                "event_label": card_event["label"],
                "event_start_ms": derived_event["start_ms"],
                "event_end_ms": derived_event["end_ms"],
                "source_path": str(source.resolve()),
                "frame_path": str(frame_path.resolve()),
                "frame": frame.model_dump(mode="json"),
            }
        )
    write_json(output / "fixture-frames.json", {"fixtures": rows})

    thumb_w, thumb_h = 640, 360
    label_h = 64
    sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + label_h) * 4), "#0d1117")
    draw = ImageDraw.Draw(sheet)
    title_font = font(24)
    detail_font = font(18)
    for index, row in enumerate(rows):
        image = Image.open(row["frame_path"]).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (index % 2) * thumb_w
        y = (index // 2) * (thumb_h + label_h)
        sheet.paste(image, (x + (thumb_w - image.width) // 2, y))
        draw.text(
            (x + 12, y + thumb_h + 6),
            row["fixture_id"],
            fill="#e6edf3",
            font=title_font,
        )
        draw.text(
            (x + 12, y + thumb_h + 35),
            f"{row['event_label']}  frame={row['frame']['frame_time_ms']} ms",
            fill="#9da7b3",
            font=detail_font,
        )
    sheet_path = output / "seed-frame-contact-sheet.jpg"
    sheet.save(sheet_path, quality=92, optimize=True)
    print(json.dumps({"fixtures": len(rows), "contact_sheet": str(sheet_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
