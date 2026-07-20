from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .geometry import normalized_to_pixels
from .models import GroundingProposal


COLORS = ["#ff3155", "#00c2ff", "#ffd43b", "#7cfc6b", "#c77dff"]
_CJK_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)


def _overlay_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _CJK_FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def draw_grounding_overlay(
    frame_path: Path, proposal: GroundingProposal, output_path: Path
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(frame_path).convert("RGB") as source:
        image = source.copy()
    if image.size != (proposal.source_width, proposal.source_height):
        raise ValueError(
            f"frame size {image.size} differs from proposal dimensions "
            f"{proposal.source_width}x{proposal.source_height}"
        )
    draw = ImageDraw.Draw(image)
    font = _overlay_font(max(12, round(min(image.size) / 45)))
    line_width = max(2, round(min(image.size) / 250))
    for index, candidate in enumerate(proposal.candidates):
        color = COLORS[index % len(COLORS)]
        pixels = normalized_to_pixels(candidate.box_2d, *image.size)
        draw.rectangle(pixels, outline=color, width=line_width)
        label = f"{index + 1}. {candidate.label}  {candidate.confidence:.2f}"
        text_box = draw.textbbox((pixels[0], pixels[1]), label, font=font, stroke_width=1)
        top = max(0, pixels[1] - (text_box[3] - text_box[1]) - 8)
        background = (pixels[0], top, min(image.width, pixels[0] + text_box[2] - text_box[0] + 8), pixels[1])
        draw.rectangle(background, fill=color)
        draw.text((pixels[0] + 4, top + 3), label, fill="#080b10", font=font)
    if not proposal.visible:
        message = f"NOT VISIBLE: {proposal.visibility_reason}"
        draw.rectangle((0, 0, image.width, max(32, image.height // 12)), fill="#ff3155")
        draw.text((10, 8), message, fill="white", font=font)
    image.save(output_path, format="PNG")
    return output_path


def draw_blind_review_overlay(
    frame_path: Path, proposal: GroundingProposal, output_path: Path
) -> Path:
    """Draw neutral candidate letters without model labels, confidence, or reasoning."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(frame_path).convert("RGB") as source:
        image = source.copy()
    if image.size != (proposal.source_width, proposal.source_height):
        raise ValueError("blind-review frame and proposal dimensions differ")
    draw = ImageDraw.Draw(image)
    font = _overlay_font(max(12, round(min(image.size) / 45)))
    line_width = max(2, round(min(image.size) / 250))
    for index, candidate in enumerate(proposal.candidates):
        color = COLORS[index % len(COLORS)]
        pixels = normalized_to_pixels(candidate.box_2d, *image.size)
        draw.rectangle(pixels, outline=color, width=line_width)
        label = f"Candidate {chr(65 + index)}"
        text_box = draw.textbbox((pixels[0], pixels[1]), label, font=font)
        label_width = text_box[2] - text_box[0] + 12
        label_height = text_box[3] - text_box[1] + 10
        top = max(0, pixels[1] - label_height)
        draw.rectangle(
            (pixels[0], top, min(image.width, pixels[0] + label_width), top + label_height),
            fill=color,
        )
        draw.text((pixels[0] + 6, top + 4), label, fill="#080b10", font=font)
    if not proposal.candidates:
        message = "NO BOX PROPOSED"
        height = max(32, image.height // 12)
        draw.rectangle((0, 0, image.width, height), fill="#6d7480")
        draw.text((10, 8), message, fill="white", font=font)
    image.save(output_path, format="PNG")
    return output_path
