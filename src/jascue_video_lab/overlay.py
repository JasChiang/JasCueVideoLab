from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .geometry import normalized_to_pixels
from .models import GroundingProposal


COLORS = ["#ff3155", "#00c2ff", "#ffd43b", "#7cfc6b", "#c77dff"]


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
    font = ImageFont.load_default(size=max(12, round(min(image.size) / 45)))
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

