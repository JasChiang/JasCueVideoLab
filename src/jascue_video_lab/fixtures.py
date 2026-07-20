from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 720
FPS = 10


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _encode(path: Path, duration_seconds: float, renderer: Callable[[float], Image.Image]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
        "-framerate", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_number in range(round(duration_seconds * FPS)):
            frame = renderer(frame_number / FPS).convert("RGB")
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
    except BaseException:
        process.kill()
        raise
    if return_code:
        raise RuntimeError(f"ffmpeg fixture encoding failed: {stderr}")


def _phone(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], screen_text: str, *, pressed: bool = False) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=32, fill="#151821", outline="#e7edf5", width=5)
    inset = 20
    screen = (x1 + inset, y1 + 48, x2 - inset, y2 - 35)
    draw.rounded_rectangle(screen, radius=20, fill="#e9f1ff")
    draw.rectangle((x1 + 60, y1 + 14, x2 - 60, y1 + 22), fill="#929db0")
    button = (x1 + 55, y1 + 300, x2 - 55, y1 + 390)
    draw.rounded_rectangle(button, radius=18, fill="#ff5e7a" if pressed else "#396afc")
    draw.text((x1 + 45, y1 + 90), screen_text, fill="#14213d", font=_font(25))
    draw.text((x1 + 82, y1 + 325), "PRESS", fill="white", font=_font(22))


def _fixture_a(t: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#202637")
    draw = ImageDraw.Draw(image)
    if t < 7:
        title, stage = "HOME", "1 / 4"
    elif t < 15:
        title, stage = "SETTINGS", "2 / 4"
    elif t < 23:
        title, stage = "FEATURE: ON", "3 / 4"
    else:
        title, stage = "DONE", "4 / 4"
    _phone(draw, (430, 55, 850, 665), title, pressed=15 <= t < 16)
    hand_x = 800 + int(45 * math.sin(t * 1.2))
    draw.ellipse((hand_x, 420, hand_x + 150, 570), fill="#e4ad89")
    draw.text((55, 55), "Silent phone UI operation", fill="white", font=_font(40))
    draw.text((55, 115), stage, fill="#5eead4", font=_font(30))
    return image


def _fixture_b(t: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#b9dbef")
    draw = ImageDraw.Draw(image)
    person_x = 220 + int(60 * math.sin(t * 0.7))
    phone_x = 690 + int(150 * math.sin(t * 1.1))
    phone_y = 410 - min(180, int(t * 35))
    draw.ellipse((person_x, 100, person_x + 180, 300), fill="#b87552", outline="#3d2c29", width=5)
    draw.rectangle((person_x - 70, 290, person_x + 250, 710), fill="#2f4b7c")
    draw.line((person_x + 220, 360, phone_x, phone_y + 120), fill="#b87552", width=55)
    _phone(draw, (phone_x, phone_y, phone_x + 160, phone_y + 280), "APP")
    draw.text((50, 40), "Person + phone motion / 16:9", fill="#14213d", font=_font(38))
    return image


def _fixture_c(t: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#111827")
    draw = ImageDraw.Draw(image)
    pressed = 2.8 <= t < 3.1
    success = t >= 3.1
    color = "#f97316" if pressed else ("#10b981" if success else "#2563eb")
    label = "PRESSED" if pressed else ("SUCCESS" if success else "TAP ME")
    draw.rounded_rectangle((460, 290, 820, 450), radius=30, fill=color)
    text_box = draw.textbbox((0, 0), label, font=_font(42))
    draw.text(((WIDTH - (text_box[2] - text_box[0])) / 2, 340), label, fill="white", font=_font(42))
    draw.text((50, 50), "Transient state lasts 0.3 seconds", fill="white", font=_font(38))
    draw.text((50, 105), f"t = {t:0.1f} s", fill="#94a3b8", font=_font(28))
    return image


def _fixture_d(t: float) -> Image.Image:
    backgrounds = ["#d8c7a4", "#b9d8c2", "#c6c0de"]
    shot = min(2, int(t // 4))
    image = Image.new("RGB", (WIDTH, HEIGHT), backgrounds[shot])
    draw = ImageDraw.Draw(image)
    if shot == 0:
        _phone(draw, (230, 145, 470, 635), "LEFT")
        _phone(draw, (810, 145, 1050, 635), "RIGHT")
        label = "SHOT 1: similar phones"
    elif shot == 1:
        _phone(draw, (430, 65, 850, 675), "LEFT")
        label = "SHOT 2: left close-up"
    else:
        _phone(draw, (470, 65, 810, 675), "RIGHT")
        hand_x = 710 + int(30 * math.sin(t * 2))
        draw.ellipse((hand_x, 350, hand_x + 210, 560), fill="#c98f6c")
        label = "SHOT 3: right operated"
    draw.text((40, 30), label, fill="#172033", font=_font(36))
    return image


def generate_fixtures(output_dir: Path) -> list[Path]:
    jobs = [
        ("A_silent_phone_ui.mp4", 30.0, _fixture_a),
        ("B_person_phone_motion_16x9.mp4", 12.0, _fixture_b),
        ("C_fast_transient_ui.mp4", 6.0, _fixture_c),
        ("D_cuts_similar_objects.mp4", 12.0, _fixture_d),
    ]
    paths = []
    for name, duration, renderer in jobs:
        path = output_dir / name
        _encode(path, duration, renderer)
        paths.append(path)
    return paths

