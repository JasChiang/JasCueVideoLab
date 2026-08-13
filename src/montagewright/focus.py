"""Which seconds of a take are in focus, measured against the take itself.

The model reads video at a frame a second off a 640-pixel proxy, which is
about the worst possible instrument for judging focus: a soft frame at that
size looks like a frame of something soft. Asked whether a clip is usable it
answered that the whole of a take was clean, listing only a tail trim -- and
the first two seconds are the lens still hunting. An edit then moved its own
in-point a second earlier to land on the gesture, straight into the softest
part of the take, while the sharp half sat inside the same eligible span.

What is measured here is the variance of the Laplacian, which is the ordinary
focus measure: sharp edges have large second derivatives and a soft picture
has none. What is deliberately *not* done is compare that number between
clips. A white tabletop with three phones on it scores 22 wide open; a
close-up of a face against a papered wall scores 22 when it is perfectly
sharp and 10 when it is not. An absolute threshold would reject the second
clip entirely and pass the first one's soft moments -- so every reading here
is a share of the same take's own best, and nothing is comparable across
files.

Nor is this a gate. Focus falls away for reasons an edit wants: a rack onto
the product, a foreground going soft as the subject steps forward, a
deliberate defocus to end on. A number cannot tell those from a lens
hunting, so the profile is reported and the judgement stays with the model
that can see what the shot is doing. The one rule taken locally is narrower
than judgement and true regardless of intent: a cut may not be *moved* into
softer footage than the frame it was planned on. Nobody asked for that.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

# Twice a second. Focus hunts and settles over tenths of a second at the
# fastest, and seventy-four clips have to be affordable.
SAMPLE_FPS = 2.0

# Wide enough to keep the high-frequency detail the measure is made of.
# The 192 that global motion is measured at would erase the difference this
# exists to see.
WIDTH = 640
HEIGHT = 360

# A frame at or above this share of the take's own best is as sharp as this
# take gets. Below the second, the lens is somewhere else.
SHARP = 0.75
SOFT = 0.55

READING = "focus-v1"


@dataclass(frozen=True)
class FocusSample:
    """One reading, as a share of the sharpest frame in the same take."""

    at_seconds: float
    share: float


def measure(source: Path) -> list[FocusSample]:
    """Sharpness over the take, each frame against the take's own best."""

    raw = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf", f"fps={SAMPLE_FPS},scale={WIDTH}:{HEIGHT},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ],
        capture_output=True, check=False,
    ).stdout
    size = WIDTH * HEIGHT
    count = len(raw) // size
    if count < 2:
        return []

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy ships with the pipeline
        return []

    frames = np.frombuffer(raw[:count * size], dtype=np.uint8)
    frames = frames.reshape(count, HEIGHT, WIDTH).astype(np.float32)
    # The four-neighbour Laplacian, and its variance over the frame.
    lap = (
        frames[:, :-2, 1:-1] + frames[:, 2:, 1:-1]
        + frames[:, 1:-1, :-2] + frames[:, 1:-1, 2:]
        - 4.0 * frames[:, 1:-1, 1:-1]
    )
    scores = lap.var(axis=(1, 2))
    best = float(scores.max())
    if best <= 0.0:
        return []
    return [
        FocusSample(
            at_seconds=round(index / SAMPLE_FPS, 3),
            share=round(float(score) / best, 3),
        )
        for index, score in enumerate(scores)
    ]


def share_at(profile: list[FocusSample], seconds: float) -> float | None:
    """What the take's focus was doing at this moment, if it is known."""

    if not profile:
        return None
    nearest = min(profile, key=lambda one: abs(one.at_seconds - seconds))
    if abs(nearest.at_seconds - seconds) > 1.0 / SAMPLE_FPS:
        return None
    return nearest.share


def windows(
    profile: list[FocusSample], starts: float, ends: float
) -> tuple[float, float]:
    """The worst and best share inside a stretch, for describing a span."""

    inside = [
        one.share for one in profile if starts - 0.001 <= one.at_seconds <= ends
    ]
    if not inside:
        return (1.0, 1.0)
    return (min(inside), max(inside))


def describe(profile: list[FocusSample], starts: float, ends: float) -> str:
    """What to tell the planner about this stretch, or nothing.

    Silence is the common case and the right one: a take that holds focus
    has nothing to say here, and a line on every source would train whoever
    reads it to skip the line.
    """

    if not profile:
        return ""
    low, high = windows(profile, starts, ends)
    if low >= SHARP:
        return ""
    sharp = [
        one.at_seconds for one in profile
        if starts - 0.001 <= one.at_seconds <= ends and one.share >= SHARP
    ]
    if not sharp:
        return (
            f"這一段整段都沒有對到這支素材最清楚的程度"
            f"（最高只有 {high:.0%}）"
        )
    return (
        f"這一段的清晰度在 {low:.0%}–{high:.0%} 之間浮動"
        f"（跟這支自己最清楚的一格比），"
        f"{_clock(min(sharp))}–{_clock(max(sharp))} 才是真正對到焦的部分"
    )


def _clock(seconds: float) -> str:
    """Every time that reaches a model is MM:SS."""

    minutes = int(seconds // 60)
    return f"{minutes}:{seconds - minutes * 60:04.1f}"


def _mean_over(
    profile: list[FocusSample], starts: float, ends: float
) -> float | None:
    inside = [
        one.share for one in profile
        if starts - 0.001 <= one.at_seconds <= ends
    ]
    return sum(inside) / len(inside) if inside else None


def moving_into_softer(
    profile: list[FocusSample],
    was: float,
    now: float,
    seconds: float,
    *,
    margin: float = 0.1,
) -> bool:
    """Whether shifting a cut's in-point lands it on softer footage.

    Not a judgement about whether soft is wrong. A shot planned on a soft
    frame stays where it was planned; this only refuses to *move* one into
    worse focus than it already had, which is the one case where nobody
    chose the softness.

    Measured over the shot's opening as well as its whole length, because
    the opening is what a move changes and what an audience sees first: the
    take that prompted this holds a sharp plateau from two seconds on, so
    both windows still contain sharp footage and only their first second
    tells them apart -- 80% against 33%.
    """

    if not profile:
        return False
    opening = min(1.0, seconds)
    for length in (opening, seconds):
        before = _mean_over(profile, was, was + length)
        after = _mean_over(profile, now, now + length)
        if before is None or after is None:
            continue
        if after < before - margin:
            return True
    return False


def cached(source: Path, library: Path) -> list[FocusSample]:
    """Measure once per file, ever. A fact about the bytes."""

    from montagewright.uploads import content_hash

    where = library / "focus" / f"{content_hash(source)[:20]}.{READING}.json"
    if where.exists():
        try:
            stored = json.loads(where.read_text(encoding="utf-8"))
            return [FocusSample(**one) for one in stored]
        except (OSError, ValueError, TypeError):
            pass
    found = measure(source)
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(
        json.dumps([asdict(one) for one in found], indent=2), encoding="utf-8"
    )
    return found
