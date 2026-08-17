"""What the tracker measured a card's subjects to be, kept beside the card.

A clip card's box is the model's answer to "where is that thing", drawn from
a 640-pixel proxy and framed the way the description reads. For a phone in a
hand the description puts the hand inside the box, so the card is wider than
the object while agreeing on centre and height -- measured across one film,
the card ran 1.38x the tracked width at the median and 2.40x at the worst,
with centres within 0.045 and heights within 1.08x.

Both numbers are correct answers to different questions, and two stages ask
different ones: Selection prices a read across the card's box, and the crop
follows the tracker's. So a read across 0.35 of frame was planned, 0.20 was
cropped, and a reveal that had two landings on paper had one on screen.

This closes that by remembering the measurement, not by editing the answer.
The card stays exactly what the model said -- it is content-addressed and
reruns must stay free -- and this sits next to it under the same key, so the
next plan for the same footage prices the object the crop will actually
follow. Nothing here is identity: a width is not evidence of which instance
was tracked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from montagewright.measure.storage import read_json, write_json

# Below this the two boxes are answering the same question and rewriting the
# card's width would be noise. Matches the crop compiler's own deadband on
# what counts as a visible difference.
MEANINGFUL_DISAGREEMENT = 0.02


def path_for(card_path: Path) -> Path:
    """The sidecar for a card, under the same content-addressed name."""

    return card_path.parent.parent / "tracked-geometry" / card_path.name


def read(card_path: Path | None) -> dict[str, dict[str, float]]:
    """Measured boxes by the label a look names, or nothing."""

    if card_path is None:
        return {}
    path = path_for(card_path)
    if not path.is_file():
        return {}
    try:
        stored = read_json(path)
    except (OSError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}
    return {
        str(label): {
            key: float(value)
            for key, value in entry.items()
            if isinstance(value, (int, float))
        }
        for label, entry in stored.items()
        if isinstance(entry, dict)
    }


def remember(
    card_path: Path | None, measured: dict[str, list[dict[str, Any]]],
) -> None:
    """Record the tracked extent of each subject this run followed.

    Merged rather than replaced: one edit sees a few of a card's subjects, and
    a subject measured in an earlier film is still the same footage. A later
    measurement of the same label wins, because the tracker that produced it
    saw the window this plan actually used.
    """

    if card_path is None or not measured:
        return
    keep = dict(read(card_path))
    for label, samples in measured.items():
        boxes = [
            one for one in samples
            if float(one.get("width") or 0.0) > 0.0
        ]
        if not boxes:
            continue
        keep[str(label)] = {
            "width": round(
                sum(float(one["width"]) for one in boxes) / len(boxes), 6
            ),
            "height": round(
                sum(float(one.get("height") or 0.0) for one in boxes)
                / len(boxes), 6
            ),
            "samples": float(len(boxes)),
        }
    if not keep:
        return
    try:
        write_json(path_for(card_path), keep)
    except OSError:
        # A cache that cannot be written is slower, not wrong.
        return


def applied(boxes: list[Any], card_path: Path | None) -> list[Any]:
    """Card boxes with a tracked extent substituted where one is known.

    Centre stays the card's: it is what the two agree on, and it is measured
    at the moment the card names rather than averaged over a window some later
    edit may not use.
    """

    tracked = read(card_path)
    if not tracked:
        return boxes
    from dataclasses import replace

    corrected = []
    for box in boxes:
        entry = tracked.get(getattr(box, "label", ""))
        if entry is None:
            corrected.append(box)
            continue
        width = float(entry.get("width") or 0.0)
        height = float(entry.get("height") or 0.0)
        if width <= 0.0 or abs(width - float(box.width)) < MEANINGFUL_DISAGREEMENT:
            corrected.append(box)
            continue
        corrected.append(replace(
            box, width=width, height=height if height > 0.0 else box.height,
        ))
    return corrected
