"""Putting corrected words back on the clock that measured them.

The recogniser is the only thing here that knows *when*. It hears badly and
times precisely: every character carries a start and an end measured from
the audio. The model hears well and cannot time anything -- asked for a
timestamp it produces a plausible number, which is the one kind of answer
that cannot be checked by looking.

So the text comes from one and the clock from the other, and this is the
join. Corrected characters are aligned against the recognised ones; wherever
they agree the recognised timing is kept exactly, and where they differ the
new characters are spread across the span the old ones occupied. Nothing is
invented and nothing drifts: the first and last moment of a line are moments
the recogniser actually measured.

This module is deliberately self-contained: it uses Python's sequence matcher
and the Apple word spans already present in a Montagewright transcript card.
It neither imports nor depends on another captioning project.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from montagewright.transcript import Word


@dataclass(frozen=True)
class Timed:
    """One character of the finished caption, and when it was said."""

    text: str
    starts_seconds: float
    ends_seconds: float
    # False when this character was not in what the recogniser heard, so its
    # time was worked out rather than measured. Worth keeping: a caption
    # made entirely of these is a caption nobody should trust the timing of.
    measured: bool = True


def _pieces(words: list[Word]) -> list[tuple[str, float, float]]:
    """The recogniser's output as characters, each with its own span.

    Chinese comes back a character at a time already; English arrives as
    whole words, and a word's characters share its span -- which is wrong by
    a few milliseconds each and right about where the word is, and that is
    the resolution any of this needs.
    """

    out: list[tuple[str, float, float]] = []
    for word in words:
        said = word.text.strip()
        if not said:
            continue
        span = max(word.ends_seconds - word.starts_seconds, 0.0)
        step = span / len(said) if len(said) > 1 else span
        for index, letter in enumerate(said):
            begins = word.starts_seconds + step * index
            out.append((letter, begins, begins + step))
    return out


def align(said: str, words: list[Word]) -> list[Timed]:
    """Lay corrected text over recognised timings.

    Punctuation the correction added belongs to the character before it --
    it takes no time of its own, and giving it a span of its own would make
    every full stop a beat of silence in a karaoke fill.
    """

    heard = _pieces(words)
    if not heard:
        return []
    if not said:
        return []

    # Alignment ignores punctuation, which the recogniser does not produce
    # and the correction adds freely; matching on it would tear otherwise
    # identical runs apart.
    marks = set("，。、！？；：…「」『』（）,.!?;:\"'()… ")
    spoken = [i for i, letter in enumerate(said) if letter not in marks]
    lean = "".join(said[i] for i in spoken)

    match = SequenceMatcher(None, [one[0] for one in heard], list(lean),
                            autojunk=False)
    placed: dict[int, tuple[float, float, bool]] = {}
    for kind, h0, h1, s0, s1 in match.get_opcodes():
        if kind == "equal":
            for step in range(s1 - s0):
                _, begins, ends = heard[h0 + step]
                placed[spoken[s0 + step]] = (begins, ends, True)
        elif kind in ("replace", "insert"):
            # Spread the new characters over whatever the old ones occupied.
            # An insertion has no span of its own, so it borrows the moment
            # between its neighbours.
            if h1 > h0:
                begins, ends = heard[h0][1], heard[h1 - 1][2]
            elif h0 < len(heard):
                begins = ends = heard[h0][1]
            else:
                begins = ends = heard[-1][2]
            count = max(s1 - s0, 1)
            step = (ends - begins) / count
            for index in range(s1 - s0):
                at = begins + step * index
                placed[spoken[s0 + index]] = (at, at + step, False)
        # "delete" means the recogniser heard something the correction
        # removed. Its time simply goes; the neighbours already cover it.

    out: list[Timed] = []
    for i, letter in enumerate(said):
        if i in placed:
            begins, ends, measured = placed[i]
            out.append(Timed(letter, begins, ends, measured))
        elif out:
            # Punctuation, or a character no opcode reached: it happens at
            # the end of whatever came before it and lasts no time.
            out.append(Timed(letter, out[-1].ends_seconds,
                             out[-1].ends_seconds, False))
        else:
            begins = heard[0][1]
            out.append(Timed(letter, begins, begins, False))
    return out


def across_lines(
    said: list[str], words: list[Word]
) -> list[tuple[float, float, list[Timed]]]:
    """Time a whole corrected transcript against the whole recognised one.

    Doing this line by line would need the model's timestamps to know which
    words each line covers, and its timestamps are the part not to trust. So
    the lines are joined, aligned against every word in one pass, and cut
    back apart: each line's span is then the span of its own characters, and
    the model's numbers are used for nothing whatever. It keeps what it is
    good at -- the words, and where a sentence ends -- and the clock stays
    entirely the recogniser's.

    A line that aligned to nothing measured comes back with a zero span. It
    is better for a caption to be visibly missing its timing than to sit
    plausibly in the wrong place.
    """

    joined = "".join(said)
    timed = align(joined, words)
    if not timed:
        return [(0.0, 0.0, []) for _ in said]

    out: list[tuple[float, float, list[Timed]]] = []
    at = 0
    for line in said:
        mine = timed[at:at + len(line)]
        at += len(line)
        if not mine:
            out.append((0.0, 0.0, []))
            continue
        # A line's edges should be moments somebody was actually speaking,
        # so trailing punctuation must not set the end. The test is span,
        # not `measured`: a corrected character holds the real span of the
        # one it replaced -- 髮 for 發 is still time somebody was talking --
        # while punctuation and pure insertions take no time at all. Judging
        # by `measured` here cut every corrected character off the end of
        # its own line.
        solid = [one for one in mine
                 if one.ends_seconds > one.starts_seconds] or mine
        out.append((solid[0].starts_seconds, solid[-1].ends_seconds, mine))
    return out


def what_was_heard(
    words: list[Word], from_seconds: float, to_seconds: float
) -> str:
    """Exactly what the recogniser produced across a span.

    The correction pass is also asked to report what it was correcting, and
    it does so imperfectly -- in one interview it wrote 髮 where the
    recogniser had said 發, silently mending the very error the field exists
    to preserve. A model asked to fix mistakes and to quote them unchanged
    in the same breath will do the first to both.

    The recogniser's own output is on disk. Read it there instead.
    """

    return "".join(
        word.text for word in words
        if word.ends_seconds > from_seconds and word.starts_seconds < to_seconds
    )


def drift(timed: list[Timed], words: list[Word]) -> float:
    """How far the finished caption has slid from what was measured.

    This is the one number that says whether the join held: text may change
    freely, but the first and last moment of a line have to be moments
    somebody actually said something.
    """

    if not timed or not words:
        return 0.0
    return max(
        abs(timed[0].starts_seconds - words[0].starts_seconds),
        abs(timed[-1].ends_seconds - words[-1].ends_seconds),
    )
