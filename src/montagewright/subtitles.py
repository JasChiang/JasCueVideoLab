"""Putting the words on the picture.

Two things decide where a subtitle can sit, and neither is a constant.

The first is the delivery shape. A 9:16 cut is watched inside an app that
draws its own things over the bottom of the frame -- the handle, the caption,
the row of buttons -- so a caption sitting where a caption traditionally sits
is a caption nobody reads. A 16:9 cut has none of that and only wants the
old title-safe margin. The band is a property of where the film is going, and
writing one number for all of them would be the execution layer deciding
something about distribution.

The second is the font. A line that fits in 16:9 does not fit in 9:16, and
Chinese does not break on spaces, so "how many characters per line" is not a
number either -- it is measured, against the actual font at the actual size.

The words themselves are drawn here rather than by ffmpeg. Its text filters
need libass or libfreetype compiled in, and the build on this machine has
neither, so a tool that depended on them would work here and not there. This
needs only `overlay`, which every build has.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

# 禁則: characters a line is not allowed to begin with. Breaking before a
# comma leaves it hanging under the start of the line above, which is the
# one typographic mistake in Chinese that everybody notices.
NEVER_STARTS = "，。、！？：；）」』】》,.!?;:)]}"

# Where to look when the system cannot be asked. Ordered by preference; a
# machine with none of them and no fontconfig cannot burn Chinese subtitles,
# and is told so rather than shown boxes.
#
# Every one of these is a guess with a shelf life. PingFang used to sit in
# /System/Library/Fonts and does not any more, which is the whole reason
# fontconfig is asked first.
FONTS: tuple[str, ...] = (
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)


# Enough colours to tell people apart, and no more. They are picked to stay
# legible over a moving picture rather than to be pretty: high value, low
# saturation, and none of them the red the interface uses for its own marks.
SPEAKER_COLOURS: tuple[tuple[int, int, int], ...] = (
    (255, 255, 255),
    (255, 226, 138),
    (168, 226, 255),
    (198, 246, 186),
    (255, 198, 224),
)


@dataclass(frozen=True)
class Style:
    """How the words look. Everything here is a decision somebody may want
    to make differently, so none of it is buried in the drawing code."""

    fill: tuple[int, int, int] = (255, 255, 255)
    outline: tuple[int, int, int] = (0, 0, 0)
    outline_alpha: int = 205
    # A share of the type size, so it holds at any frame height.
    outline_width: float = 0.09
    # Colour the words already spoken, so the line fills as it is said.
    # The recogniser timed every character, so this is measured rather than
    # paced -- which is the difference between reading along and watching a
    # progress bar.
    spoken: tuple[int, int, int] | None = None
    # A plate behind the words, for footage too busy for an outline alone.
    plate: tuple[int, int, int] | None = None
    plate_alpha: int = 150
    # Colour each speaker differently. In an interview this says who is
    # talking faster than reading the words does.
    by_speaker: bool = False
    # Scale the size the safe area asks for, for anyone who wants it bigger.
    size_scale: float = 1.0


@dataclass(frozen=True)
class TimedOverlay:
    path: Path
    left: int
    top: int
    starts_seconds: float
    ends_seconds: float
    width: int
    height: int
    cue_index: int
    state_index: int


@dataclass(frozen=True)
class PreparedSubtitleTrack:
    overlays: tuple[TimedOverlay, ...]

    @property
    def windows(self) -> list[tuple[float, float]]:
        return [
            (overlay.starts_seconds, overlay.ends_seconds)
            for overlay in self.overlays
        ]

    @property
    def boxes(self) -> list[tuple[float, float, int, int, int, int]]:
        return [
            (
                overlay.starts_seconds, overlay.ends_seconds,
                overlay.left, overlay.top, overlay.width, overlay.height,
            )
            for overlay in self.overlays
        ]


def palette_for(lines, style: Style) -> dict[str, tuple[int, int, int]]:
    """A colour per speaker, assigned in the order they first talk.

    Hashing the name instead gave two people in one interview the same
    colour, which is worse than giving them all the same one: it says they
    are the same person.
    """

    if not style.by_speaker:
        return {}
    order: list[str] = []
    for line in lines:
        who = getattr(line, "speaker", "")
        if who and who not in order:
            order.append(who)
    return {
        who: SPEAKER_COLOURS[index % len(SPEAKER_COLOURS)]
        for index, who in enumerate(order)
    }


class NoFontHere(RuntimeError):
    """Nothing on this machine can draw the characters in these lines."""


@dataclass(frozen=True)
class SafeArea:
    """Where the words are allowed to be, as fractions of the frame.

    `up_from_bottom` is to the bottom of the last line, not to the baseline
    of the first: a two-line subtitle grows upward, so that it never grows
    down into whatever the player is drawing there.
    """

    up_from_bottom: float
    side_margin: float
    # Cap height as a fraction of frame height. Vertical video is watched
    # smaller and further from the eye than a 16:9 cut on a desktop, so the
    # same apparent size is a bigger fraction of a shorter frame.
    text_height: float
    # One. A subtitle is read in a glance, and a second row asks the eye to
    # go back to the left and start again while somebody is still talking.
    # A sentence too long for one row is a sentence that should have been
    # two cues, which is what split_cues is for -- this is only the ceiling
    # for the ones that will not split: no punctuation anywhere to break on.
    # Even then it grows rather than drops a word.
    max_lines: int = 1


# Keyed by the same names the aspect flag uses.
SAFE_AREAS: dict[str, SafeArea] = {
    # Reels, Shorts and TikTok all draw a caption block and a button rail
    # over the lower fifth. Measured against a screenshot of each rather
    # than guessed: the tallest intrusion is TikTok's caption plus handle.
    "9:16": SafeArea(up_from_bottom=0.22, side_margin=0.08, text_height=0.030),
    # A feed post. The bottom is clear in the post itself, but a reshare to
    # a story puts it back, so this keeps a little more room than it needs.
    "4:5": SafeArea(up_from_bottom=0.14, side_margin=0.07, text_height=0.034),
    "1:1": SafeArea(up_from_bottom=0.12, side_margin=0.07, text_height=0.038),
    # Nothing is drawn over this one. The margin is the old title-safe rule.
    "16:9": SafeArea(up_from_bottom=0.10, side_margin=0.06, text_height=0.045),
}


def safe_area(aspect: str) -> SafeArea:
    """The band for a delivery shape, defaulting to the most constrained."""

    return SAFE_AREAS.get(aspect, SAFE_AREAS["9:16"])


# Set to a path to override everything below it.
CHOSEN: str | None = None


@lru_cache(maxsize=12)
def _asked_of_the_system(lang: str) -> list[str]:
    """Candidate font files, ranked, from fontconfig if it is installed.

    A hard-coded list of paths is a guess about somebody else's machine, and
    it was wrong on this one: /System/Library/Fonts/PingFang.ttc does not
    exist. Apple moved the CJK faces into on-demand assets under
    /System/Library/AssetsV2, where the file is perfectly ordinary and opens
    without complaint. Native applications never noticed because Core Text
    is asked for a font by name and finds it wherever it lives; only
    something reaching for a path can be wrong about where that is.
    """

    try:
        found = subprocess.run(
            ["fc-match", "--sort", "-f", "%{file}\n", f":lang={lang}"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [one for one in found.stdout.splitlines() if one.strip()]


def _can_draw(face, text: str) -> bool:
    """Whether this font has the characters, rather than boxes for them.

    Pillow draws a missing glyph as .notdef and says nothing, so a name it
    cannot spell comes out as tofu in a finished film. A private-use
    codepoint is missing from every sane font, so anything that renders
    identically to it is missing too.
    """

    absent = face.getmask("\ue001")
    for letter in set(text):
        if letter.isspace():
            continue
        mask = face.getmask(letter)
        if mask.size == absent.size and bytes(mask) == bytes(absent):
            return False
    return True


def cannot_spell(text: str, *, size: int = 40) -> str:
    """The characters the chosen font has no glyph for.

    Worth saying out loud: they are drawn as empty boxes and nothing else
    reports it, so a name the font cannot spell reaches a finished film.
    """

    face = _face(size, text=text)
    return "".join(
        sorted({
            letter for letter in text
            if not letter.isspace() and not _can_draw(face, letter)
        })
    )


# Which face inside a collection, for a language. A .ttc holds several --
# PingFang carries HK, MO, TC and SC in four weights -- and index 0 is
# whatever the file happens to list first, which for PingFang is Hong Kong
# Regular. HK and TC draw some characters differently, so a Taiwanese cut
# was being set in the wrong regional forms, and Regular is thin against a
# moving picture.
REGIONS: dict[str, tuple[str, ...]] = {
    "zh-tw": ("TC", "HK", "SC"),
    "zh-hk": ("HK", "TC", "SC"),
    "zh-cn": ("SC", "TC", "HK"),
    "ja": ("",),
}
WEIGHTS: tuple[str, ...] = ("Medium", "Semibold", "Regular")


def _best_face(path: str, size: int, lang: str):
    """Open the face inside a collection that suits the language."""

    from PIL import ImageFont

    wanted = REGIONS.get(lang.lower(), ("TC", "HK", "SC"))
    best, best_score = None, None
    for index in range(12):
        try:
            face = ImageFont.truetype(path, size, index=index)
        except OSError:
            break
        family, style = face.getname()
        region = next(
            (i for i, tag in enumerate(wanted)
             if tag and family.endswith(tag)), len(wanted)
        )
        weight = next(
            (i for i, tag in enumerate(WEIGHTS) if tag == style), len(WEIGHTS)
        )
        score = (region, weight)
        if best_score is None or score < best_score:
            best, best_score = face, score
        if index == 0 and not path.lower().endswith((".ttc", ".otc")):
            break
    return best


def _face(size: int, *, text: str = "", lang: str = "zh-tw"):
    from PIL import ImageFont

    tried: list[str] = []
    if CHOSEN:
        tried.append(CHOSEN)
    tried += _asked_of_the_system(lang)
    tried += list(FONTS)

    # fontconfig has already ranked these for the language, so rank is the
    # strong signal and coverage is the check on it. Requiring a font to
    # draw *everything* let one emoji drag the choice down the list: no
    # Chinese font qualifies, and the first that did set a street interview
    # in STIX Two Math. Ranking by coverage instead picked a handwriting
    # face that merely claims the emoji -- the glyph probe is a heuristic
    # and it is wrong about some fonts.
    #
    # So the first candidate that draws nearly all of it wins, and whatever
    # it still cannot spell is named by cannot_spell rather than chased.
    wanted = {one for one in text if not one.isspace()}
    best = None
    for candidate in tried[:24]:
        try:
            face = _best_face(candidate, size, lang) or \
                ImageFont.truetype(candidate, size)
        except OSError:
            continue
        if not wanted:
            return face
        drawn = sum(1 for one in wanted if _can_draw(face, one))
        if drawn >= len(wanted) * 0.8:
            return face
        if best is None or drawn > best[0]:
            best = (drawn, face)
    if best is not None:
        return best[1]
    raise NoFontHere(
        "no font here can draw these subtitles; they can still be written "
        "as a file. Set montagewright.subtitles.CHOSEN to a font path, or "
        "install fontconfig so the system can be asked."
    )


def wrap(text: str, face, room: int) -> list[str]:
    """Break a line to fit, measuring rather than counting.

    Chinese does not break on spaces, so a character budget is wrong for any
    line with a number, a product name or a laugh in it. This asks the font
    how wide the string actually is, and breaks at a space when there is one
    near enough to be the better break.

    Filling each line before starting the next is how a paragraph is set,
    and a subtitle is not a paragraph: it left "然後慢慢被自己 / 蒸乾", a
    full line and an orphan, which reads as a mistake on screen. Two lines
    get balanced instead.
    """

    if not text:
        return []
    # 標點懸掛: a mark at the end of a line may sit past the margin. Without
    # it, one comma turns a line that fits into two rows.
    if _width(text, face) <= room:
        return [text]
    if face.getbbox(text)[2] > room and face.getbbox(text)[2] <= room * 2:
        halved = _in_two(text, face, room)
        if halved:
            return halved
    lines: list[str] = []
    rest = text
    while rest:
        if _width(rest, face) <= room:
            lines.append(rest)
            break
        cut = len(rest)
        while cut > 1 and face.getbbox(rest[:cut])[2] > room:
            cut -= 1
        # Prefer a space, but only if it is not so far back that the line
        # ends up half empty.
        spaced = rest.rfind(" ", 0, cut + 1)
        if spaced > cut * 0.6:
            cut = spaced
        # Never leave the next line starting on a closing mark.
        while cut > 1 and rest[cut:cut + 1] in NEVER_STARTS:
            cut -= 1
        lines.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    return lines


def _width(text: str, face) -> float:
    """How wide it really needs to be, letting a closing mark hang."""

    return face.getbbox(text.rstrip(NEVER_STARTS) or text)[2]


def _in_two(text: str, face, room: int) -> list[str] | None:
    """Split once, as near the middle as the font allows."""

    best = None
    for cut in range(1, len(text)):
        head, tail = text[:cut].rstrip(), text[cut:].lstrip()
        if not head or not tail:
            continue
        # Chinese permits a visual break between most ideographs, but a
        # structural particle at the start makes the previous noun phrase
        # look broken (「梅雨 / 的味道」).  Prefer a slightly less geometric
        # balance over that grammatical stumble.
        if tail[0] in NEVER_STARTS + "的地得":
            continue
        wide_head = _width(head, face)
        wide_tail = _width(tail, face)
        if wide_head > room or wide_tail > room:
            continue
        # A breath wins first, then a light grammatical joint, then raw
        # geometry.  Purely balancing glyph widths split compounds such as
        # 「梅雨」even though the nearby「有 / 梅雨的味道」joint is still
        # comfortably balanced.
        if text[cut - 1] in "，。、！？：；,.!?;:":
            joint = 0
        elif text[cut - 1] in "是有在就也又都很更最會要想說讓把被給與和或但而":
            joint = 1
        else:
            joint = 2
        # Do not buy a grammatical joint by leaving one side visibly tiny.
        if min(wide_head, wide_tail) < max(wide_head, wide_tail) * 0.6:
            joint += 3
        score = (joint, abs(wide_head - wide_tail))
        if best is None or score < best[0]:
            best = (score, [head, tail])
    return best[1] if best else None


# Where a sentence is willing to be broken into separate cues, best first.
BREAKS: tuple[str, ...] = ("。", "！", "？", "…", "，", "、", "；", "：")

# A cue shorter than this reads as a flash rather than a line.
LEAST_SECONDS = 0.7


def split_cues(
    lines, face, room: int, *, least: float = LEAST_SECONDS, words=None,
):
    """Break long lines into separate cues, rather than into more rows.

    The transcript's idea of a line is a sentence: the median is thirteen
    characters and the tail runs to fifty-six. Wrapping the long ones filled
    two rows with fifty characters of Chinese over somebody's face, which is
    not a subtitle, it is a paragraph -- and shrinking the type to make it
    fit only made it a smaller paragraph.

    Measured word timings place each break when they are available.  A long
    corrected sentence used to be divided by character count even though the
    recogniser had measured every word; pauses and uneven delivery could make
    the next cue appear several tenths early.  Old transcripts without words
    retain the proportional fallback.
    """

    from montagewright.transcript import Line

    out = []
    for line in lines:
        if _width(line.text, face) <= room:
            out.append(line)
            continue

        pieces = _by_sense(line.text, face, room)
        span = max(line.ends_seconds - line.starts_seconds, 0.001)
        total = sum(len(one) for one in pieces) or 1
        # The corrected character clock is the strongest timing evidence we
        # have: Gemini owns the spelling, Apple owns these timestamps.  Do
        # not throw it away and rematch the pre-correction ASR words merely
        # because the sentence became too wide for one cue.
        marks = _character_marks(line)
        if not marks and words:
            marks = _word_marks(
                line.text, words, line.starts_seconds, line.ends_seconds
            )
        at = line.starts_seconds
        made = []
        consumed = 0
        for index, one in enumerate(pieces):
            begins_at = consumed
            consumed += len(one)
            proportional_end = line.starts_seconds + span * consumed / total
            # The next cue begins when its first measured word begins, not
            # when the previous word ends.  Keeping a real pause between the
            # two is the point of using the recogniser clock here.
            following = [
                starts for begins, _, starts, _ in marks
                if begins >= consumed and begins - consumed <= 2
            ]
            preceding = [
                ends for _, reached, _, ends in marks
                if reached <= consumed and ends > at
            ]
            measured = following[:1] or preceding[-1:]
            until = (
                line.ends_seconds
                if index == len(pieces) - 1
                else measured[-1] if measured else proportional_end
            )
            until = min(line.ends_seconds, max(at, until))
            made.append(Line(
                text=one, starts_seconds=at, ends_seconds=until,
                heard=line.heard, speaker=line.speaker,
                timing_source=line.timing_source,
                timing_confidence=line.timing_confidence,
                timing_locked=line.timing_locked,
                timed_text=_clock_slice(line, begins_at, consumed),
            ))
            at = until

        # A piece too brief to read is joined to the one before it, which is
        # why this is not simply "split and move on".
        joined = []
        for piece in made:
            if (
                joined
                and piece.ends_seconds - piece.starts_seconds < least
                # Only if the two of them still fit on one row. Joining up
                # to two rows to avoid a brief cue traded the fault for a
                # worse one: a wall of text instead of a quick line.
                and _width(joined[-1].text + piece.text, face) <= room
            ):
                last = joined.pop()
                joined.append(Line(
                    text=last.text + piece.text,
                    starts_seconds=last.starts_seconds,
                    ends_seconds=piece.ends_seconds,
                    heard=last.heard, speaker=last.speaker,
                ))
            else:
                joined.append(piece)
        out.extend(joined)
    return out


def _by_sense(text: str, face, room: int) -> list[str]:
    """Cut a sentence where it already pauses, then where it must."""

    pieces, rest = [], text
    while _width(rest, face) > room:
        # When all that remains fits in two rows, balance them as two cues.
        # Filling the first to the pixel edge is how a perfectly ordinary
        # phrase such as「梅雨的味道」became「梅雨的味 / 道」.  This is a
        # general orphan rule, not a vocabulary exception.
        balanced = _in_two(rest, face, room)
        if balanced:
            pieces.extend(balanced)
            rest = ""
            break
        cut = len(rest)
        while cut > 1 and face.getbbox(rest[:cut])[2] > room:
            cut -= 1
        # The last place it took a breath, if that is not too far back.
        best = max(
            (rest.rfind(mark, 0, cut + 1) for mark in BREAKS), default=-1
        )
        if best >= cut * 0.45:
            cut = best + 1
        while cut > 1 and rest[cut:cut + 1] in NEVER_STARTS:
            cut -= 1
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
        if not rest:
            break
    if rest:
        pieces.append(rest.strip())
    return [one for one in pieces if one]


def _character_marks(line) -> list[tuple[int, int, float, float]]:
    """Character spans and Apple's clock, if it still matches the copy."""

    clock = tuple(getattr(line, "timed_text", ()) or ())
    if not clock or "".join(one.text for one in clock) != line.text:
        return []
    marks = []
    where = 0
    for piece in clock:
        reached = where + len(piece.text)
        if piece.ends_seconds > piece.starts_seconds:
            marks.append((
                where, reached, piece.starts_seconds, piece.ends_seconds,
            ))
        where = reached
    return marks


def _clock_slice(line, begins: int, reaches: int):
    """Keep timing provenance on a visual fragment of one transcript line."""

    clock = tuple(getattr(line, "timed_text", ()) or ())
    if not clock or "".join(one.text for one in clock) != line.text:
        return ()
    kept, where = [], 0
    for piece in clock:
        after = where + len(piece.text)
        if after > begins and where < reaches:
            kept.append(piece)
        where = after
    return tuple(kept)


def draw_line(
    text: str,
    *,
    width: int,
    height: int,
    area: SafeArea,
    into: Path,
    style: Style | None = None,
    ink: tuple[int, int, int] | None = None,
    said_upto: int = 0,
) -> tuple[Path, int, int] | None:
    """One subtitle as a transparent image, plus where to put it.

    Returns the file and its top-left corner in the frame, or None when the
    line has nothing to draw.
    """

    from PIL import Image, ImageDraw

    style = style or Style()
    room = round(width * (1 - area.side_margin * 2))
    asked = max(12, round(height * area.text_height * style.size_scale))

    # Fit by getting smaller, never by saying less. Cutting the overflow
    # off at max_lines dropped the last word of a sentence and left a cut
    # that looked finished, which is the worst way for a subtitle to fail.
    size, face, lines = asked, _face(asked), []
    for attempt in range(6):
        size = max(12, round(asked * (1 - attempt * 0.06)))
        face = _face(size)
        lines = wrap(text, face, room)
        if len(lines) <= area.max_lines:
            break
    if not lines:
        return None

    # An outline, not a box. A box is legible and covers the picture; an
    # outline stays readable over both a bright sky and a dark interior,
    # which is what a street interview is made of.
    edge = max(2, round(size * style.outline_width))
    gap = round(size * 0.34)
    step = size + gap

    widest = max(face.getbbox(one)[2] for one in lines)
    canvas = Image.new(
        "RGBA",
        (widest + edge * 4, step * len(lines) + edge * 4),
        (0, 0, 0, 0),
    )
    pen = ImageDraw.Draw(canvas)
    if style.plate is not None:
        pen.rounded_rectangle(
            (0, 0, canvas.width - 1, canvas.height - 1),
            radius=max(4, round(size * 0.22)),
            fill=(*style.plate, style.plate_alpha),
        )
    ink = ink or style.fill
    seen = 0
    for index, one in enumerate(lines):
        at_x = (canvas.width - face.getbbox(one)[2]) // 2
        at_y = edge * 2 + index * step
        pen.text(
            (at_x, at_y), one, font=face, fill=(*ink, 255),
            stroke_width=edge, stroke_fill=(*style.outline, style.outline_alpha),
        )
        # The part already spoken, drawn over the top in its own colour.
        # Drawn as a prefix of the same string at the same origin, so the
        # letters cannot drift by a pixel against the ones underneath.
        if style.spoken is not None and said_upto > seen:
            cut = min(len(one), said_upto - seen)
            if cut > 0:
                pen.text(
                    (at_x, at_y), one[:cut], font=face,
                    fill=(*style.spoken, 255), stroke_width=edge,
                    stroke_fill=(*style.outline, style.outline_alpha),
                )
        seen += len(one)

    into.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(into)
    left = (width - canvas.width) // 2
    top = round(height * (1 - area.up_from_bottom)) - canvas.height
    return into, max(0, left), max(0, top)


def layout_boxes(
    lines: Sequence,
    *,
    aspect: str,
    width: int,
    height: int,
    work: Path,
    style: Style | None = None,
) -> list[tuple[float, float, int, int, int, int]]:
    """Return the renderer's actual timed subtitle rectangles.

    Graphics layout consumes these exact Pillow bounds instead of treating
    the entire lower third as occupied whenever any subtitle is present.
    """

    return prepare_overlays(
        lines, aspect=aspect, width=width, height=height,
        work=work, style=style,
    ).boxes


def _word_marks(text: str, words, at: float, until: float):
    """Matched character span plus the recogniser's start/end clock."""

    inside = [w for w in words if w.ends_seconds > at and w.starts_seconds < until]
    marks: list[tuple[int, int, float, float]] = []
    where = 0
    for word in inside:
        said = word.text.strip()
        if not said:
            continue
        found = text.find(said, where)
        if found < 0:
            continue
        where = found + len(said)
        # Punctuation after a word is revealed with it.
        while where < len(text) and text[where] in NEVER_STARTS:
            where += 1
        marks.append((found, where, word.starts_seconds, word.ends_seconds))
    return marks


def spans_in(text: str, words, at: float, until: float):
    """Match a cue's characters to the words that were measured.

    The recogniser times Chinese per character, so a cue is a run of them
    with the punctuation put back by the correction pass. Walking both at
    once and consuming a word only when it matches the next characters
    keeps the two in step through anything inserted between them; the
    punctuation simply belongs to the character before it.

    Returns (end_index, seconds) pairs: how much of the line has been said
    by when. Empty when the words do not line up, which is the signal to
    draw the cue as one piece rather than to guess.
    """

    marks = [
        (reached, ends)
        for _, reached, _, ends in _word_marks(text, words, at, until)
    ]
    return marks if len(marks) >= 2 else []


def prepare_overlays(
    lines: Sequence,
    *,
    aspect: str,
    width: int,
    height: int,
    work: Path,
    style: Style | None = None,
    words=None,
) -> PreparedSubtitleTrack:
    """Compile subtitle pixels once for either subtitle-only or mixed output."""

    area = safe_area(aspect)
    face = _face(max(12, round(height * area.text_height)))
    room = round(width * (1 - area.side_margin * 2))
    lines = split_cues(lines, face, room, words=words)
    style = style or Style()
    palette = palette_for(lines, style)
    unknown = cannot_spell("".join(line.text for line in lines))
    if unknown:
        print(
            f"subtitles: no glyph for {unknown} — they will be empty boxes",
            flush=True,
        )

    drawn: list[TimedOverlay] = []
    for index, line in enumerate(lines):
        ink = palette.get(getattr(line, "speaker", "")) or style.fill
        states: list[tuple[int, float, float]] = [
            (0, line.starts_seconds, line.ends_seconds)
        ]
        if style.spoken is not None and words:
            marks = spans_in(
                line.text, words, line.starts_seconds, line.ends_seconds
            )
            if marks:
                states = []
                upto, since = 0, line.starts_seconds
                for reached, when in marks:
                    if when > since:
                        states.append((upto, since, when))
                    upto, since = reached, when
                states.append((upto, since, line.ends_seconds))
        for part, (upto, since, until) in enumerate(states):
            if until <= since:
                continue
            made = draw_line(
                line.text, width=width, height=height, area=area,
                into=work / f"sub-{index:04d}-{part:03d}.png",
                style=style, ink=ink, said_upto=upto,
            )
            if made is not None:
                from PIL import Image

                path, left, top = made
                with Image.open(path) as image:
                    wide, tall = image.size
                drawn.append(TimedOverlay(
                    path, left, top, float(since), float(until),
                    wide, tall, index, part,
                ))
    return PreparedSubtitleTrack(tuple(drawn))


def burn(
    picture: Path,
    lines: Sequence,
    destination: Path,
    *,
    aspect: str,
    work: Path,
    style: Style | None = None,
    words=None,
) -> Path:
    """Composite the lines onto the picture, each over its own window.

    One overlay per line, gated by time. ffmpeg's own subtitle filters would
    be one filter instead of thirty, and are not in every build -- including
    the one this was written on.
    """

    from montagewright.measure.media import probe_video

    shape = probe_video(picture).video
    # display_*, not coded_*: the coded size is padded to the macroblock
    # grid, and a subtitle placed against padding sits a few pixels off.
    width, height = int(shape.display_width), int(shape.display_height)
    rate = shape.average_frame_rate or shape.real_frame_rate
    output_fps = (
        f"{rate.numerator}/{rate.denominator}"
        if rate is not None and rate.denominator else "30/1"
    )
    prepared = prepare_overlays(
        lines, aspect=aspect, width=width, height=height, work=work,
        style=style, words=words,
    )
    if not prepared.overlays:
        raise ValueError("no lines with anything on them")

    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(picture)]
    for overlay in prepared.overlays:
        command += ["-i", str(overlay.path)]

    steps, tag = [], "0:v"
    for index, overlay in enumerate(prepared.overlays):
        nxt = f"v{index}"
        steps.append(
            f"[{tag}][{index + 1}:v]overlay={overlay.left}:{overlay.top}:"
            f"enable='gte(t,{overlay.starts_seconds:.3f})*"
            f"lt(t,{overlay.ends_seconds:.3f})'[{nxt}]"
        )
        tag = nxt
    command += [
        "-filter_complex", ";".join(steps),
        "-map", f"[{tag}]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-r", output_fps,
        "-fps_mode", "cfr", "-c:a", "copy", str(destination),
    ]
    subprocess.run(command, check=True)
    return destination


def as_cues(lines, aspect: str, width: int, height: int, *, words=None):
    """The lines a viewer should see, whatever is going to show them.

    The file and the picture had better agree, and a fifty-six character
    cue is as unreadable in a player as it is burned into a frame.
    """

    area = safe_area(aspect)
    face = _face(max(12, round(height * area.text_height)))
    return split_cues(
        lines, face, round(width * (1 - area.side_margin * 2)), words=words,
    )


# The looks worth offering by name. Anything finer is a Style, which is
# what these are made of.
LOOKS: dict[str, Style] = {
    "plain": Style(),
    # An interview with more than one person in it.
    "speakers": Style(by_speaker=True),
    # The one social video mostly uses: the line fills as it is said.
    "spoken": Style(spoken=(255, 214, 90)),
    # For footage too busy for an outline to hold on its own.
    "plate": Style(plate=(16, 15, 13), plate_alpha=165, outline_width=0.05),
}


def look(name: str) -> Style:
    return LOOKS.get(name, LOOKS["plain"])


def fonts_here(lang: str = "zh-tw") -> list[dict]:
    """Every font on this machine that can set this language.

    Offered so nobody has to type a path. Names come from the font's own
    name table, which is what the system indexes them by; the path is
    carried alongside because FreeType can only be handed one of those.
    """

    try:
        found = subprocess.run(
            ["fc-list", "-f", "%{family[0]}\t%{file}\n", f":lang={lang}"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    seen: dict[str, str] = {}
    for row in found.stdout.splitlines():
        family, _, path = row.partition("\t")
        family, path = family.strip(), path.strip()
        # Names beginning with a dot are the interface variants the system
        # keeps for itself, and LastResort is the font of last resort --
        # the one that draws the boxes.
        if not family or not path or family.startswith("."):
            continue
        if "LastResort" in path:
            continue
        seen.setdefault(family, path)
    return [
        {"family": family, "file": path}
        for family, path in sorted(seen.items())
    ]
