"""What was said, when, and what it actually was.

A transcript card is its own artifact, not more fields on a clip card. A clip
card describes what a take looks like and is worth caching because that stays
true; a transcript is only worth paying for when the speech matters, and it is
useful on its own -- subtitling a finished video is a job that never touches
the edit.

Two halves, split the way everything else here is split. The system recogniser
is very good at when a word was said and reliably wrong about what several of
them were: it emits characters converted from a simplified model (乾 as 幹,
回 as 迴, 面 as 麵), it mishears proper nouns, and it has never heard of the
product on screen. Gemini gets the audio, the picture, and the recogniser's
own text, and corrects the words. Timings are never touched by the model --
they are measured, and a model asked for a timestamp will invent one.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from montagewright.planner import ask
from montagewright.gemini import structured_json
from montagewright.uploads import upload_now

# v3: two listenings rather than one. The recogniser's timings never reach
# the model at all now, a second pass hears the video without being shown
# what the first heard, and the correction reconciles two texts with no media
# in front of it. v2 heard the master instead of the proxy's 64 kbps
# re-encode and gained the recogniser's alternative readings. Each is a
# different answer to a different question, so none is reused -- unlike a
# proxy or a card, these are cheap to get again.
CARD_VERSION = ""  # derived below after both response schemas are defined
TOOL = Path(__file__).resolve().parents[2] / "tools" / "transcribe" / "transcribe"

# Below this a "word" is usually the recogniser splitting one syllable, and a
# subtitle cannot sit on it.
MIN_WORD_SECONDS = 0.04


class TranscriberMissing(RuntimeError):
    """The Swift tool has not been built for this machine."""


@dataclass(frozen=True)
class Word:
    text: str
    starts_seconds: float
    ends_seconds: float
    confidence: float | None = None


@dataclass(frozen=True)
class Line:
    """One subtitle: what to show, and the window it belongs in."""

    text: str
    starts_seconds: float
    ends_seconds: float
    heard: str = ""
    # Who said it, named so the frame can find them. Acoustic diarisation
    # answers "a different voice"; the picture answers "the man in the blue
    # shirt", which is the vocabulary the reframe layer already speaks -- and
    # a talking shot framed on whoever is not talking is the fault this is
    # here to make fixable.
    speaker: str = ""

    @property
    def duration(self) -> float:
        return self.ends_seconds - self.starts_seconds

    @property
    def corrected(self) -> bool:
        return bool(self.heard) and self.heard != self.text


def extract_audio(source: Path, destination: Path) -> Path:
    """Mono 16k PCM, which is what the recogniser wants and nothing more."""

    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(destination),
        ],
        check=True,
    )
    return destination


def hear(source: Path, *, locale: str = "zh-TW") -> dict[str, Any]:
    """Run the system recogniser over one file.

    Returns its answer unaltered. Whatever is wrong with the words is wrong
    at this point and is corrected later against the picture; rewriting them
    here would mean guessing without having seen anything.
    """

    if not TOOL.exists():
        raise TranscriberMissing(
            f"{TOOL} is not built. Run:\n"
            f"  swiftc -parse-as-library -O -o {TOOL} {TOOL.with_suffix('.swift').parent}/Transcribe.swift"
        )
    with tempfile.TemporaryDirectory() as work:
        wav = extract_audio(source, Path(work) / "audio.wav")
        completed = subprocess.run(
            [str(TOOL), str(wav), locale],
            capture_output=True, text=True, check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"transcription failed for {source.name}: "
            f"{completed.stderr.strip()[:300]}"
        )
    return json.loads(completed.stdout)


def words_of(payload: dict[str, Any]) -> list[Word]:
    """Every word the recogniser placed, in order."""

    words: list[Word] = []
    for utterance in payload.get("utterances", []) or []:
        for entry in utterance.get("words", []) or []:
            try:
                start = float(entry["starts_seconds"])
                end = float(entry["ends_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            text = str(entry.get("text", "")).strip()
            if not text or end - start < MIN_WORD_SECONDS:
                continue
            confidence = entry.get("confidence")
            words.append(
                Word(
                    text=text,
                    starts_seconds=start,
                    ends_seconds=end,
                    confidence=(
                        float(confidence) if confidence is not None else None
                    ),
                )
            )
    return sorted(words, key=lambda word: word.starts_seconds)


def hesitations(payload: dict[str, Any]) -> list[tuple[float, float, str, list[str]]]:
    """Stretches the recogniser had more than one reading for.

    It has always produced these and was never asked for them. The prompt
    already points the correction at the low-confidence words -- which is a
    good marker, the two characters misheard in one interview came back at
    0.72 and 0.79 with everything around them above 0.99 -- but a marker
    says only that the recogniser was unsure, not what it was unsure
    between. So the correction had to invent a replacement out of the
    picture and the sense, when the candidates it should be choosing from
    came out of the audio and were sitting in the same result.

    Choosing between readings is a judgement. Inventing one is a guess in
    the same clothes, and the difference does not show in the output.
    """

    found: list[tuple[float, float, str, list[str]]] = []
    for utterance in payload.get("utterances", []) or []:
        others = [
            str(one).strip()
            for one in (utterance.get("alternatives") or [])
            if str(one).strip()
        ]
        said = str(utterance.get("text", "")).strip()
        if not others or not said:
            continue
        try:
            start = float(utterance["starts_seconds"])
            end = float(utterance["ends_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        found.append((start, end, said, others))
    return found


# The recogniser writes these where it heard a break, and gives each one a
# span of its own.
BREAKS = "，。？！、…,.?!"


def gaps(words: list[Word], *, at_least: float = 0.35) -> list[float]:
    """Where the speaker stopped, from the recogniser's own punctuation.

    Not from the space between words: the transcriber segments a stream
    continuously, so each word's end is the next word's start and every gap
    between them is exactly zero -- ninety-five per cent of them in one
    seventy-second interview. Nothing about silence can be recovered from
    subtracting those.

    The punctuation can. A 。 or ， is the recogniser saying it heard a break,
    and the token carries the span of that break -- between a tenth of a
    second and nearly a whole one. Those spans are the pauses, measured,
    already in hand.

    A pause is still not a sentence ending: hesitating, thinking and being
    interrupted all make pauses. Which of these is a boundary is for whoever
    can hear the sentence; their answer is snapped onto these.
    """

    return [
        round(word.ends_seconds, 3)
        for word in words
        if word.text in BREAKS
        and word.ends_seconds - word.starts_seconds >= at_least / 4.0
    ]


def snap_end(seconds: float, candidates: list[float], *, within: float = 1.0) -> float:
    """Put an out-point after the pause, never before it.

    Every candidate here is the far edge of a break the recogniser marked, so
    the cut lands where the sound has finished rather than where the last
    syllable nominally ended. Taking the nearest instead of the next one is
    what clipped the final word: the pause before it is closer than the pause
    after it about half the time, and choosing it eats the word.
    """

    later = [point for point in candidates if point >= seconds - 0.02]
    if not later:
        return seconds
    return later[0] if later[0] - seconds <= within else seconds


def snap(seconds: float, candidates: list[float], *, within: float = 0.6) -> float:
    """Put a spoken boundary on the nearest measured silence.

    The model hears where a sentence ends; the recogniser measured where the
    sound stopped. Taking the model's second directly puts the cut a syllable
    early or late, and taking only the silences puts it in the middle of a
    thought -- the same split as a subject box seeded by name and measured by
    the tracker.
    """

    if not candidates:
        return seconds
    nearest = min(candidates, key=lambda point: abs(point - seconds))
    return nearest if abs(nearest - seconds) <= within else seconds


def to_srt(lines: list[Line], *, with_speaker: bool = False) -> str:
    """Standard subtitles, so this is useful without the rest of the tool.

    Who said it is structured data, not part of the line, so putting it on
    screen is a decision rather than a default. The two callers had already
    drifted -- one prefixed the name and one did not -- which is what an
    implicit choice does.
    """

    def stamp(seconds: float) -> str:
        milli = max(0, round(seconds * 1000))
        hours, milli = divmod(milli, 3_600_000)
        minutes, milli = divmod(milli, 60_000)
        secs, milli = divmod(milli, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milli:03d}"

    blocks = []
    for index, line in enumerate(lines, start=1):
        said = (
            f"{line.speaker}：{line.text}"
            if with_speaker and line.speaker
            else line.text
        )
        blocks.append(
            f"{index}\n"
            f"{stamp(line.starts_seconds)} --> {stamp(line.ends_seconds)}\n"
            f"{said}\n"
        )
    return "\n".join(blocks)


def load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("version") == CARD_VERSION else None


def save(card: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**card, "version": CARD_VERSION}, ensure_ascii=False,
                   indent=1),
        encoding="utf-8",
    )
    return path


def words_in(card: dict[str, Any]) -> list[Word]:
    """The words a card kept, if it kept any."""

    made: list[Word] = []
    for entry in card.get("words", []) or []:
        try:
            made.append(Word(
                text=str(entry.get("text", "")),
                starts_seconds=float(entry["starts_seconds"]),
                ends_seconds=float(entry["ends_seconds"]),
                confidence=(
                    None if entry.get("confidence") is None
                    else float(entry["confidence"])
                ),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return made


def words_for(
    card: dict[str, Any], source: Path, *, locale: str = "zh-TW",
    into: Path | None = None,
) -> list[Word]:
    """The words for this material, measured again if the card has none.

    The recogniser runs on this machine and costs nothing, so a card written
    before words were kept is not a reason to go without them -- or to pay
    for the correction pass a second time. What comes back is written into
    the card so the next reader does not have to ask.
    """

    kept = words_in(card)
    if kept or not source.exists():
        return kept
    try:
        heard = hear(source, locale=locale)
    except Exception:
        return []
    found = words_of(heard)
    if found and into is not None:
        card["words"] = [
            {
                "text": word.text,
                "starts_seconds": round(word.starts_seconds, 3),
                "ends_seconds": round(word.ends_seconds, 3),
                "confidence": word.confidence,
            }
            for word in found
        ]
        save(card, into)
    return found


def lines_of(card: dict[str, Any]) -> list[Line]:
    return [
        Line(
            text=str(entry.get("text", "")),
            starts_seconds=float(entry.get("starts_seconds", 0.0)),
            ends_seconds=float(entry.get("ends_seconds", 0.0)),
            heard=str(entry.get("heard", "")),
            speaker=str(entry.get("speaker", "")),
        )
        for entry in card.get("lines", []) or []
        if entry.get("text")
    ]


# Where a clipped sentence would rather begin and end.
_JOINTS = "。！？…，、；："


def _within(line: Line, *, from_seconds: float, to_seconds: float) -> str:
    """The part of a line that falls inside a window.

    There are no word timings kept, so the share of the window a piece
    occupies stands in for the share of the words -- then the ends are
    nudged to the nearest place the sentence pauses, because a subtitle
    starting mid-word reads as a fault in the tool rather than as a cut.
    """

    span = line.ends_seconds - line.starts_seconds
    if span <= 0:
        return line.text
    before = max(0.0, (from_seconds - line.starts_seconds) / span)
    after = max(0.0, (line.ends_seconds - to_seconds) / span)
    if before + after < 0.08:
        return line.text

    text = line.text
    head = round(len(text) * before)
    tail = len(text) - round(len(text) * after)
    if tail - head < 2:
        return ""

    reach = max(2, len(text) // 6)

    def joint_near(at: int) -> int:
        """The nearest place the sentence pauses, or -1.

        rfind counts from the end when given a negative start, so an
        unclamped window silently searched the wrong part of the line -- and
        snapping the head past the tail produced an inverted slice, which is
        an empty string, which is a subtitle that vanished.
        """

        low = max(0, min(at - reach, len(text)))
        high = max(low, min(at + reach, len(text)))
        found = [text.rfind(mark, low, high) for mark in _JOINTS]
        return max(found)

    # Only nudge an end that is actually being cut. Snapping the head of a
    # line the shot caught from its first word moved it past the opening
    # clause, so a sentence that started on time started two words late.
    if before > 0.02:
        moved = joint_near(head)
        if 0 <= moved + 1 < tail:
            head = moved + 1
    if after > 0.02:
        moved = joint_near(tail)
        if moved + 1 > head:
            tail = moved + 1

    said = text[head:tail].strip()
    # Two characters of a sentence, on screen for a moment, is noise. The
    # cut caught the edge of somebody talking; the words are not the point.
    return said if len(said) >= 3 else ""


def against_cut(
    shots: list[dict],
    rhythm: dict[str, dict],
    # Read-only, and a Mapping rather than a dict so a caller holding plain
    # cards can pass them: dict is invariant in its value type, so
    # dict[str, dict] is not a dict[str, dict | None].
    #
    # A source with nothing transcribed has no card, and `load` returns None
    # for one it cannot read. Both mean the same thing here: no lines.
    cards: Mapping[str, dict | None],
) -> list[Line]:
    """Every transcribed line, moved onto the timeline the shots landed on.

    A line is timed against the take it was spoken in, and the cut kept two
    seconds of that take starting somewhere in the middle. A subtitle file
    that makes the reader work out which take a line came from is not one.

    This was written inside the SRT endpoint, which was the only thing that
    needed it. Three things need it now -- the file, the track on the
    timeline, and eventually burning it into the picture -- and three copies
    of "where does this line land" is three answers to it.
    """

    timed: list[Line] = []
    cursor = 0.0
    for index, shot in enumerate(shots):
        seconds = float(rhythm.get(f"k{index:02d}", {}).get("seconds", 0.0))
        card = cards.get(str(shot.get("source_id", "")))
        start = float(shot.get("start_seconds", 0.0))
        for line in lines_of(card or {}):
            if line.ends_seconds <= start:
                continue
            if line.starts_seconds >= start + seconds:
                continue
            # A shot can hold part of a sentence. Clipping the window and
            # not the words put five seconds of talking on screen for one,
            # so the whole sentence flashed past under a shot that only
            # caught its tail.
            said = _within(
                line, from_seconds=start, to_seconds=start + seconds
            )
            if not said:
                continue
            timed.append(
                Line(
                    text=said,
                    starts_seconds=cursor
                    + max(0.0, line.starts_seconds - start),
                    ends_seconds=cursor
                    + min(seconds, line.ends_seconds - start),
                    heard=line.heard,
                    speaker=line.speaker,
                )
            )
        cursor += seconds
    return timed


def words_against_cut(
    shots: list[dict],
    rhythm: dict[str, dict],
    cards: Mapping[str, dict | None],
) -> list[Word]:
    """Every measured word, moved onto the timeline the shots landed on.

    The same shift the lines get. Without it the words stay in the time of
    the take they were spoken in, and a subtitle asked to fill as it is said
    fills to the rhythm of a completely different part of the interview.
    """

    moved: list[Word] = []
    cursor = 0.0
    for index, shot in enumerate(shots):
        seconds = float(rhythm.get(f"k{index:02d}", {}).get("seconds", 0.0))
        card = cards.get(str(shot.get("source_id", "")))
        start = float(shot.get("start_seconds", 0.0))
        for word in words_in(card or {}):
            if word.ends_seconds <= start:
                continue
            if word.starts_seconds >= start + seconds:
                continue
            moved.append(Word(
                text=word.text,
                starts_seconds=cursor + max(0.0, word.starts_seconds - start),
                ends_seconds=cursor + min(seconds, word.ends_seconds - start),
                confidence=word.confidence,
            ))
        cursor += seconds
    return moved


def _hearing_schema() -> dict[str, Any]:
    """What a second listener heard, before it is shown the first one's answer.

    Kept apart from the correction pass on purpose. A model handed the
    recogniser's transcript and asked to fix it will agree with any line that
    reads well -- and the errors hardest to catch are exactly the ones that
    read well, a plausible word that is not the word that was said. Only a
    listener that has not seen the answer can disagree with it.

    Its timestamps are rough and are used as such. They order the blocks and
    say roughly where each one sits; the clock still comes from the
    recogniser's measured per-word timings, which this never sees.
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["language", "blocks", "terms", "summary"],
        "properties": {
            "language": {
                "type": "string",
                "description": (
                    "BCP-47 for what is actually spoken. The recogniser was "
                    "run with a guess; this is the answer."
                ),
            },
            "summary": {"type": "string"},
            "terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "語音辨識器很可能聽錯的詞：人名、品牌、產品名、地名、"
                    "專業術語、固定詞組。照實際說的拼寫。你剛看完整支影片"
                    "包含畫面上的字，這是最有把握寫出這些詞的時刻。"
                ),
            },
            "blocks": {
                "type": "array",
                "description": "說了什麼，照順序，一塊 3 到 7 秒。",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["from", "to", "speaker", "said"],
                    "properties": {
                        "from": {
                            "type": "string",
                            "description": (
                                "這一塊大約從哪裡開始，MM:SS。只用來對位，"
                                "不會變成字幕時間，抓大概就好。"
                            ),
                        },
                        "to": {"type": "string"},
                        "speaker": {
                            "type": "string",
                            "description": (
                                "誰在講，用畫面上分辨得出來的描述："
                                "「戴帽子的主持人」、「穿灰藍上衣的受訪男子」。"
                                "以話的內容為準——問句是問的人講的；鏡頭對"
                                "著誰、誰握著麥克風都不是證據。"
                            ),
                        },
                        "said": {
                            "type": "string",
                            "description": (
                                "這一塊逐字說了什麼。贅字、結巴、重複、改口"
                                "全部留著，夾雜其他語言照原樣寫。順稿順掉的"
                                "每一個字都是一段對不回時間的聲音。"
                            ),
                        },
                        "unclear": {
                            "type": "string",
                            "description": "聽不清楚的地方，沒有就留空。",
                        },
                    },
                },
            },
        },
    }


def _schema() -> dict[str, Any]:
    """Flat, and with no field the model has to invent a number for."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["language", "lines", "uncertain", "summary"],
        "properties": {
            "language": {
                "type": "string",
                "description": (
                    "BCP-47 for what is actually spoken. The recogniser was "
                    "run with a guess; this is the answer."
                ),
            },
            "summary": {"type": "string"},
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    # Three fields left when the local clock won. The times
                    # were parsed and dropped -- `across_lines` aligns the
                    # corrected text onto the recogniser's own per-word
                    # measurements and never reads them -- and `heard` was
                    # overwritten from the stored words, because a model
                    # asked to correct errors and quote them unchanged in one
                    # breath corrects both, and did: it reported 髮 where the
                    # recogniser had said 發, erasing the only evidence the
                    # field carries.
                    #
                    # Removing them is not only about wasted output. Asking
                    # something that cannot measure time to state times makes
                    # it reconcile its text with numbers it invented, and the
                    # text is the half being kept.
                    "required": ["text", "speaker"],
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "改正後的字幕文字。",
                        },
                        "speaker": {
                            "type": "string",
                            "description": (
                                "誰在說這一句，用畫面上分辨得出來的描述："
                                "「戴帽子的主持人」、「穿灰藍上衣的受訪"
                                "男子」。判斷以話的內容為準——問句是問的人"
                                "講的，回答是被問的人講的；鏡頭對著誰、誰"
                                "握著麥克風都不是證據。畫面外的聲音就說"
                                "畫面外，分不出來就在 uncertain 說明。"
                            ),
                        },
                    },
                },
            },
            "uncertain": {
                "type": "array",
                "items": {"type": "string"},
                "description": "聽不清楚或無法確定的地方，各一句說明。",
            },
        },
    }


def _transcript_version() -> str:
    """Invalidate cached words when either listening contract changes."""

    import hashlib

    prompts = Path(__file__).resolve().parent / "prompts"
    payload = json.dumps(
        {"hearing": _hearing_schema(), "correction": _schema()},
        ensure_ascii=False,
        sort_keys=True,
    )
    payload += (prompts / "hearing_zh-TW.txt").read_text(encoding="utf-8")
    payload += (prompts / "transcript_zh-TW.txt").read_text(encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"montagewright-transcript-{digest}"


CARD_VERSION = _transcript_version()


def describe(
    source: Path,
    *,
    client,
    locale: str = "zh-TW",
    cache=None,
    model_id: str | None = None,
    audio: Path | None = None,
    ledger: Any | None = None,
) -> tuple[dict[str, Any], Any]:
    """Hear it locally, then have the words corrected against the picture.

    `audio` is the file the recogniser listens to, when that should not be
    the one the model watches. It ran on the proxy, whose sound is 64 kbps
    AAC re-encoded from the master's uncompressed PCM -- twenty-four times
    smaller, for a picture the model needs and a waveform it does not. That
    loss bought nothing: the recogniser runs on this machine, and the
    original is on disk beside the proxy. Measured over ninety seconds of
    street interview it cost eight differences in three hundred and
    eighty-five characters.

    Split rather than swapped, because the upload has to stay the proxy. It
    is already in the file cache from the card pass, and sending the master
    would re-upload a 4K file to say the same thing.
    """

    # Imported here, not at the top: backfill reads Word from this module,
    # so a module-level import would close the circle.
    from montagewright.backfill import across_lines, what_was_heard
    from montagewright.planner import (
        MAX_OUTPUT_TOKENS,
        MODEL_ID,
        PROMPTS,
        Usage,
        _parse,
    )

    heard = hear(audio or source, locale=locale)
    words = words_of(heard)
    silences = gaps(words)
    # What the recogniser wrote, without a single timestamp on it. Its
    # confidence stays, because that is a statement about itself rather than
    # about the clock, and it is a good one -- the two characters misheard in
    # one interview came back at 0.72 and 0.79 with everything around them
    # above 0.99.
    rough = "\n".join(
        f"{index}. {word.text}"
        + (
            f"  ←不確定 {word.confidence:.2f}"
            if word.confidence is not None and word.confidence < 0.9
            else ""
        )
        for index, word in enumerate(words)
    )

    # The readings it weighed and did not pick. Sent as its own block rather
    # than woven into the word list, because they belong to a stretch of
    # speech and not to a word: the recogniser's second reading of a phrase
    # can split it differently from its first.
    weighed = hesitations(heard)
    considered = ""
    if weighed:
        considered = "\n## 辨識器猶豫過的地方（同一段它也考慮過這些讀法）\n\n" + "\n".join(
            f"{start:.2f}–{end:.2f}  {said}"
            f"　也可能是：{'／'.join(others[:4])}"
            for start, end, said, others in weighed
        ) + "\n"

    if cache is None:
        uri = upload_now(source, client).uri
    else:
        uri, _ = cache.uri_for(source, client, mime_type="video/mp4")

    # First listening: the video, and nothing the recogniser said. Two calls
    # rather than one because the order matters more than the count -- shown
    # a transcript first, a model agrees with any line that reads well, and
    # the errors hardest to catch are exactly the ones that read well. Only a
    # listener that has not seen the answer can disagree with it.
    #
    # The upload is shared, so the second call adds no bytes and the file
    # cache already holds this proxy from the card pass.
    listening = ask(
        client,
        model=model_id or MODEL_ID,
        store=False,
        input=[
            {
                "type": "video",
                "mime_type": "video/mp4",
                "uri": uri,
                # Low, deliberately, and this has been argued once already.
                #
                # The picture is here mostly for `speaker`: who is saying
                # this line, described by how they look. That is not a note
                # -- selection is told to put the frame on whoever is
                # speaking, using the same words, because a talking shot
                # framed on the person not talking is the most obvious error
                # this makes. Telling a hat from a microphone from a
                # grey-blue shirt does not need detail.
                #
                # The one job that does is reading small print off a screen,
                # for the product names and model numbers the prompt says to
                # take from the picture rather than from memory. High
                # quadruples the cap on each frame, from seventy tokens to
                # two hundred and eighty, and video is sampled at one frame a
                # second however long the clip is: measured across the seven
                # pieces of one 326-second interview that is 22,610 tokens
                # against 90,440, so about ten cents a run rather than the
                # four thousand tokens a short clip suggested.
                #
                # And it would buy nothing on either batch to hand. Material
                # with wordmarks and model numbers on screen is product
                # footage, whose speech is rarely content, so it never
                # reaches this pass at all; the material that does reach it
                # is people talking outdoors with no text in frame. Worth
                # revisiting for a clip that is both, which would want the
                # card to record whether there is text on screen so this can
                # be asked per clip instead of guessed for all of them.
                "resolution": "low",
            },
            {
                "type": "text",
                "text": (PROMPTS / "hearing_zh-TW.txt").read_text(encoding="utf-8"),
            },
        ],
        generation_config={
            "thinking_level": "high",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        response_format=structured_json(_hearing_schema()),
        ledger=ledger,
        budget_stage="transcript",
    )
    listened = _parse(listening, what="hearing")
    usage = Usage.from_interaction(listening)

    # Second listening: two transcripts of one piece of audio, and no media
    # at all. Everything the picture had to say was said by the call above --
    # who is speaking, what is written on screen, which words a recogniser
    # would mangle -- and the question left is which of two readings of the
    # same sound is right, which is a question about two texts.
    said_by_ear = "\n".join(
        f"[{one.get('from', '')}–{one.get('to', '')}] "
        f"{one.get('speaker', '')}：{one.get('said', '')}"
        + (f"　（聽不清楚：{one['unclear']}）" if one.get("unclear") else "")
        for one in listened.get("blocks") or []
    )
    terms = [str(one).strip() for one in listened.get("terms") or [] if str(one).strip()]
    glossary = (
        "\n## 這支片裡的專有名詞（照這樣寫，遇到同音的普通詞優先用這個）\n\n"
        + "、".join(terms) + "\n"
        if terms else ""
    )

    instruction = (PROMPTS / "transcript_zh-TW.txt").read_text(encoding="utf-8")
    interaction = ask(
        client,
        model=model_id or MODEL_ID,
        store=False,
        input=[{
            "type": "text",
            "text": (
                f"{instruction}\n\n## 辨識器聽到的（照順序，沒有時間）"
                f"\n\n{rough}\n{considered}"
                f"\n## 另一個聽眾聽到的\n\n{said_by_ear}\n{glossary}"
            ),
        }],
        generation_config={
            "thinking_level": "high",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
        response_format=structured_json(_schema()),
        ledger=ledger,
        budget_stage="transcript",
    )
    payload = _parse(interaction, what="transcript")
    spent = Usage.from_interaction(interaction)
    usage = Usage(
        input_tokens=usage.input_tokens + spent.input_tokens,
        output_tokens=usage.output_tokens + spent.output_tokens,
        thought_tokens=usage.thought_tokens + spent.thought_tokens,
    )

    # The model knows the words and where a sentence ends. The recogniser
    # knows when. Take each from the one that has it: the corrected lines are
    # aligned back onto the measured per-word clock, and the model's own
    # timestamps are dropped unread. They were never checkable by looking --
    # a wrong one and a right one are the same plausible number -- and the
    # recogniser's are, because it got them from the audio.
    said = [
        str(entry.get("text", "")).strip()
        for entry in payload.get("lines", []) or []
    ]
    timings = across_lines(said, words)

    lines = []
    for entry, text, (start, end, _) in zip(
        payload.get("lines", []) or [], said, timings
    ):
        if end <= start:
            continue
        lines.append({
            "text": text,
            # Not what the model says it heard -- what the recogniser
            # actually produced across this span. Asked to correct errors
            # and quote them unchanged in one breath, the model corrects
            # both: in one interview it reported 髮 where the recogniser
            # had said 發, erasing the only evidence the field carries.
            "heard": what_was_heard(words, start, end),
            "speaker": str(entry.get("speaker", "")).strip(),
            "starts_seconds": round(start, 3),
            "ends_seconds": round(end, 3),
        })

    card = {
        "summary": payload.get("summary", ""),
        "language": payload.get("language", locale),
        "heard_with": heard.get("locale", locale),
        "duration_seconds": heard.get("duration_seconds", 0.0),
        "lines": [line for line in lines if line["text"]],
        "uncertain": payload.get("uncertain", []) or [],
        # Where the speaker actually stopped. A cut placed anywhere else in a
        # talking shot lands mid-syllable, and every consumer of this card
        # needs them, not just the one that wrote them.
        "silences": silences,
        # When each word was said. Measured locally and used here to write
        # the prompt and find the silences, then thrown away -- which put
        # word-level subtitles out of reach of a card that already knew the
        # answer. Kept without bumping the card version: an older card
        # simply has no words, and getting them again costs nothing.
        "words": [
            {
                "text": word.text,
                "starts_seconds": round(word.starts_seconds, 3),
                "ends_seconds": round(word.ends_seconds, 3),
                "confidence": word.confidence,
            }
            for word in words
        ],
    }
    return card, usage
