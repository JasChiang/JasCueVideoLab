"""One table of the events a clip was measured to contain.

Every detector in this package already finds real, frame-accurate moments --
scene cuts from ``scdet`` (carrying decoded PTS), word edges from the on-device
recogniser, beats and downbeats from the music analysis, action starts from the
tracker. They live in different shapes in different modules, and the editorial
brain, which speaks only coarse whole seconds, has no single place to point at
them.

This module is the collecting layer, not a new detector and not a new snapper.
It gathers what was already measured into one per-clip table so that

  * a coarse ``0:03`` from the model can be pulled onto the nearest *measured*
    moment (``EventTable.snap``), and
  * the model can name a moment instead of guessing its second at all
    (``EventTable.resolve`` turns ``cut:C3:002`` back into a source clock).

The snapping here does not decide policy. ``snap`` dispatches to the opinionated
functions that already exist -- ``transcript.snap`` / ``transcript.snap_end``
for spoken boundaries, ``grounding.BeatGrid.nearest_cue`` for the music -- each
of which encodes *why not the nearest one*. Flattening those into a single
minimum-distance rule is the exact mistake ``nearest_cue`` documents at length.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from .measure.models import FrozenStrictModel

if TYPE_CHECKING:  # only for typing; the leaf must not import the heavy grid
    from montagewright.grounding import BeatGrid, Cue
    from montagewright.transcript import Word
    from .measure.shots import ShotManifest


EventKind = Literal[
    "cut",  # scene boundary, from scdet decoded PTS
    "word_start",  # a word began, from the recogniser
    "word_end",  # a word finished
    "silence_edge",  # the far edge of a measured pause (where an out-point is safe)
    "beat",  # ordinary musical beat
    "accent",  # accented beat
    "downbeat",  # first beat of a bar
    "section",  # section boundary (verse/chorus turn)
    "motion_start",  # a camera/subject move began (reserved: Vision optical flow)
    "motion_settle",  # a move came to rest (reserved)
    "action_start",  # a tracked action began
    "action_complete",  # a tracked action finished
]

# Which purpose each measured kind can serve as a snap target. The music kinds
# only ever answer a music_cut; spoken kinds answer picture in/out points.
_IN_POINT_KINDS = ("cut", "word_start", "action_start", "motion_start")
_OUT_POINT_KINDS = ("silence_edge", "word_end", "action_complete", "motion_settle")
_MUSIC_KINDS = ("beat", "accent", "downbeat", "section")

SnapPurpose = Literal["in_point", "out_point", "music_cut"]


class Event(FrozenStrictModel):
    """One measured moment, addressable by a stable id.

    ``source_seconds`` is on the clip's own clock -- the same clock a span is
    read against -- so a clip shot as its own take needs no global timeline.
    ``source_pts`` is kept when the detector preserved it, so a consumer that
    wants the exact decoded frame does not have to re-derive it from seconds.
    """

    event_id: str
    clip_id: str
    kind: EventKind
    source_seconds: float = Field(ge=0.0)
    source_pts: int | None = None
    strength: float = Field(default=0.0, ge=0.0)
    label: str = ""


def _event_id(kind: str, clip_id: str, index: int) -> str:
    return f"{kind}:{clip_id}:{index:03d}"


class EventTable:
    """The measured events for one clip, grouped by kind.

    Build it from whatever detectors ran -- each ``from_*`` adds a layer, and a
    clip with no speech simply contributes no spoken layer. Nothing here re-runs
    a detector; it only files results that already exist.
    """

    def __init__(self, clip_id: str) -> None:
        self.clip_id = clip_id
        self._events: list[Event] = []
        self._by_id: dict[str, Event] = {}

    # -- collection -------------------------------------------------------

    def _add(self, kind: EventKind, seconds: float, *, pts: int | None = None,
             strength: float = 0.0, label: str = "") -> Event:
        index = sum(1 for event in self._events if event.kind == kind)
        event = Event(
            event_id=_event_id(kind, self.clip_id, index),
            clip_id=self.clip_id, kind=kind, source_seconds=float(seconds),
            source_pts=pts, strength=strength, label=label,
        )
        self._events.append(event)
        self._by_id[event.event_id] = event
        return event

    def add_cuts(self, manifest: "ShotManifest") -> "EventTable":
        """Scene boundaries as cut events, carrying the decoded PTS.

        A boundary's ``frame_time_ms`` is already the clip-local millisecond the
        cut was decoded at, so it goes straight onto the source clock.
        """

        for boundary in manifest.boundaries:
            self._add(
                "cut", boundary.frame_time_ms / 1000.0,
                pts=boundary.frame_pts, strength=boundary.score,
                label=boundary.boundary_id,
            )
        return self

    def add_words(self, words: "Iterable[Word]") -> "EventTable":
        """Word starts and ends from the recogniser (measured, never a guess)."""

        for word in words:
            self._add("word_start", word.starts_seconds, label=getattr(word, "text", ""))
            self._add("word_end", word.ends_seconds, label=getattr(word, "text", ""))
        return self

    def add_silence_edges(self, edges: "Iterable[float]") -> "EventTable":
        """The far edges of measured pauses -- the only safe place for an out-point."""

        for edge in edges:
            self._add("silence_edge", edge)
        return self

    def add_motion(self, *, starts: "Iterable[float]" = (),
                   settles: "Iterable[float]" = ()) -> "EventTable":
        """Reserved for Vision optical-flow move starts/rests (second PR)."""

        for start in starts:
            self._add("motion_start", start)
        for settle in settles:
            self._add("motion_settle", settle)
        return self

    def add_actions(self, *, starts: "Iterable[tuple[float, str]]" = (),
                    completes: "Iterable[tuple[float, str]]" = ()) -> "EventTable":
        for seconds, label in starts:
            self._add("action_start", seconds, label=label)
        for seconds, label in completes:
            self._add("action_complete", seconds, label=label)
        return self

    # -- reading ----------------------------------------------------------

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def of_kind(self, *kinds: str) -> list[Event]:
        wanted = set(kinds)
        return [event for event in self._events if event.kind in wanted]

    def seconds_of(self, *kinds: str) -> list[float]:
        return [event.source_seconds for event in self.of_kind(*kinds)]

    def resolve(self, event_id: str) -> float | None:
        """A named moment back to its source clock, for an ``event_ref`` contract.

        Returns ``None`` for an id this clip does not carry, so a caller can fall
        back to snapping a coarse second rather than trust a name that missed.
        """

        event = self._by_id.get(event_id)
        return None if event is None else event.source_seconds

    # -- snapping (delegates policy, never invents it) --------------------

    def snap(self, seconds: float, *, purpose: SnapPurpose,
             grid: "BeatGrid | None" = None, within: float | None = None) -> float:
        """Pull a coarse second onto a measured moment, per that moment's policy.

        ``in_point`` and ``out_point`` defer to the spoken-boundary snappers in
        ``transcript`` (nearest silence, and the after-the-pause rule that keeps
        a word whole). ``music_cut`` defers to ``BeatGrid.nearest_cue``, whose
        priority order (section, downbeat, accent, beat) is the reason a cut
        does not simply land on the closest beat. No policy is duplicated here.
        """

        from montagewright import transcript  # local: avoid import cycle

        if purpose == "music_cut":
            if grid is None:
                return seconds
            cue: Cue | None = grid.nearest_cue(seconds)
            return seconds if cue is None else cue.time_seconds
        if purpose == "out_point":
            candidates = self.seconds_of(*_OUT_POINT_KINDS)
            if within is None:
                return transcript.snap_end(seconds, candidates)
            return transcript.snap_end(seconds, candidates, within=within)
        # in_point
        candidates = self.seconds_of(*_IN_POINT_KINDS)
        if within is None:
            return transcript.snap(seconds, candidates)
        return transcript.snap(seconds, candidates, within=within)


def build_event_table(
    clip_id: str,
    *,
    shots: "ShotManifest | None" = None,
    transcript_payload: "dict | None" = None,
) -> EventTable:
    """Assemble one clip's table from the artifacts a run already produced.

    This is the connective tissue between the detectors and the table: it reads
    the shapes already on disk -- the ``scdet`` manifest, and the transcript
    card carrying the recogniser's words and the SpeechDetector's measured
    silences -- rather than re-running anything. A clip with no transcript
    simply lands its scene cuts and nothing spoken; a clip with no speech in a
    transcript that exists lands its cuts and whatever silence the detector saw.

    Music is deliberately not folded in here: its cues live on the track's own
    global clock, not this clip's, and the music snap takes the ``BeatGrid``
    directly.
    """

    table = EventTable(clip_id)
    if shots is not None:
        table.add_cuts(shots)
    if transcript_payload:
        from montagewright import transcript as _t  # local: avoid import cycle

        table.add_words(_t.words_of(transcript_payload))
        # The far edge of each measured pause is the only safe out-point; the
        # detector reports it directly, so we do not re-derive it from word gaps.
        table.add_silence_edges(
            silence["ends_seconds"]
            for silence in _t.detector_silences(transcript_payload)
        )
    return table


def tables_by_clip(tables: Sequence[EventTable]) -> dict[str, EventTable]:
    """Index a run's per-clip tables by clip id for ``event_ref`` resolution."""

    return {table.clip_id: table for table in tables}
