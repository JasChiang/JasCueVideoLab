"""Resolve symbolic intent into exact times.

The planner says "eight beats, cut on the beat, land on the chorus". It never
says 12.484 seconds. Names and counts resolve here, against a beat grid the
local analyser measured, so the grid owns timing and the planner owns meaning.

The previous system worked the other way: the model sent seconds, and a solver
tried afterwards to slide each cut onto a nearby beat. Measured on real
material, that produced zero aligned boundaries out of eleven -- not because
beats were scarce (the nearest one was three milliseconds away in one case)
but because durations had been stretched to their maximum to hit a target
length, leaving no room to move. Timing derived from the grid cannot get into
that position, because it never has to move anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from montagewright.schema import EDL, Clip

# Cue kinds worth cutting on. A plain beat is included because an eight-beat
# clip has to land somewhere even when no accent falls there.
CUTTABLE = frozenset({"section_boundary", "downbeat", "accent", "beat"})

# Which of them a cut would rather land on, when several are within reach.
# Every one of these is a musical event and they are not interchangeable: a
# bar resolves on its downbeat, and a cut on the beat before it changes the
# picture while the music is still leaning forward.
CUT_PREFERENCE = {
    "section_boundary": 0,
    "downbeat": 1,
    "accent": 2,
    "beat": 3,
}


@dataclass(frozen=True)
class Cue:
    cue_id: str
    time_seconds: float
    kind: str
    strength: float = 0.0


@dataclass(frozen=True)
class BeatGrid:
    """Measured musical time. Everything symbolic resolves against this."""

    bpm: float
    meter: int
    cues: tuple[Cue, ...]
    duration_seconds: float

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm

    def as_heard(
        self,
        music_from_seconds: float = 0.0,
        music_spans: "list[tuple[float, float]] | None" = None,
    ) -> "BeatGrid":
        """The same music, timed against the film instead of the file.

        Grounding counts from zero at the first frame and asks this grid
        where the nearest musical event is. The cues are stamped in the
        track's own time, and the bed does not have to start at the track's
        beginning -- the rhythm pass is told outright that a thirty-second
        cut rarely wants the first thirty seconds of a two-minute piece.

        Nothing joined the two. Pick a start of 0:30 and the film's fourth
        second was matched against the track's fourth second while the
        thirty-fourth was playing, so every cut was placed against music that
        was not sounding. It never showed because the pass has so far always
        left the start at zero -- and the field exists to be moved.

        A cue that is not in what gets played is dropped rather than shifted
        to where it would have been. It is not audible, so it is not a place
        to cut.
        """

        spans = [
            (float(a), float(b)) for a, b in (music_spans or [])
            if float(b) > float(a)
        ]
        if not spans:
            first = max(0.0, float(music_from_seconds))
            if first <= 0.0:
                return self
            spans = [(first, first + self.duration_seconds)]

        moved: list[Cue] = []
        played = 0.0
        for begins, ends in spans:
            for cue in self.cues:
                if begins - 1e-6 <= cue.time_seconds <= ends + 1e-6:
                    moved.append(
                        Cue(
                            cue_id=cue.cue_id,
                            time_seconds=round(
                                played + cue.time_seconds - begins, 6
                            ),
                            kind=cue.kind,
                            strength=cue.strength,
                        )
                    )
            played += ends - begins
        return BeatGrid(
            bpm=self.bpm,
            meter=self.meter,
            cues=tuple(sorted(moved, key=lambda one: one.time_seconds)),
            duration_seconds=played,
        )

    @property
    def named_points(self) -> dict[str, float]:
        """Section names the planner may target by name."""

        return {
            cue.cue_id: cue.time_seconds
            for cue in self.cues
            if cue.kind == "section_boundary"
        }


    def cuttable(self) -> tuple[Cue, ...]:
        return tuple(cue for cue in self.cues if cue.kind in CUTTABLE)

    def cue(self, cue_id: str | None) -> Cue | None:
        """One cue by name, so a report can say what a cut landed on."""

        return next((one for one in self.cues if one.cue_id == cue_id), None)

    def phrase_seconds(self, bars: int = 4) -> float:
        """How long a musical phrase runs, in seconds.

        Four bars by default, which is what most popular music is built in.
        A join anywhere else in the bar is audible however clean the splice.
        """

        return self.seconds_per_beat * self.meter * max(1, bars)

    def on_phrase(self, seconds: float, *, bars: int = 4) -> float:
        """The nearest phrase line to this moment.

        Where a cut inside the music has to land. A section boundary is
        better still when there is one close by -- the analyser found those
        from energy, so they are where the music itself changes -- and this
        falls back to the grid when there is not.
        """

        near = [
            point for point in self.named_points.values()
            if abs(point - seconds) <= self.phrase_seconds(bars) / 2
        ]
        if near:
            return min(near, key=lambda point: abs(point - seconds))
        span = self.phrase_seconds(bars)
        return round(max(0.0, round(seconds / span) * span), 3)

    def nearest_cue(self, seconds: float) -> Cue | None:
        """The best place to cut near here, which is not the closest one.

        This took the nearest cue by distance and nothing else, and the cue
        list is roughly three hundred beats against seventy-six downbeats, so
        the nearest one is almost always an ordinary beat. Measured on a
        finished cut: every one of the first six shots ended on the fourth
        beat of its bar -- one beat before the bar turned over, every time.
        The report called that sixteen cuts out of sixteen landed on a
        musical event, which was true and told nobody anything.

        A cut one beat early does not read as slightly early. The bar has not
        resolved, so the picture changes while the music is still leaning
        forward, and the whole film feels like it is pulling against the
        track rather than sitting on it.

        So a stronger event half a beat further away wins. Within one beat
        either side -- close enough that no length changes much -- the order
        is section, downbeat, accent, beat, and distance only breaks ties.
        Outside that window nothing is preferred, because a downbeat two
        beats away is a different edit, not the same edit placed better.
        """

        candidates = self.cuttable()
        if not candidates:
            return None
        room = self.seconds_per_beat
        near = [
            cue for cue in candidates
            if abs(cue.time_seconds - seconds) <= room + 1e-6
        ]
        if not near:
            return min(
                candidates, key=lambda cue: abs(cue.time_seconds - seconds)
            )
        return min(
            near,
            key=lambda cue: (
                CUT_PREFERENCE.get(cue.kind, len(CUT_PREFERENCE)),
                -cue.strength,
                abs(cue.time_seconds - seconds),
            ),
        )

    def cue_after(self, seconds: float) -> Cue | None:
        later = [
            cue for cue in self.cuttable() if cue.time_seconds > seconds + 1e-6
        ]
        return later[0] if later else None


def beat_grid_payload(grid: BeatGrid) -> dict:
    """A small, stable film-clock artifact shared by Web and CLI graphics."""

    return {
        "version": 1,
        "bpm": grid.bpm,
        "meter": grid.meter,
        "duration_seconds": grid.duration_seconds,
        "cues": [
            {
                "cue_id": cue.cue_id,
                "time_seconds": cue.time_seconds,
                "kind": cue.kind,
                "strength": cue.strength,
            }
            for cue in grid.cues
        ],
    }


def beat_grid_from_payload(payload: dict) -> BeatGrid:
    """Read the runtime format written beside a rendered cut."""

    return BeatGrid(
        bpm=float(payload["bpm"]),
        meter=int(payload.get("meter") or 4),
        cues=tuple(
            Cue(
                cue_id=str(entry["cue_id"]),
                time_seconds=float(entry["time_seconds"]),
                kind=str(entry["kind"]),
                strength=float(entry.get("strength") or 0.0),
            )
            for entry in payload.get("cues", [])
        ),
        duration_seconds=float(payload.get("duration_seconds") or 0.0),
    )


def read_runtime_beat_grid(path: Path) -> BeatGrid | None:
    if not path.exists():
        return None
    return beat_grid_from_payload(json.loads(path.read_text(encoding="utf-8")))


def shots_in(path: Path, *, threshold: float = 0.3) -> list[tuple[float, float]]:
    """Where an already-edited piece changes shot.

    A folder of rushes is one take per file. Something already cut is one
    file holding many, and handing it over whole means one card describing
    five minutes, one transcript, and a planner picking windows out of a
    single source as though the cuts inside it were not there. These are
    boundaries the material already has; nothing is being invented.

    A genuinely continuous take comes back as one span, which is the honest
    answer for a locked-off interview.
    """

    import subprocess

    completed = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-filter_complex",
            f"select='gt(scene,{threshold})',metadata=print:file=-",
            "-an", "-f", "null", "-",
        ],
        capture_output=True, text=True, check=False,
    )
    cuts = [
        float(line.split("pts_time:")[1].split()[0])
        for line in completed.stdout.splitlines()
        if "pts_time:" in line
    ]
    duration = _duration_of(path)
    edges = [0.0] + sorted(cuts) + [duration]
    return [
        (start, end)
        for start, end in zip(edges, edges[1:])
        if end - start >= 2.0
    ]


def _duration_of(path: Path) -> float:
    import json as _json
    import subprocess

    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(_json.loads(completed.stdout)["format"]["duration"])


def analyse_track(path: Path) -> BeatGrid:
    """Measure a track straight into a grid.

    The reviewed lock exists so a delivery can prove which analysis it was
    cut against. That provenance is worth its ceremony for a delivery and is
    pure friction for someone dropping a folder on a page to see what comes
    out, who would otherwise have to produce a lock file before the tool
    would run at all. The measurement underneath is the same one.
    """

    from montagewright.measure.music import analyze_music

    proposal = analyze_music(Path(path).expanduser().resolve())
    rate = float(proposal.master_sample_rate)
    meter = int(proposal.meter_suggestion or 4)

    # The proposal calls everything a candidate because a human was meant to
    # settle it. Nothing about the measurement changes when they do -- the
    # review decides which to keep, not when they are.
    cues = [
        Cue(
            cue_id=str(cue.cue_id),
            time_seconds=float(cue.time_ms) / 1000.0,
            kind=str(cue.kind).removesuffix("_candidate"),
            strength=float(cue.strength or 0.0),
        )
        for cue in proposal.cues
    ]
    # Downbeats and section boundaries are derived while locking rather than
    # measured, so they have to be derived here too: without them the grid
    # has nothing a planner can name, and every "land on the chorus" resolves
    # to nothing.
    beats = sorted(
        (cue for cue in cues if cue.kind == "beat"),
        key=lambda cue: cue.time_seconds,
    )
    for index, beat in enumerate(beats):
        if index % meter == 0:
            cues.append(
                Cue(
                    cue_id=f"downbeat-{index // meter:05d}",
                    time_seconds=beat.time_seconds,
                    kind="downbeat",
                    strength=beat.strength,
                )
            )
    for section in proposal.sections:
        cues.append(
            Cue(
                cue_id=str(section.label or section.section_id),
                time_seconds=float(section.start_sample) / rate,
                kind="section_boundary",
                strength=float(section.confidence or 0.0),
            )
        )
    return BeatGrid(
        bpm=float(proposal.estimated_bpm or 120.0),
        meter=meter,
        cues=tuple(sorted(cues, key=lambda cue: cue.time_seconds)),
        duration_seconds=float(proposal.duration_ms) / 1000.0,
    )


def load_beat_grid(lock_path: Path) -> BeatGrid:
    """Read a reviewed music map lock into a grid.

    The lock is the old package's format and is genuinely a measurement, so it
    is read directly rather than re-derived.
    """

    payload = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    cues = tuple(
        Cue(
            cue_id=str(entry["cue_id"]),
            time_seconds=float(entry["time_ms"]) / 1000.0,
            kind=str(entry["kind"]),
            strength=float(entry.get("strength") or 0.0),
        )
        for entry in payload.get("cues", [])
    )
    return BeatGrid(
        bpm=float(payload["bpm"]),
        meter=int(payload.get("meter") or 4),
        cues=cues,
        duration_seconds=float(payload.get("duration_ms", 0)) / 1000.0,
    )


@dataclass
class GroundedClip:
    """A clip with its place on the timeline decided."""

    clip: Clip
    timeline_in_seconds: float
    timeline_out_seconds: float
    landed_on: str | None = None
    # What kind of musical event that was. The id alone cannot be read back:
    # `mc-00040` is a position in a sorted list of candidates of every kind,
    # so a report saying nine cuts of nine landed on the music was true and
    # unauditable -- the run before this one put six consecutive cuts on the
    # fourth beat of the bar and reported sixteen out of sixteen. Ranking
    # cues by strength fixed the cutting; nothing made the claim checkable.
    landed_kind: str | None = None
    note: str | None = None
    # Set when a camera move lengthened this shot past what the rhythm asked
    # for, so the report can say why the cut runs where it does.
    move_too_short: str | None = None

    @property
    def duration_seconds(self) -> float:
        return self.timeline_out_seconds - self.timeline_in_seconds


@dataclass
class GroundedTimeline:
    clips: list[GroundedClip]
    grid: BeatGrid | None

    @property
    def duration_seconds(self) -> float:
        return self.clips[-1].timeline_out_seconds if self.clips else 0.0

    @property
    def aligned_count(self) -> int:
        """Cuts that landed on a musical event, for the delivery report."""

        return sum(1 for clip in self.clips if clip.landed_on is not None)


def _requested_duration(clip: Clip, grid: BeatGrid | None) -> float:
    """How long this clip wants to be, in seconds.

    The length is the planner's, whole. A flat per-move floor used to raise it
    here, which reads as safety and is a length decision made by a constant:
    how long a sweep needs depends on how far it travels and how much is on
    the way, and the planner is the one who watched the shot. What the floor
    is good for is saying afterwards that the move could not happen in the
    time it was given -- reported, not corrected.
    """

    if grid is not None and clip.music_sync.beats:
        return clip.music_sync.beats * grid.seconds_per_beat
    return clip.approx_out_seconds - clip.approx_in_seconds


def _floor_for(clip: Clip) -> float:
    """The least time this clip's own move can happen in.

    Estimated from the card's subject positions, which are what is known
    before grounding runs. A clip whose looks cannot be located falls back to
    the move's declared floor, because an unknown distance is not a zero one.
    """

    from montagewright.capabilities import MOVE_FLOORS
    from montagewright.reframe import seconds_needed_for

    reframe = clip.reframe
    if reframe is None or len(reframe.looks) < 2:
        return 0.0

    seen = reframe.look_boxes
    if not seen or len(seen) < len(reframe.looks):
        return MOVE_FLOORS.get(reframe.camera_move, 0.0)

    stops = [
        (one.seconds, where[0], where[1], where[2])
        for one, where in zip(reframe.looks, seen)
    ]
    return seconds_needed_for(stops, reframe.camera_energy)


def ground_timeline(edl: EDL, grid: BeatGrid | None) -> GroundedTimeline:
    """Lay the clips out in time.

    Each clip starts where the previous one ended, so the cut is continuous by
    construction. Its end is the next musical event at or after its requested
    length -- so a beat count is honoured exactly when the grid agrees, and
    rounded up to the next event when it does not, rather than being nudged
    off the grid to hit an arithmetic total.
    """

    grounded: list[GroundedClip] = []
    # The grid measured the file; grounding counts from the first frame. When
    # the bed does not start at the track's beginning those are two different
    # clocks, and every cue lookup below is in film time.
    if grid is not None:
        grid = grid.as_heard(
            getattr(edl, "music_from_seconds", 0.0),
            list(getattr(edl, "music_spans", []) or []),
        )

    # Reaching a beat is a reason to hold a shot longer; running out of
    # things to look at is a reason not to. The rhythm pass already refuses
    # to stretch a shot past the content its source proves, and then this
    # function rounded the same shot up to the next cue and undid it -- by
    # 0.34s on one shot and 0.07s on another, which the release gate
    # correctly refused, and a thirty-second film with eight good shots in
    # it was never rendered because two frames of it were unsupported.
    #
    # Which is the ordinary case, not an edge: with no speech every shot's
    # whole length is visual-only, so every shot sits at its ceiling and the
    # next cue is always past it. What an editor does here is cut on the
    # earlier beat -- short reads as intent, held-too-long reads as a
    # mistake -- and leaves the cut off the grid when there is no earlier
    # beat to take.
    has_independent_audio = bool(getattr(edl, "audio_clips", ()) or ())

    def content_ceiling(clip: Clip) -> float | None:
        claim = getattr(clip, "coverage_claim_seconds", None)
        if claim is None or has_independent_audio:
            return None
        if str(getattr(clip, "audio_role", "discard")) != "discard":
            return None
        return float(claim)

    cursor = 0.0

    for clip in edl.clips:
        wanted = _requested_duration(clip, grid)
        move = clip.reframe.camera_move if clip.reframe else "hold"
        # What this shot needs, from this shot -- the rests it asked for plus
        # the distance between its looks at the speed its energy allows. The
        # flat per-move number this replaced said every pan needs 2.5s, and
        # the real floor for the same word runs from about one second to over
        # five: it forbade a short pan across a narrow gap and permitted a
        # long one that could never arrive.
        #
        # Distances come from where the card measured the subjects, so this
        # is the estimate available before anything is grounded. The executor
        # measures again and reports against what it finds.
        floor = _floor_for(clip)
        # Not a correction. The move stays as asked and runs in the time it
        # was given; this says it will not read, so the report and the review
        # round see it instead of a shot that quietly arrives too fast.
        too_short = (
            f"{move} across this shot needs about {floor:.1f}s and has "
            f"{wanted:.2f}s"
            if floor > 0.0 and wanted < floor - 1e-6
            else None
        )
        end = cursor + wanted
        landed: str | None = None
        landed_kind: str | None = None
        note: str | None = None

        # A moment inside the shot, put where the music is. Only the cut was
        # ever aligned, so "the phone snaps shut on the downbeat and the shot
        # runs on" could not be asked for -- and a model that chose a length
        # so the action completed at the cut had that taken away again by the
        # snap, which moves the out-point by up to a beat. The two goals were
        # fighting through the same one control.
        #
        # Solved by moving the in-point rather than the length: the shot is
        # as long as the rhythm pass said, and it starts wherever it has to
        # for the named moment to fall on the beat.
        if grid is not None and clip.music_sync.anchor:
            moved, why = _put_anchor_on_the_music(clip, grid, cursor)
            if moved is not None:
                clip = clip.model_copy(update={
                    "approx_in_seconds": moved,
                    "approx_out_seconds": moved + wanted,
                })
            if why:
                note = why

        # A named point is an instruction, not a hint. `resolve_sync_point`
        # has been here since the grid was written and nothing ever called
        # it, so a shot asked to land on `chorus_1_start` was placed by
        # `nearest_cue` like any other -- which finds the nearest event to
        # the requested length, and the requested length is exactly the
        # thing being overridden. The planner named a moment in the music;
        # the answer is that moment, or a note saying the track has no such
        # moment.
        wanted_point: float | None = None
        if grid is not None and clip.music_sync.sync_to:
            wanted_point = resolve_sync_point(grid, clip.music_sync.sync_to)
            if wanted_point is None:
                note = (
                    f"no point called '{clip.music_sync.sync_to}' in this "
                    f"track; placed on the nearest event instead"
                )
            elif wanted_point <= cursor + 1e-6:
                note = (
                    f"'{clip.music_sync.sync_to}' sits at "
                    f"{wanted_point:.2f}s, before this shot begins; placed "
                    f"on the nearest event instead"
                )
                wanted_point = None

        if wanted_point is not None:
            end = wanted_point
            landed = clip.music_sync.sync_to
            found = grid.cue(landed) if grid is not None else None
            landed_kind = found.kind if found is not None else "named"
            asked_for = cursor + wanted
            if abs(end - asked_for) > grid.seconds_per_beat:
                note = (
                    f"held to '{clip.music_sync.sync_to}' at {end:.2f}s, "
                    f"{abs(end - asked_for):.2f}s from the "
                    f"{wanted:.2f}s this shot asked for"
                )
        elif grid is not None and clip.music_sync.cut_on_beat:
            cue = grid.nearest_cue(end)
            if cue is not None and cue.time_seconds > cursor + 1e-6:
                drift = abs(cue.time_seconds - end)
                end = cue.time_seconds
                landed = cue.cue_id
                landed_kind = cue.kind
                if drift > grid.seconds_per_beat / 2:
                    note = (
                        f"nearest cue sat {drift:.3f}s from the requested "
                        f"length; took it anyway to stay on the grid"
                    )
            else:
                # Every cue lies behind the cursor, i.e. the music has run
                # out. Keeping the requested length is honest; silence at the
                # tail is the mix's problem, not the edit's.
                note = "past the end of the analysed music; kept as planned"

        keeps_source_move = (
            clip.reframe is not None
            and clip.reframe.editorial_intent == "use_source_motion"
        )
        # Landing on a beat may lengthen a shot that exists to let the
        # source's own move play; it may not shorten one. `nearest_cue`
        # takes the closest event in either direction, so a cue thirteen
        # milliseconds early won, the move lost its ending, and the release
        # check refused the timeline -- over a hundredth of a second, twice
        # in one evening on two different shots. Take the first cue at or
        # after the length the move needs; if none is near, keep the length
        # and let this cut sit off the grid.
        if keeps_source_move and end < cursor + wanted - 1e-6:
            floor_end = cursor + wanted
            later = None
            if grid is not None:
                later = min(
                    (
                        cue for cue in grid.cuttable()
                        if floor_end - 1e-6 <= cue.time_seconds
                        <= floor_end + grid.seconds_per_beat + 1e-6
                    ),
                    key=lambda cue: cue.time_seconds,
                    default=None,
                )
            missed = landed
            if later is not None:
                end, landed, landed_kind = (
                    later.time_seconds, later.cue_id, later.kind
                )
            else:
                end, landed, landed_kind = floor_end, None, None
            held = (
                f"the source move needs {wanted:.2f}s, past "
                + (f"cue {missed}" if missed else "the nearest cue")
                + "; "
                + (
                    f"held to {landed} instead"
                    if landed else "kept the move whole and left the grid"
                )
            )
            note = f"{note}; {held}" if note else held

        # Content evidence is a bound of the same kind as source length, and
        # belongs beside it: one says the take has no more picture, the other
        # says the picture has no more to show. Take the last cue that fits,
        # so the cut stays on the grid; if the only cues that fit would cut
        # the move short, keep the planned length and lose the beat. Rhythm
        # is the thing that gives way here, not the content.
        ceiling = content_ceiling(clip)
        if ceiling is not None and end - cursor > ceiling + 1e-6:
            room = cursor + ceiling
            earlier = None
            if grid is not None:
                # The beat before, not any beat before. Halving a shot to
                # reach a distant cue is not "cutting a little early", it is
                # a different edit -- and one the rhythm pass never saw.
                # A shot that exists to let the source's own move play has
                # a length that is a statement about that move, not a budget
                # to spend: cutting back to the earlier beat here left an
                # authored three-second pan 2.564s to finish in, which the
                # release check refused -- rightly, and after this function
                # had already been told the shot could show three seconds.
                soonest = max(
                    cursor + max(floor, 1e-3),
                    cursor + wanted if keeps_source_move else 0.0,
                    room - grid.seconds_per_beat,
                )
                earlier = max(
                    (
                        cue for cue in grid.cuttable()
                        if soonest - 1e-6 <= cue.time_seconds <= room + 1e-6
                    ),
                    key=lambda cue: cue.time_seconds,
                    default=None,
                )
            over = end - room
            if earlier is not None:
                end = earlier.time_seconds
                landed = earlier.cue_id
                landed_kind = earlier.kind
                cut_short = (
                    f"held {over:.2f}s past what this shot can show; took "
                    f"the earlier cue {earlier.cue_id} instead"
                )
            else:
                end = max(room, cursor + wanted) if keeps_source_move else room
                landed = None
                landed_kind = None
                cut_short = (
                    f"no cue inside the {ceiling:.2f}s this shot can show; "
                    "kept the supported length and left this cut off the grid"
                )
            note = f"{note}; {cut_short}" if note else cut_short

        # Feasibility belongs in the timeline calculation. Clamping only
        # after every cut had been laid out made all later cuts move earlier
        # in the rendered film while their beat labels and reported times
        # stayed on the impossible, longer timeline.
        window = clip.usable_window
        if window is not None:
            available = max(1e-3, window[1] - clip.approx_in_seconds)
            feasible_end = cursor + available
            if end > feasible_end:
                missed = landed
                end = feasible_end
                landed = None
                landed_kind = None
                feasibility_note = (
                    f"usable source ends after {available:.2f}s; could not "
                    + (f"reach cue {missed}" if missed else "hold the requested length")
                )
                note = f"{note}; {feasibility_note}" if note else feasibility_note

        grounded.append(
            GroundedClip(
                clip=clip,
                timeline_in_seconds=cursor,
                timeline_out_seconds=end,
                landed_on=landed,
                landed_kind=landed_kind,
                note=note,
                move_too_short=too_short,
            )
        )
        cursor = end

    return GroundedTimeline(clips=grounded, grid=grid)


def apply_to_edl(edl: EDL, timeline: GroundedTimeline) -> EDL:
    """Rewrite each clip's source window to the length the grid decided.

    The in-point is what the planner chose -- that is a statement about the
    material. The out-point follows from the grounded duration, because a cut
    is a straight copy: one second of timeline is one second of source.

    A source that cannot supply the full grounded length is left as it is. The
    executor clamps it and records the shortfall, which keeps the shortening
    visible instead of silently reopening it here.
    """

    grounded_by_id = {entry.clip.clip_id: entry for entry in timeline.clips}
    rewritten = []
    for clip in edl.clips:
        grounded = grounded_by_id.get(clip.clip_id)
        if grounded is None:
            rewritten.append(clip)
            continue
        # `ground_timeline` may move the source in-point to put an action
        # anchor on the music. Carry that grounded clip forward; copying only
        # its duration made the report claim the anchor landed while render
        # still read the planner's old in-point.
        clip = grounded.clip
        out = clip.approx_in_seconds + grounded.duration_seconds
        # Not past the end of what the take is worth using. Reaching a beat
        # is a good reason to hold a shot longer and it is not a good enough
        # one to run into the part where the camera is being repositioned or
        # somebody says "again" -- and the layer that clamps lengths below
        # this one only knows how long the file is, so a second of a reset
        # and a second of a take look identical to it.
        window = clip.usable_window
        if window is not None and out > window[1]:
            out = max(clip.approx_in_seconds + 1e-3, window[1])
        rewritten.append(
            clip.model_copy(update={"approx_out_seconds": out})
        )
    return edl.model_copy(update={"clips": rewritten})


def _put_anchor_on_the_music(
    clip: Clip, grid: BeatGrid, cursor: float
) -> tuple[float | None, str | None]:
    """Where this shot has to begin for its anchored moment to hit the beat.

    The moment sits at a known second of the source. Shift the in-point and
    it slides along the film; the arithmetic is one line. What makes this
    worth a function is everything it must refuse.

    A shot may not begin outside the stretch of take it was cut from, and it
    may not run past the end of it, so the in-point that would put the moment
    on the beat is sometimes not available. That is reported rather than
    approximated: an anchor half honoured is a cut nobody asked for, placed
    where nothing asked for it.
    """

    wanted = clip.music_sync.anchor
    at = clip.moments.get(str(wanted))
    if at is None:
        return None, f"no moment called '{wanted}' in this take"

    length = clip.approx_out_seconds - clip.approx_in_seconds
    shows_at = cursor + (at - clip.approx_in_seconds)
    kind = clip.music_sync.anchor_lands_on
    candidates = [one for one in grid.cues if one.kind == kind]
    if not candidates:
        return None, f"this track has no {kind} to land '{wanted}' on"
    cue = min(candidates, key=lambda one: abs(one.time_seconds - shows_at))

    # `before` and `after` are a beat either side, which is what an editor
    # means by cutting deliberately loose.
    aimed = cue.time_seconds + {
        "before": -grid.seconds_per_beat,
        "after": grid.seconds_per_beat,
    }.get(clip.music_sync.anchor_relation, 0.0)

    begins = at - (aimed - cursor)
    window = clip.usable_window
    if window is not None and not (
        window[0] - 1e-6 <= begins and begins + length <= window[1] + 1e-6
    ):
        return None, (
            f"putting '{wanted}' on the {kind} needs this shot to start at "
            f"{begins:.2f}s, which is outside the {window[0]:.1f}-"
            f"{window[1]:.1f}s it may be cut from"
        )
    if begins < 0.0:
        return None, f"putting '{wanted}' on the {kind} needs a start before zero"
    return round(begins, 3), None


def resolve_sync_point(grid: BeatGrid, name: str) -> float | None:
    """Look up a named point such as 'chorus_1_start'.

    Unknown names return None rather than raising: a planner naming a section
    this track does not have is describing intent the material cannot serve,
    which is a note for the report, not a crash.
    """

    points = grid.named_points
    if name in points:
        return points[name]
    lowered = name.lower()
    for cue_id, seconds in points.items():
        if lowered in cue_id.lower():
            return seconds
    return None
