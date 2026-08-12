"""Drop a folder in, watch it cut, read what it decided.

The CLI already prints every stage and writes the whole account to
report.json. What it cannot do is let someone check the result against the
material without a terminal and a video player: the question after a run is
never "did it finish", it is "which take is that, and why is it framed like
that". So this serves the finished film beside the decisions that produced
it -- source file, in and out, the move, the subject, the reason, the
degradations, and the per-shot verdict.

The run itself is the CLI in a subprocess. Importing the pipeline here would
be faster and would put a long-running job in the request thread; a
subprocess is killable, its stdout is already the progress log, and a crash
takes the run down instead of the server.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from montagewright.renderer import probe_duration
from montagewright.schema import looks_of, move_of_shot, subject_of
from montagewright.uploads import default_library

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".MP4", ".MOV", ".avi", ".mkv"}
AUDIO_SUFFIXES = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".aiff", ".MP3", ".M4A", ".WAV"}
BRIEF_SUFFIXES = {".md", ".markdown", ".txt", ".MD", ".TXT"}
SPEC_SUFFIXES = {".json", ".JSON"}
# The names a request may ask for, and what each one is as a ratio. These
# were two different shapes with one name -- a tuple to validate against and
# a dict to look up -- and the lookup silently returned nothing.
ASPECTS = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0, "4:5": 4 / 5}
PAGE = Path(__file__).resolve().parent / "web" / "index.html"
SUBTITLE_FONT_LOCK = threading.Lock()
_GRAPHICS_LOCKS_GUARD = threading.Lock()
_GRAPHICS_STATE_LOCKS: dict[str, threading.Lock] = {}
_GRAPHICS_PREVIEW_LOCKS: dict[str, threading.Lock] = {}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Runs live somewhere they survive a restart. They were in a temp directory
# keyed by an in-memory dict, so closing the server threw away every finished
# cut -- and comparing this run against the last one is most of what anybody
# does with a tool like this.
RUNS_ROOT = Path(
    os.environ.get("MONTAGEWRIGHT_RUNS", Path.home() / ".cache" / "montagewright" / "runs")
)
# A sandboxed/local development server may need to create new runs inside the
# workspace while still showing older CLI/Codex runs from the normal cache.
# These roots are discovery-only: every new run is always written to
# ``RUNS_ROOT``.
LEGACY_RUNS_ROOTS = tuple(
    Path(value).expanduser()
    for value in os.environ.get("MONTAGEWRIGHT_LEGACY_RUNS", "").split(os.pathsep)
    if value.strip()
)
# A browser upload of local material copies it into the browser and writes it
# back out; the bytes were already on disk. Uploading stays for the case where
# they genuinely are not, and that case has a ceiling.
MAX_UPLOAD_BYTES = int(os.environ.get("MONTAGEWRIGHT_MAX_UPLOAD", 4 * 1024**3))
MAX_GROUNDING_UPLOAD_BYTES = int(
    os.environ.get("MONTAGEWRIGHT_MAX_GROUNDING_UPLOAD", 256 * 1024**2)
)
# A few pictures and one structured answer. Its own ceiling because it is
# spent before a run exists, so the run's --budget cannot cover it.
DRAFT_BUDGET_USD = float(os.environ.get("MONTAGEWRIGHT_DRAFT_BUDGET", "0.25"))


@dataclass
class Run:
    """One cut in progress or finished, and everything it said on the way."""

    run_id: str
    root: Path
    lines: list[str] = field(default_factory=list)
    state: str = "running"
    returncode: int | None = None
    process: subprocess.Popen | None = None
    started_at: float = field(default_factory=time.time)
    source: str = ""
    # What was run, so it can be run again into the same place. Everything
    # already paid for -- cards, transcripts, the direction, the selection --
    # is keyed on disk, so a second attempt picks up where the first stopped.
    command: list[str] = field(default_factory=list)

    @property
    def output(self) -> Path:
        return self.root / "out"

    def remember(self) -> None:
        """Write enough to rebuild this run after a restart."""

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "run.json").write_text(
            json.dumps({
                "run_id": self.run_id,
                "state": self.state,
                "started_at": self.started_at,
                "source": self.source,
                "command": self.command,
                "log": self.lines,
            }, ensure_ascii=False),
            encoding="utf-8",
        )

    def report(self) -> dict | None:
        path = self.output / "report.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Being written right now. A half-read report is not an error,
            # it is a "not yet".
            return None


RUNS: dict[str, Run] = {}


def recall() -> None:
    """Pick up runs left by an earlier server."""

    folders: list[Path] = []
    for runs_root in (RUNS_ROOT, *LEGACY_RUNS_ROOTS):
        if not runs_root.exists():
            continue
        try:
            folders.extend(sorted(runs_root.iterdir()))
        except OSError:
            continue
    for folder in folders:
        if folder.name in RUNS or not folder.is_dir():
            continue
        # Either the note this server wrote, or the one the command line
        # leaves beside its output. A cut is a cut however it was started.
        note = folder / "run.json"
        spare = folder / "out" / "command.json"
        if not note.exists() and not spare.exists():
            continue
        try:
            saved = json.loads(
                (note if note.exists() else spare).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
        if not note.exists():
            # A run from the command line keeps its own log beside the
            # output; without it the page shows a state and nothing else.
            spare_log = folder / "out" / "run.log"
            if spare_log.exists() and not saved.get("log"):
                try:
                    saved["log"] = [
                        line for line in
                        spare_log.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ][-500:]
                except OSError:
                    pass
            # A run started from the command line leaves only the note beside
            # its output, and that note is written before the work begins --
            # so "it exists" says the run started, not that it finished. A
            # run that crashed looked identical to one that worked, listed as
            # done with no shots, no spend and no film.
            #
            # The report is written at the end, so its absence is the
            # difference. `interrupted` already means this and already offers
            # to pick the run up again, which is the useful thing to do with
            # one.
            saved.setdefault(
                "state",
                "done" if (folder / "out" / "report.json").exists()
                else "interrupted",
            )
            saved.setdefault(
                "started_at", (folder / "out").stat().st_mtime
            )
        RUNS[folder.name] = Run(
            run_id=folder.name,
            root=folder,
            lines=saved.get("log", []),
            # Anything found on disk is finished as far as this process is
            # concerned: the thing that was running died with the last server.
            state=saved.get("state", "done")
            if saved.get("state") != "running"
            else "interrupted",
            started_at=float(saved.get("started_at", 0.0)),
            source=saved.get("source", ""),
            command=saved.get("command", []),
        )


def _collect(run: Run) -> None:
    """Drain the child's output into the run, and mark how it ended."""

    assert run.process is not None and run.process.stdout is not None
    for line in run.process.stdout:
        text = line.rstrip("\n")
        if text:
            run.lines.append(text)
    run.returncode = run.process.wait()
    run.state = "done" if run.returncode == 0 else "failed"
    run.remember()


def _ran_with(run, flag: str, *, after: bool = True) -> str:
    """What this run was given for a flag, or the folder it was pointed at."""

    command = run.command or []
    if flag not in command:
        return ""
    at = command.index(flag)
    if not after:
        # The rushes are the positional argument straight after `render`.
        return command[at + 1] if at + 1 < len(command) else ""
    return command[at + 1] if at + 1 < len(command) else ""


def _brief_of(run) -> str:
    """The brief this run was written against, if it was kept."""

    path = _brief_path_of(run)
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")[:4000]
    except OSError:
        return ""


def _brief_path_of(run) -> Path | None:
    """The durable Brief path, accepting both CLI flag spellings."""

    command = run.command or []
    for index, argument in enumerate(command):
        if argument == "--brief" and index + 1 < len(command):
            return Path(command[index + 1])
        if argument.startswith("--brief="):
            return Path(argument.split("=", 1)[1])
    return None


def _transcript_map(run) -> dict:
    """Which transcript belongs to which source, for this run.

    Transcripts moved into the shared library when they became worth keeping
    across runs, and they are named for the bytes they describe -- the same
    rule as the cards, for the same reason. Two readers here were still
    looking in the output directory, which nothing has written since, and
    keying by filename, which is a hash. So they found nothing and said
    nothing: the transcript tab was empty for every run, and the subtitles
    were empty for every run, and both looked like "this cut has no speech".
    """

    from montagewright.clipcard import card_map
    from montagewright.transcript import load

    found = card_map(
        run.output / "work" / "proxies",
        default_library() / "transcripts",
    )
    return {
        source_id: card
        for source_id, path in found.items()
        if (card := load(path)) is not None
    }


def _current_timeline(run: Run) -> dict:
    """The committed edit revision, or legacy-empty when none was made."""

    path = run.output / "work" / "current-timeline.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("version") not in {
            "montagewright-current-timeline-v1",
            "montagewright-current-timeline-v2",
        }:
            raise ValueError("unknown version")
        shots = value["shots"]
        if not isinstance(shots, list) or not shots:
            raise ValueError("shots must be a non-empty list")
        original = (run.report() or {}).get("selection", {}).get("shots", [])
        for shot in shots:
            selection = int(shot["selection_index"])
            if not 0 <= selection < len(original):
                raise ValueError(f"selection_index {selection} is out of range")
            if float(shot["seconds"]) <= 0:
                raise ValueError("shot duration must be positive")
        # The public film is CFR and the renderer allocates frame boundaries
        # cumulatively.  Expose that same clock to Web anchors and every
        # recut consumer; authored fractional seconds are not the rendered
        # shot boundaries (for example .515s at 30fps becomes 15 frames).
        from montagewright.executor import allocate_timeline_frames

        fps = int(value.get("output_fps") or 30)
        boundaries = allocate_timeline_frames(
            [float(shot["seconds"]) for shot in shots], fps
        )
        value["shots"] = [
            {
                **shot,
                "start_frame": start,
                "frame_count": end - start,
                "seconds": (end - start) / fps,
            }
            for shot, (start, end) in zip(shots, boundaries, strict=True)
        ]
        audio_assignments = value.setdefault("audio_assignments", [])
        if not isinstance(audio_assignments, list):
            raise ValueError("audio_assignments must be a list")
        picture_frames = boundaries[-1][1]
        seen_audio_ids: set[str] = set()
        narrative: list[tuple[int, int, str]] = []
        for audio in audio_assignments:
            audio_id = str(audio["audio_id"])
            if audio_id in seen_audio_ids:
                raise ValueError(f"duplicate audio assignment {audio_id}")
            seen_audio_ids.add(audio_id)
            start = int(audio["timeline_start_frame"])
            count = int(audio["frame_count"])
            if start < 0 or count <= 0 or start + count > picture_frames:
                raise ValueError(
                    f"audio assignment {audio_id} is outside the picture timeline"
                )
            if float(audio["out_seconds"]) <= float(audio["in_seconds"]):
                raise ValueError(f"audio assignment {audio_id} has no source duration")
            if str(audio.get("role", "narrative")) == "narrative":
                narrative.append((start, start + count, audio_id))
        narrative.sort()
        for previous, here in zip(narrative, narrative[1:]):
            if here[0] < previous[1]:
                raise ValueError(
                    f"narrative audio assignments {previous[2]} and {here[2]} overlap"
                )
        return value
    except (OSError, ValueError, TypeError, KeyError) as error:
        # This file is the commit record for the public MP4. Falling back to
        # report.json would silently put the UI/NLE/subtitles on an old cut.
        raise HTTPException(422, f"current timeline is unreadable: {error}")


def _invalidate_graphics_delivery(run: Run) -> None:
    """A saved copy/design change makes every prior graphics export stale."""

    for name in (
        "deliverable-graphics.mp4",
        "deliverable-graphics-subtitled.mp4",
        "graphics-overlay.mov",
        "timeline.xml",
        "timeline.fcpxml",
    ):
        (run.output / name).unlink(missing_ok=True)
    (run.output / "work" / "graphics-render" / "layout.json").unlink(
        missing_ok=True
    )


def _invalidate_subtitle_delivery(run: Run) -> None:
    """A subtitle edit invalidates every artifact that embeds its old text.

    A graphics-only file is included deliberately.  The public graphics
    endpoint falls back to it when the combined file is absent; retaining it
    after captions are added or changed would make that fallback look like a
    current combined export.  Graphics pixels can be rebuilt from the saved
    plan, while an incorrectly labelled delivery cannot be repaired by its
    viewer.
    """

    for name in (
        "deliverable-subtitled.mp4",
        "deliverable-graphics.mp4",
        "deliverable-graphics-subtitled.mp4",
        "graphics-overlay.mov",
        "timeline.xml",
        "timeline.fcpxml",
    ):
        (run.output / name).unlink(missing_ok=True)
    (run.output / "work" / "graphics-render" / "layout.json").unlink(
        missing_ok=True
    )


def _keyed_graphics_lock(
    locks: dict[str, threading.Lock], key: str,
) -> threading.Lock:
    """Return one process-local writer lock for a durable server resource."""

    with _GRAPHICS_LOCKS_GUARD:
        return locks.setdefault(key, threading.Lock())


def _graphics_state_lock(run: Run) -> threading.Lock:
    return _keyed_graphics_lock(
        _GRAPHICS_STATE_LOCKS,
        str((run.output / "work" / "graphics.json").resolve()),
    )


def _graphics_preview_lock(metadata_path: Path) -> threading.Lock:
    return _keyed_graphics_lock(
        _GRAPHICS_PREVIEW_LOCKS, str(metadata_path.resolve())
    )


def _cached_preview_png_is_valid(path: Path) -> bool:
    """A cache hit must name a complete PNG, not merely an existing inode."""

    try:
        with path.open("rb") as cached:
            return cached.read(len(_PNG_SIGNATURE)) == _PNG_SIGNATURE
    except FileNotFoundError:
        return False


def _load_graphics_preview_cache(
    metadata_path: Path, preview_dir: Path, graphic_id: str,
) -> dict | None:
    """Load a complete content-addressed preview or report a cache miss.

    Malformed or incomplete records are recoverable misses and are rebuilt.
    An operating-system I/O error is different: retrying the same write while
    storage is unavailable can overwrite useful evidence, so callers surface
    it as a temporary service failure instead.
    """

    try:
        raw = metadata_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    metadata = None
    try:
        candidate = json.loads(raw)
        if isinstance(candidate, dict):
            metadata = candidate
    except (ValueError, TypeError):
        return None
    if metadata is None:
        return None
    joint = metadata.get("joint_layout")
    if not isinstance(joint, dict) or graphic_id not in joint:
        return None
    for item in joint.values():
        if not isinstance(item, dict):
            return None
        filename = Path(str(item.get("url") or "")).name
        if not re.fullmatch(r"[0-9a-f]{64}\.png", filename):
            return None
        if not _cached_preview_png_is_valid(preview_dir / filename):
            return None
    return metadata


def _graphics_preview_problem(
    status_code: int,
    error_code: str,
    message: str,
    *,
    field: str = "preview",
    suggested_patch: dict[str, Any] | None = None,
) -> HTTPException:
    """A stable machine-readable error shared by editor retry/repair paths."""

    return HTTPException(
        status_code=status_code,
        detail={
            "error_code": error_code,
            "message": message,
            "field": field,
            "suggested_patch": suggested_patch or {},
        },
    )


def _graphics_validation_problem(
    error: ValueError, graphic_id: str,
) -> HTTPException:
    """Translate renderer validation into a minimal actionable contract."""

    message = str(error)
    lowered = message.lower()
    prefix = f"cues.{graphic_id}"
    if "contrast" in lowered:
        return _graphics_preview_problem(
            422, "GRAPHIC_CONTRAST_FAILED", message,
            field=f"{prefix}.style.contrast_mode",
            suggested_patch={"contrast_mode": "auto"},
        )
    if "subject evidence" in lowered:
        return _graphics_preview_problem(
            422, "GRAPHIC_SUBJECT_EVIDENCE_REQUIRED", message,
            field=f"{prefix}.composition",
            suggested_patch={"composition": "auto", "position": "auto"},
        )
    if any(word in lowered for word in (
        "safe area", "safe-area", "keepout", "outside", "position clears",
    )):
        return _graphics_preview_problem(
            422, "GRAPHIC_PLACEMENT_UNSAFE", message,
            field=f"{prefix}.position",
            suggested_patch={"position": "auto"},
        )
    if "glyph" in lowered:
        return _graphics_preview_problem(
            422, "GRAPHIC_GLYPH_UNSUPPORTED", message,
            field=f"{prefix}.primary_fact_id",
            suggested_patch={"action": "edit_copy_or_choose_font"},
        )
    if any(word in lowered for word in (
        "too long", "too tall", "cannot fit", "only", "needs a",
    )):
        return _graphics_preview_problem(
            422, "GRAPHIC_CONTENT_DOES_NOT_FIT", message,
            field=prefix,
            suggested_patch={"action": "reduce_copy_scale_or_motion"},
        )
    if "template" in lowered:
        return _graphics_preview_problem(
            422, "GRAPHIC_TEMPLATE_INVALID", message,
            field=f"{prefix}.template",
            suggested_patch={"action": "choose_compatible_template"},
        )
    return _graphics_preview_problem(
        422, "GRAPHIC_VALIDATION_FAILED", message,
        field=prefix,
        suggested_patch={"action": "review_graphic_settings"},
    )


def _timeline_blocks(run: Run) -> list[dict]:
    """Stable selection identities and source windows for the current cut."""

    current = _current_timeline(run)
    if current:
        return [dict(one) for one in current["shots"]]
    report = run.report() or {}
    shots = report.get("selection", {}).get("shots", [])
    rhythm = report.get("rhythm", {})
    resolved = report.get("source_motion_details", {})
    return [
        {
            "selection_index": index,
            "in_seconds": float(
                (resolved.get(f"k{index:02d}", {}).get("window") or
                 [shot.get("start_seconds", 0.0)])[0]
            ),
            "seconds": float((
                lambda window: window[1] - window[0]
                if len(window) >= 2 else
                rhythm.get(f"k{index:02d}", {}).get("seconds", 0.0)
            )(resolved.get(f"k{index:02d}", {}).get("window") or [])),
            "gain_db": 0.0,
        }
        for index, shot in enumerate(shots)
    ]


def _retime_graphics_for_current_cut(
    run: Run, old_blocks: list[dict], blocks: list[dict],
) -> None:
    """Resolve stable shot anchors onto a newly committed timeline."""

    from montagewright.graphics import GraphicsPlan
    from montagewright.measure.storage import write_json

    path = run.output / "work" / "graphics.json"
    if not path.exists():
        return
    plan = GraphicsPlan.model_validate_json(path.read_text(encoding="utf-8"))
    positions = {}
    cursor = 0.0
    for reel_index, block in enumerate(blocks):
        positions[int(block["selection_index"])] = (
            reel_index, cursor, block
        )
        cursor += float(block["seconds"])
    cues = []
    for cue in plan.cues:
        match = re.fullmatch(r"k(\d+)", cue.anchor_clip_id or "")
        old_reel_index = int(match.group(1)) if match else -1
        old_block = (
            old_blocks[old_reel_index]
            if 0 <= old_reel_index < len(old_blocks) else None
        )
        selection_index = (
            int(cue.anchor_selection_index)
            if cue.anchor_selection_index is not None
            else (
                int(old_block["selection_index"])
                if old_block is not None else -1
            )
        )
        placed = positions.get(selection_index)
        changes = {}
        if placed is not None and old_block is not None:
            new_reel_index, timeline_at, block = placed
            source_at = (
                float(old_block["in_seconds"]) + cue.anchor_offset_seconds
            )
            new_offset = source_at - float(block["in_seconds"])
            remaining = float(block["seconds"]) - new_offset
            if new_offset >= 0.0 and remaining >= cue.duration_seconds:
                changes.update({
                    "anchor_clip_id": f"k{new_reel_index:02d}",
                    "anchor_selection_index": selection_index,
                    "anchor_offset_seconds": round(new_offset, 6),
                    "at_seconds": round(timeline_at + new_offset, 6),
                })
            else:
                placed = None
        if placed is None:
            changes.update({
                "status": "draft",
                "editor_note": (
                    cue.editor_note + "；" if cue.editor_note else ""
                ) + "重新剪輯後原錨點已不在成片內，請重新放置",
            })
        cues.append(cue.model_copy(update=changes))
    updated = plan.model_copy(update={
        "revision": plan.revision + 1,
        "cues": cues,
    })
    write_json(path, updated.model_dump(mode="json"))


def _current_cut_decisions(run: Run) -> tuple[list[dict], dict[str, dict]]:
    """Selection and rhythm that the public MP4 currently represents."""

    report = run.report() or {}
    original = report.get("selection", {}).get("shots", [])
    current = _current_timeline(run)
    blocks = current.get("shots") or []
    if not blocks:
        return original, report.get("rhythm", {})
    shots = []
    rhythm = {}
    for index, block in enumerate(blocks):
        shot = dict(original[int(block["selection_index"])])
        shot["start_seconds"] = float(block["in_seconds"])
        shots.append(shot)
        rhythm[f"k{index:02d}"] = {"seconds": float(block["seconds"])}
    return shots, rhythm


def _current_audio_assignments(run: Run) -> list:
    """A lightweight view of the committed independent audio track.

    Subtitle derivation only needs source identity, source window, role and
    the master-timeline start.  Keeping this projection on the committed
    frame clock avoids rebuilding crops or invoking any model merely to read
    captions.
    """

    from types import SimpleNamespace

    current = _current_timeline(run)
    fps = int(current.get("output_fps") or 30)
    return [
        SimpleNamespace(
            audio_id=str(one["audio_id"]),
            source=SimpleNamespace(source_id=str(one["source_id"])),
            in_seconds=float(one["in_seconds"]),
            out_seconds=float(one["out_seconds"]),
            duration_seconds=(
                int(one["frame_count"]) / fps
            ),
            timeline_start_frame=int(one["timeline_start_frame"]),
            timeline_in_seconds=int(one["timeline_start_frame"]) / fps,
            frame_count=int(one["frame_count"]),
            role=str(one.get("role", "narrative")),
            completion=str(one.get("completion", "none")),
            gain_db=float(one.get("gain_db", 0.0) or 0.0),
            why=str(one.get("why", "")),
        )
        for one in current.get("audio_assignments", [])
    ]


def _subtitle_words(run) -> "list":
    """The measured words, on the cut's own clock."""

    from montagewright.transcript import (
        words_against_audio_assignments, words_against_cut,
    )

    audio = _current_audio_assignments(run)
    if audio:
        return words_against_audio_assignments(audio, _transcript_map(run))

    shots, rhythm = _current_cut_decisions(run)
    return words_against_cut(
        shots, rhythm,
        _transcript_map(run),
    )


def _subtitle_lines(run, *, edits: bool = True) -> "list":
    """What should appear on screen, edits included.

    The derived lines come from the transcripts and the running order. An
    edited set, if there is one, replaces them wholesale -- it was derived
    from the same cut and then corrected, so merging the two would mean
    re-deriving a line somebody had already fixed.
    """

    from montagewright.transcript import Line, against_audio_assignments, against_cut

    edited = run.output / "work" / "subtitles.json"
    if edits and edited.exists():
        try:
            saved = json.loads(edited.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = []
        if saved:
            return [
                Line(
                    text=str(one.get("text", "")),
                    starts_seconds=float(one.get("at", 0.0)),
                    ends_seconds=float(one.get("until", 0.0)),
                    heard=str(one.get("heard", "")),
                    speaker=str(one.get("speaker", "")),
                    timing_source=str(
                        one.get("timing_source", "manual")
                    ),
                    timing_confidence=str(
                        one.get("timing_confidence", "unverified")
                    ),
                    timing_locked=bool(one.get("timing_locked", False)),
                )
                for one in saved
            ]

    audio = _current_audio_assignments(run)
    if audio:
        return against_audio_assignments(audio, _transcript_map(run))
    shots, rhythm = _current_cut_decisions(run)
    return against_cut(
        shots, rhythm,
        _transcript_map(run),
    )


def _retimed(run, kept: list[dict]) -> list[dict]:
    """Put a person's edits back on the recogniser's clock.

    Someone fixing a caption is fixing the words, and the browser sends back
    whatever times the line already had. That is right until the edit
    changes how long the line takes to say -- a product name corrected from
    three characters to six, a line split in two -- and then the words are
    right and sit at the wrong moment.

    So the edited text is aligned against the measured per-word timings,
    exactly as a machine correction is. Whatever a person left alone keeps
    the timing it was measured with; only what they actually changed moves.
    Editing a caption should not silently re-time the ones around it.

    If the words are not available -- an older card that never stored
    them -- the times sent are kept as they are. A caption that does not
    move is better than one moved by a guess.
    """

    from montagewright.backfill import across_lines
    from montagewright.transcript import (
        words_against_audio_assignments, words_against_cut,
    )

    # Narrowly caught on purpose. A card that is missing or unreadable is a
    # normal thing to meet and means "no measured words here". Anything
    # else -- a shape that changed under this, a name that moved -- should
    # come out as a failure, because a broad catch here would turn a broken
    # re-timing into edits that silently keep whatever times they arrived
    # with, which looks exactly like working.
    try:
        audio = _current_audio_assignments(run)
        if audio:
            words = words_against_audio_assignments(
                audio, _transcript_map(run)
            )
        else:
            shots, rhythm = _current_cut_decisions(run)
            words = words_against_cut(shots, rhythm, _transcript_map(run))
    except (OSError, ValueError, KeyError):
        return kept
    if not words or not kept:
        return kept

    timings = across_lines([one["text"] for one in kept], words)
    out = []
    for one, (start, end, _) in zip(kept, timings):
        if one.get("timing_locked"):
            # Moving or trimming a cue is an explicit editorial timing
            # decision. Text correction may still be saved, but no automatic
            # alignment is allowed to overwrite the locked clock.
            one = dict(
                one,
                timing_source="manual",
                timing_confidence="human_locked",
            )
        elif end > start:
            one = dict(
                one,
                at=round(start, 3),
                until=round(end, 3),
                timing_source="apple_audio_time_range",
                timing_confidence="unverified",
            )
        out.append(one)
    out.sort(key=lambda one: one["at"])
    return out


def _safe_area_of(report: dict) -> dict:
    from montagewright.subtitles import safe_area

    area = safe_area(report.get("direction", {}).get("aspect", "9:16"))
    return {
        "up_from_bottom": area.up_from_bottom,
        "side_margin": area.side_margin,
        "text_height": area.text_height,
        "max_lines": area.max_lines,
    }


def _graphics_layout_evidence(run: Run, plan: Any) -> dict:
    """Map durable SAM boxes through the delivered crop for a cue window."""

    from montagewright.graphics import LayoutEvidence
    from montagewright.reframe import interpolate_crop_keyframes

    work = run.output / "work"
    crops_path = work / "crops.json"
    if not crops_path.exists():
        return {}
    try:
        crops = json.loads(crops_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    evidence = {}
    report_payload = run.report() or {}
    report_tracks = report_payload.get("subject_tracks", {})
    source_motion_details = report_payload.get("source_motion_details", {})
    current = _current_timeline(run)
    current_blocks = current.get("shots") or []
    selection_to_reel = {
        int(block["selection_index"]): index
        for index, block in enumerate(current_blocks)
    }
    original_shots = (run.report() or {}).get("selection", {}).get(
        "shots", []
    )
    for cue in plan.cues:
        clip_id = cue.anchor_clip_id
        stable_index = cue.anchor_selection_index
        reel_index = (
            selection_to_reel.get(int(stable_index), -1)
            if stable_index is not None else -1
        )
        crop_clip_id = (
            f"k{reel_index:02d}" if reel_index >= 0 else clip_id
        )
        track_clip_id = (
            f"k{int(stable_index):02d}"
            if stable_index is not None else clip_id
        )
        keys = crops.get(crop_clip_id, [])
        if not clip_id or not keys:
            continue
        durable = report_tracks.get(track_clip_id, [])
        if not durable:
            continue
        window_start = cue.anchor_offset_seconds
        window_end = window_start + cue.duration_seconds
        source_shift = 0.0
        if (
            stable_index is not None and reel_index >= 0
            and int(stable_index) < len(original_shots)
        ):
            origin_window = source_motion_details.get(
                track_clip_id, {}
            ).get("window") or []
            tracking_origin = (
                float(origin_window[0]) if origin_window
                else float(original_shots[int(stable_index)].get(
                    "start_seconds", 0.0
                ))
            )
            source_shift = (
                float(current_blocks[reel_index]["in_seconds"])
                - tracking_origin
            )
        boxes = []
        for sample in durable:
            relative = float(sample.get("seconds", 0)) - source_shift
            if not window_start - 0.05 <= relative <= window_end + 0.05:
                continue
            half_w = float(sample.get("width", 0)) / 2.0
            half_h = float(sample.get("height", 0)) / 2.0
            raw = [
                (float(sample.get("centre_x", 0)) - half_w) * 1000,
                (float(sample.get("centre_y", 0)) - half_h) * 1000,
                (float(sample.get("centre_x", 0)) + half_w) * 1000,
                (float(sample.get("centre_y", 0)) + half_h) * 1000,
            ]
            crop = interpolate_crop_keyframes(keys, relative)
            if crop is None or crop["w"] <= 0 or crop["h"] <= 0:
                continue
            x0, y0, x1, y1 = (float(value) / 1000.0 for value in raw)
            x0 = max(x0, crop["x"]); y0 = max(y0, crop["y"])
            x1 = min(x1, crop["x"] + crop["w"])
            y1 = min(y1, crop["y"] + crop["h"])
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append((
                (x0 - crop["x"]) / crop["w"],
                (y0 - crop["y"]) / crop["h"],
                (x1 - x0) / crop["w"],
                (y1 - y0) / crop["h"],
            ))
        if boxes:
            evidence[cue.graphic_id] = LayoutEvidence(
                subject_boxes=tuple(boxes),
                source="sam2.1_report_track",
            )
    return evidence


def _subtitle_render_choices(run: Run) -> tuple[str, str]:
    """The last explicit subtitle look/font, with CLI values as fallback."""

    saved = run.output / "work" / "subtitle-render.json"
    try:
        payload = json.loads(saved.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        payload = {}
    look_name = str(payload.get("look") or "")
    font = str(payload.get("font") or "")
    if not look_name and "--subtitle-look" in run.command:
        look_name = run.command[run.command.index("--subtitle-look") + 1]
    if not font and "--subtitle-font" in run.command:
        font = run.command[run.command.index("--subtitle-font") + 1]
    if look_name not in {"plain", "speakers", "spoken", "plate"}:
        look_name = "plain"
    return look_name, font


def _video_display_size(picture: Path) -> tuple[int, int]:
    """Read display dimensions without hashing a multi-gigabyte master."""

    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
            "-of", "json", str(picture),
        ],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"no video stream in {picture.name}")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    rotation = int(float(stream.get("tags", {}).get("rotate", 0) or 0))
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(float(side_data["rotation"] or 0))
            break
    if abs(rotation) % 180 == 90:
        width, height = height, width
    return width, height


def _video_timing(picture: Path) -> tuple[str, float]:
    """A lightweight rational FPS/duration probe for frame-exact previews."""

    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate:format=duration",
            "-of", "json", str(picture),
        ],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    rate = str(streams[0].get("avg_frame_rate") or "30/1") if streams else "30/1"
    if rate == "0/0":
        rate = "30/1"
    return rate, float(payload.get("format", {}).get("duration") or 0.0)


def _prepare_run_subtitle_track(
    run: Run, picture: Path, *, dimensions: tuple[int, int] | None = None,
):
    from montagewright import subtitles as typeset
    from montagewright.subtitles import (
        PreparedSubtitleTrack, look, prepare_overlays,
    )

    lines = _subtitle_lines(run)
    if not lines:
        return PreparedSubtitleTrack(())
    width, height = dimensions or _video_display_size(picture)
    aspect = (run.report() or {}).get("direction", {}).get("aspect", "9:16")
    look_name, font = _subtitle_render_choices(run)
    with SUBTITLE_FONT_LOCK:
        was, typeset.CHOSEN = typeset.CHOSEN, (font or None)
        try:
            return prepare_overlays(
                lines, aspect=aspect,
                width=width, height=height,
                work=run.output / "work" / "graphics-subtitle-render",
                style=look(look_name), words=_subtitle_words(run),
            )
        finally:
            typeset.CHOSEN = was


class _AlreadyHave(Exception):
    """The run recorded its crops, so there is nothing to rebuild."""


def _typed_path(raw: str) -> Path | None:
    """Whatever a person pasted, as a path.

    Dragging out of Finder or copying from a browser gives a `file://` URL
    with the spaces and the Chinese percent-encoded; quoting a path in a
    terminal leaves the quotes on. All of that arrives here looking like a
    path and is not one -- `Path("file:/Users/...")` is a relative directory
    called "file:", so the run started, spent four minutes writing cards, and
    only then failed on a track that was never there.
    """

    from urllib.parse import unquote, urlparse

    text = raw.strip().strip('"').strip("'")
    if not text:
        return None
    if text.startswith("file:"):
        parsed = urlparse(text)
        text = unquote(parsed.path)
    return Path(text).expanduser()


def _save(upload: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return destination


def _reference_path_lines(raw: str) -> list[str]:
    """Reference paths from the local UI, one per line or as a JSON list."""

    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        decoded = json.loads(text)
        if not isinstance(decoded, list) or not all(
            isinstance(item, str) for item in decoded
        ):
            raise ValueError("reference image paths must be a JSON string list")
        return [item.strip() for item in decoded if item.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _rewrite_uploaded_reference_paths(
    source: Path,
    destination: Path,
    provided: list[tuple[str, Path]],
    *,
    require_all: bool = False,
) -> Path:
    """Bind browser-provided image bytes to paths already named by the spec.

    This is deliberately transport-only.  It does not interpret targets,
    anchors, hashes or identity rules; ``load_grounding_spec`` remains the
    authority for all of those after this staging rewrite.
    """

    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("grounding spec must be a JSON object")
    references = payload.get("reference_images")
    if not isinstance(references, list):
        raise ValueError("grounding spec has no reference_images list")

    candidates = []
    for declared, path in provided:
        resolved = path.expanduser().resolve()
        aliases = {
            declared,
            Path(declared).as_posix(),
            Path(declared).name,
            str(resolved),
            resolved.as_posix(),
            resolved.name,
        }
        candidates.append((aliases, resolved))

    used: set[Path] = set()
    unmatched: list[str] = []
    for reference in references:
        if not isinstance(reference, dict) or not isinstance(
            reference.get("path"), str
        ):
            continue
        raw_path = reference["path"]
        exact = [
            path for aliases, path in candidates
            if raw_path in aliases or Path(raw_path).as_posix() in aliases
        ]
        matches = exact
        if not matches:
            leaf = Path(raw_path).name
            matches = [
                path for aliases, path in candidates if leaf in aliases
            ]
        matches = list(dict.fromkeys(matches))
        if len(matches) > 1:
            raise ValueError(
                f"more than one uploaded reference matches {raw_path!r}"
            )
        if matches:
            matched = matches[0]
            used.add(matched)
            reference["path"] = os.path.relpath(
                matched, destination.parent
            ).replace(os.sep, "/")
        elif require_all:
            unmatched.append(raw_path)

    unused = [
        declared for declared, path in provided
        if path.expanduser().resolve() not in used
    ]
    if unused:
        raise ValueError(
            "reference images are not named by the spec: " + ", ".join(unused)
        )
    if unmatched:
        raise ValueError(
            "uploaded or pasted specs require every reference image: "
            + ", ".join(unmatched)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return destination


def create_app() -> FastAPI:
    app = FastAPI(title="montagewright")

    @app.exception_handler(Exception)
    async def unexpected_request_failure(
        request: Request, error: Exception
    ) -> JSONResponse:
        """Keep local UI failures diagnosable without returning a blank 500.

        This application is a localhost editing workstation, not a public API.
        A generic ``Internal Server Error`` hides the one fact needed to repair
        a failed launch (for example an unreadable folder or a process limit),
        while the traceback may belong to a terminal that is no longer open.
        Keep the traceback in the server output and return the exception class
        plus message to the local editor.
        """

        error_id = uuid.uuid4().hex[:8]
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "detail": (
                    f"本機剪輯服務發生錯誤（{error_id}）："
                    f"{type(error).__name__}: {error}"
                ),
                "error_code": "internal_server_error",
                "error_id": error_id,
                "path": request.url.path,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE.read_text(encoding="utf-8")

    @app.post("/api/runs")
    async def start(
        rushes: list[UploadFile] | None = None,
        music: UploadFile | None = None,
        grounding_spec_file: UploadFile | None = None,
        reference_images: list[UploadFile] | None = None,
        source_path: str = Form(""),
        music_path: str = Form(""),
        grounding_spec_path: str = Form(""),
        grounding_spec_json: str = Form(""),
        reference_image_paths: str = Form(""),
        grounding_target_id: str = Form(""),
        grounding_target_description: str = Form(""),
        grounding_identity_cues: str = Form(""),
        grounding_exclusions: str = Form(""),
        brief: str = Form(""),
        brief_path: str = Form(""),
        base_run_id: str = Form(""),
        inherit_brief: bool = Form(False),
        aspect: str = Form("9:16"),
        seconds: float = Form(0.0),
        duration_mode: str = Form("preferred"),
        budget: float = Form(6.0),
        review: bool = Form(True),
        timeline: str = Form("none"),
        speech: str = Form("auto"),
        subtitles: str = Form("sidecar"),
        subtitle_look: str = Form("plain"),
        subtitle_font: str = Form(""),
        locale: str = Form("zh-TW"),
    ) -> JSONResponse:
        if aspect not in ASPECTS:
            raise HTTPException(
                400, f"aspect must be one of {sorted(ASPECTS)}"
            )
        if duration_mode not in {"exact", "preferred"}:
            raise HTTPException(400, "duration_mode must be exact or preferred")

        run_id = uuid.uuid4().hex[:12]
        root = RUNS_ROOT / run_id

        # Made only once there is something to put in it. Creating it first
        # meant every mistyped path left an empty folder in the runs
        # directory that nothing would ever open or clean up.
        made_root = False

        def keep() -> Path:
            nonlocal made_root
            if not made_root:
                root.mkdir(parents=True, exist_ok=True)
                made_root = True
            return root

        # A path is the ordinary case: this runs beside the material, and
        # pushing a folder of 4K through the browser to write it back to disk
        # a directory away is work nobody asked for.
        typed = _typed_path(source_path)
        if typed is not None:
            rush_dir = typed
            if not rush_dir.exists():
                raise HTTPException(400, f"{rush_dir} is not there")
            if rush_dir.is_file():
                holder = keep() / "rushes"
                holder.mkdir(parents=True, exist_ok=True)
                link = holder / rush_dir.name
                if not link.exists():
                    try:
                        os.link(rush_dir, link)
                    except OSError:
                        link.symlink_to(rush_dir)
                rush_dir = holder
            kept = sum(
                1 for path in rush_dir.iterdir()
                if path.suffix in VIDEO_SUFFIXES
            )
        else:
            rush_dir = keep() / "rushes"
            rush_dir.mkdir(parents=True, exist_ok=True)
            kept = 0
            budgeted = MAX_UPLOAD_BYTES
            for upload in rushes or []:
                # A folder upload arrives with its paths; only the leaf
                # matters, and anything that is not footage is not ours to
                # guess about.
                name = Path(upload.filename or "").name
                if not name or Path(name).suffix not in VIDEO_SUFFIXES:
                    continue
                written = _save(upload, rush_dir / name)
                budgeted -= written.stat().st_size
                if budgeted < 0:
                    shutil.rmtree(root, ignore_errors=True)
                    raise HTTPException(
                        413,
                        f"more than {MAX_UPLOAD_BYTES // 1024**3} GB uploaded; "
                        "give a path on this machine instead",
                    )
                kept += 1
        if not kept:
            raise HTTPException(400, "no video files there")

        # Grounding input is parsed by the exact same loader used by the CLI.
        # Browser files are merely staged and their declared spec paths are
        # rewritten before that validation; no model call occurs here.
        spec_path = _typed_path(grounding_spec_path)
        spec_upload = (
            grounding_spec_file
            if grounding_spec_file is not None and grounding_spec_file.filename
            else None
        )
        spec_json = grounding_spec_json.strip()
        spec_sources = sum((spec_path is not None, spec_upload is not None,
                            bool(spec_json)))
        if spec_sources > 1:
            if made_root:
                shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(
                400,
                "give one grounding spec: a path, an upload, or pasted JSON",
            )

        try:
            typed_references = _reference_path_lines(reference_image_paths)
        except (json.JSONDecodeError, ValueError) as error:
            if made_root:
                shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(400, f"invalid reference image paths: {error}")
        uploads = [
            upload for upload in reference_images or [] if upload.filename
        ]
        simple_grounding = bool(grounding_target_description.strip())
        if simple_grounding and spec_sources:
            if made_root:
                shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(
                400, "use either the simple reference fields or a grounding spec"
            )
        if (typed_references or uploads) and not (spec_sources or simple_grounding):
            if made_root:
                shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(
                400, "reference images require a target description or grounding spec"
            )
        if simple_grounding and not (typed_references or uploads):
            if made_root:
                shutil.rmtree(root, ignore_errors=True)
            raise HTTPException(400, "a grounding target needs a reference image")

        canonical_grounding: Path | None = None
        if spec_sources or simple_grounding:
            staging = keep() / "grounding-input"
            staging.mkdir(parents=True, exist_ok=True)
            grounding_upload_bytes = 0

            def count_grounding_upload(path: Path) -> None:
                nonlocal grounding_upload_bytes
                grounding_upload_bytes += path.stat().st_size
                if grounding_upload_bytes > MAX_GROUNDING_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        "grounding spec and reference uploads exceed "
                        f"{MAX_GROUNDING_UPLOAD_BYTES // 1024**2} MB; "
                        "use paths on this machine instead",
                    )

            try:
                if simple_grounding:
                    raw_spec = staging / "simple-grounding-spec.json"
                elif spec_path is not None:
                    if not spec_path.is_file():
                        raise ValueError(f"{spec_path} is not there")
                    raw_spec = spec_path
                elif spec_upload is not None:
                    raw_spec = _save(
                        spec_upload, staging / "uploaded-grounding-spec.json"
                    )
                    count_grounding_upload(raw_spec)
                else:
                    raw_spec = staging / "pasted-grounding-spec.json"
                    raw_spec.write_text(spec_json, encoding="utf-8")
                    count_grounding_upload(raw_spec)

                provided: list[tuple[str, Path]] = []
                override_root = staging / "reference-overrides"
                for raw_path in typed_references:
                    image = _typed_path(raw_path)
                    if image is None or not image.is_file():
                        raise ValueError(f"reference image is not there: {raw_path}")
                    stored = override_root / f"{uuid.uuid4().hex}-{image.name}"
                    stored.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(image, stored)
                    provided.append((raw_path, stored))
                for upload in uploads:
                    name = Path(upload.filename or "reference-image").name
                    stored = _save(
                        upload, override_root / f"{uuid.uuid4().hex}-{name}"
                    )
                    count_grounding_upload(stored)
                    provided.append((upload.filename or name, stored))

                staged_spec = raw_spec
                portable_upload = spec_upload is not None or bool(spec_json)
                if simple_grounding:
                    from montagewright.reference_grounding import (
                        build_reference_grounding_spec,
                    )

                    built = build_reference_grounding_spec(
                        raw_spec,
                        target_id=(
                            grounding_target_id.strip() or "target.primary"
                        ),
                        target_description=grounding_target_description.strip(),
                        identity_cues=tuple(
                            line.strip()
                            for line in grounding_identity_cues.splitlines()
                            if line.strip()
                        ),
                        stable_exclusions=tuple(
                            line.strip()
                            for line in grounding_exclusions.splitlines()
                            if line.strip()
                        ),
                        positive_images=tuple(path for _, path in provided),
                        created_by="web_user",
                    )
                    staged_spec = Path(str(built.source_path))
                elif provided or portable_upload:
                    staged_spec = _rewrite_uploaded_reference_paths(
                        raw_spec,
                        staging / "staged-grounding-spec.json",
                        provided,
                        require_all=portable_upload,
                    )

                from montagewright.cli import prepare_grounding_spec_artifact

                canonical_grounding, _ = prepare_grounding_spec_artifact(
                    staged_spec, keep() / "out" / "work" / "grounding-spec.json"
                )
                shutil.rmtree(staging, ignore_errors=True)
            except HTTPException:
                shutil.rmtree(root, ignore_errors=True)
                raise
            except (OSError, ValueError) as error:
                shutil.rmtree(root, ignore_errors=True)
                raise HTTPException(400, f"invalid grounding spec: {error}")

        command = [
            sys.executable, "-u", "-m", "montagewright.cli", "render",
            str(rush_dir), "--aspect", aspect,
            "--budget", str(budget),
            "--output", str(keep() / "out"),
        ]
        if canonical_grounding is not None:
            command += ["--grounding-spec", str(canonical_grounding)]
        if seconds > 0:
            command += [
                "--seconds", str(seconds), "--duration-mode", duration_mode,
            ]
        track = _typed_path(music_path)
        if track is not None:
            if not track.exists():
                if made_root:
                    shutil.rmtree(root, ignore_errors=True)
                raise HTTPException(400, f"{track} is not there")
            command += ["--music", str(track)]
        elif music is not None and music.filename:
            track = _save(music, keep() / Path(music.filename).name)
            command += ["--music", str(track)]
        if speech in {"auto", "never"}:
            command += ["--speech", speech]
        if locale.strip():
            command += ["--locale", locale.strip()]
        if inherit_brief and not brief.strip() and not brief_path.strip():
            if not base_run_id.strip():
                raise HTTPException(400, "inheriting a brief requires base_run_id")
            parent = _run(base_run_id.strip())
            brief = _brief_of(parent)
            if not brief.strip():
                raise HTTPException(400, "the base run has no readable brief")
        typed_brief = _typed_path(brief_path)
        if typed_brief is not None:
            if not typed_brief.is_file():
                if made_root:
                    shutil.rmtree(root, ignore_errors=True)
                raise HTTPException(400, f"{typed_brief} is not a brief file")
            try:
                file_brief = typed_brief.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                if made_root:
                    shutil.rmtree(root, ignore_errors=True)
                raise HTTPException(400, f"brief is unreadable: {error}")
            brief = file_brief + (
                "\n\n## Web 補充需求\n\n" + brief.strip()
                if brief.strip() else ""
            )
        if brief.strip():
            stored_brief = keep() / "brief.md"
            stored_brief.write_text(brief, encoding="utf-8")
            command += ["--brief", str(stored_brief)]
        if review:
            command += ["--review"]
        if timeline in {"premiere", "finalcut", "both"}:
            command += ["--timeline", timeline]
        if subtitles in {"none", "sidecar", "burn"}:
            command += ["--subtitles", subtitles]
        if subtitle_look in {"plain", "speakers", "spoken", "plate"}:
            command += ["--subtitle-look", subtitle_look]
        if subtitle_font and Path(subtitle_font).exists():
            command += ["--subtitle-font", subtitle_font]
        checkpoint = Path("artifacts/models/sam2.1_hiera_tiny.pt").resolve()
        if checkpoint.exists():
            command += ["--sam-checkpoint", str(checkpoint)]

        run = Run(
            run_id=run_id, root=keep(), source=str(rush_dir), command=command
        )
        run.lines.append(f"{kept} clips from {rush_dir}")
        if canonical_grounding is not None:
            run.lines.append(f"grounding spec {canonical_grounding}")
        try:
            run.remember()
            run.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except OSError as error:
            # A launch can fail before the CLI has a chance to print anything
            # (process limits, permissions, a missing interpreter).  Record a
            # truthful terminal state and give the editor an actionable 503;
            # otherwise it looks as though Gemini started and then vanished.
            run.state = "failed"
            run.returncode = -1
            run.lines.append(
                f"launch failed: {type(error).__name__}: {error}"
            )
            try:
                run.remember()
            except OSError:
                traceback.print_exc()
            RUNS[run_id] = run
            raise HTTPException(
                503,
                {
                    "message": (
                        "無法啟動本機剪輯程序："
                        f"{type(error).__name__}: {error}"
                    ),
                    "error_code": "process_launch_failed",
                    "run_id": run_id,
                },
            ) from error
        RUNS[run_id] = run
        threading.Thread(target=_collect, args=(run,), daemon=True).start()
        return JSONResponse({
            "run_id": run_id,
            "grounding_spec": (
                str(canonical_grounding) if canonical_grounding else None
            ),
        })

    @app.post("/api/grounding/draft")
    async def grounding_draft(
        reference_images: list[UploadFile] | None = None,
        reference_image_paths: str = Form(""),
    ) -> JSONResponse:
        """Read the reference pictures and propose what to say about them.

        The identity cue that made the Fold8 run work names a 5.5-inch cover
        display and a 7.6-inch inner one -- a specification someone looked
        up. Three empty textareas ask every user to be that person; the ones
        who are not leave them blank, and grounding quietly gets worse with
        nothing on screen to say so. One cheap call against the pictures the
        user already has turns writing into reviewing.

        Nothing here is authority. The draft goes back to the form as
        editable text and only becomes a lock when the person submits it.
        """

        try:
            typed = _reference_path_lines(reference_image_paths)
        except (json.JSONDecodeError, ValueError) as error:
            raise HTTPException(400, f"invalid reference image paths: {error}")
        uploads = [
            upload for upload in reference_images or [] if upload.filename
        ]
        if not typed and not uploads:
            raise HTTPException(400, "give at least one reference image")

        staging = Path(tempfile.mkdtemp(prefix="montagewright-draft-"))
        try:
            images: list[Path] = []
            budgeted = MAX_GROUNDING_UPLOAD_BYTES
            for raw_path in typed:
                image = _typed_path(raw_path)
                if image is None or not image.is_file():
                    raise HTTPException(
                        400, f"reference image is not there: {raw_path}"
                    )
                images.append(image)
            for upload in uploads:
                name = Path(upload.filename or "reference-image").name
                stored = _save(upload, staging / f"{uuid.uuid4().hex}-{name}")
                budgeted -= stored.stat().st_size
                if budgeted < 0:
                    raise HTTPException(
                        413,
                        "reference images exceed "
                        f"{MAX_GROUNDING_UPLOAD_BYTES // 1024**2} MB",
                    )
                images.append(stored)

            from starlette.concurrency import run_in_threadpool

            from montagewright.cli import _client
            from montagewright.cost import Ledger
            from montagewright.planner import MODEL_ID
            from montagewright.reference_grounding import (
                ReferenceGroundingError,
                draft_identity_from_references,
            )
            from montagewright.uploads import UploadCache, default_cache_path

            try:
                client = _client()
            except SystemExit as error:
                raise HTTPException(400, str(error))
            # The same cache the run uses. These bytes are about to be
            # uploaded again as approved anchors, and they hash the same.
            cache = UploadCache.load(default_cache_path())
            ledger = Ledger(cap_usd=DRAFT_BUDGET_USD, model_id=MODEL_ID)
            try:
                drafted = await run_in_threadpool(
                    draft_identity_from_references,
                    images, client=client, cache=cache, ledger=ledger,
                )
            except ReferenceGroundingError as error:
                raise HTTPException(502, f"draft failed: {error}")
            except Exception as error:  # noqa: BLE001 -- reported, not swallowed
                raise HTTPException(
                    502, f"draft failed: {type(error).__name__}: {error}"
                )
            if drafted is None:  # pragma: no cover -- a client exists here
                raise HTTPException(400, "no Gemini client")
            draft, _ = drafted
            return JSONResponse({
                "target_description": draft.target_description,
                "identity_cues": list(draft.identity_cues),
                "stable_exclusions": list(draft.stable_exclusions),
                "caveat": draft.caveat,
                "spent_usd": round(ledger.spent_usd, 4),
            })
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @app.get("/api/browse")
    def browse(path: str = "", kind: str = "video") -> JSONResponse:
        """List folders so a path can be clicked instead of typed.

        A browser will not hand over a real filesystem path -- a directory
        picker gives relative names and nothing else -- so typing one out was
        the only way to point this at material sitting next to the server.
        This serves the listing instead. It binds to localhost, and the person
        using it owns the disk.
        """

        here = Path(path).expanduser() if path.strip() else Path.home()
        try:
            here = here.resolve(strict=True)
        except (OSError, RuntimeError):
            raise HTTPException(404, f"{path} is not there")
        if not here.is_dir():
            here = here.parent

        looking = (
            AUDIO_SUFFIXES if kind == "audio"
            else BRIEF_SUFFIXES if kind == "file"
            else SPEC_SUFFIXES if kind == "spec"
            else VIDEO_SUFFIXES
        )
        folders = []
        for entry in sorted(here.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            try:
                clips = sum(
                    1 for child in entry.iterdir()
                    if child.suffix in looking
                )
            except PermissionError:
                continue
            folders.append({"name": entry.name, "path": str(entry), "clips": clips})
        loose = [
            {"name": entry.name, "path": str(entry)}
            for entry in sorted(here.iterdir())
            if entry.is_file() and entry.suffix in looking
        ]
        return JSONResponse({
            "here": str(here),
            "parent": str(here.parent) if here.parent != here else None,
            "folders": folders,
            "videos": loose,
        })

    @app.get("/api/runs")
    def history(limit: int = 30) -> JSONResponse:
        """Earlier cuts, so this one can be compared against them."""

        recall()
        rows = sorted(
            RUNS.values(), key=lambda run: run.started_at, reverse=True
        )[:limit]
        out = []
        for run in rows:
            report = run.report() or {}
            out.append({
                "run_id": run.run_id,
                "state": run.state,
                "started_at": run.started_at,
                "source": Path(run.source).name if run.source else "",
                "source_path": run.source,
                "seconds": report.get("duration_seconds"),
                "shots": len(report.get("selection", {}).get("shots", [])),
                "spend": round(
                    sum(report.get("spend", {}).get("by_stage", {}).values()), 4
                ) or None,
                "delivered": sum(
                    1 for entry in (report.get("shots") or {}).values()
                    if entry.get("delivered")
                ) or None,
            })
        return JSONResponse({"runs": out})

    @app.get("/api/runs/{run_id}/waveform/{which}")
    def waveform(run_id: str, which: str, width: int = 2000):
        """The sound as a picture, so the tracks can be read.

        A cut with speech under music is two things happening at once and the
        page could only play it. Seeing where the voice sits and where the bed
        steps back is the difference between trusting the mix and checking
        it.
        """

        run = _run(run_id)
        if which == "voice":
            # picture.mp4 is the cut before the bed goes under it, so it is
            # the voice alone. Older runs kept only the deliverable.
            source = next(
                (
                    run.output / name
                    for name in ("picture.mp4", "deliverable.mp4")
                    if (run.output / name).exists()
                ),
                run.output / "picture.mp4",
            )
        elif which == "music":
            # The bed as it was laid, when the render kept it. Drawing the
            # track as it arrived showed a flat wall of music and said
            # nothing about whether it steps back for the voice, which is
            # the one thing about a mix anybody checks before posting.
            source = None
            laid = run.output / "bed-as-laid.m4a"
            if laid.exists():
                source = laid
            elif "--music" in run.command:
                candidate = Path(run.command[run.command.index("--music") + 1])
                source = candidate if candidate.exists() else None
            if source is None:
                raise HTTPException(404, "this cut has no music")
        else:
            raise HTTPException(400, "voice or music")
        if not source.exists():
            raise HTTPException(404, f"no {which} to draw")

        width = max(400, min(width, 6000))
        drawn = run.output / f"wave-{which}-{width}.png"
        if not drawn.exists():
            # How long the cut is, measured off the cut. This read
            # report.json, which is written last -- so a run that stopped
            # partway drew the whole track instead of the part it used: a
            # 2m35s bed stretched across a 29s timeline, every position on
            # the lane pointing at the wrong moment, in the one view whose
            # job is showing where the sound sits.
            seconds = 0.0
            for name in ("picture.mp4", "deliverable.mp4", "preview.mp4"):
                if (run.output / name).exists():
                    seconds = probe_duration(run.output / name) or 0.0
                    if seconds:
                        break
            if not seconds:
                seconds = float((run.report() or {}).get("duration_seconds") or 0.0)
            trim = ["-t", f"{seconds:.3f}"] if seconds and which == "music" else []
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
                + trim + ["-i", str(source), "-filter_complex",
                          f"aformat=channel_layouts=mono,"
                          f"showwavespic=s={width}x120:colors=#8b8880",
                          "-frames:v", "1", str(drawn)],
                check=False,
            )
        if not drawn.exists():
            raise HTTPException(500, "could not draw it")
        return FileResponse(drawn, media_type="image/png")

    @app.get("/api/runs/{run_id}/timeline-data")
    def timeline_data(run_id: str) -> JSONResponse:
        """Each shot's place on the timeline, and how far it can be pulled.

        The handles were rendered for exactly this and nothing could reach
        them: half a second either side, invisible, unusable.
        """

        from montagewright.pipeline import probe

        run = _run(run_id)
        report = run.report() or {}
        original_shots = report.get("selection", {}).get("shots", [])
        shots = original_shots
        rhythm = report.get("rhythm", {})
        current = _current_timeline(run)
        current_blocks = current.get("shots") or []
        if current_blocks:
            shots = [
                original_shots[int(one["selection_index"])]
                for one in current_blocks
            ]
            rhythm = {
                f"k{index:02d}": {"seconds": float(one["seconds"])}
                for index, one in enumerate(current_blocks)
            }
        verdicts = report.get("shots", {})
        motion = report.get("motion", {})
        source_motion_details = report.get("source_motion_details", {})

        found: dict[str, float] = {}
        crops: dict[str, list] = {}

        # What the render actually used, if the run left it behind. Only a
        # held frame can be re-derived afterwards -- a follow came out of a
        # propagation nothing here can repeat -- so recomputing was the
        # interface drawing every move as a static box and presenting it as
        # evidence. Runs from before this was written still fall through.
        recorded = False
        trail = run.output / "work" / "crops.json"
        if trail.exists():
            try:
                crops = json.loads(trail.read_text(encoding="utf-8"))
                recorded = bool(crops)
            except (OSError, ValueError) as error:
                print(f"timeline-data: unreadable crops.json ({error})", True)

        try:
            if crops:
                raise _AlreadyHave
            plan, _, _ = _rebuild(run)
            for segment in plan.segments:
                path = segment.crop_path
                keys = (
                    [(k.seconds, k.crop) for k in path.keyframes]
                    if path is not None
                    else ([(0.0, segment.crop)] if segment.crop else [])
                )
                crops[segment.clip_id] = [
                    {
                        "at": round(at, 3), "x": round(box.x, 5),
                        "y": round(box.y, 5), "w": round(box.width, 5),
                        "h": round(box.height, 5),
                    }
                    for at, box in keys
                ]
        except _AlreadyHave:
            pass
        except Exception as error:
            # The reel is still useful without the boxes, but a silent except
            # here is how "0/13 have a crop path" looked like a fact about the
            # cut rather than a broken lookup.
            print(f"timeline-data: no crop paths ({error})", flush=True)
            crops = {}

        # How long each source runs, which is how far a shot can be pulled
        # out. Reading it costs an ffprobe -- a whole process each -- and
        # they do not depend on one another, so they are read at once
        # rather than a dozen in a row. That was a second and a half before
        # anything appeared on opening a finished cut.
        where = list((run.output / "work" / "shots").glob("*")) + list(
            Path(run.source).glob("*") if Path(run.source).exists() else []
        )
        by_stem = {path.stem: path for path in where}
        wanted = {
            shot.get("source_id", "") for shot in shots
        } - {"" } - set(found)

        def measure(source_id: str) -> tuple[str, float]:
            match = by_stem.get(source_id)
            if match is None:
                return source_id, 0.0
            try:
                return source_id, probe(source_id, match).duration_seconds
            except Exception:
                return source_id, 0.0

        if wanted:
            with ThreadPoolExecutor(max_workers=8) as crew:
                for source_id, length in crew.map(measure, sorted(wanted)):
                    found[source_id] = length

        blocks, cursor = [], 0.0
        for index, shot in enumerate(shots):
            key = f"k{index:02d}"
            source_key = (
                f"k{int(current_blocks[index]['selection_index']):02d}"
                if current_blocks else key
            )
            seconds = float(rhythm.get(key, {}).get("seconds", 0.0))
            source_id = shot.get("source_id", "")
            blocks.append({
                "clip_id": key,
                # Which shot of the report this is, which is what the take
                # endpoint is keyed by. Its position in the reel is not the
                # same number the moment anything is reordered or dropped,
                # and peeking at the take asked for /source/undefined.
                "index": (
                    int(current_blocks[index]["selection_index"])
                    if current_blocks else index
                ),
                # Where the crop actually sat, keyframe by keyframe. Without
                # it "it followed the subject" is a claim in a report; with
                # it you can watch the box move over the original.
                "crop": crops.get(key, []),
                "source_id": source_id,
                # Keep the exact frame-derived clock. Rounding every block
                # independently makes the Web reel drift from the CFR film
                # again after current-timeline already resolved its frames.
                "at": cursor,
                "seconds": seconds,
                "gain_db": (
                    float(current_blocks[index].get("gain_db", 0.0))
                    if current_blocks else 0.0
                ),
                "audio_role": (
                    str(current_blocks[index].get("audio_role", "auto"))
                    if current_blocks else str(shot.get("audio_role", "auto"))
                ),
                "audio_completion": (
                    str(current_blocks[index].get("audio_completion", "none"))
                    if current_blocks else str(shot.get("audio_completion", "none"))
                ),
                "picture_role": (
                    str(current_blocks[index].get("picture_role", "primary_action"))
                    if current_blocks else str(shot.get("picture_role", "primary_action"))
                ),
                "coverage_claim_seconds": (
                    current_blocks[index].get("coverage_claim_seconds")
                    if current_blocks
                    else shot.get("coverage_claim_seconds")
                ),
                "in_seconds": (
                    float(current_blocks[index]["in_seconds"])
                    if current_blocks
                    else float(
                        (source_motion_details.get(key, {}).get("window") or
                         [shot.get("start_seconds", 0.0)])[0]
                    )
                ),
                "source_seconds": round(found[source_id], 3),
                "subject": subject_of(shot),
                "camera_move": move_of_shot(shot),
                "motion": motion.get(source_key, {}),
                "source_motion_details": source_motion_details.get(
                    source_key, {}
                ),
                # Every stop, not only the first. A shot that settles on
                # three watches in turn was showing one name and the word
                # "pan", in the panel whose job is saying what was planned.
                "looks": [
                    {
                        "at": one.at,
                        "seconds": one.seconds,
                        "framing": one.framing,
                        "whole": one.must_be_whole,
                    }
                    for one in looks_of(shot)
                ],
                "why": shot.get("why", ""),
                "delivered": verdicts.get(source_key, {}).get("delivered"),
                "note": verdicts.get(source_key, {}).get("note", ""),
            })
            cursor += seconds
        # Where the bed came from. The rhythm pass decides it and nothing
        # else showed it, so "why does the music sound like that" had no
        # answer on the page -- and the whole point of choosing a section is
        # that it is audible.
        edl = report.get("edl") or {}
        return JSONResponse({
            "revision": int(current.get("revision", 0)),
            "output_fps": int(current.get("output_fps") or 30),
            "music_from_seconds": float(
                current.get("music_from_seconds",
                            edl.get("music_from_seconds")) or 0.0
            ),
            "music_spans": (
                current.get("music_spans")
                if current else edl.get("music_spans") or []
            ),
            # A separate, frame-clocked track. Picture edits must round-trip
            # this unchanged unless an explicit audio editor changes it;
            # otherwise a harmless B-roll reorder silently cuts the sentence.
            "audio_assignments": current.get("audio_assignments") or [],
            "blocks": blocks,
            "seconds": round(cursor, 3),
            # Whether these boxes are what the render used or the best that
            # could be worked out afterwards. A rebuilt box is a guess with
            # the same shape as evidence, and this view exists to be
            # evidence -- so it has to say which it is holding.
            "crops_are": "recorded" if recorded else "rebuilt",
            # The same band the burn uses. The preview draws subtitles over
            # the picture so they can be corrected against it, and a second
            # opinion about where they sit would make that preview a lie.
            "safe_area": _safe_area_of(report),
        })

    def _rebuild(
        run: Run,
        wanted: list[dict] | None = None,
        wanted_audio: list[dict] | None = None,
    ):
        """The render plan again, from what the report already records.

        Nothing here costs anything: the subject positions come out of the
        cards, so this is arithmetic. It is what lets a timeline be asked for
        after the fact, and a running order be changed without re-planning
        the film.
        """

        from montagewright.clipcard import card_map
        from montagewright.executor import allocate_timeline_frames, plan_render
        from montagewright.pipeline import (
            Report, follow_subjects, probe, read_crops,
        )
        from montagewright.reframe import CropPath, retime_crop_path
        from montagewright.schema import AudioClip, EDL, Clip, reframe_of

        report = run.report() or {}
        original = report.get("selection", {}).get("shots", [])
        rhythm = report.get("rhythm", {})
        current = _current_timeline(run)
        if wanted is None:
            wanted = current.get("shots") or _timeline_blocks(run)
            for one in wanted:
                if "index" not in one:
                    one["index"] = int(one["selection_index"])
        if wanted_audio is None:
            wanted_audio = list(current.get("audio_assignments") or [])
        aspect = ASPECTS.get(
            report.get("direction", {}).get("aspect", "9:16"), 9 / 16
        )
        cards = card_map(
            run.output / "work" / "proxies",
            default_library() / "cards",
        )
        clips, sources, gains = [], {}, {}
        source_paths = (
            list((run.output / "work" / "shots").glob("*"))
            + list(Path(run.source).glob("*"))
        )

        def source_for(source_id: str):
            if source_id in sources:
                return sources[source_id]
            match = next(
                (path for path in source_paths if path.stem == source_id), None
            )
            if match is None:
                raise HTTPException(404, f"{source_id} is gone")
            sources[source_id] = probe(source_id, match)
            return sources[source_id]

        for index, entry in enumerate(wanted):
            plan = original[int(entry["index"])]
            source_id = plan["source_id"]
            source_for(source_id)
            start = float(entry["in_seconds"])
            gains[f"k{index:02d}"] = float(entry.get("gain_db", 0.0) or 0.0)
            clips.append(Clip(
                clip_id=f"k{index:02d}", source_id=source_id,
                approx_in_seconds=start,
                approx_out_seconds=start + float(entry["seconds"]),
                in_looks_like=subject_of(plan),
                energy_intent=plan.get("energy", "medium"),
                audio_role=entry.get("audio_role", plan.get("audio_role", "auto")),
                audio_completion=entry.get(
                    "audio_completion", plan.get("audio_completion", "none")
                ),
                picture_role=entry.get(
                    "picture_role", plan.get("picture_role", "primary_action")
                ),
                coverage_claim_seconds=entry.get(
                    "coverage_claim_seconds", plan.get("coverage_claim_seconds")
                ),
                reframe=reframe_of(plan),
            ))
        output_fps = int(current.get("output_fps") or 30)
        picture_spans = allocate_timeline_frames(
            [float(one["seconds"]) for one in wanted], output_fps
        )
        audio_clips = []
        for one in wanted_audio:
            # Narrative is deliberately independent and remains on the
            # master clock while pictures move. Sync/ambient assignments are
            # materialised from their picture shot below by plan_render, so
            # they travel with that shot instead of being duplicated at the
            # old absolute frame after a reorder.
            if str(one.get("role", "narrative")) != "narrative":
                continue
            source_id = str(one["source_id"])
            source_for(source_id)
            starts = int(one["timeline_start_frame"])
            containing = next(
                (
                    (index, start)
                    for index, (start, end) in enumerate(picture_spans)
                    if start <= starts < end
                ),
                None,
            )
            if containing is None:
                raise HTTPException(
                    422,
                    f"audio {one.get('audio_id', '?')} starts outside the picture timeline",
                )
            clip_index, clip_start = containing
            audio_clips.append(AudioClip(
                audio_id=str(one["audio_id"]),
                source_id=source_id,
                in_seconds=float(one["in_seconds"]),
                out_seconds=float(one["out_seconds"]),
                starts_at_clip_id=f"k{clip_index:02d}",
                offset_seconds=(starts - clip_start) / output_fps,
                role=str(one.get("role", "narrative")),
                completion=str(one.get("completion", "none")),
                gain_db=float(one.get("gain_db", 0.0) or 0.0),
                why=str(one.get("why", "")),
            ))
        recorded_edl = report.get("edl") or {}
        edl = EDL(
            project_id=run.run_id,
            clips=clips,
            audio_clips=audio_clips,
            music_from_seconds=float(
                recorded_edl.get("music_from_seconds") or 0.0
            ),
            music_spans=recorded_edl.get("music_spans") or [],
        )
        # What the render actually did, where it still applies. Rebuilding
        # is the same arithmetic for a held frame and is not for anything
        # that followed a subject: that path came out of a mask propagation
        # nothing here can repeat, so the timeline was being written from a
        # guess at what the film did.
        #
        # Per shot rather than all or nothing. A recorded path is a function
        # of source time: trim it through that clock and hold a measured end
        # if a handle extends beyond it. Never replace motion with a guessed
        # static crop merely because its last SAM sample precedes shot-out.
        stored = read_crops(run.output / "work" / "crops.json")
        previous = current.get("shots") or []
        paths: dict[str, CropPath] = {}
        stale = []
        for index, entry in enumerate(wanted):
            here = f"k{index:02d}"
            selection_index = int(entry["index"])
            match = next(
                (
                    (old_index, old)
                    for old_index, old in enumerate(previous)
                    if int(old.get("selection_index", -1)) == selection_index
                ),
                None,
            )
            if match is not None:
                old_index, old = match
                was = stored.get(f"k{old_index:02d}")
                old_in = float(old["in_seconds"])
            else:
                was = stored.get(f"k{selection_index:02d}")
                old_in = float(
                    original[selection_index].get("start_seconds", 0.0)
                )
            if was is not None:
                paths[here] = retime_crop_path(
                    was,
                    old_in_seconds=old_in,
                    new_in_seconds=float(entry["in_seconds"]),
                    new_duration_seconds=float(entry["seconds"]),
                )
            else:
                stale.append(here)
        if stale:
            rebuilt = follow_subjects(
                edl, sources, target_aspect=aspect, report=Report(),
                cards=cards, checkpoint=None, client=None,
            )
            for clip_id in stale:
                if clip_id in rebuilt:
                    paths[clip_id] = rebuilt[clip_id]
        try:
            plan = plan_render(
                edl, sources, target_aspect=aspect, crop_paths=paths,
                output_size=(
                    tuple(current["output_size"])
                    if current.get("output_size") else None
                ),
                output_fps=output_fps,
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        # A shot somebody turned down stays turned down through a re-cut.
        for segment in plan.segments:
            segment.gain_db = gains.get(segment.clip_id, 0.0)
        # NLE markers and every other consumer of this rebuilt plan must read
        # reasons from the shot now at that position, not kNN of the original
        # reel. Keep report.json immutable and project a current view here.
        projected = dict(report)
        projected_selection = dict(report.get("selection") or {})
        projected_selection["shots"] = [
            original[int(entry["index"])] for entry in wanted
        ]
        projected["selection"] = projected_selection
        projected["rhythm"] = {
            f"k{index:02d}": {
                **(rhythm.get(f"k{int(entry['index']):02d}") or {}),
                "seconds": float(entry["seconds"]),
            }
            for index, entry in enumerate(wanted)
        }
        projected["shots"] = {
            f"k{index:02d}": (report.get("shots") or {}).get(
                f"k{int(entry['index']):02d}", {}
            )
            for index, entry in enumerate(wanted)
        }
        return plan, projected, aspect

    @app.post("/api/runs/{run_id}/recut")
    async def recut(run_id: str, request: Request) -> JSONResponse:
        """Render an amended running order. No model calls, so no cost.

        Everything that was decided stays decided -- the crop, the subject,
        the move. This moves cuts and drops shots, which is the part a person
        wants after watching it once and does not want to re-plan a whole
        film for.
        """

        from montagewright.renderer import render as render_cut

        run = _run(run_id)
        payload = await request.json()
        wanted = payload.get("shots", [])
        if not wanted:
            raise HTTPException(400, "nothing left to cut")
        old_blocks = _timeline_blocks(run)
        current_state = _current_timeline(run)
        current_revision = int(current_state.get("revision", 0))
        if "base_revision" not in payload:
            raise HTTPException(409, "the editor is stale; reload before recutting")
        if int(payload["base_revision"]) != current_revision:
            raise HTTPException(
                409, "the timeline changed in another editor; reload first"
            )
        if (
            current_state.get("version") == "montagewright-current-timeline-v2"
            and "audio_assignments" not in payload
        ):
            raise HTTPException(
                409, "the editor does not understand this audio timeline; reload first"
            )
        wanted_audio = payload.get(
            "audio_assignments", current_state.get("audio_assignments") or []
        )
        old_shape = [
            (
                int(one["selection_index"]),
                round(float(one["in_seconds"]), 6),
                round(float(one["seconds"]), 6),
            )
            for one in old_blocks
        ]
        new_shape = [
            (
                int(one["index"]), round(float(one["in_seconds"]), 6),
                round(float(one["seconds"]), 6),
            )
            for one in wanted
        ]
        structural_change = old_shape != new_shape
        old_audio = current_state.get("audio_assignments") or []
        audio_changed = wanted_audio != old_audio
        plan, report, _ = _rebuild(run, wanted, wanted_audio)

        # The bed and whether the voice survives were decided when the run
        # was set up; a recut is a different running order, not a different
        # film, so both carry over.
        music = None
        if "--music" in run.command:
            candidate = Path(run.command[run.command.index("--music") + 1])
            music = candidate if candidate.exists() else None
        # A failed recut must not destroy the last good film. Render the whole
        # revision beside it, verify it, then switch the public artifacts.
        # Rendering directly into run.output let an ffmpeg failure truncate
        # deliverable.mp4 while the endpoint merely returned an error.
        staging_root = run.output / "work"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(
            prefix=".recut-", dir=staging_root
        ))
        try:
            result = render_cut(
                plan, staging, music=music, keep_segments=True,
                # Speech is a property of this run, never of whatever
                # unrelated transcripts happen to exist in the library.
                keep_voice=(
                    (
                        "--speech" not in run.command
                        or run.command[run.command.index("--speech") + 1]
                        != "never"
                    ) and bool(_transcript_map(run))
                ) or any(
                    segment.audio_role
                    in {"narrative", "sync_action", "ambient_texture"}
                    for segment in plan.segments
                ),
                under_speech=str(
                    report.get("direction", {}).get("music_under_speech")
                    or "duck"
                ),
            )
            if probe_duration(result.deliverable) <= 0:
                raise RuntimeError("the recut produced no readable video")

            from montagewright.measure.storage import write_json
            from montagewright.pipeline import write_crops
            from montagewright.reframe import CropPath, Keyframe

            recorded_paths = {}
            for segment in plan.segments:
                path = segment.crop_path
                if path is None and segment.crop is not None:
                    path = CropPath([
                        Keyframe(0.0, segment.crop),
                        Keyframe(segment.duration_seconds, segment.crop),
                    ])
                if path is not None:
                    recorded_paths[segment.clip_id] = path
            write_crops(recorded_paths, staging / "work" / "crops.json")
            from montagewright.executor import allocate_timeline_frames

            frame_spans = allocate_timeline_frames(
                [one.duration_seconds for one in plan.segments],
                plan.output_fps,
            )
            manifest = {
                "version": "montagewright-current-timeline-v2",
                "revision": current_revision + 1,
                "output_fps": plan.output_fps,
                "output_size": list(plan.output_size),
                "music_from_seconds": plan.music_from_seconds,
                "music_spans": plan.music_spans,
                "shots": [
                    {
                        "selection_index": int(wanted[index]["index"]),
                        "in_seconds": segment.in_seconds,
                        "start_frame": start,
                        "frame_count": end - start,
                        "seconds": (end - start) / plan.output_fps,
                        "gain_db": segment.gain_db,
                        "audio_role": segment.audio_role,
                        "audio_completion": segment.audio_completion,
                        "picture_role": segment.picture_role,
                        "coverage_claim_seconds": segment.coverage_claim_seconds,
                    }
                    for index, (segment, (start, end)) in enumerate(
                        zip(plan.segments, frame_spans, strict=True)
                    )
                ],
                "audio_assignments": [
                    {
                        "audio_id": audio.audio_id,
                        "source_id": audio.source.source_id,
                        "in_seconds": audio.in_seconds,
                        "out_seconds": audio.out_seconds,
                        "timeline_start_frame": audio.timeline_start_frame,
                        "frame_count": audio.frame_count,
                        "role": audio.role,
                        "completion": audio.completion,
                        "gain_db": audio.gain_db,
                        "why": audio.why,
                    }
                    for audio in plan.audio_assignments
                ],
            }
            write_json(
                staging / "work" / "current-timeline.json", manifest
            )

            for name in ("picture.mp4", "deliverable.mp4", "preview.mp4"):
                os.replace(staging / name, run.output / name)
            laid = staging / "bed-as-laid.m4a"
            if laid.exists():
                os.replace(laid, run.output / laid.name)
            voice = staging / "voice-as-laid.m4a"
            if voice.exists():
                os.replace(voice, run.output / voice.name)
            elif not plan.audio_assignments:
                (run.output / "voice-as-laid.m4a").unlink(missing_ok=True)
            # Segments are not a public endpoint, so switch them after the
            # three playable artifacts are safely in place.
            old_segments = run.output / "segments"
            new_segments = staging / "segments"
            retired = run.output / ".segments-before-recut"
            if retired.exists():
                shutil.rmtree(retired)
            if old_segments.exists():
                os.replace(old_segments, retired)
            os.replace(new_segments, old_segments)
            shutil.rmtree(retired, ignore_errors=True)
            (run.output / "work").mkdir(parents=True, exist_ok=True)
            for name in ("crops.json", "current-timeline.json"):
                os.replace(staging / "work" / name, run.output / "work" / name)
            if audio_changed:
                subtitles = run.output / "work" / "subtitles.json"
                if subtitles.exists():
                    # Edited subtitle cues have no source anchor in v1. Keep
                    # the user's copy recoverable, but never burn its old
                    # absolute times onto a different running order.
                    shutil.copy2(
                        subtitles,
                        run.output / "work" / "subtitles-before-recut.json",
                    )
                    subtitles.unlink()
            if structural_change:
                _retime_graphics_for_current_cut(
                    run, old_blocks, manifest["shots"]
                )
            # These all encode absolute timing or the previous running order.
            # They are derived and will be rebuilt from current-timeline.json.
            for name in (
                "timeline.xml", "timeline.fcpxml", "subtitles.srt",
                "deliverable-subtitled.mp4", "deliverable-graphics.mp4",
                "deliverable-graphics-subtitled.mp4",
            ):
                (run.output / name).unlink(missing_ok=True)
            for wave in run.output.glob("wave-*.png"):
                wave.unlink(missing_ok=True)
            for cache in (
                run.output / "work" / "graphics-preview",
                run.output / "work" / "graphics-render",
                run.output / "work" / "graphics-subtitle-render",
            ):
                shutil.rmtree(cache, ignore_errors=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        # What was rendered, from the plan that rendered it. This read a
        # name belonging to the rebuild's own scope, so a recut that had
        # already re-encoded every segment raised on the way to saying so.
        return JSONResponse({
            "seconds": round(result.duration_seconds, 3),
            "shots": len(plan.segments),
            "revision": current_revision + 1,
            "subtitles_reset": audio_changed,
        })

    @app.get("/api/runs/{run_id}/transcripts")
    def transcripts(run_id: str) -> JSONResponse:
        """What was heard, per source, before anything was cut.

        Worth reading before the editorial passes are paid for: a transcript
        that got the product name wrong sends every later decision after the
        wrong sentence.
        """

        from montagewright.transcript import lines_of

        run = _run(run_id)
        out = {}
        for source_id, card in sorted(_transcript_map(run).items()):
            out[source_id] = {
                "language": card.get("language"),
                "summary": card.get("summary", ""),
                "lines": [
                    {
                        "text": line.text,
                        "heard": line.heard,
                        "speaker": line.speaker,
                        "starts_seconds": line.starts_seconds,
                        "ends_seconds": line.ends_seconds,
                        "corrected": line.corrected,
                    }
                    for line in lines_of(card)
                ],
            }
        return JSONResponse({"sources": out})

    def _run(run_id: str) -> Run:
        run = RUNS.get(run_id)
        if run is None:
            # Runs are picked up off disk lazily, and only the listing was
            # doing it -- so after a restart every other endpoint answered
            # "no such run" until something happened to ask for the list.
            recall()
            run = RUNS.get(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return run

    @app.get("/api/runs/{run_id}")
    def status(run_id: str, since: int = 0) -> JSONResponse:
        run = _run(run_id)
        return JSONResponse({
            "state": run.state,
            "returncode": run.returncode,
            "lines": run.lines[since:],
            "total_lines": len(run.lines),
            "report": run.report() if run.state != "running" else None,
            "has_video": (run.output / "preview.mp4").exists(),
            # What this run was made from, so another round of the same
            # material does not have to be pointed at it again. Opening a
            # cut and asking for a new round is the ordinary way to try a
            # different length, and it was starting from an empty form.
            "source_path": _ran_with(run, "render", after=False),
            "music_path": _ran_with(run, "--music"),
            "brief_text": _brief_of(run),
        })

    @app.post("/api/runs/{run_id}/resume")
    def resume(run_id: str) -> JSONResponse:
        """Run it again into the same place.

        Nothing already paid for is paid for twice: the cards, transcripts,
        the direction and the selection are all keyed on disk, so this picks
        up where the quota or the bad path stopped it.
        """

        run = _run(run_id)
        if not run.command:
            # Recorded before the command was kept. What carries the value is
            # the work directory -- cards, transcripts, the direction, the
            # selection -- and that belongs to the output, so a plain run
            # pointed back at it picks all of them up. The options it was
            # started with are lost; the money is not.
            if not run.source or not Path(run.source).exists():
                raise HTTPException(
                    400, "this run did not record how it started"
                )
            run.command = [
                sys.executable, "-u", "-m", "montagewright.cli", "render",
                run.source, "--output", str(run.output), "--review",
            ]
            run.lines.append(
                "— 這一輪是舊版存的，沒有記下當初的選項；"
                "用預設值續跑，已經算好的東西都會沿用 —"
            )
        if run.process is not None and run.process.poll() is None:
            raise HTTPException(409, "it is still going")
        run.lines.append("— 續跑 —")
        run.state = "running"
        run.process = subprocess.Popen(
            run.command,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        run.remember()
        threading.Thread(target=_collect, args=(run,), daemon=True).start()
        return JSONResponse({"state": run.state})

    @app.post("/api/runs/{run_id}/stop")
    def stop(run_id: str) -> JSONResponse:
        run = _run(run_id)
        if run.process is not None and run.process.poll() is None:
            run.process.terminate()
            run.state = "stopped"
        return JSONResponse({"state": run.state})

    @app.get("/api/runs/{run_id}/video")
    def video(run_id: str):
        path = _run(run_id).output / "preview.mp4"
        if not path.exists():
            raise HTTPException(404, "no preview yet")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/runs/{run_id}/deliverable")
    def deliverable(run_id: str):
        path = _run(run_id).output / "deliverable.mp4"
        if not path.exists():
            raise HTTPException(404, "no deliverable yet")
        return FileResponse(
            path, media_type="video/mp4", filename=f"{run_id}.mp4"
        )

    @app.get("/api/runs/{run_id}/timeline/{flavour}")
    def timeline(run_id: str, flavour: str):
        suffix = {"premiere": "xml", "finalcut": "fcpxml"}.get(flavour)
        if suffix is None:
            raise HTTPException(400, "premiere or finalcut")
        run = _run(run_id)
        path = run.output / f"timeline.{suffix}"
        if not path.exists():
            # Built on request. Asking for one up front and finding out
            # afterwards that you wanted it meant running the whole thing
            # again, and everything it needs is already written down.
            from montagewright.timeline import to_fcpxml, to_xmeml

            plan, report, _ = _rebuild(run)
            width, height = plan.output_size
            # The bed this cut was laid over, so the timeline carries it
            # too rather than opening as a silent film.
            bed = (
                run.output / "bed-as-laid.m4a"
                if (run.output / "bed-as-laid.m4a").exists() else None
            )
            if bed is None and "--music" in run.command:
                maybe = Path(run.command[run.command.index("--music") + 1])
                bed = maybe if maybe.exists() else None
            build = to_xmeml if flavour == "premiere" else to_fcpxml
            graphics = run.output / "graphics-overlay.mov"
            voice = run.output / "voice-as-laid.m4a"
            path.write_text(
                build(plan, report, name=run.output.name,
                      width=width, height=height, music=bed,
                      voice=voice if voice.exists() else None,
                      graphics=graphics if graphics.exists() else None),
                encoding="utf-8",
            )
        return FileResponse(
            path, media_type="application/xml",
            filename=f"{run_id}.{suffix}",
        )

    @app.get("/api/fonts")
    def fonts(lang: str = "", run_id: str = "") -> JSONResponse:
        """What this machine can set subtitles in.

        A flag on the command line is a flag most people never find, and
        typing a path to a font is worse than that.
        """

        from montagewright.subtitles import fonts_here

        # The language of the cut, not the language this was written in.
        # Asking for Chinese faces while somebody cuts a Japanese interview
        # hands them a list with nothing they want in it.
        if not lang and run_id:
            run = RUNS.get(run_id)
            if run is not None:
                lang = next(
                    (
                        str(card.get("language") or "")
                        for card in _transcript_map(run).values()
                        if card.get("language")
                    ),
                    "",
                )
                if not lang and "--locale" in run.command:
                    lang = run.command[run.command.index("--locale") + 1]
        return JSONResponse({
            "lang": lang or "zh-tw",
            "fonts": fonts_here(lang or "zh-tw"),
        })

    @app.get("/api/runs/{run_id}/subtitle-track")
    def subtitle_track(run_id: str) -> JSONResponse:
        """The subtitles as a track, so they can be read against the picture.

        Same lines the file gets. A line that is wrong is wrong in both, and
        the place to notice is next to the shot it is under.
        """

        run = _run(run_id)
        # What the transcripts and the running order produce, before anyone
        # touched it. Sent so the track can mark the lines that were changed
        # -- comparing against `heard` marked almost all of them, because
        # correcting what the recogniser misheard is the whole point of the
        # pass that produced them.
        derived = {
            round(line.starts_seconds, 3): line.text
            for line in _subtitle_lines(run, edits=False)
        }
        return JSONResponse({
            "lines": [
                {
                    "at": round(line.starts_seconds, 3),
                    "until": round(line.ends_seconds, 3),
                    "text": line.text,
                    "speaker": line.speaker,
                    "heard": line.heard,
                    "derived": derived.get(round(line.starts_seconds, 3), ""),
                    "timing_source": line.timing_source,
                    "timing_confidence": line.timing_confidence,
                    "timing_locked": line.timing_locked,
                }
                for line in _subtitle_lines(run)
            ],
            "edited": (run.output / "work" / "subtitles.json").exists(),
        })

    @app.get("/api/runs/{run_id}/graphics-track")
    def graphics_track(run_id: str) -> JSONResponse:
        """The independent editorial-graphics track and its copy provenance."""

        from montagewright.graphics import (
            GRAPHIC_PRESET_REGISTRY_VERSION, CopyFact, GraphicsPlan,
            graphic_presets_for_editor,
            templates_for_editor,
        )

        run = _run(run_id)
        source = run.output / "work" / "graphics.json"
        if source.exists():
            try:
                plan = GraphicsPlan.model_validate_json(
                    source.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as error:
                raise HTTPException(422, f"graphics plan is unreadable: {error}")
        else:
            approved_copy = run.output / "work" / "approved-copy.json"
            if approved_copy.exists():
                try:
                    payload = json.loads(approved_copy.read_text(encoding="utf-8"))
                    plan = GraphicsPlan(facts=payload.get("facts", []))
                except (OSError, ValueError, TypeError) as error:
                    raise HTTPException(
                        422, f"approved brief copy is unreadable: {error}"
                    )
            else:
                plan = GraphicsPlan()
        # A brief can gain approved copy after a manual graphics track was
        # first saved. Keep the copy bank current without overwriting any
        # cue or persisting until the editor actually saves.
        approved_copy = run.output / "work" / "approved-copy.json"
        if approved_copy.exists():
            try:
                trusted = json.loads(
                    approved_copy.read_text(encoding="utf-8")
                ).get("facts", [])
                known = {fact.fact_id for fact in plan.facts}
                plan = GraphicsPlan.model_validate({
                    **plan.model_dump(mode="json"),
                    "facts": [
                        *plan.model_dump(mode="json")["facts"],
                        *(fact for fact in trusted
                          if fact.get("fact_id") not in known),
                    ],
                })
            except (OSError, ValueError, TypeError) as error:
                raise HTTPException(
                    422, f"approved brief copy is unreadable: {error}"
                )
        by_id = {fact.fact_id: fact for fact in plan.facts}
        layout_path = run.output / "work" / "graphics-render" / "layout.json"
        try:
            layout = (
                json.loads(layout_path.read_text(encoding="utf-8"))
                if layout_path.exists() else {}
            )
        except (OSError, ValueError):
            layout = {}
        candidate_path = run.output / "work" / "brief-candidates.json"
        brief_path = None
        try:
            if candidate_path.exists():
                candidate_data = json.loads(
                    candidate_path.read_text(encoding="utf-8")
                )
            else:
                brief_path = _brief_path_of(run)
                if brief_path is not None and brief_path.exists():
                    from montagewright.brief import load_brief

                    candidate_data = load_brief(brief_path).candidates_json()
                else:
                    candidate_data = {
                        "brief_sha256": "", "candidates": [],
                        "instructions": [],
                    }
        except (OSError, ValueError) as error:
            raise HTTPException(422, f"brief candidates are unreadable: {error}")
        # Candidate provenance is server-issued just like approved copy. It
        # is not approval, but exposing these immutable facts lets the Web UI
        # cite ordinary Brief prose without manufacturing a source label.
        candidate_facts = candidate_data.get("facts", []) or []
        if candidate_facts:
            try:
                known = {fact.fact_id for fact in plan.facts}
                plan = GraphicsPlan.model_validate({
                    **plan.model_dump(mode="json"),
                    "facts": [
                        *plan.model_dump(mode="json")["facts"],
                        *(fact for fact in candidate_facts
                          if fact.get("fact_id") not in known),
                    ],
                })
            except (ValueError, TypeError) as error:
                raise HTTPException(
                    422, f"brief candidate facts are unreadable: {error}"
                )
        # Older clients were once able to persist an evidence-looking label
        # without the artifact that proves it. Reconcile those records while
        # loading: verified server facts keep their lineage; everything else
        # becomes an unapproved user draft instead of being grandfathered as
        # OCR/transcript/Gemini evidence forever.
        try:
            trusted_catalog = {
                fact.fact_id: fact
                for fact in [
                    *(
                        CopyFact.model_validate(raw)
                        for raw in (
                            json.loads(approved_copy.read_text(encoding="utf-8"))
                            .get("facts", [])
                            if approved_copy.exists() else []
                        )
                    ),
                    *(
                        CopyFact.model_validate(raw)
                        for raw in candidate_facts
                    ),
                ]
            }
            downgraded: set[str] = set()
            reconciled_facts = []
            for fact in plan.facts:
                trusted = trusted_catalog.get(fact.fact_id)
                if fact.source_kind == "user" or (
                    trusted is not None
                    and fact.model_dump() == trusted.model_dump()
                ):
                    reconciled_facts.append(fact)
                    continue
                downgraded.add(fact.fact_id)
                reconciled_facts.append(fact.model_copy(update={
                    "source_kind": "user",
                    "source_reference": "legacy-unverified",
                    "source_sha256": "",
                    "text_sha256": "",
                    "allowed_kinds": [],
                    "approved": False,
                    "approved_by": None,
                }))
            if downgraded:
                plan = GraphicsPlan(
                    version=plan.version,
                    revision=plan.revision,
                    brand=plan.brand,
                    facts=reconciled_facts,
                    cues=[
                        cue.model_copy(update={"status": "draft"})
                        if cue.primary_fact_id in downgraded
                        or cue.secondary_fact_id in downgraded
                        else cue
                        for cue in plan.cues
                    ],
                )
        except (OSError, ValueError, TypeError) as error:
            raise HTTPException(422, f"graphics provenance is unreadable: {error}")
        by_id = {fact.fact_id: fact for fact in plan.facts}
        return JSONResponse({
            **plan.model_dump(mode="json"),
            "templates": templates_for_editor(),
            "presets": graphic_presets_for_editor(),
            "preset_registry_version": GRAPHIC_PRESET_REGISTRY_VERSION,
            "layout": layout,
            "brief_sha256": candidate_data.get("brief_sha256", ""),
            "brief_source": str(brief_path) if brief_path else "",
            "brief_candidates": candidate_data.get("candidates", []),
            "brief_instructions": candidate_data.get("instructions", []),
            "resolved": [
                {
                    **cue.model_dump(mode="json"),
                    "primary_text": by_id[cue.primary_fact_id].exact_text,
                    "secondary_text": (
                        by_id[cue.secondary_fact_id].exact_text
                        if cue.secondary_fact_id else ""
                    ),
                }
                for cue in plan.cues
            ],
        })

    @app.put("/api/runs/{run_id}/graphics-track")
    async def edit_graphics_track(
        run_id: str, request: Request
    ) -> JSONResponse:
        """Save edits immediately; approved copy remains a typed contract."""

        from montagewright.graphics import (
            CopyFact,
            GraphicsPlan,
            validate_brief_authority,
            validate_for_render,
        )

        run = _run(run_id)
        try:
            plan = GraphicsPlan.model_validate(await request.json())
        except ValueError as error:
            raise HTTPException(422, str(error))
        destination = run.output / "work" / "graphics.json"
        stored_plan = GraphicsPlan()
        if destination.exists():
            try:
                stored_plan = GraphicsPlan.model_validate_json(
                    destination.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as error:
                raise HTTPException(422, f"stored graphics are unreadable: {error}")
            if plan.revision != stored_plan.revision:
                raise HTTPException(
                    409,
                    "字卡已在另一個視窗更新；請重新整理後再修改",
                )
        elif plan.revision != 0:
            # Revision is assigned by this server. A client cannot create a
            # new authority record at an arbitrary future revision.
            raise HTTPException(
                409, "字卡尚未建立；請以 revision 0 重新儲存"
            )
        trusted_human = {
            fact.fact_id: fact for fact in stored_plan.facts
            if fact.approved_by == "human_review"
        }
        untrusted_human = [
            fact.fact_id for fact in plan.facts
            if fact.approved_by == "human_review"
            and (
                fact.fact_id not in trusted_human
                or fact.model_dump() != trusted_human[fact.fact_id].model_dump()
            )
        ]
        if untrusted_human:
            raise HTTPException(
                422,
                "人工核准必須經由明確的核准動作，不能在一般存檔中宣稱："
                + ", ".join(untrusted_human),
            )
        approved_copy = run.output / "work" / "approved-copy.json"
        try:
            authority = (
                [
                    CopyFact.model_validate(fact)
                    for fact in json.loads(
                        approved_copy.read_text(encoding="utf-8")
                    ).get("facts", [])
                ]
                if approved_copy.exists() else []
            )
            validate_brief_authority(plan, authority)
        except (OSError, ValueError, TypeError) as error:
            raise HTTPException(422, str(error))
        # A generic PUT may add user-authored drafts, but it may not claim an
        # evidence provenance. Non-user facts must be byte-for-byte records
        # already issued by this run's server artifacts or its stored plan.
        trusted_non_user = {
            fact.fact_id: fact for fact in authority
            if fact.source_kind != "user"
        }
        # Facts already committed by this server remain server-owned evidence
        # on later revisions. Without this, a legitimate OCR/transcript/model
        # fact can be loaded but the next ordinary style edit rejects it as a
        # forgery even when the bytes are unchanged.
        trusted_non_user.update({
            fact.fact_id: fact for fact in stored_plan.facts
            if fact.source_kind != "user"
        })
        candidate_path = run.output / "work" / "brief-candidates.json"
        try:
            if candidate_path.exists():
                candidate_payload = json.loads(
                    candidate_path.read_text(encoding="utf-8")
                )
            else:
                brief_path = _brief_path_of(run)
                if brief_path is not None and brief_path.exists():
                    from montagewright.brief import load_brief

                    candidate_payload = load_brief(brief_path).candidates_json()
                else:
                    candidate_payload = {"facts": []}
            for raw in candidate_payload.get("facts", []) or []:
                fact = CopyFact.model_validate(raw)
                trusted_non_user[fact.fact_id] = fact
        except (OSError, ValueError, TypeError) as error:
            raise HTTPException(422, f"brief candidates are unreadable: {error}")
        forged = [
            fact.fact_id for fact in plan.facts
            if fact.source_kind != "user"
            and (
                fact.fact_id not in trusted_non_user
                or fact.model_dump()
                != trusted_non_user[fact.fact_id].model_dump()
            )
        ]
        if forged:
            raise HTTPException(
                422,
                "文字來源必須由伺服器證據建立，不能由一般存檔宣稱："
                + ", ".join(forged),
            )
        source = next(
            (
                run.output / name
                for name in ("deliverable.mp4", "picture.mp4")
                if (run.output / name).exists()
            ),
            None,
        )
        duration = probe_duration(source) if source else 0.0
        warnings = (
            validate_for_render(plan, duration_seconds=duration)
            if duration else []
        )
        from montagewright.measure.storage import write_json
        from starlette.concurrency import run_in_threadpool

        submitted_revision = plan.revision
        lock = _graphics_state_lock(run)
        await run_in_threadpool(lock.acquire)
        try:
            try:
                current_revision = (
                    GraphicsPlan.model_validate_json(
                        destination.read_text(encoding="utf-8")
                    ).revision
                    if destination.exists() else 0
                )
            except (OSError, ValueError) as error:
                raise HTTPException(
                    422, f"stored graphics are unreadable: {error}"
                )
            if submitted_revision != current_revision:
                raise HTTPException(
                    409,
                    "字卡已在另一個視窗更新；請重新整理後再修改",
                )
            plan = plan.model_copy(update={"revision": current_revision + 1})
            try:
                write_json(destination, plan)
                _invalidate_graphics_delivery(run)
            except OSError as error:
                raise HTTPException(
                    503, f"graphics storage is temporarily unavailable: {error}"
                )
        finally:
            lock.release()
        return JSONResponse({
            "facts": len(plan.facts), "cues": len(plan.cues),
            "warnings": warnings, "revision": plan.revision,
        })

    @app.post("/api/runs/{run_id}/approve-graphic/{graphic_id}")
    async def approve_graphic(
        run_id: str, graphic_id: str, request: Request
    ) -> JSONResponse:
        """Record an explicit human copy-review action on one saved cue."""

        import hashlib
        from montagewright.graphics import GraphicsPlan
        from montagewright.measure.storage import write_json

        run = _run(run_id)
        destination = run.output / "work" / "graphics.json"
        if not destination.exists():
            raise HTTPException(404, "save the graphics draft before approval")
        try:
            wanted_revision = int((await request.json()).get("revision", -1))
            plan = GraphicsPlan.model_validate_json(
                destination.read_text(encoding="utf-8")
            )
            cue = next(
                cue for cue in plan.cues if cue.graphic_id == graphic_id
            )
        except (OSError, ValueError, StopIteration) as error:
            raise HTTPException(422, f"cannot approve graphic: {error}")
        if wanted_revision != plan.revision:
            raise HTTPException(
                409, "字卡已在另一個視窗更新；請重新整理後再核准"
            )
        facts = list(plan.facts)
        updates: dict[str, str] = {}
        for which, fact_id in (
            ("primary", cue.primary_fact_id),
            ("secondary", cue.secondary_fact_id),
        ):
            if not fact_id:
                continue
            fact = plan.fact(fact_id)
            approved_id = fact_id
            if fact.source_kind != "user":
                approved_id = f"user.{cue.graphic_id}.{which}"
                facts = [item for item in facts if item.fact_id != approved_id]
            approved = fact.model_copy(update={
                "fact_id": approved_id,
                "source_kind": "user",
                "source_reference": "web-ui-explicit-review",
                "source_sha256": "",
                "text_sha256": hashlib.sha256(
                    fact.exact_text.encode("utf-8")
                ).hexdigest(),
                "allowed_kinds": [],
                "approved": True,
                "approved_by": "human_review",
            })
            facts = [
                approved if item.fact_id == fact_id else item for item in facts
            ] if approved_id == fact_id else [*facts, approved]
            updates[f"{which}_fact_id"] = approved_id
        approved_cue = cue.model_copy(update={**updates, "status": "approved"})
        cues = [
            approved_cue if item.graphic_id == graphic_id else item
            for item in plan.cues
        ]
        try:
            plan = GraphicsPlan.model_validate({
                **plan.model_dump(mode="json"),
                "revision": plan.revision + 1,
                "facts": [fact.model_dump(mode="json") for fact in facts],
                "cues": [item.model_dump(mode="json") for item in cues],
            })
        except ValueError as error:
            raise HTTPException(422, f"cannot approve graphic: {error}")
        from starlette.concurrency import run_in_threadpool

        lock = _graphics_state_lock(run)
        await run_in_threadpool(lock.acquire)
        try:
            try:
                current_revision = GraphicsPlan.model_validate_json(
                    destination.read_text(encoding="utf-8")
                ).revision
            except (OSError, ValueError) as error:
                raise HTTPException(
                    422, f"stored graphics are unreadable: {error}"
                )
            if current_revision != wanted_revision:
                raise HTTPException(
                    409, "字卡已在另一個視窗更新；請重新整理後再核准"
                )
            try:
                write_json(destination, plan)
                _invalidate_graphics_delivery(run)
            except OSError as error:
                raise HTTPException(
                    503, f"graphics storage is temporarily unavailable: {error}"
                )
        finally:
            lock.release()
        return JSONResponse(plan.model_dump(mode="json"))

    @app.post("/api/runs/{run_id}/burn-graphics")
    def burn_graphics_track(run_id: str) -> JSONResponse:
        """Render approved cards over a copy, never over the clean master."""

        from montagewright.graphics import (
            CopyFact,
            GraphicsPlan,
            burn_graphics,
            render_graphics_overlay,
            validate_brief_authority,
        )
        from montagewright.grounding import read_runtime_beat_grid

        run = _run(run_id)
        stored = run.output / "work" / "graphics.json"
        if not stored.exists():
            raise HTTPException(404, "this run has no graphics track")
        try:
            plan = GraphicsPlan.model_validate_json(
                stored.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise HTTPException(422, str(error))
        approved_copy = run.output / "work" / "approved-copy.json"
        try:
            authority = (
                [
                    CopyFact.model_validate(fact)
                    for fact in json.loads(
                        approved_copy.read_text(encoding="utf-8")
                    ).get("facts", [])
                ]
                if approved_copy.exists() else []
            )
            validate_brief_authority(plan, authority)
        except (OSError, ValueError, TypeError) as error:
            raise HTTPException(422, str(error))
        clean = next(
            (
                run.output / name
                for name in ("deliverable.mp4", "picture.mp4")
                if (run.output / name).exists()
            ),
            None,
        )
        if clean is None:
            raise HTTPException(404, "this run has no finished cut")
        merged_graphics = run.output / "deliverable-graphics-subtitled.mp4"
        try:
            beat_grid = read_runtime_beat_grid(
                run.output / "work" / "graphics-beat-grid.json"
            )
            subtitle_track = _prepare_run_subtitle_track(run, clean)
            subtitles = subtitle_track.windows
            subtitle_boxes = subtitle_track.boxes
            subtitle_overlays = list(subtitle_track.overlays)
            destination = (
                merged_graphics if subtitle_overlays
                else run.output / "deliverable-graphics.mp4"
            )
            made = burn_graphics(
                clean, plan, destination,
                work=run.output / "work" / "graphics-render",
                subtitle_windows=subtitles,
                subtitle_boxes=subtitle_boxes,
                subtitle_overlays=subtitle_overlays,
                layout_evidence=_graphics_layout_evidence(run, plan),
                beat_grid=beat_grid,
            )
            overlay = render_graphics_overlay(
                clean, plan, run.output / "graphics-overlay.mov",
                work=run.output / "work" / "graphics-render",
                subtitle_windows=subtitles,
                subtitle_boxes=subtitle_boxes,
                layout_evidence=_graphics_layout_evidence(run, plan),
                beat_grid=beat_grid,
            )
            for stale_timeline in (
                run.output / "timeline.xml", run.output / "timeline.fcpxml"
            ):
                stale_timeline.unlink(missing_ok=True)
            if not subtitle_overlays:
                merged_graphics.unlink(missing_ok=True)
        except (
            OSError, RuntimeError, ValueError, subprocess.CalledProcessError,
        ) as error:
            raise HTTPException(422, str(error))
        return JSONResponse({
            "file": made.name,
            "overlay": overlay.name,
            "cues": sum(cue.status == "approved" for cue in plan.cues),
            "layout": json.loads(
                (run.output / "work" / "graphics-render" / "layout.json")
                .read_text(encoding="utf-8")
            ),
        })

    @app.post("/api/runs/{run_id}/graphics-preview/{graphic_id}")
    async def compile_graphic_preview(
        run_id: str, graphic_id: str, request: Request
    ) -> JSONResponse:
        """Compile one card with the production Pillow renderer for the UI."""

        import hashlib
        from functools import partial
        from starlette.concurrency import run_in_threadpool
        from montagewright.graphics import (
            GRAPHICS_RENDERER_VERSION, GraphicsPlan, _layout_frames,
            _swept_rect, resolve_graphic_window,
            compile_graphic,
        )
        from montagewright.grounding import (
            beat_grid_payload, read_runtime_beat_grid,
        )

        run = _run(run_id)
        try:
            plan = GraphicsPlan.model_validate(await request.json())
        except ValueError as error:
            field = "plan"
            errors = getattr(error, "errors", lambda: [])()
            if errors:
                field = ".".join(str(part) for part in errors[0].get("loc", ()))
            raise _graphics_preview_problem(
                422, "GRAPHICS_PLAN_INVALID", str(error), field=field,
                suggested_patch={"action": "correct_field"},
            )
        try:
            next(cue for cue in plan.cues if cue.graphic_id == graphic_id)
        except StopIteration:
            raise _graphics_preview_problem(
                422, "GRAPHIC_NOT_FOUND",
                f"no graphic named {graphic_id} exists in this plan",
                field="graphic_id",
                suggested_patch={"action": "select_existing_graphic"},
            )
        picture = next(
            (
                run.output / name
                for name in ("deliverable.mp4", "picture.mp4")
                if (run.output / name).exists()
            ),
            None,
        )
        if picture is None:
            raise HTTPException(404, "no picture available for graphics preview")
        try:
            width, height = await run_in_threadpool(_video_display_size, picture)
            output_fps, picture_duration = await run_in_threadpool(
                _video_timing, picture
            )
            picture_stat = picture.stat()
            subtitle_track = await run_in_threadpool(partial(
                _prepare_run_subtitle_track, run, picture,
                dimensions=(width, height),
            ))
        except (
            OSError, RuntimeError, ValueError, subprocess.SubprocessError,
        ) as error:
            raise _graphics_preview_problem(
                503, "PREVIEW_SOURCE_UNAVAILABLE", str(error),
                field="preview.source",
                suggested_patch={"action": "retry"},
            )
        subtitle_boxes_all = subtitle_track.boxes
        beat_grid = read_runtime_beat_grid(
            run.output / "work" / "graphics-beat-grid.json"
        )
        evidence_map = _graphics_layout_evidence(run, plan)
        digest = hashlib.sha256(
            json.dumps(
                {
                    "target_graphic_id": graphic_id,
                    "plan": plan.model_dump(mode="json"),
                    "width": width, "height": height,
                    "output_fps": output_fps,
                    "picture_duration": picture_duration,
                    "picture": {
                        "size": picture_stat.st_size,
                        "mtime_ns": picture_stat.st_mtime_ns,
                    },
                    "renderer_version": GRAPHICS_RENDERER_VERSION,
                    "beat_grid": (
                        beat_grid_payload(beat_grid) if beat_grid is not None
                        else None
                    ),
                    "subtitle_keepout": subtitle_boxes_all,
                    "evidence": {
                        key: {
                            "subject_boxes": item.subject_boxes,
                            "source": item.source,
                        }
                        for key, item in evidence_map.items()
                    },
                },
                ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        preview_dir = run.output / "work" / "graphics-preview"
        metadata_path = preview_dir / f"{digest}.json"
        try:
            metadata = _load_graphics_preview_cache(
                metadata_path, preview_dir, graphic_id
            )
        except OSError as error:
            raise _graphics_preview_problem(
                503, "PREVIEW_CACHE_UNAVAILABLE", str(error),
                field="preview.cache",
                suggested_patch={"action": "retry"},
            )
        async def compile_missing_preview() -> dict:
            plan_order = {
                item.graphic_id: index for index, item in enumerate(plan.cues)
            }
            eligible = [
                item for item in plan.cues
                if item.status == "approved" or item.graphic_id == graphic_id
            ]
            # Layout only the temporal collision component containing the
            # target. A title at 00:03 cannot affect one at 01:10, and
            # recompiling the whole film on every keystroke blocks the editor.
            component = {graphic_id}
            changed = True
            while changed:
                changed = False
                members = [item for item in eligible if item.graphic_id in component]
                for item in eligible:
                    if item.graphic_id in component:
                        continue
                    if any(
                        item.at_seconds < other.at_seconds + other.duration_seconds
                        and item.at_seconds + item.duration_seconds > other.at_seconds
                        for other in members
                    ):
                        component.add(item.graphic_id)
                        changed = True
            preview_cues = [
                item for item in eligible if item.graphic_id in component
            ]
            # Preview the exact joint solution this plan will use once the
            # selected draft is approved, so approval cannot swap two cards.
            preview_cues.sort(key=lambda item: (
                item.position == "auto", plan_order[item.graphic_id]
            ))
            cue_windows = {
                item.graphic_id: resolve_graphic_window(
                    item, beat_grid=beat_grid, output_fps=output_fps,
                    timeline_duration=picture_duration,
                )
                for item in preview_cues
            }
            earlier: list[tuple[Any, Any]] = []
            card = metadata = None
            joint_layout: dict[str, dict] = {}
            for current in preview_cues:
                current_start, current_end = cue_windows[current.graphic_id]
                current_boxes = [
                    (left, top, wide, tall)
                    for start, end, left, top, wide, tall in subtitle_boxes_all
                    if current_start < end and current_end > start
                ]
                overlapping = [
                    (other, other_card) for other, other_card in earlier
                    if current_start < cue_windows[other.graphic_id][1]
                    and current_end > cue_windows[other.graphic_id][0]
                    and (
                        current.collision_policy == "avoid"
                        or other.collision_policy == "avoid"
                    )
                ]
                keepouts = [*current_boxes, *(
                    _swept_rect(other, other_card, width, height)
                    for other, other_card in overlapping
                )]
                child_digest = hashlib.sha256(
                    f"{digest}:{current.graphic_id}".encode("utf-8")
                ).hexdigest()
                current_path = preview_dir / f"{child_digest}.png"
                try:
                    frames = await run_in_threadpool(
                        partial(
                            _layout_frames, picture,
                            current.model_copy(update={"at_seconds": current_start}),
                            cache_dir=preview_dir / "frames",
                        )
                    )
                except (
                    OSError, RuntimeError, subprocess.SubprocessError,
                ) as error:
                    raise _graphics_preview_problem(
                        503, "PREVIEW_FRAME_DECODE_FAILED", str(error),
                        field="preview.source",
                        suggested_patch={"action": "retry"},
                    )
                if len(frames) < 3:
                    raise _graphics_preview_problem(
                        503, "PREVIEW_FRAME_DECODE_FAILED",
                        f"only {len(frames)}/3 picture samples were decoded",
                        field="preview.source",
                        suggested_patch={"action": "retry"},
                    )
                try:
                    current_card, current_metadata = await run_in_threadpool(
                        partial(
                            compile_graphic, current, plan,
                            width=width, height=height, into=current_path,
                            frames=frames,
                            evidence=evidence_map.get(current.graphic_id),
                            forbidden_positions=set(), keepout_rects=keepouts,
                            beat_grid=beat_grid,
                            output_fps=output_fps,
                            timeline_duration=picture_duration,
                        )
                    )
                except ValueError as error:
                    raise _graphics_validation_problem(
                        error, current.graphic_id
                    )
                except (OSError, RuntimeError) as error:
                    raise _graphics_preview_problem(
                        503, "PREVIEW_RENDER_UNAVAILABLE", str(error),
                        field="preview.renderer",
                        suggested_patch={"action": "retry"},
                    )
                current_metadata.update({
                    "z_index": (
                        current.z_index if current.z_index is not None
                        else plan_order[current.graphic_id]
                    ),
                    "plan_order": plan_order[current.graphic_id],
                    "card_width": current_card.width,
                    "card_height": current_card.height,
                    "left": current_card.left,
                    "top": current_card.top,
                    "avoids_graphic_ids": [
                        other.graphic_id for other, _ in overlapping
                    ],
                    "url": (
                        f"/api/runs/{run_id}/graphics-preview-file/"
                        f"{child_digest}.png"
                    ),
                })
                joint_layout[current.graphic_id] = dict(current_metadata)
                if current.graphic_id == graphic_id:
                    card, metadata = current_card, current_metadata
                earlier.append((current, current_card))
            if card is None or metadata is None:
                raise _graphics_preview_problem(
                    422, "GRAPHIC_NOT_COMPILED",
                    "graphic preview could not be compiled",
                    field=f"cues.{graphic_id}",
                    suggested_patch={"action": "review_graphic_settings"},
                )
            from montagewright.measure.storage import write_json

            metadata.update({
                "card_width": card.width, "card_height": card.height,
                "left": card.left, "top": card.top,
                "joint_layout": joint_layout,
            })
            try:
                write_json(metadata_path, metadata)
            except OSError as error:
                raise _graphics_preview_problem(
                    503, "PREVIEW_CACHE_UNAVAILABLE", str(error),
                    field="preview.cache",
                    suggested_patch={"action": "retry"},
                )
            return metadata

        if metadata is None:
            lock = _graphics_preview_lock(metadata_path)
            await run_in_threadpool(lock.acquire)
            try:
                # A second request may have completed while this one waited.
                try:
                    metadata = _load_graphics_preview_cache(
                        metadata_path, preview_dir, graphic_id
                    )
                except OSError as error:
                    raise _graphics_preview_problem(
                        503, "PREVIEW_CACHE_UNAVAILABLE", str(error),
                        field="preview.cache",
                        suggested_patch={"action": "retry"},
                    )
                if metadata is None:
                    metadata = await compile_missing_preview()
            finally:
                lock.release()
        # Preview files outlive the in-memory run alias that first compiled
        # them.  A server restart may reopen the same output directory under
        # its durable run id; never return the stale alias embedded in cached
        # metadata.  The digest filename is the authority, the URL is merely
        # a request-scoped projection.
        for item in metadata.get("joint_layout", {}).values():
            filename = Path(str(item.get("url") or "")).name
            if re.fullmatch(r"[0-9a-f]{64}\.png", filename):
                item["url"] = (
                    f"/api/runs/{run_id}/graphics-preview-file/{filename}"
                )
        current_url = metadata["joint_layout"][graphic_id]["url"]
        return JSONResponse({
            "frame_width": width, "frame_height": height,
            **metadata,
            # Keep this after **metadata: old cache records also contain a
            # top-level URL and dict expansion would otherwise restore it.
            "url": current_url,
        })

    @app.get("/api/runs/{run_id}/graphics-preview-file/{filename}")
    def graphic_preview_file(run_id: str, filename: str):
        if not re.fullmatch(r"[0-9a-f]{64}\.png", filename):
            raise HTTPException(404, "no such graphics preview")
        path = _run(run_id).output / "work" / "graphics-preview" / filename
        try:
            valid = _cached_preview_png_is_valid(path)
        except OSError as error:
            raise _graphics_preview_problem(
                503, "PREVIEW_CACHE_UNAVAILABLE", str(error),
                field="preview.cache",
                suggested_patch={"action": "retry"},
            )
        if not valid:
            if not path.exists():
                raise HTTPException(404, "no such graphics preview")
            raise _graphics_preview_problem(
                503, "PREVIEW_CACHE_CORRUPT",
                "the cached preview is not a complete PNG",
                field="preview.cache",
                suggested_patch={"action": "regenerate_preview"},
            )
        # The filename is the SHA-256 content digest. It can be cached forever;
        # a changed card receives a different URL instead of mutating this one.
        return FileResponse(
            path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/api/runs/{run_id}/graphics-burned")
    def graphics_burned(run_id: str):
        run = _run(run_id)
        made = run.output / "deliverable-graphics-subtitled.mp4"
        if not made.exists():
            made = run.output / "deliverable-graphics.mp4"
        if not made.exists():
            raise HTTPException(404, "graphics have not been rendered")
        return FileResponse(
            made, media_type="video/mp4", filename=f"{run_id}-graphics.mp4"
        )

    @app.put("/api/runs/{run_id}/subtitle-track")
    async def edit_subtitle_track(run_id: str, request: Request) -> JSONResponse:
        """Keep an edited set of lines beside the ones that were derived.

        Gemini fixes most of what the recogniser mishears and not all of it,
        and the name of a product is exactly the kind of word it gets wrong.
        Edits are kept separately rather than written back over the
        transcript: the transcript is what was heard, which stays true, and
        this is what should appear on screen.
        """

        run = _run(run_id)
        sent = (await request.json()).get("lines", [])
        kept = [
            {
                "at": round(float(one.get("at", 0.0)), 3),
                "until": round(float(one.get("until", 0.0)), 3),
                "text": str(one.get("text", "")),
                "speaker": str(one.get("speaker", "")),
                "heard": str(one.get("heard", "")),
                "timing_source": str(
                    one.get("timing_source", "apple_audio_time_range")
                ),
                "timing_confidence": str(
                    one.get("timing_confidence", "unverified")
                ),
                "timing_locked": bool(one.get("timing_locked", False)),
            }
            for one in sent
            if str(one.get("text", "")).strip()
        ]
        kept.sort(key=lambda one: one["at"])
        kept = _retimed(run, kept)
        destination = run.output / "work" / "subtitles.json"
        from montagewright.measure.storage import write_json

        try:
            # Publish the new subtitle authority atomically before retiring
            # outputs that contain the old text. A failed save leaves the last
            # known-good delivery intact; a successful one can never coexist
            # with a stale subtitle/graphics composite.
            write_json(destination, kept)
            _invalidate_subtitle_delivery(run)
        except OSError as error:
            raise HTTPException(
                503, f"subtitle storage is temporarily unavailable: {error}"
            )
        # The re-timed lines go back, not just a count. The browser sent the
        # times the lines used to have; if it keeps them it will draw the
        # captions at moments the server has already moved them off.
        return JSONResponse({
            "lines": len(kept),
            "timed": [
                {
                    "at": one["at"], "until": one["until"],
                    "text": one["text"],
                    "timing_source": one.get("timing_source", ""),
                    "timing_confidence": one.get("timing_confidence", ""),
                    "timing_locked": bool(one.get("timing_locked", False)),
                }
                for one in kept
            ],
        })

    @app.post("/api/runs/{run_id}/burn-subtitles")
    def burn_subtitles(
        run_id: str, look: str = "plain", font: str = ""
    ) -> JSONResponse:
        """Put the words on the picture, using the lines as they now read.

        Costs nothing: the transcript was paid for once and the corrections
        were made by hand. Kept as its own file so the cut without them is
        still there -- burned-in text cannot be taken out again, and which
        one gets posted is not this tool's decision.
        """

        from montagewright import subtitles as typeset
        from montagewright.subtitles import NoFontHere, burn
        from montagewright.subtitles import look as looks_like

        run = _run(run_id)
        said = _subtitle_lines(run)
        if not said:
            raise HTTPException(404, "nothing was transcribed in this run")
        source = next(
            (
                run.output / name
                for name in ("deliverable.mp4", "picture.mp4")
                if (run.output / name).exists()
            ),
            None,
        )
        if source is None:
            raise HTTPException(404, "this run has no finished cut")
        aspect = (run.report() or {}).get("direction", {}).get("aspect", "9:16")
        # Set for this render only; the next one asks again.
        with SUBTITLE_FONT_LOCK:
            was, typeset.CHOSEN = typeset.CHOSEN, (font or None)
            try:
                made = burn(
                    source, said, run.output / "deliverable-subtitled.mp4",
                    aspect=aspect, work=run.output / "work" / "subs",
                    style=looks_like(look), words=_subtitle_words(run),
                )
            except NoFontHere as error:
                raise HTTPException(422, str(error))
            finally:
                typeset.CHOSEN = was
        from montagewright.measure.storage import write_json

        write_json(
            run.output / "work" / "subtitle-render.json",
            {"look": look, "font": font},
        )
        return JSONResponse({
            "file": made.name, "lines": len(said), "aspect": aspect,
            "look": look,
        })

    @app.get("/api/runs/{run_id}/burned")
    def burned(run_id: str):
        run = _run(run_id)
        made = run.output / "deliverable-subtitled.mp4"
        if not made.exists():
            raise HTTPException(404, "not burned yet")
        return FileResponse(made, media_type="video/mp4",
                            filename=f"{run_id}-subtitled.mp4")

    @app.get("/api/runs/{run_id}/subtitles")
    def subtitles(run_id: str):
        """Every transcript in the run, as one SRT against the finished cut.

        The lines are timed against their own source, so they are shifted
        onto the timeline the shots landed on -- a subtitle file that needs
        the reader to work out which take a line came from is not one.
        """

        from montagewright.subtitles import NoFontHere, as_cues
        from montagewright.transcript import to_srt

        run = _run(run_id)
        timed = _subtitle_lines(run)
        aspect = (run.report() or {}).get("direction", {}).get("aspect", "9:16")
        picture = next(
            (
                run.output / name
                for name in ("picture.mp4", "deliverable.mp4")
                if (run.output / name).exists()
            ),
            None,
        )
        width, height = (
            _video_display_size(picture) if picture is not None
            else ((1920, 1080) if aspect == "16:9" else (1080, 1920))
        )
        try:
            timed = as_cues(
                timed, aspect, width, height, words=_subtitle_words(run),
            )
        except NoFontHere:
            # Without a font there is nothing to measure against, and a long
            # cue in a file is better than no file.
            pass
        if not timed:
            raise HTTPException(404, "nothing was transcribed in this run")
        path = run.output / "subtitles.srt"
        # Several people in one cut; the file says which is which.
        path.write_text(to_srt(timed, with_speaker=True), encoding="utf-8")
        return FileResponse(
            path, media_type="text/plain", filename=f"{run_id}.srt"
        )

    @app.get("/api/runs/{run_id}/source/{which}")
    def source_clip(run_id: str, which: str):
        """The take this shot was cut from, uncropped.

        Checking that a crop followed anything means seeing what it was
        moving across. The rendered segment cannot show that -- it is the
        answer, not the working.

        Addressed by source id, falling back to the shot's position. It was
        the position alone, which meant the URL changed at every cut even
        when the next shot came out of the same file -- so the browser
        dropped a take it already had and fetched it again under a new name.
        Two shots from one take now resolve to one URL, which is also the
        only way its cache can help: at one to two seconds a shot, a reload
        per cut is a reload per second.
        """

        run = _run(run_id)
        report = run.report() or {}
        shots = report.get("selection", {}).get("shots", [])
        known = {str(shot.get("source_id", "")) for shot in shots}
        if which in known:
            source_id = which
        elif which.isdigit() and int(which) < len(shots):
            source_id = shots[int(which)]["source_id"]
        else:
            raise HTTPException(404, "no such shot")

        # The proxy, when there is one. This view exists to show where the
        # crop sat, and the proxy is the same framing at a five-hundredth of
        # the bytes -- the original is 128MB of 4K that the browser has to
        # decode in full to draw a rectangle on, and every scrub reopens it.
        proxy = run.output / "work" / "proxies" / f"{source_id}.mp4"
        if proxy.exists():
            return FileResponse(proxy, media_type="video/mp4")

        match = next(
            (
                path for path in
                list((run.output / "work" / "shots").glob("*"))
                + list(Path(run.source).glob("*"))
                if path.stem == source_id
            ),
            None,
        )
        if match is None:
            raise HTTPException(404, f"{source_id} is gone")
        return FileResponse(match, media_type="video/mp4")

    @app.get("/api/runs/{run_id}/thumb/{index}")
    def thumb(run_id: str, index: int, at: float = 0.35):
        """A frame from a shot, for recognising it by sight.

        A list of filenames is not how anybody knows which take is which.
        Kept on disk once made: pulling a frame is cheap, doing it for every
        shot on every repaint is not.
        """

        run = _run(run_id)
        made = run.output / "work" / "thumbs" / f"{index:03d}.jpg"
        if not made.exists():
            # Every shot has two files, the cut and the one with handles,
            # and a glob returns whichever it likes first. Rejecting the
            # handles file rather than passing over it lost the frame for
            # every shot whose handles happened to be listed first.
            segment = next(
                (
                    path for path in
                    sorted((run.output / "segments").glob(f"{index:03d}-*.mp4"))
                    if "handles" not in path.name
                ),
                None,
            )
            if segment is None:
                raise HTTPException(404, "no such shot")
            made.parent.mkdir(parents=True, exist_ok=True)
            length = probe_duration(segment)
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-ss", f"{max(0.0, length * at):.3f}", "-i", str(segment),
                 "-frames:v", "1", "-vf", "scale=-2:180", str(made)],
                check=False,
            )
        if not made.exists():
            raise HTTPException(404, "could not read a frame")
        return FileResponse(made, media_type="image/jpeg")

    @app.get("/api/runs/{run_id}/shot/{index}")
    def shot(run_id: str, index: int):
        """One rendered shot on its own. Faults hide in the whole cut."""

        run = _run(run_id)
        matches = sorted((run.output / "segments").glob(f"{index:03d}-*.mp4"))
        real = [p for p in matches if not p.name.endswith(".handles.mp4")]
        if not real:
            raise HTTPException(404, "no such shot")
        return FileResponse(real[0], media_type="video/mp4")

    return app


app = create_app()


def main() -> int:
    import uvicorn
    from montagewright.environment import load_project_env

    load_project_env()
    uvicorn.run(
        app,
        host=os.environ.get("MONTAGEWRIGHT_HOST", "127.0.0.1"),
        port=int(os.environ.get("MONTAGEWRIGHT_PORT", "8765")),
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
