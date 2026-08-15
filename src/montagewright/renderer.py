"""Render a compiled plan with ffmpeg.

Segments are cut individually and then concatenated, rather than assembled in
one filter graph. A graph is marginally faster and much harder to debug: when
one shot is wrong you want to open that shot, not bisect a filter chain. The
segment files are also the natural cache boundary once only part of a cut
changes between review rounds.

Every render produces two files. The deliverable is full resolution; the
preview is small enough to send somewhere. The review loop reads the preview
by design -- judging a cut from a low-resolution copy is what a director does
with a viewing link, and it keeps the reviewer's attention on the edit rather
than on grain.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from dataclasses import replace

from montagewright.executor import RenderPlan, Segment, allocate_timeline_frames
from montagewright.reframe import ffmpeg_crop_filters

# Short-form platforms normalise to roughly this; matching it here means the
# cut sounds the same locally as it will after upload.
TARGET_LUFS = -14.0
# Headroom below full scale. Lossy re-encoding on the way to a platform adds
# its own overshoot, so a master that already touches 0 dBFS clips there.
TRUE_PEAK_CEILING_DB = -1.5
# alimiter takes a linear ceiling, not decibels. Passing "-1.5dB" is accepted
# and silently clamped, which is how a cut measured at +0.017 dBFS came out of
# a filter chain that asked for -1.5.
TRUE_PEAK_CEILING_LINEAR = 10 ** (TRUE_PEAK_CEILING_DB / 20)
PREVIEW_HEIGHT = 640

# Level the voice before anything is laid under it. A street interview runs
# from a shouted answer to a mumbled one -- fourteen decibels apart in one
# take here -- so a bed placed under the average sits comfortably under the
# loud speaker and nearly on top of the quiet one. speechnorm is built for
# this: it lifts quiet speech without pumping the gaps the way a compressor
# aimed at music would.
VOICE_LEVELLER = "speechnorm=e=12.5:r=0.0001:l=1"

# How far under the voice the bed sits. Measured against the voice, not
# subtracted from the music: a mastered track reduced by a fixed amount lands
# wherever that track happened to be mastered, and a street interview averages
# around -22 dBFS, which is exactly where "the music minus 12" put the bed --
# the same level as the speech it was supposed to be under.
BED_BELOW_VOICE_DB = 14.0
# How the bed gets out of the way. Attack short enough to be down before the
# first syllable lands, release long enough that it does not pump between
# words -- a bed that comes back up inside a sentence is more distracting
# than one that never moved.
DUCK_THRESHOLD = 0.03
DUCK_RATIO = 8
DUCK_ATTACK_MS = 20
DUCK_RELEASE_MS = 600

# Extra material kept either side of each cut, never shown. A transition needs
# two shots to overlap, and an exact cut leaves nothing to overlap with; a
# nudge of a quarter of a second later needs frames that were never rendered.
# Editors call these handles and always cut them.
HANDLE_SECONDS = 0.5


class RenderError(RuntimeError):
    """ffmpeg refused, and the command plus its stderr are attached."""


@dataclass(frozen=True)
class Handles:
    """How much spare material a segment carries, and where."""

    head_seconds: float
    tail_seconds: float


@dataclass(frozen=True)
class RenderResult:
    deliverable: Path
    preview: Path
    segment_paths: tuple[Path, ...]
    duration_seconds: float

    def sizes_mb(self) -> dict[str, float]:
        return {
            "deliverable": self.deliverable.stat().st_size / 1_048_576,
            "preview": self.preview.stat().st_size / 1_048_576,
        }


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    hardware_unavailable = (
        "h264_videotoolbox" in command
        and any(
            marker in completed.stderr
            for marker in (
                "Cannot create compression session",
                "hardware encoder may be busy",
                "Could not open encoder before EOF",
            )
        )
    )
    if completed.returncode != 0 and hardware_unavailable:
        # Listing an encoder only proves that this ffmpeg build knows its
        # name.  VideoToolbox can still refuse a session at render time when
        # the hardware pool is busy or a particular frame shape is not
        # supported.  Retry the identical edit in software; do not hide
        # unrelated filter, media, or filesystem failures behind a fallback.
        software = [
            "libx264" if token == "h264_videotoolbox" else token
            for token in command
        ]
        completed = subprocess.run(
            software, capture_output=True, text=True, check=False
        )
        command = software
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.strip().splitlines()[-15:])
        raise RenderError(
            f"ffmpeg failed ({completed.returncode})\n"
            f"  {' '.join(command[:12])} ...\n{tail}"
        )
    return completed


def _encoder(preferred: str, fallback: str) -> str:
    """Prefer the hardware encoder, but do not fail without it.

    VideoToolbox is the fast path on this hardware and absent everywhere else.
    A renderer that only works on one laptop is not much of a renderer.
    """

    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return preferred if preferred in probe.stdout else fallback


def _level(path: Path) -> float:
    """Mean level of a file's audio, in dBFS.

    Both sides have to be measured for "under the voice" to mean anything.
    A track mastered loud and a field recording of someone talking in traffic
    are twenty decibels apart before anything is decided.
    """

    completed = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", str(path),
            "-af", "volumedetect", "-f", "null", "-",
        ],
        capture_output=True, text=True, check=False,
    )
    for line in completed.stderr.splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].split("dB")[0])
    return -20.0


def probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RenderError(f"ffprobe could not read {path}")
    return float(json.loads(completed.stdout)["format"]["duration"])


def _render_segment(
    segment: Segment, destination: Path, *, video_encoder: str,
    output_size: tuple[int, int] = (1080, 1920),
    output_fps: int = 30,
    output_frames: int | None = None,
) -> tuple[Path, "Handles"]:
    """Cut one shot, cropping if the plan asked for it."""

    source = segment.source
    first = segment.usable_from_seconds
    last = segment.usable_to_seconds or source.duration_seconds
    head = min(HANDLE_SECONDS, max(0.0, segment.in_seconds - first))
    tail = min(HANDLE_SECONDS, max(0.0, last - segment.out_seconds))

    # Establish the delivery clock before any frame-evaluated crop motion.
    # In particular, perspective's `on` counter follows the frames it sees;
    # putting CFR after it made identical authored moves run at different
    # speeds for 24, 30 and 60 fps sources.
    filters: list[str] = [f"fps=fps={output_fps}:round=near"]
    if segment.crop_path is not None and not segment.crop_path.is_static:
        # A following camera. The x expression is evaluated per frame, so the
        # motion lives in the same filter as the crop rather than in a
        # separate command stream.
        filters.extend(ffmpeg_crop_filters(
            segment.crop_path, source.width, source.height, output_size,
            output_fps=output_fps,
        ))
    elif segment.crop is not None:
        x, y, width, height = segment.crop.to_pixels(source.width, source.height)
        filters.append(f"crop={width}:{height}:{x}:{y}")
        filters.append(f"scale={output_size[0]}:{output_size[1]}")
    # Source cameras frequently carry a non-square display aspect ratio.  A
    # 1080x1920 render that inherits a 256:81 SAR is advertised to players as
    # 16:9 even though its pixels are portrait.  The delivery canvas always
    # uses square pixels, so make that metadata explicit before encoding.
    filters.append("setsar=1")
    # Mixed 23.976/25/29.97/30/60 footage is now already on one explicit CFR
    # editing timeline. Editorial times remain seconds; this is the single
    # boundary where they are quantised to deliverable frames.
    handle_filters = [f"fps=fps={output_fps}:round=near"]
    if segment.crop_path is not None and not segment.crop_path.is_static:
        handle_filters.extend(ffmpeg_crop_filters(
            segment.crop_path, source.width, source.height, output_size,
            output_fps=output_fps, clock_offset_seconds=head,
        ))
    elif segment.crop is not None:
        x, y, width, height = segment.crop.to_pixels(source.width, source.height)
        handle_filters.extend([
            f"crop={width}:{height}:{x}:{y}",
            f"scale={output_size[0]}:{output_size[1]}",
        ])
    handle_filters.append("setsar=1")
    if output_frames is not None:
        filters.extend([
            "tpad=stop_mode=clone:stop_duration=1",
            f"trim=end_frame={output_frames}",
            "setpts=PTS-STARTPTS",
        ])

    # The delivered segment is cut exactly. Handles are written alongside it
    # as their own file, so a transition or a nudge has material without the
    # timeline paying for it -- the concat stays frame-exact because every
    # segment is already the length it is meant to be.
    audio = []
    if segment.audio_role == "discard":
        # Keep a silent audio stream rather than dropping it. Every rendered
        # segment must have the same stream layout for frame-exact concat.
        audio = ["-af", "volume=0"]
    elif abs(segment.gain_db) > 0.01:
        audio = ["-af", f"volume={segment.gain_db:.2f}dB"]

    # Handles reach back to the start of the file and forward to its end,
    # which is the wrong boundary: half a second before a take is usually the
    # camera still being aimed, and half a second after it is often somebody
    # saying "again". A handle exists to be pulled, so one that opens onto a
    # reset is worse than none.
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        # Seeking before -i decodes from the preceding keyframe, which is both
        # faster and frame-accurate here because the output is re-encoded.
        "-ss", f"{segment.in_seconds:.6f}",
        "-to", f"{segment.out_seconds:.6f}",
        "-i", str(segment.source.path),
    ]
    if filters:
        command += ["-vf", ",".join(filters)]
    command += audio
    command += [
        "-c:v", video_encoder, "-b:v", "12M",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p", "-r", str(output_fps), "-fps_mode", "cfr",
    ]
    if output_frames is not None:
        command += [
            "-frames:v", str(output_frames),
            "-t", f"{output_frames / output_fps:.9f}",
        ]
    command.append(str(destination))
    _run(command)

    if head > 0.0 or tail > 0.0:
        spare = destination.with_name(f"{destination.stem}.handles.mp4")
        _run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{segment.in_seconds - head:.6f}",
                "-to", f"{segment.out_seconds + tail:.6f}",
                "-i", str(segment.source.path),
            ]
            + (["-vf", ",".join(handle_filters)] if handle_filters else [])
            + audio
            + [
                "-c:v", video_encoder, "-b:v", "12M",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p", "-r", str(output_fps),
                "-fps_mode", "cfr", str(spare),
            ]
        )
    return destination, Handles(head_seconds=head, tail_seconds=tail)


def _concat(
    segment_paths: list[tuple[Path, Handles, float]],
    destination: Path,
    work_dir: Path,
) -> Path:
    """Join the segments without touching the streams again.

    The segments already share a codec, pixel format, and sample rate because
    this module wrote all of them, so a stream copy is exact. Re-encoding here
    would be a second generation loss for no gain.

    Handles are trimmed at render time rather than here. Asking the concat
    demuxer for inpoint/outpoint on a copied stream lands on the nearest
    keyframe, which drifted a two-shot test by 124ms -- and every cut in this
    pipeline is placed against a measured beat, so drift that accumulates
    across a timeline is not a rounding detail, it is the alignment gone.
    """

    listing = work_dir / "concat.txt"
    listing.write_text(
        "".join(f"file '{path.resolve()}'\n" for path, _, _ in segment_paths),
        encoding="utf-8",
    )
    _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c", "copy", str(destination),
        ]
    )
    return destination


# Long enough to read as an ending rather than a glitch, short enough not to
# eat the last shot. A bed that simply stops mid-phrase is the most audible
# thing in an otherwise finished cut.
MUSIC_FADE_SECONDS = 1.5


# Long enough to hide a join, short enough that neither side is smeared. A
# butt splice between two pieces of music clicks even on a phrase line,
# because the waveform does not happen to be at zero.
MUSIC_JOIN_SECONDS = 0.12


def _spliced(
    spans: "list[tuple[float, float]]", duration: float
) -> tuple[str, str]:
    """Play these pieces of the track in order, joined and trimmed to fit.

    A two-minute piece cut to thirty seconds keeps its shape this way: the
    opening, the part with the energy, the ending, with the middle taken out.
    The alternative is half a piece that stops.

    Each piece is taken whole and they are crossfaded into each other, which
    is the join an editor makes -- a butt splice clicks even on a phrase line,
    since the waveform is not at zero just because the bar is. What comes out
    is then trimmed to the picture, so a set of spans that overshoots is
    shortened rather than refused.
    """

    parts = []
    for index, (began, ended) in enumerate(spans):
        parts.append(
            f"[1:a]atrim={began:.6f}:{ended:.6f},asetpts=PTS-STARTPTS[m{index}];"
        )
    chain = "".join(parts)
    current = "[m0]"
    for index in range(1, len(spans)):
        nxt = f"[j{index}]"
        chain += (
            f"{current}[m{index}]acrossfade="
            f"d={MUSIC_JOIN_SECONDS}:c1=tri:c2=tri{nxt};"
        )
        current = nxt
    # Trailing semicolon belongs to the caller's chain, and the label has to
    # come off so this reads as one filter run like the simple case does.
    return chain, f"{current}atrim=0:{duration:.6f},asetpts=PTS-STARTPTS"


def _mux_music(
    picture: Path,
    music: Path,
    destination: Path,
    *,
    video_encoder: str,
    keep_voice: bool = False,
    under_speech: str = "duck",
    music_from_seconds: float = 0.0,
    music_spans: "list[tuple[float, float]] | None" = None,
    fade_out_seconds: float = MUSIC_FADE_SECONDS,
) -> Path:
    """Lay a music bed under the cut and normalise the result.

    The music is trimmed to the picture, never the other way round: stretching
    a track to fit a cut is audible, and holding the picture to fit the track
    means padding it with something nobody chose.

    When the picture carries speech, the music goes under it rather than over
    it. This used to be unconditional -- `-map 0:v:0 -map [music]` threw the
    source audio away, which is right for b-roll and destroys an interview,
    where what was said is the whole content. A fixed lower level is not the
    answer either: quiet enough never to bury a sentence is too quiet to be
    doing anything in the gaps. The voice drives the compressor, so the bed
    steps back for each line and comes up between them.
    """

    duration = probe_duration(picture)
    # From wherever the rhythm pass pointed, not from zero. Taking the first
    # thirty seconds of a two-minute track means scoring the film with the
    # intro, which is written to have no energy yet.
    start = max(0.0, float(music_from_seconds))
    if start > 0.0:
        spare = max(0.0, (probe_duration(music) or 0.0) - duration)
        start = min(start, spare)
    # The bed as one chain ending in [bed], because a spliced one has to
    # take [1:a] several times and cannot be written as a suffix.
    if music_spans:
        before, tail = _spliced(music_spans, duration)
    else:
        before = ""
        tail = f"[1:a]atrim={start:.6f}:{start + duration:.6f},asetpts=PTS-STARTPTS"
    # And it ends rather than stopping. A bed cut off mid-phrase is the most
    # audible thing in a finished cut; a second and a half of fade is what
    # makes it sound like an ending.
    fade = (
        f",afade=t=out:st={max(0.0, duration - fade_out_seconds):.3f}"
        f":d={min(fade_out_seconds, duration):.3f}"
        if fade_out_seconds > 0.0 else ""
    )
    bed_gain = (
        _level(picture) - _level(music) - BED_BELOW_VOICE_DB
        if keep_voice
        else 0.0
    )
    if keep_voice and under_speech == "bed":
        # Steady, all the way through. Ducking a film that is speech from end
        # to end means the bed climbs into every breath and gets pushed down
        # again by the next line -- busier than simply sitting behind it. Which
        # of the two a cut wants is an editorial call, and it is made by the
        # layer that watched the material rather than by a compressor.
        chain = (
            f"{before}{tail}{fade},volume={bed_gain:.2f}dB[bed];"
            f"[0:a]{VOICE_LEVELLER}[voice];"
            "[voice][bed]amix=inputs=2:duration=first:normalize=0,"
            f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK_CEILING_DB}:LRA=11,"
            f"alimiter=limit={TRUE_PEAK_CEILING_LINEAR:.6f}:level=disabled"
            "[out]"
        )
    elif keep_voice:
        chain = (
            f"{before}{tail}{fade},volume={bed_gain:.2f}dB[bed];"
            # The voice is the sidechain trigger, not part of the output of
            # this branch -- asplit because one copy steers the compressor
            # and the other is what anyone actually hears.
            f"[0:a]{VOICE_LEVELLER},asplit=2[voice][key];"
            f"[bed][key]sidechaincompress="
            f"threshold={DUCK_THRESHOLD}:ratio={DUCK_RATIO}:"
            f"attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}[ducked];"
            "[voice][ducked]amix=inputs=2:duration=first:normalize=0,"
            f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK_CEILING_DB}:LRA=11,"
            f"alimiter=limit={TRUE_PEAK_CEILING_LINEAR:.6f}:level=disabled"
            "[out]"
        )
    else:
        chain = (
            f"{before}{tail}{fade},"
            f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK_CEILING_DB}:LRA=11,"
            # loudnorm in one pass predicts its true peak rather than
            # measuring it, and overshoots often enough to matter: this cut
            # came back at +0.017 dBFS against a -1.5 request. A limiter after
            # it holds the ceiling for real, which is what stops a platform's
            # own re-encode from clipping what we sent.
            f"alimiter=limit={TRUE_PEAK_CEILING_LINEAR:.6f}:level=disabled"
            "[out]"
        )
    # Keep the bed as it ended up, not as it arrived. Whether the music
    # steps back for the voice is the one thing about a mix somebody wants
    # to check before posting, and the only honest way to show it is to draw
    # the track that was actually laid under the words. Inferring it from
    # where the speech is would be a picture of the intention.
    # A filter label is consumed once, so the bed cannot simply be named
    # twice -- it is split, one copy into the mix and one into a file.
    bed_out = []
    if "[ducked];" in chain:
        chain = chain.replace(
            "[ducked];", "[ducked];[ducked]asplit=2[duck_mix][bed_only];", 1
        ).replace("[voice][ducked]amix", "[voice][duck_mix]amix", 1)
        bed_out = ["-map", "[bed_only]"]
    elif "[bed];" in chain and "[voice][bed]amix" in chain:
        chain = chain.replace(
            "[bed];", "[bed];[bed]asplit=2[bed_mix][bed_only];", 1
        ).replace("[voice][bed]amix", "[voice][bed_mix]amix", 1)
        bed_out = ["-map", "[bed_only]"]

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(picture), "-i", str(music),
        "-filter_complex", chain,
        "-map", "0:v:0", "-map", "[out]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(destination),
    ]
    if bed_out:
        command += bed_out + [
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            str(destination.parent / "bed-as-laid.m4a"),
        ]
    _run(command)
    return destination


def _preview(source: Path, destination: Path, *, video_encoder: str) -> Path:
    """A copy small enough to send over a slow link."""

    _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vf", f"scale=-2:{PREVIEW_HEIGHT},setsar=1",
            "-c:v", video_encoder, "-b:v", "1200k",
            "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            str(destination),
        ]
    )
    return destination


def _lay_audio_assignments(
    picture: Path, assignments, destination: Path, *, output_fps: int,
) -> Path:
    """Lay continuous source audio independently of picture segment cuts."""

    duration = probe_duration(picture)
    ordered = sorted(assignments, key=lambda one: one.timeline_in_seconds)
    previous_end = 0.0
    for assignment in ordered:
        begins = assignment.timeline_in_seconds
        ends = begins + assignment.duration_seconds
        if begins < -1e-6 or ends > duration + 0.05:
            raise RenderError(
                f"audio {assignment.audio_id} falls outside the picture "
                f"timeline ({begins:.3f}–{ends:.3f}s of {duration:.3f}s)"
            )
        if assignment.role == "narrative" and begins < previous_end - 1e-6:
            raise RenderError(
                f"narrative audio {assignment.audio_id} overlaps another "
                "narrative assignment"
            )
        if assignment.role == "narrative":
            previous_end = ends

    inputs = ["-i", str(picture)]
    filters = [
        f"anullsrc=r=48000:cl=stereo,atrim=duration={duration:.6f}[silence]"
    ]
    labels = ["[silence]"]
    for index, assignment in enumerate(ordered, start=1):
        inputs += ["-i", str(assignment.source.path)]
        delay_samples = max(
            0, round(assignment.timeline_start_frame * 48000 / output_fps)
        )
        label = f"a{index}"
        filters.append(
            f"[{index}:a]atrim={assignment.in_seconds:.6f}:"
            f"{assignment.out_seconds:.6f},asetpts=PTS-STARTPTS,"
            f"atrim=duration={assignment.duration_seconds:.6f},"
            "aresample=48000,aformat=channel_layouts=stereo,"
            f"volume={assignment.gain_db:.2f}dB,"
            f"adelay={delay_samples}S:all=1[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=first:normalize=0:"
        "dropout_transition=0,"
        f"atrim=duration={duration:.6f},"
        f"alimiter=limit={TRUE_PEAK_CEILING_LINEAR:.6f}:level=disabled[out]"
    )
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs, "-filter_complex", ";".join(filters),
        "-map", "0:v:0", "-map", "[out]", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.6f}",
        str(destination),
    ])
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(destination), "-vn", "-c:a", "aac", "-b:a", "192k",
        str(destination.parent / "voice-as-laid.m4a"),
    ])
    return destination


def render(
    plan: RenderPlan,
    output_dir: Path,
    *,
    music: Path | None = None,
    keep_segments: bool = True,
    keep_voice: bool = False,
    under_speech: str = "duck",
) -> RenderResult:
    """Render a plan to a deliverable and a preview.

    Raises only when ffmpeg itself fails. Whether the cut is any good is the
    review loop's question, and it needs a finished file to answer it.
    """

    if not plan.segments:
        raise RenderError("nothing to render: the plan has no segments")

    output_dir = output_dir.expanduser().resolve()
    segment_dir = output_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)

    video_encoder = _encoder("h264_videotoolbox", "libx264")

    segment_paths: list[tuple[Path, "Handles", float]] = []
    boundaries = allocate_timeline_frames(
        [segment.duration_seconds for segment in plan.segments],
        plan.output_fps,
    )
    for index, (segment, (start_frame, end_frame)) in enumerate(
        zip(plan.segments, boundaries)
    ):
        segment_frames = end_frame - start_frame
        if segment_frames < 1:
            raise RenderError(
                f"{segment.clip_id} is shorter than one {plan.output_fps} fps "
                "timeline frame"
            )
        destination = segment_dir / f"{index:03d}-{segment.clip_id}.mp4"
        picture_segment = (
            replace(segment, audio_role="discard")
            if plan.audio_track_explicit else segment
        )
        rendered, handles = _render_segment(
            picture_segment, destination, video_encoder=video_encoder,
            output_size=plan.output_size,
            output_fps=plan.output_fps,
            output_frames=segment_frames,
        )
        segment_paths.append((rendered, handles, segment.duration_seconds))

    picture = _concat(segment_paths, output_dir / "picture.mp4", output_dir)
    mix_picture = picture
    if plan.audio_assignments:
        mix_picture = _lay_audio_assignments(
            picture, plan.audio_assignments, output_dir / "picture-with-audio.mp4",
            output_fps=plan.output_fps,
        )
    kept_paths = [path for path, _, _ in segment_paths]

    deliverable = output_dir / "deliverable.mp4"
    if music is not None:
        _mux_music(
            mix_picture, music, deliverable,
            video_encoder=video_encoder,
            # An explicit laid track is authoritative. A caller's legacy
            # keep_voice switch must never discard audio the EDL assigned.
            keep_voice=keep_voice or bool(plan.audio_assignments),
            under_speech=under_speech,
            music_from_seconds=plan.music_from_seconds,
            music_spans=plan.music_spans or None,
        )
    else:
        shutil.copyfile(mix_picture, deliverable)

    preview = _preview(
        deliverable, output_dir / "preview.mp4", video_encoder=video_encoder
    )

    if not keep_segments:
        shutil.rmtree(segment_dir, ignore_errors=True)
        kept_paths = []

    return RenderResult(
        deliverable=deliverable,
        preview=preview,
        segment_paths=tuple(kept_paths),
        duration_seconds=probe_duration(deliverable),
    )
