import json
import subprocess
from pathlib import Path

from montagewright.executor import AudioAssignment, RenderPlan, Segment, Source
from montagewright.renderer import render


def _colour_clip(path: Path, *, fps: int, colour: str) -> None:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c={colour}:s=320x180:d=0.6:r={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True)


def _silent_colour_clip(path: Path, *, seconds: float, colour: str) -> None:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i",
        f"color=c={colour}:s=160x90:d={seconds}:r=30",
        "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={seconds}",
        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(path),
    ], check=True)


def _pcm_peak(path: Path, start: float, seconds: float = .15) -> int:
    from array import array

    raw = subprocess.check_output([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-t", str(seconds), "-i", str(path),
        "-vn", "-f", "s16le", "-ac", "1", "-ar", "48000", "-",
    ])
    samples = array("h")
    samples.frombytes(raw)
    return max((abs(one) for one in samples), default=0)


def test_one_audio_assignment_runs_continuously_across_three_picture_cuts(
    tmp_path: Path, monkeypatch,
) -> None:
    """A B-roll cut changes pictures, never the sentence underneath it."""

    import montagewright.renderer as renderer

    pictures = []
    for index, colour in enumerate(("red", "green", "blue")):
        path = tmp_path / f"picture-{index}.mp4"
        _silent_colour_clip(path, seconds=1, colour=colour)
        pictures.append(Source(f"p{index}", path, 1, 160, 90))
    voice_path = tmp_path / "voice.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=black:s=160x90:d=2:r=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(voice_path),
    ], check=True)
    voice = Source("voice", voice_path, 2, 160, 90)
    assignment = AudioAssignment(
        audio_id="a00", source=voice, in_seconds=0, out_seconds=2,
        timeline_in_seconds=.5, timeline_start_frame=15, frame_count=60,
        role="narrative", completion="complete_thought",
    )
    plan = RenderPlan(
        project_id="jl-cut", output_size=(160, 90), output_fps=30,
        segments=[
            Segment(f"k{index:02d}", source, 0, 1)
            for index, source in enumerate(pictures)
        ],
        audio_assignments=[assignment],
        audio_track_explicit=True,
    )
    monkeypatch.setattr(renderer, "_encoder", lambda *_: "libx264")

    made = render(plan, tmp_path / "out")

    # Silence before/after, one uninterrupted tone through both picture cuts.
    assert _pcm_peak(made.deliverable, .1) < 200
    assert _pcm_peak(made.deliverable, .75) > 1000
    assert _pcm_peak(made.deliverable, 1.25) > 1000
    assert _pcm_peak(made.deliverable, 2.15) > 1000
    assert _pcm_peak(made.deliverable, 2.75) < 200
    assert (tmp_path / "out" / "voice-as-laid.m4a").exists()


def test_subtitles_follow_the_audio_assignment_not_picture_boundaries() -> None:
    from types import SimpleNamespace
    from montagewright.transcript import against_audio_assignments

    assignment = SimpleNamespace(
        role="narrative", source=SimpleNamespace(source_id="voice"),
        in_seconds=4.0, out_seconds=6.0, duration_seconds=2.0,
        timeline_in_seconds=1.5,
    )
    cards = {"voice": {"lines": [{
        "text": "這一句跨過三個畫面",
        "starts_seconds": 4.0, "ends_seconds": 6.0,
    }]}}

    lines = against_audio_assignments([assignment], cards)

    assert len(lines) == 1
    assert lines[0].text == "這一句跨過三個畫面"
    assert lines[0].starts_seconds == 1.5
    assert lines[0].ends_seconds == 3.5


def test_nle_exports_use_the_laid_voice_track_without_source_audio_leakage(
    tmp_path: Path,
) -> None:
    from montagewright.timeline import to_fcpxml, to_xmeml

    picture = tmp_path / "picture.mp4"
    voice = tmp_path / "voice-as-laid.m4a"
    picture.touch()
    voice.touch()
    source = Source("picture", picture, 2, 160, 90)
    plan = RenderPlan(
        project_id="nle-voice", output_size=(160, 90), output_fps=30,
        segments=[Segment("k00", source, 0, 2)],
        audio_track_explicit=True,
    )

    premiere = to_xmeml(
        plan, {}, name="voice", width=160, height=90, voice=voice
    )
    finalcut = to_fcpxml(
        plan, {}, name="voice", width=160, height=90, voice=voice
    )

    assert voice.resolve().as_uri() in premiere
    assert voice.resolve().as_uri() in finalcut
    assert 'id="voice-laid"' in premiere
    assert 'audioRole="dialogue"' in finalcut
    # With an authoritative laid dialogue stem, picture assets are video-only.
    assert 'name="picture" start="0s" hasVideo="1" hasAudio="0"' in finalcut


def test_audio_assignment_uses_the_same_master_frame_clock_as_picture() -> None:
    from montagewright.executor import plan_render
    from montagewright.schema import AudioClip, Clip, EDL

    source = Source("A", Path("A.mp4"), 20, 160, 90)
    edl = EDL(
        project_id="frame-clock",
        clips=[
            Clip(
                clip_id="k00", source_id="A",
                approx_in_seconds=0, approx_out_seconds=.515,
                in_looks_like="first", energy_intent="medium",
            ),
            Clip(
                clip_id="k01", source_id="A",
                approx_in_seconds=1, approx_out_seconds=2,
                in_looks_like="second", energy_intent="medium",
            ),
        ],
        audio_clips=[AudioClip(
            audio_id="a00", source_id="A", in_seconds=4, out_seconds=4.8,
            starts_at_clip_id="k01", offset_seconds=.1,
            role="narrative", completion="complete_thought",
        )],
    )

    plan = plan_render(edl, {"A": source}, output_fps=30)

    # .515s is cumulatively allocated to frame 15, then .1s adds 3 frames.
    assert plan.audio_assignments[0].timeline_start_frame == 18
    assert plan.audio_assignments[0].timeline_in_seconds == .6


def test_web_current_timeline_v2_round_trips_independent_audio(
    tmp_path: Path,
) -> None:
    import json
    from montagewright.webapp import Run, _current_timeline

    run = Run("audio-v2", tmp_path / "run")
    (run.output / "work").mkdir(parents=True)
    (run.output / "report.json").write_text(json.dumps({
        "selection": {"shots": [{"source_id": "A"}, {"source_id": "B"}]},
    }), encoding="utf-8")
    assignment = {
        "audio_id": "a00", "source_id": "voice",
        "in_seconds": 4.0, "out_seconds": 5.0,
        "timeline_start_frame": 12, "frame_count": 30,
        "role": "narrative", "completion": "complete_thought",
        "gain_db": 0.0, "why": "one thought across the cut",
    }
    (run.output / "work" / "current-timeline.json").write_text(json.dumps({
        "version": "montagewright-current-timeline-v2", "revision": 2,
        "output_fps": 30, "output_size": [160, 90],
        "shots": [
            {"selection_index": 0, "in_seconds": 0, "seconds": 1},
            {"selection_index": 1, "in_seconds": 0, "seconds": 1},
        ],
        "audio_assignments": [assignment],
    }), encoding="utf-8")

    current = _current_timeline(run)

    assert current["revision"] == 2
    assert current["audio_assignments"] == [assignment]
    assert [one["start_frame"] for one in current["shots"]] == [0, 30]


def test_audio_assignment_cannot_run_past_the_picture() -> None:
    import pytest
    from montagewright.executor import plan_render
    from montagewright.schema import AudioClip, Clip, EDL

    source = Source("A", Path("A.mp4"), 20, 160, 90)
    edl = EDL(
        project_id="too-long",
        clips=[Clip(
            clip_id="k00", source_id="A", approx_in_seconds=0,
            approx_out_seconds=1, in_looks_like="picture",
            energy_intent="medium",
        )],
        audio_clips=[AudioClip(
            audio_id="a00", source_id="A", in_seconds=4, out_seconds=6,
            starts_at_clip_id="k00", role="narrative",
            completion="complete_thought",
        )],
    )

    with pytest.raises(ValueError, match="outside"):
        plan_render(edl, {"A": source}, output_fps=30)


def test_detached_voice_does_not_drop_a_later_sync_sound() -> None:
    from montagewright.executor import plan_render
    from montagewright.schema import AudioClip, Clip, EDL

    sources = {
        name: Source(name, Path(f"{name}.mp4"), 10, 160, 90)
        for name in ("voice", "action")
    }
    edl = EDL(
        project_id="mixed-audio",
        clips=[
            Clip(
                clip_id="k00", source_id="voice", approx_in_seconds=0,
                approx_out_seconds=1, in_looks_like="b-roll",
                energy_intent="medium", audio_role="discard",
            ),
            Clip(
                clip_id="k01", source_id="action", approx_in_seconds=2,
                approx_out_seconds=3, in_looks_like="open the box",
                energy_intent="medium", audio_role="sync_action",
                audio_completion="complete_action_sound",
            ),
        ],
        audio_clips=[AudioClip(
            audio_id="a00", source_id="voice", in_seconds=4,
            out_seconds=5, starts_at_clip_id="k00", role="narrative",
            completion="complete_thought",
        )],
    )

    plan = plan_render(edl, sources, output_fps=30)

    assert plan.audio_track_explicit is True
    assert [(one.role, one.source.source_id) for one in plan.audio_assignments] == [
        ("narrative", "voice"), ("sync_action", "action")
    ]


def test_mixed_source_fps_becomes_one_cfr_timeline(
    tmp_path: Path, monkeypatch,
):
    import montagewright.renderer as renderer

    first = tmp_path / "24.mp4"
    second = tmp_path / "60.mp4"
    _colour_clip(first, fps=24, colour="red")
    _colour_clip(second, fps=60, colour="blue")
    sources = [
        Source("a", first, .6, 320, 180),
        Source("b", second, .6, 320, 180),
    ]
    plan = RenderPlan(
        project_id="mixed-fps", output_size=(320, 180), output_fps=30,
        segments=[
            Segment("k00", sources[0], 0, .5),
            Segment("k01", sources[1], 0, .5),
        ],
    )
    monkeypatch.setattr(renderer, "_encoder", lambda *_: "libx264")

    made = render(plan, tmp_path / "out")

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames",
        "-of", "json", str(made.deliverable),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["avg_frame_rate"] == "30/1"
    assert stream["r_frame_rate"] == "30/1"
    assert int(stream["nb_frames"]) == 30


def test_fractional_shots_share_one_global_frame_rounding(
    tmp_path: Path, monkeypatch,
):
    import montagewright.renderer as renderer

    source_path = tmp_path / "source.mp4"
    _colour_clip(source_path, fps=60, colour="green")
    source = Source("a", source_path, .6, 320, 180)
    plan = RenderPlan(
        project_id="global-frame-allocation", output_size=(320, 180),
        output_fps=30,
        segments=[
            Segment(f"k{index:02d}", source, 0, .515)
            for index in range(10)
        ],
    )
    monkeypatch.setattr(renderer, "_encoder", lambda *_: "libx264")

    made = render(plan, tmp_path / "out")

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_frames",
        "-of", "json", str(made.deliverable),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["avg_frame_rate"] == "30/1"
    # 5.15 seconds is quantised once to the nearest timeline frame. Rounding
    # every .515-second shot separately would incorrectly produce 160 frames.
    assert int(stream["nb_frames"]) == 155


def test_dynamic_crop_width_really_pushes_in_at_same_speed_for_mixed_fps(
    tmp_path: Path, monkeypatch,
):
    """Verify zoom pixels and timing agree for 24/30/60 fps sources."""

    from PIL import Image
    import montagewright.renderer as renderer
    from montagewright.executor import CropBox
    from montagewright.reframe import CropPath, Keyframe

    path = CropPath([
        Keyframe(0, CropBox(.3418, 0, .3164, 1)),
        Keyframe(2, CropBox(.39585, .1708, .2083, .6584)),
    ])
    monkeypatch.setattr(renderer, "_encoder", lambda *_: "libx264")

    series = []
    for source_fps in (24, 30, 60):
        source_path = tmp_path / f"target-{source_fps}.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i",
            f"color=black:s=320x180:d=2:r={source_fps}",
            "-vf", "drawbox=x=135:y=65:w=50:h=50:color=white:t=fill",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source_path),
        ], check=True)
        source = Source(f"target-{source_fps}", source_path, 2, 320, 180)
        plan = RenderPlan(
            project_id=f"real-push-{source_fps}",
            output_size=(180, 320), output_fps=30,
            segments=[Segment(
                "k00", source, 0, 2, crop=path.keyframes[0].crop,
                crop_path=path,
            )],
        )
        made = render(plan, tmp_path / f"push-{source_fps}")

        widths = []
        for frame in (2, 15, 30, 45, 57):
            still = tmp_path / f"fps-{source_fps}-frame-{frame}.png"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(made.deliverable),
                "-vf", f"select='eq(n,{frame})'", "-frames:v", "1",
                str(still),
            ], check=True)
            image = Image.open(still).convert("L")
            ink = [
                (x, y)
                for y in range(image.height)
                for x in range(image.width)
                if image.getpixel((x, y)) > 200
            ]
            widths.append(
                max(x for x, _ in ink) - min(x for x, _ in ink) + 1
            )
        series.append(widths)

    assert series[1][-1] > series[1][0] * 1.4
    for widths in series[1:]:
        assert widths == series[0]


def test_a_sped_up_segment_lands_its_screen_frames_not_its_source_frames(
    tmp_path: Path, monkeypatch,
):
    import montagewright.renderer as renderer

    # Two seconds of source, read whole, but played at twice speed: the
    # timeline was allocated on screen seconds, so it must deliver one second
    # -- thirty frames at 30fps -- not the two seconds the window spans.
    source_path = tmp_path / "src.mp4"
    _silent_colour_clip(source_path, seconds=2.0, colour="orange")
    source = Source("a", source_path, 2.0, 160, 90)
    plan = RenderPlan(
        project_id="sped", output_size=(160, 90), output_fps=30,
        segments=[Segment("k00", source, 0.0, 2.0, speed_ratio=2.0)],
    )
    monkeypatch.setattr(renderer, "_encoder", lambda *_: "libx264")

    made = render(plan, tmp_path / "out")

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_frames",
        "-of", "json", str(made.deliverable),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["avg_frame_rate"] == "30/1"
    assert int(stream["nb_frames"]) == 30


def test_a_slowed_segment_stretches_its_source_window_on_the_timeline(
    tmp_path: Path, monkeypatch,
):
    import montagewright.renderer as renderer

    # One second of source at half speed occupies two seconds of screen: the
    # renderer fills the extra frames rather than running short.
    source_path = tmp_path / "src.mp4"
    _silent_colour_clip(source_path, seconds=1.2, colour="teal")
    source = Source("a", source_path, 1.2, 160, 90)
    plan = RenderPlan(
        project_id="slowed", output_size=(160, 90), output_fps=30,
        segments=[Segment("k00", source, 0.0, 1.0, speed_ratio=0.5)],
    )
    monkeypatch.setattr(renderer, "_encoder", lambda *_: "libx264")

    made = render(plan, tmp_path / "out")

    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames", "-of", "json",
        str(made.deliverable),
    ], check=True, capture_output=True, text=True)
    assert int(json.loads(probe.stdout)["streams"][0]["nb_frames"]) == 60


def test_atempo_chains_extreme_ratios_into_stable_steps():
    from montagewright.renderer import _atempo_chain

    assert _atempo_chain(2.0) == "atempo=2.000000000"
    assert _atempo_chain(0.5) == "atempo=0.500000000"
    # 4x is out of one atempo's stable range, so it composes as two doublings.
    assert _atempo_chain(4.0) == "atempo=2.000000000,atempo=2.000000000"
    # Quarter speed composes as two halvings.
    assert _atempo_chain(0.25) == "atempo=0.500000000,atempo=0.500000000"
