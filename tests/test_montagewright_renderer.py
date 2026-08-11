import json
import subprocess
from pathlib import Path

from montagewright.executor import RenderPlan, Segment, Source
from montagewright.renderer import render


def _colour_clip(path: Path, *, fps: int, colour: str) -> None:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c={colour}:s=320x180:d=0.6:r={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ], check=True)


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
