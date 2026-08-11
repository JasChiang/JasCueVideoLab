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
