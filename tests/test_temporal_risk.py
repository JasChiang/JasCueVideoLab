from __future__ import annotations

import subprocess
from pathlib import Path

from jascue_video_lab.temporal_risk import (
    TemporalDifferenceSample,
    build_temporal_risk_windows,
    scan_temporal_risk_windows,
)


def _sample(
    index: int,
    time_ms: int,
    *,
    mean_delta: float = 0.0,
    changed_fraction: float = 0.0,
    boundary: bool = False,
) -> TemporalDifferenceSample:
    return TemporalDifferenceSample(
        sample_index=index,
        sample_time_ms=time_ms,
        mean_absolute_luma_delta=mean_delta,
        changed_pixel_fraction=changed_fraction,
        near_shot_boundary=boundary,
    )


def test_short_entry_and_exit_changes_become_one_dense_review_window() -> None:
    peaks, windows = build_temporal_risk_windows(
        [
            _sample(1, 250),
            _sample(2, 500, mean_delta=0.08, changed_fraction=0.12),
            _sample(3, 750),
            _sample(4, 1000, mean_delta=0.07, changed_fraction=0.11),
            _sample(5, 1250),
        ],
        duration_ms=2000,
        padding_ms=250,
        merge_gap_ms=250,
    )

    assert [peak.sample_time_ms for peak in peaks] == [500, 1000]
    assert len(windows) == 1
    assert windows[0].start_ms == 250
    assert windows[0].end_ms == 1251
    assert windows[0].evidence_sample_indexes == (2, 4)
    assert "localized_pixel_change" in windows[0].reasons


def test_known_shot_boundary_is_excluded_unless_explicitly_requested() -> None:
    sample = _sample(
        1,
        1000,
        mean_delta=0.5,
        changed_fraction=0.9,
        boundary=True,
    )

    excluded, excluded_windows = build_temporal_risk_windows(
        [sample],
        duration_ms=2000,
        include_shot_boundaries=False,
    )
    included, included_windows = build_temporal_risk_windows(
        [sample],
        duration_ms=2000,
        include_shot_boundaries=True,
    )

    assert excluded == ()
    assert excluded_windows == ()
    assert included[0].reasons[-1] == "shot_boundary_change"
    assert len(included_windows) == 1


def test_ffmpeg_scan_finds_transient_visual_change_without_clip_card(
    tmp_path: Path,
) -> None:
    video = tmp_path / "transient.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:r=8:d=2",
            "-vf",
            (
                "drawbox=x=40:y=20:w=40:h=30:color=white:t=fill:"
                "enable='between(t,0.75,1.0)'"
            ),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    output = tmp_path / "temporal-risk.json"

    scan = scan_temporal_risk_windows(
        video,
        duration_ms=2000,
        sampling_fps=8,
        analysis_width=160,
        analysis_height=90,
        mean_delta_threshold=0.02,
        changed_fraction_threshold=0.04,
        pixel_delta_threshold=20,
        padding_ms=250,
        merge_gap_ms=250,
        output_path=output,
    )

    assert scan.decoded_sample_count >= 15
    assert len(scan.risk_peaks) >= 2
    assert any(window.start_ms <= 750 < window.end_ms for window in scan.windows)
    assert output.is_file()
    assert "not semantic events" in scan.warning
