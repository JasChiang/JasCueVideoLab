"""Recall-only local scan for short visual changes that coarse VLM sampling may miss."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from PIL import Image, ImageChops
from pydantic import Field, model_validator

from .media import sha256_file
from .models import StrictModel
from .shots import ShotManifest
from .storage import utc_now, write_json


TEMPORAL_RISK_SCANNER_VERSION = "temporal-risk-window-scanner-v1"


class TemporalDifferenceSample(StrictModel):
    sample_index: int = Field(ge=1)
    sample_time_ms: int = Field(ge=0)
    mean_absolute_luma_delta: float = Field(ge=0.0, le=1.0)
    changed_pixel_fraction: float = Field(ge=0.0, le=1.0)
    near_shot_boundary: bool = False


class TemporalRiskPeak(StrictModel):
    sample_index: int = Field(ge=1)
    sample_time_ms: int = Field(ge=0)
    score: float = Field(ge=0.0)
    reasons: tuple[
        Literal[
            "mean_luma_change",
            "localized_pixel_change",
            "shot_boundary_change",
        ],
        ...,
    ] = Field(min_length=1)
    mean_absolute_luma_delta: float = Field(ge=0.0, le=1.0)
    changed_pixel_fraction: float = Field(ge=0.0, le=1.0)


class TemporalRiskWindow(StrictModel):
    window_id: str = Field(pattern=r"^risk-window-[0-9]{4}$")
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    peak_score: float = Field(gt=0.0)
    peak_time_ms: int = Field(ge=0)
    reasons: tuple[str, ...] = Field(min_length=1)
    evidence_sample_indexes: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_window(self) -> "TemporalRiskWindow":
        if self.start_ms >= self.end_ms:
            raise ValueError("temporal risk windows must be non-empty")
        if not self.start_ms <= self.peak_time_ms < self.end_ms:
            raise ValueError("peak_time_ms must fall within the risk window")
        if len(self.evidence_sample_indexes) != len(
            set(self.evidence_sample_indexes)
        ):
            raise ValueError("risk-window evidence samples must be unique")
        return self


class TemporalRiskScan(StrictModel):
    artifact_type: Literal["temporal_risk_scan_v1"] = "temporal_risk_scan_v1"
    scanner_version: Literal["temporal-risk-window-scanner-v1"] = (
        TEMPORAL_RISK_SCANNER_VERSION
    )
    video_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    duration_ms: int = Field(gt=0)
    sampling_fps: float = Field(gt=0.0, le=12.0)
    analysis_width: int = Field(gt=0)
    analysis_height: int = Field(gt=0)
    mean_delta_threshold: float = Field(gt=0.0, le=1.0)
    changed_fraction_threshold: float = Field(gt=0.0, le=1.0)
    pixel_delta_threshold: int = Field(ge=1, le=255)
    include_shot_boundaries: bool
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decoded_sample_count: int = Field(ge=1)
    risk_peaks: tuple[TemporalRiskPeak, ...]
    windows: tuple[TemporalRiskWindow, ...]
    generated_at: str = Field(min_length=1)
    warning: str = Field(min_length=1)


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_temporal_risk_windows(
    samples: Sequence[TemporalDifferenceSample],
    *,
    duration_ms: int,
    mean_delta_threshold: float = 0.04,
    changed_fraction_threshold: float = 0.08,
    include_shot_boundaries: bool = False,
    padding_ms: int = 500,
    merge_gap_ms: int = 500,
) -> tuple[tuple[TemporalRiskPeak, ...], tuple[TemporalRiskWindow, ...]]:
    """Convert local change measurements into recall-oriented search windows.

    The output is not an event claim.  It only identifies intervals that merit
    denser exact-frame review independently of any existing Clip Card event.
    """

    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if not 0 < mean_delta_threshold <= 1:
        raise ValueError("mean_delta_threshold must be within (0, 1]")
    if not 0 < changed_fraction_threshold <= 1:
        raise ValueError("changed_fraction_threshold must be within (0, 1]")
    if padding_ms < 0 or merge_gap_ms < 0:
        raise ValueError("padding and merge gap must be non-negative")
    times = [sample.sample_time_ms for sample in samples]
    if times != sorted(times):
        raise ValueError("temporal difference samples must be time ordered")

    peaks: list[TemporalRiskPeak] = []
    for sample in samples:
        reasons: list[
            Literal[
                "mean_luma_change",
                "localized_pixel_change",
                "shot_boundary_change",
            ]
        ] = []
        if sample.mean_absolute_luma_delta >= mean_delta_threshold:
            reasons.append("mean_luma_change")
        if sample.changed_pixel_fraction >= changed_fraction_threshold:
            reasons.append("localized_pixel_change")
        if sample.near_shot_boundary:
            if include_shot_boundaries:
                reasons.append("shot_boundary_change")
            else:
                continue
        if not reasons:
            continue
        score = max(
            sample.mean_absolute_luma_delta / mean_delta_threshold,
            sample.changed_pixel_fraction / changed_fraction_threshold,
        )
        peaks.append(
            TemporalRiskPeak(
                sample_index=sample.sample_index,
                sample_time_ms=sample.sample_time_ms,
                score=round(score, 6),
                reasons=tuple(reasons),
                mean_absolute_luma_delta=sample.mean_absolute_luma_delta,
                changed_pixel_fraction=sample.changed_pixel_fraction,
            )
        )

    clusters: list[list[TemporalRiskPeak]] = []
    for peak in peaks:
        if (
            not clusters
            or peak.sample_time_ms - clusters[-1][-1].sample_time_ms
            > merge_gap_ms + 2 * padding_ms
        ):
            clusters.append([peak])
        else:
            clusters[-1].append(peak)

    windows: list[TemporalRiskWindow] = []
    for index, cluster in enumerate(clusters, start=1):
        peak = max(cluster, key=lambda item: (item.score, -item.sample_time_ms))
        start_ms = max(0, cluster[0].sample_time_ms - padding_ms)
        end_ms = min(duration_ms, cluster[-1].sample_time_ms + padding_ms + 1)
        if end_ms <= start_ms:
            end_ms = min(duration_ms, start_ms + 1)
        windows.append(
            TemporalRiskWindow(
                window_id=f"risk-window-{index:04d}",
                start_ms=start_ms,
                end_ms=end_ms,
                peak_score=peak.score,
                peak_time_ms=peak.sample_time_ms,
                reasons=tuple(
                    sorted({reason for item in cluster for reason in item.reasons})
                ),
                evidence_sample_indexes=tuple(
                    item.sample_index for item in cluster
                ),
            )
        )
    return tuple(peaks), tuple(windows)


def _decode_difference_samples(
    video_path: Path,
    *,
    sampling_fps: float,
    analysis_width: int,
    analysis_height: int,
    pixel_delta_threshold: int,
    shot_manifest: ShotManifest | None,
) -> tuple[list[TemporalDifferenceSample], int]:
    filter_graph = (
        f"fps={sampling_fps},"
        f"scale={analysis_width}:{analysis_height}:"
        "force_original_aspect_ratio=decrease,"
        f"pad={analysis_width}:{analysis_height}:(ow-iw)/2:(oh-ih)/2,"
        "format=gray"
    )
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-an",
            "-vf",
            filter_graph,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("failed to open FFmpeg temporal-risk pipes")
    frame_size = analysis_width * analysis_height
    previous: Image.Image | None = None
    samples: list[TemporalDifferenceSample] = []
    decoded_count = 0
    boundary_times = (
        [boundary.frame_time_ms for boundary in shot_manifest.boundaries]
        if shot_manifest is not None
        else []
    )
    boundary_tolerance_ms = max(1, round(500 / sampling_fps))
    while True:
        payload = process.stdout.read(frame_size)
        if not payload:
            break
        if len(payload) != frame_size:
            process.kill()
            raise RuntimeError("FFmpeg returned a truncated temporal-risk frame")
        current = Image.frombytes(
            "L", (analysis_width, analysis_height), payload
        )
        if previous is not None:
            difference = ImageChops.difference(previous, current)
            histogram = difference.histogram()
            absolute_total = sum(value * count for value, count in enumerate(histogram))
            changed = sum(histogram[pixel_delta_threshold:])
            sample_time_ms = round(decoded_count * 1000 / sampling_fps)
            samples.append(
                TemporalDifferenceSample(
                    sample_index=decoded_count,
                    sample_time_ms=sample_time_ms,
                    mean_absolute_luma_delta=round(
                        absolute_total / (frame_size * 255), 8
                    ),
                    changed_pixel_fraction=round(changed / frame_size, 8),
                    near_shot_boundary=any(
                        abs(sample_time_ms - boundary_time)
                        <= boundary_tolerance_ms
                        for boundary_time in boundary_times
                    ),
                )
            )
        previous = current
        decoded_count += 1
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"FFmpeg temporal-risk scan failed: {stderr.strip()}")
    if decoded_count == 0:
        raise ValueError("temporal-risk scan decoded no video frames")
    return samples, decoded_count


def scan_temporal_risk_windows(
    video_path: Path,
    *,
    duration_ms: int,
    sampling_fps: float = 4.0,
    analysis_width: int = 256,
    analysis_height: int = 256,
    mean_delta_threshold: float = 0.04,
    changed_fraction_threshold: float = 0.08,
    pixel_delta_threshold: int = 20,
    include_shot_boundaries: bool = False,
    padding_ms: int = 500,
    merge_gap_ms: int = 500,
    shot_manifest: ShotManifest | None = None,
    output_path: Path | None = None,
) -> TemporalRiskScan:
    """Decode a small local proxy and produce independent dense-review windows."""

    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not 0 < sampling_fps <= 12:
        raise ValueError("sampling_fps must be within (0, 12]")
    if analysis_width <= 0 or analysis_height <= 0:
        raise ValueError("analysis dimensions must be positive")
    if not 1 <= pixel_delta_threshold <= 255:
        raise ValueError("pixel_delta_threshold must be within 1..255")
    samples, decoded_count = _decode_difference_samples(
        video_path,
        sampling_fps=sampling_fps,
        analysis_width=analysis_width,
        analysis_height=analysis_height,
        pixel_delta_threshold=pixel_delta_threshold,
        shot_manifest=shot_manifest,
    )
    peaks, windows = build_temporal_risk_windows(
        samples,
        duration_ms=duration_ms,
        mean_delta_threshold=mean_delta_threshold,
        changed_fraction_threshold=changed_fraction_threshold,
        include_shot_boundaries=include_shot_boundaries,
        padding_ms=padding_ms,
        merge_gap_ms=merge_gap_ms,
    )
    request = {
        "scanner_version": TEMPORAL_RISK_SCANNER_VERSION,
        "source_sha256": sha256_file(video_path),
        "duration_ms": duration_ms,
        "sampling_fps": sampling_fps,
        "analysis_width": analysis_width,
        "analysis_height": analysis_height,
        "mean_delta_threshold": mean_delta_threshold,
        "changed_fraction_threshold": changed_fraction_threshold,
        "pixel_delta_threshold": pixel_delta_threshold,
        "include_shot_boundaries": include_shot_boundaries,
        "padding_ms": padding_ms,
        "merge_gap_ms": merge_gap_ms,
        "shot_manifest": (
            {
                "detector": shot_manifest.detector,
                "threshold": shot_manifest.threshold,
                "boundaries": [
                    boundary.model_dump(mode="json")
                    for boundary in shot_manifest.boundaries
                ],
            }
            if shot_manifest is not None
            else None
        ),
    }
    scan = TemporalRiskScan(
        video_path=str(video_path.resolve()),
        source_sha256=request["source_sha256"],
        duration_ms=duration_ms,
        sampling_fps=sampling_fps,
        analysis_width=analysis_width,
        analysis_height=analysis_height,
        mean_delta_threshold=mean_delta_threshold,
        changed_fraction_threshold=changed_fraction_threshold,
        pixel_delta_threshold=pixel_delta_threshold,
        include_shot_boundaries=include_shot_boundaries,
        request_sha256=_canonical_sha256(request),
        decoded_sample_count=decoded_count,
        risk_peaks=peaks,
        windows=windows,
        generated_at=utc_now(),
        warning=(
            "Risk windows are recall-only local visual-change signals, not "
            "semantic events or frame-accurate edit decisions."
        ),
    )
    if output_path is not None:
        write_json(output_path, scan)
    return scan
