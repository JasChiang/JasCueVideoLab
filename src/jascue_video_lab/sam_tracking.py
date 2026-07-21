from __future__ import annotations

import json
import math
import re
import subprocess
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

from .geometry import box_iou, center_distance
from .media import probe_video, sha256_file
from .models import (
    Rational,
    SharedSam21AnalysisFrame,
    SharedSam21BBoxSeed,
    SharedSam21SessionManifest,
    SharedSam21SessionTarget,
    SharedSam21SessionTiming,
    SegmentationModelProvenance,
    SegmentationSample,
    SegmentationTrack,
    TrackerAgreementReport,
    TrackerAgreementSample,
    SemanticIdentityStatus,
    TrackingState,
)
from .shots import ShotBoundary, ShotManifest, ShotSegment, detect_shots_ffmpeg
from .storage import utc_now, write_json


SAM21_TINY_MODEL_ID = "sam2.1_hiera_tiny"
SAM21_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM21_IMPLEMENTATION_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


@dataclass(frozen=True)
class _AnalysisFrame:
    path: Path
    source_pts: int
    timeline_time_ms: int


@dataclass(frozen=True)
class _SeedShot:
    start_time_ms: int
    end_time_ms: int
    start_frame_pts: int | None
    boundary_score: float | None


@dataclass(frozen=True)
class _MaterializedMaskObservation:
    """Small per-frame record; full-resolution float logits are never retained."""

    mask_path: str | None
    mask_sha256: str | None
    mask_area_pixels: int
    mask_area_ratio: float
    connected_components: int
    derived_tracking_box: list[int] | None
    center_2d: list[float] | None
    mean_positive_probability: float | None


_SHOWINFO_FRAME_RE = re.compile(
    r"showinfo[^\n]*\bn:\s*(?P<index>\d+)\s+pts:\s*(?P<pts>-?\d+)\s+"
    r"pts_time:(?P<pts_time>-?[0-9.]+)"
)


def _require_segmentation_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as error:  # pragma: no cover - exercised by an optional install
        raise RuntimeError(
            "SAM 2.1 tracking requires the segmentation extra; see README setup instructions"
        ) from error
    return np, torch, build_sam2_video_predictor


def resolve_device(requested: str, torch: Any) -> str:
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable in this runtime")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this runtime")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalized_box_to_xyxy(
    box_2d: Sequence[int], width: int, height: int
) -> list[float]:
    if len(box_2d) != 4:
        raise ValueError("box_2d must contain four x-first normalized coordinates")
    x_min, y_min, x_max, y_max = box_2d
    if not (0 <= x_min < x_max <= 1000 and 0 <= y_min < y_max <= 1000):
        raise ValueError("box_2d must be [xmin,ymin,xmax,ymax] within 0..1000")
    return [
        x_min * width / 1000,
        y_min * height / 1000,
        x_max * width / 1000,
        y_max * height / 1000,
    ]


def pad_normalized_box(box_2d: Sequence[int], padding_ratio: float) -> list[int]:
    if not 0 <= padding_ratio <= 1:
        raise ValueError("padding_ratio must be in [0, 1]")
    normalized_box_to_xyxy(box_2d, 1000, 1000)
    x_min, y_min, x_max, y_max = box_2d
    x_padding = round((x_max - x_min) * padding_ratio)
    y_padding = round((y_max - y_min) * padding_ratio)
    return [
        max(0, x_min - x_padding),
        max(0, y_min - y_padding),
        min(1000, x_max + x_padding),
        min(1000, y_max + y_padding),
    ]


def normalized_polygon_to_mask(
    polygon_xy: Sequence[Sequence[int]], width: int, height: int
) -> Any:
    """Rasterize a normalized x/y polygon into an analysis-resolution binary mask."""
    np, _, _ = _require_segmentation_dependencies()
    if width <= 0 or height <= 0:
        raise ValueError("mask dimensions must be positive")
    if len(polygon_xy) < 3:
        raise ValueError("mask polygon must contain at least three points")
    pixels: list[tuple[int, int]] = []
    for point in polygon_xy:
        if len(point) != 2:
            raise ValueError("each mask polygon point must contain x and y")
        x, y = point
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            raise ValueError("mask polygon coordinates must be within 0..1000")
        pixels.append(
            (
                round(x * (width - 1) / 1000),
                round(y * (height - 1) / 1000),
            )
        )
    image = Image.new("1", (width, height), 0)
    ImageDraw.Draw(image).polygon(pixels, fill=1)
    mask = np.asarray(image, dtype=bool)
    if not mask.any():
        raise ValueError("rasterized mask polygon is empty")
    return mask


def binary_mask_geometry(mask: Any) -> dict[str, Any]:
    """Return canonical x-first geometry for a 2-D NumPy-compatible binary mask."""
    np, _, _ = _require_segmentation_dependencies()
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    height, width = binary.shape
    area = int(binary.sum())
    if area == 0:
        return {
            "area_pixels": 0,
            "area_ratio": 0.0,
            "box_2d": None,
            "center_2d": None,
        }
    ys, xs = np.nonzero(binary)
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max()) + 1
    y_max = int(ys.max()) + 1
    normalized = [
        max(0, min(999, round(x_min * 1000 / width))),
        max(0, min(999, round(y_min * 1000 / height))),
        max(1, min(1000, round(x_max * 1000 / width))),
        max(1, min(1000, round(y_max * 1000 / height))),
    ]
    normalized[2] = max(normalized[0] + 1, normalized[2])
    normalized[3] = max(normalized[1] + 1, normalized[3])
    return {
        "area_pixels": area,
        "area_ratio": area / (width * height),
        "box_2d": normalized,
        "center_2d": [
            round((normalized[0] + normalized[2]) / 2, 3),
            round((normalized[1] + normalized[3]) / 2, 3),
        ],
    }


def approximate_connected_components(mask: Any, max_side: int = 160) -> int:
    """Count materially sized components after downsampling for a cheap drift signal."""
    np, _, _ = _require_segmentation_dependencies()
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    if not binary.any():
        return 0
    step = max(1, math.ceil(max(binary.shape) / max_side))
    small = binary[::step, ::step]
    seen = np.zeros(small.shape, dtype=bool)
    component_sizes: list[int] = []
    height, width = small.shape
    for y, x in zip(*np.nonzero(small & ~seen), strict=False):
        if seen[y, x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(y), int(x))])
        seen[y, x] = True
        component_size = 0
        while queue:
            cy, cx = queue.popleft()
            component_size += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and small[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        component_sizes.append(component_size)
    minimum_size = max(2, round(int(small.sum()) * 0.005))
    return sum(size >= minimum_size for size in component_sizes)


def classify_tracking_state(
    *,
    area_ratio: float,
    connected_components: int,
    mean_positive_probability: float | None,
    previous_area_ratios: Sequence[float],
    center_2d: Sequence[float] | None,
    previous_center_2d: Sequence[float] | None,
    shot_boundary: bool,
) -> tuple[TrackingState, list[str]]:
    if area_ratio <= 0 or center_2d is None:
        return TrackingState.LOST, ["mask_empty"]
    drift_reasons: list[str] = []
    low_confidence_reasons: list[str] = []
    if shot_boundary:
        drift_reasons.append("shot_boundary_requires_new_seed")
    if previous_area_ratios:
        ordered = sorted(previous_area_ratios[-5:])
        reference = ordered[len(ordered) // 2]
        ratio = area_ratio / reference if reference > 0 else math.inf
        if ratio < 0.4 or ratio > 2.5:
            drift_reasons.append(f"mask_area_jump:{ratio:.3f}x")
    if previous_center_2d is not None:
        distance = math.dist(center_2d, previous_center_2d)
        if distance > 220:
            drift_reasons.append(f"center_jump:{distance:.1f}/1000")
    if connected_components > 4:
        low_confidence_reasons.append(f"mask_fragmented:{connected_components}_components")
    if mean_positive_probability is not None and mean_positive_probability < 0.6:
        low_confidence_reasons.append(
            f"weak_positive_logits:{mean_positive_probability:.3f}"
        )
    if drift_reasons:
        return TrackingState.DRIFT_SUSPECTED, drift_reasons + low_confidence_reasons
    if low_confidence_reasons:
        return TrackingState.LOW_CONFIDENCE, low_confidence_reasons
    return TrackingState.TRACKED, ["geometry_gates_passed"]


def _timeline_ms_from_pts(
    pts: int,
    *,
    source_start_pts: int,
    time_base_numerator: int,
    time_base_denominator: int,
) -> int:
    relative_seconds = Fraction(
        (pts - source_start_pts) * time_base_numerator,
        time_base_denominator,
    )
    return round(relative_seconds * 1000)


def _timeline_ms_in_interval_from_pts(
    pts: int,
    *,
    source_start_pts: int,
    time_base_numerator: int,
    time_base_denominator: int,
    start_time_ms: int,
    end_time_ms: int,
) -> int:
    """Represent an exact in-range PTS in the same half-open integer-ms interval."""
    relative_seconds = Fraction(
        (pts - source_start_pts) * time_base_numerator,
        time_base_denominator,
    )
    if not Fraction(start_time_ms, 1000) <= relative_seconds < Fraction(
        end_time_ms, 1000
    ):
        raise ValueError("source PTS is outside the selected analysis interval")
    return min(end_time_ms - 1, max(start_time_ms, round(relative_seconds * 1000)))


def _normalize_shot_manifest(
    manifest: ShotManifest,
    *,
    duration_ms: int,
    source_start_pts: int,
    time_base_numerator: int,
    time_base_denominator: int,
) -> ShotManifest:
    """Map detector PTS to the video's zero-based playback timeline.

    FFmpeg preserves non-zero stream PTS. Shot selection, however, receives the
    same zero-based times used by the local player and grounding pipeline. Keep
    the original frame PTS while deriving local milliseconds from the stream
    time base instead of trusting rounded or container-relative timestamps.
    """
    by_time: dict[int, ShotBoundary] = {}
    for boundary in manifest.boundaries:
        local_ms = _timeline_ms_from_pts(
            boundary.frame_pts,
            source_start_pts=source_start_pts,
            time_base_numerator=time_base_numerator,
            time_base_denominator=time_base_denominator,
        )
        if not 0 < local_ms < duration_ms:
            continue
        normalized = boundary.model_copy(update={"frame_time_ms": local_ms})
        existing = by_time.get(local_ms)
        if existing is None or normalized.score > existing.score:
            by_time[local_ms] = normalized
    boundaries = [by_time[time_ms] for time_ms in sorted(by_time)]
    starts: list[tuple[int, int | None, float | None]] = [
        (0, source_start_pts, None)
    ]
    starts.extend(
        (boundary.frame_time_ms, boundary.frame_pts, boundary.score)
        for boundary in boundaries
    )
    shots = [
        ShotSegment(
            shot_id=f"shot-{index + 1:04d}",
            start_time_ms=start_ms,
            end_time_ms=(starts[index + 1][0] if index + 1 < len(starts) else duration_ms),
            start_frame_pts=start_pts,
            boundary_source="video_start" if index == 0 else "ffmpeg_scdet",
            boundary_score=score,
        )
        for index, (start_ms, start_pts, score) in enumerate(starts)
    ]
    return manifest.model_copy(
        update={
            "duration_ms": duration_ms,
            "detector": f"{manifest.detector}; local time derived from source PTS",
            "timeline_basis": "local_ms_from_decoded_pts",
            "source_start_pts": source_start_pts,
            "source_time_base": Rational(
                numerator=time_base_numerator,
                denominator=time_base_denominator,
            ),
            "boundaries": boundaries,
            "shots": shots,
        }
    )


def _seed_shot(shots: Sequence[ShotSegment], seed_time_ms: int) -> _SeedShot:
    for shot in shots:
        if shot.start_time_ms <= seed_time_ms < shot.end_time_ms:
            return _SeedShot(
                start_time_ms=shot.start_time_ms,
                end_time_ms=shot.end_time_ms,
                start_frame_pts=shot.start_frame_pts,
                boundary_score=shot.boundary_score,
            )
    raise ValueError("seed_time_ms is outside the decoded video timeline")


def resolve_tracking_interval(
    manifest: ShotManifest,
    *,
    seed_time_ms: int,
    allowed_start_ms: int,
    allowed_end_ms: int,
) -> tuple[int, int]:
    """Resolve the half-open, shot-local interval consumed by a tracker.

    This pure helper is shared by cache-key construction and execution so a
    cached SAM artifact cannot hide a change in shot boundaries.
    """
    if not 0 <= allowed_start_ms < allowed_end_ms <= manifest.duration_ms:
        raise ValueError("allowed tracking interval must be inside the video duration")
    shot = _seed_shot(manifest.shots, seed_time_ms)
    start_ms = max(shot.start_time_ms, allowed_start_ms)
    end_ms = min(shot.end_time_ms, allowed_end_ms)
    if not start_ms <= seed_time_ms < end_ms:
        raise ValueError(
            "seed must lie inside the intersection of the allowed interval and seed shot"
        )
    return start_ms, end_ms


def require_bbox_track_request_match(
    track: SegmentationTrack,
    *,
    video_path: Path,
    asset_id: str,
    target_description: str,
    seed_time_ms: int,
    seed_box_2d: Sequence[int],
    seed_box_padding_ratio: float,
    analysis_fps: float,
    analysis_start_ms: int,
    analysis_end_ms: int,
    checkpoint_sha256: str,
    seed_frame_pts: int | None = None,
    seed_frame_sha256: str | None = None,
    seed_source_width: int | None = None,
    seed_source_height: int | None = None,
) -> None:
    """Fail closed if a cached SAM artifact is not this bbox-seed request."""
    expected = {
        "method": "bbox_seed_sam2_video_mask_propagation",
        "asset_id": asset_id,
        "video_path": str(video_path.expanduser().resolve()),
        "target_description": target_description,
        "seed_time_ms": seed_time_ms,
        "semantic_seed_box": list(seed_box_2d),
        "seed_prompt_type": "box",
        "sam_prompt_box": pad_normalized_box(
            seed_box_2d,
            seed_box_padding_ratio,
        ),
        "seed_box_padding_ratio": seed_box_padding_ratio,
        "analysis_fps": analysis_fps,
        "analysis_start_ms": analysis_start_ms,
        "analysis_end_ms": analysis_end_ms,
        "target_id": None,
        "shared_session_id": None,
        "analysis_frames_manifest_sha256": None,
        "seed_frame_pts": seed_frame_pts,
        "seed_frame_sha256": seed_frame_sha256,
        "seed_source_width": seed_source_width,
        "seed_source_height": seed_source_height,
    }
    mismatches = {
        field: {"expected": value, "actual": getattr(track, field)}
        for field, value in expected.items()
        if getattr(track, field) != value
    }
    provenance_expected = {
        "model_id": SAM21_TINY_MODEL_ID,
        "implementation": "facebookresearch/sam2",
        "implementation_revision": SAM21_IMPLEMENTATION_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
    }
    for field, value in provenance_expected.items():
        actual = getattr(track.model_provenance, field)
        if actual != value:
            mismatches[f"model_provenance.{field}"] = {
                "expected": value,
                "actual": actual,
            }
    if track.sam_prompt_mask_polygon_xy is not None:
        mismatches["sam_prompt_mask_polygon_xy"] = {
            "expected": None,
            "actual": track.sam_prompt_mask_polygon_xy,
        }
    if mismatches:
        raise ValueError(f"cached SAM track does not match bbox seed request: {mismatches}")


def _extract_analysis_frames(
    video_path: Path,
    frames_dir: Path,
    analysis_fps: float,
    max_side: int,
    *,
    start_time_ms: int,
    end_time_ms: int,
    source_start_pts: int,
    time_base_numerator: int,
    time_base_denominator: int,
    required_source_pts: Sequence[int] = (),
) -> tuple[list[_AnalysisFrame], int, int]:
    if analysis_fps <= 0 or analysis_fps > 60:
        raise ValueError("analysis_fps must be in (0, 60]")
    if max_side < 320:
        raise ValueError("max_side must be at least 320")
    if not 0 <= start_time_ms < end_time_ms:
        raise ValueError("analysis frame interval must be a non-empty half-open interval")
    required_pts = list(required_source_pts)
    if len(required_pts) != len(set(required_pts)):
        raise ValueError("required_source_pts must not contain duplicates")
    for pts in required_pts:
        relative_seconds = Fraction(
            (pts - source_start_pts) * time_base_numerator,
            time_base_denominator,
        )
        if not Fraction(start_time_ms, 1000) <= relative_seconds < Fraction(
            end_time_ms, 1000
        ):
            raise ValueError(
                "required source PTS is outside the exact selected analysis interval: "
                f"{pts}"
            )
        timeline_time_ms = _timeline_ms_in_interval_from_pts(
            pts,
            source_start_pts=source_start_pts,
            time_base_numerator=time_base_numerator,
            time_base_denominator=time_base_denominator,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )
        if not start_time_ms <= timeline_time_ms < end_time_ms:
            raise ValueError(
                "required source PTS is outside the selected analysis interval: "
                f"{pts} -> {timeline_time_ms} ms not in "
                f"[{start_time_ms}, {end_time_ms})"
            )
    frames_dir.mkdir(parents=True, exist_ok=True)
    if any(frames_dir.iterdir()):
        raise FileExistsError(f"analysis frame directory is not empty: {frames_dir}")
    time_base = Fraction(time_base_numerator, time_base_denominator)
    source_origin_seconds = Fraction(source_start_pts) * time_base
    absolute_start_seconds = source_origin_seconds + Fraction(start_time_ms, 1000)
    absolute_end_seconds = source_origin_seconds + Fraction(end_time_ms, 1000)
    interval_seconds = Fraction(1, 1) / Fraction(str(analysis_fps))
    start_seconds_text = f"{float(absolute_start_seconds):.9f}"
    interval_seconds_text = f"{float(interval_seconds):.9f}"
    current_bucket = f"floor((t-{start_seconds_text})/{interval_seconds_text})"
    previous_bucket = (
        f"floor(((prev_pts*TB)-{start_seconds_text})/{interval_seconds_text})"
    )
    regular_select = (
        f"gte(t\\,{start_seconds_text})*"
        f"lt(t\\,{float(absolute_end_seconds):.9f})*"
        f"(isnan(prev_pts)+gt({current_bucket}\\,{previous_bucket}))"
    )
    forced_select = "+".join(f"eq(pts\\,{pts})" for pts in sorted(required_pts))
    select = (
        f"({regular_select})+({forced_select})" if forced_select else regular_select
    )
    filter_graph = (
        f"select={select},"
        f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease,"
        "showinfo"
    )
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-copyts",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-vf",
            filter_graph,
            "-an",
            "-fps_mode",
            "vfr",
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(frames_dir / "%06d.jpg"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"FFmpeg analysis-frame extraction failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    paths = sorted(frames_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if not paths:
        raise RuntimeError("FFmpeg produced no analysis frames")
    matches = list(_SHOWINFO_FRAME_RE.finditer(completed.stderr))
    if len(matches) != len(paths):
        raise RuntimeError(
            "could not establish one-to-one source PTS lineage for analysis frames: "
            f"{len(paths)} files but {len(matches)} showinfo records"
        )
    frames: list[_AnalysisFrame] = []
    for expected_index, (path, match) in enumerate(zip(paths, matches, strict=True)):
        showinfo_index = int(match.group("index"))
        if showinfo_index != expected_index:
            raise RuntimeError(
                "FFmpeg showinfo frame indices are not contiguous from zero: "
                f"expected {expected_index}, got {showinfo_index}"
            )
        source_pts = int(match.group("pts"))
        relative_seconds = Fraction(
            (source_pts - source_start_pts) * time_base_numerator,
            time_base_denominator,
        )
        if not Fraction(start_time_ms, 1000) <= relative_seconds < Fraction(
            end_time_ms, 1000
        ):
            raise RuntimeError(
                "FFmpeg emitted an analysis frame outside the selected seed shot: "
                f"PTS {source_pts} not in [{start_time_ms}, {end_time_ms}) ms"
            )
        timeline_time_ms = _timeline_ms_in_interval_from_pts(
            source_pts,
            source_start_pts=source_start_pts,
            time_base_numerator=time_base_numerator,
            time_base_denominator=time_base_denominator,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )
        frames.append(
            _AnalysisFrame(
                path=path,
                source_pts=source_pts,
                timeline_time_ms=timeline_time_ms,
            )
        )
    emitted_pts = {frame.source_pts for frame in frames}
    missing_required_pts = sorted(set(required_pts) - emitted_pts)
    if missing_required_pts:
        raise RuntimeError(
            "FFmpeg did not emit every required source PTS: "
            f"{missing_required_pts}"
        )
    with Image.open(paths[0]) as first:
        width, height = first.size
    return frames, width, height


def _scene_cut_scores(frame_paths: Sequence[Path]) -> list[float | None]:
    scores: list[float | None] = [None]
    previous: list[float] | None = None
    for path in frame_paths:
        with Image.open(path).convert("RGB") as source:
            histogram = source.resize((64, 36)).histogram()
        total = sum(histogram)
        current = [value / total for value in histogram]
        if previous is not None:
            scores.append(min(1.0, sum(abs(a - b) for a, b in zip(current, previous)) / 2))
        previous = current
    return scores[: len(frame_paths)]


def _save_mask(mask: Any, output_path: Path) -> str:
    np, _, _ = _require_segmentation_dependencies()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(output_path)
    return sha256_file(output_path)


def _render_overlay(
    frame_path: Path,
    mask: Any,
    box_2d: Sequence[int] | None,
    state: TrackingState,
    output_path: Path,
) -> None:
    np, _, _ = _require_segmentation_dependencies()
    with Image.open(frame_path).convert("RGBA") as source:
        image = source.copy()
    rejected = state in {TrackingState.DRIFT_SUSPECTED, TrackingState.LOST}
    overlay_rgb = (255, 55, 75) if rejected else (0, 255, 170)
    alpha = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 105)
    color = Image.new("RGBA", image.size, (*overlay_rgb, 0))
    color.putalpha(alpha)
    image = Image.alpha_composite(image, color)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, round(min(image.size) / 35)))
    if box_2d is not None:
        x_min, y_min, x_max, y_max = box_2d
        pixels = (
            round(x_min * image.width / 1000),
            round(y_min * image.height / 1000),
            round(x_max * image.width / 1000),
            round(y_max * image.height / 1000),
        )
        outline = "#ff374b" if rejected else "#00ffaa"
        draw.rectangle(pixels, outline=outline, width=max(2, image.width // 400))
    draw.rectangle((0, 0, image.width, max(34, image.height // 12)), fill="#101820cc")
    draw.text((12, 8), f"SAM 2.1 mask | {state.value}", fill="white", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=90)


def _render_video(overlays_dir: Path, output_path: Path, analysis_fps: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(analysis_fps),
            "-i",
            str(overlays_dir / "%06d.jpg"),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )


def _logits_by_object_id(obj_ids: Any, mask_logits: Any) -> dict[str, Any]:
    """Map SAM's current output tensor without copying it into retained CPU arrays."""
    if obj_ids is None:
        raise RuntimeError("SAM did not return object IDs for multi-object output")
    resolved_ids = [str(obj_id) for obj_id in list(obj_ids)]
    if len(resolved_ids) != len(set(resolved_ids)):
        raise RuntimeError("SAM returned duplicate object IDs")
    if len(mask_logits) != len(resolved_ids):
        raise RuntimeError("SAM object IDs and mask logits have different batch sizes")
    return {
        obj_id: mask_logits[offset, 0]
        for offset, obj_id in enumerate(resolved_ids)
    }


def _materialize_mask_observation(
    *,
    np: Any,
    logits: Any,
    width: int,
    height: int,
    sample_index: int,
    target_output_dir: Path,
) -> _MaterializedMaskObservation:
    """Immediately reduce one SAM output to a mask artifact and compact metadata."""
    values = logits.detach().float().cpu().numpy()
    if values.shape != (height, width):
        raise RuntimeError(
            "SAM mask dimensions do not match analysis frames: "
            f"expected {(height, width)}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError("SAM returned non-finite mask logits")
    binary = np.asarray(values > 0, dtype=bool)
    geometry = binary_mask_geometry(binary)
    components = approximate_connected_components(binary)
    if geometry["area_pixels"]:
        probabilities = 1 / (1 + np.exp(-np.clip(values[binary], -30, 30)))
        mean_probability = float(probabilities.mean())
        mask_rel = Path("masks") / f"{sample_index:06d}.png"
        mask_hash = _save_mask(binary, target_output_dir / mask_rel)
        mask_path = str(mask_rel)
    else:
        mean_probability = None
        mask_hash = None
        mask_path = None
        stale_mask_path = target_output_dir / "masks" / f"{sample_index:06d}.png"
        if stale_mask_path.exists():
            stale_mask_path.unlink()
    return _MaterializedMaskObservation(
        mask_path=mask_path,
        mask_sha256=mask_hash,
        mask_area_pixels=geometry["area_pixels"],
        mask_area_ratio=geometry["area_ratio"],
        connected_components=components,
        derived_tracking_box=geometry["box_2d"],
        center_2d=geometry["center_2d"],
        mean_positive_probability=mean_probability,
    )


def _expected_observation_sources(
    sample_index: int, seed_index: int, *, reverse_seed_overlap: bool
) -> set[str]:
    if sample_index < seed_index:
        return {"reverse"}
    if sample_index > seed_index:
        return {"forward"}
    sources = {"prompt", "forward"}
    if reverse_seed_overlap:
        sources.add("reverse")
    return sources


def _record_direction_frame(
    *,
    direction: str,
    frame_idx: int,
    expected_indexes: set[int],
    seen_indexes: set[int],
) -> None:
    if frame_idx not in expected_indexes:
        raise RuntimeError(
            f"SAM {direction} propagation returned out-of-range frame {frame_idx}"
        )
    if frame_idx in seen_indexes:
        raise RuntimeError(
            f"SAM {direction} propagation returned duplicate frame {frame_idx}"
        )
    seen_indexes.add(frame_idx)


def _validate_shared_observation_coverage(
    *,
    frame_count: int,
    seed_indexes: dict[str, int],
    observation_sources: dict[str, dict[int, set[str]]],
    expected_forward_indexes: set[int],
    seen_forward_indexes: set[int],
    expected_reverse_indexes: set[int],
    seen_reverse_indexes: set[int],
) -> None:
    if seen_forward_indexes != expected_forward_indexes:
        missing = sorted(expected_forward_indexes - seen_forward_indexes)
        raise RuntimeError(f"SAM forward propagation frame coverage mismatch: missing={missing}")
    if seen_reverse_indexes != expected_reverse_indexes:
        missing = sorted(expected_reverse_indexes - seen_reverse_indexes)
        raise RuntimeError(f"SAM reverse propagation frame coverage mismatch: missing={missing}")
    expected_sample_indexes = set(range(frame_count))
    reverse_seed_overlap = bool(expected_reverse_indexes)
    for target_id, seed_index in seed_indexes.items():
        actual_indexes = set(observation_sources[target_id])
        if actual_indexes != expected_sample_indexes:
            missing = sorted(expected_sample_indexes - actual_indexes)
            extra = sorted(actual_indexes - expected_sample_indexes)
            raise RuntimeError(
                f"SAM target {target_id!r} frame coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        for sample_index in range(frame_count):
            expected_sources = _expected_observation_sources(
                sample_index,
                seed_index,
                reverse_seed_overlap=reverse_seed_overlap,
            )
            actual_sources = observation_sources[target_id][sample_index]
            if actual_sources != expected_sources:
                raise RuntimeError(
                    f"SAM target {target_id!r} frame {sample_index} source coverage "
                    f"mismatch: expected={sorted(expected_sources)}, "
                    f"actual={sorted(actual_sources)}"
                )


def _resolve_shared_seed_indexes(
    *,
    targets: Sequence[SharedSam21BBoxSeed],
    frame_records: Sequence[SharedSam21AnalysisFrame],
) -> dict[str, int]:
    """Resolve upstream evidence by exact source PTS, never nearest timeline time."""
    by_source_pts = {frame.source_pts: frame for frame in frame_records}
    if len(by_source_pts) != len(frame_records):
        raise RuntimeError("analysis frame manifest contains duplicate source PTS values")
    resolved: dict[str, int] = {}
    for target in targets:
        frame = by_source_pts.get(target.seed_frame_pts)
        if frame is None:
            raise ValueError(
                f"seed source PTS is not present in analysis frames for {target.target_id!r}"
            )
        if frame.analysis_sample_time_ms != target.seed_time_ms:
            raise ValueError(
                f"seed time does not match decoded source PTS for {target.target_id!r}"
            )
        resolved[target.target_id] = frame.sample_index
    return resolved


def _build_segmentation_samples(
    *,
    np: Any,
    analysis_frames: Sequence[_AnalysisFrame],
    width: int,
    height: int,
    seed_index: int,
    observations_by_index: dict[int, _MaterializedMaskObservation],
    seed_shot: _SeedShot,
    analysis_start_ms: int,
    target_output_dir: Path,
) -> list[SegmentationSample]:
    begins_at_shot_boundary = (
        seed_shot.start_time_ms > 0 and analysis_start_ms == seed_shot.start_time_ms
    )
    cut_scores: list[float | None] = [None] * len(analysis_frames)
    if begins_at_shot_boundary:
        cut_scores[0] = seed_shot.boundary_score
    overlays_dir = target_output_dir / "overlays"
    samples: list[SegmentationSample] = []
    previous_areas: list[float] = []
    previous_center: Sequence[float] | None = None
    for index, analysis_frame in enumerate(analysis_frames):
        observation = observations_by_index[index]
        if observation.mask_path is not None:
            mask_artifact = target_output_dir / observation.mask_path
            if sha256_file(mask_artifact) != observation.mask_sha256:
                raise RuntimeError(f"materialized mask changed before track assembly: {mask_artifact}")
            with Image.open(mask_artifact).convert("L") as saved_mask:
                binary = np.asarray(saved_mask, dtype=np.uint8) > 0
        else:
            binary = np.zeros((height, width), bool)
        comparison_areas = [] if index == seed_index else previous_areas
        comparison_center = None if index == seed_index else previous_center
        state, reasons = classify_tracking_state(
            area_ratio=observation.mask_area_ratio,
            connected_components=observation.connected_components,
            mean_positive_probability=observation.mean_positive_probability,
            previous_area_ratios=comparison_areas,
            center_2d=observation.center_2d,
            previous_center_2d=comparison_center,
            shot_boundary=False,
        )
        shot_boundary = index == 0 and begins_at_shot_boundary
        if shot_boundary:
            reasons.append("tracker_initialized_inside_new_shot")
        semantic_status = (
            SemanticIdentityStatus.SEED_GROUNDED
            if index == seed_index
            else SemanticIdentityStatus.REVALIDATION_REQUIRED
            if state in {TrackingState.DRIFT_SUSPECTED, TrackingState.LOST}
            else SemanticIdentityStatus.NOT_REVALIDATED
        )
        samples.append(
            SegmentationSample(
                sample_index=index,
                analysis_sample_time_ms=analysis_frame.timeline_time_ms,
                source_pts=analysis_frame.source_pts,
                timing_basis="decoded_source_pts",
                mask_path=observation.mask_path,
                mask_sha256=observation.mask_sha256,
                mask_area_pixels=observation.mask_area_pixels,
                mask_area_ratio=round(observation.mask_area_ratio, 8),
                connected_components=observation.connected_components,
                derived_tracking_box=observation.derived_tracking_box,
                center_2d=observation.center_2d,
                mean_positive_probability=(
                    round(observation.mean_positive_probability, 6)
                    if observation.mean_positive_probability is not None
                    else None
                ),
                scene_cut_score=(
                    round(cut_scores[index], 6)
                    if cut_scores[index] is not None
                    else None
                ),
                shot_boundary=shot_boundary,
                tracking_state=state,
                state_reasons=reasons,
                semantic_identity_status=semantic_status,
            )
        )
        _render_overlay(
            analysis_frame.path,
            binary,
            observation.derived_tracking_box,
            state,
            overlays_dir / f"{index:06d}.jpg",
        )
        if index == seed_index:
            previous_areas.clear()
            previous_center = None
        if observation.mask_area_ratio > 0:
            previous_areas.append(observation.mask_area_ratio)
            previous_center = observation.center_2d
    return samples


def track_bboxes_shared_sam21(
    *,
    video_path: Path,
    checkpoint_path: Path,
    targets: Sequence[SharedSam21BBoxSeed],
    output_dir: Path,
    asset_id: str,
    analysis_fps: float = 2.0,
    max_side: int = 960,
    device: str = "auto",
    ffmpeg_scdet_threshold: float = 4.0,
    seed_box_padding_ratio: float = 0.0,
    allowed_start_ms: int | None = None,
    allowed_end_ms: int | None = None,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = False,
) -> SharedSam21SessionManifest:
    """Track bbox-seeded objects in one decode and one SAM inference state."""
    request_targets = [SharedSam21BBoxSeed.model_validate(target) for target in targets]
    if len(request_targets) < 2:
        raise ValueError("shared SAM tracking requires at least two targets")
    target_ids = [target.target_id for target in request_targets]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("shared SAM target_id values must be unique")
    if not 0 <= ffmpeg_scdet_threshold <= 100:
        raise ValueError("ffmpeg_scdet_threshold must be in [0, 100]")
    if not 0 <= seed_box_padding_ratio <= 1:
        raise ValueError("seed_box_padding_ratio must be in [0, 1]")
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"shared SAM output directory is not empty: {output_dir}")
    total_started = monotonic()
    media = probe_video(video_path)
    if asset_id != media.asset_id:
        raise ValueError("bbox seed asset_id does not match the supplied tracking video")
    if (allowed_start_ms is None) != (allowed_end_ms is None):
        raise ValueError("allowed_start_ms and allowed_end_ms must be provided together")
    if allowed_start_ms is not None and not (
        0 <= allowed_start_ms < allowed_end_ms <= media.duration_ms
    ):
        raise ValueError("allowed tracking interval must be inside the video duration")
    source_start_pts = media.video.start_pts or 0
    time_base_numerator = media.video.time_base.numerator
    time_base_denominator = media.video.time_base.denominator
    shot_started = monotonic()
    detected_shots = detect_shots_ffmpeg(video_path, threshold=ffmpeg_scdet_threshold)
    shot_manifest = _normalize_shot_manifest(
        detected_shots,
        duration_ms=media.duration_ms,
        source_start_pts=source_start_pts,
        time_base_numerator=time_base_numerator,
        time_base_denominator=time_base_denominator,
    )
    shot_detection_seconds = monotonic() - shot_started
    intervals = {
        resolve_tracking_interval(
            shot_manifest,
            seed_time_ms=target.seed_time_ms,
            allowed_start_ms=allowed_start_ms or 0,
            allowed_end_ms=(
                allowed_end_ms if allowed_end_ms is not None else media.duration_ms
            ),
        )
        for target in request_targets
    }
    if len(intervals) != 1:
        raise ValueError("all shared SAM seeds must resolve to the same shot-local interval")
    analysis_start_ms, analysis_end_ms = intervals.pop()
    seed_shots = [_seed_shot(shot_manifest.shots, target.seed_time_ms) for target in request_targets]
    shot_ranges = {(shot.start_time_ms, shot.end_time_ms) for shot in seed_shots}
    if len(shot_ranges) != 1:
        raise ValueError("all shared SAM seeds must be inside the same shot")
    seed_shot = seed_shots[0]
    shot_segment = next(
        shot
        for shot in shot_manifest.shots
        if shot.start_time_ms == seed_shot.start_time_ms
        and shot.end_time_ms == seed_shot.end_time_ms
    )

    np, torch, build_sam2_video_predictor = _require_segmentation_dependencies()
    resolved_device = resolve_device(device, torch)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    session_id = f"shared-sam21-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "shots.json", shot_manifest)
    extract_started = monotonic()
    analysis_frames, width, height = _extract_analysis_frames(
        video_path,
        output_dir / "analysis-frames",
        analysis_fps,
        max_side,
        start_time_ms=analysis_start_ms,
        end_time_ms=analysis_end_ms,
        source_start_pts=source_start_pts,
        time_base_numerator=time_base_numerator,
        time_base_denominator=time_base_denominator,
        required_source_pts=sorted(
            {target.seed_frame_pts for target in request_targets}
        ),
    )
    extraction_seconds = monotonic() - extract_started
    frame_records = [
        SharedSam21AnalysisFrame(
            sample_index=index,
            analysis_sample_time_ms=frame.timeline_time_ms,
            source_pts=frame.source_pts,
            path=str(Path("analysis-frames") / frame.path.name),
            sha256=sha256_file(frame.path),
        )
        for index, frame in enumerate(analysis_frames)
    ]
    write_json(
        output_dir / "analysis-frames-manifest.json",
        {
            "timing_basis": "decoded_source_pts",
            "frames": [frame.model_dump(mode="json") for frame in frame_records],
        },
    )
    frames_manifest_sha256 = sha256_file(output_dir / "analysis-frames-manifest.json")
    seed_indexes = _resolve_shared_seed_indexes(
        targets=request_targets,
        frame_records=frame_records,
    )
    target_dirs = {
        target.target_id: output_dir / "targets" / target.target_id
        for target in request_targets
    }
    for target_dir in target_dirs.values():
        target_dir.mkdir(parents=True, exist_ok=False)

    init_started = monotonic()
    predictor = build_sam2_video_predictor(
        SAM21_CONFIG,
        str(checkpoint_path),
        device=resolved_device,
        apply_postprocessing=False,
    )
    inference_state = predictor.init_state(
        video_path=str(output_dir / "analysis-frames"),
        offload_video_to_cpu=offload_video_to_cpu,
        # Meta's own macOS demo keeps MPS state resident and only offloads video
        # frames. Offloading recurrent state forces a transfer every frame, so it
        # remains opt-in and is always recorded in the session manifest.
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=False,
    )
    initialization_seconds = monotonic() - init_started
    observations_by_target: dict[
        str, dict[int, _MaterializedMaskObservation]
    ] = {
        target.target_id: {} for target in request_targets
    }
    observation_sources: dict[str, dict[int, set[str]]] = {
        target.target_id: {} for target in request_targets
    }
    prompt_boxes: dict[str, list[int]] = {}
    reverse_seed_overlap = max(seed_indexes.values()) > 0

    def record_observation(
        *, target_id: str, frame_idx: int, source: str, logits: Any
    ) -> None:
        if not 0 <= frame_idx < len(analysis_frames):
            raise RuntimeError(
                f"SAM target {target_id!r} returned out-of-range frame {frame_idx}"
            )
        expected_sources = _expected_observation_sources(
            frame_idx,
            seed_indexes[target_id],
            reverse_seed_overlap=reverse_seed_overlap,
        )
        if source not in expected_sources:
            raise RuntimeError(
                f"SAM target {target_id!r} returned unexpected {source} output "
                f"for frame {frame_idx}"
            )
        sources = observation_sources[target_id].setdefault(frame_idx, set())
        if source in sources:
            raise RuntimeError(
                f"SAM target {target_id!r} returned duplicate {source} output "
                f"for frame {frame_idx}"
            )
        observation = _materialize_mask_observation(
            np=np,
            logits=logits,
            width=width,
            height=height,
            sample_index=frame_idx,
            target_output_dir=target_dirs[target_id],
        )
        observations_by_target[target_id][frame_idx] = observation
        sources.add(source)

    prompt_started = monotonic()
    for target in request_targets:
        prompt_box = pad_normalized_box(target.seed_box_2d, seed_box_padding_ratio)
        prompt_boxes[target.target_id] = prompt_box
        _, obj_ids, seed_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=seed_indexes[target.target_id],
            obj_id=target.target_id,
            box=np.asarray(
                normalized_box_to_xyxy(prompt_box, width, height), dtype=np.float32
            ),
        )
        outputs = _logits_by_object_id(obj_ids, seed_logits)
        if target.target_id not in outputs:
            raise RuntimeError(
                f"SAM prompt output omitted target_id {target.target_id!r}"
            )
        record_observation(
            target_id=target.target_id,
            frame_idx=seed_indexes[target.target_id],
            source="prompt",
            logits=outputs[target.target_id],
        )
    prompt_seconds = monotonic() - prompt_started

    target_id_set = set(target_ids)
    expected_forward_indexes = set(
        range(min(seed_indexes.values()), len(analysis_frames))
    )
    seen_forward_indexes: set[int] = set()
    forward_started = monotonic()
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=min(seed_indexes.values()), reverse=False
    ):
        _record_direction_frame(
            direction="forward",
            frame_idx=frame_idx,
            expected_indexes=expected_forward_indexes,
            seen_indexes=seen_forward_indexes,
        )
        outputs = _logits_by_object_id(obj_ids, mask_logits)
        if set(outputs) != target_id_set:
            raise RuntimeError(
                "SAM forward propagation object coverage mismatch: "
                f"expected={sorted(target_id_set)}, actual={sorted(outputs)}"
            )
        for target_id, logits in outputs.items():
            if frame_idx >= seed_indexes[target_id]:
                record_observation(
                    target_id=target_id,
                    frame_idx=frame_idx,
                    source="forward",
                    logits=logits,
                )
    forward_seconds = monotonic() - forward_started
    expected_reverse_indexes = (
        set(range(0, max(seed_indexes.values()) + 1))
        if reverse_seed_overlap
        else set()
    )
    seen_reverse_indexes: set[int] = set()
    reverse_started = monotonic()
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=max(seed_indexes.values()), reverse=True
    ):
        _record_direction_frame(
            direction="reverse",
            frame_idx=frame_idx,
            expected_indexes=expected_reverse_indexes,
            seen_indexes=seen_reverse_indexes,
        )
        outputs = _logits_by_object_id(obj_ids, mask_logits)
        if set(outputs) != target_id_set:
            raise RuntimeError(
                "SAM reverse propagation object coverage mismatch: "
                f"expected={sorted(target_id_set)}, actual={sorted(outputs)}"
            )
        for target_id, logits in outputs.items():
            if frame_idx <= seed_indexes[target_id]:
                record_observation(
                    target_id=target_id,
                    frame_idx=frame_idx,
                    source="reverse",
                    logits=logits,
                )
    reverse_seconds = monotonic() - reverse_started
    _validate_shared_observation_coverage(
        frame_count=len(analysis_frames),
        seed_indexes=seed_indexes,
        observation_sources=observation_sources,
        expected_forward_indexes=expected_forward_indexes,
        seen_forward_indexes=seen_forward_indexes,
        expected_reverse_indexes=expected_reverse_indexes,
        seen_reverse_indexes=seen_reverse_indexes,
    )
    for target in request_targets:
        seed_observation = observations_by_target[target.target_id][
            seed_indexes[target.target_id]
        ]
        if seed_observation.mask_area_pixels == 0:
            raise RuntimeError(f"SAM produced an empty seed mask for {target.target_id!r}")
    for frame, record in zip(analysis_frames, frame_records, strict=True):
        if sha256_file(frame.path) != record.sha256:
            raise RuntimeError("shared immutable analysis frames changed during SAM inference")

    provenance = SegmentationModelProvenance(
        model_id=SAM21_TINY_MODEL_ID,
        implementation="facebookresearch/sam2",
        implementation_revision=SAM21_IMPLEMENTATION_REVISION,
        checkpoint_sha256=checkpoint_sha256,
        device=resolved_device,
        torch_version=torch.__version__,
        generated_at=utc_now(),
    )
    artifact_started = monotonic()
    target_members: list[SharedSam21SessionTarget] = []
    for target in request_targets:
        target_dir = target_dirs[target.target_id]
        samples = _build_segmentation_samples(
            np=np,
            analysis_frames=analysis_frames,
            width=width,
            height=height,
            seed_index=seed_indexes[target.target_id],
            observations_by_index=observations_by_target[target.target_id],
            seed_shot=seed_shot,
            analysis_start_ms=analysis_start_ms,
            target_output_dir=target_dir,
        )
        state_counts = Counter(sample.tracking_state for sample in samples)
        inference_elapsed = prompt_seconds + forward_seconds + reverse_seconds
        track = SegmentationTrack(
            method="bbox_seed_sam2_video_mask_propagation",
            asset_id=asset_id,
            video_path=str(video_path.resolve()),
            target_description=target.target_description,
            seed_source=target.seed_source,
            seed_time_ms=target.seed_time_ms,
            seed_sample_index=seed_indexes[target.target_id],
            seed_frame_pts=target.seed_frame_pts,
            seed_frame_sha256=target.seed_frame_sha256,
            seed_source_width=target.seed_source_width,
            seed_source_height=target.seed_source_height,
            semantic_seed_box=target.seed_box_2d,
            seed_prompt_type="box",
            sam_prompt_box=prompt_boxes[target.target_id],
            sam_prompt_mask_polygon_xy=None,
            seed_box_padding_ratio=seed_box_padding_ratio,
            refined_seed_mask_path=str(
                Path("masks") / f"{seed_indexes[target.target_id]:06d}.png"
            ),
            analysis_fps=analysis_fps,
            analysis_width=width,
            analysis_height=height,
            analysis_start_ms=analysis_start_ms,
            analysis_end_ms=analysis_end_ms,
            source_start_pts=source_start_pts,
            source_time_base={
                "numerator": time_base_numerator,
                "denominator": time_base_denominator,
            },
            timing_warning=(
                "Samples share one immutable decoded-frame set and one SAM inference state. "
                "Per-track elapsed_seconds reports shared prompt and propagation wall time, "
                "not additive target cost. PTS remains authoritative."
            ),
            semantic_warning=(
                "Each target has an independent bbox semantic seed. Non-seed samples are "
                "geometry propagation, not semantic identity confirmation."
            ),
            total_samples=len(samples),
            state_counts=dict(state_counts),
            elapsed_seconds=round(inference_elapsed, 3),
            effective_fps=(
                round(len(samples) / inference_elapsed, 4) if inference_elapsed else 0
            ),
            model_provenance=provenance,
            samples=samples,
            target_id=target.target_id,
            shared_session_id=session_id,
            analysis_frames_manifest_sha256=frames_manifest_sha256,
        )
        track_path = target_dir / "segmentation-track.json"
        write_json(track_path, track)
        _render_video(target_dir / "overlays", target_dir / "segmentation-debug.mp4", analysis_fps)
        (target_dir / "summary.json").write_text(
            track.model_dump_json(indent=2, exclude={"samples"}) + "\n",
            encoding="utf-8",
        )
        target_members.append(
            SharedSam21SessionTarget(
                target_id=target.target_id,
                target_description=target.target_description,
                seed_time_ms=target.seed_time_ms,
                seed_sample_index=seed_indexes[target.target_id],
                seed_frame_pts=target.seed_frame_pts,
                seed_frame_sha256=target.seed_frame_sha256,
                seed_source_width=target.seed_source_width,
                seed_source_height=target.seed_source_height,
                track_path=str(
                    Path("targets") / target.target_id / "segmentation-track.json"
                ),
                track_sha256=sha256_file(track_path),
                state_counts=dict(state_counts),
            )
        )
    artifact_seconds = monotonic() - artifact_started
    total_seconds = monotonic() - total_started
    manifest = SharedSam21SessionManifest(
        artifact_type="shared_sam21_multi_object_tracking_session",
        method="bbox_seed_shared_sam2_video_mask_propagation",
        session_id=session_id,
        asset_id=asset_id,
        video_path=str(video_path.resolve()),
        shot_id=shot_segment.shot_id,
        analysis_fps=analysis_fps,
        analysis_width=width,
        analysis_height=height,
        analysis_start_ms=analysis_start_ms,
        analysis_end_ms=analysis_end_ms,
        source_start_pts=source_start_pts,
        source_time_base={
            "numerator": time_base_numerator,
            "denominator": time_base_denominator,
        },
        analysis_frames_path="analysis-frames-manifest.json",
        analysis_frames_manifest_sha256=frames_manifest_sha256,
        analysis_frames=frame_records,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        target_count=len(target_members),
        targets=target_members,
        model_provenance=provenance,
        timing=SharedSam21SessionTiming(
            shot_detection_seconds=round(shot_detection_seconds, 6),
            analysis_frame_extraction_seconds=round(extraction_seconds, 6),
            predictor_initialization_seconds=round(initialization_seconds, 6),
            prompt_seconds=round(prompt_seconds, 6),
            forward_propagation_seconds=round(forward_seconds, 6),
            reverse_propagation_seconds=round(reverse_seconds, 6),
            target_artifact_seconds=round(artifact_seconds, 6),
            total_seconds=round(total_seconds, 6),
        ),
        warning=(
            "All targets share one decode, predictor, and inference state, but retain "
            "independent masks, geometry states, semantic status, and artifact provenance."
        ),
        generated_at=utc_now(),
    )
    write_json(output_dir / "shared-session.json", manifest)
    return manifest


def track_bbox_sam21(
    *,
    video_path: Path,
    checkpoint_path: Path,
    seed_time_ms: int,
    seed_box_2d: Sequence[int],
    target_description: str,
    output_dir: Path,
    seed_source: str,
    asset_id: str | None = None,
    seed_frame_pts: int | None = None,
    seed_frame_sha256: str | None = None,
    seed_source_width: int | None = None,
    seed_source_height: int | None = None,
    analysis_fps: float = 2.0,
    max_side: int = 960,
    device: str = "auto",
    ffmpeg_scdet_threshold: float = 4.0,
    seed_box_padding_ratio: float = 0.0,
    allowed_start_ms: int | None = None,
    allowed_end_ms: int | None = None,
    seed_mask_polygon_xy: Sequence[Sequence[int]] | None = None,
) -> SegmentationTrack:
    """Refine a semantic bbox with SAM 2.1 and propagate only inside its seed shot."""
    if seed_mask_polygon_xy is not None:
        raise ValueError(
            "polygon seeds are disabled for the primary SAM path; provide a bbox seed"
        )
    if not 0 <= ffmpeg_scdet_threshold <= 100:
        raise ValueError("ffmpeg_scdet_threshold must be in [0, 100]")
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    media = probe_video(video_path)
    if asset_id is not None and asset_id != media.asset_id:
        raise ValueError(
            "bbox seed asset_id does not match the supplied tracking video"
        )
    seed_lineage = (
        seed_frame_pts,
        seed_frame_sha256,
        seed_source_width,
        seed_source_height,
    )
    if any(value is not None for value in seed_lineage) and not all(
        value is not None for value in seed_lineage
    ):
        raise ValueError("seed frame lineage fields must be provided together")
    if seed_frame_pts is not None:
        if asset_id is None:
            raise ValueError("exact seed frame lineage requires asset_id")
        decoded_seed_time_ms = _timeline_ms_from_pts(
            seed_frame_pts,
            source_start_pts=media.video.start_pts or 0,
            time_base_numerator=media.video.time_base.numerator,
            time_base_denominator=media.video.time_base.denominator,
        )
        if decoded_seed_time_ms != seed_time_ms:
            raise ValueError("seed_time_ms does not match seed_frame_pts")
    if (allowed_start_ms is None) != (allowed_end_ms is None):
        raise ValueError("allowed_start_ms and allowed_end_ms must be provided together")
    if allowed_start_ms is not None and not (
        0 <= allowed_start_ms < allowed_end_ms <= media.duration_ms
    ):
        raise ValueError("allowed tracking interval must be inside the video duration")
    source_start_pts = media.video.start_pts or 0
    time_base_numerator = media.video.time_base.numerator
    time_base_denominator = media.video.time_base.denominator
    detected_shots = detect_shots_ffmpeg(
        video_path,
        threshold=ffmpeg_scdet_threshold,
    )
    shot_manifest = _normalize_shot_manifest(
        detected_shots,
        duration_ms=media.duration_ms,
        source_start_pts=source_start_pts,
        time_base_numerator=time_base_numerator,
        time_base_denominator=time_base_denominator,
    )
    seed_shot = _seed_shot(shot_manifest.shots, seed_time_ms)
    analysis_start_ms, analysis_end_ms = resolve_tracking_interval(
        shot_manifest,
        seed_time_ms=seed_time_ms,
        allowed_start_ms=allowed_start_ms or 0,
        allowed_end_ms=(
            allowed_end_ms if allowed_end_ms is not None else media.duration_ms
        ),
    )
    np, torch, build_sam2_video_predictor = _require_segmentation_dependencies()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "shots.json", shot_manifest)
    analysis_frames, width, height = _extract_analysis_frames(
        video_path,
        output_dir / "analysis-frames",
        analysis_fps,
        max_side,
        start_time_ms=analysis_start_ms,
        end_time_ms=analysis_end_ms,
        source_start_pts=source_start_pts,
        time_base_numerator=time_base_numerator,
        time_base_denominator=time_base_denominator,
        required_source_pts=([seed_frame_pts] if seed_frame_pts is not None else ()),
    )
    if seed_frame_pts is not None:
        matching_indexes = [
            index
            for index, frame in enumerate(analysis_frames)
            if frame.source_pts == seed_frame_pts
        ]
        if len(matching_indexes) != 1:
            raise RuntimeError(
                "exact seed source PTS must occur exactly once in analysis frames"
            )
        seed_index = matching_indexes[0]
    else:
        seed_index = min(
            range(len(analysis_frames)),
            key=lambda index: abs(analysis_frames[index].timeline_time_ms - seed_time_ms),
        )
    resolved_device = resolve_device(device, torch)
    started = monotonic()
    predictor = build_sam2_video_predictor(
        SAM21_CONFIG,
        str(checkpoint_path),
        device=resolved_device,
        apply_postprocessing=False,
    )
    inference_state = predictor.init_state(
        video_path=str(output_dir / "analysis-frames"),
        offload_video_to_cpu=True,
        # Keep recurrent state on the compute device.  On MPS, Meta's reference
        # demo offloads video frames to CPU to avoid fragmentation but does not
        # offload state, which otherwise creates a transfer on every frame.
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    sam_prompt_box = pad_normalized_box(seed_box_2d, seed_box_padding_ratio)
    seed_box_xyxy = normalized_box_to_xyxy(sam_prompt_box, width, height)
    _, _, seed_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=seed_index,
        obj_id="target",
        box=np.asarray(seed_box_xyxy, dtype=np.float32),
    )
    logits_by_index: dict[int, Any] = {
        seed_index: seed_logits[0, 0].detach().float().cpu().numpy()
    }
    for frame_idx, _, mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=seed_index, reverse=False
    ):
        logits_by_index[frame_idx] = mask_logits[0, 0].detach().float().cpu().numpy()
    for frame_idx, _, mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=seed_index, reverse=True
    ):
        logits_by_index[frame_idx] = mask_logits[0, 0].detach().float().cpu().numpy()
    elapsed = monotonic() - started
    cut_scores: list[float | None] = [None] * len(analysis_frames)
    begins_at_shot_boundary = (
        seed_shot.start_time_ms > 0
        and analysis_start_ms == seed_shot.start_time_ms
    )
    if begins_at_shot_boundary:
        cut_scores[0] = seed_shot.boundary_score
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    samples: list[SegmentationSample] = []
    previous_areas: list[float] = []
    previous_center: Sequence[float] | None = None
    for index, analysis_frame in enumerate(analysis_frames):
        frame_path = analysis_frame.path
        logits = logits_by_index.get(index)
        binary = (
            np.asarray(logits > 0, dtype=bool)
            if logits is not None
            else np.zeros((height, width), bool)
        )
        geometry = binary_mask_geometry(binary)
        components = approximate_connected_components(binary)
        if geometry["area_pixels"]:
            probabilities = 1 / (1 + np.exp(-np.clip(logits[binary], -30, 30)))
            mean_probability = float(probabilities.mean())
            mask_rel = Path("masks") / f"{index:06d}.png"
            mask_hash = _save_mask(binary, output_dir / mask_rel)
            mask_path = str(mask_rel)
        else:
            mean_probability = None
            mask_hash = None
            mask_path = None
        cut_score = cut_scores[index]
        shot_boundary = index == 0 and begins_at_shot_boundary
        comparison_areas = [] if index == seed_index else previous_areas
        comparison_center = None if index == seed_index else previous_center
        state, reasons = classify_tracking_state(
            area_ratio=geometry["area_ratio"],
            connected_components=components,
            mean_positive_probability=mean_probability,
            previous_area_ratios=comparison_areas,
            center_2d=geometry["center_2d"],
            previous_center_2d=comparison_center,
            # The predictor was initialized after the cut, so the first frame is
            # evidence of a reset rather than a cross-shot drift condition.
            shot_boundary=False,
        )
        if shot_boundary:
            reasons.append("tracker_initialized_inside_new_shot")
        semantic_status = (
            SemanticIdentityStatus.SEED_GROUNDED
            if index == seed_index
            else SemanticIdentityStatus.REVALIDATION_REQUIRED
            if state in {TrackingState.DRIFT_SUSPECTED, TrackingState.LOST}
            else SemanticIdentityStatus.NOT_REVALIDATED
        )
        sample = SegmentationSample(
            sample_index=index,
            analysis_sample_time_ms=analysis_frame.timeline_time_ms,
            source_pts=analysis_frame.source_pts,
            timing_basis="decoded_source_pts",
            mask_path=mask_path,
            mask_sha256=mask_hash,
            mask_area_pixels=geometry["area_pixels"],
            mask_area_ratio=round(geometry["area_ratio"], 8),
            connected_components=components,
            derived_tracking_box=geometry["box_2d"],
            center_2d=geometry["center_2d"],
            mean_positive_probability=(
                round(mean_probability, 6) if mean_probability is not None else None
            ),
            scene_cut_score=round(cut_score, 6) if cut_score is not None else None,
            shot_boundary=shot_boundary,
            tracking_state=state,
            state_reasons=reasons,
            semantic_identity_status=semantic_status,
        )
        samples.append(sample)
        _render_overlay(
            frame_path,
            binary,
            geometry["box_2d"],
            state,
            overlays_dir / f"{index:06d}.jpg",
        )
        if index == seed_index:
            previous_areas.clear()
            previous_center = None
        if geometry["area_ratio"] > 0:
            previous_areas.append(geometry["area_ratio"])
            previous_center = geometry["center_2d"]
    state_counts = Counter(sample.tracking_state for sample in samples)
    track = SegmentationTrack(
        method="bbox_seed_sam2_video_mask_propagation",
        asset_id=asset_id or media.asset_id,
        video_path=str(video_path.resolve()),
        target_description=target_description,
        seed_source=seed_source,
        seed_time_ms=seed_time_ms,
        seed_sample_index=seed_index,
        seed_frame_pts=seed_frame_pts,
        seed_frame_sha256=seed_frame_sha256,
        seed_source_width=seed_source_width,
        seed_source_height=seed_source_height,
        semantic_seed_box=list(seed_box_2d),
        seed_prompt_type="box",
        sam_prompt_box=sam_prompt_box,
        sam_prompt_mask_polygon_xy=None,
        seed_box_padding_ratio=seed_box_padding_ratio,
        refined_seed_mask_path=str(Path("masks") / f"{seed_index:06d}.png"),
        analysis_fps=analysis_fps,
        analysis_width=width,
        analysis_height=height,
        analysis_start_ms=analysis_start_ms,
        analysis_end_ms=analysis_end_ms,
        source_start_pts=source_start_pts,
        source_time_base={
            "numerator": time_base_numerator,
            "denominator": time_base_denominator,
        },
        timing_warning=(
            "analysis_sample_time_ms is derived from each selected decoded frame's source PTS "
            "relative to stream start_pts; source_pts preserves the decoded source timestamp. "
            "The constant-rate debug video is not an edit timeline."
        ),
        semantic_warning=(
            "SAM propagates geometry from one semantic seed. Non-seed samples are not semantic "
            "identity confirmations unless a separate re-grounding result is attached."
        ),
        total_samples=len(samples),
        state_counts=dict(state_counts),
        elapsed_seconds=round(elapsed, 3),
        effective_fps=round(len(samples) / elapsed, 4) if elapsed else 0,
        model_provenance=SegmentationModelProvenance(
            model_id=SAM21_TINY_MODEL_ID,
            implementation="facebookresearch/sam2",
            implementation_revision=SAM21_IMPLEMENTATION_REVISION,
            checkpoint_sha256=sha256_file(checkpoint_path),
            device=resolved_device,
            torch_version=torch.__version__,
            generated_at=utc_now(),
        ),
        samples=samples,
    )
    write_json(output_dir / "segmentation-track.json", track)
    _render_video(overlays_dir, output_dir / "segmentation-debug.mp4", analysis_fps)
    (output_dir / "summary.json").write_text(
        json.dumps(
            track.model_dump(mode="json", exclude={"samples"}),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return track


def compare_segmentation_to_bbox_track(
    segmentation_path: Path, reference_path: Path, output_path: Path
) -> TrackerAgreementReport:
    segmentation = SegmentationTrack.model_validate_json(
        segmentation_path.read_text(encoding="utf-8")
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_rows = [row for row in reference.get("samples", []) if row.get("box_2d")]
    if not reference_rows:
        raise ValueError("reference bbox track contains no usable boxes")
    samples: list[TrackerAgreementSample] = []
    for sample in segmentation.samples:
        if sample.derived_tracking_box is None:
            continue
        nearest = min(
            reference_rows,
            key=lambda row: abs(row["decoded_time_ms"] - sample.analysis_sample_time_ms),
        )
        iou = box_iou(sample.derived_tracking_box, nearest["box_2d"])
        distance = center_distance(sample.derived_tracking_box, nearest["box_2d"])
        samples.append(
            TrackerAgreementSample(
                analysis_sample_time_ms=sample.analysis_sample_time_ms,
                reference_time_ms=nearest["decoded_time_ms"],
                segmentation_box=sample.derived_tracking_box,
                reference_box=nearest["box_2d"],
                bbox_iou=round(iou, 6),
                center_distance_normalized=round(distance, 6),
            )
        )
    if not samples:
        raise ValueError("no segmentation and bbox samples could be aligned")
    ious = [sample.bbox_iou for sample in samples]
    distances = [sample.center_distance_normalized for sample in samples]
    report = TrackerAgreementReport(
        interpretation="tracker_agreement_not_accuracy",
        segmentation_path=str(segmentation_path),
        reference_path=str(reference_path),
        reference_method=str(reference.get("method", "unknown")),
        aligned_samples=len(samples),
        mean_bbox_iou=round(sum(ious) / len(ious), 6),
        min_bbox_iou=min(ious),
        mean_center_distance_normalized=round(sum(distances) / len(distances), 6),
        max_center_distance_normalized=max(distances),
        warning=(
            "This compares two model-derived tracks. It measures agreement, not accuracy, and "
            "must not be presented as human ground truth."
        ),
        samples=samples,
    )
    write_json(output_path, report)
    return report
