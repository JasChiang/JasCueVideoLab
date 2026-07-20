from __future__ import annotations

import json
import math
import subprocess
from collections import Counter, deque
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

from .media import sha256_file
from .geometry import box_iou, center_distance
from .models import (
    SegmentationModelProvenance,
    SegmentationSample,
    SegmentationTrack,
    TrackerAgreementReport,
    TrackerAgreementSample,
    SemanticIdentityStatus,
    TrackingState,
)
from .shots import detect_shots_ffmpeg
from .storage import utc_now, write_json


SAM21_TINY_MODEL_ID = "sam2.1_hiera_tiny"
SAM21_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM21_IMPLEMENTATION_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


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


def crossed_shot_boundary(
    sample_index: int, seed_index: int, boundary_indices: Sequence[int]
) -> bool:
    if sample_index > seed_index:
        return any(seed_index < boundary <= sample_index for boundary in boundary_indices)
    if sample_index < seed_index:
        return any(sample_index < boundary <= seed_index for boundary in boundary_indices)
    return False


def _extract_analysis_frames(
    video_path: Path, frames_dir: Path, analysis_fps: float, max_side: int
) -> tuple[list[Path], int, int]:
    if analysis_fps <= 0 or analysis_fps > 60:
        raise ValueError("analysis_fps must be in (0, 60]")
    if max_side < 320:
        raise ValueError("max_side must be at least 320")
    frames_dir.mkdir(parents=True, exist_ok=True)
    if any(frames_dir.iterdir()):
        raise FileExistsError(f"analysis frame directory is not empty: {frames_dir}")
    filter_graph = (
        f"fps={analysis_fps},"
        f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            filter_graph,
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(frames_dir / "%06d.jpg"),
        ],
        check=True,
    )
    paths = sorted(frames_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    if not paths:
        raise RuntimeError("FFmpeg produced no analysis frames")
    with Image.open(paths[0]) as first:
        width, height = first.size
    return paths, width, height


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
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(output_path)
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
    alpha = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 105, mode="L")
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
    analysis_fps: float = 2.0,
    max_side: int = 960,
    device: str = "auto",
    ffmpeg_scdet_threshold: float = 4.0,
    seed_box_padding_ratio: float = 0.0,
) -> SegmentationTrack:
    """Refine a Gemini/manual bbox into a SAM mask and propagate it in both directions."""
    np, torch, build_sam2_video_predictor = _require_segmentation_dependencies()
    if not 0 <= ffmpeg_scdet_threshold <= 100:
        raise ValueError("ffmpeg_scdet_threshold must be in [0, 100]")
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths, width, height = _extract_analysis_frames(
        video_path, output_dir / "analysis-frames", analysis_fps, max_side
    )
    shot_manifest = detect_shots_ffmpeg(
        video_path,
        threshold=ffmpeg_scdet_threshold,
        output_path=output_dir / "shots.json",
    )
    seed_index = min(
        range(len(frame_paths)),
        key=lambda index: abs(round(index * 1000 / analysis_fps) - seed_time_ms),
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
        offload_state_to_cpu=resolved_device != "cpu",
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
    cut_scores: list[float | None] = [None] * len(frame_paths)
    boundary_pts: dict[int, int] = {}
    for boundary in shot_manifest.boundaries:
        sample_index = min(
            range(len(frame_paths)),
            key=lambda index: abs(round(index * 1000 / analysis_fps) - boundary.frame_time_ms),
        )
        if cut_scores[sample_index] is None or boundary.score > cut_scores[sample_index]:
            cut_scores[sample_index] = boundary.score
            boundary_pts[sample_index] = boundary.frame_pts
    boundary_indices = sorted(boundary_pts)
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    samples: list[SegmentationSample] = []
    previous_areas: list[float] = []
    previous_center: Sequence[float] | None = None
    for index, frame_path in enumerate(frame_paths):
        logits = logits_by_index.get(index)
        binary = np.asarray(logits > 0, dtype=bool) if logits is not None else np.zeros((height, width), bool)
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
        shot_boundary = index in boundary_pts
        comparison_areas = [] if index == seed_index else previous_areas
        comparison_center = None if index == seed_index else previous_center
        state, reasons = classify_tracking_state(
            area_ratio=geometry["area_ratio"],
            connected_components=components,
            mean_positive_probability=mean_probability,
            previous_area_ratios=comparison_areas,
            center_2d=geometry["center_2d"],
            previous_center_2d=comparison_center,
            shot_boundary=shot_boundary,
        )
        outside_seed_shot = crossed_shot_boundary(index, seed_index, boundary_indices)
        if outside_seed_shot and state != TrackingState.LOST:
            state = TrackingState.DRIFT_SUSPECTED
            if "outside_seed_shot_reacquisition_required" not in reasons:
                reasons.append("outside_seed_shot_reacquisition_required")
        semantic_status = (
            SemanticIdentityStatus.SEED_GROUNDED
            if index == seed_index
            else SemanticIdentityStatus.REVALIDATION_REQUIRED
            if outside_seed_shot
            or shot_boundary
            or state in {TrackingState.DRIFT_SUSPECTED, TrackingState.LOST}
            else SemanticIdentityStatus.NOT_REVALIDATED
        )
        sample = SegmentationSample(
            sample_index=index,
            analysis_sample_time_ms=round(index * 1000 / analysis_fps),
            source_pts=boundary_pts.get(index),
            timing_basis="uniform_ffmpeg_analysis_sample",
            mask_path=mask_path,
            mask_sha256=mask_hash,
            mask_area_pixels=geometry["area_pixels"],
            mask_area_ratio=round(geometry["area_ratio"], 8),
            connected_components=components,
            derived_tracking_box=geometry["box_2d"],
            center_2d=geometry["center_2d"],
            mean_positive_probability=(round(mean_probability, 6) if mean_probability is not None else None),
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
        method="gemini_bbox_seed_sam2_video_mask_propagation",
        asset_id=asset_id or f"sha256:{sha256_file(video_path)}",
        video_path=str(video_path.resolve()),
        target_description=target_description,
        seed_source=seed_source,
        seed_time_ms=seed_time_ms,
        seed_sample_index=seed_index,
        semantic_seed_box=list(seed_box_2d),
        sam_prompt_box=sam_prompt_box,
        seed_box_padding_ratio=seed_box_padding_ratio,
        refined_seed_mask_path=str(Path("masks") / f"{seed_index:06d}.png"),
        analysis_fps=analysis_fps,
        analysis_width=width,
        analysis_height=height,
        timing_warning=(
            "analysis_sample_time_ms is a uniform FFmpeg sampling clock, not original frame PTS "
            "and not a frame-accurate edit point"
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
