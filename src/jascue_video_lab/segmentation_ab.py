from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageDraw

from .gemini import GeminiLabClient
from .geometry import box_iou, native_yxyx_to_canonical_xyxy
from .models import ExtractedFrame, GroundingProposal, SegmentationTrack
from .overlay import _overlay_font
from .sam_tracking import track_bbox_sam21
from .storage import read_json, write_json


def _draw_gemini_segmentation(
    frame_path: Path,
    candidate: Any,
    output_path: Path,
) -> None:
    with Image.open(frame_path).convert("RGBA") as source:
        image = source.copy()
    width, height = image.size
    polygon = [
        (round(x * width / 1000), round(y * height / 1000))
        for x, y in candidate.mask
    ]
    fill = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill)
    fill_draw.polygon(polygon, fill=(0, 220, 150, 95), outline=(0, 255, 180, 255), width=8)
    image = Image.alpha_composite(image, fill)
    draw = ImageDraw.Draw(image)
    y_min, x_min, y_max, x_max = candidate.box_2d_yxyx
    box = (
        round(x_min * width / 1000),
        round(y_min * height / 1000),
        round(x_max * width / 1000),
        round(y_max * height / 1000),
    )
    line_width = max(4, round(min(image.size) / 240))
    draw.rectangle(box, outline="#ff3155", width=line_width)
    font = _overlay_font(max(18, round(min(image.size) / 45)))
    label = f"Gemini polygon | {candidate.label} | {candidate.confidence:.2f}"
    text_box = draw.textbbox((box[0], box[1]), label, font=font)
    draw.rectangle(text_box, fill="#ff3155")
    draw.text((box[0], box[1]), label, fill="#101010", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=94)


def _mask_iou(left_path: Path, right_path: Path) -> float:
    import numpy as np

    with Image.open(left_path).convert("1") as left_image:
        left = np.asarray(left_image, dtype=bool)
    with Image.open(right_path).convert("1") as right_image:
        right = np.asarray(right_image, dtype=bool)
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    return intersection / union if union else 1.0


def _compare_tracks(
    bbox_track: SegmentationTrack,
    polygon_track: SegmentationTrack,
    bbox_root: Path,
    polygon_root: Path,
) -> dict[str, Any]:
    if bbox_track.total_samples != polygon_track.total_samples:
        raise ValueError("A/B tracks must have the same number of samples")
    rows: list[dict[str, Any]] = []
    for bbox_sample, polygon_sample in zip(bbox_track.samples, polygon_track.samples, strict=True):
        if bbox_sample.sample_index != polygon_sample.sample_index:
            raise ValueError("A/B sample indices differ")
        mask_iou = None
        if bbox_sample.mask_path and polygon_sample.mask_path:
            mask_iou = _mask_iou(
                bbox_root / bbox_sample.mask_path,
                polygon_root / polygon_sample.mask_path,
            )
        bbox_agreement = None
        if bbox_sample.derived_tracking_box and polygon_sample.derived_tracking_box:
            bbox_agreement = box_iou(
                bbox_sample.derived_tracking_box,
                polygon_sample.derived_tracking_box,
            )
        rows.append(
            {
                "sample_index": bbox_sample.sample_index,
                "analysis_sample_time_ms": bbox_sample.analysis_sample_time_ms,
                "mask_iou": round(mask_iou, 6) if mask_iou is not None else None,
                "bbox_iou": round(bbox_agreement, 6) if bbox_agreement is not None else None,
                "bbox_seed_area_ratio": bbox_sample.mask_area_ratio,
                "polygon_seed_area_ratio": polygon_sample.mask_area_ratio,
                "bbox_seed_state": bbox_sample.tracking_state,
                "polygon_seed_state": polygon_sample.tracking_state,
            }
        )
    mask_ious = [row["mask_iou"] for row in rows if row["mask_iou"] is not None]
    bbox_ious = [row["bbox_iou"] for row in rows if row["bbox_iou"] is not None]
    seed_index = bbox_track.seed_sample_index
    return {
        "interpretation": (
            "A/B agreement is not accuracy. Independent human or pixel ground truth is still required."
        ),
        "total_samples": len(rows),
        "seed_sample_index": seed_index,
        "seed_mask_iou": rows[seed_index]["mask_iou"],
        "mean_mask_iou": round(sum(mask_ious) / len(mask_ious), 6),
        "min_mask_iou": min(mask_ious),
        "mean_derived_bbox_iou": round(sum(bbox_ious) / len(bbox_ious), 6),
        "min_derived_bbox_iou": min(bbox_ious),
        "bbox_seed_state_counts": bbox_track.state_counts,
        "polygon_seed_state_counts": polygon_track.state_counts,
        "samples": rows,
    }


def _render_review_sheet(
    bbox_root: Path,
    polygon_root: Path,
    track: SegmentationTrack,
    output_path: Path,
) -> None:
    indices = sorted({0, track.seed_sample_index, track.total_samples // 2, track.total_samples - 1})
    cell_width, cell_height = 640, 360
    header_height = 42
    canvas = Image.new("RGB", (cell_width * 2, (cell_height + header_height) * len(indices)), "#111")
    draw = ImageDraw.Draw(canvas)
    font = _overlay_font(24)
    for row_index, sample_index in enumerate(indices):
        y = row_index * (cell_height + header_height)
        draw.text((12, y + 8), f"bbox seed | sample {sample_index}", fill="white", font=font)
        draw.text(
            (cell_width + 12, y + 8),
            f"Gemini polygon seed | sample {sample_index}",
            fill="white",
            font=font,
        )
        for column, root in enumerate((bbox_root, polygon_root)):
            with Image.open(root / "overlays" / f"{sample_index:06d}.jpg").convert("RGB") as frame:
                fitted = frame.resize((cell_width, cell_height))
            canvas.paste(fitted, (column * cell_width, y + header_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def run_segmentation_seed_ab(
    *,
    case_dir: Path,
    checkpoint_path: Path,
    target_description: str,
    event_description: str,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    grounding_dir = case_dir / "grounding"
    grounding = GroundingProposal.model_validate(read_json(grounding_dir / "grounding.json"))
    baseline = SegmentationTrack.model_validate(read_json(case_dir / "sam21" / "segmentation-track.json"))
    frame_path = grounding_dir / "frame.png"
    frame = ExtractedFrame(
        path=str(frame_path),
        requested_time_ms=grounding.frame_time_ms,
        frame_time_ms=grounding.frame_time_ms,
        frame_pts=grounding.frame_pts,
        frame_hash=grounding.frame_hash,
        width=grounding.source_width,
        height=grounding.source_height,
    )
    media = SimpleNamespace(asset_id=grounding.asset_id)
    client = GeminiLabClient()
    started = monotonic()
    try:
        segmentation = client.segment_frame(
            media=media,
            frame=frame,
            event_id=grounding.event_id,
            event_description=event_description,
            entity_id=grounding.entity_id,
            target_description=target_description,
            run_id=run_id,
            output_dir=output_dir / "gemini",
        )
    finally:
        client.close()
    gemini_seconds = monotonic() - started
    if not segmentation.visible or not segmentation.candidates:
        raise ValueError("Gemini segmentation target was not visible")
    candidate = segmentation.candidates[0]
    _draw_gemini_segmentation(
        frame_path,
        candidate,
        output_dir / "gemini" / "segmentation-debug.png",
    )
    canonical_box = native_yxyx_to_canonical_xyxy(candidate.box_2d_yxyx)
    common = {
        "video_path": case_dir / "tracking-source.mp4",
        "checkpoint_path": checkpoint_path,
        "seed_time_ms": baseline.seed_time_ms,
        "seed_box_2d": canonical_box,
        "target_description": target_description,
        "asset_id": grounding.asset_id,
        "analysis_fps": baseline.analysis_fps,
        "max_side": max(baseline.analysis_width, baseline.analysis_height),
        "device": "auto",
        "ffmpeg_scdet_threshold": 4.0,
        "seed_box_padding_ratio": 0.0,
    }
    bbox_root = output_dir / "sam-bbox-seed"
    polygon_root = output_dir / "sam-polygon-seed"
    bbox_track = track_bbox_sam21(
        **common,
        output_dir=bbox_root,
        seed_source="gemini_segmentation_response_bbox",
    )
    polygon_track = track_bbox_sam21(
        **common,
        output_dir=polygon_root,
        seed_source="gemini_segmentation_response_polygon",
        seed_mask_polygon_xy=candidate.mask,
    )
    comparison = _compare_tracks(bbox_track, polygon_track, bbox_root, polygon_root)
    write_json(output_dir / "comparison.json", comparison)
    timing = {
        "gemini_segmentation_seconds": round(gemini_seconds, 3),
        "sam_bbox_seed_seconds": bbox_track.elapsed_seconds,
        "sam_polygon_seed_seconds": polygon_track.elapsed_seconds,
    }
    write_json(output_dir / "timing.json", timing)
    _render_review_sheet(bbox_root, polygon_root, polygon_track, output_dir / "review-sheet.jpg")
    document = f"""<!doctype html><html lang=\"zh-Hant\"><meta charset=\"utf-8\">
<title>Gemini segmentation seed A/B</title>
<style>body{{font:16px system-ui;background:#111;color:#eee;max-width:1200px;margin:auto;padding:24px}}img,video{{max-width:100%}}code{{color:#7ee0b8}}section{{margin:28px 0}}</style>
<h1>{html.escape(grounding.event_id)}：Gemini segmentation seed A/B</h1>
<p>目標：{html.escape(target_description)}</p>
<p><strong>注意：</strong>A/B agreement 不是 accuracy，仍需人工或 pixel ground truth。</p>
<section><h2>Gemini 單幀 polygon</h2><img src=\"gemini/segmentation-debug.png\"></section>
<section><h2>固定樣本比較</h2><img src=\"review-sheet.jpg\"></section>
<section><h2>bbox seed</h2><video controls src=\"sam-bbox-seed/segmentation-debug.mp4\"></video></section>
<section><h2>Gemini polygon seed</h2><video controls src=\"sam-polygon-seed/segmentation-debug.mp4\"></video></section>
<pre>{html.escape(json.dumps(comparison, ensure_ascii=False, indent=2))}</pre>
</html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")
    return {"comparison": comparison, "timing": timing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_segmentation_seed_ab(
        case_dir=args.case_dir,
        checkpoint_path=args.checkpoint,
        target_description=args.target,
        event_description=args.event,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
