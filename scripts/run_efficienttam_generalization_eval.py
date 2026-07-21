#!/usr/bin/env python3
"""Run auditable EfficientTAM-Ti MPS tracking over diverse local fixtures.

The script intentionally treats the bbox seeds as proposals, not ground truth. Each
fixture uses one EfficientTAM inference state and adds every target before forward
and reverse propagation, exercising the shared-session multi-object path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_CONFIG = (
    ROOT / "fixtures" / "annotations" / "efficienttam_generalization_v1.json"
)
UPSTREAM: Path
CHECKPOINT: Path
OUTPUT: Path


_SHOWINFO_FRAME_RE = re.compile(
    r"showinfo[^\n]*\bn:\s*(?P<index>\d+)\s+pts:\s*(?P<pts>-?\d+)\s+"
    r"pts_time:(?P<pts_time>-?[0-9.]+)"
)


def probe_source(path: Path) -> dict[str, int]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=start_pts,time_base",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    time_base = Fraction(stream["time_base"])
    return {
        "duration_ms": round(float(payload["format"]["duration"]) * 1000),
        "start_pts": int(stream.get("start_pts") or 0),
        "time_base_numerator": time_base.numerator,
        "time_base_denominator": time_base.denominator,
    }


def timeline_ms_from_pts(pts: int, source: dict[str, int]) -> int:
    return round(
        (pts - source["start_pts"])
        * source["time_base_numerator"]
        * 1000
        / source["time_base_denominator"]
    )


def extract_analysis_frames(
    video_path: Path,
    frames_dir: Path,
    analysis_fps: float,
    max_side: int,
    *,
    start_time_ms: int,
    end_time_ms: int,
    source: dict[str, int],
    required_source_pts: Sequence[int],
) -> tuple[list[dict[str, Any]], int, int]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    if any(frames_dir.iterdir()):
        raise FileExistsError(f"analysis frame directory is not empty: {frames_dir}")
    time_base = Fraction(
        source["time_base_numerator"], source["time_base_denominator"]
    )
    source_origin_seconds = Fraction(source["start_pts"]) * time_base
    absolute_start_seconds = source_origin_seconds + Fraction(start_time_ms, 1000)
    absolute_end_seconds = source_origin_seconds + Fraction(end_time_ms, 1000)
    interval_seconds = Fraction(1, 1) / Fraction(str(analysis_fps))
    start_seconds_text = f"{float(absolute_start_seconds):.9f}"
    interval_seconds_text = f"{float(interval_seconds):.9f}"
    current_bucket = f"floor((t-{start_seconds_text})/{interval_seconds_text})"
    previous_bucket = f"floor(((prev_pts*TB)-{start_seconds_text})/{interval_seconds_text})"
    regular_select = (
        f"gte(t\\,{start_seconds_text})*"
        f"lt(t\\,{float(absolute_end_seconds):.9f})*"
        f"(isnan(prev_pts)+gt({current_bucket}\\,{previous_bucket}))"
    )
    forced_select = "+".join(
        f"eq(pts\\,{pts})" for pts in sorted(required_source_pts)
    )
    select = f"({regular_select})+({forced_select})" if forced_select else regular_select
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
            (
                f"select={select},"
                f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease,"
                "showinfo"
            ),
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
        raise RuntimeError(f"FFmpeg extraction failed: {completed.stderr.strip()}")
    paths = sorted(frames_dir.glob("*.jpg"), key=lambda item: int(item.stem))
    matches = list(_SHOWINFO_FRAME_RE.finditer(completed.stderr))
    if len(matches) != len(paths):
        raise RuntimeError(
            f"frame lineage mismatch: {len(paths)} files, {len(matches)} PTS records"
        )
    frames: list[dict[str, Any]] = []
    for index, (path, match) in enumerate(zip(paths, matches, strict=True)):
        if int(match.group("index")) != index:
            raise RuntimeError("FFmpeg frame indices are not contiguous")
        pts = int(match.group("pts"))
        frames.append(
            {
                "path": path,
                "source_pts": pts,
                "timeline_time_ms": timeline_ms_from_pts(pts, source),
            }
        )
    emitted = {item["source_pts"] for item in frames}
    missing = sorted(set(required_source_pts) - emitted)
    if missing:
        raise RuntimeError(f"required source PTS not emitted: {missing}")
    with Image.open(paths[0]) as first:
        width, height = first.size
    return frames, width, height


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame-manifest",
        type=Path,
        required=True,
        help="Exact-frame fixture manifest produced by the local preparation step.",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        required=True,
        help="Directory containing upstream/ and checkpoints/efficienttam_ti.pt.",
    )
    parser.add_argument(
        "--fixture-config",
        type=Path,
        default=DEFAULT_FIXTURE_CONFIG,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_to_pixels(box: list[int], width: int, height: int) -> list[float]:
    return [
        box[0] * width / 1000,
        box[1] * height / 1000,
        box[2] * width / 1000,
        box[3] * height / 1000,
    ]


def mask_stats(mask: np.ndarray) -> dict[str, Any]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "area_pixels": 0,
            "area_ratio": 0.0,
            "box_2d": None,
            "center_2d": None,
        }
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    return {
        "area_pixels": int(mask.sum()),
        "area_ratio": round(float(mask.mean()), 9),
        "box_2d": [
            round(x0 * 1000 / width, 3),
            round(y0 * 1000 / height, 3),
            round(x1 * 1000 / width, 3),
            round(y1 * 1000 / height, 3),
        ],
        "center_2d": [
            round(float(xs.mean()) * 1000 / width, 3),
            round(float(ys.mean()) * 1000 / height, 3),
        ],
    }


def encode_review(review_frames: Path, frame_count: int, fps: float, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            str(review_frames / "%06d.jpg"),
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-vf",
            "scale=in_range=full:out_range=tv,format=yuv420p",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def render_review(
    frame_paths: list[Path],
    masks: dict[int, dict[int, np.ndarray]],
    targets: list[dict[str, Any]],
    review_dir: Path,
    fixture_id: str,
) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    count = len(frame_paths)
    for index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"could not read analysis frame: {frame_path}")
        overlay = frame.copy()
        for object_id, target in enumerate(targets, start=1):
            mask = masks[object_id][index]
            color = tuple(int(value) for value in target["color_bgr"])
            overlay[mask] = color
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, color, 2, cv2.LINE_AA)
            geometry = mask_stats(mask)
            if geometry["box_2d"] is not None:
                height, width = mask.shape
                x0, y0, x1, y1 = canonical_to_pixels(geometry["box_2d"], width, height)
                cv2.rectangle(frame, (round(x0), round(y0)), (round(x1), round(y1)), color, 2)
        frame = cv2.addWeighted(frame, 0.66, overlay, 0.34, 0)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
        title = f"EfficientTAM-Ti MPS | {fixture_id} | shared session | {index + 1}/{count}"
        cv2.putText(
            frame,
            title,
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(
            str(review_dir / f"{index:06d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 91],
        )


def run_fixture(
    predictor: Any,
    fixture: dict[str, Any],
    targets: list[dict[str, Any]],
    *,
    fps: float,
    max_side: int,
) -> dict[str, Any]:
    fixture_id = fixture["fixture_id"]
    fixture_dir = OUTPUT / fixture_id
    frames_dir = fixture_dir / "analysis-frames"
    review_dir = fixture_dir / "review-frames"
    masks_dir = fixture_dir / "masks"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    source = Path(fixture["source_path"])
    media = probe_source(source)
    start_ms = int(fixture["event_start_ms"])
    end_ms = min(int(fixture["event_end_ms"]), media["duration_ms"])
    seed_pts = int(fixture["frame"]["frame_pts"])

    if frames_dir.exists() and any(frames_dir.iterdir()):
        for path in frames_dir.glob("*.jpg"):
            path.unlink()
    frames, width, height = extract_analysis_frames(
        source,
        frames_dir,
        fps,
        max_side,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        source=media,
        required_source_pts=[seed_pts],
    )
    seed_index = next(
        index for index, frame in enumerate(frames) if frame["source_pts"] == seed_pts
    )
    frame_paths = [frame["path"] for frame in frames]

    phase_seconds: dict[str, float] = {}
    started = time.perf_counter()
    t0 = time.perf_counter()
    state = predictor.init_state(
        str(frames_dir),
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    torch.mps.synchronize()
    phase_seconds["state_init"] = time.perf_counter() - t0

    prompt_results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for object_id, target in enumerate(targets, start=1):
        out_frame, out_ids, logits = predictor.add_new_points_or_box(
            state,
            frame_idx=seed_index,
            obj_id=object_id,
            box=np.asarray(canonical_to_pixels(target["box_2d"], width, height), dtype=np.float32),
        )
        prompt_results.append(
            {
                "object_id": object_id,
                "target_id": target["target_id"],
                "output_frame": int(out_frame),
                "visible_object_ids": [int(value) for value in out_ids],
                "logit_shape": list(logits.shape),
            }
        )
    torch.mps.synchronize()
    phase_seconds["all_bbox_prompts"] = time.perf_counter() - t0

    masks: dict[int, dict[int, np.ndarray]] = {
        object_id: {} for object_id in range(1, len(targets) + 1)
    }
    directions: dict[int, str] = {}
    yields = 0
    t0 = time.perf_counter()
    for frame_index, object_ids, logits in predictor.propagate_in_video(
        state, start_frame_idx=seed_index, reverse=False
    ):
        for position, object_id in enumerate(object_ids):
            masks[int(object_id)][int(frame_index)] = (
                logits[position, 0] > 0
            ).detach().cpu().numpy()
        directions[int(frame_index)] = "seed" if frame_index == seed_index else "forward"
        yields += 1
    torch.mps.synchronize()
    phase_seconds["forward"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for frame_index, object_ids, logits in predictor.propagate_in_video(
        state, start_frame_idx=seed_index, reverse=True
    ):
        for position, object_id in enumerate(object_ids):
            masks[int(object_id)][int(frame_index)] = (
                logits[position, 0] > 0
            ).detach().cpu().numpy()
        directions[int(frame_index)] = "seed" if frame_index == seed_index else "reverse"
        yields += 1
    torch.mps.synchronize()
    phase_seconds["reverse"] = time.perf_counter() - t0

    expected = list(range(len(frames)))
    for object_id in masks:
        if sorted(masks[object_id]) != expected:
            missing = sorted(set(expected) - set(masks[object_id]))
            raise RuntimeError(f"{fixture_id} object {object_id} missing frames: {missing[:10]}")

    t0 = time.perf_counter()
    tracks: list[dict[str, Any]] = []
    for object_id, target in enumerate(targets, start=1):
        target_mask_dir = masks_dir / target["target_id"]
        target_mask_dir.mkdir(parents=True, exist_ok=True)
        samples: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            mask = masks[object_id][index]
            mask_path = target_mask_dir / f"{index:06d}.png"
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path, optimize=True)
            samples.append(
                {
                    "sample_index": index,
                    "source_pts": frame["source_pts"],
                    "timeline_time_ms": frame["timeline_time_ms"],
                    "direction": directions[index],
                    "mask_path": str(mask_path.relative_to(fixture_dir)),
                    "mask_sha256": sha256(mask_path),
                    **mask_stats(mask),
                }
            )
        track = {
            "target_id": target["target_id"],
            "label": target["label"],
            "seed_source": "manual_bbox_proposal_pending_human_review",
            "seed_box_2d": target["box_2d"],
            "seed_sample_index": seed_index,
            "samples": samples,
        }
        track_path = fixture_dir / f"track-{target['target_id']}.json"
        write_json(track_path, track)
        tracks.append({"target_id": target["target_id"], "path": str(track_path), "track": track})
    phase_seconds["mask_serialization"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    render_review(frame_paths, masks, targets, review_dir, fixture_id)
    video_path = fixture_dir / "review.mp4"
    encode_review(review_dir, len(frames), fps, video_path)
    phase_seconds["review_render_and_encode"] = time.perf_counter() - t0
    phase_seconds["total"] = time.perf_counter() - started

    summary_targets = []
    for item in tracks:
        samples = item["track"]["samples"]
        areas = np.asarray([sample["area_ratio"] for sample in samples], dtype=float)
        seed_area = float(samples[seed_index]["area_ratio"])
        summary_targets.append(
            {
                "target_id": item["target_id"],
                "empty_masks": int(np.count_nonzero(areas == 0)),
                "min_area_ratio": round(float(areas.min()), 9),
                "max_area_ratio": round(float(areas.max()), 9),
                "seed_area_ratio": round(seed_area, 9),
                "max_area_over_seed": None if seed_area == 0 else round(float(areas.max() / seed_area), 4),
                "min_area_over_seed": None if seed_area == 0 else round(float(areas.min() / seed_area), 4),
            }
        )

    result = {
        "fixture_id": fixture_id,
        "status": "tracking_complete_pending_human_review",
        "event_interval_ms": [start_ms, end_ms],
        "analysis": {
            "fps": fps,
            "max_side": max_side,
            "width": width,
            "height": height,
            "frame_count": len(frames),
            "seed_index": seed_index,
            "seed_source_pts": seed_pts,
            "seed_frame_hash": fixture["frame"]["frame_hash"],
        },
        "shared_session": {
            "predictor_instances_for_fixture": 1,
            "inference_states_for_fixture": 1,
            "target_count": len(targets),
            "prompt_results": prompt_results,
            "forward_and_reverse_yields": yields,
        },
        "target_summaries": summary_targets,
        "timing_seconds": phase_seconds,
        "review_video": str(video_path),
    }
    write_json(fixture_dir / "result.json", result)

    predictor.reset_state(state)
    del state, masks
    torch.mps.empty_cache()
    return result


def main() -> None:
    global CHECKPOINT, OUTPUT, UPSTREAM

    args = parse_args()
    frame_manifest_path = args.frame_manifest.expanduser().resolve(strict=True)
    fixture_config_path = args.fixture_config.expanduser().resolve(strict=True)
    runtime_root = args.runtime_root.expanduser().resolve(strict=True)
    UPSTREAM = runtime_root / "upstream"
    CHECKPOINT = runtime_root / "checkpoints" / "efficienttam_ti.pt"
    OUTPUT = args.output.expanduser().resolve()
    sys.path.insert(0, str(UPSTREAM))

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be unset or 0")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    frame_manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    fixture_config = json.loads(fixture_config_path.read_text(encoding="utf-8"))
    frames_by_id = {item["fixture_id"]: item for item in frame_manifest["fixtures"]}
    configs = {item["fixture_id"]: item for item in fixture_config["fixtures"]}
    if set(frames_by_id) != set(configs):
        raise RuntimeError("fixture IDs differ between exact-frame manifest and seed config")

    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

    OUTPUT.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    warnings_seen: list[str] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        predictor = build_efficienttam_video_predictor(
            "configs/efficienttam/efficienttam_ti.yaml",
            str(CHECKPOINT),
            device="mps",
            apply_postprocessing=True,
        )
        torch.mps.synchronize()
        model_load_seconds = time.perf_counter() - t0
        results = []
        for index, fixture_id in enumerate(configs, start=1):
            print(f"[{index}/{len(configs)}] {fixture_id}", flush=True)
            result = run_fixture(
                predictor,
                frames_by_id[fixture_id],
                configs[fixture_id]["targets"],
                fps=float(fixture_config["analysis_fps"]),
                max_side=int(fixture_config["analysis_max_side"]),
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "fixture_id": fixture_id,
                        "frames": result["analysis"]["frame_count"],
                        "targets": result["shared_session"]["target_count"],
                        "seconds": round(result["timing_seconds"]["total"], 3),
                    }
                ),
                flush=True,
            )
        warnings_seen = sorted(
            {f"{type(item.message).__name__}: {item.message}" for item in records}
        )

    report = {
        "status": "tracking_complete_pending_human_review",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "model": "EfficientTAM-Ti",
            "device": "mps",
            "prompt": f"{fixture_config.get('seed_source', 'unspecified_bbox_source')}; box-only",
            "session_contract": "one predictor and one inference state per fixture; all targets added before propagation",
            "propagation": "forward and reverse from the exact seed frame",
            "model_load_seconds": model_load_seconds,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "warnings": warnings_seen,
        },
        "provenance": {
            "checkpoint_sha256": sha256(CHECKPOINT),
            "fixture_config_sha256": sha256(fixture_config_path),
            "exact_frame_manifest_sha256": sha256(frame_manifest_path),
        },
        "fixtures": results,
    }
    write_json(OUTPUT / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
