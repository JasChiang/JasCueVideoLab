from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence


def normalized_box_to_xywh(
    box_2d: Sequence[int], width: int, height: int
) -> tuple[float, float, float, float]:
    if len(box_2d) != 4:
        raise ValueError("box_2d must contain four x-first normalized coordinates")
    x_min, y_min, x_max, y_max = box_2d
    if not (0 <= x_min < x_max <= 1000 and 0 <= y_min < y_max <= 1000):
        raise ValueError("box_2d must be canonical [xmin,ymin,xmax,ymax] within 0..1000")
    x = x_min * width / 1000
    y = y_min * height / 1000
    return x, y, (x_max - x_min) * width / 1000, (y_max - y_min) * height / 1000


def xywh_to_normalized_box(
    xywh: Sequence[float], width: int, height: int
) -> list[int]:
    x, y, box_width, box_height = xywh
    x_min = max(0, min(999, round(x * 1000 / width)))
    y_min = max(0, min(999, round(y * 1000 / height)))
    x_max = max(x_min + 1, min(1000, round((x + box_width) * 1000 / width)))
    y_max = max(y_min + 1, min(1000, round((y + box_height) * 1000 / height)))
    return [x_min, y_min, x_max, y_max]


def _require_cv2():
    try:
        import cv2
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Dynamic tracking requires: uv sync --extra tracking") from error
    return cv2


def _clamped_rect(
    box: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x, y, box_width, box_height = box
    x1 = max(0, min(width - 1, round(x)))
    y1 = max(0, min(height - 1, round(y)))
    x2 = max(x1 + 1, min(width, round(x + box_width)))
    y2 = max(y1 + 1, min(height, round(y + box_height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _appearance_histogram(frame: Any, box: Sequence[float], cv2: Any) -> Any | None:
    rect = _clamped_rect(box, frame.shape[1], frame.shape[0])
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).flatten()


def _appearance_similarity(reference: Any, frame: Any, box: Sequence[float], cv2: Any) -> float:
    current = _appearance_histogram(frame, box, cv2)
    if reference is None or current is None:
        return 0.0
    distance = float(cv2.compareHist(reference, current, cv2.HISTCMP_BHATTACHARYYA))
    return max(0.0, min(1.0, 1.0 - distance))


def _sample_video(
    video_path: Path,
    frames_dir: Path,
    analysis_fps: float,
    max_side: int,
    cv2: Any,
) -> tuple[list[dict[str, Any]], int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("OpenCV reported invalid video dimensions")
    scale = min(1.0, max_side / max(source_width, source_height))
    tracking_width = max(1, round(source_width * scale))
    tracking_height = max(1, round(source_height * scale))
    sample_interval_ms = 1000.0 / analysis_fps
    next_sample_ms = 0.0
    decoded_frame_index = 0
    samples: list[dict[str, Any]] = []
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_time_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if decoded_time_ms + 0.5 < next_sample_ms:
                decoded_frame_index += 1
                continue
            if (frame.shape[1], frame.shape[0]) != (tracking_width, tracking_height):
                frame = cv2.resize(
                    frame, (tracking_width, tracking_height), interpolation=cv2.INTER_AREA
                )
            sample_index = len(samples)
            frame_path = frames_dir / f"f{sample_index:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f"failed to save sampled frame {frame_path}")
            samples.append(
                {
                    "sample_index": sample_index,
                    "decoded_frame_index": decoded_frame_index,
                    "decoded_time_ms": round(decoded_time_ms, 3),
                    "frame_path": str(frame_path),
                }
            )
            next_sample_ms += sample_interval_ms
            decoded_frame_index += 1
    finally:
        capture.release()
    if not samples:
        raise RuntimeError("no frames were decoded")
    return samples, tracking_width, tracking_height


def _track_direction(
    samples: list[dict[str, Any]],
    seed_index: int,
    seed_box: tuple[float, float, float, float],
    direction: int,
    appearance_threshold: float,
    cv2: Any,
) -> dict[int, dict[str, Any]]:
    seed_frame = cv2.imread(samples[seed_index]["frame_path"])
    if seed_frame is None:
        raise RuntimeError("could not read tracker seed frame")
    tracker = cv2.TrackerCSRT.create()
    tracker_seed = tuple(round(value) for value in seed_box)
    tracker.init(seed_frame, tracker_seed)
    reference_histogram = _appearance_histogram(seed_frame, seed_box, cv2)
    output: dict[int, dict[str, Any]] = {}
    index = seed_index + direction
    while 0 <= index < len(samples):
        frame = cv2.imread(samples[index]["frame_path"])
        if frame is None:
            break
        ok, box = tracker.update(frame)
        similarity = _appearance_similarity(reference_histogram, frame, box, cv2) if ok else 0.0
        accepted = bool(ok and similarity >= appearance_threshold)
        output[index] = {
            "tracker_success": bool(ok),
            "appearance_similarity": round(similarity, 6),
            "accepted": accepted,
            "tracking_box_xywh": [round(float(value), 3) for value in box] if ok else None,
        }
        if not accepted:
            break
        index += direction
    return output


def _render_tracking_video(
    samples: list[dict[str, Any]],
    records: dict[int, dict[str, Any]],
    output_path: Path,
    analysis_fps: float,
    width: int,
    height: int,
    label: str,
    cv2: Any,
) -> None:
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), analysis_fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the MP4 debug writer")
    try:
        for sample in samples:
            frame = cv2.imread(sample["frame_path"])
            record = records.get(sample["sample_index"])
            if record and record.get("tracking_box_xywh"):
                rect = _clamped_rect(record["tracking_box_xywh"], width, height)
                if rect:
                    x1, y1, x2, y2 = rect
                    color = (50, 220, 50) if record["accepted"] else (30, 30, 230)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(
                        frame,
                        f"{label} sim={record['appearance_similarity']:.2f}",
                        (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                frame,
                f"decoded {sample['decoded_time_ms'] / 1000:.3f}s",
                (16, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()


def _render_tracking_html(output_dir: Path, result: dict[str, Any]) -> Path:
    output = output_dir / "index.html"
    label = html.escape(str(result["target_description"]))
    summary = json.dumps(
        {
            "method": result["method"],
            "seed_source": result["seed_source"],
            "seed_time_ms": result["seed_time_ms"],
            "accepted_samples": result["accepted_samples"],
            "total_samples": result["total_samples"],
            "coverage": result["coverage"],
            "warning": result["warning"],
        },
        ensure_ascii=False,
        indent=2,
    )
    output.write_text(
        f"""<!doctype html>
<html lang="zh-Hant"><meta charset="utf-8"><title>Dynamic tracking debug</title>
<style>body{{font:16px system-ui;background:#111;color:#eee;max-width:1100px;margin:30px auto;padding:0 20px}}video{{width:100%;background:#000}}pre{{white-space:pre-wrap;background:#222;padding:16px;border-radius:8px}}.warn{{color:#ffd166}}</style>
<h1>Dynamic tracking debug</h1><p>Target: {label}</p>
<p class="warn">這是 CSRT proposal，不是 Gemini 原生 tracking，也不是 production SpatialTrack。</p>
<video controls src="tracking-debug.mp4"></video><pre>{html.escape(summary)}</pre>
</html>""",
        encoding="utf-8",
    )
    return output


def track_bbox_csrt(
    *,
    video_path: Path,
    seed_time_ms: int,
    seed_box_2d: Sequence[int],
    target_description: str,
    output_dir: Path,
    seed_source: str,
    analysis_fps: float = 15.0,
    max_side: int = 960,
    appearance_threshold: float = 0.25,
) -> dict[str, Any]:
    if analysis_fps <= 0 or analysis_fps > 60:
        raise ValueError("analysis_fps must be in (0, 60]")
    if max_side < 320:
        raise ValueError("max_side must be at least 320")
    if not 0 <= appearance_threshold <= 1:
        raise ValueError("appearance_threshold must be in [0, 1]")
    cv2 = _require_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples, width, height = _sample_video(
        video_path, output_dir / "sampled-frames", analysis_fps, max_side, cv2
    )
    seed_index = min(
        range(len(samples)), key=lambda index: abs(samples[index]["decoded_time_ms"] - seed_time_ms)
    )
    seed_box = normalized_box_to_xywh(seed_box_2d, width, height)
    records = {
        seed_index: {
            "tracker_success": True,
            "appearance_similarity": 1.0,
            "accepted": True,
            "tracking_box_xywh": [round(value, 3) for value in seed_box],
        }
    }
    records.update(_track_direction(samples, seed_index, seed_box, 1, appearance_threshold, cv2))
    records.update(_track_direction(samples, seed_index, seed_box, -1, appearance_threshold, cv2))
    sample_rows: list[dict[str, Any]] = []
    for sample in samples:
        index = sample["sample_index"]
        record = records.get(index)
        row = dict(sample)
        row.pop("frame_path", None)
        if record:
            row.update(record)
            if record.get("tracking_box_xywh"):
                row["box_2d"] = xywh_to_normalized_box(record["tracking_box_xywh"], width, height)
        else:
            row.update(
                {
                    "tracker_success": False,
                    "appearance_similarity": None,
                    "accepted": False,
                    "tracking_box_xywh": None,
                    "box_2d": None,
                }
            )
        sample_rows.append(row)
    accepted = sum(bool(row["accepted"]) for row in sample_rows)
    result = {
        "method": "OpenCV TrackerCSRT seeded by one canonical x-first bbox",
        "seed_source": seed_source,
        "video_path": str(video_path.resolve()),
        "target_description": target_description,
        "seed_time_ms": seed_time_ms,
        "seed_sample_time_ms": samples[seed_index]["decoded_time_ms"],
        "seed_box_2d": list(seed_box_2d),
        "analysis_fps": analysis_fps,
        "tracking_width": width,
        "tracking_height": height,
        "appearance_threshold": appearance_threshold,
        "total_samples": len(samples),
        "accepted_samples": accepted,
        "coverage": round(accepted / len(samples), 6),
        "warning": (
            "Tracker times are decoder sample times, not original frame PTS. CSRT boxes are experimental "
            "propagation proposals and require periodic Gemini or human revalidation."
        ),
        "samples": sample_rows,
    }
    (output_dir / "tracking.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _render_tracking_video(
        samples,
        records,
        output_dir / "tracking-debug.mp4",
        analysis_fps,
        width,
        height,
        target_description,
        cv2,
    )
    _render_tracking_html(output_dir, result)
    return result
