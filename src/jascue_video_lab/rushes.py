from __future__ import annotations

import hashlib
import html
import json
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path
from time import monotonic
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .billing import summarize_usage_and_list_price
from .gemini import GeminiLabClient
from .media import probe_video, sha256_file
from .models import RushClip, RushFrame, RushesCatalog, RushesEditPlan
from .shots import ShotManifest, detect_shots_ffmpeg
from .storage import read_json, utc_now, write_json


def _probe_clip(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration,size:stream=width,height,r_frame_rate",
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
    return {
        "duration_ms": round(float(payload["format"]["duration"]) * 1000),
        "size_bytes": int(payload["format"]["size"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream["r_frame_rate"],
    }


def _format_mmss(milliseconds: int) -> str:
    seconds = max(0, milliseconds // 1000)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _catalog_fingerprint(clips: list[RushClip], interval_ms: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"interval_ms={interval_ms}\n".encode())
    for clip in clips:
        digest.update(
            f"{clip.clip_id}|{clip.sha256}|{clip.duration_ms}|{clip.width}x{clip.height}\n".encode()
        )
    return f"sha256:{digest.hexdigest()}"


def _label_frame(source_path: Path, output_path: Path, label: str) -> None:
    with Image.open(source_path).convert("RGB") as source:
        image = source.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(18, image.width // 30))
    height = max(48, image.height // 8)
    draw.rectangle((0, 0, image.width, height), fill="#080b10")
    draw.text((14, 10), label, fill="#ffffff", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def _render_contact_sheets(frame_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns, rows = 4, 4
    cell_width, cell_height = 320, 180
    page_size = columns * rows
    for page_index, start in enumerate(range(0, len(frame_paths), page_size), start=1):
        canvas = Image.new("RGB", (cell_width * columns, cell_height * rows), "#101418")
        for local_index, path in enumerate(frame_paths[start : start + page_size]):
            with Image.open(path).convert("RGB") as source:
                frame = source.resize((cell_width, cell_height))
            x = (local_index % columns) * cell_width
            y = (local_index // columns) * cell_height
            canvas.paste(frame, (x, y))
        canvas.save(output_dir / f"page-{page_index:03d}.jpg", quality=88)


def create_rushes_catalog(
    source_directory: Path,
    output_dir: Path,
    *,
    sample_interval_ms: int = 2000,
    max_width: int = 640,
) -> RushesCatalog:
    if sample_interval_ms < 500:
        raise ValueError("sample_interval_ms must be at least 500")
    sources = sorted(
        path
        for path in source_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".m4v"}
    )
    if not sources:
        raise ValueError(f"no video files found in {source_directory}")
    output_dir.mkdir(parents=True, exist_ok=True)
    clips: list[RushClip] = []
    for path in sources:
        metadata = _probe_clip(path)
        clips.append(
            RushClip(
                clip_id=path.stem,
                path=str(path.resolve()),
                sha256=sha256_file(path),
                **metadata,
            )
        )
    frames_dir = output_dir / "catalog-frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames: list[RushFrame] = []
    frame_paths: list[Path] = []
    next_frame_number = 1
    for clip in clips:
        raw_dir = output_dir / "catalog-raw" / clip.clip_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        fps_value = 1000 / sample_interval_ms
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                clip.path,
                "-vf",
                f"fps={fps_value},scale={max_width}:-2",
                "-q:v",
                "2",
                "-start_number",
                "0",
                str(raw_dir / "%06d.jpg"),
            ],
            check=True,
        )
        raw_paths = sorted(raw_dir.glob("*.jpg"), key=lambda path: int(path.stem))
        for local_index, raw_path in enumerate(raw_paths):
            requested_time_ms = local_index * sample_interval_ms
            if requested_time_ms >= clip.duration_ms:
                continue
            frame_id = f"RF{next_frame_number:06d}"
            output_path = frames_dir / f"{frame_id}.jpg"
            _label_frame(
                raw_path,
                output_path,
                f"{frame_id}  |  {clip.clip_id}  |  {_format_mmss(requested_time_ms)}",
            )
            frames.append(
                RushFrame(
                    frame_id=frame_id,
                    clip_id=clip.clip_id,
                    requested_time_ms=requested_time_ms,
                    image_path=str(Path("catalog-frames") / output_path.name),
                )
            )
            frame_paths.append(output_path)
            next_frame_number += 1
    if not frames:
        raise RuntimeError("catalog extraction produced no frames")
    reel_path = output_dir / "analysis-reel.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "1",
            "-start_number",
            "1",
            "-i",
            str(frames_dir / "RF%06d.jpg"),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "24",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(reel_path),
        ],
        check=True,
    )
    _render_contact_sheets(frame_paths, output_dir / "contact-sheets")
    catalog = RushesCatalog(
        catalog_id=_catalog_fingerprint(clips, sample_interval_ms),
        source_directory=str(source_directory.resolve()),
        sample_interval_ms=sample_interval_ms,
        total_duration_ms=sum(clip.duration_ms for clip in clips),
        clips=clips,
        frames=frames,
        analysis_reel_path=str(reel_path.resolve()),
        generated_at=utc_now(),
    )
    write_json(output_dir / "catalog.json", catalog)
    return catalog


def _crop_filter(aspect_ratio: str, focus: str) -> tuple[str, tuple[int, int]]:
    if aspect_ratio == "16:9":
        return (
            "fps=30,scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080",
            (1920, 1080),
        )
    x_expression = {"left": "0", "center": "(iw-ow)/2", "right": "iw-ow"}[focus]
    return f"fps=30,scale=-2:1920,crop=1080:1920:{x_expression}:0", (1080, 1920)


def _segment_bounds(
    *, center_ms: int, requested_duration_ms: int, clip_duration_ms: int, shot: Any
) -> tuple[int, int]:
    start = center_ms - requested_duration_ms // 2
    end = center_ms + requested_duration_ms // 2
    if start < shot.start_time_ms:
        end += shot.start_time_ms - start
        start = shot.start_time_ms
    if end > shot.end_time_ms:
        start -= end - shot.end_time_ms
        end = shot.end_time_ms
    start = max(shot.start_time_ms, 0, start)
    end = min(shot.end_time_ms, clip_duration_ms, end)
    if end - start < 1000:
        raise ValueError("selected representative frame has less than one second inside its shot")
    return start, end


def render_rushes_edit(
    catalog: RushesCatalog,
    plan: RushesEditPlan,
    output_dir: Path,
    *,
    scdet_threshold: float = 4.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    shot_cache: dict[str, ShotManifest] = {}
    render_result: dict[str, Any] = {"timelines": []}
    for timeline in plan.timelines:
        slug = timeline.aspect_ratio.replace(":", "x")
        timeline_dir = output_dir / slug
        segments_dir = timeline_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[dict[str, Any]] = []
        for index, selected in enumerate(timeline.shots):
            frame = frames[selected.representative_frame_id]
            clip = clips[frame.clip_id]
            if clip.clip_id not in shot_cache:
                shot_cache[clip.clip_id] = detect_shots_ffmpeg(
                    Path(clip.path),
                    threshold=scdet_threshold,
                    output_path=output_dir / "shots" / f"{clip.clip_id}.json",
                )
            shot = next(
                item
                for item in shot_cache[clip.clip_id].shots
                if item.start_time_ms <= frame.requested_time_ms < item.end_time_ms
            )
            start_ms, end_ms = _segment_bounds(
                center_ms=frame.requested_time_ms,
                requested_duration_ms=round(selected.suggested_duration_seconds * 1000),
                clip_duration_ms=clip.duration_ms,
                shot=shot,
            )
            filter_graph, dimensions = _crop_filter(
                timeline.aspect_ratio, selected.vertical_focus
            )
            segment_path = segments_dir / f"{index:03d}-{selected.select_id}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start_ms / 1000:.3f}",
                    "-i",
                    clip.path,
                    "-t",
                    f"{(end_ms - start_ms) / 1000:.3f}",
                    "-vf",
                    filter_graph,
                    "-an",
                    "-c:v",
                    "h264_videotoolbox",
                    "-b:v",
                    "8M",
                    "-pix_fmt",
                    "yuv420p",
                    str(segment_path),
                ],
                check=True,
            )
            rendered.append(
                {
                    "select_id": selected.select_id,
                    "representative_frame_id": frame.frame_id,
                    "source_clip_id": clip.clip_id,
                    "source_path": clip.path,
                    "source_in_ms": start_ms,
                    "source_out_ms": end_ms,
                    "source_shot_id": shot.shot_id,
                    "vertical_focus": selected.vertical_focus,
                    "output_dimensions": list(dimensions),
                    "segment_path": str(segment_path.resolve()),
                }
            )
        concat_path = segments_dir / "concat.txt"
        concat_path.write_text(
            "".join(f"file '{Path(item['segment_path']).name}'\n" for item in rendered),
            encoding="utf-8",
        )
        output_path = timeline_dir / f"rough-cut-{slug}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path.resolve()),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path.resolve()),
            ],
            check=True,
        )
        render_result["timelines"].append(
            {
                "aspect_ratio": timeline.aspect_ratio,
                "output_path": str(output_path.resolve()),
                "shots": rendered,
            }
        )
    write_json(output_dir / "render-manifest.json", render_result)
    _render_review_html(catalog, plan, render_result, output_dir)
    return render_result


def _render_review_html(
    catalog: RushesCatalog,
    plan: RushesEditPlan,
    render_result: dict[str, Any],
    output_dir: Path,
) -> None:
    timelines = {item["aspect_ratio"]: item for item in render_result["timelines"]}
    sections: list[str] = []
    for timeline in plan.timelines:
        rendered = timelines[timeline.aspect_ratio]
        video_rel = Path(rendered["output_path"]).relative_to(output_dir.resolve())
        rows = []
        for selected, item in zip(timeline.shots, rendered["shots"], strict=True):
            frame = next(
                frame for frame in catalog.frames if frame.frame_id == selected.representative_frame_id
            )
            frame_rel = Path("..") / frame.image_path
            rows.append(
                "<tr>"
                f"<td><img src=\"{html.escape(str(frame_rel))}\"></td>"
                f"<td>{html.escape(item['source_clip_id'])}</td>"
                f"<td>{item['source_in_ms']/1000:.3f}–{item['source_out_ms']/1000:.3f}s</td>"
                f"<td>{html.escape(selected.role)}</td>"
                f"<td>{html.escape(selected.visual_description)}</td>"
                f"<td>{html.escape('; '.join(selected.quality_risks) or 'none')}</td>"
                "</tr>"
            )
        sections.append(
            f"<section><h2>{timeline.aspect_ratio} — {html.escape(timeline.title)}</h2>"
            f"<p>{html.escape(timeline.editorial_intent)}</p>"
            f"<video controls src=\"{html.escape(str(video_rel))}\"></video>"
            "<table><thead><tr><th>frame</th><th>clip</th><th>source range</th><th>role</th><th>description</th><th>risks</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )
    (output_dir / "index.html").write_text(
        """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Rushes rough-cut review</title>
<style>body{font:15px system-ui;background:#101214;color:#eee;max-width:1500px;margin:24px auto;padding:0 20px}section{background:#1b1f24;padding:20px;margin:20px 0;border-radius:12px}video{width:min(100%,960px);max-height:70vh;background:#000}table{border-collapse:collapse;width:100%;margin-top:18px}th,td{border:1px solid #3b424a;padding:8px;text-align:left;vertical-align:top}img{width:200px}code{color:#7cf}</style>
<h1>Rushes rough-cut review</h1><p>這是 Gemini frame-ID selects 經本機 FFmpeg shot-boundary 驗證後的實驗 rough cut，不是 final edit。</p>"""
        + "".join(sections)
        + "</html>",
        encoding="utf-8",
    )


def run_rushes_experiment(
    source_directory: Path,
    output_dir: Path,
    *,
    prompt_template: str,
    sample_interval_ms: int = 2000,
    scdet_threshold: float = 4.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    started = monotonic()
    catalog_path = output_dir / "catalog.json"
    if catalog_path.exists():
        catalog = RushesCatalog.model_validate(read_json(catalog_path))
    else:
        stage = monotonic()
        catalog = create_rushes_catalog(
            source_directory,
            output_dir,
            sample_interval_ms=sample_interval_ms,
        )
        timings["catalog_seconds"] = round(monotonic() - stage, 3)
    reel_path = Path(catalog.analysis_reel_path)
    reel_media = probe_video(reel_path)
    upload_dir = output_dir / "file-cache" / reel_media.sha256 / "upload"
    client = GeminiLabClient()
    try:
        stage = monotonic()
        uploaded, reused = client.ensure_video_upload(reel_path, upload_dir)
        timings["file_api_seconds"] = round(monotonic() - stage, 3)
        stage = monotonic()
        plan = client.plan_rushes_edit(
            catalog=catalog,
            uploaded=uploaded,
            prompt_template=prompt_template,
            project_id="rushes-selects",
            run_id=f"rushes-{uuid.uuid4().hex[:8]}",
            run_dir=output_dir / "gemini",
        )
        timings["gemini_plan_seconds"] = round(monotonic() - stage, 3)
    finally:
        client.close()
    stage = monotonic()
    render_result = render_rushes_edit(
        catalog,
        plan,
        output_dir / "renders",
        scdet_threshold=scdet_threshold,
    )
    timings["render_seconds"] = round(monotonic() - stage, 3)
    timings["total_seconds"] = round(monotonic() - started, 3)
    pricing = summarize_usage_and_list_price(output_dir / "gemini")
    write_json(output_dir / "pricing.json", pricing)
    write_json(
        output_dir / "timing.json",
        {**timings, "file_api_reused": reused, "generated_at": utc_now()},
    )
    result = {
        "catalog_path": str(catalog_path.resolve()),
        "plan_path": str((output_dir / "gemini" / "rushes_edit_plan.json").resolve()),
        "review_path": str((output_dir / "renders" / "index.html").resolve()),
        "renders": render_result,
        "timing": timings,
        "pricing": pricing,
    }
    write_json(output_dir / "result.json", result)
    return result
