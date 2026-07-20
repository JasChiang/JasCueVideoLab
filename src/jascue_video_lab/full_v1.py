from __future__ import annotations

import hashlib
import html
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from time import monotonic
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .billing import summarize_usage_and_list_price, summarize_usage_files
from .gemini import GeminiLabClient, MODEL_ID, VISUAL_EVIDENCE_SYSTEM_INSTRUCTION
from .media import create_analysis_proxy, extract_frame, has_audio_stream, probe_video, sha256_file
from .models import (
    ClipShotCatalog,
    DenseEventSelection,
    DenseFrame,
    DenseFrameCatalog,
    DerivedClipEvent,
    DerivedClipTimeline,
    FullClipCard,
    FullClipEvent,
    FeatureEditPlan,
    GroundingProposal,
    RushesCatalog,
    SegmentationTrack,
    ShotRepresentativeFrame,
)
from .overlay import draw_grounding_overlay
from .sam_tracking import track_bbox_sam21
from .schema import gemini_response_schema
from .shots import ShotManifest, detect_shots_ffmpeg
from .storage import append_error, read_json, utc_now, write_json


def mmss_to_ms(value: str) -> int:
    minutes, seconds = (int(part) for part in value.split(":"))
    if seconds > 59:
        raise ValueError("MM:SS seconds must be within 00..59")
    return (minutes * 60 + seconds) * 1000


def derive_clip_timeline(card: FullClipCard, shots: ShotManifest) -> DerivedClipTimeline:
    if card.source_asset_id == card.proxy_asset_id:
        raise ValueError("source and proxy identities must remain distinct")
    if shots.duration_ms != card.duration_ms:
        raise ValueError("shot manifest and Clip Card durations differ")
    events: list[DerivedClipEvent] = []
    for event in card.events:
        start_ms = mmss_to_ms(event.start_mmss)
        end_ms = mmss_to_ms(event.end_mmss)
        keyframe_ms = (
            mmss_to_ms(event.recommended_keyframe_mmss)
            if event.recommended_keyframe_mmss is not None
            else None
        )
        shot_ids = [
            shot.shot_id
            for shot in shots.shots
            if shot.start_time_ms < end_ms and shot.end_time_ms > start_ms
        ]
        if not shot_ids:
            raise ValueError(f"event {event.event_id} does not intersect a decoded shot")
        events.append(
            DerivedClipEvent(
                event_id=event.event_id,
                start_mmss=event.start_mmss,
                end_mmss=event.end_mmss,
                recommended_keyframe_mmss=event.recommended_keyframe_mmss,
                start_ms=start_ms,
                end_ms=end_ms,
                recommended_keyframe_ms=keyframe_ms,
                shot_ids=shot_ids,
                boundary_source="gemini_mmss_local_conversion",
                exact_frame_required=(
                    event.dense_refinement != "not_needed" or keyframe_ms is None
                ),
            )
        )
    return DerivedClipTimeline(
        source_asset_id=card.source_asset_id,
        duration_ms=card.duration_ms,
        events=events,
        generated_at=utc_now(),
    )


def create_shot_catalog(
    video_path: Path,
    source_asset_id: str,
    manifest: ShotManifest,
    output_dir: Path,
) -> ClipShotCatalog:
    """Keep one lightweight audit JPEG per shot; this is not Grounding evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[ShotRepresentativeFrame] = []
    for index, shot in enumerate(manifest.shots, start=1):
        requested_ms = (shot.start_time_ms + shot.end_time_ms) // 2
        frame_id = f"CF{index:06d}"
        frame_path = output_dir / "frames" / f"{frame_id}.jpg"
        extracted = extract_frame(video_path, requested_ms, frame_path, max_width=960)
        frames.append(
            ShotRepresentativeFrame(
                frame_id=frame_id,
                shot_id=shot.shot_id,
                role="middle",
                requested_time_ms=requested_ms,
                frame_time_ms=extracted.frame_time_ms,
                frame_pts=extracted.frame_pts,
                frame_hash=extracted.frame_hash,
                image_path=str(frame_path.resolve()),
            )
        )
    catalog = ClipShotCatalog(
        source_asset_id=source_asset_id,
        duration_ms=manifest.duration_ms,
        frames=frames,
        generated_at=utc_now(),
    )
    write_json(output_dir / "shot-catalog.json", catalog)
    return catalog


def _render_dense_transport(source_path: Path, output_path: Path, frame_id: str, max_width: int) -> None:
    with Image.open(source_path).convert("RGB") as source:
        width = min(max_width, source.width)
        height = round(source.height * width / source.width)
        resized = source.resize((width, height))
    font = ImageFont.load_default(size=max(18, width // 28))
    bar_height = max(48, height // 9)
    image = Image.new("RGB", (width, height + bar_height), "#080b10")
    image.paste(resized, (0, bar_height))
    draw = ImageDraw.Draw(image)
    draw.text((14, 10), frame_id, fill="#ffffff", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=90)


def _render_dense_contact_sheets(frames: list[DenseFrame], output_dir: Path) -> tuple[list[str], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns, rows = 4, 4
    cell_width, cell_height = 320, 200
    page_size = columns * rows
    paths: list[str] = []
    hashes: list[str] = []
    for page_number, start in enumerate(range(0, len(frames), page_size), start=1):
        canvas = Image.new("RGB", (cell_width * columns, cell_height * rows), "#101418")
        for local_index, frame in enumerate(frames[start : start + page_size]):
            with Image.open(frame.transport_image_path).convert("RGB") as source:
                fitted = ImageOps.pad(
                    source,
                    (cell_width, cell_height),
                    method=Image.Resampling.LANCZOS,
                    color="#101418",
                    centering=(0.5, 0.5),
                )
            x = (local_index % columns) * cell_width
            y = (local_index // columns) * cell_height
            canvas.paste(fitted, (x, y))
        path = output_dir / f"page-{page_number:03d}.jpg"
        canvas.save(path, quality=88)
        paths.append(str(path.resolve()))
        hashes.append(sha256_file(path))
    return paths, hashes


def create_dense_event_catalog(
    video_path: Path,
    source_asset_id: str,
    event: FullClipEvent,
    output_dir: Path,
    *,
    sampling_fps: float,
    max_width: int = 960,
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
) -> DenseFrameCatalog:
    if sampling_fps <= 0 or sampling_fps > 8:
        raise ValueError("dense sampling_fps must be within (0, 8]")
    source_media = probe_video(video_path)
    if source_media.asset_id != source_asset_id:
        raise ValueError("dense source identity differs from Clip Card")
    event_start_ms = mmss_to_ms(event.start_mmss)
    event_end_ms = mmss_to_ms(event.end_mmss)
    start_ms = event_start_ms if window_start_ms is None else window_start_ms
    end_ms = event_end_ms if window_end_ms is None else window_end_ms
    if not event_start_ms <= start_ms < end_ms <= event_end_ms:
        raise ValueError("dense window must remain inside the coarse event interval")
    interval_ms = max(1, round(1000 / sampling_fps))
    requested_times = set(range(start_ms, end_ms, interval_ms))
    if event.recommended_keyframe_mmss is not None:
        keyframe_ms = mmss_to_ms(event.recommended_keyframe_mmss)
        if start_ms <= keyframe_ms < end_ms:
            requested_times.add(keyframe_ms)
    requested_times = {
        min(max(0, value), source_media.duration_ms - 1) for value in requested_times
    }
    ordered_times = sorted(requested_times)
    if len(ordered_times) > 3600:
        raise ValueError("dense event catalog exceeds the 3600-image request limit")
    frames: list[DenseFrame] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, requested_ms in enumerate(ordered_times, start=1):
        frame_id = f"DF{index:06d}"
        source_path = output_dir / "analysis-frames" / f"{frame_id}.jpg"
        transport_path = output_dir / "transport-frames" / f"{frame_id}.jpg"
        extracted = extract_frame(
            video_path,
            requested_ms,
            source_path,
            max_width=max_width,
        )
        _render_dense_transport(source_path, transport_path, frame_id, max_width)
        frames.append(
            DenseFrame(
                frame_id=frame_id,
                event_id=event.event_id,
                requested_time_ms=requested_ms,
                frame_time_ms=extracted.frame_time_ms,
                frame_pts=extracted.frame_pts,
                frame_hash=extracted.frame_hash,
                width=extracted.width,
                height=extracted.height,
                image_path=str(source_path.resolve()),
                transport_image_path=str(transport_path.resolve()),
                transport_image_hash=sha256_file(transport_path),
            )
        )
    contact_sheet_paths, contact_sheet_hashes = _render_dense_contact_sheets(
        frames, output_dir / "contact-sheets"
    )
    catalog = DenseFrameCatalog(
        source_asset_id=source_asset_id,
        event_id=event.event_id,
        sampling_fps=sampling_fps,
        source_start_ms=start_ms,
        source_end_ms=end_ms,
        frames=frames,
        contact_sheet_paths=contact_sheet_paths,
        contact_sheet_hashes=contact_sheet_hashes,
        generated_at=utc_now(),
    )
    write_json(output_dir / "dense-catalog.json", catalog)
    return catalog


def dense_sampling_fps(event: FullClipEvent) -> float:
    text = " ".join(
        [event.label, event.description, *event.dense_refinement_reasons]
    ).lower()
    fast_markers = ("快速", "短暫", "0.2", "0.3", "0.4", "0.5", "fast", "transient", "ui")
    if event.dense_refinement == "required" and any(marker in text for marker in fast_markers):
        return 8.0
    if event.dense_refinement in {"required", "recommended"}:
        return 4.0
    return 2.0


def dense_window_for_event(
    event: FullClipEvent,
    shots: ShotManifest,
    *,
    window_ms: int = 4000,
) -> tuple[int, int, str]:
    if window_ms < 1000 or window_ms > 5000:
        raise ValueError("dense refinement window must be between 1000 and 5000 ms")
    event_start = mmss_to_ms(event.start_mmss)
    event_end = mmss_to_ms(event.end_mmss)
    center = (
        mmss_to_ms(event.recommended_keyframe_mmss)
        if event.recommended_keyframe_mmss is not None
        else (event_start + event_end) // 2
    )
    shot = next(
        (
            candidate
            for candidate in shots.shots
            if candidate.start_time_ms <= center < candidate.end_time_ms
        ),
        None,
    )
    if shot is None:
        raise ValueError(f"event {event.event_id} keyframe does not belong to a decoded shot")
    half = window_ms // 2
    start = max(event_start, shot.start_time_ms, center - half)
    end = min(event_end, shot.end_time_ms, center + (window_ms - half))
    if end - start < 250:
        raise ValueError(f"event {event.event_id} has no usable shot-local dense window")
    return start, end, shot.shot_id


def _cache_fingerprint(prompt: str) -> dict[str, str]:
    schema_json = json.dumps(
        gemini_response_schema(FullClipCard), sort_keys=True, separators=(",", ":")
    )
    return {
        "model": MODEL_ID,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(schema_json.encode()).hexdigest(),
    }


def _saved_request_matches_prompt(run_dir: Path, prompt: str) -> bool:
    request_path = run_dir / "clip_card.request.json"
    if not request_path.exists():
        return False
    try:
        request = read_json(request_path)
        request_inputs = request["input"]
        return bool(
            request.get("model") == MODEL_ID
            and request.get("system_instruction") == VISUAL_EVIDENCE_SYSTEM_INSTRUCTION
            and request_inputs
            and request_inputs[0].get("text", "").startswith(prompt)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _revalidate_saved_clip_card(
    run_dir: Path,
    source_asset_id: str,
    proxy_asset_id: str,
    duration_ms: int,
    prompt: str,
) -> FullClipCard | None:
    raw_output_path = run_dir / "clip_card.raw_output.json"
    if not raw_output_path.exists() or not _saved_request_matches_prompt(run_dir, prompt):
        return None
    try:
        raw_output = read_json(raw_output_path)
        card = FullClipCard.model_validate_json(raw_output["output_text"])
        if (
            card.source_asset_id != source_asset_id
            or card.proxy_asset_id != proxy_asset_id
            or card.duration_ms != duration_ms
        ):
            return None
        raw_interaction_path = run_dir / "clip_card.raw_interaction.json"
        interaction_id = None
        if raw_interaction_path.exists():
            interaction_id = read_json(raw_interaction_path).get("id") or None
        card = card.model_copy(
            update={
                "model_provenance": card.model_provenance.model_copy(
                    update={"interaction_id": interaction_id}
                )
            }
        )
        write_json(run_dir / "clip_card.json", card)
        write_json(
            run_dir / "clip_card.schema_validation.json",
            {
                "ok": True,
                "errors": [],
                "revalidated_from_saved_raw_output": True,
            },
        )
        return card
    except (KeyError, TypeError, ValueError):
        return None


def _archive_clip_card_attempt(run_dir: Path) -> Path | None:
    existing = sorted(run_dir.glob("clip_card.*.json"))
    if not existing:
        return None
    history_dir = run_dir / "history" / utc_now().replace(":", "-")
    history_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.copy2(path, history_dir / path.name)
    return history_dir


def _render_review(
    card: FullClipCard,
    timeline: DerivedClipTimeline,
    selections: dict[str, DenseEventSelection],
    output_dir: Path,
) -> Path:
    derived = {event.event_id: event for event in timeline.events}
    rows: list[str] = []
    for event in card.events:
        local = derived[event.event_id]
        selection = selections.get(event.event_id)
        dense_text = "not run"
        if selection is not None:
            dense_text = (
                html.escape(str(selection.recommended_frame_id))
                if selection.visible
                else "not visible"
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(event.event_id)}</td>"
            f"<td>{html.escape(event.start_mmss)}–{html.escape(event.end_mmss)}</td>"
            f"<td>{local.start_ms}–{local.end_ms}</td>"
            f"<td>{html.escape(event.label)}</td>"
            f"<td>{html.escape(event.description)}</td>"
            f"<td>{html.escape(event.dense_refinement)} / {dense_text}</td>"
            f"<td>{html.escape(', '.join(local.shot_ids))}</td>"
            "</tr>"
        )
    document = f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>Full v1 clip review</title>
<style>body{{font:15px system-ui;background:#101418;color:#eee;max-width:1500px;margin:24px auto;padding:0 20px}}video{{width:min(100%,960px);background:#000}}table{{border-collapse:collapse;width:100%;margin-top:20px}}th,td{{border:1px solid #3b424a;padding:8px;vertical-align:top}}code{{color:#7ee0b8}}</style>
<h1>Full v1 clip review</h1>
<p>Gemini event time is coarse <code>MM:SS</code>. Milliseconds are local conversions; dense DF IDs map to exact source PTS and hashes.</p>
<video controls src="analysis-proxy.mp4"></video>
<p>{html.escape(card.summary)}</p>
<table><thead><tr><th>event</th><th>Gemini MM:SS</th><th>local ms</th><th>label</th><th>description</th><th>dense</th><th>shots</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</html>"""
    path = output_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def run_full_clip(
    video_path: Path,
    output_dir: Path,
    *,
    clip_card_prompt: str,
    dense_prompt: str,
    proxy_max_side: int = 1280,
    proxy_fps: int = 30,
    audio_mode: str = "auto",
    scdet_threshold: float = 4.0,
    temperature: float = 0.2,
    dense_mode: str = "none",
    dense_event_ids: set[str] | None = None,
    dense_window_ms: int = 4000,
    dense_fps_override: float | None = None,
    prepare_only: bool = False,
    file_cache_root: Path | None = None,
) -> dict[str, Any]:
    if dense_mode not in {"none", "required", "flagged", "all"}:
        raise ValueError("dense_mode must be none, required, flagged, or all")
    if dense_fps_override not in {None, 4.0, 8.0}:
        raise ValueError("dense_fps_override must be 4 or 8 FPS")
    if audio_mode not in {"auto", "off", "required"}:
        raise ValueError("audio_mode must be auto, off, or required")
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    started = monotonic()
    stage = monotonic()
    source_media = probe_video(video_path)
    source_has_audio = has_audio_stream(video_path)
    if audio_mode == "required" and not source_has_audio:
        raise ValueError("audio_mode=required but the source has no audio stream")
    include_audio = source_has_audio and audio_mode != "off"
    write_json(output_dir / "source.media.json", source_media)
    write_json(
        output_dir / "private-source.json",
        {
            "privacy": "local-only; excluded from public reports",
            "path": str(video_path.resolve()),
            "source_asset_id": source_media.asset_id,
        },
    )
    proxy_path = output_dir / "analysis-proxy.mp4"
    if proxy_path.exists():
        proxy_media = probe_video(proxy_path)
        proxy_record = read_json(output_dir / "analysis-proxy.json")
        proxy_has_audio = has_audio_stream(proxy_path)
        if proxy_has_audio != include_audio:
            raise ValueError(
                "existing analysis proxy audio policy differs from --audio-mode; "
                "use a new output directory"
            )
        proxy_record.update(
            {
                "audio_mode": audio_mode,
                "source_has_audio": source_has_audio,
                "proxy_has_audio": proxy_has_audio,
            }
        )
        write_json(output_dir / "analysis-proxy.json", proxy_record)
    else:
        proxy_media, proxy_record = create_analysis_proxy(
            video_path,
            proxy_path,
            max_side=proxy_max_side,
            fps=proxy_fps,
            preserve_audio=include_audio,
        )
        proxy_record["audio_mode"] = audio_mode
        write_json(output_dir / "analysis-proxy.json", proxy_record)
    timings["probe_and_proxy_seconds"] = round(monotonic() - stage, 3)

    stage = monotonic()
    shots = detect_shots_ffmpeg(
        video_path,
        threshold=scdet_threshold,
        output_path=output_dir / "shots.json",
    )
    shot_catalog_path = output_dir / "shot-catalog" / "shot-catalog.json"
    if shot_catalog_path.exists():
        shot_catalog = ClipShotCatalog.model_validate(read_json(shot_catalog_path))
    else:
        shot_catalog = create_shot_catalog(
            video_path,
            source_media.asset_id,
            shots,
            output_dir / "shot-catalog",
        )
    timings["shot_catalog_seconds"] = round(monotonic() - stage, 3)

    if prepare_only:
        timings["total_seconds"] = round(monotonic() - started, 3)
        result = {
            "source_asset_id": source_media.asset_id,
            "proxy_asset_id": proxy_media.asset_id,
            "duration_ms": source_media.duration_ms,
            "audio_mode": audio_mode,
            "source_has_audio": source_has_audio,
            "proxy_has_audio": bool(proxy_record["proxy_has_audio"]),
            "shot_catalog_path": str(shot_catalog_path.resolve()),
            "shot_count": len(shots.shots),
            "shot_representative_frame_count": len(shot_catalog.frames),
            "event_count": 0,
            "dense_event_count": 0,
            "prepare_only": True,
            "execution_pricing": summarize_usage_files([]),
            "timing": timings,
        }
        write_json(output_dir / "preparation.json", result)
        write_json(
            output_dir / "preparation-timing.json",
            {**timings, "generated_at": utc_now()},
        )
        return result

    client = GeminiLabClient(temperature=temperature)
    selections: dict[str, DenseEventSelection] = {}
    execution_interactions: list[Path] = []
    try:
        stage = monotonic()
        upload_dir = _shared_upload_dir(
            proxy_media.sha256,
            file_cache_root=file_cache_root or DEFAULT_FILE_CACHE_ROOT,
            legacy_search_root=Path(__file__).resolve().parents[2] / "artifacts",
        )
        uploaded, reused = client.ensure_video_upload(proxy_path, upload_dir)
        timings["file_api_seconds"] = round(monotonic() - stage, 3)

        cache_key = {
            **_cache_fingerprint(clip_card_prompt),
            "source_asset_id": source_media.asset_id,
            "proxy_asset_id": proxy_media.asset_id,
        }
        card_path = output_dir / "gemini" / "clip-card" / "clip_card.json"
        cache_path = output_dir / "gemini" / "clip-card" / "cache-key.json"
        run_dir = output_dir / "gemini" / "clip-card"
        if (
            card_path.exists()
            and cache_path.exists()
            and read_json(cache_path) == cache_key
            and _saved_request_matches_prompt(run_dir, clip_card_prompt)
        ):
            card = FullClipCard.model_validate(read_json(card_path))
            card_reused = True
        else:
            card = _revalidate_saved_clip_card(
                run_dir,
                source_media.asset_id,
                proxy_media.asset_id,
                source_media.duration_ms,
                clip_card_prompt,
            )
            if card is None:
                stage = monotonic()
                _archive_clip_card_attempt(run_dir)
                card = client.analyze_full_clip(
                    source_media=source_media,
                    proxy_media=proxy_media,
                    uploaded=uploaded,
                    prompt_template=clip_card_prompt,
                    run_id=f"full-clip-{uuid.uuid4().hex[:8]}",
                    run_dir=run_dir,
                )
                execution_interactions.append(run_dir / "clip_card.raw_interaction.json")
                timings["gemini_clip_card_seconds"] = round(monotonic() - stage, 3)
                card_reused = False
            else:
                card_reused = True
                timings["gemini_clip_card_seconds"] = 0.0
            write_json(cache_path, cache_key)

        timeline = derive_clip_timeline(card, shots)
        write_json(output_dir / "derived-timeline.json", timeline)

        requested_dense_ids = dense_event_ids or set()
        unknown_requested = requested_dense_ids - {event.event_id for event in card.events}
        if unknown_requested:
            raise ValueError(f"unknown --dense-event IDs: {sorted(unknown_requested)}")
        for event in card.events:
            should_dense = event.event_id in requested_dense_ids or {
                    "none": False,
                    "required": event.dense_refinement == "required",
                    "flagged": event.dense_refinement in {"required", "recommended"},
                    "all": True,
                }[dense_mode]
            if not should_dense:
                continue
            event_dir = output_dir / "dense" / event.event_id
            dense_catalog_path = event_dir / "dense-catalog.json"
            if dense_catalog_path.exists():
                dense_catalog = DenseFrameCatalog.model_validate(read_json(dense_catalog_path))
            else:
                stage = monotonic()
                dense_start_ms, dense_end_ms, dense_shot_id = dense_window_for_event(
                    event,
                    shots,
                    window_ms=dense_window_ms,
                )
                dense_catalog = create_dense_event_catalog(
                    video_path,
                    source_media.asset_id,
                    event,
                    event_dir,
                    sampling_fps=dense_fps_override or dense_sampling_fps(event),
                    window_start_ms=dense_start_ms,
                    window_end_ms=dense_end_ms,
                )
                write_json(
                    event_dir / "dense-window.json",
                    {
                        "event_id": event.event_id,
                        "shot_id": dense_shot_id,
                        "start_ms": dense_start_ms,
                        "end_ms": dense_end_ms,
                        "window_ms": dense_end_ms - dense_start_ms,
                        "source": "local_window_around_coarse_mmss_keyframe",
                    },
                )
                timings[f"dense_extract_{event.event_id}_seconds"] = round(
                    monotonic() - stage, 3
                )
            stage = monotonic()
            selection = client.select_dense_event_frames(
                event=event,
                catalog=dense_catalog,
                prompt_template=dense_prompt,
                run_id=f"dense-{uuid.uuid4().hex[:8]}",
                run_dir=event_dir / "gemini",
            )
            execution_interactions.append(
                event_dir / "gemini" / "dense_selection.raw_interaction.json"
            )
            timings[f"dense_gemini_{event.event_id}_seconds"] = round(
                monotonic() - stage, 3
            )
            selections[event.event_id] = selection
    finally:
        client.close()

    review_path = _render_review(card, timeline, selections, output_dir)
    pricing = summarize_usage_and_list_price(output_dir / "gemini")
    dense_pricing = summarize_usage_and_list_price(output_dir / "dense")
    pricing["dense_requests"] = dense_pricing
    pricing["estimated_total_cost_usd"] = round(
        pricing["estimated_total_cost_usd"]
        + dense_pricing["estimated_total_cost_usd"],
        8,
    )
    execution_pricing = summarize_usage_files(
        [path for path in execution_interactions if path.exists()],
        relative_to=output_dir,
    )
    write_json(output_dir / "pricing.json", pricing)
    write_json(output_dir / "execution-pricing.json", execution_pricing)
    timings["total_seconds"] = round(monotonic() - started, 3)
    write_json(
        output_dir / "timing.json",
        {
            **timings,
            "file_api_reused": reused,
            "clip_card_reused": card_reused,
            "generated_at": utc_now(),
        },
    )
    result = {
        "source_asset_id": source_media.asset_id,
        "proxy_asset_id": proxy_media.asset_id,
        "clip_card_path": str(card_path.resolve()),
        "derived_timeline_path": str((output_dir / "derived-timeline.json").resolve()),
        "shot_catalog_path": str(shot_catalog_path.resolve()),
        "shot_count": len(shots.shots),
        "shot_representative_frame_count": len(shot_catalog.frames),
        "event_count": len(card.events),
        "duration_ms": source_media.duration_ms,
        "audio_mode": audio_mode,
        "source_has_audio": source_has_audio,
        "proxy_has_audio": bool(proxy_record["proxy_has_audio"]),
        "dense_event_count": len(selections),
        "review_path": str(review_path.resolve()),
        "pricing": pricing,
        "execution_pricing": execution_pricing,
        "timing": timings,
    }
    write_json(output_dir / "result.json", result)
    return result


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
DEFAULT_FILE_CACHE_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "full-v1-file-cache"


def selected_clip_ids_from_feature_plan(
    catalog: RushesCatalog,
    plan: FeatureEditPlan,
) -> list[str]:
    """Resolve the unique source clips actually referenced by an edit plan."""
    if plan.catalog_id != catalog.catalog_id:
        raise ValueError("feature plan and rushes catalog IDs differ")
    frames = {frame.frame_id: frame for frame in catalog.frames}
    selected: list[str] = []
    seen: set[str] = set()
    for chapter in plan.chapters:
        for frame_id in (chapter.horizontal_frame_id, chapter.vertical_frame_id):
            if frame_id is None:
                continue
            if frame_id not in frames:
                raise ValueError(f"feature plan references unknown frame ID: {frame_id}")
            clip_id = frames[frame_id].clip_id
            if clip_id not in seen:
                seen.add(clip_id)
                selected.append(clip_id)
    return selected


def run_selected_full_clips(
    *,
    catalog_path: Path,
    plan_path: Path,
    prepared_library_dir: Path,
    output_dir: Path,
    clip_card_prompt: str,
    dense_prompt: str,
    max_clips: int | None = None,
    temperature: float = 0.2,
    audio_mode: str = "auto",
    prepare_only: bool = False,
    file_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Run Full Clip Cards only for clips already selected by a feature plan."""
    if max_clips is not None and max_clips < 1:
        raise ValueError("max_clips must be at least one")
    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    plan = FeatureEditPlan.model_validate(read_json(plan_path))
    selected_ids = selected_clip_ids_from_feature_plan(catalog, plan)
    if max_clips is not None:
        selected_ids = selected_ids[:max_clips]
    clips = {clip.clip_id: clip for clip in catalog.clips}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "selected-clip-cards.json"
    entries_by_id: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if (
            previous.get("catalog_id") == catalog.catalog_id
            and previous.get("project_id") == plan.project_id
            and previous.get("selection_source") == "feature_edit_plan"
        ):
            entries_by_id = {
                str(entry["clip_id"]): entry for entry in previous.get("clips", [])
            }
    total_cost = 0.0
    started = monotonic()

    for position, clip_id in enumerate(selected_ids, start=1):
        clip = clips.get(clip_id)
        if clip is None:
            raise ValueError(f"selected frame belongs to unknown clip: {clip_id}")
        clip_dir = prepared_library_dir / "clips" / clip.sha256[:16]
        if not (clip_dir / "analysis-proxy.mp4").exists():
            raise FileNotFoundError(
                f"prepared analysis proxy missing for selected clip {clip_id}: {clip_dir}"
            )
        clip_started = monotonic()
        try:
            if prepare_only:
                result = {
                    "source_asset_id": f"sha256:{clip.sha256}",
                    "event_count": None,
                    "execution_pricing": {"estimated_total_cost_usd": 0.0},
                }
            else:
                result = run_full_clip(
                    Path(clip.path),
                    clip_dir,
                    clip_card_prompt=clip_card_prompt,
                    dense_prompt=dense_prompt,
                    temperature=temperature,
                    audio_mode=audio_mode,
                    dense_mode="none",
                    prepare_only=False,
                    file_cache_root=file_cache_root,
                )
            execution_cost = float(result["execution_pricing"]["estimated_total_cost_usd"])
            lifetime_cost = float(
                result.get("pricing", {}).get("estimated_total_cost_usd", execution_cost)
            )
            total_cost += execution_cost
            entries_by_id[clip_id] = {
                "position": position,
                "clip_id": clip_id,
                "source_asset_id": result["source_asset_id"],
                "clip_run": str(clip_dir.resolve()),
                "status": "prepared_local" if prepare_only else "ok",
                "event_count": None if prepare_only else result["event_count"],
                "execution_cost_usd": execution_cost,
                "artifact_lifetime_cost_usd": lifetime_cost,
                "elapsed_seconds": round(monotonic() - clip_started, 3),
            }
        except Exception as error:
            append_error(output_dir / "errors", f"clip-{position:03d}", error)
            entries_by_id[clip_id] = {
                "position": position,
                "clip_id": clip_id,
                "source_asset_id": f"sha256:{clip.sha256}",
                "clip_run": str(clip_dir.resolve()),
                "status": "error",
                "error_type": type(error).__name__,
                "elapsed_seconds": round(monotonic() - clip_started, 3),
            }

        entries = [entries_by_id[item] for item in selected_ids if item in entries_by_id]
        write_json(
            manifest_path,
            {
                "catalog_id": catalog.catalog_id,
                "project_id": plan.project_id,
                "selection_source": "feature_edit_plan",
                "prepare_only": prepare_only,
                "selected_clip_count": len(selected_ids),
                "clips": entries,
                "generated_at": utc_now(),
            },
        )

    entries = [entries_by_id[item] for item in selected_ids if item in entries_by_id]
    succeeded = sum(entry["status"] != "error" for entry in entries)
    result = {
        "catalog_id": catalog.catalog_id,
        "project_id": plan.project_id,
        "selected_clip_count": len(selected_ids),
        "succeeded": succeeded,
        "failed": len(selected_ids) - succeeded,
        "prepare_only": prepare_only,
        "estimated_new_cost_usd": round(total_cost, 8),
        "elapsed_seconds": round(monotonic() - started, 3),
        "manifest_path": str(manifest_path.resolve()),
    }
    write_json(output_dir / "result.json", result)
    return result


def _shared_upload_dir(
    proxy_sha256: str,
    *,
    file_cache_root: Path,
    legacy_search_root: Path,
) -> Path:
    """Resolve a SHA-keyed cache and migrate exact legacy metadata when available."""
    target = file_cache_root / proxy_sha256 / "upload"
    if target.exists():
        return target
    pattern = f"**/file-cache/{proxy_sha256}/upload/file_cache.json"
    for cache_record in legacy_search_root.glob(pattern):
        source = cache_record.parent
        if source == target:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)
        write_json(
            target / "migration.json",
            {
                "proxy_sha256": proxy_sha256,
                "source": str(source.resolve()),
                "migrated_at": utc_now(),
                "verification": "remote ACTIVE state must still be checked before reuse",
            },
        )
        break
    return target


def run_full_library(
    source_dir: Path,
    output_dir: Path,
    *,
    clip_card_prompt: str,
    dense_prompt: str,
    recursive: bool = False,
    max_clips: int | None = None,
    proxy_max_side: int = 1280,
    proxy_fps: int = 30,
    audio_mode: str = "auto",
    scdet_threshold: float = 4.0,
    temperature: float = 0.2,
    prepare_only: bool = False,
    file_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Create one resumable Clip Card per unique source without automatic geometry work."""
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if audio_mode not in {"auto", "off", "required"}:
        raise ValueError("audio_mode must be auto, off, or required")
    if max_clips is not None and max_clips < 1:
        raise ValueError("max_clips must be at least one")
    iterator = source_dir.rglob("*") if recursive else source_dir.glob("*")
    paths = sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda path: path.name.casefold(),
    )
    if max_clips is not None:
        paths = paths[:max_clips]
    if not paths:
        raise ValueError("source directory contains no supported video files")

    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    started = monotonic()
    public_entries: list[dict[str, Any]] = []
    private_entries: list[dict[str, Any]] = []
    seen_assets: dict[str, str] = {}
    total_cost = 0.0
    total_duration_ms = 0
    succeeded = 0
    failed = 0
    duplicates = 0

    for position, path in enumerate(paths, start=1):
        clip_started = monotonic()
        private_entry: dict[str, Any] = {
            "position": position,
            "path": str(path.resolve()),
        }
        private_entries.append(private_entry)
        try:
            media = probe_video(path)
            total_duration_ms += media.duration_ms
            private_entry.update(
                {
                    "source_asset_id": media.asset_id,
                    "duration_ms": media.duration_ms,
                }
            )
            if media.asset_id in seen_assets:
                duplicates += 1
                public_entries.append(
                    {
                        "source_asset_id": media.asset_id,
                        "duration_ms": media.duration_ms,
                        "status": "duplicate_skipped",
                        "canonical_clip_run": seen_assets[media.asset_id],
                    }
                )
                private_entry["status"] = "duplicate_skipped"
                continue

            asset_key = media.sha256[:16]
            clip_dir = clips_dir / asset_key
            seen_assets[media.asset_id] = f"clips/{asset_key}"
            result = run_full_clip(
                path,
                clip_dir,
                clip_card_prompt=clip_card_prompt,
                dense_prompt=dense_prompt,
                proxy_max_side=proxy_max_side,
                proxy_fps=proxy_fps,
                audio_mode=audio_mode,
                scdet_threshold=scdet_threshold,
                temperature=temperature,
                dense_mode="none",
                prepare_only=prepare_only,
                file_cache_root=file_cache_root,
            )
            clip_cost = float(
                result["execution_pricing"]["estimated_total_cost_usd"]
            )
            artifact_lifetime_cost = (
                float(result["pricing"]["estimated_total_cost_usd"])
                if "pricing" in result
                else 0.0
            )
            total_cost += clip_cost
            succeeded += 1
            public_entries.append(
                {
                    "source_asset_id": media.asset_id,
                    "duration_ms": media.duration_ms,
                    "source_has_audio": result["source_has_audio"],
                    "proxy_has_audio": result["proxy_has_audio"],
                    "status": "prepared_local" if prepare_only else "ok",
                    "clip_run": f"clips/{asset_key}",
                    "event_count": result["event_count"] if not prepare_only else None,
                    "shot_count": result["shot_count"],
                    "dense_event_count": 0,
                    "estimated_cost_usd": clip_cost,
                    "artifact_lifetime_cost_usd": artifact_lifetime_cost,
                    "elapsed_seconds": round(monotonic() - clip_started, 3),
                }
            )
            private_entry["status"] = "prepared_local" if prepare_only else "ok"
        except Exception as error:
            failed += 1
            append_error(output_dir / "errors", f"clip-{position:04d}", error)
            public_entries.append(
                {
                    "source_asset_id": private_entry.get("source_asset_id", "unresolved"),
                    "duration_ms": private_entry.get("duration_ms"),
                    "status": "error",
                    "error_type": type(error).__name__,
                }
            )
            private_entry["status"] = "error"
            private_entry["error"] = f"{type(error).__name__}: {error}"
        finally:
            write_json(
                output_dir / "private-library.json",
                {
                    "privacy": "local-only; filenames and paths must not enter public reports",
                    "source_directory": str(source_dir.resolve()),
                    "clips": private_entries,
                },
            )
            write_json(
                output_dir / "library-index.json",
                {
                    "method": (
                        "local_proxy_and_shot_preparation"
                        if prepare_only
                        else "full_proxy_to_mmss_clip_card"
                    ),
                    "geometry_policy": "deferred_until_event_selected",
                    "dense_policy": "disabled_by_default_local_event_fallback_only",
                    "audio_mode": audio_mode,
                    "clips": public_entries,
                    "generated_at": utc_now(),
                },
            )

    result = {
        "input_clip_count": len(paths),
        "unique_clip_count": len(seen_assets),
        "succeeded": succeeded,
        "failed": failed,
        "duplicates_skipped": duplicates,
        "total_duration_ms": total_duration_ms,
        "estimated_total_cost_usd": round(total_cost, 8),
        "cost_scope": "new Gemini requests made during this execution only",
        "prepare_only": prepare_only,
        "audio_mode": audio_mode,
        "elapsed_seconds": round(monotonic() - started, 3),
        "library_index_path": str((output_dir / "library-index.json").resolve()),
        "private_library_path": str((output_dir / "private-library.json").resolve()),
    }
    write_json(output_dir / "result.json", result)
    return result


def _extract_tracking_source(
    source_path: Path,
    start_ms: int,
    end_ms: int,
    output_path: Path,
) -> None:
    if end_ms <= start_ms:
        raise ValueError("tracking source interval must be non-empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
            str(source_path),
            "-t",
            f"{(end_ms - start_ms) / 1000:.3f}",
            "-map",
            "0:v:0",
            "-vf",
            "scale=1920:1920:force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-an",
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


def run_full_event_geometry(
    clip_run_dir: Path,
    event_id: str,
    *,
    grounding_prompt: str,
    checkpoint_path: Path | None = None,
    target_entity_id: str | None = None,
    target_description: str | None = None,
    sam_analysis_fps: float = 2.0,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Ground one selected Clip Card event and optionally propagate SAM inside its interval."""
    if bool(target_entity_id) != bool(target_description):
        raise ValueError("target entity ID and description must be provided together")
    card = FullClipCard.model_validate(
        read_json(clip_run_dir / "gemini" / "clip-card" / "clip_card.json")
    )
    timeline = DerivedClipTimeline.model_validate(
        read_json(clip_run_dir / "derived-timeline.json")
    )
    source_record = read_json(clip_run_dir / "private-source.json")
    source_path = Path(source_record["path"])
    media = probe_video(source_path)
    if media.asset_id != card.source_asset_id:
        raise ValueError("private source identity differs from Clip Card")
    try:
        event = next(item for item in card.events if item.event_id == event_id)
        derived = next(item for item in timeline.events if item.event_id == event_id)
    except StopIteration as error:
        raise ValueError(f"unknown Clip Card event: {event_id}") from error
    if target_entity_id is None:
        if not event.grounding_targets:
            raise ValueError(f"event {event_id} has no proposed Grounding target")
        selected_target = event.grounding_targets[0]
        target_entity_id = selected_target.entity_id
        target_description = selected_target.target_description

    dense_selection_path = clip_run_dir / "dense" / event_id / "gemini" / "dense_selection.json"
    seed_source = "clip_card_recommended_mmss"
    requested_time_ms = derived.recommended_keyframe_ms
    if dense_selection_path.exists():
        dense_selection = DenseEventSelection.model_validate(read_json(dense_selection_path))
        if dense_selection.visible and dense_selection.recommended_frame_id:
            dense_catalog = DenseFrameCatalog.model_validate(
                read_json(clip_run_dir / "dense" / event_id / "dense-catalog.json")
            )
            dense_frame = next(
                frame
                for frame in dense_catalog.frames
                if frame.frame_id == dense_selection.recommended_frame_id
            )
            requested_time_ms = dense_frame.requested_time_ms
            seed_source = f"dense_frame_id:{dense_frame.frame_id}"
    if requested_time_ms is None:
        requested_time_ms = (derived.start_ms + derived.end_ms) // 2
        seed_source = "local_event_midpoint_no_model_keyframe"

    geometry_dir = clip_run_dir / "geometry" / event_id
    grounding_dir = geometry_dir / "grounding"
    frame_path = grounding_dir / "frame.png"
    frame = extract_frame(source_path, requested_time_ms, frame_path)
    write_json(
        grounding_dir / "frame.json",
        {
            **frame.model_dump(mode="json"),
            "seed_source": seed_source,
            "coarse_event_start_mmss": event.start_mmss,
            "coarse_event_end_mmss": event.end_mmss,
        },
    )
    started = monotonic()
    grounding_path = grounding_dir / "grounding.json"
    if grounding_path.exists():
        proposal = GroundingProposal.model_validate(read_json(grounding_path))
        grounding_reused = True
    else:
        client = GeminiLabClient(temperature=temperature)
        try:
            proposal = client.ground_frame(
                media=media,
                frame=frame,
                event_id=event.event_id,
                event_description=event.description,
                entity_id=str(target_entity_id),
                target_description=str(target_description),
                prompt_template=grounding_prompt,
                run_id=f"full-ground-{uuid.uuid4().hex[:8]}",
                output_dir=grounding_dir,
            )
        finally:
            client.close()
        grounding_reused = False
    grounding_seconds = round(monotonic() - started, 3)
    draw_grounding_overlay(frame_path, proposal, grounding_dir / "debug.png")

    track: SegmentationTrack | None = None
    if checkpoint_path is not None:
        if not proposal.visible or not proposal.candidates:
            raise ValueError(f"Gemini Grounding target is not visible for {event_id}")
        track_dir = geometry_dir / "sam21"
        track_path = track_dir / "segmentation-track.json"
        if track_path.exists():
            track = SegmentationTrack.model_validate(read_json(track_path))
        else:
            tracking_source = geometry_dir / "tracking-source.mp4"
            if not tracking_source.exists():
                _extract_tracking_source(
                    source_path,
                    derived.start_ms,
                    derived.end_ms,
                    tracking_source,
                )
            candidate = max(proposal.candidates, key=lambda item: item.confidence)
            track = track_bbox_sam21(
                video_path=tracking_source,
                checkpoint_path=checkpoint_path,
                seed_time_ms=max(0, frame.frame_time_ms - derived.start_ms),
                seed_box_2d=candidate.box_2d,
                target_description=str(target_description),
                output_dir=track_dir,
                seed_source=str(grounding_path),
                asset_id=proposal.asset_id,
                analysis_fps=sam_analysis_fps,
                max_side=960,
                device="auto",
                ffmpeg_scdet_threshold=4.0,
                seed_box_padding_ratio=0.04,
            )
    pricing = summarize_usage_and_list_price(geometry_dir)
    execution_pricing = summarize_usage_files(
        [grounding_dir / "grounding.raw_interaction.json"] if not grounding_reused else [],
        relative_to=geometry_dir,
    )
    result = {
        "event_id": event_id,
        "target_entity_id": target_entity_id,
        "target_description": target_description,
        "seed_source": seed_source,
        "frame_pts": frame.frame_pts,
        "frame_time_ms": frame.frame_time_ms,
        "frame_hash": frame.frame_hash,
        "grounding_visible": proposal.visible,
        "grounding_candidate_count": len(proposal.candidates),
        "grounding_reused": grounding_reused,
        "grounding_seconds": grounding_seconds,
        "sam_track_path": (
            str((geometry_dir / "sam21" / "segmentation-track.json").resolve())
            if track is not None
            else None
        ),
        "sam_total_samples": track.total_samples if track is not None else 0,
        "pricing": pricing,
        "execution_pricing": execution_pricing,
    }
    write_json(geometry_dir / "result.json", result)
    return result
