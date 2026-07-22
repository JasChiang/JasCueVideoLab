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
from .gemini import (
    GeminiLabClient,
    GroundingIdentityReference,
    MODEL_ID,
    VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
)
from .grounding_selection import (
    require_grounding_request_match,
    require_tracking_seed_candidate,
)
from .identity_checkpoints import plan_identity_checkpoints
from .media import create_analysis_proxy, extract_frame, has_audio_stream, probe_video, sha256_file
from .models import (
    ClipShotCatalog,
    DenseEventSelection,
    DenseFrame,
    DenseFrameCatalog,
    DerivedClipEvent,
    DerivedClipTimeline,
    EvidenceIdentityContractV2,
    EvidenceQueryLock,
    EvidenceQueryLockV2,
    EvidenceQueryTargetRef,
    EvidenceTargetIdentityV2,
    FullClipCard,
    FullClipEvent,
    FeatureEditPlan,
    GeminiNativeGroundingProposal,
    GroundingProposal,
    MatchStatus,
    PredicateRequiredAt,
    PredicateStatus,
    RushesCatalog,
    SegmentationTrack,
    ShotRepresentativeFrame,
)
from .overlay import draw_grounding_overlay
from .query_refinement import (
    QueryTemporalDecision,
    QueryTemporalSelection,
    build_query_temporal_fingerprint,
    validate_query_temporal_evidence_bundle,
    write_query_temporal_consumer_lineage,
)
from .sam_tracking import (
    SAM21_CONFIG,
    SAM21_IMPLEMENTATION_REVISION,
    require_bbox_track_request_match,
    resolve_tracking_interval,
    track_bbox_sam21,
)
from .schema import gemini_response_schema
from .shots import ShotManifest, detect_shots_ffmpeg
from .storage import append_error, read_json, utc_now, write_json


def _usage_artifact_snapshot(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in root.rglob("*raw_interaction.json")
    }


def _incremental_usage_since(
    root: Path,
    before: dict[str, str],
) -> dict[str, Any]:
    if not root.exists():
        return summarize_usage_files([])
    changed = [
        path
        for path in root.rglob("*raw_interaction.json")
        if before.get(str(path.relative_to(root))) != sha256_file(path)
    ]
    return summarize_usage_files(changed, relative_to=root)


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
            or card.model_provenance.model_id != MODEL_ID
        ):
            return None
        raw_interaction_path = run_dir / "clip_card.raw_interaction.json"
        interaction_id = None
        if raw_interaction_path.exists():
            raw_interaction = read_json(raw_interaction_path)
            if raw_interaction.get("model") != MODEL_ID:
                return None
            interaction_id = raw_interaction.get("id") or None
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

    client = GeminiLabClient()
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
        usage_before = _usage_artifact_snapshot(clip_dir)
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
                    audio_mode=audio_mode,
                    dense_mode="none",
                    prepare_only=False,
                    file_cache_root=file_cache_root,
                )
            execution_pricing = _incremental_usage_since(clip_dir, usage_before)
            execution_cost = float(execution_pricing["estimated_total_cost_usd"])
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
                "execution_pricing": execution_pricing,
                "artifact_lifetime_cost_usd": lifetime_cost,
                "elapsed_seconds": round(monotonic() - clip_started, 3),
            }
        except Exception as error:
            append_error(output_dir / "errors", f"clip-{position:03d}", error)
            execution_pricing = _incremental_usage_since(clip_dir, usage_before)
            execution_cost = float(execution_pricing["estimated_total_cost_usd"])
            lifetime_pricing = summarize_usage_and_list_price(clip_dir)
            total_cost += execution_cost
            entries_by_id[clip_id] = {
                "position": position,
                "clip_id": clip_id,
                "source_asset_id": f"sha256:{clip.sha256}",
                "clip_run": str(clip_dir.resolve()),
                "status": "error",
                "error_type": type(error).__name__,
                "execution_cost_usd": execution_cost,
                "artifact_lifetime_cost_usd": lifetime_pricing[
                    "estimated_total_cost_usd"
                ],
                "execution_pricing": execution_pricing,
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
        clip_dir: Path | None = None
        usage_before: dict[str, str] = {}
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
            usage_before = _usage_artifact_snapshot(clip_dir)
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
                dense_mode="none",
                prepare_only=prepare_only,
                file_cache_root=file_cache_root,
            )
            execution_pricing = _incremental_usage_since(clip_dir, usage_before)
            clip_cost = float(execution_pricing["estimated_total_cost_usd"])
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
                    "execution_pricing": execution_pricing,
                    "artifact_lifetime_cost_usd": artifact_lifetime_cost,
                    "elapsed_seconds": round(monotonic() - clip_started, 3),
                }
            )
            private_entry["status"] = "prepared_local" if prepare_only else "ok"
        except Exception as error:
            failed += 1
            append_error(output_dir / "errors", f"clip-{position:04d}", error)
            execution_pricing = (
                _incremental_usage_since(clip_dir, usage_before)
                if clip_dir is not None
                else summarize_usage_files([])
            )
            clip_cost = float(execution_pricing["estimated_total_cost_usd"])
            lifetime_pricing = (
                summarize_usage_and_list_price(clip_dir)
                if clip_dir is not None
                else summarize_usage_files([])
            )
            total_cost += clip_cost
            public_entries.append(
                {
                    "source_asset_id": private_entry.get("source_asset_id", "unresolved"),
                    "duration_ms": private_entry.get("duration_ms"),
                    "status": "error",
                    "error_type": type(error).__name__,
                    "estimated_cost_usd": clip_cost,
                    "artifact_lifetime_cost_usd": lifetime_pricing[
                        "estimated_total_cost_usd"
                    ],
                    "execution_pricing": execution_pricing,
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


def _query_lock_target_description(target: EvidenceQueryTargetRef) -> str:
    """Render a locked target without adding domain assumptions."""
    parts = [target.target_description]
    if target.positive_attributes:
        parts.append("必須可由畫面確認的特徵：" + "；".join(target.positive_attributes))
    if target.negative_attributes:
        parts.append("不得混淆為：" + "；".join(target.negative_attributes))
    return "\n".join(parts)


def _query_lock_v2_target_description(target: EvidenceTargetIdentityV2) -> str:
    """Render persistent identity separately from temporary event state."""
    parts = [target.target_description, f"identity scope: {target.scope.value}"]
    if target.identity_cues:
        parts.append("可持續辨識的 identity cues：" + "；".join(target.identity_cues))
    if target.context_cues:
        parts.append(
            "只可作輔助、不得取代 identity 的 context cues："
            + "；".join(target.context_cues)
        )
    if target.stable_exclusions:
        parts.append("必須排除的相似實例或描繪：" + "；".join(target.stable_exclusions))
    if target.parent_target_id is not None:
        parts.append("parent target ID：" + target.parent_target_id)
    return "\n".join(parts)


def _select_query_lock_target(
    query_lock: EvidenceQueryLock | EvidenceQueryLockV2,
    query_target_id: str | None = None,
) -> EvidenceQueryTargetRef | EvidenceTargetIdentityV2:
    targets = (
        query_lock.targets
        if isinstance(query_lock, EvidenceQueryLock)
        else query_lock.identity.targets
    )
    if not targets:
        raise ValueError("query lock has no target suitable for geometry")
    if query_target_id is None:
        if len(targets) != 1:
            raise ValueError(
                "query lock contains multiple targets; select one with --query-target-id"
            )
        return targets[0]
    try:
        return next(
            target for target in targets if target.target_id == query_target_id
        )
    except StopIteration as error:
        raise ValueError(
            f"query lock does not contain target: {query_target_id}"
        ) from error


def _query_lock_event_description(
    event: FullClipEvent,
    query_lock: EvidenceQueryLock,
) -> str:
    """Combine observed event context with an immutable, generic evidence contract."""
    parts = [f"coarse event（僅供情境，不是空間證據）：{event.description}"]
    if query_lock.observable_predicate:
        parts.append("要驗證的可觀察條件：" + query_lock.observable_predicate)
    if query_lock.predicate_phases is not None:
        parts.extend(
            [
                "條件前證據：" + query_lock.predicate_phases.precondition,
                "條件頂點證據：" + query_lock.predicate_phases.apex,
                "條件後證據：" + query_lock.predicate_phases.postcondition,
            ]
        )
    if query_lock.required_evidence:
        parts.append("必要證據：" + "；".join(query_lock.required_evidence))
    if query_lock.negative_constraints:
        parts.append("排除條件：" + "；".join(query_lock.negative_constraints))
    return "\n".join(parts)


def _query_lock_v2_identity_context(
    identity: EvidenceIdentityContractV2,
    target: EvidenceTargetIdentityV2,
) -> str:
    """Exact-frame Grounding context for v2; intentionally excludes predicates."""
    parts = [
        "本次只驗證指定 identity，且只使用指定原始影格。",
        "不得用事件敘述、時間或前後影格推測座標；predicate 由獨立 temporal gate 處理。",
    ]
    if target.context_cues:
        parts.append("輔助情境線索：" + "；".join(target.context_cues))
    if target.stable_exclusions:
        parts.append("identity exclusions：" + "；".join(target.stable_exclusions))
    for ancestor in identity.ancestors(target.target_id):
        ancestor_parts = [
            f"parent instance disambiguator {ancestor.target_id}: "
            + ancestor.target_description
        ]
        if ancestor.identity_cues:
            ancestor_parts.append("identity cues=" + "；".join(ancestor.identity_cues))
        if ancestor.context_cues:
            ancestor_parts.append("context only=" + "；".join(ancestor.context_cues))
        if ancestor.stable_exclusions:
            ancestor_parts.append("exclude=" + "；".join(ancestor.stable_exclusions))
        parts.append("；".join(ancestor_parts))
    if identity.ancestors(target.target_id):
        parts.append(
            "parent instance material only disambiguates the selected subpart; "
            "the bbox must still tightly bound the requested child target."
        )
    return "\n".join(parts)


def _load_evidence_query_lock(path: Path) -> EvidenceQueryLock | EvidenceQueryLockV2:
    payload = read_json(path)
    if payload.get("contract_version") == "evidence-query-lock-v2":
        return EvidenceQueryLockV2.model_validate(payload)
    return EvidenceQueryLock.model_validate(payload)


def _resolve_query_identity_references(
    identity: EvidenceIdentityContractV2,
    target: EvidenceTargetIdentityV2,
    reference_dir: Path | None,
) -> tuple[GroundingIdentityReference, ...]:
    anchor_sources = (target, *identity.ancestors(target.target_id))
    anchors = [
        (source, role, anchor)
        for source in anchor_sources
        for role, source_anchors in (
            ("positive", source.positive_anchors),
            ("negative", source.negative_anchors),
        )
        for anchor in source_anchors
    ]
    if not anchors:
        return ()
    if reference_dir is None:
        raise ValueError(
            "query identity anchors require --query-reference-dir containing "
            "content-addressed crop images"
        )
    suffixes = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif")
    references: list[GroundingIdentityReference] = []
    # Gemini Grounding accepts four references. Select deterministically:
    # child before nearest/farther ancestors, and positive before negative
    # within each identity. The full approved lock remains saved for audit;
    # the exact transmitted subset is included in the Grounding request hash.
    for source, role, anchor in anchors[:4]:
        matches = [
            reference_dir / f"{anchor.crop_sha256}{suffix}" for suffix in suffixes
        ]
        path = next((candidate for candidate in matches if candidate.is_file()), None)
        if path is None:
            raise ValueError(
                "query identity reference is missing for crop SHA-256 "
                f"{anchor.crop_sha256}"
            )
        references.append(
            GroundingIdentityReference(
                reference_id=(
                    f"{role}:{source.target_id}:{anchor.frame_id}:"
                    f"{anchor.crop_sha256[:12]}"
                ),
                role=role,
                target_id=target.target_id,
                anchor_target_id=source.target_id,
                description=(
                    "same locked target instance"
                    if source.target_id == target.target_id and role == "positive"
                    else "explicit confuser that must not be selected"
                    if source.target_id == target.target_id
                    else "parent instance context for child-target disambiguation"
                    if role == "positive"
                    else "confusing parent instance that must be excluded"
                ),
                path=path,
                sha256=anchor.crop_sha256,
            )
        )
    return tuple(references)


def _matching_dense_seed_frame(
    selection: DenseEventSelection,
    catalog: DenseFrameCatalog,
    *,
    target_entity_id: str,
    target_description: str,
) -> DenseFrame | None:
    """Return a dense seed only when its semantic target and lineage still match."""
    if selection.source_asset_id != catalog.source_asset_id:
        raise ValueError("dense selection and catalog source assets differ")
    if selection.event_id != catalog.event_id:
        raise ValueError("dense selection and catalog event IDs differ")
    if (
        not selection.visible
        or selection.match_status != MatchStatus.MATCHED
        or selection.recommended_frame_id is None
        or selection.target_entity_id != target_entity_id
        or selection.target_description != target_description
    ):
        return None
    try:
        return next(
            frame
            for frame in catalog.frames
            if frame.frame_id == selection.recommended_frame_id
        )
    except StopIteration as error:
        raise ValueError("dense selection references a frame absent from its catalog") from error


def _query_temporal_seed_frame(
    decision: QueryTemporalDecision,
) -> tuple[Any | None, str]:
    """Choose a Grounding seed without changing predicate semantics."""
    if (
        decision.match_status != MatchStatus.MATCHED
        or decision.predicate_status != PredicateStatus.SATISFIED
    ):
        raise ValueError("query temporal decision is not matched+satisfied")
    if decision.required_at == PredicateRequiredAt.CANDIDATE:
        # Candidate eligibility does not require the eventual SAM seed itself to
        # exhibit a transient action or state.
        return None, "candidate_predicate_gate_only"
    if decision.required_at == PredicateRequiredAt.SEED:
        return decision.seed_frame, "query_predicate_seed_frame"
    if decision.required_at == PredicateRequiredAt.TRANSITION:
        return decision.apex_frame, "query_predicate_transition_apex"
    if decision.required_at == PredicateRequiredAt.INTERVAL:
        frames = decision.interval_sample_frames
        if not frames:
            raise ValueError("positive interval decision has no sampled frames")
        return frames[len(frames) // 2], "query_predicate_interval_mid_sample"
    raise ValueError(f"unsupported predicate application: {decision.required_at}")


def run_query_predicate_refinement(
    clip_run_dir: Path,
    event_id: str,
    *,
    query_lock_path: Path,
    query_target_id: str | None,
    prompt_template: str,
    sampling_fps: float = 8.0,
    window_ms: int = 4000,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Explicitly perform at most one paid temporal call for one locked event."""
    lock = _load_evidence_query_lock(query_lock_path)
    if not isinstance(lock, EvidenceQueryLockV2):
        raise ValueError("query predicate refinement requires an approved QueryLock v2")
    if lock.predicate is None:
        raise ValueError("query predicate refinement requires a predicate contract")
    locked_target = _select_query_lock_target(lock, query_target_id)
    if not isinstance(locked_target, EvidenceTargetIdentityV2):
        raise TypeError("QueryLock v2 resolved an incompatible target contract")
    grounding_target_id = locked_target.target_id
    if grounding_target_id not in lock.predicate.participant_target_ids:
        raise ValueError("selected temporal target must be a predicate participant")
    card = FullClipCard.model_validate(
        read_json(clip_run_dir / "gemini" / "clip-card" / "clip_card.json")
    )
    try:
        event = next(item for item in card.events if item.event_id == event_id)
    except StopIteration as error:
        raise ValueError(f"unknown Clip Card event: {event_id}") from error
    source_path = Path(read_json(clip_run_dir / "private-source.json")["path"])
    media = probe_video(source_path)
    if media.asset_id != card.source_asset_id:
        raise ValueError("private source identity differs from Clip Card")
    shots = ShotManifest.model_validate(read_json(clip_run_dir / "shots.json"))
    start_ms, end_ms, shot_id = dense_window_for_event(
        event,
        shots,
        window_ms=window_ms,
    )
    root = output_dir or clip_run_dir / "query-refinement" / event_id
    fps_key = str(sampling_fps).replace(".", "p")
    catalog_dir = root / f"dense-{fps_key}-{start_ms}-{end_ms}"
    catalog_path = catalog_dir / "dense-catalog.json"
    existing_candidates = [
        clip_run_dir / "dense" / event_id / "dense-catalog.json",
        catalog_path,
    ]
    catalog: DenseFrameCatalog | None = None
    catalog_reused = False
    selected_catalog_path = catalog_path
    for existing in existing_candidates:
        if not existing.exists():
            continue
        candidate = DenseFrameCatalog.model_validate(read_json(existing))
        if (
            candidate.source_asset_id == media.asset_id
            and candidate.event_id == event_id
            and candidate.sampling_fps == sampling_fps
            and candidate.source_start_ms == start_ms
            and candidate.source_end_ms == end_ms
        ):
            catalog = candidate
            catalog_reused = True
            selected_catalog_path = existing
            break
    if catalog is None:
        catalog = create_dense_event_catalog(
            source_path,
            media.asset_id,
            event,
            catalog_dir,
            sampling_fps=sampling_fps,
            window_start_ms=start_ms,
            window_end_ms=end_ms,
        )
    write_json(
        root / "dense-window.json",
        {
            "source_asset_id": media.asset_id,
            "event_id": event_id,
            "shot_id": shot_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "sampling_fps": sampling_fps,
            "catalog_reused": catalog_reused,
        },
    )
    response_schema = gemini_response_schema(QueryTemporalSelection)
    fingerprint = build_query_temporal_fingerprint(
        query_lock=lock,
        grounding_target_id=grounding_target_id,
        catalog=catalog,
        model_id=MODEL_ID,
        prompt_template=prompt_template,
        system_instruction=VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
        response_schema=response_schema,
    )
    run_dir = root / f"request-{fingerprint.request_sha256[:16]}"
    decision_path = run_dir / "query_temporal.decision.json"
    reused = False
    if decision_path.exists():
        decision = validate_query_temporal_evidence_bundle(
            decision_path,
            query_lock=lock,
            expected_system_instruction=VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
            expected_model_id=MODEL_ID,
            expected_prompt_template=prompt_template,
        )
        expected = {
            "source_asset_id": media.asset_id,
            "event_id": event_id,
            "query_id": lock.query_id,
            "grounding_target_id": grounding_target_id,
            "identity_sha256": fingerprint.identity_sha256,
            "predicate_sha256": fingerprint.predicate_sha256,
            "catalog_sha256": fingerprint.catalog_sha256,
            "request_sha256": fingerprint.request_sha256,
            "required_at": lock.predicate.required_at,
            "model_id": MODEL_ID,
        }
        actual = {
            **{
                key: getattr(decision, key)
                for key in expected
                if key != "model_id"
            },
            "model_id": decision.model_provenance.model_id,
        }
        if actual != expected:
            raise ValueError("cached query temporal decision lineage does not match request")
        write_query_temporal_consumer_lineage(
            run_dir,
            query_lock=lock,
            grounding_target_id=grounding_target_id,
            request_sha256=fingerprint.request_sha256,
        )
        reused = True
    else:
        client = GeminiLabClient()
        try:
            decision = client.refine_query_lock_frames(
                query_lock=lock,
                grounding_target_id=grounding_target_id,
                catalog=catalog,
                prompt_template=prompt_template,
                run_id=f"query-temporal-{uuid.uuid4().hex[:8]}",
                run_dir=run_dir,
            )
        finally:
            client.close()
    pricing = summarize_usage_and_list_price(run_dir)
    execution_pricing = summarize_usage_files(
        [] if reused else [run_dir / "query_temporal.raw_interaction.json"],
        relative_to=root,
    )
    result = {
        "source_asset_id": media.asset_id,
        "event_id": event_id,
        "query_id": lock.query_id,
        "grounding_target_id": grounding_target_id,
        "query_lock_sha256": lock.definition_sha256(),
        "component_hashes": lock.component_hashes(),
        "request_sha256": fingerprint.request_sha256,
        "decision_path": str(decision_path.resolve()),
        "catalog_path": str(selected_catalog_path.resolve()),
        "catalog_reused": catalog_reused,
        "gemini_request_reused": reused,
        "api_calls_this_execution": 0 if reused else 1,
        "coverage_claim": decision.coverage_claim,
        "match_status": decision.match_status,
        "predicate_status": decision.predicate_status,
        "pricing": pricing,
        "execution_pricing": execution_pricing,
    }
    write_json(root / "result.json", result)
    return result


def run_full_event_geometry(
    clip_run_dir: Path,
    event_id: str,
    *,
    grounding_prompt: str,
    checkpoint_path: Path | None = None,
    target_entity_id: str | None = None,
    target_description: str | None = None,
    accept_proposed_target: bool = False,
    grounding_candidate_number: int | None = None,
    query_lock_path: Path | None = None,
    query_target_id: str | None = None,
    query_reference_dir: Path | None = None,
    predicate_decision_path: Path | None = None,
    predicate_prompt_template: str | None = None,
    sam_analysis_fps: float = 2.0,
    identity_checkpoint_budget: int = 2,
) -> dict[str, Any]:
    """Ground one selected Clip Card event and optionally propagate SAM inside its interval."""
    if bool(target_entity_id) != bool(target_description):
        raise ValueError("target entity ID and description must be provided together")
    if target_entity_id is not None and accept_proposed_target:
        raise ValueError(
            "explicit target fields cannot be combined with --accept-proposed-target"
        )
    if checkpoint_path is None and grounding_candidate_number is not None:
        raise ValueError(
            "--grounding-candidate-number is only consumed when --sam-checkpoint is provided"
        )
    if query_target_id is not None and query_lock_path is None:
        raise ValueError("--query-target-id requires --query-lock")
    if query_reference_dir is not None and query_lock_path is None:
        raise ValueError("--query-reference-dir requires --query-lock")
    if predicate_decision_path is not None and query_lock_path is None:
        raise ValueError("--predicate-decision requires --query-lock")
    if identity_checkpoint_budget < 0 or identity_checkpoint_budget > 8:
        raise ValueError("identity checkpoint budget must be within 0..8")
    if query_lock_path is not None and (
        target_entity_id is not None or accept_proposed_target
    ):
        raise ValueError(
            "a query lock is the immutable target source and cannot be combined with "
            "explicit/proposed target flags"
        )
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
    query_lock: EvidenceQueryLock | EvidenceQueryLockV2 | None = None
    query_lock_sha256: str | None = None
    query_component_hashes: dict[str, str] | None = None
    grounding_identity_sha256: str | None = None
    identity_references: tuple[GroundingIdentityReference, ...] = ()
    query_temporal_decision: QueryTemporalDecision | None = None
    query_lock_artifact_path: Path | None = None
    grounding_event_description = event.description
    target_lock_source = "explicit_target"
    if query_lock_path is not None:
        query_lock = _load_evidence_query_lock(query_lock_path)
        locked_target = _select_query_lock_target(query_lock, query_target_id)
        target_entity_id = locked_target.target_id
        query_lock_sha256 = query_lock.definition_sha256()
        if isinstance(query_lock, EvidenceQueryLockV2):
            if not isinstance(locked_target, EvidenceTargetIdentityV2):
                raise TypeError("v2 query lock resolved an incompatible target contract")
            target_description = _query_lock_v2_target_description(locked_target)
            query_component_hashes = query_lock.component_hashes()
            grounding_identity_sha256 = query_component_hashes["identity_sha256"]
            identity_references = _resolve_query_identity_references(
                query_lock.identity,
                locked_target,
                query_reference_dir,
            )
            target_lock_source = (
                f"query_lock_v2:{query_lock.query_id}:revision:{query_lock.revision}:"
                f"approval:{query_lock.approval.approval_source.value}"
            )
            grounding_event_description = _query_lock_v2_identity_context(
                query_lock.identity, locked_target
            )
        else:
            if not isinstance(locked_target, EvidenceQueryTargetRef):
                raise TypeError("v1 query lock resolved an incompatible target contract")
            target_description = _query_lock_target_description(locked_target)
            grounding_identity_sha256 = query_lock_sha256
            target_lock_source = (
                f"query_lock:{query_lock.query_id}:revision:{query_lock.revision}"
            )
            grounding_event_description = _query_lock_event_description(event, query_lock)
    elif target_entity_id is None:
        if not accept_proposed_target:
            proposed = [target.entity_id for target in event.grounding_targets]
            raise ValueError(
                "an explicit Grounding target is required; provide target entity ID and "
                f"description, or explicitly accept the sole proposal. Proposed IDs: {proposed}"
            )
        if len(event.grounding_targets) != 1:
            raise ValueError(
                "--accept-proposed-target requires exactly one proposed Grounding target; "
                "multiple or absent proposals require explicit user selection"
            )
        selected_target = event.grounding_targets[0]
        target_entity_id = selected_target.entity_id
        target_description = selected_target.target_description
        target_lock_source = "explicit_acceptance_of_single_clip_card_proposal"

    if isinstance(query_lock, EvidenceQueryLockV2):
        if query_lock.predicate is not None:
            if predicate_decision_path is None:
                raise ValueError(
                    "QueryLock v2 contains a predicate; run refine-query-predicate and "
                    "provide --predicate-decision before exact-frame Grounding"
                )
            if predicate_prompt_template is None:
                raise ValueError(
                    "predicate decision validation requires the current predicate prompt"
                )
            query_temporal_decision = validate_query_temporal_evidence_bundle(
                predicate_decision_path,
                query_lock=query_lock,
                expected_system_instruction=VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
                expected_model_id=MODEL_ID,
                expected_prompt_template=predicate_prompt_template,
            )
            expected_temporal = {
                "source_asset_id": media.asset_id,
                "event_id": event_id,
                "query_id": query_lock.query_id,
                "identity_sha256": query_lock.component_hashes()["identity_sha256"],
                "predicate_sha256": query_lock.component_hashes()["predicate_sha256"],
                "required_at": query_lock.predicate.required_at,
                "model_id": MODEL_ID,
            }
            actual_temporal = {
                **{
                    key: getattr(query_temporal_decision, key)
                    for key in expected_temporal
                    if key != "model_id"
                },
                "model_id": query_temporal_decision.model_provenance.model_id,
            }
            if actual_temporal != expected_temporal:
                raise ValueError(
                    "predicate decision does not match the selected QueryLock/event/model"
                )
            if (
                query_temporal_decision.grounding_target_id
                not in query_lock.predicate.participant_target_ids
            ):
                raise ValueError(
                    "predicate decision temporal target is not a locked predicate participant"
                )
            write_query_temporal_consumer_lineage(
                predicate_decision_path.parent,
                query_lock=query_lock,
                grounding_target_id=query_temporal_decision.grounding_target_id,
                request_sha256=query_temporal_decision.request_sha256,
            )
        elif predicate_decision_path is not None:
            raise ValueError("predicate decision was supplied for a lock without a predicate")
    elif predicate_decision_path is not None:
        raise ValueError("predicate decisions are supported only by QueryLock v2")

    dense_selection_path = clip_run_dir / "dense" / event_id / "gemini" / "dense_selection.json"
    seed_source = "clip_card_recommended_mmss"
    predicate_gate_source: str | None = None
    requested_time_ms = derived.recommended_keyframe_ms
    expected_dense_frame_pts: int | None = None
    tracking_allowed_start_ms = derived.start_ms
    tracking_allowed_end_ms = derived.end_ms
    tracking_eligibility_source = "derived_event_interval"
    if query_temporal_decision is not None:
        temporal_frame, temporal_seed_source = _query_temporal_seed_frame(
            query_temporal_decision
        )
        if temporal_frame is not None:
            requested_time_ms = temporal_frame.requested_time_ms
            expected_dense_frame_pts = temporal_frame.frame_pts
            seed_source = temporal_seed_source
        else:
            predicate_gate_source = temporal_seed_source
        if query_temporal_decision.required_at == PredicateRequiredAt.INTERVAL:
            interval_frames = query_temporal_decision.interval_sample_frames
            if len(interval_frames) < 2:
                raise ValueError("positive interval predicate has no usable evidence bracket")
            final_step_ms = max(
                1,
                interval_frames[-1].requested_time_ms
                - interval_frames[-2].requested_time_ms,
            )
            tracking_allowed_start_ms = max(
                derived.start_ms, interval_frames[0].frame_time_ms
            )
            tracking_allowed_end_ms = min(
                derived.end_ms,
                interval_frames[-1].frame_time_ms + final_step_ms,
            )
            tracking_eligibility_source = (
                "query_predicate_contiguous_sampled_evidence_bracket"
            )
    # Existing dense selections were generated under the Clip Card target, not
    # under an EvidenceQueryLock. Until dense artifacts carry a lock hash, never
    # reuse them for a lock-governed request, even if an entity ID happens to match.
    if (
        dense_selection_path.exists()
        and query_lock is None
        and query_temporal_decision is None
    ):
        dense_selection = DenseEventSelection.model_validate(read_json(dense_selection_path))
        dense_catalog = DenseFrameCatalog.model_validate(
            read_json(clip_run_dir / "dense" / event_id / "dense-catalog.json")
        )
        dense_frame = _matching_dense_seed_frame(
            dense_selection,
            dense_catalog,
            target_entity_id=str(target_entity_id),
            target_description=str(target_description),
        )
        if dense_frame is not None:
            requested_time_ms = dense_frame.requested_time_ms
            expected_dense_frame_pts = dense_frame.frame_pts
            seed_source = f"dense_frame_id:{dense_frame.frame_id}:target_matched"
    if requested_time_ms is None:
        requested_time_ms = (derived.start_ms + derived.end_ms) // 2
        seed_source = "local_event_midpoint_no_model_keyframe"

    geometry_dir = clip_run_dir / "geometry" / event_id
    if query_lock is not None:
        query_lock_artifact_path = (
            geometry_dir / f"query-lock-{query_lock_sha256[:16]}.json"
            if isinstance(query_lock, EvidenceQueryLockV2)
            else geometry_dir / "query-lock.json"
        )
        write_json(
            query_lock_artifact_path,
            {
                "definition_sha256": query_lock_sha256,
                "component_hashes": query_component_hashes,
                "query_lock": query_lock.model_dump(mode="json"),
            },
        )
    frame_path = geometry_dir / "evidence-frames" / f"requested-{requested_time_ms}.png"
    frame = extract_frame(source_path, requested_time_ms, frame_path)
    if expected_dense_frame_pts is not None and frame.frame_pts != expected_dense_frame_pts:
        raise ValueError(
            "dense frame lineage mismatch: the original-resolution extraction did not "
            f"resolve to the selected frame PTS ({frame.frame_pts} != "
            f"{expected_dense_frame_pts})"
        )
    grounding_request_key = {
        "contract_version": "exact-frame-grounding-v2",
        "model_id": MODEL_ID,
        "source_asset_id": media.asset_id,
        "event_id": event.event_id,
        "frame_hash": frame.frame_hash,
        "frame_pts": frame.frame_pts,
        "frame_time_ms": frame.frame_time_ms,
        "source_width": frame.width,
        "source_height": frame.height,
        "dense_frame_expected_pts": expected_dense_frame_pts,
        "target_entity_id": target_entity_id,
        "target_description": target_description,
        "target_lock_source": (
            "evidence_query_lock_v2_identity"
            if isinstance(query_lock, EvidenceQueryLockV2)
            else target_lock_source
        ),
        # A v2 Grounding cache depends on identity, not predicate or framing.
        # The complete lock hash is saved as lineage outside this fingerprint.
        "query_lock_sha256": (
            query_lock_sha256
            if isinstance(query_lock, EvidenceQueryLock)
            else None
        ),
        "grounding_identity_sha256": grounding_identity_sha256,
        "identity_reference_selection_version": (
            "child-then-nearest-ancestors-positive-then-negative-v1"
        ),
        "identity_references": [
            {
                "reference_id": reference.reference_id,
                "role": reference.role,
                "target_id": reference.target_id,
                "anchor_target_id": (
                    reference.anchor_target_id or reference.target_id
                ),
                "sha256": reference.sha256,
            }
            for reference in identity_references
        ],
        "event_description": grounding_event_description,
        "prompt_sha256": hashlib.sha256(grounding_prompt.encode("utf-8")).hexdigest(),
        "system_instruction_sha256": hashlib.sha256(
            VISUAL_EVIDENCE_SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "response_schema_sha256": hashlib.sha256(
            json.dumps(
                gemini_response_schema(GeminiNativeGroundingProposal),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "thinking_level": "low",
    }
    grounding_fingerprint = hashlib.sha256(
        json.dumps(
            grounding_request_key,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    grounding_dir = geometry_dir / "grounding" / f"bbox-{grounding_fingerprint[:16]}"
    write_json(
        grounding_dir / "request-key.json",
        {
            **grounding_request_key,
            "request_fingerprint": grounding_fingerprint,
        },
    )
    if isinstance(query_lock, EvidenceQueryLockV2):
        write_json(
            grounding_dir / f"query-lineage-{query_lock_sha256[:16]}.json",
            {
                "query_lock_sha256": query_lock_sha256,
                "component_hashes": query_component_hashes,
                "temporal_decision_request_sha256": (
                    query_temporal_decision.request_sha256
                    if query_temporal_decision is not None
                    else None
                ),
            },
        )
    if isinstance(query_lock, EvidenceQueryLockV2):
        write_json(grounding_dir / "frame.json", frame)
        write_json(
            grounding_dir / f"query-consumer-{query_lock_sha256[:16]}.json",
            {
                "query_lock_sha256": query_lock_sha256,
                "seed_source": seed_source,
                "predicate_gate_source": predicate_gate_source,
                "coarse_event_start_mmss": event.start_mmss,
                "coarse_event_end_mmss": event.end_mmss,
                "temporal_decision_request_sha256": (
                    query_temporal_decision.request_sha256
                    if query_temporal_decision is not None
                    else None
                ),
            },
        )
    else:
        write_json(
            grounding_dir / "frame.json",
            {
                **frame.model_dump(mode="json"),
                "seed_source": seed_source,
                "predicate_gate_source": predicate_gate_source,
                "coarse_event_start_mmss": event.start_mmss,
                "coarse_event_end_mmss": event.end_mmss,
            },
        )
    started = monotonic()
    grounding_path = grounding_dir / "grounding.json"
    if grounding_path.exists():
        proposal = GroundingProposal.model_validate(read_json(grounding_path))
        require_grounding_request_match(
            proposal,
            asset_id=media.asset_id,
            event_id=event.event_id,
            entity_id=str(target_entity_id),
            frame_pts=frame.frame_pts,
            frame_time_ms=frame.frame_time_ms,
            frame_hash=frame.frame_hash,
            source_width=frame.width,
            source_height=frame.height,
            model_id=MODEL_ID,
        )
        grounding_reused = True
    else:
        client = GeminiLabClient()
        try:
            proposal = client.ground_frame(
                media=media,
                frame=frame,
                event_id=event.event_id,
                event_description=grounding_event_description,
                entity_id=str(target_entity_id),
                target_description=str(target_description),
                prompt_template=grounding_prompt,
                run_id=f"full-ground-{uuid.uuid4().hex[:8]}",
                output_dir=grounding_dir,
                identity_references=identity_references,
            )
        finally:
            client.close()
        grounding_reused = False
    grounding_seconds = round(monotonic() - started, 3)
    draw_grounding_overlay(frame_path, proposal, grounding_dir / "debug.png")

    track: SegmentationTrack | None = None
    selected_seed = None
    track_path: Path | None = None
    identity_checkpoint_plan_path: Path | None = None
    if checkpoint_path is not None:
        if not proposal.visible or not proposal.candidates:
            raise ValueError(f"Gemini Grounding target is not visible for {event_id}")
        selected_seed = require_tracking_seed_candidate(
            proposal,
            candidate_number=grounding_candidate_number,
            require_predicate_satisfied=bool(
                isinstance(query_lock, EvidenceQueryLock)
                and query_lock.observable_predicate
            ),
        )
        tracking_scdet_threshold = 4.0
        tracking_seed_box_padding_ratio = 0.04
        tracking_max_side = 960
        tracking_device = "auto"
        tracking_shots = detect_shots_ffmpeg(
            source_path,
            threshold=tracking_scdet_threshold,
        )
        expected_analysis_start_ms, expected_analysis_end_ms = resolve_tracking_interval(
            tracking_shots,
            seed_time_ms=frame.frame_time_ms,
            allowed_start_ms=tracking_allowed_start_ms,
            allowed_end_ms=tracking_allowed_end_ms,
        )
        checkpoint_sha256 = sha256_file(checkpoint_path)
        seed_manifest = {
            "contract_version": "bbox-seed-v2-exact-pts",
            "asset_id": proposal.asset_id,
            "event_id": proposal.event_id,
            "entity_id": proposal.entity_id,
            "target_description": target_description,
            "target_lock_source": (
                "evidence_query_lock_v2_identity"
                if isinstance(query_lock, EvidenceQueryLockV2)
                else target_lock_source
            ),
            "query_lock_sha256": (
                query_lock_sha256
                if isinstance(query_lock, EvidenceQueryLock)
                else None
            ),
            "grounding_identity_sha256": grounding_identity_sha256,
            "frame_hash": proposal.frame_hash,
            "frame_pts": proposal.frame_pts,
            "candidate_number": selected_seed.candidate_number,
            "candidate_index": selected_seed.candidate_index,
            "candidate_selection_source": selected_seed.selection_source,
            "box_2d": list(selected_seed.candidate.box_2d),
            "seed_type": "gemini_bbox",
            "source_start_ms": tracking_allowed_start_ms,
            "source_end_ms": tracking_allowed_end_ms,
            "tracking_eligibility_source": tracking_eligibility_source,
            "normalized_seed_shot_start_ms": expected_analysis_start_ms,
            "normalized_seed_shot_end_ms": expected_analysis_end_ms,
            "analysis_fps": sam_analysis_fps,
            "analysis_max_side": tracking_max_side,
            "ffmpeg_scdet_threshold": tracking_scdet_threshold,
            "seed_box_padding_ratio": tracking_seed_box_padding_ratio,
            "device_request": tracking_device,
            "sam_config": SAM21_CONFIG,
            "sam_implementation_revision": SAM21_IMPLEMENTATION_REVISION,
            "checkpoint_sha256": checkpoint_sha256,
        }
        seed_fingerprint = hashlib.sha256(
            json.dumps(seed_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        track_dir = geometry_dir / "sam21" / f"bbox-{seed_fingerprint[:16]}"
        seed_manifest_path = track_dir / "seed-selection.json"
        write_json(
            seed_manifest_path,
            {
                **seed_manifest,
                "seed_fingerprint": seed_fingerprint,
            },
        )
        if isinstance(query_lock, EvidenceQueryLockV2):
            write_json(
                track_dir / f"query-lineage-{query_lock_sha256[:16]}.json",
                {
                    "query_lock_sha256": query_lock_sha256,
                    "component_hashes": query_component_hashes,
                    "temporal_decision_request_sha256": (
                        query_temporal_decision.request_sha256
                        if query_temporal_decision is not None
                        else None
                    ),
                },
            )
        track_path = track_dir / "segmentation-track.json"
        if track_path.exists():
            track = SegmentationTrack.model_validate(read_json(track_path))
        else:
            track = track_bbox_sam21(
                video_path=source_path,
                checkpoint_path=checkpoint_path,
                seed_time_ms=frame.frame_time_ms,
                seed_box_2d=selected_seed.candidate.box_2d,
                target_description=str(target_description),
                output_dir=track_dir,
                seed_source=str(seed_manifest_path),
                asset_id=proposal.asset_id,
                seed_frame_pts=proposal.frame_pts,
                seed_frame_sha256=proposal.frame_hash,
                seed_source_width=proposal.source_width,
                seed_source_height=proposal.source_height,
                analysis_fps=sam_analysis_fps,
                max_side=tracking_max_side,
                device=tracking_device,
                ffmpeg_scdet_threshold=tracking_scdet_threshold,
                seed_box_padding_ratio=tracking_seed_box_padding_ratio,
                allowed_start_ms=tracking_allowed_start_ms,
                allowed_end_ms=tracking_allowed_end_ms,
            )
        require_bbox_track_request_match(
            track,
            video_path=source_path,
            asset_id=proposal.asset_id,
            target_description=str(target_description),
            seed_time_ms=frame.frame_time_ms,
            seed_box_2d=selected_seed.candidate.box_2d,
            seed_box_padding_ratio=tracking_seed_box_padding_ratio,
            analysis_fps=sam_analysis_fps,
            analysis_start_ms=expected_analysis_start_ms,
            analysis_end_ms=expected_analysis_end_ms,
            checkpoint_sha256=checkpoint_sha256,
            seed_frame_pts=proposal.frame_pts,
            seed_frame_sha256=proposal.frame_hash,
            seed_source_width=proposal.source_width,
            seed_source_height=proposal.source_height,
        )
        identity_plan = plan_identity_checkpoints(
            track.samples,
            asset_id=track.asset_id,
            track_fingerprint=sha256_file(track_path),
            identity_sha256=grounding_identity_sha256,
            max_model_checks=identity_checkpoint_budget,
            seed_sample_index=track.seed_sample_index,
        )
        identity_checkpoint_plan_path = (
            track_dir
            / "identity-checkpoint-plans"
            / f"plan-{identity_plan.planning_request_sha256[:16]}.json"
        )
        if identity_checkpoint_plan_path.exists():
            existing_identity_plan = read_json(identity_checkpoint_plan_path)
            if existing_identity_plan != identity_plan.model_dump(mode="json"):
                raise ValueError("cached identity checkpoint plan was modified")
        write_json(identity_checkpoint_plan_path, identity_plan)
        write_json(
            track_dir / "identity-checkpoint-plan.current.json",
            {
                "artifact_type": "identity_checkpoint_plan_pointer_v1",
                "planning_request_sha256": identity_plan.planning_request_sha256,
                "path": str(identity_checkpoint_plan_path.resolve()),
            },
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
        "target_lock_source": target_lock_source,
        "query_lock_sha256": query_lock_sha256,
        "query_lock_artifact_path": (
            str(query_lock_artifact_path.resolve())
            if query_lock_artifact_path is not None
            else None
        ),
        "query_component_hashes": query_component_hashes,
        "grounding_identity_sha256": grounding_identity_sha256,
        "query_temporal_decision_path": (
            str(predicate_decision_path.resolve())
            if predicate_decision_path is not None
            else None
        ),
        "query_temporal_request_sha256": (
            query_temporal_decision.request_sha256
            if query_temporal_decision is not None
            else None
        ),
        "query_temporal_coverage_claim": (
            query_temporal_decision.coverage_claim
            if query_temporal_decision is not None
            else None
        ),
        "identity_reference_count": len(identity_references),
        "grounding_request_fingerprint": grounding_fingerprint,
        "grounding_path": str(grounding_path.resolve()),
        "grounding_debug_path": str((grounding_dir / "debug.png").resolve()),
        "grounding_candidate_number": (
            selected_seed.candidate_number if selected_seed is not None else None
        ),
        "grounding_candidate_index": (
            selected_seed.candidate_index if selected_seed is not None else None
        ),
        "grounding_candidate_selection_source": (
            selected_seed.selection_source if selected_seed is not None else None
        ),
        "sam_seed_type": "bbox" if checkpoint_path is not None else None,
        "seed_source": seed_source,
        "predicate_gate_source": predicate_gate_source,
        "tracking_eligibility_source": tracking_eligibility_source,
        "tracking_allowed_start_ms": tracking_allowed_start_ms,
        "tracking_allowed_end_ms": tracking_allowed_end_ms,
        "frame_pts": frame.frame_pts,
        "frame_time_ms": frame.frame_time_ms,
        "frame_hash": frame.frame_hash,
        "grounding_visible": proposal.visible,
        "grounding_candidate_count": len(proposal.candidates),
        "grounding_reused": grounding_reused,
        "grounding_seconds": grounding_seconds,
        "sam_track_path": (
            str(track_path.resolve()) if track is not None and track_path is not None else None
        ),
        "sam_total_samples": track.total_samples if track is not None else 0,
        "identity_checkpoint_budget": identity_checkpoint_budget,
        "identity_checkpoint_plan_path": (
            str(identity_checkpoint_plan_path.resolve())
            if identity_checkpoint_plan_path is not None
            else None
        ),
        "identity_checkpoint_model_calls_made": 0,
        "pricing": pricing,
        "execution_pricing": execution_pricing,
    }
    write_json(geometry_dir / "result.json", result)
    return result
