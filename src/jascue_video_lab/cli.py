from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from PIL import Image

from .ab_review import render_grounding_ab_review
from .billing import summarize_usage_and_list_price
from .compare import compare_runs
from .feature_cut import run_feature_cut_experiment
from .fixtures import generate_fixtures
from .full_v1 import run_full_clip, run_full_event_geometry, run_full_library
from .gemini import GeminiLabClient
from .media import extract_frame, probe_video, sha256_file
from .models import (
    ContentMap,
    ExtractedFrame,
    GroundingProposal,
    MediaInfo,
    TargetCandidateMap,
    TemporalEvent,
    TemporalMap,
    RushesCatalog,
    RushesEditPlan,
)
from .overlay import draw_grounding_overlay
from .review import render_manual_review
from .repeat import run_repeated_grounding
from .rushes import create_rushes_catalog, render_rushes_edit, run_rushes_experiment
from .sam_tracking import compare_segmentation_to_bbox_track, track_bbox_sam21
from .shots import detect_shots_ffmpeg
from .storage import append_error, read_json, write_json
from .timeline import render_direct_moment_timeline, render_temporal_timeline, render_timeline
from .tracking import track_bbox_csrt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_name(value: str) -> str:
    clean = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return clean[:100] or "unnamed"


def _load_prompt(name: str) -> str:
    return (PROJECT_ROOT / "prompts" / name).read_text(encoding="utf-8")


def _default_artifact_root(asset_hash: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "artifacts" / asset_hash[:12] / stamp


def _artifact_upload_source(artifact_root: Path) -> Path:
    identity_path = artifact_root / "upload_source_identity.json"
    if identity_path.exists():
        identity = read_json(identity_path)
        if not identity.get("used_analysis_proxy", False):
            sources = sorted(artifact_root.glob("source.*"))
            if sources:
                return sources[0]
    proxy = artifact_root / "analysis-proxy.mp4"
    if proxy.exists():
        return proxy
    sources = sorted(artifact_root.glob("source.*"))
    if not sources:
        raise FileNotFoundError(f"no analysis-proxy.mp4 or source.* in {artifact_root}")
    return sources[0]


def _find_proposals(run_dir: Path) -> list[tuple[GroundingProposal, Path]]:
    proposals = []
    for path in run_dir.glob("events/*/groundings/*/grounding.json"):
        proposal = GroundingProposal.model_validate(read_json(path))
        overlay = path.parent / "debug.png"
        if overlay.exists():
            proposals.append((proposal, overlay))
    return proposals


def command_probe(args: argparse.Namespace) -> int:
    media = probe_video(args.video)
    if args.output:
        write_json(args.output, media)
    print(media.model_dump_json(indent=2))
    return 0


def command_extract(args: argparse.Namespace) -> int:
    frame = extract_frame(args.video, args.time_ms, args.output)
    metadata = args.output.with_suffix(args.output.suffix + ".json")
    write_json(metadata, frame)
    print(frame.model_dump_json(indent=2))
    return 0


def command_upload(args: argparse.Namespace) -> int:
    total_started = monotonic()
    media = probe_video(args.video)
    artifact_root = args.output
    artifact_root.mkdir(parents=True, exist_ok=True)
    existing_media_path = artifact_root / "media.json"
    existing_upload_asset_id: str | None = None
    if existing_media_path.exists():
        existing_media = MediaInfo.model_validate(read_json(existing_media_path))
        if existing_media.asset_id != media.asset_id:
            raise ValueError(
                "artifact root already belongs to a different source asset; use a new output directory"
            )
        existing_upload_asset_id = existing_media.asset_id
    upload_identity_path = artifact_root / "upload_source_identity.json"
    if upload_identity_path.exists():
        saved_identity = read_json(upload_identity_path)
        existing_upload_asset_id = saved_identity.get("upload_asset_id")
    elif (artifact_root / "analysis_proxy.json").exists():
        saved_proxy = read_json(artifact_root / "analysis_proxy.json")
        existing_upload_asset_id = saved_proxy.get("proxy_media", {}).get("asset_id")
    write_json(existing_media_path, media)
    source_link = artifact_root / ("source" + args.video.suffix.lower())
    if not source_link.exists():
        source_link.symlink_to(args.video.resolve(strict=True))
    upload_source = source_link
    upload_asset_id = media.asset_id
    proxy_record: dict[str, object] | None = None
    if args.analysis_proxy:
        proxy_media = probe_video(args.analysis_proxy)
        duration_delta_ms = abs(proxy_media.duration_ms - media.duration_ms)
        if duration_delta_ms > args.max_proxy_duration_delta_ms:
            raise ValueError(
                f"proxy duration differs by {duration_delta_ms} ms; maximum is "
                f"{args.max_proxy_duration_delta_ms} ms"
            )
        proxy_link = artifact_root / "analysis-proxy.mp4"
        if not proxy_link.exists():
            proxy_link.symlink_to(args.analysis_proxy.resolve(strict=True))
        upload_source = proxy_link
        upload_asset_id = proxy_media.asset_id
        proxy_record = {
            "purpose": "Gemini semantic video analysis only; original source remains authoritative",
            "proxy_media": proxy_media.model_dump(mode="json"),
            "duration_delta_ms": duration_delta_ms,
            "original_bytes": args.video.stat().st_size,
            "proxy_bytes": args.analysis_proxy.stat().st_size,
            "byte_reduction_ratio": round(
                1 - args.analysis_proxy.stat().st_size / args.video.stat().st_size, 8
            ),
            "grounding_source": str(source_link),
            "upload_source": str(proxy_link),
        }
    if (
        (artifact_root / "upload" / "file_upload_initial.json").exists()
        and existing_upload_asset_id
        and existing_upload_asset_id != upload_asset_id
        and not args.force_reupload
    ):
        raise ValueError(
            "saved File API object was created from a different upload source; "
            "use a new artifact root or explicitly pass --force-reupload"
        )
    if proxy_record is not None:
        write_json(artifact_root / "analysis_proxy.json", proxy_record)
    upload_started = monotonic()
    client = GeminiLabClient(temperature=args.temperature)
    try:
        uploaded, reused = client.ensure_video_upload(
            upload_source,
            artifact_root / "upload",
            force_reupload=args.force_reupload,
        )
    finally:
        client.close()
    write_json(
        upload_identity_path,
        {
            "source_asset_id": media.asset_id,
            "upload_asset_id": upload_asset_id,
            "used_analysis_proxy": bool(args.analysis_proxy),
        },
    )
    upload_elapsed = monotonic() - upload_started
    timing = {
        "upload_and_file_processing_seconds": round(upload_elapsed, 6),
        "total_prepare_seconds": round(monotonic() - total_started, 6),
    }
    write_json(artifact_root / "upload_timing.json", timing)
    result = {
        "artifact_root": str(artifact_root),
        "asset_id": media.asset_id,
        "uploaded_file_name": uploaded.name,
        "uploaded_state": uploaded.state.name if uploaded.state else None,
        "reused_file_api_object": reused,
        "used_analysis_proxy": bool(args.analysis_proxy),
        "proxy": proxy_record,
        "timing": timing,
    }
    write_json(artifact_root / "upload_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_target_suggestions(
    *,
    artifact_root: Path,
    output_dir: Path,
    runs: int,
    temperature: float,
) -> int:
    started = monotonic()
    media = MediaInfo.model_validate(read_json(artifact_root / "media.json"))
    output_dir.mkdir(parents=True, exist_ok=True)
    client = GeminiLabClient(temperature=temperature)
    summaries: list[dict[str, object]] = []
    failures = 0
    try:
        uploaded, reused = client.ensure_video_upload(
            _artifact_upload_source(artifact_root), artifact_root / "upload"
        )
        for run_number in range(1, runs + 1):
            run_dir = output_dir / f"run-{run_number:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_id = f"target-candidates-{run_number:02d}-{uuid.uuid4().hex[:8]}"
            run_started = monotonic()
            try:
                candidate_map = client.suggest_targets(
                    media=media,
                    uploaded=uploaded,
                    prompt_template=_load_prompt("target_candidates_mmss_zh-TW.txt"),
                    run_id=run_id,
                    run_dir=run_dir,
                )
                summaries.append(
                    {
                        "run": f"run-{run_number:02d}",
                        "schema_valid": True,
                        "elapsed_seconds": round(monotonic() - run_started, 6),
                        "candidates": [
                            {
                                "candidate_id": candidate.candidate_id,
                                "label": candidate.label,
                                "entity_kind": candidate.entity_kind.value,
                                "target_description": candidate.target_description,
                                "representative_timestamp_mmss": (
                                    candidate.representative_timestamp_mmss
                                ),
                                "confidence": candidate.confidence,
                            }
                            for candidate in candidate_map.candidates
                        ],
                    }
                )
            except Exception as error:
                failures += 1
                append_error(run_dir, "target_candidate_pipeline", error)
                summaries.append(
                    {
                        "run": f"run-{run_number:02d}",
                        "schema_valid": False,
                        "elapsed_seconds": round(monotonic() - run_started, 6),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    finally:
        client.close()
    pricing = summarize_usage_and_list_price(output_dir)
    result = {
        "model": "gemini-3.5-flash",
        "method": "user-selectable target candidates before timing or Grounding",
        "file_api_object_reused": reused,
        "runs_requested": runs,
        "runs_succeeded": runs - failures,
        "failure_count": failures,
        "elapsed_seconds": round(monotonic() - started, 6),
        "pricing": pricing,
        "runs": summaries,
        "next_step": (
            "Pass --candidate-map RUN/target_candidates.json and --candidate-id ID "
            "to direct-moment-repeat."
        ),
    }
    write_json(output_dir / "pricing.json", pricing)
    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 1


def command_suggest_targets(args: argparse.Namespace) -> int:
    return _run_target_suggestions(
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
        runs=args.runs,
        temperature=args.temperature,
    )


def command_serve_review(args: argparse.Namespace) -> int:
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_network:
        raise ValueError("non-loopback host requires explicit --allow-network")
    import uvicorn

    uvicorn.run(
        "jascue_video_lab.webapp:app",
        host=args.host,
        port=args.port,
        reload=False,
        access_log=True,
    )
    return 0


def command_pricing(args: argparse.Namespace) -> int:
    summary = summarize_usage_and_list_price(args.artifact_root)
    output = args.output or (args.artifact_root / "pricing.json")
    write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_fixtures(args: argparse.Namespace) -> int:
    paths = generate_fixtures(args.output_dir)
    for path in paths:
        media = probe_video(path)
        write_json(path.with_suffix(".media.json"), media)
        print(path)
    return 0


def command_timeline(args: argparse.Namespace) -> int:
    content = ContentMap.model_validate(read_json(args.run_dir / "content_map.json"))
    output = args.output or (args.run_dir / "index.html")
    render_timeline(
        content_map=content,
        video_path=args.video,
        proposals=_find_proposals(args.run_dir),
        output_path=output,
    )
    print(output)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    report = compare_runs(args.run_dirs, args.output, args.annotations)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def command_review(args: argparse.Namespace) -> int:
    output = render_manual_review(args.artifact_root, args.annotations, args.output_dir)
    print(output.resolve())
    return 0


def command_ground_repeat(args: argparse.Namespace) -> int:
    summary = run_repeated_grounding(
        artifact_root=args.artifact_root,
        frame_json=args.frame_json,
        prompt_template=_load_prompt("grounding_native_yxyx_zh-TW.txt"),
        event_id=args.event_id,
        event_description=args.event_description,
        entity_id=args.entity_id,
        target_description=args.target_description,
        output_dir=args.output_dir,
        runs=args.runs,
        temperature=args.temperature,
        reference_box=args.reference_box,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 1


def command_track_csrt(args: argparse.Namespace) -> int:
    if args.grounding_json:
        grounding = GroundingProposal.model_validate(read_json(args.grounding_json))
        if not grounding.visible or not grounding.candidates:
            raise ValueError("grounding seed must be visible and contain at least one candidate")
        seed_time_ms = grounding.frame_time_ms
        seed_box = grounding.candidates[0].box_2d
        seed_source = f"Gemini GroundingProposal:{args.grounding_json}"
    else:
        if args.seed_time_ms is None or args.seed_box is None:
            raise ValueError("provide --grounding-json or both --seed-time-ms and --seed-box")
        seed_time_ms = args.seed_time_ms
        seed_box = args.seed_box
        seed_source = "manual canonical bbox"
    result = track_bbox_csrt(
        video_path=args.video,
        seed_time_ms=seed_time_ms,
        seed_box_2d=seed_box,
        target_description=args.target_description,
        output_dir=args.output_dir,
        seed_source=seed_source,
        analysis_fps=args.analysis_fps,
        max_side=args.max_side,
        appearance_threshold=args.appearance_threshold,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "samples"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_track_sam21(args: argparse.Namespace) -> int:
    if args.grounding_json:
        grounding = GroundingProposal.model_validate(read_json(args.grounding_json))
        if not grounding.visible or not grounding.candidates:
            raise ValueError("grounding seed must be visible and contain at least one candidate")
        seed_time_ms = grounding.frame_time_ms
        seed_box = grounding.candidates[0].box_2d
        seed_source = f"Gemini GroundingProposal:{args.grounding_json}"
        asset_id = grounding.asset_id
    else:
        if args.seed_time_ms is None or args.seed_box is None:
            raise ValueError("provide --grounding-json or both --seed-time-ms and --seed-box")
        seed_time_ms = args.seed_time_ms
        seed_box = args.seed_box
        seed_source = "manual canonical bbox"
        asset_id = None
    result = track_bbox_sam21(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        seed_time_ms=seed_time_ms,
        seed_box_2d=seed_box,
        target_description=args.target_description,
        output_dir=args.output_dir,
        seed_source=seed_source,
        asset_id=asset_id,
        analysis_fps=args.analysis_fps,
        max_side=args.max_side,
        device=args.device,
        ffmpeg_scdet_threshold=args.ffmpeg_scdet_threshold,
        seed_box_padding_ratio=args.seed_box_padding_ratio,
    )
    print(result.model_dump_json(indent=2, exclude={"samples"}))
    return 0


def command_compare_trackers(args: argparse.Namespace) -> int:
    result = compare_segmentation_to_bbox_track(
        args.segmentation_json, args.reference_bbox_json, args.output
    )
    print(result.model_dump_json(indent=2, exclude={"samples"}))
    return 0


def command_detect_shots(args: argparse.Namespace) -> int:
    result = detect_shots_ffmpeg(
        args.video, threshold=args.threshold, output_path=args.output
    )
    print(result.model_dump_json(indent=2))
    return 0


def command_catalog_rushes(args: argparse.Namespace) -> int:
    catalog = create_rushes_catalog(
        args.source_directory,
        args.output_dir,
        sample_interval_ms=args.sample_interval_ms,
        max_width=args.max_width,
    )
    print(
        json.dumps(
            {
                "catalog_id": catalog.catalog_id,
                "clip_count": len(catalog.clips),
                "frame_count": len(catalog.frames),
                "total_duration_ms": catalog.total_duration_ms,
                "analysis_reel_path": catalog.analysis_reel_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_rushes_run(args: argparse.Namespace) -> int:
    result = run_rushes_experiment(
        args.source_directory,
        args.output_dir,
        prompt_template=_load_prompt("rushes_selects_zh-TW.txt"),
        sample_interval_ms=args.sample_interval_ms,
        temperature=args.temperature,
        scdet_threshold=args.scdet_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_render_rushes(args: argparse.Namespace) -> int:
    catalog = RushesCatalog.model_validate(read_json(args.catalog_json))
    plan = RushesEditPlan.model_validate(read_json(args.plan_json))
    if plan.catalog_id != catalog.catalog_id:
        raise ValueError("edit plan catalog_id does not match catalog.json")
    result = render_rushes_edit(
        catalog,
        plan,
        args.output_dir,
        scdet_threshold=args.scdet_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_feature_cut(args: argparse.Namespace) -> int:
    result = run_feature_cut_experiment(
        catalog_path=args.catalog_json,
        brief_path=args.brief_json,
        checkpoint_path=args.sam_checkpoint,
        output_dir=args.output_dir,
        plan_prompt=_load_prompt("feature_cut_selects_zh-TW.txt"),
        grounding_prompt=_load_prompt("grounding_native_yxyx_zh-TW.txt"),
        temperature=args.temperature,
        scdet_threshold=args.scdet_threshold,
        sam_analysis_fps=args.sam_analysis_fps,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_full_clip(args: argparse.Namespace) -> int:
    result = run_full_clip(
        args.video,
        args.output_dir,
        clip_card_prompt=_load_prompt("full_clip_card_mmss_zh-TW.txt"),
        dense_prompt=_load_prompt("dense_event_frame_selection_zh-TW.txt"),
        proxy_max_side=args.proxy_max_side,
        proxy_fps=args.proxy_fps,
        audio_mode=args.audio_mode,
        scdet_threshold=args.scdet_threshold,
        temperature=args.temperature,
        dense_mode=args.dense_mode,
        dense_event_ids=set(args.dense_event),
        dense_window_ms=args.dense_window_ms,
        dense_fps_override=args.dense_fps,
        file_cache_root=args.file_cache_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_full_library(args: argparse.Namespace) -> int:
    result = run_full_library(
        args.source_dir,
        args.output_dir,
        clip_card_prompt=_load_prompt("full_clip_card_mmss_zh-TW.txt"),
        dense_prompt=_load_prompt("dense_event_frame_selection_zh-TW.txt"),
        recursive=args.recursive,
        max_clips=args.max_clips,
        proxy_max_side=args.proxy_max_side,
        proxy_fps=args.proxy_fps,
        audio_mode=args.audio_mode,
        scdet_threshold=args.scdet_threshold,
        temperature=args.temperature,
        prepare_only=args.prepare_only,
        file_cache_root=args.file_cache_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_full_ground_event(args: argparse.Namespace) -> int:
    result = run_full_event_geometry(
        args.clip_run_dir,
        args.event_id,
        grounding_prompt=_load_prompt("grounding_native_yxyx_zh-TW.txt"),
        checkpoint_path=args.sam_checkpoint,
        target_entity_id=args.target_entity_id,
        target_description=args.target_description,
        sam_analysis_fps=args.sam_analysis_fps,
        temperature=args.temperature,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_review_ab(args: argparse.Namespace) -> int:
    output = render_grounding_ab_review(
        explicit_summary=args.explicit_summary,
        generic_summary=args.generic_summary,
        output_dir=args.output_dir,
    )
    print(output.resolve())
    return 0


def command_temporal_repeat(args: argparse.Namespace) -> int:
    media = MediaInfo.model_validate(read_json(args.artifact_root / "media.json"))
    source = args.artifact_root / "source.mp4"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = GeminiLabClient(temperature=args.temperature)
    summaries: list[dict[str, object]] = []
    failures = 0
    try:
        uploaded, _ = client.ensure_video_upload(
            _artifact_upload_source(args.artifact_root), args.artifact_root / "upload"
        )
        for run_number in range(1, args.runs + 1):
            run_id = f"temporal-{run_number:02d}-{uuid.uuid4().hex[:8]}"
            run_dir = args.output_dir / f"run-{run_number:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                temporal = client.analyze_temporal_video(
                    media=media,
                    uploaded=uploaded,
                    prompt_template=_load_prompt("temporal_map_zh-TW.txt"),
                    run_id=run_id,
                    run_dir=run_dir,
                )
                for event in temporal.events:
                    frame_dir = run_dir / "keyframes" / _safe_name(event.event_id)
                    frame = extract_frame(source, event.recommended_keyframe_ms, frame_dir / "frame.png")
                    write_json(frame_dir / "frame.json", frame)
                summaries.append(
                    {
                        "run": f"run-{run_number:02d}",
                        "schema_valid": True,
                        "event_count": len(temporal.events),
                        "events": [
                            {
                                "event_id": event.event_id,
                                "label": event.label,
                                "start_ms": event.start_ms,
                                "end_ms": event.end_ms,
                                "recommended_keyframe_ms": event.recommended_keyframe_ms,
                            }
                            for event in temporal.events
                        ],
                    }
                )
            except Exception as error:
                failures += 1
                summaries.append(
                    {
                        "run": f"run-{run_number:02d}",
                        "schema_valid": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    finally:
        client.close()
    result = {
        "model": "gemini-3.5-flash",
        "method": "temporal-first reduced schema; no entity/layout/card fields",
        "runs_requested": args.runs,
        "runs_succeeded": args.runs - failures,
        "failure_count": failures,
        "runs": summaries,
    }
    write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 1


def command_storyboard_temporal(args: argparse.Namespace) -> int:
    media = MediaInfo.model_validate(read_json(args.artifact_root / "media.json"))
    source = args.artifact_root / "source.mp4"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, object]] = []
    for index, requested_ms in enumerate(range(0, media.duration_ms, args.interval_ms)):
        frame_id = f"f{index:03d}"
        frame_dir = args.output_dir / "frames" / frame_id
        frame = extract_frame(source, requested_ms, frame_dir / "frame.png")
        thumbnail_path = frame_dir / "storyboard.jpg"
        with Image.open(frame.path) as image:
            image = image.convert("RGB")
            image.thumbnail((args.max_width, args.max_width))
            image.save(thumbnail_path, format="JPEG", quality=85, optimize=True)
        frames.append(
            {
                "frame_id": frame_id,
                "requested_time_ms": requested_ms,
                "frame_pts": frame.frame_pts,
                "frame_time_ms": frame.frame_time_ms,
                "image_path": str(thumbnail_path.resolve()),
                "image_hash": sha256_file(thumbnail_path),
            }
        )
        write_json(frame_dir / "frame.json", frame)
    write_json(args.output_dir / "frame_index.json", frames)

    client = GeminiLabClient(temperature=args.temperature)
    try:
        indexed = client.analyze_indexed_storyboard(
            media=media,
            frames=frames,
            prompt_template=_load_prompt("indexed_storyboard_zh-TW.txt"),
            run_id=f"storyboard-{uuid.uuid4().hex[:8]}",
            run_dir=args.output_dir,
        )
    finally:
        client.close()

    positions = {str(frame["frame_id"]): index for index, frame in enumerate(frames)}
    temporal_events: list[TemporalEvent] = []
    for event in indexed.events:
        first_index = positions[event.first_frame_id]
        last_index = positions[event.last_frame_id]
        recommended_index = positions[event.recommended_frame_id]
        start_ms = int(frames[first_index]["frame_time_ms"])
        end_ms = (
            int(frames[last_index + 1]["frame_time_ms"])
            if last_index + 1 < len(frames)
            else media.duration_ms
        )
        temporal_events.append(
            TemporalEvent(
                event_id=event.event_id,
                start_ms=start_ms,
                end_ms=end_ms,
                label=event.label,
                observable_evidence=event.observable_evidence,
                recommended_keyframe_ms=int(frames[recommended_index]["frame_time_ms"]),
                keyframe_reason=f"Selected immutable storyboard frame {event.recommended_frame_id}",
                confidence=event.confidence,
                boundary_precision=event.boundary_precision,
            )
        )
    temporal = TemporalMap(
        asset_id=indexed.asset_id,
        duration_ms=indexed.duration_ms,
        summary=indexed.summary,
        events=temporal_events,
        uncertainties=indexed.uncertainties,
        model_provenance=indexed.model_provenance,
    )
    write_json(args.output_dir / "temporal_map.derived_from_pts.json", temporal)
    event_images: dict[str, Path] = {}
    grounding_results: list[dict[str, object]] = []
    ground_client = GeminiLabClient(temperature=args.temperature)
    try:
        for event, indexed_event in zip(temporal.events, indexed.events, strict=True):
            frame_index = positions[indexed_event.recommended_frame_id]
            frame_dir = args.output_dir / "frames" / str(frames[frame_index]["frame_id"])
            frame = ExtractedFrame.model_validate(read_json(frame_dir / "frame.json"))
            grounding_dir = args.output_dir / "events" / _safe_name(event.event_id) / "grounding"
            try:
                proposal = ground_client.ground_frame(
                    media=media,
                    frame=frame,
                    event_id=event.event_id,
                    event_description=event.observable_evidence,
                    entity_id=indexed_event.grounding_target_id,
                    target_description=indexed_event.grounding_target_description,
                    prompt_template=_load_prompt("grounding_native_yxyx_zh-TW.txt"),
                    run_id=f"storyboard-ground-{event.event_id}-{uuid.uuid4().hex[:8]}",
                    output_dir=grounding_dir,
                )
                overlay_path = grounding_dir / "debug.png"
                draw_grounding_overlay(Path(frame.path), proposal, overlay_path)
                event_images[event.event_id] = overlay_path
                grounding_results.append(
                    {
                        "event_id": event.event_id,
                        "entity_id": indexed_event.grounding_target_id,
                        "visible": proposal.visible,
                        "candidate_count": len(proposal.candidates),
                        "frame_time_ms": proposal.frame_time_ms,
                    }
                )
            except Exception as error:
                append_error(args.output_dir, f"storyboard_grounding:{event.event_id}", error)
                event_images[event.event_id] = Path(str(frames[frame_index]["image_path"]))
                grounding_results.append(
                    {"event_id": event.event_id, "ok": False, "error": str(error)}
                )
    finally:
        ground_client.close()
    render_temporal_timeline(
        temporal_map=temporal,
        video_path=source,
        event_images=event_images,
        output_path=args.output_dir / "index.html",
        sampling_interval_ms=args.interval_ms,
    )
    result = {
        "ok": True,
        "model": "gemini-3.5-flash",
        "method": "Gemini selects immutable frame IDs; local FFmpeg PTS supplies all times",
        "interval_ms": args.interval_ms,
        "frame_count": len(frames),
        "event_count": len(temporal.events),
        "grounding_results": grounding_results,
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _mmss_to_ms(value: str) -> int:
    minutes, seconds = (int(part) for part in value.split(":"))
    return (minutes * 60 + seconds) * 1000


def _jpeg_transport_frame(frame: ExtractedFrame, moment_dir: Path) -> ExtractedFrame:
    output = moment_dir / "frame.transport.jpg"
    with Image.open(frame.path) as image:
        image.convert("RGB").save(output, format="JPEG", quality=95, subsampling=0, optimize=True)
    transport = frame.model_copy(
        update={"path": str(output.resolve()), "frame_hash": sha256_file(output)}
    )
    write_json(
        moment_dir / "frame.transport.json",
        {
            **transport.model_dump(mode="json"),
            "transport": "same-dimension JPEG quality=95 subsampling=0",
            "source_frame_path": frame.path,
            "source_frame_hash": frame.frame_hash,
        },
    )
    return transport


def command_direct_moment_repeat(args: argparse.Namespace) -> int:
    pipeline_started = monotonic()
    media = MediaInfo.model_validate(read_json(args.artifact_root / "media.json"))
    source = args.artifact_root / "source.mp4"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.target_id) != bool(args.target_description):
        raise ValueError("--target-id and --target-description must be provided together")
    if bool(args.candidate_map) != bool(args.candidate_id):
        raise ValueError("--candidate-map and --candidate-id must be provided together")
    if args.target_id and args.candidate_id:
        raise ValueError("use either an explicit target or a saved candidate, not both")

    target_id = args.target_id
    target_description = args.target_description
    if args.candidate_map:
        candidates = TargetCandidateMap.model_validate(read_json(args.candidate_map))
        selected = next(
            (
                candidate
                for candidate in candidates.candidates
                if candidate.candidate_id == args.candidate_id
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"candidate_id {args.candidate_id!r} not found in {args.candidate_map}")
        if candidates.asset_id != media.asset_id:
            raise ValueError("candidate map belongs to a different asset")
        target_id = selected.candidate_id
        target_description = selected.target_description

    if not target_id:
        print(
            "No target was specified; proposing user-selectable candidates before timing or Grounding.",
            file=sys.stderr,
        )
        return _run_target_suggestions(
            artifact_root=args.artifact_root,
            output_dir=args.output_dir,
            runs=args.runs,
            temperature=args.temperature,
        )
    client = GeminiLabClient(temperature=args.temperature)
    summaries: list[dict[str, object]] = []
    analysis_failures = 0
    grounding_failures = 0
    try:
        uploaded, _ = client.ensure_video_upload(
            _artifact_upload_source(args.artifact_root), args.artifact_root / "upload"
        )
        for run_number in range(1, args.runs + 1):
            run_id = f"direct-mmss-{run_number:02d}-{uuid.uuid4().hex[:8]}"
            run_dir = args.output_dir / f"run-{run_number:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_started = monotonic()
            timing: dict[str, object] = {"groundings": []}
            try:
                analysis_started = monotonic()
                moments = client.analyze_direct_moments(
                    media=media,
                    uploaded=uploaded,
                    prompt_template=_load_prompt(
                        "target_moments_mmss_zh-TW.txt"
                        if target_id
                        else "direct_moments_mmss_zh-TW.txt"
                    ),
                    run_id=run_id,
                    run_dir=run_dir,
                    locked_target_id=target_id,
                    locked_target_description=target_description,
                )
                timing["video_analysis_seconds"] = round(monotonic() - analysis_started, 6)
                run_summary: dict[str, object] = {
                    "run": f"run-{run_number:02d}",
                    "schema_valid": True,
                    "moment_count": len(moments.moments),
                    "moments": [
                        {
                            "moment_id": moment.moment_id,
                            "timestamp_mmss": moment.timestamp_mmss,
                            "timestamp_ms": _mmss_to_ms(moment.timestamp_mmss),
                            "label": moment.label,
                            "grounding_target_id": moment.grounding_target_id,
                        }
                        for moment in moments.moments
                    ],
                }
                if run_number <= args.ground_runs:
                    timeline_results: list[tuple[str, int, int, Path, GroundingProposal]] = []
                    grounding_summary: list[dict[str, object]] = []
                    for moment in moments.moments[: args.ground_moments_per_run]:
                        requested_ms = _mmss_to_ms(moment.timestamp_mmss)
                        moment_dir = run_dir / "moments" / _safe_name(moment.moment_id)
                        extraction_started = monotonic()
                        grounding_started: float | None = None
                        try:
                            frame = extract_frame(source, requested_ms, moment_dir / "frame.png")
                            write_json(moment_dir / "frame.json", frame)
                            grounding_frame = (
                                _jpeg_transport_frame(frame, moment_dir)
                                if args.ground_transport_jpeg
                                else frame
                            )
                            extraction_seconds = monotonic() - extraction_started
                            grounding_dir = moment_dir / "grounding"
                            grounding_started = monotonic()
                            proposal = client.ground_frame(
                                media=media,
                                frame=grounding_frame,
                                event_id=moment.moment_id,
                                event_description=f"{moment.label}；{moment.observable_evidence}",
                                entity_id=moment.grounding_target_id,
                                target_description=moment.grounding_target_description,
                                prompt_template=_load_prompt("grounding_native_yxyx_zh-TW.txt"),
                                run_id=run_id,
                                output_dir=grounding_dir,
                            )
                            grounding_seconds = monotonic() - grounding_started
                            overlay_path = grounding_dir / "debug.png"
                            draw_grounding_overlay(Path(frame.path), proposal, overlay_path)
                            timeline_results.append(
                                (
                                    moment.moment_id,
                                    requested_ms,
                                    frame.frame_time_ms,
                                    overlay_path,
                                    proposal,
                                )
                            )
                            grounding_row = {
                                "moment_id": moment.moment_id,
                                "ok": True,
                                "requested_ms": requested_ms,
                                "frame_time_ms": frame.frame_time_ms,
                                "visible": proposal.visible,
                                "candidate_count": len(proposal.candidates),
                                "frame_extraction_seconds": round(extraction_seconds, 6),
                                "grounding_seconds": round(grounding_seconds, 6),
                            }
                        except Exception as error:
                            grounding_failures += 1
                            extraction_seconds = monotonic() - extraction_started
                            grounding_seconds = (
                                monotonic() - grounding_started
                                if grounding_started is not None
                                else 0.0
                            )
                            append_error(
                                run_dir,
                                f"direct_moment_grounding:{moment.moment_id}",
                                error,
                            )
                            grounding_row = {
                                "moment_id": moment.moment_id,
                                "ok": False,
                                "requested_ms": requested_ms,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "frame_extraction_seconds": round(extraction_seconds, 6),
                                "grounding_seconds": round(grounding_seconds, 6),
                            }
                        grounding_summary.append(grounding_row)
                        timing["groundings"].append(grounding_row)
                    if timeline_results:
                        render_direct_moment_timeline(
                            moment_map=moments,
                            video_path=source,
                            results=timeline_results,
                            output_path=run_dir / "index.html",
                        )
                    run_summary["groundings"] = grounding_summary
                timing["run_total_seconds"] = round(monotonic() - run_started, 6)
                write_json(run_dir / "timing.json", timing)
                summaries.append(run_summary)
            except Exception as error:
                analysis_failures += 1
                append_error(run_dir, "direct_moment_pipeline", error)
                timing["run_total_seconds"] = round(monotonic() - run_started, 6)
                write_json(run_dir / "timing.json", timing)
                summaries.append(
                    {
                        "run": f"run-{run_number:02d}",
                        "schema_valid": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    finally:
        client.close()
    pricing = summarize_usage_and_list_price(args.output_dir)
    write_json(args.output_dir / "pricing.json", pricing)
    result = {
        "model": "gemini-3.5-flash",
        "method": "direct Gemini MM:SS salient moments",
        "runs_requested": args.runs,
        "runs_succeeded": args.runs - analysis_failures,
        "grounded_runs": min(args.ground_runs, args.runs),
        "analysis_failure_count": analysis_failures,
        "grounding_failure_count": grounding_failures,
        "failure_count": analysis_failures + grounding_failures,
        "pipeline_elapsed_seconds": round(monotonic() - pipeline_started, 6),
        "pricing": pricing,
        "runs": summaries,
    }
    write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if analysis_failures == 0 and grounding_failures == 0 else 1


def command_run(args: argparse.Namespace) -> int:
    media = probe_video(args.video)
    artifact_root = args.output or _default_artifact_root(media.sha256)
    artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(artifact_root / "media.json", media)
    source_link = artifact_root / ("source" + args.video.suffix.lower())
    if not source_link.exists():
        source_link.symlink_to(args.video.resolve(strict=True))
    content_prompt = _load_prompt("content_map_zh-TW.txt")
    grounding_prompt = _load_prompt("grounding_native_yxyx_zh-TW.txt")
    failures = 0
    semantic_failures = 0
    client: GeminiLabClient | None = None
    run_dirs: list[Path] = []
    try:
        client = GeminiLabClient(temperature=args.temperature)
        uploaded, _ = client.ensure_video_upload(
            source_link,
            artifact_root / "upload",
            force_reupload=args.force_reupload,
        )
        for run_number in range(1, args.runs + 1):
            run_id = f"run-{run_number:02d}-{uuid.uuid4().hex[:8]}"
            run_dir = artifact_root / f"run-{run_number:02d}"
            run_dirs.append(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                run_dir / "run.json",
                {
                    "run_id": run_id,
                    "model": "gemini-3.5-flash",
                    "temperature": args.temperature,
                    "asset_id": media.asset_id,
                    "semantic_time_is_frame_accurate": False,
                },
            )
            try:
                content = client.analyze_video(
                    media=media,
                    uploaded=uploaded,
                    prompt_template=content_prompt,
                    run_id=run_id,
                    run_dir=run_dir,
                    repair_attempts=args.content_repair_attempts,
                )
            except Exception:
                failures += 1
                continue
            entities = {entity.entity_id: entity for entity in content.entities}
            proposals: list[tuple[GroundingProposal, Path]] = []
            semantic_checks: list[dict[str, object]] = []
            for event in content.events:
                if event.recommended_keyframe_ms is None:
                    continue
                event_dir = run_dir / "events" / _safe_name(event.event_id)
                try:
                    frame = extract_frame(
                        args.video,
                        event.recommended_keyframe_ms,
                        event_dir / "frame.png",
                    )
                    write_json(event_dir / "frame.json", frame)
                except Exception as error:
                    append_error(run_dir, f"extract_frame:{event.event_id}", error)
                    failures += 1
                    continue
                ordered_ids = list(dict.fromkeys(event.primary_entity_ids + event.entity_ids))
                for entity_id in ordered_ids[: args.ground_per_event]:
                    entity = entities.get(entity_id)
                    if entity is None:
                        continue
                    grounding_dir = event_dir / "groundings" / _safe_name(entity_id)
                    try:
                        proposal = client.ground_frame(
                            media=media,
                            frame=frame,
                            event_id=event.event_id,
                            event_description=event.description,
                            entity_id=entity_id,
                            target_description=(
                                f"{entity.label}；可區分特徵：{entity.distinguishing_features}"
                            ),
                            prompt_template=grounding_prompt,
                            run_id=run_id,
                            output_dir=grounding_dir,
                        )
                        overlay_path = grounding_dir / "debug.png"
                        draw_grounding_overlay(Path(frame.path), proposal, overlay_path)
                        proposals.append((proposal, overlay_path))
                        is_primary_or_required = entity_id in set(
                            event.primary_entity_ids + event.required_entity_ids
                        )
                        passed = proposal.visible and bool(proposal.candidates)
                        semantic_checks.append(
                            {
                                "event_id": event.event_id,
                                "entity_id": entity_id,
                                "recommended_keyframe_ms": event.recommended_keyframe_ms,
                                "frame_time_ms": proposal.frame_time_ms,
                                "primary_or_required": is_primary_or_required,
                                "visible": proposal.visible,
                                "candidate_count": len(proposal.candidates),
                                "passed": passed if is_primary_or_required else None,
                                "visibility_reason": proposal.visibility_reason,
                            }
                        )
                        if is_primary_or_required and not passed:
                            semantic_failures += 1
                    except Exception as error:
                        append_error(run_dir, f"grounding:{event.event_id}:{entity_id}", error)
                        failures += 1
            write_json(
                run_dir / "semantic_keyframe_validation.json",
                {
                    "ok": all(
                        check["passed"] is not False
                        for check in semantic_checks
                    ),
                    "method": (
                        "A primary/required entity must be visible with at least one candidate "
                        "at the Content Map recommended keyframe. This is an automated Gemini "
                        "cross-check, not independent human ground truth."
                    ),
                    "checks": semantic_checks,
                },
            )
            render_timeline(
                content_map=content,
                video_path=source_link,
                proposals=proposals,
                output_path=run_dir / "index.html",
            )
    except Exception as error:
        append_error(artifact_root, "pipeline", error)
        failures += 1
    finally:
        if client is not None:
            client.close()
    annotations = args.annotations if args.annotations and args.annotations.exists() else None
    compare_runs(run_dirs, artifact_root / "comparison.json", annotations)
    write_json(
        artifact_root / "result.json",
        {
            "ok": failures == 0 and semantic_failures == 0,
            "execution_ok": failures == 0,
            "semantic_gate_ok": semantic_failures == 0,
            "failure_count": failures,
            "semantic_failure_count": semantic_failures,
            "artifact_root": str(artifact_root.resolve()),
            "run_count": args.runs,
        },
    )
    print(artifact_root.resolve())
    return 0 if failures == 0 and semantic_failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jascue-video-lab",
        description="Gemini 3.5 Flash video understanding and single-frame grounding lab",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe", help="Inspect media with ffprobe and calculate SHA-256")
    probe_parser.add_argument("video", type=Path)
    probe_parser.add_argument("--output", type=Path)
    probe_parser.set_defaults(handler=command_probe)

    upload_parser = subparsers.add_parser(
        "upload", help="Prepare an artifact and upload an original video or analysis proxy"
    )
    upload_parser.add_argument("video", type=Path)
    upload_parser.add_argument("--analysis-proxy", type=Path)
    upload_parser.add_argument("--max-proxy-duration-delta-ms", type=int, default=100)
    upload_parser.add_argument("--temperature", type=float, default=0.2)
    upload_parser.add_argument(
        "--force-reupload",
        action="store_true",
        help="Ignore an ACTIVE saved File API object and upload again",
    )
    upload_parser.add_argument("--output", type=Path, required=True)
    upload_parser.set_defaults(handler=command_upload)

    pricing_parser = subparsers.add_parser(
        "pricing", help="Summarize saved raw interaction usage using public list prices"
    )
    pricing_parser.add_argument("artifact_root", type=Path)
    pricing_parser.add_argument("--output", type=Path)
    pricing_parser.set_defaults(handler=command_pricing)

    serve_parser = subparsers.add_parser(
        "serve-review",
        help="Start the local blind-review web app",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Explicitly allow binding beyond this Mac; no authentication is provided",
    )
    serve_parser.set_defaults(handler=command_serve_review)

    extract_parser = subparsers.add_parser("extract", help="Extract orientation-corrected frame with exact PTS")
    extract_parser.add_argument("video", type=Path)
    extract_parser.add_argument("time_ms", type=int)
    extract_parser.add_argument("output", type=Path)
    extract_parser.set_defaults(handler=command_extract)

    fixture_parser = subparsers.add_parser("make-fixtures", help="Generate four real synthetic video fixtures")
    fixture_parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "fixtures" / "generated")
    fixture_parser.set_defaults(handler=command_fixtures)

    run_parser = subparsers.add_parser("run", help="Run the live vertical slice")
    run_parser.add_argument("video", type=Path)
    run_parser.add_argument("--runs", type=int, default=3)
    run_parser.add_argument("--ground-per-event", type=int, default=1)
    run_parser.add_argument("--temperature", type=float, default=0.2)
    run_parser.add_argument(
        "--content-repair-attempts",
        type=int,
        default=1,
        help="Retry an invalid Content Map with its saved contract errors (default: 1)",
    )
    run_parser.add_argument("--output", type=Path)
    run_parser.add_argument("--annotations", type=Path)
    run_parser.add_argument(
        "--resume-upload",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    run_parser.add_argument(
        "--force-reupload",
        action="store_true",
        help="Ignore an ACTIVE saved File API object and upload again",
    )
    run_parser.set_defaults(handler=command_run)

    timeline_parser = subparsers.add_parser("timeline", help="Rebuild timeline from a completed run")
    timeline_parser.add_argument("run_dir", type=Path)
    timeline_parser.add_argument("video", type=Path)
    timeline_parser.add_argument("--output", type=Path)
    timeline_parser.set_defaults(handler=command_timeline)

    compare_parser = subparsers.add_parser("compare", help="Compare completed run directories")
    compare_parser.add_argument("run_dirs", nargs="+", type=Path)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--annotations", type=Path)
    compare_parser.set_defaults(handler=command_compare)

    review_parser = subparsers.add_parser(
        "review", help="Render a static human-vs-Gemini bbox review matrix"
    )
    review_parser.add_argument("artifact_root", type=Path)
    review_parser.add_argument("--annotations", type=Path, required=True)
    review_parser.add_argument("--output-dir", type=Path, required=True)
    review_parser.set_defaults(handler=command_review)

    repeat_parser = subparsers.add_parser(
        "ground-repeat", help="Repeat Grounding on the exact same extracted frame"
    )
    repeat_parser.add_argument("artifact_root", type=Path)
    repeat_parser.add_argument("frame_json", type=Path)
    repeat_parser.add_argument("--event-id", required=True)
    repeat_parser.add_argument("--event-description", required=True)
    repeat_parser.add_argument("--entity-id", required=True)
    repeat_parser.add_argument("--target-description", required=True)
    repeat_parser.add_argument("--runs", type=int, default=5)
    repeat_parser.add_argument("--temperature", type=float, default=0.2)
    repeat_parser.add_argument("--reference-box", type=int, nargs=4)
    repeat_parser.add_argument("--output-dir", type=Path, required=True)
    repeat_parser.set_defaults(handler=command_ground_repeat)

    ab_parser = subparsers.add_parser(
        "review-ab", help="Render a static A/B review for two repeated Grounding experiments"
    )
    ab_parser.add_argument("explicit_summary", type=Path)
    ab_parser.add_argument("generic_summary", type=Path)
    ab_parser.add_argument("--output-dir", type=Path, required=True)
    ab_parser.set_defaults(handler=command_review_ab)

    temporal_parser = subparsers.add_parser(
        "temporal-repeat",
        help="A/B test a reduced timing-only schema against an already uploaded video",
    )
    temporal_parser.add_argument("artifact_root", type=Path)
    temporal_parser.add_argument("--runs", type=int, default=3)
    temporal_parser.add_argument("--temperature", type=float, default=0.2)
    temporal_parser.add_argument("--output-dir", type=Path, required=True)
    temporal_parser.set_defaults(handler=command_temporal_repeat)

    storyboard_parser = subparsers.add_parser(
        "storyboard-temporal",
        help="Build a PTS-indexed storyboard and let Gemini select frame IDs, never timestamps",
    )
    storyboard_parser.add_argument("artifact_root", type=Path)
    storyboard_parser.add_argument("--interval-ms", type=int, default=4000)
    storyboard_parser.add_argument("--max-width", type=int, default=768)
    storyboard_parser.add_argument("--temperature", type=float, default=0.2)
    storyboard_parser.add_argument("--output-dir", type=Path, required=True)
    storyboard_parser.set_defaults(handler=command_storyboard_temporal)

    candidates_parser = subparsers.add_parser(
        "suggest-targets",
        help="When no target was requested, propose user-selectable objects without Grounding",
    )
    candidates_parser.add_argument("artifact_root", type=Path)
    candidates_parser.add_argument("--runs", type=int, default=1)
    candidates_parser.add_argument("--temperature", type=float, default=0.2)
    candidates_parser.add_argument("--output-dir", type=Path, required=True)
    candidates_parser.set_defaults(handler=command_suggest_targets)

    direct_parser = subparsers.add_parser(
        "direct-moment-repeat",
        help="Ask Gemini directly for official MM:SS screenshot moments and optionally ground them",
    )
    direct_parser.add_argument("artifact_root", type=Path)
    direct_parser.add_argument("--runs", type=int, default=3)
    direct_parser.add_argument("--ground-runs", type=int, default=1)
    direct_parser.add_argument("--ground-moments-per-run", type=int, default=1)
    direct_parser.add_argument("--ground-transport-jpeg", action="store_true")
    direct_parser.add_argument("--target-id")
    direct_parser.add_argument("--target-description")
    direct_parser.add_argument("--candidate-map", type=Path)
    direct_parser.add_argument("--candidate-id")
    direct_parser.add_argument("--temperature", type=float, default=0.2)
    direct_parser.add_argument("--output-dir", type=Path, required=True)
    direct_parser.set_defaults(handler=command_direct_moment_repeat)

    tracking_parser = subparsers.add_parser(
        "track-csrt",
        help="Experimental bbox propagation on an isolated optional OpenCV path",
    )
    tracking_parser.add_argument("video", type=Path)
    tracking_parser.add_argument("--grounding-json", type=Path)
    tracking_parser.add_argument("--seed-time-ms", type=int)
    tracking_parser.add_argument("--seed-box", type=int, nargs=4)
    tracking_parser.add_argument("--target-description", required=True)
    tracking_parser.add_argument("--analysis-fps", type=float, default=15.0)
    tracking_parser.add_argument("--max-side", type=int, default=960)
    tracking_parser.add_argument("--appearance-threshold", type=float, default=0.25)
    tracking_parser.add_argument("--output-dir", type=Path, required=True)
    tracking_parser.set_defaults(handler=command_track_csrt)

    sam_tracking_parser = subparsers.add_parser(
        "track-sam21",
        help="Experimental Gemini/manual bbox to SAM 2.1 mask propagation",
    )
    sam_tracking_parser.add_argument("video", type=Path)
    sam_tracking_parser.add_argument("--checkpoint", type=Path, required=True)
    sam_tracking_parser.add_argument("--grounding-json", type=Path)
    sam_tracking_parser.add_argument("--seed-time-ms", type=int)
    sam_tracking_parser.add_argument("--seed-box", type=int, nargs=4)
    sam_tracking_parser.add_argument("--target-description", required=True)
    sam_tracking_parser.add_argument("--analysis-fps", type=float, default=2.0)
    sam_tracking_parser.add_argument("--max-side", type=int, default=960)
    sam_tracking_parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    sam_tracking_parser.add_argument("--ffmpeg-scdet-threshold", type=float, default=4.0)
    sam_tracking_parser.add_argument("--seed-box-padding-ratio", type=float, default=0.0)
    sam_tracking_parser.add_argument("--output-dir", type=Path, required=True)
    sam_tracking_parser.set_defaults(handler=command_track_sam21)

    tracker_comparison_parser = subparsers.add_parser(
        "compare-trackers",
        help="Compare SAM mask-derived boxes with a bbox tracker as agreement, not accuracy",
    )
    tracker_comparison_parser.add_argument("segmentation_json", type=Path)
    tracker_comparison_parser.add_argument("reference_bbox_json", type=Path)
    tracker_comparison_parser.add_argument("--output", type=Path, required=True)
    tracker_comparison_parser.set_defaults(handler=command_compare_trackers)

    shot_parser = subparsers.add_parser(
        "detect-shots", help="Detect exact decoded-frame shot boundaries with FFmpeg scdet"
    )
    shot_parser.add_argument("video", type=Path)
    shot_parser.add_argument("--threshold", type=float, default=4.0)
    shot_parser.add_argument("--output", type=Path, required=True)
    shot_parser.set_defaults(handler=command_detect_shots)

    catalog_parser = subparsers.add_parser(
        "catalog-rushes", help="Build a labeled immutable-frame-ID catalog reel from rushes"
    )
    catalog_parser.add_argument("source_directory", type=Path)
    catalog_parser.add_argument("--sample-interval-ms", type=int, default=2000)
    catalog_parser.add_argument("--max-width", type=int, default=640)
    catalog_parser.add_argument("--output-dir", type=Path, required=True)
    catalog_parser.set_defaults(handler=command_catalog_rushes)

    rushes_parser = subparsers.add_parser(
        "rushes-run", help="Catalog rushes, ask Gemini for frame-ID selects, and render rough cuts"
    )
    rushes_parser.add_argument("source_directory", type=Path)
    rushes_parser.add_argument("--sample-interval-ms", type=int, default=2000)
    rushes_parser.add_argument("--scdet-threshold", type=float, default=4.0)
    rushes_parser.add_argument("--temperature", type=float, default=0.2)
    rushes_parser.add_argument("--output-dir", type=Path, required=True)
    rushes_parser.set_defaults(handler=command_rushes_run)

    render_rushes_parser = subparsers.add_parser(
        "render-rushes", help="Render a validated frame-ID edit plan without another model call"
    )
    render_rushes_parser.add_argument("catalog_json", type=Path)
    render_rushes_parser.add_argument("plan_json", type=Path)
    render_rushes_parser.add_argument("--scdet-threshold", type=float, default=4.0)
    render_rushes_parser.add_argument("--output-dir", type=Path, required=True)
    render_rushes_parser.set_defaults(handler=command_render_rushes)

    feature_cut_parser = subparsers.add_parser(
        "feature-cut",
        help="Build brief-ordered 16:9 and SAM-tracked 9:16 feature review cuts",
    )
    feature_cut_parser.add_argument("catalog_json", type=Path)
    feature_cut_parser.add_argument("brief_json", type=Path)
    feature_cut_parser.add_argument("--sam-checkpoint", type=Path, required=True)
    feature_cut_parser.add_argument("--sam-analysis-fps", type=float, default=2.0)
    feature_cut_parser.add_argument("--scdet-threshold", type=float, default=4.0)
    feature_cut_parser.add_argument("--temperature", type=float, default=0.2)
    feature_cut_parser.add_argument("--output-dir", type=Path, required=True)
    feature_cut_parser.set_defaults(handler=command_feature_cut)

    full_clip_parser = subparsers.add_parser(
        "full-clip",
        help="Analyze one complete proxy into an MM:SS Clip Card and dense frame-ID evidence",
    )
    full_clip_parser.add_argument("video", type=Path)
    full_clip_parser.add_argument("--proxy-max-side", type=int, default=1280)
    full_clip_parser.add_argument("--proxy-fps", type=int, default=30)
    full_clip_parser.add_argument(
        "--audio-mode",
        choices=["auto", "off", "required"],
        default="auto",
        help="auto preserves audio when present; off strips it; required rejects silent sources",
    )
    full_clip_parser.add_argument("--scdet-threshold", type=float, default=4.0)
    full_clip_parser.add_argument("--temperature", type=float, default=0.2)
    full_clip_parser.add_argument(
        "--dense-mode",
        choices=["none", "required", "flagged", "all"],
        default="none",
    )
    full_clip_parser.add_argument(
        "--dense-event",
        action="append",
        default=[],
        help="Explicit event ID to refine; may be repeated and overrides dense-mode for that event",
    )
    full_clip_parser.add_argument("--dense-window-ms", type=int, default=4000)
    full_clip_parser.add_argument(
        "--dense-fps",
        type=float,
        choices=[4.0, 8.0],
        help="Explicit local fallback FPS for selected dense events",
    )
    full_clip_parser.add_argument("--output-dir", type=Path, required=True)
    full_clip_parser.add_argument("--file-cache-root", type=Path)
    full_clip_parser.set_defaults(handler=command_full_clip)

    full_library_parser = subparsers.add_parser(
        "full-library",
        help="Build resumable per-clip MM:SS Clip Cards for a rushes directory",
    )
    full_library_parser.add_argument("source_dir", type=Path)
    full_library_parser.add_argument("--recursive", action="store_true")
    full_library_parser.add_argument("--max-clips", type=int)
    full_library_parser.add_argument("--proxy-max-side", type=int, default=1280)
    full_library_parser.add_argument("--proxy-fps", type=int, default=30)
    full_library_parser.add_argument(
        "--audio-mode",
        choices=["auto", "off", "required"],
        default="auto",
        help="auto preserves audio when present; off strips it; required rejects silent sources",
    )
    full_library_parser.add_argument("--scdet-threshold", type=float, default=4.0)
    full_library_parser.add_argument("--temperature", type=float, default=0.2)
    full_library_parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create local proxies, hashes, shots, and audit frames without Gemini network calls",
    )
    full_library_parser.add_argument("--output-dir", type=Path, required=True)
    full_library_parser.add_argument("--file-cache-root", type=Path)
    full_library_parser.set_defaults(handler=command_full_library)

    full_ground_parser = subparsers.add_parser(
        "full-ground-event",
        help="Ground one selected Clip Card event and optionally propagate SAM in that interval",
    )
    full_ground_parser.add_argument("clip_run_dir", type=Path)
    full_ground_parser.add_argument("event_id")
    full_ground_parser.add_argument("--target-entity-id")
    full_ground_parser.add_argument("--target-description")
    full_ground_parser.add_argument("--sam-checkpoint", type=Path)
    full_ground_parser.add_argument("--sam-analysis-fps", type=float, default=2.0)
    full_ground_parser.add_argument("--temperature", type=float, default=0.2)
    full_ground_parser.set_defaults(handler=command_full_ground_event)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "runs", 1) < 1:
        parser.error("--runs must be at least 1")
    if getattr(args, "ground_per_event", 1) < 1:
        parser.error("--ground-per-event must be at least 1")
    if getattr(args, "content_repair_attempts", 0) < 0:
        parser.error("--content-repair-attempts cannot be negative")
    if getattr(args, "interval_ms", 250) < 250:
        parser.error("--interval-ms must be at least 250")
    if getattr(args, "ground_runs", 0) < 0:
        parser.error("--ground-runs cannot be negative")
    if getattr(args, "ground_moments_per_run", 1) < 1:
        parser.error("--ground-moments-per-run must be at least 1")
    try:
        status = args.handler(args)
    except KeyboardInterrupt:
        status = 130
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        status = 1
    raise SystemExit(status)


if __name__ == "__main__":
    main()
