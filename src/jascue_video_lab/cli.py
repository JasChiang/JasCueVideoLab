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
from .full_v1 import (
    run_query_predicate_refinement,
    run_full_clip,
    run_full_event_geometry,
    run_full_library,
    run_selected_full_clips,
)
from .gemini import GeminiLabClient, MODEL_ID
from .grounding_selection import require_tracking_seed_candidate
from .media import extract_frame, probe_video, sha256_file
from .music import MusicMapLock, MusicMapProposal, analyze_music, review_music_map
from .music_cues import (
    CuePlanLock,
    CuePlanProposal,
    SemanticMusicPairingProposal,
    VisualSyncMap,
    apply_music_first_cue_lock,
    derive_brief_visual_sync_map,
    derive_visual_sync_map,
    plan_music_cues,
    render_cue_review,
    review_cue_plan,
)
from .models import (
    ContentMap,
    ExtractedFrame,
    FeatureEditBrief,
    GroundingProposal,
    MediaInfo,
    TargetCandidateMap,
    TemporalEvent,
    TemporalMap,
    RushesCatalog,
    RushesEditPlan,
    SharedSam21TrackingRequest,
)
from .multi_tracking import (
    compare_aligned_segmentation_tracks,
    render_multi_segmentation_review,
)
from .overlay import draw_grounding_overlay
from .review import render_manual_review
from .repeat import run_repeated_grounding
from .rushes import create_rushes_catalog, render_rushes_edit, run_rushes_experiment
from .sam_tracking import (
    compare_segmentation_to_bbox_track,
    track_bbox_sam21,
    track_bboxes_shared_sam21,
)
from .shots import detect_shots_ffmpeg
from .storage import append_error, read_json, write_json
from .temporal_risk import scan_temporal_risk_windows
from .timeline import render_direct_moment_timeline, render_temporal_timeline, render_timeline
from .tracking import track_bbox_csrt
from .trim_intent import review_trim_decision, run_trim_intent_event


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
    client = GeminiLabClient()
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
) -> int:
    started = monotonic()
    media = MediaInfo.model_validate(read_json(artifact_root / "media.json"))
    output_dir.mkdir(parents=True, exist_ok=True)
    client = GeminiLabClient()
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
        "model": MODEL_ID,
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
        reference_box=args.reference_box,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failure_count"] == 0 else 1


def command_track_csrt(args: argparse.Namespace) -> int:
    if args.grounding_json:
        grounding = GroundingProposal.model_validate(read_json(args.grounding_json))
        tracking_media = probe_video(args.video)
        if grounding.asset_id != tracking_media.asset_id:
            raise ValueError(
                "GroundingProposal asset_id does not match the supplied tracking video"
            )
        selected_seed = require_tracking_seed_candidate(
            grounding,
            candidate_number=args.grounding_candidate_number,
        )
        seed_time_ms = grounding.frame_time_ms
        seed_box = selected_seed.candidate.box_2d
        seed_source = (
            f"GroundingProposal:{args.grounding_json}:"
            f"candidate-number:{selected_seed.candidate_number}:"
            f"selection:{selected_seed.selection_source}"
        )
    else:
        if args.grounding_candidate_number is not None:
            raise ValueError("--grounding-candidate-number requires --grounding-json")
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
        tracking_media = probe_video(args.video)
        if grounding.asset_id != tracking_media.asset_id:
            raise ValueError(
                "GroundingProposal asset_id does not match the supplied tracking video"
            )
        selected_seed = require_tracking_seed_candidate(
            grounding,
            candidate_number=args.grounding_candidate_number,
        )
        seed_time_ms = grounding.frame_time_ms
        seed_box = selected_seed.candidate.box_2d
        seed_source = (
            f"GroundingProposal:{args.grounding_json}:"
            f"candidate-number:{selected_seed.candidate_number}:"
            f"selection:{selected_seed.selection_source}"
        )
        asset_id = grounding.asset_id
        seed_frame_pts = grounding.frame_pts
        seed_frame_sha256 = grounding.frame_hash
        seed_source_width = grounding.source_width
        seed_source_height = grounding.source_height
    else:
        if args.grounding_candidate_number is not None:
            raise ValueError("--grounding-candidate-number requires --grounding-json")
        if args.seed_time_ms is None or args.seed_box is None:
            raise ValueError("provide --grounding-json or both --seed-time-ms and --seed-box")
        seed_time_ms = args.seed_time_ms
        seed_box = args.seed_box
        seed_source = "manual canonical bbox"
        asset_id = None
        seed_frame_pts = None
        seed_frame_sha256 = None
        seed_source_width = None
        seed_source_height = None
    result = track_bbox_sam21(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        seed_time_ms=seed_time_ms,
        seed_box_2d=seed_box,
        target_description=args.target_description,
        output_dir=args.output_dir,
        seed_source=seed_source,
        asset_id=asset_id,
        seed_frame_pts=seed_frame_pts,
        seed_frame_sha256=seed_frame_sha256,
        seed_source_width=seed_source_width,
        seed_source_height=seed_source_height,
        analysis_fps=args.analysis_fps,
        max_side=args.max_side,
        device=args.device,
        ffmpeg_scdet_threshold=args.ffmpeg_scdet_threshold,
        seed_box_padding_ratio=args.seed_box_padding_ratio,
    )
    print(result.model_dump_json(indent=2, exclude={"samples"}))
    return 0


def command_track_shared_sam21(args: argparse.Namespace) -> int:
    request = SharedSam21TrackingRequest.model_validate(read_json(args.targets_json))
    result = track_bboxes_shared_sam21(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        targets=request.targets,
        output_dir=args.output_dir,
        asset_id=request.asset_id,
        analysis_fps=args.analysis_fps,
        max_side=args.max_side,
        device=args.device,
        ffmpeg_scdet_threshold=args.ffmpeg_scdet_threshold,
        seed_box_padding_ratio=args.seed_box_padding_ratio,
        allowed_start_ms=args.allowed_start_ms,
        allowed_end_ms=args.allowed_end_ms,
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )
    print(result.model_dump_json(indent=2, exclude={"analysis_frames"}))
    return 0


def command_render_multi_sam21(args: argparse.Namespace) -> int:
    result = render_multi_segmentation_review(
        track_json_paths=args.track_json,
        labels=args.label,
        output_dir=args.output_dir,
        display_fps=args.display_fps,
        analysis_frames_dir=args.analysis_frames_dir,
    )
    print(result.model_dump_json(indent=2))
    return 0


def command_compare_sam21_tracks(args: argparse.Namespace) -> int:
    result = compare_aligned_segmentation_tracks(
        args.track_a_json, args.track_b_json, args.output
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


def command_scan_temporal_risk(args: argparse.Namespace) -> int:
    media = probe_video(args.video)
    shot_manifest = (
        detect_shots_ffmpeg(args.video, threshold=args.shot_threshold)
        if args.use_shot_boundaries
        else None
    )
    result = scan_temporal_risk_windows(
        args.video,
        duration_ms=media.duration_ms,
        sampling_fps=args.sampling_fps,
        analysis_width=args.analysis_width,
        analysis_height=args.analysis_height,
        mean_delta_threshold=args.mean_delta_threshold,
        changed_fraction_threshold=args.changed_fraction_threshold,
        pixel_delta_threshold=args.pixel_delta_threshold,
        include_shot_boundaries=args.include_shot_boundaries,
        padding_ms=args.padding_ms,
        merge_gap_ms=args.merge_gap_ms,
        shot_manifest=shot_manifest,
        output_path=args.output,
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
    brief_path = args.brief_json
    plan_prompt = _load_prompt("feature_cut_selects_zh-TW.txt")
    if args.music_first_cue_lock is not None:
        cue_lock_path = args.music_first_cue_lock.expanduser().resolve(strict=True)
        cue_lock = CuePlanLock.model_validate(read_json(cue_lock_path))
        visual_path = Path(cue_lock.plan.visual_sync_map_path).expanduser().resolve(
            strict=True
        )
        visual_map = VisualSyncMap.model_validate(read_json(visual_path))
        original_brief = FeatureEditBrief.model_validate(read_json(args.brief_json))
        guided_brief = apply_music_first_cue_lock(
            original_brief,
            visual_map=visual_map,
            cue_lock=cue_lock,
        )
        input_dir = args.output_dir / "music-first-input"
        input_dir.mkdir(parents=True, exist_ok=True)
        brief_path = input_dir / "brief.music-first.json"
        write_json(brief_path, guided_brief)
        context = {
            "contract_version": "music-first-feature-context-v1",
            "original_brief_path": str(args.brief_json.expanduser().resolve(strict=True)),
            "original_brief_sha256": sha256_file(args.brief_json),
            "music_lock_sha256": cue_lock.plan.music_lock_sha256,
            "cue_plan_lock_path": str(cue_lock_path),
            "cue_plan_lock_sha256": sha256_file(cue_lock_path),
            "global_music_strategy": None,
            "chapter_slots": [
                {
                    "feature_id": chapter.feature_id,
                    "target_duration_seconds": chapter.target_duration_seconds,
                }
                for chapter in guided_brief.chapters
            ],
        }
        if cue_lock.plan.semantic_pairing_used:
            semantic_path = Path(
                cue_lock.plan.semantic_pairing_path or ""
            ).expanduser().resolve(strict=True)
            semantic = SemanticMusicPairingProposal.model_validate(
                read_json(semantic_path)
            )
            context["global_music_strategy"] = semantic.global_strategy
            context["section_interpretations"] = [
                row.model_dump(mode="json")
                for row in semantic.section_interpretations
            ]
        write_json(input_dir / "music-first-context.json", context)
        plan_prompt += (
            "\n\n## 已核准的 music-first 剪輯約束\n"
            "下列資料在選片前建立。請選擇能配合各音樂段落角色、動作完整性"
            "與指定長度的素材；不得為了卡點捏造畫面證據。精確節拍已由本機"
            " CuePlan 鎖定，不得自行改寫時間。\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )
    result = run_feature_cut_experiment(
        catalog_path=args.catalog_json,
        brief_path=brief_path,
        checkpoint_path=args.sam_checkpoint,
        output_dir=args.output_dir,
        plan_prompt=plan_prompt,
        grounding_prompt=_load_prompt("grounding_native_yxyx_zh-TW.txt"),
        scdet_threshold=args.scdet_threshold,
        sam_analysis_fps=args.sam_analysis_fps,
        trim_decision_paths=args.trim_decision,
        allow_proposed_trim_preview=args.allow_proposed_trim_preview,
        reuse_feature_plan=args.reuse_feature_plan,
        aspect=args.aspect,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_build_brief_sync_map(args: argparse.Namespace) -> int:
    visual_map = derive_brief_visual_sync_map(
        args.brief_json,
        aspect_ratio=args.aspect,
        default_flex_ms=args.default_flex_ms,
    )
    write_json(args.output, visual_map)
    print(
        json.dumps(
            {
                "visual_sync_map_path": str(args.output.resolve()),
                "source_kind": visual_map.source_kind,
                "point_count": len(visual_map.points),
                "project_duration_ms": visual_map.project_duration_ms,
                "next_step": "plan-semantic-music",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_analyze_music(args: argparse.Namespace) -> int:
    proposal = analyze_music(args.music)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = args.output_dir / "music-map.proposal.json"
    write_json(proposal_path, proposal)
    write_json(
        args.output_dir / "private-source.json",
        {
            "purpose": "local review only; excluded from public methodology artifacts",
            "path": str(args.music.expanduser().resolve(strict=True)),
            "sha256": proposal.source_sha256,
        },
    )
    print(
        json.dumps(
            {
                "proposal_path": str(proposal_path.resolve()),
                "music_id": proposal.music_id,
                "duration_ms": proposal.duration_ms,
                "estimated_bpm": proposal.estimated_bpm,
                "tempo_confidence": proposal.tempo_confidence,
                "cue_count": len(proposal.cues),
                "section_count": len(proposal.sections),
                "requires_human_review": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_review_music_map(args: argparse.Namespace) -> int:
    proposal = MusicMapProposal.model_validate(read_json(args.proposal_json))
    first_downbeat_sample = (
        round(args.first_downbeat_ms * proposal.master_sample_rate / 1000)
        if args.first_downbeat_ms is not None
        else None
    )
    review, lock = review_music_map(
        proposal,
        proposal_path=args.proposal_json,
        reviewer=args.reviewer,
        decision=args.decision,
        notes=args.notes,
        bpm=args.bpm,
        first_downbeat_sample=first_downbeat_sample,
        meter=args.meter,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / "music-map.review.json"
    write_json(review_path, review)
    result: dict[str, object] = {
        "review_path": str(review_path.resolve()),
        "decision": review.decision,
        "lock_path": None,
    }
    if lock is not None:
        lock_path = args.output_dir / "music-map.lock.json"
        write_json(lock_path, lock)
        result["lock_path"] = str(lock_path.resolve())
        result["locked_bpm"] = lock.bpm
        result["locked_meter"] = lock.meter
        result["locked_cue_count"] = len(lock.cues)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


DEFAULT_MUSIC_FILE_CACHE_ROOT = (
    Path(__file__).resolve().parents[2] / "artifacts" / "music-file-cache"
)


def command_build_visual_sync_map(args: argparse.Namespace) -> int:
    visual_map = derive_visual_sync_map(
        args.render_manifest,
        aspect_ratio=args.aspect,
        default_flex_ms=args.default_flex_ms,
    )
    write_json(args.output, visual_map)
    print(
        json.dumps(
            {
                "visual_sync_map_path": str(args.output.resolve()),
                "aspect_ratio": visual_map.aspect_ratio,
                "project_duration_ms": visual_map.project_duration_ms,
                "point_count": len(visual_map.points),
                "flexibility_authorization": visual_map.flexibility_authorization,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_plan_semantic_music(args: argparse.Namespace) -> int:
    music_lock = MusicMapLock.model_validate(read_json(args.music_lock))
    visual_map = VisualSyncMap.model_validate(read_json(args.visual_sync_map))
    music_source = args.music.expanduser().resolve(strict=True)
    if sha256_file(music_source) != music_lock.music_id.removeprefix("sha256:"):
        raise ValueError("music file does not match the approved MusicMap lock")
    visual_digest = sha256_file(args.visual_sync_map.expanduser().resolve(strict=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    music_sha256 = sha256_file(music_source)
    file_cache_root = (
        args.file_cache_root.expanduser().resolve()
        if args.file_cache_root is not None
        else DEFAULT_MUSIC_FILE_CACHE_ROOT
    )
    upload_dir = file_cache_root / music_sha256 / "upload"
    client = GeminiLabClient()
    try:
        uploaded, reused = client.ensure_video_upload(
            music_source,
            upload_dir,
            force_reupload=args.force_reupload,
        )
        write_json(
            args.output_dir / "upload-cache-binding.json",
            {
                "contract_version": "music-file-cache-binding-v1",
                "music_sha256": music_sha256,
                "cache_directory": str(upload_dir.resolve()),
                "file_api_reused": reused,
            },
        )
        proposal = client.plan_music_semantic_pairing(
            music_lock=music_lock,
            visual_map=visual_map,
            visual_sync_map_sha256=visual_digest,
            uploaded_audio=uploaded,
            prompt_template=_load_prompt("music_semantic_pairing_zh-TW.txt"),
            run_id=f"music-semantic-{uuid.uuid4().hex[:12]}",
            run_dir=args.output_dir,
            reuse_raw_output=args.reuse_raw_output,
        )
    finally:
        client.close()
    pricing = summarize_usage_and_list_price(args.output_dir)
    write_json(args.output_dir / "pricing.json", pricing)
    print(
        json.dumps(
            {
                "proposal_path": str(
                    (args.output_dir / "semantic-music-pairing.proposal.json").resolve()
                ),
                "model": MODEL_ID,
                "file_api_reused": reused,
                "section_interpretation_count": len(
                    proposal.section_interpretations
                ),
                "pairing_count": len(proposal.pairings),
                "pricing": pricing,
                "requires_human_review": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_plan_music_cues(args: argparse.Namespace) -> int:
    music_lock = MusicMapLock.model_validate(read_json(args.music_lock))
    visual_map = VisualSyncMap.model_validate(read_json(args.visual_sync_map))
    semantic_pairing = (
        SemanticMusicPairingProposal.model_validate(read_json(args.semantic_pairing))
        if args.semantic_pairing is not None
        else None
    )
    plan = plan_music_cues(
        music_lock,
        visual_map,
        music_lock_path=args.music_lock,
        visual_sync_map_path=args.visual_sync_map,
        preset=args.preset,
        semantic_pairing=semantic_pairing,
        semantic_pairing_path=args.semantic_pairing,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "cue-plan.proposal.json"
    write_json(plan_path, plan)
    review_path: Path | None = None
    if args.music is not None:
        if sha256_file(args.music.expanduser().resolve(strict=True)) != music_lock.music_id.removeprefix(
            "sha256:"
        ):
            raise ValueError("review music file does not match the locked MusicMap")
        review_path = render_cue_review(
            music_path=args.music,
            video_path=args.video,
            visual_map=visual_map,
            plan=plan,
            output_path=args.output_dir / "cue-review.html",
        )
    print(
        json.dumps(
            {
                "cue_plan_path": str(plan_path.resolve()),
                "review_path": str(review_path) if review_path else None,
                "aligned_count": plan.aligned_count,
                "unmatched_count": plan.unmatched_count,
                "hard_unmatched_count": plan.hard_unmatched_count,
                "changes_applied": False,
                "requires_human_review": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def command_review_cue_plan(args: argparse.Namespace) -> int:
    plan = CuePlanProposal.model_validate(read_json(args.cue_plan))
    review, lock = review_cue_plan(
        plan,
        cue_plan_path=args.cue_plan,
        reviewer=args.reviewer,
        decision=args.decision,
        notes=args.notes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / "cue-plan.review.json"
    write_json(review_path, review)
    result: dict[str, object] = {
        "review_path": str(review_path.resolve()),
        "decision": review.decision,
        "lock_path": None,
    }
    if lock is not None:
        lock_path = args.output_dir / "cue-plan.lock.json"
        write_json(lock_path, lock)
        result["lock_path"] = str(lock_path.resolve())
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
        dense_mode=args.dense_mode,
        dense_event_ids=set(args.dense_event),
        dense_window_ms=args.dense_window_ms,
        dense_fps_override=args.dense_fps,
        file_cache_root=args.file_cache_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_trim_event(args: argparse.Namespace) -> int:
    result = run_trim_intent_event(
        args.clip_run_dir,
        args.event_id,
        args.output_dir,
        prompt_template=_load_prompt("trim_intent_frame_selection_zh-TW.txt"),
        editorial_intent=args.editorial_intent,
        sampling_fps=args.sampling_fps,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_review_trim(args: argparse.Namespace) -> int:
    decision = review_trim_decision(
        args.decision_json,
        args.output,
        reviewer=args.reviewer,
        decision=args.decision,
        notes=args.notes,
    )
    print(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, indent=2))
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
        prepare_only=args.prepare_only,
        file_cache_root=args.file_cache_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_full_selected(args: argparse.Namespace) -> int:
    result = run_selected_full_clips(
        catalog_path=args.catalog_json,
        plan_path=args.plan_json,
        prepared_library_dir=args.prepared_library,
        output_dir=args.output_dir,
        clip_card_prompt=_load_prompt("full_clip_card_mmss_zh-TW.txt"),
        dense_prompt=_load_prompt("dense_event_frame_selection_zh-TW.txt"),
        max_clips=args.max_clips,
        audio_mode=args.audio_mode,
        prepare_only=args.prepare_only,
        file_cache_root=args.file_cache_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["failed"] == 0 else 1


def command_full_ground_event(args: argparse.Namespace) -> int:
    result = run_full_event_geometry(
        args.clip_run_dir,
        args.event_id,
        grounding_prompt=_load_prompt("grounding_native_yxyx_zh-TW.txt"),
        checkpoint_path=args.sam_checkpoint,
        target_entity_id=args.target_entity_id,
        target_description=args.target_description,
        accept_proposed_target=args.accept_proposed_target,
        grounding_candidate_number=args.grounding_candidate_number,
        query_lock_path=args.query_lock,
        query_target_id=args.query_target_id,
        query_reference_dir=args.query_reference_dir,
        predicate_decision_path=args.predicate_decision,
        predicate_prompt_template=_load_prompt(
            "query_predicate_frame_selection_zh-TW.txt"
        ),
        sam_analysis_fps=args.sam_analysis_fps,
        identity_checkpoint_budget=args.identity_checkpoint_budget,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_refine_query_predicate(args: argparse.Namespace) -> int:
    result = run_query_predicate_refinement(
        args.clip_run_dir,
        args.event_id,
        query_lock_path=args.query_lock,
        query_target_id=args.query_target_id,
        prompt_template=_load_prompt("query_predicate_frame_selection_zh-TW.txt"),
        sampling_fps=args.sampling_fps,
        window_ms=args.window_ms,
        output_dir=args.output_dir,
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
    client = GeminiLabClient()
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
        "model": MODEL_ID,
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

    client = GeminiLabClient()
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
    ground_client = GeminiLabClient()
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
        "model": MODEL_ID,
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
        )
    client = GeminiLabClient()
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
        "model": MODEL_ID,
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
        client = GeminiLabClient()
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
                    "model": MODEL_ID,
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
        description=f"{MODEL_ID} video understanding and single-frame grounding lab",
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
    temporal_parser.add_argument("--output-dir", type=Path, required=True)
    temporal_parser.set_defaults(handler=command_temporal_repeat)

    storyboard_parser = subparsers.add_parser(
        "storyboard-temporal",
        help="Build a PTS-indexed storyboard and let Gemini select frame IDs, never timestamps",
    )
    storyboard_parser.add_argument("artifact_root", type=Path)
    storyboard_parser.add_argument("--interval-ms", type=int, default=4000)
    storyboard_parser.add_argument("--max-width", type=int, default=768)
    storyboard_parser.add_argument("--output-dir", type=Path, required=True)
    storyboard_parser.set_defaults(handler=command_storyboard_temporal)

    candidates_parser = subparsers.add_parser(
        "suggest-targets",
        help="When no target was requested, propose user-selectable objects without Grounding",
    )
    candidates_parser.add_argument("artifact_root", type=Path)
    candidates_parser.add_argument("--runs", type=int, default=1)
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
    direct_parser.add_argument("--output-dir", type=Path, required=True)
    direct_parser.set_defaults(handler=command_direct_moment_repeat)

    tracking_parser = subparsers.add_parser(
        "track-csrt",
        help="Experimental bbox propagation on an isolated optional OpenCV path",
    )
    tracking_parser.add_argument("video", type=Path)
    tracking_parser.add_argument("--grounding-json", type=Path)
    tracking_parser.add_argument(
        "--grounding-candidate-number",
        type=int,
        help="1-based operator selection matching the number drawn on the debug overlay",
    )
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
    sam_tracking_parser.add_argument(
        "--grounding-candidate-number",
        type=int,
        help="1-based operator selection matching the number drawn on the debug overlay",
    )
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

    shared_sam_parser = subparsers.add_parser(
        "track-shared-sam21",
        help="Track multiple bbox-seeded objects in one SAM 2.1 inference state",
    )
    shared_sam_parser.add_argument("video", type=Path)
    shared_sam_parser.add_argument("--checkpoint", type=Path, required=True)
    shared_sam_parser.add_argument(
        "--targets-json",
        type=Path,
        required=True,
        help="SharedSam21TrackingRequest JSON containing two or more bbox targets",
    )
    shared_sam_parser.add_argument("--analysis-fps", type=float, default=2.0)
    shared_sam_parser.add_argument("--max-side", type=int, default=960)
    shared_sam_parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    shared_sam_parser.add_argument("--ffmpeg-scdet-threshold", type=float, default=4.0)
    shared_sam_parser.add_argument("--seed-box-padding-ratio", type=float, default=0.0)
    shared_sam_parser.add_argument("--allowed-start-ms", type=int)
    shared_sam_parser.add_argument("--allowed-end-ms", type=int)
    shared_sam_parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep decoded video tensors on CPU by default; use "
            "--no-offload-video-to-cpu only after measuring device memory"
        ),
    )
    shared_sam_parser.add_argument(
        "--offload-state-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload recurrent state to CPU; normally slower on MPS",
    )
    shared_sam_parser.add_argument("--output-dir", type=Path, required=True)
    shared_sam_parser.set_defaults(handler=command_track_shared_sam21)

    multi_sam_parser = subparsers.add_parser(
        "render-multi-sam21",
        help="Combine aligned SAM tracks into a normal-duration manual-review MP4",
    )
    multi_sam_parser.add_argument("track_json", type=Path, nargs="+")
    multi_sam_parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Repeat once per track, in the same order as the track JSON arguments",
    )
    multi_sam_parser.add_argument("--display-fps", type=float, default=30.0)
    multi_sam_parser.add_argument(
        "--analysis-frames-dir",
        type=Path,
        help=(
            "Explicit shared session analysis-frames directory; validates the "
            "adjacent manifest, decoded PTS, dimensions, and frame hashes"
        ),
    )
    multi_sam_parser.add_argument("--output-dir", type=Path, required=True)
    multi_sam_parser.set_defaults(handler=command_render_multi_sam21)

    segmentation_comparison_parser = subparsers.add_parser(
        "compare-sam21-tracks",
        help="Compare two aligned SAM mask tracks as symmetric agreement, not accuracy",
    )
    segmentation_comparison_parser.add_argument("track_a_json", type=Path)
    segmentation_comparison_parser.add_argument("track_b_json", type=Path)
    segmentation_comparison_parser.add_argument("--output", type=Path, required=True)
    segmentation_comparison_parser.set_defaults(handler=command_compare_sam21_tracks)

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

    risk_parser = subparsers.add_parser(
        "scan-temporal-risk",
        help=(
            "Find recall-only local visual-change windows independently of "
            "Gemini Clip Card events"
        ),
    )
    risk_parser.add_argument("video", type=Path)
    risk_parser.add_argument("--sampling-fps", type=float, default=4.0)
    risk_parser.add_argument("--analysis-width", type=int, default=256)
    risk_parser.add_argument("--analysis-height", type=int, default=256)
    risk_parser.add_argument("--mean-delta-threshold", type=float, default=0.04)
    risk_parser.add_argument(
        "--changed-fraction-threshold", type=float, default=0.08
    )
    risk_parser.add_argument("--pixel-delta-threshold", type=int, default=20)
    risk_parser.add_argument("--padding-ms", type=int, default=500)
    risk_parser.add_argument("--merge-gap-ms", type=int, default=500)
    risk_parser.add_argument(
        "--use-shot-boundaries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run FFmpeg scdet so known cuts can be excluded from transient windows",
    )
    risk_parser.add_argument("--shot-threshold", type=float, default=4.0)
    risk_parser.add_argument(
        "--include-shot-boundaries",
        action="store_true",
        help="Include known shot-boundary changes in the recall windows",
    )
    risk_parser.add_argument("--output", type=Path, required=True)
    risk_parser.set_defaults(handler=command_scan_temporal_risk)

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
    feature_cut_parser.add_argument(
        "--aspect",
        choices=["both", "9x16", "16x9"],
        default="both",
        help=(
            "Render both aspect ratios (default), only the vertical 9:16 cut, "
            "or only the horizontal 16:9 cut. Unrequested geometry is not computed."
        ),
    )
    feature_cut_parser.add_argument(
        "--trim-decision",
        type=Path,
        action="append",
        default=[],
        help=(
            "Explicitly human-approved trim decision JSON; repeat for multiple selected events. "
            "Proposed or rejected decisions are refused."
        ),
    )
    feature_cut_parser.add_argument(
        "--reuse-feature-plan",
        action="store_true",
        help=(
            "Explicitly reuse the saved editorial feature plan in output-dir; "
            "geometry and rendered segments are still recomputed and provenance is recorded."
        ),
    )
    feature_cut_parser.add_argument(
        "--allow-proposed-trim-preview",
        action="store_true",
        help=(
            "Render an explicitly unreviewed proposal cut for human review. "
            "The manifest remains marked as requiring review; rejected trims are never accepted."
        ),
    )
    feature_cut_parser.add_argument(
        "--music-first-cue-lock",
        type=Path,
        help=(
            "Approved pre-selection CuePlan lock. Its music-aware chapter slots "
            "are applied before Gemini selects media; post-render cue planning "
            "remains a QC/refinement stage."
        ),
    )
    feature_cut_parser.add_argument("--output-dir", type=Path, required=True)
    feature_cut_parser.set_defaults(handler=command_feature_cut)

    analyze_music_parser = subparsers.add_parser(
        "analyze-music",
        help=(
            "Analyze a local music track into a review-required beat, accent, "
            "energy, and section proposal without Gemini"
        ),
    )
    analyze_music_parser.add_argument("music", type=Path)
    analyze_music_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_music_parser.set_defaults(handler=command_analyze_music)

    review_music_parser = subparsers.add_parser(
        "review-music-map",
        help="Approve or reject a MusicMap proposal and create an immutable lock",
    )
    review_music_parser.add_argument("proposal_json", type=Path)
    review_music_parser.add_argument("--reviewer", required=True)
    review_music_parser.add_argument(
        "--decision", choices=["approved", "rejected"], required=True
    )
    review_music_parser.add_argument("--notes", default="")
    review_music_parser.add_argument("--bpm", type=float)
    review_music_parser.add_argument(
        "--first-downbeat-ms",
        type=int,
        help=(
            "Reviewed first downbeat on the music timeline. If omitted on approval, "
            "the analyzer proposal is used."
        ),
    )
    review_music_parser.add_argument("--meter", type=int)
    review_music_parser.add_argument("--output-dir", type=Path, required=True)
    review_music_parser.set_defaults(handler=command_review_music_map)

    brief_sync_parser = subparsers.add_parser(
        "build-brief-sync-map",
        help=(
            "Build pre-selection visual intents from an editorial brief so Gemini "
            "can interpret and pair the music before media selection"
        ),
    )
    brief_sync_parser.add_argument("brief_json", type=Path)
    brief_sync_parser.add_argument(
        "--aspect", choices=["16:9", "9:16"], default="16:9"
    )
    brief_sync_parser.add_argument(
        "--default-flex-ms",
        type=int,
        default=3_000,
        help="Maximum reviewed timing movement around provisional chapter boundaries",
    )
    brief_sync_parser.add_argument("--output", type=Path, required=True)
    brief_sync_parser.set_defaults(handler=command_build_brief_sync_map)

    visual_sync_parser = subparsers.add_parser(
        "build-visual-sync-map",
        help=(
            "Derive chapter-boundary visual sync points from a rendered feature-cut "
            "manifest; no source trim is changed"
        ),
    )
    visual_sync_parser.add_argument("render_manifest", type=Path)
    visual_sync_parser.add_argument(
        "--aspect", choices=["16:9", "9:16"], required=True
    )
    visual_sync_parser.add_argument(
        "--default-flex-ms",
        type=int,
        default=0,
        help=(
            "Explicitly authorize this much timing movement around derived boundaries. "
            "Default 0 keeps the map read-only."
        ),
    )
    visual_sync_parser.add_argument("--output", type=Path, required=True)
    visual_sync_parser.set_defaults(handler=command_build_visual_sync_map)

    semantic_music_parser = subparsers.add_parser(
        "plan-semantic-music",
        help=(
            "Optionally let Gemini interpret one music track and pair locked cue IDs "
            "with existing visual event IDs; exact timing remains local"
        ),
    )
    semantic_music_parser.add_argument("music", type=Path)
    semantic_music_parser.add_argument("music_lock", type=Path)
    semantic_music_parser.add_argument("visual_sync_map", type=Path)
    semantic_music_parser.add_argument(
        "--force-reupload",
        action="store_true",
        help="Ignore an ACTIVE saved File API object and upload the music again",
    )
    semantic_music_parser.add_argument(
        "--reuse-raw-output",
        action="store_true",
        help=(
            "Canonicalize and revalidate the saved paid response without "
            "creating another Gemini interaction"
        ),
    )
    semantic_music_parser.add_argument(
        "--file-cache-root",
        type=Path,
        help=(
            "Shared SHA-256 keyed File API cache. Defaults to "
            "artifacts/music-file-cache so multiple aspect ratios reuse one upload."
        ),
    )
    semantic_music_parser.add_argument("--output-dir", type=Path, required=True)
    semantic_music_parser.set_defaults(handler=command_plan_semantic_music)

    cue_plan_parser = subparsers.add_parser(
        "plan-music-cues",
        help=(
            "Globally align approved visual sync points to an approved MusicMap; "
            "the output is a review proposal and does not edit media"
        ),
    )
    cue_plan_parser.add_argument("music_lock", type=Path)
    cue_plan_parser.add_argument("visual_sync_map", type=Path)
    cue_plan_parser.add_argument(
        "--preset",
        choices=["narrative", "balanced", "montage"],
        default="balanced",
    )
    cue_plan_parser.add_argument(
        "--semantic-pairing",
        type=Path,
        help=(
            "Optional Gemini semantic pairing proposal. It only adds ranking "
            "preferences; local timing windows and global ordering remain authoritative."
        ),
    )
    cue_plan_parser.add_argument(
        "--music",
        type=Path,
        help="Matching local music file used only to build the HTML review player",
    )
    cue_plan_parser.add_argument(
        "--video",
        type=Path,
        help="Optional rendered picture edit used only in the HTML review player",
    )
    cue_plan_parser.add_argument("--output-dir", type=Path, required=True)
    cue_plan_parser.set_defaults(handler=command_plan_music_cues)

    cue_review_parser = subparsers.add_parser(
        "review-cue-plan",
        help="Approve or reject a hash-bound CuePlan proposal",
    )
    cue_review_parser.add_argument("cue_plan", type=Path)
    cue_review_parser.add_argument("--reviewer", required=True)
    cue_review_parser.add_argument(
        "--decision", choices=["approved", "rejected"], required=True
    )
    cue_review_parser.add_argument("--notes", default="")
    cue_review_parser.add_argument("--output-dir", type=Path, required=True)
    cue_review_parser.set_defaults(handler=command_review_cue_plan)

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
    full_library_parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create local proxies, hashes, shots, and audit frames without Gemini network calls",
    )
    full_library_parser.add_argument("--output-dir", type=Path, required=True)
    full_library_parser.add_argument("--file-cache-root", type=Path)
    full_library_parser.set_defaults(handler=command_full_library)

    full_selected_parser = subparsers.add_parser(
        "full-selected",
        help="Run Full Clip Cards only for source clips referenced by a feature edit plan",
    )
    full_selected_parser.add_argument("catalog_json", type=Path)
    full_selected_parser.add_argument("plan_json", type=Path)
    full_selected_parser.add_argument("--prepared-library", type=Path, required=True)
    full_selected_parser.add_argument("--max-clips", type=int)
    full_selected_parser.add_argument(
        "--audio-mode",
        choices=["auto", "off", "required"],
        default="auto",
    )
    full_selected_parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Resolve and validate selected prepared proxies without Gemini network calls",
    )
    full_selected_parser.add_argument("--file-cache-root", type=Path)
    full_selected_parser.add_argument("--output-dir", type=Path, required=True)
    full_selected_parser.set_defaults(handler=command_full_selected)

    trim_event_parser = subparsers.add_parser(
        "trim-event",
        help="Refine one Clip Card event into frame-ID/PTS trim phases for human review",
    )
    trim_event_parser.add_argument("clip_run_dir", type=Path)
    trim_event_parser.add_argument("event_id")
    trim_event_parser.add_argument(
        "--editorial-intent",
        default=(
            "保留可理解的完整動作與結果；若存在疑似刻意停留或適合文字的負空間，"
            "保留為待人工確認的 hold proposal。"
        ),
    )
    trim_event_parser.add_argument(
        "--sampling-fps", type=float, choices=[2.0, 4.0, 8.0], default=4.0
    )
    trim_event_parser.add_argument("--output-dir", type=Path, required=True)
    trim_event_parser.set_defaults(handler=command_trim_event)

    review_trim_parser = subparsers.add_parser(
        "review-trim",
        help="Approve or reject a saved trim proposal without another model call",
    )
    review_trim_parser.add_argument("decision_json", type=Path)
    review_trim_parser.add_argument("--decision", choices=["approved", "rejected"], required=True)
    review_trim_parser.add_argument("--reviewer", required=True)
    review_trim_parser.add_argument("--notes", default="")
    review_trim_parser.add_argument("--output", type=Path, required=True)
    review_trim_parser.set_defaults(handler=command_review_trim)

    refine_query_parser = subparsers.add_parser(
        "refine-query-predicate",
        help=(
            "Explicitly use one Gemini request to select lock-aware DF evidence IDs; "
            "no bbox or SAM is performed"
        ),
    )
    refine_query_parser.add_argument("clip_run_dir", type=Path)
    refine_query_parser.add_argument("event_id")
    refine_query_parser.add_argument("--query-lock", type=Path, required=True)
    refine_query_parser.add_argument(
        "--query-target-id",
        help="Required when the QueryLock contains multiple identity targets",
    )
    refine_query_parser.add_argument(
        "--sampling-fps", type=float, choices=[4.0, 8.0], default=8.0
    )
    refine_query_parser.add_argument(
        "--window-ms",
        type=int,
        default=4000,
        help="Shot-local dense window in milliseconds (1000..5000)",
    )
    refine_query_parser.add_argument("--output-dir", type=Path)
    refine_query_parser.set_defaults(handler=command_refine_query_predicate)

    full_ground_parser = subparsers.add_parser(
        "full-ground-event",
        help="Ground one selected Clip Card event and optionally propagate SAM in that interval",
    )
    full_ground_parser.add_argument("clip_run_dir", type=Path)
    full_ground_parser.add_argument("event_id")
    full_ground_parser.add_argument("--target-entity-id")
    full_ground_parser.add_argument("--target-description")
    full_ground_parser.add_argument(
        "--query-lock",
        type=Path,
        help="Immutable, domain-neutral evidence query contract used instead of target flags",
    )
    full_ground_parser.add_argument(
        "--query-target-id",
        help="Select one target when the query lock contains multiple target references",
    )
    full_ground_parser.add_argument(
        "--query-reference-dir",
        type=Path,
        help=(
            "Directory of content-addressed identity crops named <sha256>.<image-ext>; "
            "required only when a v2 lock contains positive or negative anchors"
        ),
    )
    full_ground_parser.add_argument(
        "--predicate-decision",
        type=Path,
        help=(
            "Resolved query_temporal.decision.json from refine-query-predicate; "
            "required when a QueryLock v2 contains a predicate"
        ),
    )
    full_ground_parser.add_argument(
        "--accept-proposed-target",
        action="store_true",
        help="Explicitly accept the Clip Card target only when exactly one proposal exists",
    )
    full_ground_parser.add_argument(
        "--grounding-candidate-number",
        type=int,
        help="1-based human/operator selection matching the number drawn on the debug overlay",
    )
    full_ground_parser.add_argument("--sam-checkpoint", type=Path)
    full_ground_parser.add_argument("--sam-analysis-fps", type=float, default=2.0)
    full_ground_parser.add_argument(
        "--identity-checkpoint-budget",
        type=int,
        default=2,
        help=(
            "Verify at most this many risk-triggered exact frames (0..8). "
            "Planning is local; each selected frame may make one Gemini image call."
        ),
    )
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
