from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from .ab_review import render_grounding_ab_review
from .compare import compare_runs
from .fixtures import generate_fixtures
from .gemini import GeminiLabClient
from .media import extract_frame, probe_video, sha256_file
from .models import ContentMap, ExtractedFrame, GroundingProposal, MediaInfo, TemporalEvent, TemporalMap
from .overlay import draw_grounding_overlay
from .review import render_manual_review
from .repeat import run_repeated_grounding
from .storage import append_error, read_json, write_json
from .timeline import render_direct_moment_timeline, render_temporal_timeline, render_timeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_name(value: str) -> str:
    clean = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return clean[:100] or "unnamed"


def _load_prompt(name: str) -> str:
    return (PROJECT_ROOT / "prompts" / name).read_text(encoding="utf-8")


def _default_artifact_root(asset_hash: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "artifacts" / asset_hash[:12] / stamp


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
        uploaded = client.resume_video_upload(args.artifact_root / "upload")
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


def command_direct_moment_repeat(args: argparse.Namespace) -> int:
    media = MediaInfo.model_validate(read_json(args.artifact_root / "media.json"))
    source = args.artifact_root / "source.mp4"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = GeminiLabClient(temperature=args.temperature)
    summaries: list[dict[str, object]] = []
    failures = 0
    try:
        uploaded = client.resume_video_upload(args.artifact_root / "upload")
        for run_number in range(1, args.runs + 1):
            run_id = f"direct-mmss-{run_number:02d}-{uuid.uuid4().hex[:8]}"
            run_dir = args.output_dir / f"run-{run_number:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                moments = client.analyze_direct_moments(
                    media=media,
                    uploaded=uploaded,
                    prompt_template=_load_prompt("direct_moments_mmss_zh-TW.txt"),
                    run_id=run_id,
                    run_dir=run_dir,
                )
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
                    for moment in moments.moments:
                        requested_ms = _mmss_to_ms(moment.timestamp_mmss)
                        moment_dir = run_dir / "moments" / _safe_name(moment.moment_id)
                        frame = extract_frame(source, requested_ms, moment_dir / "frame.png")
                        write_json(moment_dir / "frame.json", frame)
                        grounding_dir = moment_dir / "grounding"
                        proposal = client.ground_frame(
                            media=media,
                            frame=frame,
                            event_id=moment.moment_id,
                            event_description=f"{moment.label}；{moment.observable_evidence}",
                            entity_id=moment.grounding_target_id,
                            target_description=moment.grounding_target_description,
                            prompt_template=_load_prompt("grounding_native_yxyx_zh-TW.txt"),
                            run_id=run_id,
                            output_dir=grounding_dir,
                        )
                        overlay_path = grounding_dir / "debug.png"
                        draw_grounding_overlay(Path(frame.path), proposal, overlay_path)
                        timeline_results.append(
                            (moment.moment_id, requested_ms, frame.frame_time_ms, overlay_path, proposal)
                        )
                        grounding_summary.append(
                            {
                                "moment_id": moment.moment_id,
                                "requested_ms": requested_ms,
                                "frame_time_ms": frame.frame_time_ms,
                                "visible": proposal.visible,
                                "candidate_count": len(proposal.candidates),
                            }
                        )
                    render_direct_moment_timeline(
                        moment_map=moments,
                        video_path=source,
                        results=timeline_results,
                        output_path=run_dir / "index.html",
                    )
                    run_summary["groundings"] = grounding_summary
                summaries.append(run_summary)
            except Exception as error:
                failures += 1
                append_error(run_dir, "direct_moment_pipeline", error)
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
        "method": "direct Gemini MM:SS salient moments",
        "runs_requested": args.runs,
        "runs_succeeded": args.runs - failures,
        "grounded_runs": min(args.ground_runs, args.runs),
        "failure_count": failures,
        "runs": summaries,
    }
    write_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if failures == 0 else 1


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
        if args.resume_upload:
            uploaded = client.resume_video_upload(artifact_root / "upload")
        else:
            uploaded = client.upload_video(source_link, artifact_root / "upload")
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
        help="Resume a saved File API upload instead of uploading the video again",
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

    direct_parser = subparsers.add_parser(
        "direct-moment-repeat",
        help="Ask Gemini directly for official MM:SS screenshot moments and optionally ground them",
    )
    direct_parser.add_argument("artifact_root", type=Path)
    direct_parser.add_argument("--runs", type=int, default=3)
    direct_parser.add_argument("--ground-runs", type=int, default=1)
    direct_parser.add_argument("--temperature", type=float, default=0.2)
    direct_parser.add_argument("--output-dir", type=Path, required=True)
    direct_parser.set_defaults(handler=command_direct_moment_repeat)
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
