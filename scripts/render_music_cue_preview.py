#!/usr/bin/env python3
"""Render an explicitly unreviewed music-aware preview from a CuePlan proposal.

This adapter never upgrades a proposal into an approved edit.  It applies only
order-preserving alignments already proven to be inside the VisualSyncMap
authorization windows.  Existing rendered chapter segments are gently retimed
to the proposed boundaries, then mixed with the locked source music.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from jascue_video_lab.media import sha256_file
from jascue_video_lab.storage import utc_now, write_json


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_preview_timing(
    visual_map: dict[str, Any],
    cue_plan: dict[str, Any],
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    points = visual_map["points"]
    alignments = {
        row["visual_event_id"]: row for row in cue_plan["alignments"]
    }
    original = [int(point["project_time_ms"]) for point in points]
    proposed: list[int] = []
    audit: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        row = alignments[point["visual_event_id"]]
        use_alignment = (
            row["status"] == "aligned"
            and row["within_authorized_window"] is True
            and row["proposed_project_time_ms"] is not None
            and index not in {0, len(points) - 1}
        )
        chosen = (
            int(row["proposed_project_time_ms"])
            if use_alignment
            else int(point["project_time_ms"])
        )
        proposed.append(chosen)
        audit.append(
            {
                "visual_event_id": point["visual_event_id"],
                "feature_id": point["feature_id"],
                "original_project_time_ms": int(point["project_time_ms"]),
                "applied_project_time_ms": chosen,
                "delta_ms": chosen - int(point["project_time_ms"]),
                "music_cue_id": row.get("music_cue_id") if use_alignment else None,
                "music_cue_kind": row.get("music_cue_kind") if use_alignment else None,
                "application_status": (
                    "applied_inside_authorized_window"
                    if use_alignment
                    else "kept_original_boundary"
                ),
            }
        )
    if proposed[0] != 0 or proposed[-1] != original[-1]:
        raise ValueError("preview must preserve timeline start and end")
    if any(right <= left for left, right in zip(proposed, proposed[1:])):
        raise ValueError("proposed boundaries must remain strictly ordered")
    return original, proposed, audit


def render(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_json(args.render_manifest)
    visual_map = read_json(args.visual_sync_map)
    cue_plan = read_json(args.cue_plan)
    if cue_plan.get("changes_applied") is not False:
        raise ValueError("input CuePlan must remain an unapplied proposal")
    if cue_plan.get("requires_human_review") is not True:
        raise ValueError("preview input must explicitly require human review")
    if sha256_file(args.visual_sync_map) != cue_plan["visual_sync_map_sha256"]:
        raise ValueError("CuePlan does not match the VisualSyncMap artifact")

    manifest_key = "horizontal" if args.aspect == "16:9" else "vertical"
    chapter_rows = manifest[manifest_key]["chapters"]
    original, proposed, boundary_audit = build_preview_timing(
        visual_map, cue_plan
    )
    if len(chapter_rows) != len(original) - 1:
        raise ValueError("rendered chapters do not match visual boundaries")

    segment_paths = [Path(row["segment_path"]).resolve(strict=True) for row in chapter_rows]
    segment_audit: list[dict[str, Any]] = []
    filter_parts: list[str] = []
    for index, (row, segment_path) in enumerate(zip(chapter_rows, segment_paths)):
        original_duration = original[index + 1] - original[index]
        applied_duration = proposed[index + 1] - proposed[index]
        ratio = applied_duration / original_duration
        if not 0.9 <= ratio <= 1.1:
            raise ValueError(
                f"preview retime exceeds ten percent for {row['feature_id']}: {ratio}"
            )
        filter_parts.append(
            f"[{index}:v]setpts={ratio:.9f}*PTS,"
            "fps=30,format=yuv420p"
            f"[v{index}]"
        )
        segment_audit.append(
            {
                "feature_id": row["feature_id"],
                "segment_path": str(segment_path),
                "segment_sha256": sha256_file(segment_path),
                "original_duration_ms": original_duration,
                "applied_duration_ms": applied_duration,
                "retime_ratio": round(ratio, 9),
                "source_clip_id": row["source_clip_id"],
                "source_in_ms": row["source_in_ms"],
                "source_out_ms": row["source_out_ms"],
            }
        )
    concat_inputs = "".join(f"[v{index}]" for index in range(len(segment_paths)))
    filter_parts.append(
        f"{concat_inputs}concat=n={len(segment_paths)}:v=1:a=0[vout]"
    )
    total_seconds = proposed[-1] / 1000
    fade_start = max(0.0, total_seconds - 1.0)
    music_index = len(segment_paths)
    filter_parts.append(
        f"[{music_index}:a]atrim=0:{total_seconds:.6f},"
        "asetpts=PTS-STARTPTS,"
        f"afade=t=out:st={fade_start:.6f}:d=1[aout]"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-y"]
    for path in segment_paths:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-i",
            str(args.music.resolve(strict=True)),
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(args.output),
        ]
    )
    subprocess.run(command, check=True)
    application = {
        "contract_version": "music-cue-preview-application-v1",
        "aspect_ratio": args.aspect,
        "status": "rendered_unreviewed_preview",
        "requires_human_review": True,
        "approval_status": "not_approved",
        "timing_method": (
            "Gemini semantic cue preference plus local sample-accurate, "
            "order-preserving CuePlan; per-segment preview retime"
        ),
        "source_audio_policy": "replaced_with_music",
        "cue_plan_path": str(args.cue_plan.resolve(strict=True)),
        "cue_plan_sha256": sha256_file(args.cue_plan),
        "visual_sync_map_path": str(args.visual_sync_map.resolve(strict=True)),
        "visual_sync_map_sha256": sha256_file(args.visual_sync_map),
        "render_manifest_path": str(args.render_manifest.resolve(strict=True)),
        "render_manifest_sha256": sha256_file(args.render_manifest),
        "music_path": str(args.music.resolve(strict=True)),
        "music_sha256": sha256_file(args.music),
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "original_project_duration_ms": original[-1],
        "applied_project_duration_ms": proposed[-1],
        "boundaries": boundary_audit,
        "segments": segment_audit,
        "warnings": [
            "This is a review preview, not an approved production timeline.",
            "Small per-segment retimes demonstrate cue application; source-handle-aware "
            "re-trimming remains a future production path.",
            "Picture-edit geometry and semantic identity warnings remain unchanged.",
        ],
        "generated_at": utc_now(),
    }
    write_json(args.audit, application)
    return application


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("render_manifest", type=Path)
    parser.add_argument("visual_sync_map", type=Path)
    parser.add_argument("cue_plan", type=Path)
    parser.add_argument("music", type=Path)
    parser.add_argument("--aspect", choices=("16:9", "9:16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    result = render(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
