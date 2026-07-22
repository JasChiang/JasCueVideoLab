#!/usr/bin/env python3
"""Join editorial candidates, geometry, and render decisions for 9:16 review."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from jascue_video_lab.feature_cut import (
    _track_geometry_fingerprint,
    _usable_track_centers,
    _vertical_crop_geometry,
)
from jascue_video_lab.media import probe_video, sha256_file
from jascue_video_lab.models import FramingRegionIntent, SegmentationTrack
from jascue_video_lab.storage import read_json, write_json


def _manifest_video(
    video_path: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    """Bind an optional audit player to the exact rendered vertical output."""

    vertical = manifest.get("vertical")
    if not isinstance(vertical, dict):
        raise ValueError("render manifest has no vertical section")
    expected = vertical.get("media")
    if not isinstance(expected, dict):
        raise ValueError(
            "--video requires render manifest vertical.media provenance"
        )
    required = ("sha256", "width", "height", "duration_seconds")
    missing = [key for key in required if key not in expected]
    if missing:
        raise ValueError(
            "--video cannot be verified because vertical.media is missing: "
            + ", ".join(missing)
        )

    media = probe_video(video_path)
    expected_duration_ms = round(float(expected["duration_seconds"]) * 1000)
    comparisons = {
        "sha256": (media.sha256, expected["sha256"]),
        "width": (media.video.coded_width, expected["width"]),
        "height": (media.video.coded_height, expected["height"]),
        "duration_ms": (media.duration_ms, expected_duration_ms),
    }
    mismatches = [
        key for key, (actual, wanted) in comparisons.items() if actual != wanted
    ]
    if mismatches:
        detail = ", ".join(
            f"{key}={comparisons[key][0]!r} (expected {comparisons[key][1]!r})"
            for key in mismatches
        )
        raise ValueError(
            "--video does not match render manifest vertical media: " + detail
        )
    return {
        "path": media.path,
        "sha256": media.sha256,
        "width": media.video.coded_width,
        "height": media.video.coded_height,
        "duration_ms": media.duration_ms,
        "validation": "exact_hash_dimensions_duration_match",
    }


def _contained_session_file(session_dir: Path, relative_path: str) -> Path:
    """Resolve a shared-session member without allowing manifest path escape."""

    root = session_dir.resolve(strict=True)
    candidate = (root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"shared SAM track escapes its session: {relative_path}"
        ) from error
    if not candidate.is_file():
        raise ValueError(f"shared SAM track is missing: {relative_path}")
    return candidate


def _region_payloads(
    rendered: dict[str, object],
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    raw_regions = rendered.get("vertical_regions") or []
    if not isinstance(raw_regions, list):
        raise ValueError("rendered vertical_regions must be a list")
    for raw_region in raw_regions:
        region = FramingRegionIntent.model_validate(raw_region)
        regions.append(region.model_dump(mode="json"))
    return regions


def _shared_geometry_fingerprint(
    regions: list[dict[str, Any]],
    track_fingerprints: list[str],
) -> str:
    """Reproduce the renderer's ordered multi-region track fingerprint."""

    return hashlib.sha256(
        json.dumps(
            {"regions": regions, "tracks": track_fingerprints},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _resolve_manifest_tracks(
    *,
    render_manifest_dir: Path,
    feature_id: str,
    rendered: dict[str, object],
) -> tuple[list[tuple[Path, SegmentationTrack]], dict[str, object] | None]:
    """Resolve only track artifacts cryptographically selected by the render.

    A directory glob is discovery, never provenance.  Single-object tracks must
    match the chapter's exact geometry fingerprint.  Multi-object tracks must
    additionally be named and hash-verified by one shared-session manifest, and
    their ordered composite must match the chapter fingerprint.
    """

    expected = rendered.get("track_geometry_fingerprint")
    if expected is None:
        return [], None
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(
            f"invalid track_geometry_fingerprint for feature {feature_id}"
        )

    manifest_root = render_manifest_dir.resolve(strict=True)
    geometry_root = (
        manifest_root / "geometry" / feature_id / "vertical"
    ).resolve(strict=False)
    try:
        geometry_root.relative_to(manifest_root)
    except ValueError as error:
        raise ValueError(f"unsafe feature_id in render manifest: {feature_id}") from error
    if not geometry_root.is_dir():
        raise ValueError(
            f"rendered track fingerprint has no geometry directory: {feature_id}"
        )

    all_regions = _region_payloads(rendered)
    required_regions = [
        region for region in all_regions if region["role"] == "required"
    ]
    required_region_ids = [str(region["region_id"]) for region in required_regions]
    region_geometry_root = geometry_root
    if all_regions:
        region_key = hashlib.sha256(
            json.dumps(
                all_regions,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        region_geometry_root = geometry_root / f"regions-{region_key}"
    matches: list[
        tuple[list[tuple[Path, SegmentationTrack]], dict[str, object]]
    ] = []

    if len(required_regions) >= 2:
        for session_path in sorted(
            region_geometry_root.glob("**/shared-session.json")
        ):
            payload = read_json(session_path)
            if not isinstance(payload, dict):
                continue
            targets = payload.get("targets")
            if not isinstance(targets, list):
                continue
            target_ids = [
                target.get("target_id") if isinstance(target, dict) else None
                for target in targets
            ]
            # Unrelated or stale sessions are not part of this render lineage.
            if target_ids != required_region_ids:
                continue

            tracks: list[tuple[Path, SegmentationTrack]] = []
            track_fingerprints: list[str] = []
            track_artifacts: list[dict[str, object]] = []
            valid_session = True
            for target in targets:
                assert isinstance(target, dict)
                relative_path = target.get("track_path")
                expected_sha256 = target.get("track_sha256")
                if not isinstance(relative_path, str) or not isinstance(
                    expected_sha256, str
                ):
                    valid_session = False
                    break
                try:
                    track_path = _contained_session_file(
                        session_path.parent, relative_path
                    )
                except (OSError, ValueError):
                    valid_session = False
                    break
                actual_sha256 = sha256_file(track_path)
                if actual_sha256 != expected_sha256:
                    valid_session = False
                    break
                try:
                    track = SegmentationTrack.model_validate(read_json(track_path))
                except (OSError, ValueError):
                    valid_session = False
                    break
                fingerprint = _track_geometry_fingerprint(track)
                tracks.append((track_path, track))
                track_fingerprints.append(fingerprint)
                track_artifacts.append(
                    {
                        "path": str(track_path.resolve()),
                        "sha256": actual_sha256,
                        "geometry_fingerprint": fingerprint,
                        "target_id": target["target_id"],
                    }
                )
            if not valid_session:
                continue
            composite = _shared_geometry_fingerprint(
                required_regions, track_fingerprints
            )
            if composite != expected:
                continue
            matches.append(
                (
                    tracks,
                    {
                        "match_kind": "shared_session_composite_geometry_fingerprint",
                        "manifest_track_geometry_fingerprint": expected,
                        "shared_session_manifest_path": str(session_path.resolve()),
                        "shared_session_manifest_sha256": sha256_file(session_path),
                        "track_artifacts": track_artifacts,
                    },
                )
            )

    if len(required_regions) == 1:
        region = required_regions[0]
        region_track_root = (
            region_geometry_root / "regions" / str(region["region_id"])
        )
        for track_path in sorted(
            region_track_root.glob("**/segmentation-track.json")
        ):
            try:
                track = SegmentationTrack.model_validate(read_json(track_path))
            except (OSError, ValueError):
                continue
            fingerprint = _track_geometry_fingerprint(track)
            composite = _shared_geometry_fingerprint(
                required_regions, [fingerprint]
            )
            if composite != expected:
                continue
            track_sha256 = sha256_file(track_path)
            matches.append(
                (
                    [(track_path, track)],
                    {
                        "match_kind": (
                            "single_required_region_composite_geometry_fingerprint"
                        ),
                        "manifest_track_geometry_fingerprint": expected,
                        "track_artifacts": [
                            {
                                "path": str(track_path.resolve()),
                                "sha256": track_sha256,
                                "geometry_fingerprint": fingerprint,
                                "target_id": region["region_id"],
                            }
                        ],
                    },
                )
            )

    single_track_candidates = (
        geometry_root.glob("**/segmentation-track.json")
        if not required_regions
        else []
    )
    for track_path in sorted(single_track_candidates):
        relative_parts = track_path.relative_to(geometry_root).parts
        if "shared-sam21" in relative_parts:
            continue
        try:
            track = SegmentationTrack.model_validate(read_json(track_path))
        except (OSError, ValueError):
            continue
        fingerprint = _track_geometry_fingerprint(track)
        if fingerprint != expected:
            continue
        track_sha256 = sha256_file(track_path)
        matches.append(
            (
                [(track_path, track)],
                {
                    "match_kind": "single_track_geometry_fingerprint",
                    "manifest_track_geometry_fingerprint": expected,
                    "track_artifacts": [
                        {
                            "path": str(track_path.resolve()),
                            "sha256": track_sha256,
                            "geometry_fingerprint": fingerprint,
                        }
                    ],
                },
            )
        )

    if not matches:
        raise ValueError(
            f"no hash/fingerprint-verified track lineage matches feature {feature_id}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"ambiguous track lineage for feature {feature_id}: {len(matches)} matches"
        )
    return matches[0]


def _adapt_legacy_plan(
    plan: dict[str, object],
    brief: dict[str, object],
    vertical: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Preserve legacy clip identity without inventing an asset fingerprint."""

    brief_by_id = {
        item["feature_id"]: item
        for item in brief["chapters"]  # type: ignore[index]
    }
    legacy_shots: list[dict[str, object]] = []
    for chapter in plan["chapters"]:  # type: ignore[index]
        feature_id = chapter["feature_id"]
        brief_chapter = brief_by_id[feature_id]
        rendered = vertical[feature_id]
        legacy_shots.append(
            {
                "feature_id": feature_id,
                "title": brief_chapter["title"],
                "vertical_candidate_id": "legacy_selected",
                "candidates": [
                    {
                        "candidate_id": "legacy_selected",
                        # A clip identifier is not a content-addressed asset ID.
                        # Keep the two namespaces separate so downstream audit
                        # consumers cannot mistake this adapter value for a
                        # verified source fingerprint.
                        "source_asset_id": None,
                        "source_clip_id": rendered.get("source_clip_id"),
                        "event_id": chapter.get("vertical_event_id"),
                        "frame_id": chapter["vertical_frame_id"],
                        "observed_visual_evidence": chapter[
                            "observed_visual_evidence"
                        ],
                        "selection_reason": chapter["selection_reason"],
                        "quality_risks": chapter["quality_risks"],
                        "vertical_strategy": chapter["vertical_strategy"],
                        "vertical_target_description": chapter[
                            "vertical_target_description"
                        ],
                        "vertical_crop_mode": brief_chapter["vertical_crop_mode"],
                        "vertical_regions": brief_chapter.get("vertical_regions", []),
                    }
                ],
            }
        )
    return {
        "project_id": plan["project_id"],
        "shots": legacy_shots,
    }


def _alternative_candidate(candidate: dict[str, object]) -> dict[str, object]:
    """Return audit-safe candidate identity across legacy and current plans."""

    return {
        "candidate_id": candidate["candidate_id"],
        "source_asset_id": candidate.get("source_asset_id"),
        "source_clip_id": candidate.get("source_clip_id"),
        "event_id": candidate.get("event_id"),
        "frame_id": candidate["frame_id"],
        "observed_visual_evidence": candidate["observed_visual_evidence"],
        "selection_reason": candidate["selection_reason"],
        "quality_risks": candidate["quality_risks"],
    }


def _track_audit(
    *,
    rendered: dict[str, object],
    track_files: list[Path],
    matching_tracks: list[tuple[Path, SegmentationTrack]],
    track_lineage: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """Report rendered geometry first; recomputation is comparison-only."""

    # Compatibility callers may still pass track_files, but the audit must
    # never publish a discovered path that was not verified as a manifest
    # match.  matching_tracks is the authoritative resolved set.
    del track_files
    verified_paths = [path for path, _track in matching_tracks]
    if "crop_keyframes" in rendered:
        # crop_keyframes are the exact geometry serialized by the renderer.
        # They remain authoritative for both single- and multi-region renders;
        # recomputing with today's algorithm could otherwise rewrite history.
        geometry_keys = (
            "crop_width_normalized",
            "crop_height_normalized",
            "max_target_width_normalized",
            "max_target_height_normalized",
            "geometry_feasible",
            "full_containment_feasible",
            "controlled_clip_applied",
            "containment_failure_count",
            "minimum_visible_required_width_fraction",
            "minimum_visible_required_height_fraction",
            "minimum_visible_required_area_fraction",
            "max_crop_x_speed_pixels_per_second",
            "max_crop_y_speed_pixels_per_second",
            "max_crop_speed_pixels_per_second",
            "max_crop_acceleration_pixels_per_second_squared",
            "source_x_edge_contact_count",
            "source_y_edge_contact_count",
            "source_boundary_contact_count",
            "source_boundary_contact_ratio",
            "crop_coordinate_space",
            "crop_x_values_pixels",
            "crop_y_values_pixels",
            "crop_keyframes",
        )
        return {
            "source": "render_manifest_authoritative_geometry",
            "authoritative_for_rendered_output": True,
            "track_paths": [str(path.resolve()) for path in verified_paths],
            "track_lineage": track_lineage,
            **{key: rendered[key] for key in geometry_keys if key in rendered},
        }

    if len(matching_tracks) != 1:
        return None

    track_file, track = matching_tracks[0]
    times, centers, boxes = _usable_track_centers(track)
    if not times:
        return None
    if track.seed_source_width is None or track.seed_source_height is None:
        return {
            "source": "current_algorithm_comparison_unavailable",
            "authoritative_for_rendered_output": False,
            "comparison_warning": (
                "legacy track has no orientation-corrected seed dimensions; "
                "the current 2D crop solver cannot safely reinterpret its coordinates"
            ),
            "track_path": str(track_file.resolve()),
            "target_description": track.target_description,
            "track_lineage": track_lineage,
        }
    _, crop_geometry = _vertical_crop_geometry(
        times,
        centers,
        boxes,
        source_width=track.seed_source_width,
        source_height=track.seed_source_height,
        safety_multiplier=float(rendered.get("target_safety_multiplier", 1.0)),
        overflow_policy=rendered.get(
            "vertical_overflow_policy",
            rendered.get("overflow_policy", "preserve_all"),
        ),
        edge_priority=rendered.get(
            "vertical_edge_priority",
            rendered.get("edge_priority", "balanced"),
        ),
    )
    return {
        "source": "current_algorithm_comparison_not_rendered_geometry",
        "authoritative_for_rendered_output": False,
        "comparison_warning": (
            "render manifest did not preserve crop_keyframes; this geometry was "
            "recomputed with the current algorithm and may differ from the rendered video"
        ),
        "track_path": str(track_file.resolve()),
        "target_description": track.target_description,
        "analysis_fps": track.analysis_fps,
        "analysis_start_ms": track.analysis_start_ms,
        "analysis_end_ms": track.analysis_end_ms,
        "state_counts": {str(key): value for key, value in track.state_counts.items()},
        "track_lineage": track_lineage,
        **crop_geometry,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan_json", type=Path)
    parser.add_argument("render_manifest_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--budget-plan", type=Path)
    parser.add_argument(
        "--brief-json",
        type=Path,
        help="Required when plan_json is a legacy single-select feature plan",
    )
    args = parser.parse_args()

    plan = read_json(args.plan_json)
    manifest = read_json(args.render_manifest_json)
    verified_video = (
        _manifest_video(args.video, manifest) if args.video is not None else None
    )
    vertical = {item["feature_id"]: item for item in manifest["vertical"]["chapters"]}
    alternatives_preserved = "shots" in plan
    if not alternatives_preserved:
        if not args.brief_json:
            raise ValueError("legacy feature plans require --brief-json")
        brief = read_json(args.brief_json)
        plan = _adapt_legacy_plan(plan, brief, vertical)
    shots_by_id = {item["feature_id"]: item for item in plan["shots"]}
    ordered_shots = plan["shots"]
    if args.budget_plan:
        budget = read_json(args.budget_plan)
        ordered_shots = [shots_by_id[feature_id] for feature_id in budget["sequence"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timeline_cursor_ms = 0
    audits: list[dict[str, object]] = []
    rows: list[str] = []

    for shot in ordered_shots:
        feature_id = shot["feature_id"]
        rendered = vertical[feature_id]
        selected = next(
            candidate
            for candidate in shot["candidates"]
            if candidate["candidate_id"] == shot["vertical_candidate_id"]
        )
        alternatives = [
            _alternative_candidate(candidate)
            for candidate in shot["candidates"]
            if candidate["candidate_id"] != shot["vertical_candidate_id"]
        ]
        duration_ms = rendered.get("duration_ms")
        if not isinstance(duration_ms, int) or duration_ms <= 0:
            duration_ms = int(rendered["source_out_ms"]) - int(
                rendered["source_in_ms"]
            )
        matching_tracks, track_lineage = _resolve_manifest_tracks(
            render_manifest_dir=args.render_manifest_json.parent,
            feature_id=feature_id,
            rendered=rendered,
        )
        track_files = [path for path, _track in matching_tracks]
        track_audit = _track_audit(
            rendered=rendered,
            track_files=track_files,
            matching_tracks=matching_tracks,
            track_lineage=track_lineage,
        )
        explanation = (
            f"以「{rendered.get('target_description') or '未指定主體'}」作為構圖主體；"
            f"計畫模式為 {rendered.get('vertical_crop_mode', selected['vertical_crop_mode'])}，實際策略為 "
            f"{rendered.get('applied_strategy')}."
        )
        if rendered.get("fallback_reason"):
            explanation += f" 退回原因：{rendered['fallback_reason']}。"
        elif rendered.get("applied_strategy") == "tracked_crop":
            explanation += (
                " crop x／y 先由 SAM required-region union 建立逐時刻合法範圍，"
                "再於兩軸合法範圍內平滑並逐段內插。"
            )
        if rendered.get("controlled_clip_applied"):
            explanation += " 此段明示允許裁掉 required union 的一部分，必須人工確認取捨。"
        elif rendered.get("secondary_context_clipping_allowed"):
            explanation += " 此段只允許犧牲 required region 以外的次要脈絡。"

        risk_codes = list(rendered.get("risk_codes") or [])
        automated_checks = {
            "tracking_coverage_passed": rendered.get("coverage_passed"),
            "required_region_containment_passed": (
                rendered.get("containment_failure_count") == 0
                if rendered.get("containment_failure_count") is not None
                else None
            ),
            "full_containment_feasible": rendered.get("full_containment_feasible"),
            "fallback_free": rendered.get("fallback_reason") is None,
            "max_crop_speed_pixels_per_second": rendered.get(
                "max_crop_speed_pixels_per_second"
            ),
            "max_crop_x_speed_pixels_per_second": rendered.get(
                "max_crop_x_speed_pixels_per_second"
            ),
            "max_crop_y_speed_pixels_per_second": rendered.get(
                "max_crop_y_speed_pixels_per_second"
            ),
            "max_crop_acceleration_pixels_per_second_squared": rendered.get(
                "max_crop_acceleration_pixels_per_second_squared"
            ),
        }
        audit = {
            "feature_id": feature_id,
            "title": shot["title"],
            "timeline_start_ms": timeline_cursor_ms,
            "timeline_end_ms": timeline_cursor_ms + duration_ms,
            "source_clip_id": rendered.get("source_clip_id"),
            "source_in_ms": rendered.get("source_in_ms"),
            "source_out_ms": rendered.get("source_out_ms"),
            "selected_candidate": selected,
            "alternative_candidates": alternatives,
            "alternative_candidates_preserved": alternatives_preserved,
            "alternatives_unavailable_reason": (
                None
                if alternatives_preserved
                else "legacy_single-select_plan_did_not_preserve_top_k"
            ),
            "applied_strategy": rendered.get("applied_strategy"),
            "fallback_reason": rendered.get("fallback_reason"),
            "target_description": rendered.get("target_description"),
            "vertical_regions": rendered.get("vertical_regions") or [],
            "vertical_overflow_policy": rendered.get(
                "vertical_overflow_policy", "preserve_all"
            ),
            "vertical_edge_priority": rendered.get(
                "vertical_edge_priority", "balanced"
            ),
            "vertical_crop_mode": selected["vertical_crop_mode"],
            "effective_vertical_crop_mode": rendered.get(
                "vertical_crop_mode", selected["vertical_crop_mode"]
            ),
            "subject_clipping_allowed": rendered.get("subject_clipping_allowed", False),
            "grounding_debug": rendered.get("grounding_debug"),
            "track_audit": track_audit,
            "automated_checks": automated_checks,
            "risk_codes": risk_codes,
            "requires_gemini_review": bool(
                rendered.get("requires_gemini_review") or risk_codes
            ),
            "decision_explanation": explanation,
            "human_review": {
                "status": "pending",
                "preferred_action": None,
                "horizontal_bias": None,
                "replacement_candidate_id": None,
                "notes": "",
            },
        }
        audits.append(audit)
        timeline_cursor_ms += duration_ms
        debug_paths = list(rendered.get("grounding_debugs") or [])
        if not debug_paths and rendered.get("grounding_debug"):
            debug_paths = [rendered["grounding_debug"]]
        debug_html = "".join(
            f"<a href='{html.escape(str(debug))}'><img src='{html.escape(str(debug))}'></a>"
            for debug in debug_paths
        ) or "none"
        alternatives_html = "<br>".join(
            f"{html.escape(item['candidate_id'])}: {html.escape(item['observed_visual_evidence'])}"
            for item in alternatives
        ) or (
            "none"
            if alternatives_preserved
            else "未保存：舊版 single-select plan 沒有 Top-K"
        )
        rows.append(
            "<tr>"
            f"<td>{timeline_cursor_ms / 1000 - duration_ms / 1000:.3f}–{timeline_cursor_ms / 1000:.3f}s</td>"
            f"<td>{html.escape(feature_id)}<br>{html.escape(shot['title'])}</td>"
            f"<td>{html.escape(str(rendered.get('target_description') or 'none'))}</td>"
            f"<td>{html.escape(str(rendered.get('applied_strategy')))}<br>"
            f"mode={html.escape(str(rendered.get('vertical_crop_mode', selected['vertical_crop_mode'])))}<br>"
            f"fallback={html.escape(str(rendered.get('fallback_reason') or 'none'))}</td>"
            f"<td>{html.escape(explanation)}</td>"
            f"<td>{alternatives_html}</td>"
            f"<td>{debug_html}</td>"
            "</tr>"
        )

    payload = {
        "method": "vertical_crop_audit_v3_manifest_bound_2d_geometry",
        "interpretation": "renderer_decision_and_geometry_not_human_approval",
        "project_id": plan["project_id"],
        "video": verified_video["path"] if verified_video else None,
        "video_validation": verified_video,
        "budget_plan": str(args.budget_plan.resolve()) if args.budget_plan else None,
        "total_duration_ms": timeline_cursor_ms,
        "chapters": audits,
    }
    write_json(args.output_dir / "vertical-crop-audit.json", payload)
    video_html = (
        f"<video controls src='{html.escape(str(verified_video['path']))}'></video>"
        if verified_video
        else ""
    )
    (args.output_dir / "vertical-crop-audit.html").write_text(
        """<!doctype html><html lang='zh-Hant'><meta charset='utf-8'><title>9:16 crop audit</title>
<style>body{font:14px system-ui;background:#101214;color:#eee;margin:20px}video{width:min(420px,100%);max-height:75vh;background:#000}table{border-collapse:collapse;width:100%;margin-top:18px}th,td{border:1px solid #3b424a;padding:8px;vertical-align:top;text-align:left}img{width:220px}small{color:#9ca3af}</style>
<h1>9:16 自動裁切稽核</h1><p>本頁保存模型候選、實際 target、SAM crop 決策與替代拍帶；所有 human_review 初始為 pending。</p>"""
        + video_html
        + "<table><thead><tr><th>timeline</th><th>shot</th><th>target</th><th>mode</th><th>reason</th><th>alternatives</th><th>grounding</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></html>",
        encoding="utf-8",
    )
    print((args.output_dir / "vertical-crop-audit.html").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
