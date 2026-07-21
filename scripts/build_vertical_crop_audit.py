#!/usr/bin/env python3
"""Join editorial candidates, geometry, and render decisions for 9:16 review."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from jascue_video_lab.feature_cut import (
    _track_geometry_fingerprint,
    _usable_track_centers,
    _vertical_crop_geometry,
)
from jascue_video_lab.models import SegmentationTrack
from jascue_video_lab.storage import read_json, write_json


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
    vertical = {item["feature_id"]: item for item in manifest["vertical"]["chapters"]}
    alternatives_preserved = "shots" in plan
    if not alternatives_preserved:
        if not args.brief_json:
            raise ValueError("legacy feature plans require --brief-json")
        brief = read_json(args.brief_json)
        brief_by_id = {item["feature_id"]: item for item in brief["chapters"]}
        legacy_shots: list[dict[str, object]] = []
        for chapter in plan["chapters"]:
            feature_id = chapter["feature_id"]
            brief_chapter = brief_by_id[feature_id]
            legacy_shots.append(
                {
                    "feature_id": feature_id,
                    "title": brief_chapter["title"],
                    "vertical_candidate_id": "legacy_selected",
                    "candidates": [
                        {
                            "candidate_id": "legacy_selected",
                            "source_asset_id": chapter["vertical_source_asset_id"],
                            "event_id": chapter["vertical_event_id"],
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
                        }
                    ],
                }
            )
        plan = {
            "project_id": plan["project_id"],
            "shots": legacy_shots,
        }
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
            {
                "candidate_id": candidate["candidate_id"],
                "source_asset_id": candidate["source_asset_id"],
                "event_id": candidate["event_id"],
                "frame_id": candidate["frame_id"],
                "observed_visual_evidence": candidate["observed_visual_evidence"],
                "selection_reason": candidate["selection_reason"],
                "quality_risks": candidate["quality_risks"],
            }
            for candidate in shot["candidates"]
            if candidate["candidate_id"] != shot["vertical_candidate_id"]
        ]
        duration_ms = int(rendered["source_out_ms"]) - int(rendered["source_in_ms"])
        track_files = sorted(
            args.render_manifest_json.parent.glob(
                f"geometry/{feature_id}/vertical/**/segmentation-track.json"
            )
        )
        track_audit: dict[str, object] | None = None
        matching_tracks: list[tuple[Path, SegmentationTrack]] = []
        for track_file in track_files:
            candidate_track = SegmentationTrack.model_validate(read_json(track_file))
            if _track_geometry_fingerprint(candidate_track) == rendered.get(
                "track_geometry_fingerprint"
            ):
                matching_tracks.append((track_file, candidate_track))
        if len(matching_tracks) == 1:
            track_file, track = matching_tracks[0]
            times, centers, boxes = _usable_track_centers(track)
            if times:
                _, crop_geometry = _vertical_crop_geometry(times, centers, boxes)
                track_audit = {
                    "track_path": str(track_file.resolve()),
                    "target_description": track.target_description,
                    "analysis_fps": track.analysis_fps,
                    "analysis_start_ms": track.analysis_start_ms,
                    "analysis_end_ms": track.analysis_end_ms,
                    "state_counts": {
                        str(key): value for key, value in track.state_counts.items()
                    },
                    **crop_geometry,
                }
        explanation = (
            f"以「{rendered.get('target_description') or '未指定主體'}」作為構圖主體；"
            f"計畫模式為 {selected['vertical_crop_mode']}，實際策略為 "
            f"{rendered.get('applied_strategy')}."
        )
        if rendered.get("fallback_reason"):
            explanation += f" 退回原因：{rendered['fallback_reason']}。"
        elif rendered.get("applied_strategy") == "tracked_crop":
            explanation += " crop x 由 SAM mask 中心經雙向平滑後逐段內插。"
        if rendered.get("subject_clipping_allowed"):
            explanation += " 此段允許犧牲次要脈絡，不能解讀為完整保留所有可見人物或物件。"

        audit = {
            "feature_id": feature_id,
            "title": shot["title"],
            "timeline_start_ms": timeline_cursor_ms,
            "timeline_end_ms": timeline_cursor_ms + duration_ms,
            "source_clip_id": rendered["source_clip_id"],
            "source_in_ms": rendered["source_in_ms"],
            "source_out_ms": rendered["source_out_ms"],
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
            "vertical_crop_mode": selected["vertical_crop_mode"],
            "subject_clipping_allowed": rendered.get("subject_clipping_allowed", False),
            "grounding_debug": rendered.get("grounding_debug"),
            "track_audit": track_audit,
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
        debug = rendered.get("grounding_debug")
        debug_html = (
            f"<a href='{html.escape(str(debug))}'><img src='{html.escape(str(debug))}'></a>"
            if debug
            else "none"
        )
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
            f"mode={html.escape(selected['vertical_crop_mode'])}<br>"
            f"fallback={html.escape(str(rendered.get('fallback_reason') or 'none'))}</td>"
            f"<td>{html.escape(explanation)}</td>"
            f"<td>{alternatives_html}</td>"
            f"<td>{debug_html}</td>"
            "</tr>"
        )

    payload = {
        "method": "vertical_crop_audit_v1",
        "interpretation": "renderer_decision_and_geometry_not_human_approval",
        "project_id": plan["project_id"],
        "video": str(args.video.resolve()) if args.video else None,
        "budget_plan": str(args.budget_plan.resolve()) if args.budget_plan else None,
        "total_duration_ms": timeline_cursor_ms,
        "chapters": audits,
    }
    write_json(args.output_dir / "vertical-crop-audit.json", payload)
    video_html = (
        f"<video controls src='{html.escape(str(args.video.resolve()))}'></video>"
        if args.video
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
