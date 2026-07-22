#!/usr/bin/env python3
"""Create auditable Trim Intent proposals for selected feature-plan events."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from time import monotonic

from jascue_video_lab.billing import summarize_usage_and_list_price
from jascue_video_lab.models import FeatureEditBrief, RushesCatalog
from jascue_video_lab.storage import read_json, utc_now, write_json
from jascue_video_lab.trim_intent import run_video_trim_intent_event


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan_json", type=Path)
    parser.add_argument("brief_json", type=Path)
    parser.add_argument("catalog_json", type=Path)
    parser.add_argument("prepared_library", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    started = monotonic()
    plan = read_json(args.plan_json)
    brief = FeatureEditBrief.model_validate(read_json(args.brief_json))
    catalog = RushesCatalog.model_validate(read_json(args.catalog_json))
    if plan.get("project_id") != brief.project_id or plan.get("catalog_id") != catalog.catalog_id:
        raise ValueError("feature plan differs from brief or catalog")
    chapters = plan.get("chapters")
    if not isinstance(chapters, list):
        raise ValueError("feature plan chapters must be a list")
    expected = [chapter.feature_id for chapter in brief.chapters]
    actual = [chapter.get("feature_id") for chapter in chapters]
    if actual != expected:
        raise ValueError("feature plan must preserve brief chapters in order")

    clips_by_asset = {f"sha256:{clip.sha256}": clip for clip in catalog.clips}
    frames_by_id = {frame.frame_id: frame for frame in catalog.frames}
    brief_by_id = {chapter.feature_id: chapter for chapter in brief.chapters}
    prompt = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "trim_intent_video_mmss_zh-TW.txt"
    ).read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for selected in chapters:
        feature_id = str(selected["feature_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", feature_id):
            raise ValueError(f"unsafe feature_id: {feature_id}")
        if selected.get("evidence_status") == "not_found":
            results.append({"feature_id": feature_id, "status": "not_found"})
            continue
        horizontal = (
            str(selected["horizontal_source_asset_id"]),
            str(selected["horizontal_event_id"]),
            str(selected["horizontal_frame_id"]),
        )
        vertical = (
            str(selected["vertical_source_asset_id"]),
            str(selected["vertical_event_id"]),
            str(selected["vertical_frame_id"]),
        )
        selections = [("shared", *horizontal)] if horizontal == vertical else [
            ("horizontal", *horizontal),
            ("vertical", *vertical),
        ]
        chapter = brief_by_id[feature_id]
        editorial_intent = (
            f"章節目的：{chapter.title}。"
            + "；".join(chapter.detail_lines)
            + f"。目標片長約 {chapter.target_duration_seconds:g} 秒，但完整且可理解的動作、結果、"
            "可疑 hold 與 reset 證據優先於硬湊秒數；若同一事件有連續可用階段，不得把新階段"
            "自動當成 reset。所有剪點仍需真人審核。"
        )
        for aspect, asset_id, event_id, frame_id in selections:
            clip = clips_by_asset.get(asset_id)
            if clip is None:
                raise ValueError(f"unknown selected source asset: {asset_id}")
            frame = frames_by_id.get(frame_id)
            if frame is None or frame.clip_id != clip.clip_id:
                raise ValueError(f"selected frame does not belong to source asset: {frame_id}")
            run_dir = args.prepared_library / "clips" / clip.sha256[:16]
            output = args.output_dir / feature_id
            if aspect != "shared":
                output = output / aspect
            try:
                result = run_video_trim_intent_event(
                    run_dir,
                    event_id,
                    frame.requested_time_ms,
                    output,
                    prompt_template=prompt,
                    editorial_intent=editorial_intent,
                )
                results.append(
                    {
                        "feature_id": feature_id,
                        "aspect": aspect,
                        "source_asset_id": asset_id,
                        "event_id": event_id,
                        "anchor_frame_id": frame_id,
                        "status": "ok",
                        **result,
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "feature_id": feature_id,
                        "aspect": aspect,
                        "source_asset_id": asset_id,
                        "event_id": event_id,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )

    pricing = summarize_usage_and_list_price(args.output_dir)
    manifest = {
        "project_id": brief.project_id,
        "catalog_id": catalog.catalog_id,
        "status": "ok" if not failures else "partial_failure",
        "mode": "direct_video_mmss_to_source_pts",
        "results": results,
        "failures": failures,
        "pricing": pricing,
        "elapsed_seconds": round(monotonic() - started, 3),
        "generated_at": utc_now(),
    }
    write_json(args.output_dir / "trim-proposals-manifest.json", manifest)
    print((args.output_dir / "trim-proposals-manifest.json").resolve())
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
