#!/usr/bin/env python3
"""Reconcile an open-ended edit against a global duration budget.

This pass may keep, drop, or reorder already-rendered complete segments. It
does not silently shorten a locally reviewed action or create new geometry.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Literal

from google import genai
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jascue_video_lab.billing import summarize_usage_files
from jascue_video_lab.gemini import MODEL_ID, _raw_dump
from jascue_video_lab.models import ModelProvenance
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import read_json, utc_now, write_json


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SegmentDecision(StrictModel):
    feature_id: str
    action: Literal["keep", "drop"]
    reason: str


class BudgetPlan(StrictModel):
    project_id: str
    target_min_seconds: float
    target_max_seconds: float
    sequence: list[str] = Field(min_length=1)
    decisions: list[SegmentDecision]
    strategy_summary: str
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_ids(self) -> "BudgetPlan":
        if len(self.sequence) != len(set(self.sequence)):
            raise ValueError("sequence IDs must be unique")
        decision_ids = [item.feature_id for item in self.decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("decision IDs must be unique")
        kept = {item.feature_id for item in self.decisions if item.action == "keep"}
        if set(self.sequence) != kept:
            raise ValueError("sequence must contain exactly the kept decisions")
        return self


def concat_segments(paths: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_path = output.with_suffix(".concat.txt")
    concat_path.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in paths), encoding="utf-8"
    )
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
            str(concat_path),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("open_plan_json", type=Path)
    parser.add_argument("render_manifest_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-min", type=float, default=60.0)
    parser.add_argument("--target-max", type=float, default=90.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.target_min <= args.target_max:
        raise ValueError("invalid duration range")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    open_plan = read_json(args.open_plan_json)
    manifest = read_json(args.render_manifest_json)
    horizontal = {item["feature_id"]: item for item in manifest["horizontal"]["chapters"]}
    vertical = {item["feature_id"]: item for item in manifest["vertical"]["chapters"]}
    shots_by_id = {item["feature_id"]: item for item in open_plan["shots"]}
    if set(horizontal) != set(vertical) or set(horizontal) != set(shots_by_id):
        raise ValueError("open plan and rendered timelines differ")

    segments: list[dict[str, object]] = []
    for shot in open_plan["shots"]:
        feature_id = shot["feature_id"]
        horizontal_item = horizontal[feature_id]
        vertical_item = vertical[feature_id]
        duration_ms = horizontal_item.get("duration_ms")
        if not isinstance(duration_ms, int) or duration_ms <= 0:
            duration_ms = int(horizontal_item["source_out_ms"]) - int(
                horizontal_item["source_in_ms"]
            )
        vertical_duration_ms = vertical_item.get("duration_ms")
        if not isinstance(vertical_duration_ms, int) or vertical_duration_ms <= 0:
            vertical_duration_ms = int(vertical_item["source_out_ms"]) - int(
                vertical_item["source_in_ms"]
            )
        if duration_ms != vertical_duration_ms:
            raise ValueError(f"aspect durations differ for {feature_id}")
        segments.append(
            {
                "feature_id": feature_id,
                "title": shot["title"],
                "editorial_role": shot["editorial_role"],
                "intended_effect": shot["intended_effect"],
                "actual_duration_seconds": round(duration_ms / 1000, 3),
                "selected_visual_evidence": next(
                    item
                    for item in shot["candidates"]
                    if item["candidate_id"] == shot["horizontal_candidate_id"]
                )["observed_visual_evidence"],
                "quality_risks": sorted(
                    {
                        risk
                        for candidate in shot["candidates"]
                        for risk in candidate["quality_risks"]
                    }
                ),
            }
        )

    provenance = ModelProvenance(
        model_id=MODEL_ID,
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version=importlib.metadata.version("google-genai"),
        run_id=f"budget-{uuid.uuid4().hex[:8]}",
        generated_at=utc_now(),
        interaction_id=None,
    )
    prompt = f"""
你是短影音的全片時長協調器。這是一支沒有使用者內容 brief 的 open edit；前一階段已根據完整 Clip Card library 選片，並讓 Gemini 重看入選影片，產生不可在本階段任意截短的完整片段。

請只做三件事：保留、刪除、重新排序完整片段。不得改寫內容證據、不得創造新片段、不得要求在動作中間剪斷。

規則：
1. 最終所有保留片段的實際秒數總和必須在 {args.target_min:g}–{args.target_max:g} 秒。
2. 必須保留一個 hook 與一個 closing，但可以重新排序其他片段以改善節奏。
3. 優先保留視覺差異、動作結果、人物／產品交替與可理解的資訊推進；刪除重複、拖長或對整體故事貢獻較低的段落。
4. decisions 必須恰好涵蓋所有輸入 feature_id；sequence 只能列 action=keep 的 ID，並表示最終播放順序。
5. 不得使用模型記憶或品牌知識；只能依下方保存的可見證據與實際片長判斷。

project_id 必須原樣回傳：{open_plan['project_id']}
target_min_seconds 必須原樣回傳：{args.target_min}
target_max_seconds 必須原樣回傳：{args.target_max}
model_provenance 必須原樣回傳：
{provenance.model_dump_json(indent=2)}

## 原始 open-edit 故事線
{json.dumps({'theme': open_plan['inferred_theme'], 'story_arc': open_plan['story_arc']}, ensure_ascii=False, indent=2)}

## 已渲染完整片段
{json.dumps(segments, ensure_ascii=False, indent=2)}
""".strip()
    request = {
        "model": MODEL_ID,
        "system_instruction": (
            "Use only the supplied segment evidence and exact durations. "
            "You may keep, drop, or reorder whole segments; never invent content."
        ),
        "store": False,
        "input": [{"type": "text", "text": prompt}],
        "generation_config": {"temperature": args.temperature, "thinking_level": "high"},
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": gemini_response_schema(BudgetPlan),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "budget.request.json", request)
    client = genai.Client(api_key=api_key)
    try:
        interaction = client.interactions.create(**request)
    finally:
        client.close()
    write_json(args.output_dir / "budget.raw_interaction.json", _raw_dump(interaction))
    write_json(
        args.output_dir / "budget.raw_output.json", {"output_text": interaction.output_text}
    )
    budget = BudgetPlan.model_validate_json(interaction.output_text)
    if budget.project_id != open_plan["project_id"]:
        raise ValueError("model changed project ID")
    if (
        budget.target_min_seconds != args.target_min
        or budget.target_max_seconds != args.target_max
    ):
        raise ValueError("model changed duration contract")
    expected_ids = set(shots_by_id)
    decision_ids = {item.feature_id for item in budget.decisions}
    if decision_ids != expected_ids:
        raise ValueError("budget decisions do not cover the full rendered timeline")
    duration_by_id = {
        item["feature_id"]: float(item["actual_duration_seconds"]) for item in segments
    }
    total_seconds = sum(duration_by_id[feature_id] for feature_id in budget.sequence)
    if not args.target_min <= total_seconds <= args.target_max:
        raise ValueError(f"budget sequence duration is out of range: {total_seconds:.3f}s")
    roles = {shot["feature_id"]: shot["editorial_role"] for shot in open_plan["shots"]}
    if not any(roles[item] == "hook" for item in budget.sequence):
        raise ValueError("budget sequence dropped every hook")
    if not any(roles[item] == "closing" for item in budget.sequence):
        raise ValueError("budget sequence dropped every closing")

    budget = budget.model_copy(
        update={
            "model_provenance": budget.model_provenance.model_copy(
                update={"interaction_id": getattr(interaction, "id", None) or ""}
            )
        }
    )
    write_json(args.output_dir / "budget-plan.json", budget)
    write_json(
        args.output_dir / "budget-validation.json",
        {
            "ok": True,
            "duration_seconds": round(total_seconds, 3),
            "kept_count": len(budget.sequence),
            "dropped_count": len(expected_ids) - len(budget.sequence),
        },
    )
    write_json(
        args.output_dir / "pricing.json",
        summarize_usage_files(
            [args.output_dir / "budget.raw_interaction.json"],
            relative_to=args.output_dir,
        ),
    )
    for aspect, items in (("16x9", horizontal), ("9x16", vertical)):
        paths = [Path(items[feature_id]["segment_path"]) for feature_id in budget.sequence]
        concat_segments(paths, args.output_dir / f"open-edit-budgeted-{aspect}.mp4")
    print((args.output_dir / "budget-plan.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
