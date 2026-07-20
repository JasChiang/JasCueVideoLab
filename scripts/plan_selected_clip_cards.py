#!/usr/bin/env python3
"""Ask Gemini to turn selected Full Clip Cards into an auditable narrative plan."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import uuid
from pathlib import Path
from typing import Literal

from google import genai
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jascue_video_lab.billing import summarize_usage_files
from jascue_video_lab.gemini import MODEL_ID, _raw_dump
from jascue_video_lab.models import FeatureEditBrief, FeatureEditPlan, FullClipCard, RushesCatalog
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import read_json, utc_now, write_json


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NarrativeChapter(StrictModel):
    order: int = Field(ge=1)
    feature_id: str
    source_asset_id: str
    event_id: str
    source_in_mmss: str = Field(pattern=r"^[0-9]{2,}:[0-5][0-9]$")
    source_out_mmss: str = Field(pattern=r"^[0-9]{2,}:[0-5][0-9]$")
    narrative_role: Literal[
        "hook", "identity", "design", "ai_feature", "camera", "lifestyle", "ecosystem", "closing"
    ]
    observed_evidence: str
    selection_reason: str
    voiceover_line: str
    transition_note: str
    quality_risks: list[str]


class ClipCardNarrativePlan(StrictModel):
    project_id: str
    title: str
    strategy_summary: str
    target_duration_seconds: int = Field(ge=60, le=90)
    chapters: list[NarrativeChapter] = Field(min_length=1, max_length=16)
    uncertainties: list[str]
    model_provenance: dict[str, str | None]

    @model_validator(mode="after")
    def validate_order(self) -> "ClipCardNarrativePlan":
        orders = [chapter.order for chapter in self.chapters]
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError("chapter order must be contiguous from 1")
        return self


def mmss_to_seconds(value: str) -> int:
    minutes, seconds = (int(part) for part in value.split(":"))
    return minutes * 60 + seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_json", type=Path)
    parser.add_argument("feature_plan_json", type=Path)
    parser.add_argument("brief_json", type=Path)
    parser.add_argument("prepared_library", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--thinking-level", choices=["minimal", "low", "medium", "high"], default="low")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    catalog = RushesCatalog.model_validate(read_json(args.catalog_json))
    feature_plan = FeatureEditPlan.model_validate(read_json(args.feature_plan_json))
    brief = FeatureEditBrief.model_validate(read_json(args.brief_json))
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    evidence: list[dict[str, object]] = []
    expected: dict[str, tuple[str, FullClipCard]] = {}
    for chapter in feature_plan.chapters:
        frame_id = chapter.horizontal_frame_id or chapter.vertical_frame_id
        if frame_id is None:
            continue
        clip = clips[frames[frame_id].clip_id]
        card_path = (
            args.prepared_library
            / "clips"
            / clip.sha256[:16]
            / "gemini"
            / "clip-card"
            / "clip_card.json"
        )
        card = FullClipCard.model_validate(read_json(card_path))
        expected[chapter.feature_id] = (card.source_asset_id, card)
        evidence.append(
            {
                "feature_id": chapter.feature_id,
                "source_asset_id": card.source_asset_id,
                "clip_card": card.model_dump(mode="json"),
            }
        )

    run_id = f"clip-card-plan-{uuid.uuid4().hex[:8]}"
    provenance = {
        "model_id": MODEL_ID,
        "api": "gemini_interactions",
        "sdk": "google-genai",
        "sdk_version": importlib.metadata.version("google-genai"),
        "run_id": run_id,
        "generated_at": utc_now(),
        "interaction_id": None,
    }
    prompt = f"""
你是資深產品短影音剪輯規劃師。請只根據使用者 brief 與 11 份已驗證 Full Clip Card，規劃一支 60–90 秒、能吸引觀眾了解新機的 16:9 picture edit。

規則：
1. brief 是產品名稱與規格的唯一 claim source；Clip Card 只證明畫面拍到什麼。若 Clip Card OCR／型號與 brief 衝突，列入 uncertainties，不得改寫 brief。
2. 每個 brief feature_id 必須恰好出現一次，但可以重新排序。前 3 個 chapter 應優先呈現最有吸引力的結果畫面，形成 result-first hook。
3. source_asset_id 與 event_id 必須從對應 feature 的 Clip Card 原樣選取。
4. source_in_mmss／source_out_mmss 必須落在所選 event 的 [start_mmss,end_mmss] 內，使用 MM:SS，不輸出毫秒。這仍是 coarse review cut，不是 frame-accurate cut。
5. 片段不要平均分配；完整功能示範可較長，靜態產品特寫應較短。所有片段總長必須在 60–90 秒。
6. voiceover_line 使用自然的繁體中文，不能宣稱 Clip Card 沒有證明且 brief 也沒有提供的事實。
7. 不設計像素字卡位置，不輸出 bbox/crop，不使用 transcript。

project_id 必須原樣回傳：{brief.project_id}
model_provenance 必須先原樣回傳以下資料（interaction_id 為 null）：
{json.dumps(provenance, ensure_ascii=False, indent=2)}

## 使用者 brief
{brief.model_dump_json(indent=2)}

## 對應的 Full Clip Card evidence
{json.dumps(evidence, ensure_ascii=False, indent=2)}
""".strip()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "model": MODEL_ID,
        "store": False,
        "input": [{"type": "text", "text": prompt}],
        "generation_config": {
            "temperature": args.temperature,
            "thinking_level": args.thinking_level,
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": gemini_response_schema(ClipCardNarrativePlan),
        },
    }
    write_json(args.output_dir / "narrative-plan.request.json", request)
    client = genai.Client(api_key=api_key)
    try:
        interaction = client.interactions.create(**request)
    finally:
        client.close()
    write_json(args.output_dir / "narrative-plan.raw_interaction.json", _raw_dump(interaction))
    write_json(args.output_dir / "narrative-plan.raw_output.json", {"output_text": interaction.output_text})
    plan = ClipCardNarrativePlan.model_validate_json(interaction.output_text)

    if plan.project_id != brief.project_id:
        raise ValueError("Gemini changed immutable project_id")
    expected_ids = set(expected)
    actual_ids = [chapter.feature_id for chapter in plan.chapters]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError("narrative plan must use every selected feature exactly once")
    total_seconds = 0
    for chapter in plan.chapters:
        expected_asset, card = expected[chapter.feature_id]
        if chapter.source_asset_id != expected_asset:
            raise ValueError(f"wrong source asset for {chapter.feature_id}")
        event = next((item for item in card.events if item.event_id == chapter.event_id), None)
        if event is None:
            raise ValueError(f"unknown event for {chapter.feature_id}: {chapter.event_id}")
        start = mmss_to_seconds(chapter.source_in_mmss)
        end = mmss_to_seconds(chapter.source_out_mmss)
        if not (mmss_to_seconds(event.start_mmss) <= start < end <= mmss_to_seconds(event.end_mmss)):
            raise ValueError(f"chapter trim falls outside event: {chapter.feature_id}")
        total_seconds += end - start
    if not 60 <= total_seconds <= 90:
        raise ValueError(f"actual planned duration is outside 60–90 seconds: {total_seconds}")
    final = plan.model_copy(
        update={
            "model_provenance": {
                **plan.model_provenance,
                "interaction_id": getattr(interaction, "id", None) or "",
            }
        }
    )
    write_json(args.output_dir / "narrative-plan.schema-validation.json", {"ok": True})
    write_json(args.output_dir / "narrative-plan.json", final)
    pricing = summarize_usage_files(
        [args.output_dir / "narrative-plan.raw_interaction.json"],
        relative_to=args.output_dir,
    )
    write_json(args.output_dir / "pricing.json", pricing)
    print(json.dumps({"duration_seconds": total_seconds, "pricing": pricing}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
