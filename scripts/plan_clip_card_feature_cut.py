#!/usr/bin/env python3
"""Plan an auditable feature cut from a complete Clip Card library.

The model may only select immutable catalog frame IDs backed by a validated
Clip Card event. Local validation projects the richer audit plan into the
FeatureEditPlan consumed by the existing Grounding and tracking renderer.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import uuid
from pathlib import Path
from typing import Literal

from google import genai
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from jascue_video_lab.billing import summarize_usage_files
from jascue_video_lab.feature_cut import write_external_feature_plan_projection
from jascue_video_lab.gemini import MODEL_ID, _raw_dump
from jascue_video_lab.models import (
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    FullClipCard,
    ModelProvenance,
    RushesCatalog,
)
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import read_json, utc_now, write_json


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClipCardFeatureSelect(StrictModel):
    feature_id: str
    evidence_status: Literal["supported", "partial", "not_found"]
    horizontal_source_asset_id: str | None = None
    horizontal_event_id: str | None = None
    horizontal_frame_id: str | None = Field(default=None, pattern=r"^RF[0-9]{6}$")
    vertical_source_asset_id: str | None = None
    vertical_event_id: str | None = None
    vertical_frame_id: str | None = Field(default=None, pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str
    selection_reason: str
    horizontal_strategy: Literal["original", "tracked_reframe"]
    horizontal_zoom_intent: Literal["none", "subtle", "detail"]
    horizontal_target_description: str | None
    vertical_strategy: Literal["tracked_crop", "fit_with_background"]
    vertical_target_description: str | None
    quality_risks: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_evidence_fields(self) -> "ClipCardFeatureSelect":
        ids = (
            self.horizontal_source_asset_id,
            self.horizontal_event_id,
            self.horizontal_frame_id,
            self.vertical_source_asset_id,
            self.vertical_event_id,
            self.vertical_frame_id,
        )
        if self.evidence_status == "not_found":
            if any(value is not None for value in ids):
                raise ValueError("not_found chapters cannot reference source evidence")
        elif any(value is None for value in ids):
            raise ValueError("supported/partial chapters require both source/event/frame triples")
        if self.horizontal_strategy == "tracked_reframe":
            if self.horizontal_zoom_intent == "none" or not self.horizontal_target_description:
                raise ValueError("tracked_reframe requires zoom intent and target")
        elif self.horizontal_zoom_intent != "none":
            raise ValueError("original horizontal strategy must use zoom intent none")
        if self.vertical_strategy == "tracked_crop" and not self.vertical_target_description:
            raise ValueError("tracked_crop requires a target")
        return self


class ClipCardFeaturePlan(StrictModel):
    project_id: str
    catalog_id: str
    title: str
    strategy_summary: str
    chapters: list[ClipCardFeatureSelect]
    uncertainties: list[str]
    model_provenance: ModelProvenance


def mmss(milliseconds: int) -> str:
    total = max(0, milliseconds // 1000)
    return f"{total // 60:02d}:{total % 60:02d}"


def compact_card(card: FullClipCard) -> dict[str, object]:
    return {
        "source_asset_id": card.source_asset_id,
        "duration_ms": card.duration_ms,
        "summary": card.summary,
        "content_type": card.content_type,
        "clip_uses": card.clip_uses,
        "portrait_reframe_feasibility": card.portrait_reframe_feasibility,
        "uncertainties": card.uncertainties,
        "events": [
            {
                "event_id": event.event_id,
                "start_mmss": event.start_mmss,
                "end_mmss": event.end_mmss,
                "recommended_keyframe_mmss": event.recommended_keyframe_mmss,
                "label": event.label,
                "description": event.description,
                "observable_evidence": event.observable_evidence,
                "action_completeness": event.action_completeness,
                "editing_uses": event.editing_uses,
                "quality_risks": event.quality_risks,
                "framing_intent": event.framing_intent,
                "grounding_targets": [
                    {
                        "entity_id": target.entity_id,
                        "target_kind": target.target_kind,
                        "target_description": target.target_description,
                        "purpose": target.purpose,
                    }
                    for target in event.grounding_targets
                ],
            }
            for event in card.events
        ],
    }


def validate_plan_contract(
    plan: ClipCardFeaturePlan,
    *,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
    cards: dict[str, FullClipCard],
) -> None:
    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("model changed immutable project or catalog ID")
    expected_features = [chapter.feature_id for chapter in brief.chapters]
    if [chapter.feature_id for chapter in plan.chapters] != expected_features:
        raise ValueError("plan must preserve every brief chapter exactly once and in order")
    brief_by_id = {chapter.feature_id: chapter for chapter in brief.chapters}
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    for chapter in plan.chapters:
        brief_chapter = brief_by_id[chapter.feature_id]
        if chapter.evidence_status == "not_found":
            continue
        if (
            brief_chapter.vertical_primary_target_description
            and chapter.vertical_strategy != "tracked_crop"
        ):
            raise ValueError(
                f"{chapter.feature_id} has an explicit vertical primary target and must use "
                "tracked_crop; fit_with_background is not allowed"
            )
        triples = (
            (
                chapter.horizontal_source_asset_id,
                chapter.horizontal_event_id,
                chapter.horizontal_frame_id,
            ),
            (
                chapter.vertical_source_asset_id,
                chapter.vertical_event_id,
                chapter.vertical_frame_id,
            ),
        )
        for asset_id, event_id, frame_id in triples:
            assert asset_id is not None and event_id is not None and frame_id is not None
            card = cards.get(asset_id)
            if card is None:
                raise ValueError(f"unknown selected asset: {asset_id}")
            event = next((item for item in card.events if item.event_id == event_id), None)
            if event is None:
                raise ValueError(f"unknown selected event: {asset_id}/{event_id}")
            frame = frames.get(frame_id)
            if frame is None:
                raise ValueError(f"unknown selected frame: {frame_id}")
            selected_clip = clips[frame.clip_id]
            if f"sha256:{selected_clip.sha256}" != asset_id:
                raise ValueError(f"frame does not belong to selected asset: {frame_id}")
            frame_mmss = mmss(frame.requested_time_ms)
            if not event.start_mmss <= frame_mmss < event.end_mmss:
                raise ValueError(f"frame lies outside selected event: {frame_id}")


def project_feature_contracts(
    plan: ClipCardFeaturePlan,
    *,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
) -> FeatureEditPlan:
    """Deterministically project the richer Clip Card plan for the renderer."""

    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("source plan differs from projection catalog/brief")
    projected = [
        FeatureChapterSelect(
            feature_id=chapter.feature_id,
            evidence_status=chapter.evidence_status,
            horizontal_frame_id=chapter.horizontal_frame_id,
            vertical_frame_id=chapter.vertical_frame_id,
            observed_visual_evidence=chapter.observed_visual_evidence,
            selection_reason=chapter.selection_reason,
            horizontal_strategy=chapter.horizontal_strategy,
            horizontal_zoom_intent=chapter.horizontal_zoom_intent,
            horizontal_target_description=chapter.horizontal_target_description,
            vertical_strategy=chapter.vertical_strategy,
            vertical_target_description=chapter.vertical_target_description,
            quality_risks=chapter.quality_risks,
            confidence=chapter.confidence,
        )
        for chapter in plan.chapters
    ]
    return FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id=catalog.catalog_id,
        title=plan.title,
        chapters=projected,
        uncertainties=plan.uncertainties,
        model_provenance=plan.model_provenance,
    )


def reproject_external_feature_plan(
    *,
    source_plan: ClipCardFeaturePlan,
    catalog: RushesCatalog,
    brief: FeatureEditBrief,
    source_artifacts: dict[str, Path],
) -> tuple[FeatureEditBrief, FeatureEditPlan]:
    """Registered deterministic projector used by provenance validation."""

    del source_artifacts
    return brief, project_feature_contracts(source_plan, brief=brief, catalog=catalog)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_json", type=Path)
    parser.add_argument("brief_json", type=Path)
    parser.add_argument("prepared_library", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument(
        "--thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default="high",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    catalog = RushesCatalog.model_validate(read_json(args.catalog_json))
    brief = FeatureEditBrief.model_validate(read_json(args.brief_json))
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    asset_to_clip = {f"sha256:{clip.sha256}": clip for clip in catalog.clips}

    cards: dict[str, FullClipCard] = {}
    for clip in catalog.clips:
        path = (
            args.prepared_library
            / "clips"
            / clip.sha256[:16]
            / "gemini"
            / "clip-card"
            / "clip_card.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"Clip Card missing for {clip.clip_id}: {path}")
        card = FullClipCard.model_validate(read_json(path))
        expected_asset = f"sha256:{clip.sha256}"
        if card.source_asset_id != expected_asset:
            raise ValueError(f"Clip Card asset mismatch for {clip.clip_id}")
        cards[expected_asset] = card

    frame_map: dict[str, list[dict[str, object]]] = {}
    for frame in catalog.frames:
        clip = clips[frame.clip_id]
        frame_map.setdefault(f"sha256:{clip.sha256}", []).append(
            {
                "frame_id": frame.frame_id,
                "local_mmss": mmss(frame.requested_time_ms),
            }
        )

    run_id = f"clip-card-feature-plan-{uuid.uuid4().hex[:8]}"
    provenance = ModelProvenance(
        model_id=MODEL_ID,
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version=importlib.metadata.version("google-genai"),
        run_id=run_id,
        generated_at=utc_now(),
        interaction_id=None,
    )
    evidence = [
        {
            "clip_id": asset_to_clip[asset_id].clip_id,
            "clip_card": compact_card(card),
            "available_catalog_frames": frame_map[asset_id],
        }
        for asset_id, card in cards.items()
    ]
    prompt = f"""
你是 evidence-bound 的資深短影音挑帶剪輯師。請使用完整 Clip Card library，為使用者 brief 的每個 chapter 選擇橫式與直式 take。你只能引用輸入列出的 source_asset_id、event_id 與 RF frame_id。

規則：
1. brief 是允許使用的產品 claim，不是畫面證據；observed_visual_evidence 只能寫 Clip Card 直接支持的內容。
2. 每個 brief feature_id 必須依原順序恰好回傳一次。優先完整動作、清楚結果、低遮擋、低反光與不同 take；不要因為檔名或常識補完。
3. selected frame 的 local_mmss 必須位於所引用 event 的 [start_mmss,end_mmss)；不得自行創造 frame ID 或 timestamp。
4. 若可見型號、文字、數字或物件身分與 brief 衝突，優先改選沒有衝突的 take；沒有可靠 take 時用 partial 或 not_found 並保存風險。
5. 橫式與直式可以選不同來源。9:16 優先遵守 brief 的 vertical_primary_target_description；tracked target 必須是單一、可持續辨識的實例或局部。
6. bbox、mask、crop 座標與精確 cut point 均由後續 Grounding／tracker／FFmpeg 處理；本階段不得輸出座標。
7. confidence 是 proposal，不是人工真值。

project_id 必須原樣回傳：{brief.project_id}
catalog_id 必須原樣回傳：{catalog.catalog_id}
model_provenance 必須先原樣回傳：
{provenance.model_dump_json(indent=2)}

## 使用者 brief
{brief.model_dump_json(indent=2)}

## 完整 Clip Card evidence 與可選 RF frame IDs
{json.dumps(evidence, ensure_ascii=False, indent=2)}
""".strip()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    request = {
        "model": MODEL_ID,
        "system_instruction": (
            "Provided Clip Cards and RF frame maps are the only evidence. "
            "Never replace visible evidence with model memory or likely product knowledge."
        ),
        "store": False,
        "input": [{"type": "text", "text": prompt}],
        "generation_config": {
            "temperature": args.temperature,
            "thinking_level": args.thinking_level,
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": gemini_response_schema(ClipCardFeaturePlan),
        },
    }
    client = genai.Client(api_key=api_key)
    try:
        interaction = None
        plan = None
        previous_output = ""
        previous_error = ""
        for attempt in range(1, args.repair_attempts + 2):
            attempt_request = request
            if attempt > 1:
                repair_prompt = (
                    prompt
                    + "\n\n## 前次輸出未通過本機 contract\n"
                    + previous_error
                    + "\n請重新產生完整結果，不得只回傳修補片段。前次輸出如下：\n"
                    + previous_output
                )
                attempt_request = {
                    **request,
                    "input": [{"type": "text", "text": repair_prompt}],
                }
            write_json(
                args.output_dir / f"clip-card-feature-plan.attempt-{attempt:02d}.request.json",
                attempt_request,
            )
            current = client.interactions.create(**attempt_request)
            raw = _raw_dump(current)
            write_json(
                args.output_dir
                / f"clip-card-feature-plan.attempt-{attempt:02d}.raw_interaction.json",
                raw,
            )
            write_json(
                args.output_dir / f"clip-card-feature-plan.attempt-{attempt:02d}.raw_output.json",
                {"output_text": current.output_text},
            )
            try:
                plan = ClipCardFeaturePlan.model_validate_json(current.output_text)
                validate_plan_contract(
                    plan,
                    brief=brief,
                    catalog=catalog,
                    cards=cards,
                )
                interaction = current
                write_json(args.output_dir / "clip-card-feature-plan.request.json", attempt_request)
                write_json(args.output_dir / "clip-card-feature-plan.raw_interaction.json", raw)
                write_json(
                    args.output_dir / "clip-card-feature-plan.raw_output.json",
                    {"output_text": current.output_text},
                )
                break
            except (ValidationError, ValueError) as error:
                previous_output = current.output_text
                previous_error = str(error)
                write_json(
                    args.output_dir
                    / f"clip-card-feature-plan.attempt-{attempt:02d}.schema-validation.json",
                    {"ok": False, "error_type": type(error).__name__, "error": str(error)},
                )
        if interaction is None or plan is None:
            raise ValueError(
                f"Clip Card feature plan failed after {args.repair_attempts + 1} attempts: "
                f"{previous_error}"
            )
    finally:
        client.close()
    final_audit = plan.model_copy(
        update={
            "model_provenance": plan.model_provenance.model_copy(
                update={"interaction_id": getattr(interaction, "id", None) or ""}
            )
        }
    )
    final_plan = project_feature_contracts(
        final_audit,
        brief=brief,
        catalog=catalog,
    )
    write_json(args.output_dir / "clip-card-feature-plan.json", final_audit)
    write_json(args.output_dir / "feature_edit_plan.json", final_plan)
    write_json(
        args.output_dir / "clip-card-feature-plan.schema-validation.json",
        {"ok": True, "clip_card_count": len(cards), "frame_count": len(frames)},
    )
    write_external_feature_plan_projection(
        plan_dir=args.output_dir,
        projection_contract_id="clip-card-feature-cut-v1",
        catalog_path=args.catalog_json,
        brief_path=args.brief_json,
        feature_plan_path=args.output_dir / "feature_edit_plan.json",
        source_plan_path=args.output_dir / "clip-card-feature-plan.json",
        source_request_path=args.output_dir / "clip-card-feature-plan.request.json",
        source_artifacts={
            "source_raw_interaction": (
                args.output_dir / "clip-card-feature-plan.raw_interaction.json"
            ),
            "source_raw_output": (
                args.output_dir / "clip-card-feature-plan.raw_output.json"
            ),
        },
    )
    pricing = summarize_usage_files(
        sorted(args.output_dir.glob("clip-card-feature-plan.attempt-*.raw_interaction.json")),
        relative_to=args.output_dir,
    )
    write_json(args.output_dir / "pricing.json", pricing)
    print(
        json.dumps(
            {
                "clip_card_count": len(cards),
                "chapter_count": len(final_plan.chapters),
                "pricing": pricing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
