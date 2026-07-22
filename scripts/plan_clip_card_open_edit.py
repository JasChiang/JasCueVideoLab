#!/usr/bin/env python3
"""Infer an auditable edit from a complete Clip Card library without a content brief.

The only editorial constraints are duration, aspect-ratio deliverables, and
evidence-only behavior. Gemini must preserve alternatives for every timeline
slot before selecting the horizontal and vertical representatives.
"""

from __future__ import annotations

import argparse
import html
import importlib.metadata
import json
import os
import uuid
from pathlib import Path
from typing import Literal

from google import genai
from pydantic import BaseModel, ConfigDict, Field, model_validator

from jascue_video_lab.billing import summarize_usage_files
from jascue_video_lab.feature_cut import write_external_feature_plan_projection
from jascue_video_lab.gemini import MODEL_ID, _raw_dump
from jascue_video_lab.media import sha256_file
from jascue_video_lab.models import (
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    FramingRegionIntent,
    FullClipCard,
    ModelProvenance,
    RushesCatalog,
)
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import read_json, utc_now, write_json


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerticalOverflowProposal(StrictModel):
    """Non-executable model suggestion for a later human framing decision."""

    proposed_policy: Literal["controlled_clip"]
    proposed_edge_priority: Literal[
        "balanced", "preserve_start", "preserve_end"
    ] = "balanced"
    rationale: str = Field(min_length=1)


class OpenEditCandidate(StrictModel):
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64)
    source_asset_id: str
    event_id: str
    frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str
    selection_reason: str
    quality_risks: list[str]
    horizontal_strategy: Literal["original", "tracked_reframe"]
    horizontal_zoom_intent: Literal["none", "subtle", "detail"]
    horizontal_target_description: str | None
    vertical_strategy: Literal["tracked_crop", "fit_with_background"]
    vertical_target_description: str | None
    vertical_crop_mode: Literal["strict", "primary_center"]
    vertical_regions: list[FramingRegionIntent] = Field(default_factory=list, max_length=4)
    # Model output is never an execution authorization.  A human-reviewed,
    # hash-bound policy sidecar is the only path to controlled_clip.
    vertical_overflow_policy: Literal["preserve_all"] = "preserve_all"
    vertical_edge_priority: Literal["balanced"] = "balanced"
    vertical_overflow_proposal: VerticalOverflowProposal | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_geometry_intent(self) -> "OpenEditCandidate":
        if self.horizontal_strategy == "tracked_reframe":
            if self.horizontal_zoom_intent == "none" or not self.horizontal_target_description:
                raise ValueError("tracked_reframe requires zoom intent and target")
        elif self.horizontal_zoom_intent != "none":
            raise ValueError("original horizontal strategy must use zoom intent none")
        required_regions = [
            region for region in self.vertical_regions if region.role == "required"
        ]
        if self.vertical_strategy == "tracked_crop" and not (
            self.vertical_target_description or required_regions
        ):
            raise ValueError("tracked_crop requires a target description or required region")
        if self.vertical_regions and not required_regions:
            raise ValueError("vertical regions must include at least one required region")
        if self.vertical_regions and self.vertical_strategy != "tracked_crop":
            raise ValueError("vertical regions are crop constraints and require tracked_crop")
        if (
            self.vertical_overflow_proposal is not None
            and self.vertical_strategy != "tracked_crop"
        ):
            raise ValueError("overflow proposals only apply to tracked_crop candidates")
        return self


class OpenEditShot(StrictModel):
    feature_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    title: str
    editorial_role: Literal[
        "hook",
        "setup",
        "action",
        "result",
        "beauty",
        "lifestyle",
        "transition",
        "closing",
    ]
    intended_effect: str
    target_duration_seconds: float = Field(ge=3.0, le=10.0)
    candidates: list[OpenEditCandidate] = Field(min_length=2, max_length=4)
    horizontal_candidate_id: str = Field(
        pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64
    )
    vertical_candidate_id: str = Field(
        pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64
    )

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> "OpenEditShot":
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique within a timeline slot")
        if self.horizontal_candidate_id not in ids or self.vertical_candidate_id not in ids:
            raise ValueError("selected candidate must be present in candidates")
        return self

    def candidate(self, candidate_id: str) -> OpenEditCandidate:
        return next(item for item in self.candidates if item.candidate_id == candidate_id)


class OpenEditPlan(StrictModel):
    project_id: str
    catalog_id: str
    inferred_title: str
    inferred_theme: str
    intended_audience_hypothesis: str
    story_arc: str
    shots: list[OpenEditShot] = Field(min_length=10, max_length=16)
    excluded_patterns: list[str]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_timeline(self) -> "OpenEditPlan":
        ids = [shot.feature_id for shot in self.shots]
        if len(ids) != len(set(ids)):
            raise ValueError("feature IDs must be unique")
        duration = sum(shot.target_duration_seconds for shot in self.shots)
        if not 60.0 <= duration <= 90.0:
            raise ValueError("selected timeline duration must be 60-90 seconds")
        if self.shots[0].editorial_role != "hook":
            raise ValueError("first shot must be a hook")
        if self.shots[-1].editorial_role != "closing":
            raise ValueError("last shot must be a closing")
        for aspect in ("horizontal", "vertical"):
            selected = [
                shot.candidate(
                    shot.horizontal_candidate_id
                    if aspect == "horizontal"
                    else shot.vertical_candidate_id
                ).frame_id
                for shot in self.shots
            ]
            if len(selected) != len(set(selected)):
                raise ValueError(f"duplicate selected frame in {aspect} timeline")
        return self


def compact_card(card: FullClipCard) -> dict[str, object]:
    return {
        "source_asset_id": card.source_asset_id,
        "duration_ms": card.duration_ms,
        "summary": card.summary,
        "content_type": card.content_type,
        "clip_uses": card.clip_uses,
        "portrait_reframe_feasibility": card.portrait_reframe_feasibility,
        "uncertainties": card.uncertainties,
        "entities": [
            {
                "entity_id": entity.entity_id,
                "kind": entity.kind,
                "label": entity.label,
                "distinguishing_features": entity.distinguishing_features,
                "evidence": entity.evidence,
            }
            for entity in card.entities
        ],
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
                "primary_entity_ids": event.primary_entity_ids,
                "required_entity_ids": event.required_entity_ids,
                "optional_entity_ids": event.optional_entity_ids,
                "avoid_overlay_entity_ids": event.avoid_overlay_entity_ids,
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


def mmss(milliseconds: int) -> str:
    total = max(0, milliseconds // 1000)
    return f"{total // 60:02d}:{total % 60:02d}"


def validate_evidence(
    plan: OpenEditPlan,
    *,
    project_id: str,
    catalog: RushesCatalog,
    cards: dict[str, FullClipCard],
) -> None:
    if plan.project_id != project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("model changed immutable project or catalog ID")
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    for shot in plan.shots:
        for candidate in shot.candidates:
            card = cards.get(candidate.source_asset_id)
            if card is None:
                raise ValueError(f"unknown candidate asset: {candidate.source_asset_id}")
            event = next((item for item in card.events if item.event_id == candidate.event_id), None)
            if event is None:
                raise ValueError(
                    f"unknown candidate event: {candidate.source_asset_id}/{candidate.event_id}"
                )
            frame = frames.get(candidate.frame_id)
            if frame is None:
                raise ValueError(f"unknown candidate frame: {candidate.frame_id}")
            clip = clips[frame.clip_id]
            if f"sha256:{clip.sha256}" != candidate.source_asset_id:
                raise ValueError(f"candidate frame belongs to another asset: {candidate.frame_id}")
            frame_time = mmss(frame.requested_time_ms)
            if not event.start_mmss <= frame_time < event.end_mmss:
                raise ValueError(f"candidate frame lies outside event: {candidate.frame_id}")


def project_feature_contracts(
    plan: OpenEditPlan,
) -> tuple[FeatureEditBrief, FeatureEditPlan, dict[str, object]]:
    brief_chapters: list[FeatureChapterBrief] = []
    selected_chapters: list[FeatureChapterSelect] = []
    audit_chapters: list[dict[str, object]] = []
    for shot in plan.shots:
        horizontal = shot.candidate(shot.horizontal_candidate_id)
        vertical = shot.candidate(shot.vertical_candidate_id)
        vertical_target_description = _projected_vertical_target_description(vertical)
        brief_chapters.append(
            FeatureChapterBrief(
                feature_id=shot.feature_id,
                title=shot.title,
                detail_lines=[
                    f"editorial_role={shot.editorial_role}",
                    shot.intended_effect,
                ],
                target_duration_seconds=shot.target_duration_seconds,
                vertical_primary_target_description=(
                    vertical_target_description
                    if vertical.vertical_strategy == "tracked_crop"
                    else None
                ),
                vertical_crop_mode=vertical.vertical_crop_mode,
                vertical_regions=vertical.vertical_regions,
                vertical_overflow_policy="preserve_all",
                vertical_edge_priority="balanced",
            )
        )
        selected_chapters.append(
            FeatureChapterSelect(
                feature_id=shot.feature_id,
                evidence_status="supported",
                horizontal_frame_id=horizontal.frame_id,
                vertical_frame_id=vertical.frame_id,
                observed_visual_evidence=(
                    f"16:9: {horizontal.observed_visual_evidence} "
                    f"9:16: {vertical.observed_visual_evidence}"
                ),
                selection_reason=(
                    f"16:9 {horizontal.selection_reason}; 9:16 {vertical.selection_reason}"
                ),
                horizontal_strategy=horizontal.horizontal_strategy,
                horizontal_zoom_intent=horizontal.horizontal_zoom_intent,
                horizontal_target_description=horizontal.horizontal_target_description,
                vertical_strategy=vertical.vertical_strategy,
                vertical_target_description=vertical_target_description,
                quality_risks=sorted(
                    set(
                        horizontal.quality_risks
                        + vertical.quality_risks
                        + (
                            ["model_proposed_controlled_clip_requires_human_policy"]
                            if vertical.vertical_overflow_proposal is not None
                            else []
                        )
                    )
                ),
                confidence=min(horizontal.confidence, vertical.confidence),
            )
        )
        audit_chapters.append(
            {
                "feature_id": shot.feature_id,
                "evidence_status": "supported",
                "horizontal_source_asset_id": horizontal.source_asset_id,
                "horizontal_event_id": horizontal.event_id,
                "horizontal_frame_id": horizontal.frame_id,
                "vertical_source_asset_id": vertical.source_asset_id,
                "vertical_event_id": vertical.event_id,
                "vertical_frame_id": vertical.frame_id,
                "execution_vertical_overflow_policy": "preserve_all",
                "model_vertical_overflow_proposal": (
                    vertical.vertical_overflow_proposal.model_dump(mode="json")
                    if vertical.vertical_overflow_proposal is not None
                    else None
                ),
            }
        )
    brief = FeatureEditBrief(
        project_id=plan.project_id,
        title=plan.inferred_title,
        target_duration_seconds=sum(shot.target_duration_seconds for shot in plan.shots),
        render_title_overlays=False,
        vertical_fallback_strategy="center_crop",
        chapters=brief_chapters,
    )
    feature_plan = FeatureEditPlan(
        project_id=plan.project_id,
        catalog_id=plan.catalog_id,
        title=plan.inferred_title,
        chapters=selected_chapters,
        uncertainties=plan.uncertainties,
        model_provenance=plan.model_provenance,
    )
    trim_plan: dict[str, object] = {
        "project_id": plan.project_id,
        "catalog_id": plan.catalog_id,
        "title": plan.inferred_title,
        "chapters": audit_chapters,
    }
    return brief, feature_plan, trim_plan


def reproject_external_feature_plan(
    *,
    source_plan: OpenEditPlan,
    catalog: RushesCatalog,
    brief: FeatureEditBrief,
    source_artifacts: dict[str, Path],
) -> tuple[FeatureEditBrief, FeatureEditPlan]:
    """Registered deterministic projector used by provenance validation."""

    del brief, source_artifacts
    if source_plan.catalog_id != catalog.catalog_id:
        raise ValueError("open-edit source plan differs from projection catalog")
    projected_brief, projected_plan, _ = project_feature_contracts(source_plan)
    return projected_brief, projected_plan


def _projected_vertical_target_description(candidate: OpenEditCandidate) -> str | None:
    """Project region-only framing intent into the legacy single-target contract.

    ``FeatureChapterSelect`` still requires a non-empty target description for
    ``tracked_crop``.  Open-edit candidates may instead express the stronger,
    multi-region contract.  Keep those regions intact on the brief and derive a
    deterministic, domain-neutral union description for consumers that still
    read the single-target field.
    """

    if candidate.vertical_target_description:
        return candidate.vertical_target_description
    required_regions = sorted(
        (
            region
            for region in candidate.vertical_regions
            if region.role == "required"
        ),
        key=lambda region: region.region_id,
    )
    if not required_regions:
        return None
    members = "; ".join(
        (
            f"region_id={region.region_id}, kind={region.kind}, "
            f"target={region.target_description}"
        )
        for region in required_regions
    )
    return f"Preserve the union of all required framing regions: {members}"


def render_candidate_board(plan: OpenEditPlan, catalog: RushesCatalog, output: Path) -> None:
    frames = {frame.frame_id: frame for frame in catalog.frames}
    rows: list[str] = []
    for index, shot in enumerate(plan.shots, start=1):
        cells: list[str] = []
        for candidate in shot.candidates:
            frame = frames[candidate.frame_id]
            image_path = Path(frame.image_path).resolve()
            badges = []
            if candidate.candidate_id == shot.horizontal_candidate_id:
                badges.append("16:9")
            if candidate.candidate_id == shot.vertical_candidate_id:
                badges.append("9:16")
            cells.append(
                "<div class='candidate'>"
                f"<img src='{html.escape(str(image_path))}'>"
                f"<h4>{html.escape(candidate.candidate_id)} {' / '.join(badges)}</h4>"
                f"<p>{html.escape(candidate.observed_visual_evidence)}</p>"
                f"<small>{html.escape(candidate.selection_reason)}</small>"
                "</div>"
            )
        rows.append(
            "<section>"
            f"<h2>{index:02d}. {html.escape(shot.title)} · {shot.target_duration_seconds:g}s"
            f" <small>{html.escape(shot.editorial_role)}</small></h2>"
            f"<p>{html.escape(shot.intended_effect)}</p>"
            f"<div class='grid'>{''.join(cells)}</div>"
            "</section>"
        )
    output.write_text(
        """<!doctype html><html lang='zh-Hant'><meta charset='utf-8'>
<title>No-brief candidate board</title><style>
body{font:15px system-ui;background:#101214;color:#eee;max-width:1500px;margin:24px auto;padding:0 20px}
section{background:#1b1f24;padding:18px;margin:18px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.candidate{background:#111;padding:10px;border-radius:8px}.candidate img{width:100%;aspect-ratio:16/9;object-fit:cover}small{color:#9ca3af}
</style>"""
        f"<h1>{html.escape(plan.inferred_title)}</h1>"
        f"<p><strong>模型自行推論主題：</strong>{html.escape(plan.inferred_theme)}</p>"
        f"<p><strong>故事線：</strong>{html.escape(plan.story_arc)}</p>"
        + "".join(rows)
        + "</html>",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_json", type=Path)
    parser.add_argument("prepared_library", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--project-id", default="open-edit-no-brief")
    parser.add_argument(
        "--reuse-raw-output",
        action="store_true",
        help="Revalidate the saved raw response without creating another API request",
    )
    parser.add_argument(
        "--thinking-level",
        choices=["low", "high"],
        default="high",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not args.reuse_raw_output and not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required")
    catalog = RushesCatalog.model_validate(read_json(args.catalog_json))
    cards: dict[str, FullClipCard] = {}
    for clip in catalog.clips:
        card_path = (
            args.prepared_library
            / "clips"
            / clip.sha256[:16]
            / "gemini"
            / "clip-card"
            / "clip_card.json"
        )
        if not card_path.exists():
            continue
        card = FullClipCard.model_validate(read_json(card_path))
        expected_asset = f"sha256:{clip.sha256}"
        if card.source_asset_id != expected_asset:
            raise ValueError(f"Clip Card asset mismatch: {clip.clip_id}")
        cards[expected_asset] = card
    if len(cards) < 2:
        raise ValueError("at least two validated Clip Cards are required")

    frame_map: dict[str, list[dict[str, object]]] = {}
    for frame in catalog.frames:
        clip = next(item for item in catalog.clips if item.clip_id == frame.clip_id)
        asset_id = f"sha256:{clip.sha256}"
        if asset_id in cards:
            frame_map.setdefault(asset_id, []).append(
                {"frame_id": frame.frame_id, "local_mmss": mmss(frame.requested_time_ms)}
            )
    evidence = [
        {
            "clip_card": compact_card(card),
            "available_catalog_frames": frame_map[asset_id],
        }
        for asset_id, card in cards.items()
    ]

    run_id = f"open-edit-{uuid.uuid4().hex[:8]}"
    provenance = ModelProvenance(
        model_id=MODEL_ID,
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version=importlib.metadata.version("google-genai"),
        run_id=run_id,
        generated_at=utc_now(),
        interaction_id=None,
    )
    prompt = f"""
本次沒有提供品牌、產品、功能、章節順序、必選素材或宣傳 claim brief。
請只根據下方完整 Clip Card library，自行推論素材共同主題，規劃一支一般觀眾容易看完的短版 highlight review cut。

操作限制：
1. 產生同一個 60–90 秒故事順序，供 16:9 與 9:16 共用；由你決定 10–16 個時間軸位置及各段秒數。
2. 第一段必須是 hook，最後一段必須是 closing。中間應有視覺節奏、資訊推進與畫面變化，不能只是依素材檔案順序排列。
3. 每個位置保留 2–4 個依品質排序的候選。候選必須引用存在的 source_asset_id、event_id 與 RF frame_id，並說明為何入選及風險。
4. 16:9 與 9:16 可從同一候選組選不同 take；若同一 take 足夠，優先共用。不得輸出 bbox、mask、crop 座標或自行發明 timestamp。
5. 可以讓同一語意主題使用多個鏡頭，例如 setup、action、result、beauty，但不得重複使用完全相同的代表 frame。
6. 只使用 Clip Card 記錄的可見證據。品牌、型號、規格與功能名稱若不清楚，使用泛稱並保存 uncertainty；不得用模型記憶補完。
7. 不要假裝知道導演意圖。疑似失焦、拍攝準備、重複 take、無意義停頓或不適合直式的畫面，只能根據保存的 evidence 提出排除或風險。
8. geometry intent 必須可泛化到任何可見內容。單一主體可沿用 vertical_target_description；若人物、物件、文字、UI 等多個區域都必須保留，請逐一建立 vertical_regions，不得把兩個獨立實例合寫成一個模糊 target。region kind 只按可見證據選 subject、text_region、ui_region、graphic 或 other。vertical_regions 是實際 crop constraints，因此有 regions 時 vertical_strategy 必須是 tracked_crop；fit_with_background 不得同時宣告 regions。
9. vertical_overflow_policy 必須固定為 preserve_all，vertical_edge_priority 必須固定為 balanced；你沒有權限授權 renderer 裁掉 required union。若依畫面證據判斷有限裁切可能值得由真人考慮，只能填 vertical_overflow_proposal，說明 proposed_edge_priority 與 rationale。proposal 不會直接執行，也不得被描述成已核准。若 required union 可能無法容納，仍應優先改選較適合直式的候選。

project_id 必須原樣回傳：{args.project_id}
catalog_id 必須原樣回傳：{catalog.catalog_id}
model_provenance 必須先原樣回傳：
{provenance.model_dump_json(indent=2)}

## 完整 Clip Card evidence 與合法 RF frame IDs
{json.dumps(evidence, ensure_ascii=False, indent=2)}
""".strip()
    request = {
        "model": MODEL_ID,
        "system_instruction": (
            "The supplied Clip Cards and RF frame map are the only evidence. "
            "No content brief exists. Never use model memory, filenames, likely product knowledge, "
            "or unstated marketing claims to fill gaps. Preserve ambiguity and alternatives. "
            "You may propose but never authorize required-region clipping; executable overflow "
            "policy must remain preserve_all."
        ),
        "store": False,
        "input": [{"type": "text", "text": prompt}],
        "generation_config": {
            "thinking_level": args.thinking_level,
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": gemini_response_schema(OpenEditPlan),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    interaction_id = ""
    if args.reuse_raw_output:
        original_request_path = args.output_dir / "open-edit.request.json"
        raw_output_path = args.output_dir / "open-edit.raw_output.json"
        raw_interaction_path = args.output_dir / "open-edit.raw_interaction.json"
        for required_path in (
            original_request_path,
            raw_output_path,
            raw_interaction_path,
        ):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"--reuse-raw-output requires original artifact: {required_path}"
                )
        original_request = read_json(original_request_path)
        raw_interaction = read_json(raw_interaction_path)
        artifact_models = {
            "original_request": str(original_request.get("model") or ""),
            "raw_interaction": str(raw_interaction.get("model") or ""),
        }
        mismatched_models = {
            source: model
            for source, model in artifact_models.items()
            if model != MODEL_ID
        }
        if mismatched_models:
            raise ValueError(
                "--reuse-raw-output model mismatch: "
                f"expected {MODEL_ID!r}, got {mismatched_models}. "
                "Reuse the File API upload cache instead, or explicitly run with the "
                "artifact's original JASCUE_GEMINI_MODEL."
            )
        reprojection_request_path = (
            args.output_dir / "open-edit.reprojection-request.json"
        )
        write_json(reprojection_request_path, request)
        write_json(
            args.output_dir / "open-edit.raw-output-reuse.json",
            {
                "interpretation": (
                    "saved_model_response_revalidated_and_projected_no_new_model_call"
                ),
                "original_request_path": str(original_request_path.resolve()),
                "original_request_sha256": sha256_file(original_request_path),
                "raw_output_path": str(raw_output_path.resolve()),
                "raw_output_sha256": sha256_file(raw_output_path),
                "current_reprojection_request_path": str(
                    reprojection_request_path.resolve()
                ),
                "current_reprojection_request_sha256": sha256_file(
                    reprojection_request_path
                ),
                "reused_at": utc_now(),
            },
        )
        raw_output = read_json(raw_output_path)
        output_text = str(raw_output["output_text"])
        interaction_id = str(raw_interaction.get("id") or "")
    else:
        write_json(args.output_dir / "open-edit.request.json", request)
        client = genai.Client(api_key=api_key)
        try:
            interaction = client.interactions.create(**request)
        finally:
            client.close()
        raw_interaction = _raw_dump(interaction)
        output_text = interaction.output_text
        interaction_id = getattr(interaction, "id", None) or ""
        write_json(args.output_dir / "open-edit.raw_interaction.json", raw_interaction)
        write_json(
            args.output_dir / "open-edit.raw_output.json",
            {"output_text": output_text},
        )
    plan = OpenEditPlan.model_validate_json(output_text)
    validate_evidence(plan, project_id=args.project_id, catalog=catalog, cards=cards)
    plan = plan.model_copy(
        update={
            "model_provenance": plan.model_provenance.model_copy(
                update={"interaction_id": interaction_id}
            )
        }
    )
    brief, feature_plan, trim_plan = project_feature_contracts(plan)
    write_json(args.output_dir / "open-edit-plan.json", plan)
    write_json(args.output_dir / "brief.json", brief)
    plan_dir = args.output_dir / "gemini-plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    write_json(plan_dir / "feature_edit_plan.json", feature_plan)
    write_json(plan_dir / "open-edit-trim-plan.json", trim_plan)
    write_json(
        args.output_dir / "open-edit.schema-validation.json",
        {
            "ok": True,
            "clip_card_count": len(cards),
            "catalog_frame_count": len(catalog.frames),
            "candidate_count": sum(len(shot.candidates) for shot in plan.shots),
            "selected_shot_count": len(plan.shots),
            "target_duration_seconds": brief.target_duration_seconds,
        },
    )
    projection_artifacts = {
        "source_raw_interaction": args.output_dir / "open-edit.raw_interaction.json",
        "source_raw_output": args.output_dir / "open-edit.raw_output.json",
    }
    for role, path in (
        (
            "original_request",
            args.output_dir / "open-edit.request.json",
        ),
        (
            "raw_output_reuse_record",
            args.output_dir / "open-edit.raw-output-reuse.json",
        ),
        (
            "current_reprojection_request",
            args.output_dir / "open-edit.reprojection-request.json",
        ),
    ):
        if args.reuse_raw_output and path.exists():
            projection_artifacts[role] = path
    projection_request_path = (
        args.output_dir / "open-edit.reprojection-request.json"
        if args.reuse_raw_output
        else args.output_dir / "open-edit.request.json"
    )
    write_external_feature_plan_projection(
        plan_dir=plan_dir,
        projection_contract_id="clip-card-open-edit-v1",
        catalog_path=args.catalog_json,
        brief_path=args.output_dir / "brief.json",
        feature_plan_path=plan_dir / "feature_edit_plan.json",
        source_plan_path=args.output_dir / "open-edit-plan.json",
        source_request_path=projection_request_path,
        source_artifacts=projection_artifacts,
    )
    write_json(
        args.output_dir / "pricing.json",
        summarize_usage_files(
            [args.output_dir / "open-edit.raw_interaction.json"],
            relative_to=args.output_dir,
        ),
    )
    render_candidate_board(plan, catalog, args.output_dir / "candidate-board.html")
    print((args.output_dir / "open-edit-plan.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
