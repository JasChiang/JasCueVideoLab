#!/usr/bin/env python3
"""Plan an auditable feature cut from a complete Clip Card library.

The model may only select immutable catalog frame IDs backed by a validated
Clip Card event. Local validation projects the richer audit plan into the
FeatureEditPlan consumed by the existing Grounding and tracking renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from google import genai
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from jascue_video_lab.billing import summarize_usage_files
from jascue_video_lab.feature_cut import write_external_feature_plan_projection
from jascue_video_lab.gemini import MODEL_ID, _raw_dump
from jascue_video_lab.media import sha256_file
from jascue_video_lab.models import (
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    FeatureHorizontalCandidate,
    FeatureVerticalCandidate,
    FramingRegionIntent,
    FullClipCard,
    ModelProvenance,
    RushesCatalog,
)
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import read_json, utc_now, write_json


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Keep these two v1 models byte-for-byte schema compatible with the historical
# source request.  In particular, do not add class docstrings: Pydantic emits
# them as JSON Schema descriptions and provenance validation compares schemas.
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


class ResolvedEntityRef(StrictModel):
    """An auditable link from a planner region back to one Clip Card event."""

    entity_id: str = Field(min_length=1)
    event_relation: Literal[
        "event_member",
        "primary",
        "required",
        "optional",
        "avoid_overlay",
        "grounding_target",
    ]


class ResolvedFramingRegion(StrictModel):
    """Domain-neutral crop evidence resolved to immutable Clip Card entities.

    ``hard_core`` is content that must remain visible, ``soft_extent`` is useful
    context that may be sacrificed, and ``overlay_keepout`` is content that a
    later layout system should avoid covering.  ``atomic`` regions intentionally
    refer to one entity; a ``union`` makes a multi-entity constraint explicit.
    """

    region_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")
    target_description: str = Field(min_length=1)
    kind: Literal["subject", "text_region", "ui_region", "graphic", "other"]
    constraint_role: Literal["hard_core", "soft_extent", "overlay_keepout"]
    composition: Literal["atomic", "union"] = "atomic"
    atomic: bool = False
    entity_refs: list[ResolvedEntityRef] = Field(min_length=1, max_length=4)
    observable_relation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_entity_refs(self) -> "ResolvedFramingRegion":
        ids = [ref.entity_id for ref in self.entity_refs]
        if len(ids) != len(set(ids)):
            raise ValueError("resolved region entity refs must be unique")
        if self.composition == "atomic" and len(ids) != 1:
            raise ValueError("atomic resolved regions must reference exactly one entity")
        if self.composition == "union" and len(ids) < 2:
            raise ValueError("union resolved regions must reference at least two entities")
        if self.atomic and self.constraint_role != "hard_core":
            raise ValueError("atomic crop content must use hard_core constraint role")
        return self


class ClipCardFeatureCandidate(StrictModel):
    """One ranked take whose source and geometry intent remain auditable."""

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64)
    source_asset_id: str
    event_id: str
    frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str = Field(min_length=1, max_length=600)
    selection_reason: str = Field(min_length=1, max_length=500)
    quality_risks: list[str] = Field(max_length=8)
    horizontal_strategy: Literal["original", "tracked_reframe"]
    horizontal_zoom_intent: Literal["none", "subtle", "detail"]
    horizontal_target_description: str | None
    vertical_strategy: Literal["tracked_crop", "fit_with_background"]
    vertical_crop_mode: Literal["strict", "primary_center"] = "strict"
    vertical_target_description: str | None
    resolved_regions: list[ResolvedFramingRegion] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_candidate(self) -> "ClipCardFeatureCandidate":
        if self.horizontal_strategy == "tracked_reframe":
            if self.horizontal_zoom_intent == "none" or not self.horizontal_target_description:
                raise ValueError("tracked_reframe candidate requires zoom intent and target")
        elif self.horizontal_zoom_intent != "none":
            raise ValueError("original candidate must use zoom intent none")
        hard_regions = [
            region
            for region in self.resolved_regions
            if region.constraint_role == "hard_core"
        ]
        if self.vertical_strategy == "tracked_crop" and not (
            self.vertical_target_description or hard_regions
        ):
            raise ValueError("tracked_crop candidate requires a target or hard-core region")
        if self.resolved_regions and self.vertical_strategy != "tracked_crop":
            raise ValueError("resolved crop regions require tracked_crop")
        region_ids = [region.region_id for region in self.resolved_regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("candidate resolved region IDs must be unique")
        return self


class ClipCardFeatureSelectV2(StrictModel):
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
    candidates: list[ClipCardFeatureCandidate] = Field(default_factory=list, max_length=4)
    horizontal_candidate_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]+$"
    )
    vertical_candidate_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]+$"
    )

    @model_validator(mode="after")
    def validate_evidence_fields(self) -> "ClipCardFeatureSelectV2":
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
            selected_vertical = next(
                (
                    candidate
                    for candidate in self.candidates
                    if candidate.candidate_id == self.vertical_candidate_id
                ),
                None,
            )
            selected_hard_regions = (
                [
                    region
                    for region in selected_vertical.resolved_regions
                    if region.constraint_role == "hard_core"
                ]
                if selected_vertical is not None
                else []
            )
            if not selected_hard_regions:
                raise ValueError("tracked_crop requires a target or selected hard-core region")
        if self.candidates:
            candidate_ids = [candidate.candidate_id for candidate in self.candidates]
            if len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError("candidate IDs must be unique within a chapter")
            references = [
                (candidate.source_asset_id, candidate.event_id, candidate.frame_id)
                for candidate in self.candidates
            ]
            if len(references) != len(set(references)):
                raise ValueError("Top-K candidates must reference distinct evidence frames")
            if self.evidence_status == "not_found":
                raise ValueError("not_found chapters cannot preserve candidates")
            if self.horizontal_candidate_id not in candidate_ids:
                raise ValueError("horizontal candidate must be present in candidates")
            if self.vertical_candidate_id not in candidate_ids:
                raise ValueError("vertical candidate must be present in candidates")
            by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
            horizontal = by_id[self.horizontal_candidate_id]
            vertical = by_id[self.vertical_candidate_id]
            if (
                self.horizontal_source_asset_id,
                self.horizontal_event_id,
                self.horizontal_frame_id,
            ) != (horizontal.source_asset_id, horizontal.event_id, horizontal.frame_id):
                raise ValueError("legacy horizontal selection must match selected candidate")
            if (
                self.vertical_source_asset_id,
                self.vertical_event_id,
                self.vertical_frame_id,
            ) != (vertical.source_asset_id, vertical.event_id, vertical.frame_id):
                raise ValueError("legacy vertical selection must match selected candidate")
            if (
                self.horizontal_strategy,
                self.horizontal_zoom_intent,
                self.horizontal_target_description,
            ) != (
                horizontal.horizontal_strategy,
                horizontal.horizontal_zoom_intent,
                horizontal.horizontal_target_description,
            ):
                raise ValueError("legacy horizontal geometry must match selected candidate")
            if (
                self.vertical_strategy,
                self.vertical_target_description,
            ) != (
                vertical.vertical_strategy,
                vertical.vertical_target_description,
            ):
                raise ValueError("legacy vertical geometry must match selected candidate")
        elif self.horizontal_candidate_id is not None or self.vertical_candidate_id is not None:
            raise ValueError("candidate IDs require a candidate list")
        return self


class ClipCardFeaturePlanV2(StrictModel):
    contract_version: Literal["legacy-v1", "clip-card-feature-cut-v2"]
    project_id: str
    catalog_id: str
    title: str
    strategy_summary: str
    chapters: list[ClipCardFeatureSelectV2]
    uncertainties: list[str]
    model_provenance: ModelProvenance

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_contract(cls, value: Any) -> Any:
        # Previously saved planner JSON has no contract_version or alternatives.
        # Keep it readable and deterministically projectable without pretending
        # that a legacy single selection is a genuine Top-K result.
        if isinstance(value, dict) and "contract_version" not in value:
            return {"contract_version": "legacy-v1", **value}
        return value

    @model_validator(mode="after")
    def validate_contract_version(self) -> "ClipCardFeaturePlanV2":
        if self.contract_version == "clip-card-feature-cut-v2":
            for chapter in self.chapters:
                if chapter.evidence_status == "not_found":
                    if chapter.candidates:
                        raise ValueError("v2 not_found chapters cannot contain candidates")
                    continue
                if not 2 <= len(chapter.candidates) <= 4:
                    raise ValueError("v2 chapters must preserve Top-K 2-4 candidates")
        elif any(chapter.candidates for chapter in self.chapters):
            raise ValueError("legacy-v1 plans cannot claim v2 candidate alternatives")
        return self


class ClipCardFeatureCandidateV3(StrictModel):
    """One ranked take; local evidence owns descriptions and crop regions."""

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", min_length=1, max_length=64)
    source_asset_id: str
    event_id: str
    frame_id: str = Field(pattern=r"^RF[0-9]{6}$")
    observed_visual_evidence: str
    selection_reason: str
    quality_risks: list[str]
    horizontal_strategy: Literal["original", "tracked_reframe"]
    horizontal_zoom_intent: Literal["none", "subtle", "detail"]
    horizontal_focus_entity_id: str | None = None
    vertical_strategy: Literal["tracked_crop", "fit_with_background"]
    vertical_crop_mode: Literal["strict", "primary_center"] = "strict"
    framing_intent: str = Field(min_length=1, max_length=300)
    required_entity_ids: list[str] = Field(default_factory=list, max_length=4)
    preferred_entity_ids: list[str] = Field(default_factory=list, max_length=4)
    sacrificable_entity_ids: list[str] = Field(default_factory=list, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_candidate(self) -> "ClipCardFeatureCandidateV3":
        if self.horizontal_strategy == "tracked_reframe":
            if self.horizontal_zoom_intent == "none" or not self.horizontal_focus_entity_id:
                raise ValueError("tracked_reframe requires a focus entity and zoom intent")
        elif self.horizontal_zoom_intent != "none" or self.horizontal_focus_entity_id:
            raise ValueError("original horizontal strategy cannot declare a focus entity or zoom")
        if self.vertical_strategy == "tracked_crop" and not self.required_entity_ids:
            raise ValueError("tracked_crop requires at least one required entity")
        classified = (
            self.required_entity_ids
            + self.preferred_entity_ids
            + self.sacrificable_entity_ids
        )
        if len(classified) != len(set(classified)):
            raise ValueError("vertical semantic entity roles must be disjoint and unique")
        return self


class ClipCardFeatureSelectV3(StrictModel):
    """Selection-only chapter; rank-one mirror fields are projected locally."""

    feature_id: str
    evidence_status: Literal["supported", "partial", "not_found"]
    candidates: list[ClipCardFeatureCandidateV3] = Field(default_factory=list, max_length=4)
    horizontal_candidate_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]+$"
    )
    vertical_candidate_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_-]+$"
    )

    @model_validator(mode="after")
    def validate_selection(self) -> "ClipCardFeatureSelectV3":
        if self.evidence_status == "not_found":
            if self.candidates or self.horizontal_candidate_id or self.vertical_candidate_id:
                raise ValueError("not_found chapters cannot reference candidates")
            return self
        if not 2 <= len(self.candidates) <= 4:
            raise ValueError("v3 chapters must preserve Top-K 2-4 candidates")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate IDs must be unique within a chapter")
        references = [
            (candidate.source_asset_id, candidate.event_id, candidate.frame_id)
            for candidate in self.candidates
        ]
        if len(references) != len(set(references)):
            raise ValueError("Top-K candidates must reference distinct evidence frames")
        if self.horizontal_candidate_id not in candidate_ids:
            raise ValueError("horizontal candidate must be present in candidates")
        if self.vertical_candidate_id not in candidate_ids:
            raise ValueError("vertical candidate must be present in candidates")
        return self


class ClipCardFeaturePlanV3(StrictModel):
    """Cost-bounded model output containing editorial choices, not mirrors."""

    contract_version: Literal["clip-card-feature-cut-v3"]
    project_id: str
    catalog_id: str
    title: str
    strategy_summary: str
    chapters: list[ClipCardFeatureSelectV3]
    uncertainties: list[str]
    model_provenance: ModelProvenance


class SelectedEvidenceEntity(StrictModel):
    entity_id: str
    kind: str
    label: str
    distinguishing_features: str


class SelectedEvidenceGroundingTarget(StrictModel):
    entity_id: str
    target_description: str


class SelectedEvidenceEvent(StrictModel):
    source_asset_id: str
    event_id: str
    entity_ids: list[str]
    primary_entity_ids: list[str]
    required_entity_ids: list[str]
    optional_entity_ids: list[str]
    avoid_overlay_entity_ids: list[str]
    entities: list[SelectedEvidenceEntity]
    grounding_targets: list[SelectedEvidenceGroundingTarget]


class SelectedClipCardEvidence(StrictModel):
    """Hash-bound local evidence required to reproduce a v3 projection."""

    contract_version: Literal["clip-card-feature-cut-selected-evidence-v1"]
    events: list[SelectedEvidenceEvent]


FEATURE_PLAN_NORMALIZATION_VERSION = "clip-card-feature-plan-normalization-v1"


def canonicalize_feature_plan_output(
    output_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Canonicalize only two explicitly ordered schema contradictions.

    The function is deliberately narrow and deterministic.  It never changes
    editorial selections or evidence references.  Explicit
    ``horizontal_strategy=original`` has conservative precedence: local
    normalization disables contradictory zoom and tracking focus rather than
    promoting a non-tracking choice into executable tracking.
    """

    payload = json.loads(output_text)
    if not isinstance(payload, dict):
        raise ValueError("feature planner output must be a JSON object")
    changes: list[dict[str, Any]] = []
    chapters = payload.get("chapters")
    if not isinstance(chapters, list):
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), changes
    for chapter_index, chapter in enumerate(chapters):
        if not isinstance(chapter, dict):
            continue
        candidates = chapter.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            strategy = candidate.get("horizontal_strategy")
            zoom = candidate.get("horizontal_zoom_intent")
            focus = candidate.get("horizontal_focus_entity_id")
            base = f"$.chapters[{chapter_index}].candidates[{candidate_index}]"
            if strategy == "original" and zoom in {"subtle", "detail"}:
                candidate["horizontal_zoom_intent"] = "none"
                changes.append(
                    {
                        "json_path": f"{base}.horizontal_zoom_intent",
                        "before": zoom,
                        "after": "none",
                        "rule": "explicit_original_strategy_disables_zoom",
                    }
                )
            if strategy == "original" and focus is not None:
                candidate["horizontal_focus_entity_id"] = None
                changes.append(
                    {
                        "json_path": f"{base}.horizontal_focus_entity_id",
                        "before": focus,
                        "after": None,
                        "rule": "explicit_original_strategy_has_no_focus_entity",
                    }
                )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")), changes


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_feature_normalization_artifacts(
    *,
    output_dir: Path,
    artifact_stem: str,
    raw_output_path: Path,
    raw_output_text: str,
) -> tuple[str, Path, Path]:
    canonical_text, changes = canonicalize_feature_plan_output(raw_output_text)
    canonical_path = output_dir / f"{artifact_stem}.canonical_output.json"
    audit_path = output_dir / f"{artifact_stem}.normalization-audit.json"
    write_json(canonical_path, {"output_text": canonical_text})
    write_json(
        audit_path,
        {
            "contract_version": FEATURE_PLAN_NORMALIZATION_VERSION,
            "interpretation": "conditional_schema_contradictions_only",
            "raw_output_path": str(raw_output_path.resolve()),
            "raw_output_artifact_sha256": sha256_file(raw_output_path),
            "input_output_text_sha256": _text_sha256(raw_output_text),
            "canonical_output_path": str(canonical_path.resolve()),
            "canonical_output_artifact_sha256": sha256_file(canonical_path),
            "canonical_output_text_sha256": _text_sha256(canonical_text),
            "changes": changes,
            "change_count": len(changes),
            "created_at": utc_now(),
        },
    )
    return canonical_text, canonical_path, audit_path


def _resolve_feature_reuse_artifacts(output_dir: Path) -> dict[str, Any]:
    """Resolve one complete, non-mixed paid-response artifact set."""

    sets = (
        {
            "kind": "canonical",
            "request": output_dir / "clip-card-feature-plan.request.json",
            "raw_output": output_dir / "clip-card-feature-plan.raw_output.json",
            "raw_interaction": output_dir / "clip-card-feature-plan.raw_interaction.json",
        },
        {
            "kind": "attempt-01",
            "request": output_dir / "clip-card-feature-plan.attempt-01.request.json",
            "raw_output": output_dir / "clip-card-feature-plan.attempt-01.raw_output.json",
            "raw_interaction": output_dir
            / "clip-card-feature-plan.attempt-01.raw_interaction.json",
        },
    )
    incomplete: list[str] = []
    for artifact_set in sets:
        paths = [artifact_set[key] for key in ("request", "raw_output", "raw_interaction")]
        present = [path.exists() for path in paths]
        if all(present):
            return artifact_set
        if any(present):
            incomplete.append(str(artifact_set["kind"]))
    detail = f"; incomplete sets: {incomplete}" if incomplete else ""
    raise FileNotFoundError(
        "--reuse-raw-output requires one complete canonical or attempt-01 "
        f"request/raw-output/raw-interaction set{detail}"
    )


def _verified_feature_raw_output_text(
    *, raw_output: dict[str, Any], raw_interaction: dict[str, Any]
) -> str:
    """Return a paid response only when both independently saved copies agree."""

    output_text = raw_output.get("output_text")
    interaction_text = raw_interaction.get("output_text")
    if not isinstance(output_text, str) or not isinstance(interaction_text, str):
        raise ValueError(
            "--reuse-raw-output requires string output_text in both raw artifacts"
        )
    if output_text != interaction_text:
        raise ValueError(
            "--reuse-raw-output artifact mismatch: raw interaction output_text "
            "does not exactly match raw output output_text"
        )
    return output_text


def _assert_fresh_feature_namespace_empty(output_dir: Path) -> None:
    existing = sorted(output_dir.glob("clip-card-feature-plan*"))
    if existing:
        raise FileExistsError(
            "fresh feature planning refuses an existing paid artifact namespace; "
            "use --reuse-raw-output or a new output directory: "
            + ", ".join(path.name for path in existing[:8])
        )


def _assert_projection_request_hash(
    *, pointer_path: Path, plan_dir: Path, expected_request_path: Path
) -> None:
    pointer = read_json(pointer_path)
    record = read_json(plan_dir / str(pointer["record_path"]))
    expected = sha256_file(expected_request_path)
    if record.get("source_request_sha256") != expected:
        raise RuntimeError(
            "external projection source request does not match the original paid request"
        )


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
                "entity_ids": event.entity_ids,
                "primary_entity_ids": event.primary_entity_ids,
                "required_entity_ids": event.required_entity_ids,
                "optional_entity_ids": event.optional_entity_ids,
                "avoid_overlay_entity_ids": event.avoid_overlay_entity_ids,
                "entity_relations": [
                    {
                        "entity_id": entity_id,
                        "relations": [
                            relation
                            for relation, members in (
                                ("event_member", event.entity_ids),
                                ("primary", event.primary_entity_ids),
                                ("required", event.required_entity_ids),
                                ("optional", event.optional_entity_ids),
                                ("avoid_overlay", event.avoid_overlay_entity_ids),
                                (
                                    "grounding_target",
                                    [target.entity_id for target in event.grounding_targets],
                                ),
                            )
                            if entity_id in members
                        ],
                    }
                    for entity_id in sorted(
                        set(
                            event.entity_ids
                            + event.primary_entity_ids
                            + event.required_entity_ids
                            + event.optional_entity_ids
                            + event.avoid_overlay_entity_ids
                            + [target.entity_id for target in event.grounding_targets]
                        )
                    )
                ],
                "card_opportunities": [
                    {
                        "kind": opportunity.kind,
                        "rationale": opportunity.rationale,
                        "entity_ids": opportunity.entity_ids,
                    }
                    for opportunity in event.card_opportunities
                ],
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


def compact_card_v3(card: FullClipCard) -> dict[str, object]:
    """Compact selection evidence without locally derivable relation mirrors.

    The model still sees every event and the entity IDs needed to choose a take,
    but it does not receive duplicated relation expansions, per-entity evidence,
    or card-layout records that are irrelevant to editorial ranking.
    """

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
                "entity_ids": event.entity_ids,
                "primary_entity_ids": event.primary_entity_ids,
                "required_entity_ids": event.required_entity_ids,
                "optional_entity_ids": event.optional_entity_ids,
                "avoid_overlay_entity_ids": event.avoid_overlay_entity_ids,
                "grounding_target_entity_ids": [
                    target.entity_id for target in event.grounding_targets
                ],
            }
            for event in card.events
        ],
    }


def validate_plan_contract(
    plan: ClipCardFeaturePlanV2,
    *,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
    cards: dict[str, FullClipCard],
    require_v2: bool = True,
) -> None:
    if require_v2 and plan.contract_version != "clip-card-feature-cut-v2":
        raise ValueError("new feature planning requests require the v2 Top-K contract")
    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("model changed immutable project or catalog ID")
    expected_features = [chapter.feature_id for chapter in brief.chapters]
    if [chapter.feature_id for chapter in plan.chapters] != expected_features:
        raise ValueError("plan must preserve every brief chapter exactly once and in order")
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}

    def validate_evidence_reference(
        *,
        asset_id: str,
        event_id: str,
        frame_id: str,
    ) -> tuple[FullClipCard, Any]:
        card = cards.get(asset_id)
        if card is None:
            raise ValueError(f"unknown selected asset: {asset_id}")
        event = next((item for item in card.events if item.event_id == event_id), None)
        if event is None:
            raise ValueError(f"unknown selected event: {asset_id}/{event_id}")
        frame = frames.get(frame_id)
        if frame is None:
            raise ValueError(f"unknown selected frame: {frame_id}")
        selected_clip = clips.get(frame.clip_id)
        if selected_clip is None or f"sha256:{selected_clip.sha256}" != asset_id:
            raise ValueError(f"frame does not belong to selected asset: {frame_id}")
        frame_mmss = mmss(frame.requested_time_ms)
        if not event.start_mmss <= frame_mmss < event.end_mmss:
            raise ValueError(f"frame lies outside selected event: {frame_id}")
        return card, event

    relation_fields = {
        "event_member": "entity_ids",
        "primary": "primary_entity_ids",
        "required": "required_entity_ids",
        "optional": "optional_entity_ids",
        "avoid_overlay": "avoid_overlay_entity_ids",
    }

    def validate_region_lineage(
        *, candidate: ClipCardFeatureCandidate, card: FullClipCard, event: Any
    ) -> None:
        known_entities = {entity.entity_id for entity in card.entities}
        grounding_entities = {target.entity_id for target in event.grounding_targets}
        for region in candidate.resolved_regions:
            for ref in region.entity_refs:
                if ref.entity_id not in known_entities:
                    raise ValueError(
                        f"candidate region references unknown entity: {candidate.candidate_id}/"
                        f"{region.region_id}/{ref.entity_id}"
                    )
                if ref.event_relation == "grounding_target":
                    valid_relation = ref.entity_id in grounding_entities
                else:
                    valid_relation = ref.entity_id in getattr(
                        event, relation_fields[ref.event_relation]
                    )
                if not valid_relation:
                    raise ValueError(
                        f"candidate region relation is not backed by its event: "
                        f"{candidate.candidate_id}/{region.region_id}/{ref.entity_id}/"
                        f"{ref.event_relation}"
                    )

    for chapter in plan.chapters:
        if chapter.evidence_status == "not_found":
            continue
        # A brief target states editorial priority, not the geometry algorithm.
        # Local preflight may legitimately choose a stable fit strategy when a
        # moving crop is unnecessary or cannot preserve the required extent.
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
            validate_evidence_reference(
                asset_id=asset_id, event_id=event_id, frame_id=frame_id
            )
        for candidate in chapter.candidates:
            card, event = validate_evidence_reference(
                asset_id=candidate.source_asset_id,
                event_id=candidate.event_id,
                frame_id=candidate.frame_id,
            )
            validate_region_lineage(candidate=candidate, card=card, event=event)


def validate_plan_contract_v3(
    plan: ClipCardFeaturePlanV3,
    *,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
    cards: dict[str, FullClipCard],
) -> None:
    """Validate model choices while deriving no semantic values from the model."""

    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("model changed immutable project or catalog ID")
    expected_features = [chapter.feature_id for chapter in brief.chapters]
    if [chapter.feature_id for chapter in plan.chapters] != expected_features:
        raise ValueError("plan must preserve every brief chapter exactly once and in order")
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    for chapter in plan.chapters:
        if chapter.evidence_status == "not_found":
            continue
        for candidate in chapter.candidates:
            card = cards.get(candidate.source_asset_id)
            if card is None:
                raise ValueError(f"unknown selected asset: {candidate.source_asset_id}")
            event = next(
                (item for item in card.events if item.event_id == candidate.event_id), None
            )
            if event is None:
                raise ValueError(
                    f"unknown selected event: {candidate.source_asset_id}/{candidate.event_id}"
                )
            frame = frames.get(candidate.frame_id)
            if frame is None:
                raise ValueError(f"unknown selected frame: {candidate.frame_id}")
            clip = clips.get(frame.clip_id)
            if clip is None or f"sha256:{clip.sha256}" != candidate.source_asset_id:
                raise ValueError(
                    f"frame does not belong to selected asset: {candidate.frame_id}"
                )
            frame_mmss = mmss(frame.requested_time_ms)
            if not event.start_mmss <= frame_mmss < event.end_mmss:
                raise ValueError(f"frame lies outside selected event: {candidate.frame_id}")
            event_entities = set(
                event.entity_ids
                + event.primary_entity_ids
                + event.required_entity_ids
                + event.optional_entity_ids
                + event.avoid_overlay_entity_ids
                + [target.entity_id for target in event.grounding_targets]
            )
            selected_entities = set(
                candidate.required_entity_ids
                + candidate.preferred_entity_ids
                + candidate.sacrificable_entity_ids
            )
            if candidate.horizontal_focus_entity_id:
                selected_entities.add(candidate.horizontal_focus_entity_id)
            unknown = sorted(selected_entities - event_entities)
            if unknown:
                raise ValueError(
                    f"candidate focus entities are not backed by its event: "
                    f"{candidate.candidate_id}/{unknown}"
                )
            classified = set(
                candidate.required_entity_ids
                + candidate.preferred_entity_ids
                + candidate.sacrificable_entity_ids
            )
            unclassified = sorted(
                (set(event.primary_entity_ids) | set(event.required_entity_ids))
                - classified
            )
            if unclassified:
                raise ValueError(
                    f"candidate did not classify important event entities: "
                    f"{candidate.candidate_id}/{unclassified}"
                )


def build_selected_clip_card_evidence(
    plan: ClipCardFeaturePlanV3,
    *,
    cards: dict[str, FullClipCard],
) -> SelectedClipCardEvidence:
    """Snapshot only locally validated events referenced by the v3 source plan."""

    keys = sorted(
        {
            (candidate.source_asset_id, candidate.event_id)
            for chapter in plan.chapters
            for candidate in chapter.candidates
        }
    )
    events: list[SelectedEvidenceEvent] = []
    for asset_id, event_id in keys:
        card = cards.get(asset_id)
        if card is None:
            raise ValueError(f"cannot snapshot unknown asset: {asset_id}")
        event = next((item for item in card.events if item.event_id == event_id), None)
        if event is None:
            raise ValueError(f"cannot snapshot unknown event: {asset_id}/{event_id}")
        referenced_ids = set(
            event.entity_ids
            + event.primary_entity_ids
            + event.required_entity_ids
            + event.optional_entity_ids
            + event.avoid_overlay_entity_ids
            + [target.entity_id for target in event.grounding_targets]
        )
        entities_by_id = {entity.entity_id: entity for entity in card.entities}
        events.append(
            SelectedEvidenceEvent(
                source_asset_id=asset_id,
                event_id=event_id,
                entity_ids=list(event.entity_ids),
                primary_entity_ids=list(event.primary_entity_ids),
                required_entity_ids=list(event.required_entity_ids),
                optional_entity_ids=list(event.optional_entity_ids),
                avoid_overlay_entity_ids=list(event.avoid_overlay_entity_ids),
                entities=[
                    SelectedEvidenceEntity(
                        entity_id=entity_id,
                        kind=entities_by_id[entity_id].kind.value,
                        label=entities_by_id[entity_id].label,
                        distinguishing_features=(
                            entities_by_id[entity_id].distinguishing_features
                        ),
                    )
                    for entity_id in sorted(referenced_ids)
                ],
                grounding_targets=[
                    SelectedEvidenceGroundingTarget(
                        entity_id=target.entity_id,
                        target_description=target.target_description,
                    )
                    for target in event.grounding_targets
                ],
            )
        )
    return SelectedClipCardEvidence(
        contract_version="clip-card-feature-cut-selected-evidence-v1",
        events=events,
    )


def _selected_first_candidates_v3(
    chapter: ClipCardFeatureSelectV3, selected_candidate_id: str | None
) -> list[ClipCardFeatureCandidateV3]:
    if not chapter.candidates:
        return []
    if selected_candidate_id is None:
        raise ValueError("candidate contract is missing its selected candidate ID")
    selected = next(
        candidate
        for candidate in chapter.candidates
        if candidate.candidate_id == selected_candidate_id
    )
    return [selected] + [
        candidate
        for candidate in chapter.candidates
        if candidate.candidate_id != selected_candidate_id
    ]


def _region_kind(entity_kind: str) -> Literal[
    "subject", "text_region", "ui_region", "graphic", "other"
]:
    if entity_kind == "text_region":
        return "text_region"
    if entity_kind in {"phone_screen", "screen", "ui_element"}:
        return "ui_region"
    if entity_kind == "logo":
        return "graphic"
    if entity_kind == "other":
        return "other"
    return "subject"


def _safe_region_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip("-.") or "region"


def _event_index(
    evidence: SelectedClipCardEvidence,
) -> dict[tuple[str, str], SelectedEvidenceEvent]:
    index: dict[tuple[str, str], SelectedEvidenceEvent] = {}
    for event in evidence.events:
        key = (event.source_asset_id, event.event_id)
        if key in index:
            raise ValueError(f"duplicate selected evidence event: {key}")
        index[key] = event
    return index


def _target_description(event: SelectedEvidenceEvent, entity_id: str) -> str:
    target = next(
        (item for item in event.grounding_targets if item.entity_id == entity_id), None
    )
    if target is not None:
        return target.target_description
    entity = next((item for item in event.entities if item.entity_id == entity_id), None)
    if entity is None:
        raise ValueError(
            f"selected evidence is missing entity {event.source_asset_id}/"
            f"{event.event_id}/{entity_id}"
        )
    details = entity.distinguishing_features.strip()
    return f"{entity.label}; {details}" if details else entity.label


def _observable_entity_relations(
    event: SelectedEvidenceEvent, entity_id: str
) -> list[str]:
    relations = [
        relation
        for relation, members in (
            ("event_member", event.entity_ids),
            ("primary", event.primary_entity_ids),
            ("required", event.required_entity_ids),
            ("optional", event.optional_entity_ids),
            ("avoid_overlay", event.avoid_overlay_entity_ids),
            ("grounding_target", [item.entity_id for item in event.grounding_targets]),
        )
        if entity_id in members
    ]
    return [f"event_relation={relation}" for relation in relations]


def _project_candidate_regions_v3(
    candidate: ClipCardFeatureCandidateV3,
    event: SelectedEvidenceEvent,
) -> list[FramingRegionIntent]:
    if candidate.vertical_strategy != "tracked_crop":
        return []
    hard_ids = list(candidate.required_entity_ids)
    preferred_ids = [
        entity_id
        for entity_id in candidate.preferred_entity_ids
        if entity_id not in hard_ids
    ]
    overlay_ids = [
        entity_id
        for entity_id in event.avoid_overlay_entity_ids
        if entity_id not in hard_ids and entity_id not in preferred_ids
        and entity_id not in candidate.sacrificable_entity_ids
    ]
    roles = [
        *( (entity_id, "required") for entity_id in hard_ids ),
        *( (entity_id, "preferred") for entity_id in preferred_ids ),
        *( (entity_id, "avoid_overlay") for entity_id in overlay_ids ),
    ]
    if len(roles) > 8:
        raise ValueError(
            f"locally derived crop contract exceeds eight regions: {candidate.candidate_id}"
        )
    entities = {entity.entity_id: entity for entity in event.entities}
    projected: list[FramingRegionIntent] = []
    for entity_id, role in roles:
        entity = entities.get(entity_id)
        if entity is None:
            raise ValueError(
                f"selected evidence is missing crop entity: {candidate.candidate_id}/"
                f"{entity_id}"
            )
        kind = _region_kind(entity.kind)
        atomic = role == "required" and kind in {
            "text_region",
            "ui_region",
            "graphic",
        }
        projected.append(
            FramingRegionIntent(
                region_id=(
                    f"{_safe_region_token(candidate.candidate_id)}."
                    f"{_safe_region_token(role)}.{_safe_region_token(entity_id)}"
                ),
                entity_id=entity_id,
                target_description=_target_description(event, entity_id),
                kind=kind,
                role=role,
                atomic=atomic,
                minimum_visible_fraction=1.0 if role == "required" else None,
                observable_relations=list(
                    dict.fromkeys(
                        _observable_entity_relations(event, entity_id)
                        + [f"editorial_framing_intent={candidate.framing_intent}"]
                    )
                ),
                exclusions=[],
            )
        )
    return projected


def project_feature_contracts_v3(
    plan: ClipCardFeaturePlanV3,
    *,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
    selected_evidence: SelectedClipCardEvidence,
) -> FeatureEditPlan:
    """Project v3 editorial choices using only hash-bound local Clip Cards."""

    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("source plan differs from projection catalog/brief")
    index = _event_index(selected_evidence)
    projected: list[FeatureChapterSelect] = []
    for chapter in plan.chapters:
        if chapter.evidence_status == "not_found":
            projected.append(
                FeatureChapterSelect(
                    feature_id=chapter.feature_id,
                    evidence_status="not_found",
                    horizontal_frame_id=None,
                    vertical_frame_id=None,
                    observed_visual_evidence="No supported Clip Card evidence was selected.",
                    selection_reason="The evidence-bound planner returned not_found.",
                    horizontal_strategy="original",
                    horizontal_zoom_intent="none",
                    horizontal_target_description=None,
                    vertical_strategy="fit_with_background",
                    vertical_target_description=None,
                    quality_risks=["No supported source evidence."],
                    confidence=0.0,
                )
            )
            continue
        horizontal_options = _selected_first_candidates_v3(
            chapter, chapter.horizontal_candidate_id
        )
        vertical_options = _selected_first_candidates_v3(
            chapter, chapter.vertical_candidate_id
        )
        horizontal_primary = horizontal_options[0]
        vertical_primary = vertical_options[0]

        def evidence_event(candidate: ClipCardFeatureCandidateV3) -> SelectedEvidenceEvent:
            event = index.get((candidate.source_asset_id, candidate.event_id))
            if event is None:
                raise ValueError(
                    f"selected evidence artifact is missing candidate event: "
                    f"{candidate.candidate_id}"
                )
            return event

        def horizontal_target(candidate: ClipCardFeatureCandidateV3) -> str | None:
            if candidate.horizontal_focus_entity_id is None:
                return None
            return _target_description(
                evidence_event(candidate), candidate.horizontal_focus_entity_id
            )

        def vertical_target(candidate: ClipCardFeatureCandidateV3) -> str | None:
            if not candidate.required_entity_ids:
                return None
            event = evidence_event(candidate)
            descriptions = [
                _target_description(event, entity_id)
                for entity_id in candidate.required_entity_ids
            ]
            return " | ".join(descriptions)

        horizontal_primary_target = horizontal_target(horizontal_primary)
        vertical_primary_target = vertical_target(vertical_primary)
        observed = horizontal_primary.observed_visual_evidence
        reason = horizontal_primary.selection_reason
        if horizontal_primary.candidate_id != vertical_primary.candidate_id:
            observed = (
                f"16:9: {observed} 9:16: {vertical_primary.observed_visual_evidence}"
            )
            reason = f"16:9: {reason} 9:16: {vertical_primary.selection_reason}"
        quality_risks = list(
            dict.fromkeys(horizontal_primary.quality_risks + vertical_primary.quality_risks)
        )
        projected.append(
            FeatureChapterSelect(
                feature_id=chapter.feature_id,
                evidence_status=chapter.evidence_status,
                horizontal_frame_id=horizontal_primary.frame_id,
                vertical_frame_id=vertical_primary.frame_id,
                observed_visual_evidence=observed,
                selection_reason=reason,
                horizontal_strategy=horizontal_primary.horizontal_strategy,
                horizontal_zoom_intent=horizontal_primary.horizontal_zoom_intent,
                horizontal_target_description=horizontal_primary_target,
                vertical_strategy=vertical_primary.vertical_strategy,
                vertical_target_description=vertical_primary_target,
                quality_risks=quality_risks,
                confidence=min(horizontal_primary.confidence, vertical_primary.confidence),
                horizontal_candidates=[
                    FeatureHorizontalCandidate(
                        candidate_id=candidate.candidate_id,
                        rank=rank,
                        source_asset_id=candidate.source_asset_id,
                        event_id=candidate.event_id,
                        frame_id=candidate.frame_id,
                        observed_visual_evidence=candidate.observed_visual_evidence,
                        selection_reason=candidate.selection_reason,
                        strategy=candidate.horizontal_strategy,
                        zoom_intent=candidate.horizontal_zoom_intent,
                        target_description=horizontal_target(candidate),
                        quality_risks=candidate.quality_risks,
                        confidence=candidate.confidence,
                    )
                    for rank, candidate in enumerate(horizontal_options, start=1)
                ],
                vertical_candidates=[
                    FeatureVerticalCandidate(
                        candidate_id=candidate.candidate_id,
                        rank=rank,
                        source_asset_id=candidate.source_asset_id,
                        event_id=candidate.event_id,
                        frame_id=candidate.frame_id,
                        observed_visual_evidence=candidate.observed_visual_evidence,
                        selection_reason=candidate.selection_reason,
                        strategy=candidate.vertical_strategy,
                        crop_mode=candidate.vertical_crop_mode,
                        target_description=vertical_target(candidate),
                        regions=_project_candidate_regions_v3(
                            candidate, evidence_event(candidate)
                        ),
                        quality_risks=candidate.quality_risks,
                        confidence=candidate.confidence,
                    )
                    for rank, candidate in enumerate(vertical_options, start=1)
                ],
            )
        )
    return FeatureEditPlan(
        project_id=brief.project_id,
        catalog_id=catalog.catalog_id,
        title=plan.title,
        chapters=projected,
        uncertainties=plan.uncertainties,
        model_provenance=plan.model_provenance,
    )


def _selected_first_candidates(
    chapter: ClipCardFeatureSelectV2, selected_candidate_id: str | None
) -> list[ClipCardFeatureCandidate]:
    if not chapter.candidates:
        return []
    if selected_candidate_id is None:
        raise ValueError("candidate contract is missing its selected candidate ID")
    selected = next(
        candidate
        for candidate in chapter.candidates
        if candidate.candidate_id == selected_candidate_id
    )
    return [selected] + [
        candidate
        for candidate in chapter.candidates
        if candidate.candidate_id != selected_candidate_id
    ]


def _project_candidate_regions(
    candidate: ClipCardFeatureCandidate,
) -> list[FramingRegionIntent]:
    """Flatten auditable unions into executable single-entity regions."""

    role_map = {
        "hard_core": "required",
        "soft_extent": "preferred",
        "overlay_keepout": "avoid_overlay",
    }
    projected: list[FramingRegionIntent] = []
    for region in candidate.resolved_regions:
        union = len(region.entity_refs) > 1
        for index, ref in enumerate(region.entity_refs, start=1):
            region_id = (
                f"{region.region_id}.member-{index}" if union else region.region_id
            )
            member_description = (
                f"{region.target_description}; exact member entity_id={ref.entity_id}, "
                f"event_relation={ref.event_relation}"
                if union
                else region.target_description
            )
            projected.append(
                FramingRegionIntent(
                    region_id=region_id,
                    entity_id=ref.entity_id,
                    target_description=member_description,
                    kind=region.kind,
                    role=role_map[region.constraint_role],
                    atomic=region.atomic,
                    minimum_visible_fraction=(
                        1.0
                        if region.constraint_role == "hard_core" or region.atomic
                        else None
                    ),
                    observable_relations=[
                        f"event_relation={ref.event_relation}",
                        region.observable_relation,
                    ],
                    exclusions=[],
                )
            )
    return projected


def _upgrade_legacy_feature_plan(
    plan: ClipCardFeaturePlan,
) -> ClipCardFeaturePlanV2:
    return ClipCardFeaturePlanV2.model_validate(
        {"contract_version": "legacy-v1", **plan.model_dump(mode="json")}
    )


def project_feature_contracts(
    plan: ClipCardFeaturePlan | ClipCardFeaturePlanV2,
    *,
    brief: FeatureEditBrief,
    catalog: RushesCatalog,
    preserve_runtime_candidates: bool | None = None,
) -> FeatureEditPlan:
    """Deterministically project the richer Clip Card plan for the renderer."""

    legacy_source = isinstance(plan, ClipCardFeaturePlan)
    if legacy_source:
        plan = _upgrade_legacy_feature_plan(plan)
    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("source plan differs from projection catalog/brief")
    if preserve_runtime_candidates is None:
        preserve_runtime_candidates = not legacy_source and (
            plan.contract_version == "clip-card-feature-cut-v2"
        )
    projected: list[FeatureChapterSelect] = []
    for chapter in plan.chapters:
        horizontal_options = _selected_first_candidates(
            chapter, chapter.horizontal_candidate_id
        )
        vertical_options = _selected_first_candidates(
            chapter, chapter.vertical_candidate_id
        )
        projected.append(
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
                horizontal_candidates=[
                    FeatureHorizontalCandidate(
                        candidate_id=candidate.candidate_id,
                        rank=rank,
                        source_asset_id=candidate.source_asset_id,
                        event_id=candidate.event_id,
                        frame_id=candidate.frame_id,
                        observed_visual_evidence=candidate.observed_visual_evidence,
                        selection_reason=candidate.selection_reason,
                        strategy=candidate.horizontal_strategy,
                        zoom_intent=candidate.horizontal_zoom_intent,
                        target_description=candidate.horizontal_target_description,
                        quality_risks=candidate.quality_risks,
                        confidence=candidate.confidence,
                    )
                    for rank, candidate in enumerate(horizontal_options, start=1)
                ] if preserve_runtime_candidates else [],
                vertical_candidates=[
                    FeatureVerticalCandidate(
                        candidate_id=candidate.candidate_id,
                        rank=rank,
                        source_asset_id=candidate.source_asset_id,
                        event_id=candidate.event_id,
                        frame_id=candidate.frame_id,
                        observed_visual_evidence=candidate.observed_visual_evidence,
                        selection_reason=candidate.selection_reason,
                        strategy=candidate.vertical_strategy,
                        crop_mode=candidate.vertical_crop_mode,
                        target_description=candidate.vertical_target_description,
                        regions=_project_candidate_regions(candidate),
                        quality_risks=candidate.quality_risks,
                        confidence=candidate.confidence,
                    )
                    for rank, candidate in enumerate(vertical_options, start=1)
                ] if preserve_runtime_candidates else [],
            )
        )
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
    """Reproduce the legacy v1 candidate-free projection exactly."""

    del source_artifacts
    if not isinstance(source_plan, ClipCardFeaturePlan):
        raise ValueError("clip-card-feature-cut-v1 requires its exact legacy source schema")
    return brief, project_feature_contracts(
        source_plan,
        brief=brief,
        catalog=catalog,
        preserve_runtime_candidates=False,
    )


def reproject_external_feature_plan_v2(
    *,
    source_plan: ClipCardFeaturePlanV2,
    catalog: RushesCatalog,
    brief: FeatureEditBrief,
    source_artifacts: dict[str, Path],
) -> tuple[FeatureEditBrief, FeatureEditPlan]:
    """Reproduce the v2 Top-K runtime-candidate projection exactly."""

    del source_artifacts
    if (
        not isinstance(source_plan, ClipCardFeaturePlanV2)
        or source_plan.contract_version != "clip-card-feature-cut-v2"
    ):
        raise ValueError(
            "clip-card-feature-cut-v2 requires a clip-card-feature-cut-v2 source plan"
        )
    return brief, project_feature_contracts(
        source_plan,
        brief=brief,
        catalog=catalog,
        preserve_runtime_candidates=True,
    )


def reproject_external_feature_plan_v3(
    *,
    source_plan: ClipCardFeaturePlanV3,
    catalog: RushesCatalog,
    brief: FeatureEditBrief,
    source_artifacts: dict[str, Path],
) -> tuple[FeatureEditBrief, FeatureEditPlan]:
    """Reproduce the v3 projection from choices plus hash-bound local evidence."""

    if not isinstance(source_plan, ClipCardFeaturePlanV3):
        raise ValueError("clip-card-feature-cut-v3 requires its exact v3 source schema")
    evidence_path = source_artifacts.get("selected_clip_card_evidence")
    if evidence_path is None:
        raise ValueError("clip-card-feature-cut-v3 requires selected_clip_card_evidence")
    selected_evidence = SelectedClipCardEvidence.model_validate(read_json(evidence_path))
    return brief, project_feature_contracts_v3(
        source_plan,
        brief=brief,
        catalog=catalog,
        selected_evidence=selected_evidence,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_json", type=Path)
    parser.add_argument("brief_json", type=Path)
    parser.add_argument("prepared_library", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=0,
        help=(
            "Opt-in full paid retries after schema/lineage validation failure. "
            "The default is zero so one planning command makes at most one "
            "Gemini request."
        ),
    )
    parser.add_argument(
        "--thinking-level",
        choices=["low", "high"],
        default="high",
    )
    parser.add_argument(
        "--reuse-raw-output",
        action="store_true",
        help=(
            "Canonicalize, revalidate, and project an existing paid response "
            "without creating another API request"
        ),
    )
    args = parser.parse_args()
    if args.repair_attempts < 0:
        parser.error("--repair-attempts must be zero or greater")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not args.reuse_raw_output and not api_key:
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
            "clip_card": compact_card_v3(card),
            "available_catalog_frames": frame_map[asset_id],
        }
        for asset_id, card in cards.items()
    ]
    prompt = f"""
你是 evidence-bound 的資深短影音挑帶剪輯師。請使用完整 Clip Card library，為使用者 brief 的每個 chapter 保留有排序的候選 take，再分別選出橫式與直式代表。你只能引用輸入列出的 source_asset_id、event_id、entity_id 與 RF frame_id。

規則：
1. brief 是允許使用的產品 claim，不是畫面證據；observed_visual_evidence 只能寫 Clip Card 直接支持的內容。
2. 每個 brief feature_id 必須依原順序恰好回傳一次。supported／partial chapter 必須保留 2–4 個依品質排序、evidence frame 不重複的 candidates；優先完整動作、清楚結果、低遮擋、低反光與不同 take。not_found 不得虛構候選。
3. selected frame 的 local_mmss 必須位於所引用 event 的 [start_mmss,end_mmss)；不得自行創造 frame ID 或 timestamp。RF frame_id 必須從 available_catalog_frames 逐字複製並保留全部六位數與前導零，例如 RF000204 不可縮成 RF00204。
4. 若可見型號、文字、數字或物件身分與 brief 衝突，優先改選沒有衝突的 take；沒有可靠 take 時用 partial 或 not_found 並保存風險。
5. 每個 candidate 都必須保存可直接重試的 16:9 strategy／zoom／horizontal_focus_entity_id，以及 9:16 strategy、framing_intent 和 brief-specific entity priorities。橫式與直式可以從同一候選組選不同來源；horizontal_candidate_id／vertical_candidate_id 必須指向 candidates。不要重複輸出 rank-1 asset/event/frame mirror、target description 或 resolved crop regions；程式會從所選 candidate 與 hash-bound Clip Card evidence 確定性補出。
   - horizontal_strategy=original 時，horizontal_zoom_intent 必須是 none，而且 horizontal_focus_entity_id 必須是 null；原始構圖不需要追蹤焦點。
   - horizontal_strategy=tracked_reframe 時，horizontal_zoom_intent 必須是 subtle 或 detail，而且 horizontal_focus_entity_id 必須引用該 event 中一個可見 entity。
6. 9:16 應把 brief 的 vertical_primary_target_description 視為內容優先序，不是強制演算法。只有需要動態跟隨且存在可靠 target 時才用 tracked_crop；若穩定構圖已可保留內容，或窄裁切無法安全包含必要範圍，可以使用 fit_with_background。不得只因 brief 有 primary target 就強制 tracked_crop。
7. required_entity_ids、preferred_entity_ids、sacrificable_entity_ids 是針對本 brief 與本 aspect 的編輯決定，三組必須互斥，清單順序代表優先序，且只能引用該 event 已列出的 entity。每個 event 的 primary_entity_ids 與 required_entity_ids 都必須被歸入三組之一；不得把未觀察到的 entity 加入。tracked_crop 至少要有一個 required entity。
8. framing_intent 只需簡潔描述本候選的構圖取捨；不得輸出座標、bbox、mask、target description 或 verbose region contract。程式會把這些 entity priority ID 與 Clip Card entity/grounding target 資料轉成 domain-neutral hard-core、soft-extent 與 overlay keepout regions。
9. bbox、mask、crop 座標與精確 cut point 均由後續 Grounding／tracker／FFmpeg 處理；本階段不得輸出座標。
10. confidence 是 proposal，不是人工真值；候選排序仍須由可見 evidence 與風險說明支持。

contract_version 必須原樣回傳：clip-card-feature-cut-v3
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
            "Never replace visible evidence with model memory or likely product knowledge. "
            "Preserve 2-4 auditable alternatives for every supported chapter. Return concise "
            "brief-specific entity priorities, but never duplicate descriptions, rank-one "
            "mirror fields, or verbose crop regions that local Clip Cards can derive. A brief "
            "target is editorial intent, not authorization to force a tracked crop. "
            "For original horizontal framing, zoom must be none and focus entity must be null; "
            "only tracked_reframe may name a horizontal focus entity."
        ),
        "store": False,
        "input": [{"type": "text", "text": prompt}],
        "generation_config": {
            "thinking_level": args.thinking_level,
        },
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": gemini_response_schema(ClipCardFeaturePlanV3),
        },
    }
    plan: ClipCardFeaturePlanV3 | None = None
    interaction_id = ""
    source_request_path: Path
    source_raw_output_path: Path
    source_raw_interaction_path: Path
    canonical_output_path: Path
    normalization_audit_path: Path
    extra_projection_artifacts: dict[str, Path] = {}
    if args.reuse_raw_output:
        artifacts = _resolve_feature_reuse_artifacts(args.output_dir)
        source_request_path = artifacts["request"]
        source_raw_output_path = artifacts["raw_output"]
        source_raw_interaction_path = artifacts["raw_interaction"]
        original_request = read_json(source_request_path)
        raw_interaction = read_json(source_raw_interaction_path)
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
                "Run with the artifact's original JASCUE_GEMINI_MODEL instead."
            )
        reprojection_request_path = (
            args.output_dir / "clip-card-feature-plan.reprojection-request.json"
        )
        write_json(reprojection_request_path, request)
        raw_output = read_json(source_raw_output_path)
        output_text = _verified_feature_raw_output_text(
            raw_output=raw_output,
            raw_interaction=raw_interaction,
        )
        output_text, canonical_output_path, normalization_audit_path = (
            _write_feature_normalization_artifacts(
                output_dir=args.output_dir,
                artifact_stem="clip-card-feature-plan",
                raw_output_path=source_raw_output_path,
                raw_output_text=output_text,
            )
        )
        reuse_record_path = args.output_dir / "clip-card-feature-plan.raw-output-reuse.json"
        write_json(
            reuse_record_path,
            {
                "interpretation": (
                    "saved_model_response_canonicalized_revalidated_and_projected_"
                    "with_no_new_model_call"
                ),
                "artifact_set": artifacts["kind"],
                "original_request_path": str(artifacts["request"].resolve()),
                "original_request_sha256": sha256_file(artifacts["request"]),
                "raw_output_path": str(source_raw_output_path.resolve()),
                "raw_output_sha256": sha256_file(source_raw_output_path),
                "raw_interaction_path": str(source_raw_interaction_path.resolve()),
                "raw_interaction_sha256": sha256_file(source_raw_interaction_path),
                "current_reprojection_request_path": str(
                    reprojection_request_path.resolve()
                ),
                "current_reprojection_request_sha256": sha256_file(
                    reprojection_request_path
                ),
                "normalization_audit_path": str(normalization_audit_path.resolve()),
                "normalization_audit_sha256": sha256_file(normalization_audit_path),
                "reused_at": utc_now(),
            },
        )
        interaction_id = str(raw_interaction.get("id") or "")
        plan = ClipCardFeaturePlanV3.model_validate_json(output_text)
        validate_plan_contract_v3(plan, brief=brief, catalog=catalog, cards=cards)
        if plan.model_provenance.model_id != MODEL_ID:
            raise ValueError(
                "--reuse-raw-output model provenance mismatch: "
                f"expected {MODEL_ID!r}, got {plan.model_provenance.model_id!r}"
            )
        extra_projection_artifacts = {
            "original_request": artifacts["request"],
            "current_reprojection_request": reprojection_request_path,
            "raw_output_reuse_record": reuse_record_path,
        }
    else:
        _assert_fresh_feature_namespace_empty(args.output_dir)
        client = genai.Client(api_key=api_key)
        try:
            previous_error = ""
            for attempt in range(1, args.repair_attempts + 2):
                attempt_request = request
                if attempt > 1:
                    repair_prompt = (
                        prompt
                        + "\n\n## 前次輸出未通過本機 contract\n"
                        + previous_error[:6000]
                        + "\n請重新產生完整結果，不得只回傳修補片段。完整 evidence 已在上方，"
                        "不要重複沿用前次不合法輸出。"
                    )
                    attempt_request = {
                        **request,
                        "input": [{"type": "text", "text": repair_prompt}],
                        "generation_config": {"thinking_level": "low"},
                    }
                attempt_stem = f"clip-card-feature-plan.attempt-{attempt:02d}"
                attempt_request_path = args.output_dir / f"{attempt_stem}.request.json"
                attempt_raw_interaction_path = (
                    args.output_dir / f"{attempt_stem}.raw_interaction.json"
                )
                attempt_raw_output_path = args.output_dir / f"{attempt_stem}.raw_output.json"
                write_json(attempt_request_path, attempt_request)
                current = client.interactions.create(**attempt_request)
                raw = _raw_dump(current)
                write_json(attempt_raw_interaction_path, raw)
                write_json(attempt_raw_output_path, {"output_text": current.output_text})
                try:
                    canonical_text, attempt_canonical_path, attempt_audit_path = (
                        _write_feature_normalization_artifacts(
                            output_dir=args.output_dir,
                            artifact_stem=attempt_stem,
                            raw_output_path=attempt_raw_output_path,
                            raw_output_text=current.output_text,
                        )
                    )
                    plan = ClipCardFeaturePlanV3.model_validate_json(canonical_text)
                    validate_plan_contract_v3(
                        plan,
                        brief=brief,
                        catalog=catalog,
                        cards=cards,
                    )
                    interaction_id = getattr(current, "id", None) or ""
                    source_request_path = args.output_dir / "clip-card-feature-plan.request.json"
                    source_raw_interaction_path = (
                        args.output_dir / "clip-card-feature-plan.raw_interaction.json"
                    )
                    source_raw_output_path = (
                        args.output_dir / "clip-card-feature-plan.raw_output.json"
                    )
                    write_json(source_request_path, attempt_request)
                    write_json(source_raw_interaction_path, raw)
                    write_json(source_raw_output_path, {"output_text": current.output_text})
                    canonical_text, canonical_output_path, normalization_audit_path = (
                        _write_feature_normalization_artifacts(
                            output_dir=args.output_dir,
                            artifact_stem="clip-card-feature-plan",
                            raw_output_path=source_raw_output_path,
                            raw_output_text=current.output_text,
                        )
                    )
                    break
                except (ValidationError, ValueError) as error:
                    plan = None
                    previous_error = str(error)
                    write_json(
                        args.output_dir / f"{attempt_stem}.schema-validation.json",
                        {
                            "ok": False,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    )
            if plan is None:
                raise ValueError(
                    f"Clip Card feature plan failed after {args.repair_attempts + 1} "
                    f"attempts: {previous_error}"
                )
        finally:
            client.close()
    assert plan is not None
    final_audit = plan.model_copy(
        update={
            "model_provenance": plan.model_provenance.model_copy(
                update={"interaction_id": interaction_id}
            )
        }
    )
    selected_evidence = build_selected_clip_card_evidence(
        final_audit,
        cards=cards,
    )
    final_plan = project_feature_contracts_v3(
        final_audit,
        brief=brief,
        catalog=catalog,
        selected_evidence=selected_evidence,
    )
    write_json(args.output_dir / "clip-card-feature-plan.json", final_audit)
    write_json(
        args.output_dir / "selected-clip-card-evidence.json",
        selected_evidence,
    )
    write_json(args.output_dir / "feature_edit_plan.json", final_plan)
    write_json(
        args.output_dir / "clip-card-feature-plan.schema-validation.json",
        {"ok": True, "clip_card_count": len(cards), "frame_count": len(frames)},
    )
    projection_pointer = write_external_feature_plan_projection(
        plan_dir=args.output_dir,
        projection_contract_id="clip-card-feature-cut-v3",
        catalog_path=args.catalog_json,
        brief_path=args.brief_json,
        feature_plan_path=args.output_dir / "feature_edit_plan.json",
        source_plan_path=args.output_dir / "clip-card-feature-plan.json",
        source_request_path=source_request_path,
        source_artifacts={
            "source_raw_interaction": source_raw_interaction_path,
            # The projection validator must parse the exact canonical source
            # used to build the saved plan.  The immutable paid response stays
            # separately hash-bound for audit and replay.
            "source_raw_output": canonical_output_path,
            "original_raw_output": source_raw_output_path,
            "canonicalized_output": canonical_output_path,
            "normalization_audit": normalization_audit_path,
            "selected_clip_card_evidence": (
                args.output_dir / "selected-clip-card-evidence.json"
            ),
            **extra_projection_artifacts,
        },
    )
    if args.reuse_raw_output:
        _assert_projection_request_hash(
            pointer_path=projection_pointer,
            plan_dir=args.output_dir,
            expected_request_path=artifacts["request"],
        )
    usage_paths = sorted(
        args.output_dir.glob("clip-card-feature-plan.attempt-*.raw_interaction.json")
    )
    if not usage_paths:
        usage_paths = [source_raw_interaction_path]
    pricing = summarize_usage_files(
        usage_paths,
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
