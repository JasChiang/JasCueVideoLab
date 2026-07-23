from __future__ import annotations

import hashlib
import html
import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .media import sha256_file
from .models import (
    DenseFrameCatalog,
    FeatureEditBrief,
    FrozenStrictModel,
    ModelProvenance,
    StrictModel,
    TrimIntentDecision,
    TrimIntentProposal,
)
from .music import CuePriority, LockedMusicCue, MusicMapLock
from .storage import read_json, utc_now


VISUAL_SYNC_CONTRACT_VERSION = "visual-sync-map-v1"
CUE_PLAN_CONTRACT_VERSION = "cue-plan-proposal-v1"
CUE_PLAN_LOCK_VERSION = "cue-plan-lock-v1"


class VisualSyncPriority(StrEnum):
    HARD = "hard"
    PREFERRED = "preferred"
    OPTIONAL = "optional"


class VisualSyncPoint(FrozenStrictModel):
    visual_event_id: str = Field(pattern=r"^vs-[0-9]{4}$")
    feature_id: str | None = None
    phase: Literal[
        "timeline_start",
        "chapter_start",
        "cut",
        "action_apex",
        "contact",
        "reveal",
        "ui_change",
        "text",
        "punch_in",
        "hold_start",
        "setup_start",
        "action_start",
        "result_start",
        "reset_start",
        "ending_pose",
    ]
    sync_mode: Literal["hard", "soft", "structural"] = "soft"
    project_time_ms: int = Field(ge=0)
    flex_before_ms: int = Field(default=0, ge=0, le=10_000)
    flex_after_ms: int = Field(default=0, ge=0, le=10_000)
    priority: VisualSyncPriority
    allowed_cue_kinds: tuple[
        Literal[
            "section_boundary",
            "downbeat",
            "beat",
            "accent",
            "ending_hit",
        ],
        ...,
    ]
    evidence_refs: tuple[str, ...] = ()
    semantic_description: str = ""
    editorial_note: str = ""

    @model_validator(mode="after")
    def validate_allowed_cues(self) -> "VisualSyncPoint":
        if not self.allowed_cue_kinds:
            raise ValueError("visual sync point must allow at least one cue kind")
        return self


class VisualSyncMap(StrictModel):
    contract_version: Literal["visual-sync-map-v1"] = VISUAL_SYNC_CONTRACT_VERSION
    project_id: str
    aspect_ratio: Literal["16:9", "9:16"]
    render_manifest_path: str
    render_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_kind: Literal["render_manifest", "editorial_brief"] = "render_manifest"
    project_duration_ms: int = Field(gt=0)
    timing_basis: Literal[
        "rendered_segment_duration_ms",
        "editorial_brief_target_duration_ms",
    ] = (
        "rendered_segment_duration_ms"
    )
    flexibility_authorization: Literal[
        "read_only_boundaries",
        "operator_authorized_default_flex",
        "evidence_authored",
    ]
    points: list[VisualSyncPoint]
    uncertainties: list[str] = Field(default_factory=list)
    generated_at: str

    @model_validator(mode="after")
    def validate_points(self) -> "VisualSyncMap":
        times = [point.project_time_ms for point in self.points]
        ids = [point.visual_event_id for point in self.points]
        if len(ids) != len(set(ids)):
            raise ValueError("visual sync point IDs must be unique")
        if times != sorted(times):
            raise ValueError("visual sync points must be chronological")
        if any(time > self.project_duration_ms for time in times):
            raise ValueError("visual sync point lies outside the project timeline")
        if self.flexibility_authorization == "read_only_boundaries" and any(
            point.flex_before_ms or point.flex_after_ms for point in self.points
        ):
            raise ValueError("read-only visual boundaries cannot carry edit flexibility")
        return self


def derive_brief_visual_sync_map(
    brief_path: Path,
    *,
    aspect_ratio: Literal["16:9", "9:16"],
    default_flex_ms: int = 3_000,
    target_duration_ms: int | None = None,
) -> VisualSyncMap:
    """Create pre-selection visual intents so music can be planned first.

    These are editorial targets, not claims about observed media. Gemini may
    pair them with musical roles before candidate selection, while exact
    source evidence remains the responsibility of the later feature planner.
    """
    if not 0 <= default_flex_ms <= 10_000:
        raise ValueError("default_flex_ms must be between 0 and 10000")
    path = brief_path.expanduser().resolve(strict=True)
    brief = FeatureEditBrief.model_validate(read_json(path))
    original_duration_ms = round(
        sum(chapter.target_duration_seconds for chapter in brief.chapters) * 1000
    )
    project_duration_ms = target_duration_ms or original_duration_ms
    if not 60_000 <= project_duration_ms <= 90_000:
        raise ValueError("music-first project duration must remain between 60 and 90 seconds")
    minimum = len(brief.chapters) * 3_000
    maximum = len(brief.chapters) * 10_000
    if not minimum <= project_duration_ms <= maximum:
        raise ValueError("music-first duration cannot satisfy per-chapter duration limits")
    duration_scale = project_duration_ms / original_duration_ms
    elapsed = 0
    points: list[VisualSyncPoint] = []
    for index, chapter in enumerate(brief.chapters):
        phase: Literal["timeline_start", "chapter_start"] = (
            "timeline_start" if index == 0 else "chapter_start"
        )
        points.append(
            VisualSyncPoint(
                visual_event_id=f"vs-{len(points) + 1:04d}",
                feature_id=chapter.feature_id,
                phase=phase,
                sync_mode="structural" if index == 0 else "soft",
                project_time_ms=elapsed,
                flex_before_ms=0 if index == 0 else default_flex_ms,
                flex_after_ms=0 if index == 0 else default_flex_ms,
                priority=(
                    VisualSyncPriority.HARD
                    if index == 0
                    else VisualSyncPriority.PREFERRED
                ),
                allowed_cue_kinds=(
                    ("section_boundary", "downbeat", "accent")
                    if index == 0
                    else ("section_boundary", "downbeat", "accent", "beat")
                ),
                evidence_refs=(f"editorial-brief:{sha256_file(path)}",),
                semantic_description=(
                    chapter.title
                    + (
                        " — " + "; ".join(chapter.detail_lines)
                        if chapter.detail_lines
                        else ""
                    )
                ),
                editorial_note=(
                    "Pre-selection editorial intent only; it is not observed "
                    "visual evidence and cannot authorize source geometry."
                ),
            )
        )
        elapsed += round(chapter.target_duration_seconds * 1000 * duration_scale)
    elapsed = project_duration_ms
    points.append(
        VisualSyncPoint(
            visual_event_id=f"vs-{len(points) + 1:04d}",
            feature_id=brief.chapters[-1].feature_id,
            phase="ending_pose",
            sync_mode="hard",
            project_time_ms=elapsed,
            flex_before_ms=default_flex_ms,
            flex_after_ms=default_flex_ms,
            priority=VisualSyncPriority.HARD,
            allowed_cue_kinds=(
                "ending_hit",
                "section_boundary",
                "downbeat",
                "accent",
            ),
            evidence_refs=(f"editorial-brief:{sha256_file(path)}",),
            semantic_description="Resolve the editorial arc and preserve a closing hold.",
            editorial_note="Pre-selection desired ending; source evidence is resolved later.",
        )
    )
    return VisualSyncMap(
        project_id=brief.project_id,
        aspect_ratio=aspect_ratio,
        render_manifest_path=str(path),
        render_manifest_sha256=sha256_file(path),
        source_kind="editorial_brief",
        project_duration_ms=elapsed,
        timing_basis="editorial_brief_target_duration_ms",
        flexibility_authorization=(
            "operator_authorized_default_flex"
            if default_flex_ms
            else "read_only_boundaries"
        ),
        points=points,
        uncertainties=[
            "This map precedes media selection; every selected clip still requires "
            "independent visual evidence, trim, and geometry validation."
        ],
        generated_at=utc_now(),
    )


def apply_music_first_cue_lock(
    brief: FeatureEditBrief,
    *,
    visual_map: VisualSyncMap,
    cue_lock: CuePlanLock,
) -> FeatureEditBrief:
    """Project an approved pre-selection cue schedule into chapter durations."""
    if visual_map.source_kind != "editorial_brief":
        raise ValueError("music-first feature planning requires an editorial-brief sync map")
    bound_brief_path = Path(visual_map.render_manifest_path).expanduser().resolve(
        strict=True
    )
    if visual_map.render_manifest_sha256 != sha256_file(bound_brief_path):
        raise ValueError("music-first editorial brief lineage no longer matches disk")
    if FeatureEditBrief.model_validate(read_json(bound_brief_path)) != brief:
        raise ValueError("music-first cue plan is bound to a different editorial brief")
    if cue_lock.plan.visual_sync_map_sha256 != sha256_file(
        Path(cue_lock.plan.visual_sync_map_path).expanduser().resolve(strict=True)
    ):
        raise ValueError("CuePlan visual map lineage no longer matches disk")
    if visual_map.project_id != brief.project_id:
        raise ValueError("music-first visual map belongs to another project")
    aligned = {
        row.visual_event_id: row.proposed_project_time_ms
        for row in cue_lock.plan.alignments
        if row.status == "aligned" and row.proposed_project_time_ms is not None
    }
    boundaries: list[int] = []
    for point in visual_map.points:
        if point.phase in {"timeline_start", "chapter_start", "ending_pose"}:
            boundaries.append(aligned.get(point.visual_event_id, point.project_time_ms))
    if len(boundaries) != len(brief.chapters) + 1:
        raise ValueError("music-first cue plan does not cover all editorial boundaries")
    if boundaries[0] != 0 or boundaries != sorted(boundaries):
        raise ValueError("music-first cue boundaries must be chronological from zero")
    durations = [
        (end - start) / 1000
        for start, end in zip(boundaries, boundaries[1:])
    ]
    if any(not 3.0 <= duration <= 10.0 for duration in durations):
        raise ValueError("music-first cue schedule violates 3-10 second chapter limits")
    return brief.model_copy(
        update={
            "target_duration_seconds": sum(durations),
            "chapters": [
                chapter.model_copy(update={"target_duration_seconds": duration})
                for chapter, duration in zip(brief.chapters, durations, strict=True)
            ],
        }
    )


class CueAlignment(FrozenStrictModel):
    visual_event_id: str
    status: Literal["aligned", "unmatched"]
    sync_mode: Literal["hard", "soft", "structural"]
    original_project_time_ms: int = Field(ge=0)
    proposed_project_time_ms: int | None = Field(default=None, ge=0)
    delta_ms: int | None = None
    music_cue_id: str | None = None
    music_cue_kind: str | None = None
    music_sample_index: int | None = Field(default=None, ge=0)
    alignment_score: float | None = None
    within_authorized_window: bool
    reason: str
    evidence_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_status(self) -> "CueAlignment":
        aligned_values = (
            self.proposed_project_time_ms,
            self.delta_ms,
            self.music_cue_id,
            self.music_cue_kind,
            self.music_sample_index,
            self.alignment_score,
        )
        if self.status == "aligned":
            if any(value is None for value in aligned_values):
                raise ValueError("aligned cue rows require complete cue evidence")
            if not self.within_authorized_window:
                raise ValueError("aligned cue must remain inside the authorized window")
        elif any(value is not None for value in aligned_values):
            raise ValueError("unmatched cue rows cannot claim a music target")
        return self


class MusicSectionInterpretation(FrozenStrictModel):
    section_id: str
    role: Literal[
        "opening",
        "build",
        "peak",
        "breathing_space",
        "closing",
        "neutral",
    ]
    energy_level: Literal["low", "medium", "high"]
    motion_character: Literal[
        "steady",
        "accelerating",
        "decelerating",
        "pulsing",
        "sparse",
        "free_form",
    ]
    emotional_character: tuple[str, ...]
    recommended_visual_roles: tuple[str, ...]
    audible_evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class SemanticCuePairing(FrozenStrictModel):
    visual_event_id: str
    preferred_cue_ids: tuple[str, ...] = Field(min_length=1, max_length=5)
    sync_mode: Literal["hard", "soft", "structural"]
    rhythmic_intent: Literal[
        "section_transition",
        "strong_hit",
        "subtle_accent",
        "breathing_space",
        "ending_resolution",
    ]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class SemanticMusicPairingProposal(StrictModel):
    contract_version: Literal["semantic-music-pairing-v1"] = (
        "semantic-music-pairing-v1"
    )
    music_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    music_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    visual_sync_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_strategy: str
    section_interpretations: list[MusicSectionInterpretation]
    pairings: list[SemanticCuePairing]
    uncertainties: list[str]
    requires_human_review: Literal[True] = True
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_unique_references(self) -> "SemanticMusicPairingProposal":
        section_ids = [item.section_id for item in self.section_interpretations]
        visual_ids = [item.visual_event_id for item in self.pairings]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("semantic music section references must be unique")
        if len(visual_ids) != len(set(visual_ids)):
            raise ValueError("semantic visual event references must be unique")
        return self

class CuePlanProposal(StrictModel):
    contract_version: Literal["cue-plan-proposal-v1"] = CUE_PLAN_CONTRACT_VERSION
    plan_id: str = Field(pattern=r"^cue-plan-[0-9a-f]{12}$")
    preset: Literal["narrative", "balanced", "montage"]
    music_lock_path: str
    music_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    music_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    visual_sync_map_path: str
    visual_sync_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_pairing_path: str | None = None
    semantic_pairing_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    semantic_pairing_used: bool = False
    project_duration_ms: int = Field(gt=0)
    music_duration_ms: int = Field(gt=0)
    alignments: list[CueAlignment]
    aligned_count: int = Field(ge=0)
    unmatched_count: int = Field(ge=0)
    hard_unmatched_count: int = Field(ge=0)
    changes_applied: Literal[False] = False
    requires_human_review: Literal[True] = True
    uncertainties: list[str]
    generated_at: str

    @model_validator(mode="after")
    def validate_counts(self) -> "CuePlanProposal":
        aligned = sum(row.status == "aligned" for row in self.alignments)
        unmatched = len(self.alignments) - aligned
        if aligned != self.aligned_count or unmatched != self.unmatched_count:
            raise ValueError("cue plan summary counts do not match alignments")
        if self.semantic_pairing_used != bool(
            self.semantic_pairing_path and self.semantic_pairing_sha256
        ):
            raise ValueError("semantic pairing usage and lineage fields disagree")
        return self


class CuePlanReview(StrictModel):
    contract_version: Literal["cue-plan-review-v1"] = "cue-plan-review-v1"
    cue_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer: str = Field(min_length=1)
    reviewed_at: str
    decision: Literal["approved", "rejected"]
    notes: str = ""


class CuePlanLock(StrictModel):
    contract_version: Literal["cue-plan-lock-v1"] = CUE_PLAN_LOCK_VERSION
    cue_plan_path: str
    cue_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review: CuePlanReview
    plan: CuePlanProposal
    definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_lock(self) -> "CuePlanLock":
        if self.review.decision != "approved":
            raise ValueError("cue plan lock requires explicit human approval")
        if self.review.cue_plan_sha256 != self.cue_plan_sha256:
            raise ValueError("cue plan review is not bound to this plan")
        return self


def _canonical_hash(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude={"definition_sha256"})
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _trim_visual_sync_points(
    chapter: dict[str, Any],
    *,
    project_segment_start_ms: int,
    segment_duration_ms: int,
    default_flex_ms: int,
) -> list[VisualSyncPoint]:
    decision_value = chapter.get("trim_decision_path")
    if not isinstance(decision_value, str) or not decision_value:
        return []
    decision_path = Path(decision_value).expanduser().resolve(strict=True)
    decision = TrimIntentDecision.model_validate(read_json(decision_path))
    if (
        decision.approval_status != "approved"
        or decision.requires_human_review
        or decision.human_review is None
        or decision.source_in_ms is None
    ):
        raise ValueError(
            "VisualSyncMap only accepts phase evidence from a human-approved TrimIntentDecision"
        )
    proposal_path = Path(decision.proposal_path).expanduser().resolve(strict=True)
    catalog_path = Path(decision.catalog_path).expanduser().resolve(strict=True)
    proposal = TrimIntentProposal.model_validate(read_json(proposal_path))
    catalog = DenseFrameCatalog.model_validate(read_json(catalog_path))
    if (
        proposal.source_asset_id != decision.source_asset_id
        or proposal.event_id != decision.event_id
        or catalog.source_asset_id != decision.source_asset_id
        or catalog.event_id != decision.event_id
    ):
        raise ValueError("approved Trim Intent lineage disagrees across artifacts")
    frames = {frame.frame_id: frame for frame in catalog.frames}
    mapping: dict[
        str,
        tuple[
            Literal[
                "setup_start",
                "action_start",
                "result_start",
                "hold_start",
                "reset_start",
            ],
            VisualSyncPriority,
            Literal["hard", "soft", "structural"],
            tuple[
                Literal[
                    "section_boundary",
                    "downbeat",
                    "beat",
                    "accent",
                    "ending_hit",
                ],
                ...,
            ],
        ],
    ] = {
        "setup_start": (
            "setup_start",
            VisualSyncPriority.OPTIONAL,
            "structural",
            ("section_boundary", "downbeat"),
        ),
        "action_start": (
            "action_start",
            VisualSyncPriority.PREFERRED,
            "soft",
            ("downbeat", "accent", "beat"),
        ),
        "result_start": (
            "result_start",
            VisualSyncPriority.PREFERRED,
            "soft",
            ("section_boundary", "downbeat", "accent"),
        ),
        "hold_start": (
            "hold_start",
            VisualSyncPriority.OPTIONAL,
            "soft",
            ("downbeat", "accent", "beat"),
        ),
        "reset_start": (
            "reset_start",
            VisualSyncPriority.OPTIONAL,
            "structural",
            ("section_boundary", "downbeat"),
        ),
    }
    result: list[VisualSyncPoint] = []
    for selection in proposal.selections:
        contract = mapping.get(selection.phase)
        if contract is None:
            continue
        frame = frames.get(selection.frame_id)
        if frame is None:
            raise ValueError(
                f"approved Trim Intent references unknown dense frame {selection.frame_id}"
            )
        local_ms = frame.frame_time_ms - decision.source_in_ms
        if not 0 <= local_ms <= segment_duration_ms:
            continue
        phase, priority, sync_mode, allowed = contract
        result.append(
            VisualSyncPoint(
                visual_event_id="vs-0001",
                feature_id=str(chapter.get("feature_id") or "") or None,
                phase=phase,
                sync_mode=sync_mode,
                project_time_ms=project_segment_start_ms + local_ms,
                flex_before_ms=default_flex_ms,
                flex_after_ms=default_flex_ms,
                priority=priority,
                allowed_cue_kinds=allowed,
                evidence_refs=(
                    f"trim-decision:{sha256_file(decision_path)}",
                    f"trim-proposal:{sha256_file(proposal_path)}",
                    f"dense-frame:{frame.frame_id}:{frame.frame_hash}",
                ),
                semantic_description=(
                    f"{selection.phase}: {proposal.observed_phase_evidence}"
                ),
                editorial_note=(
                    "Exact dense-frame phase from a human-approved Trim Intent. "
                    "A non-zero flex still requires downstream source-handle and "
                    "geometry validation before changing the edit."
                ),
            )
        )
    return result


def derive_visual_sync_map(
    render_manifest_path: Path,
    *,
    aspect_ratio: Literal["16:9", "9:16"],
    default_flex_ms: int = 0,
) -> VisualSyncMap:
    if not 0 <= default_flex_ms <= 10_000:
        raise ValueError("default_flex_ms must be between 0 and 10000")
    path = render_manifest_path.expanduser().resolve(strict=True)
    manifest = read_json(path)
    key = "horizontal" if aspect_ratio == "16:9" else "vertical"
    timeline = manifest.get(key)
    if not isinstance(timeline, dict) or timeline.get("status") not in {
        None,
        "rendered",
    }:
        raise ValueError(f"render manifest does not contain a rendered {aspect_ratio} cut")
    chapters = timeline.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("rendered timeline has no chapter entries")
    durations: list[int] = []
    for chapter in chapters:
        duration_value = chapter.get("duration_ms")
        if duration_value is None:
            source_in = chapter.get("source_in_ms")
            source_out = chapter.get("source_out_ms")
            if source_in is None or source_out is None:
                raise ValueError(
                    "render manifest chapter lacks duration_ms and source in/out"
                )
            duration_value = int(source_out) - int(source_in)
        durations.append(int(duration_value))
    if any(duration <= 0 for duration in durations):
        raise ValueError("render manifest contains a non-positive segment duration")
    points = [
        VisualSyncPoint(
            visual_event_id="vs-0001",
            feature_id=str(chapters[0].get("feature_id") or "") or None,
            phase="timeline_start",
            sync_mode="structural",
            project_time_ms=0,
            priority=VisualSyncPriority.HARD,
            allowed_cue_kinds=("section_boundary", "downbeat", "accent"),
            evidence_refs=(f"render-manifest:{sha256_file(path)}",),
            semantic_description=str(
                chapters[0].get("semantic_intent")
                or chapters[0].get("observed_visual_evidence")
                or "Start of the picture edit."
            ),
            editorial_note="Start of the approved picture-edit sequence.",
        )
    ]
    elapsed = 0
    for index, (chapter, duration) in enumerate(
        zip(chapters, durations, strict=True), start=1
    ):
        points.extend(
            _trim_visual_sync_points(
                chapter,
                project_segment_start_ms=elapsed,
                segment_duration_ms=duration,
                default_flex_ms=default_flex_ms,
            )
        )
        elapsed += duration
        if index < len(chapters):
            next_chapter = chapters[index]
            points.append(
                VisualSyncPoint(
                    visual_event_id=f"vs-{len(points) + 1:04d}",
                    feature_id=str(next_chapter.get("feature_id") or "") or None,
                    phase="cut",
                    sync_mode="soft",
                    project_time_ms=elapsed,
                    flex_before_ms=default_flex_ms,
                    flex_after_ms=default_flex_ms,
                    priority=VisualSyncPriority.PREFERRED,
                    allowed_cue_kinds=("section_boundary", "downbeat", "accent", "beat"),
                    evidence_refs=(
                        f"previous-feature:{chapter.get('feature_id')}",
                        f"next-feature:{next_chapter.get('feature_id')}",
                    ),
                    semantic_description=(
                        "Transition from "
                        + str(
                            chapter.get("semantic_intent")
                            or chapter.get("observed_visual_evidence")
                            or chapter.get("feature_id")
                        )
                        + " to "
                        + str(
                            next_chapter.get("semantic_intent")
                            or next_chapter.get("observed_visual_evidence")
                            or next_chapter.get("feature_id")
                        )
                    ),
                    editorial_note=(
                        "Existing rendered chapter boundary. A non-zero window is "
                        "operator authorization for a downstream re-edit proposal, "
                        "not proof that action-safe source handles exist."
                    ),
                )
            )
    points.append(
        VisualSyncPoint(
            visual_event_id=f"vs-{len(points) + 1:04d}",
            feature_id=str(chapters[-1].get("feature_id") or "") or None,
            phase="ending_pose",
            sync_mode="hard",
            project_time_ms=elapsed,
            flex_before_ms=default_flex_ms,
            flex_after_ms=default_flex_ms,
            priority=VisualSyncPriority.HARD,
            allowed_cue_kinds=("ending_hit", "section_boundary", "downbeat", "accent"),
            evidence_refs=(f"last-feature:{chapters[-1].get('feature_id')}",),
            semantic_description=str(
                chapters[-1].get("semantic_intent")
                or chapters[-1].get("observed_visual_evidence")
                or "End of the picture edit."
            ),
            editorial_note="End of the current picture edit; preserve an intentional visual hold.",
        )
    )
    points = [
        point.model_copy(update={"visual_event_id": f"vs-{index:04d}"})
        for index, point in enumerate(
            sorted(
                points,
                key=lambda point: (
                    point.project_time_ms,
                    point.phase,
                    point.feature_id or "",
                ),
            ),
            start=1,
        )
    ]
    return VisualSyncMap(
        project_id=str(manifest.get("project_id") or "unknown-project"),
        aspect_ratio=aspect_ratio,
        render_manifest_path=str(path),
        render_manifest_sha256=sha256_file(path),
        project_duration_ms=elapsed,
        flexibility_authorization=(
            "operator_authorized_default_flex"
            if default_flex_ms
            else "read_only_boundaries"
        ),
        points=points,
        generated_at=utc_now(),
    )


def _preset_cue_weight(
    preset: Literal["narrative", "balanced", "montage"], cue: LockedMusicCue
) -> float:
    base = {
        CuePriority.HARD: 1.5,
        CuePriority.PREFERRED: 1.0,
        CuePriority.OPTIONAL: 0.45,
    }[cue.priority]
    kind_weight = {
        "narrative": {
            "section_boundary": 1.4,
            "ending_hit": 1.4,
            "downbeat": 1.0,
            "accent": 0.75,
            "beat": 0.2,
        },
        "balanced": {
            "section_boundary": 1.3,
            "ending_hit": 1.4,
            "downbeat": 1.15,
            "accent": 1.0,
            "beat": 0.5,
        },
        "montage": {
            "section_boundary": 1.15,
            "ending_hit": 1.25,
            "downbeat": 1.15,
            "accent": 1.1,
            "beat": 0.9,
        },
    }[preset][cue.kind]
    return base * kind_weight


def _visual_weight(priority: VisualSyncPriority) -> float:
    return {
        VisualSyncPriority.HARD: 2.0,
        VisualSyncPriority.PREFERRED: 1.2,
        VisualSyncPriority.OPTIONAL: 0.65,
    }[priority]


def _candidate_score(
    point: VisualSyncPoint,
    cue: LockedMusicCue,
    preset: Literal["narrative", "balanced", "montage"],
    semantic_preferences: dict[str, set[str]] | None = None,
) -> float | None:
    if cue.kind not in point.allowed_cue_kinds:
        return None
    minimum = point.project_time_ms - point.flex_before_ms
    maximum = point.project_time_ms + point.flex_after_ms
    if not minimum <= cue.time_ms <= maximum:
        return None
    span = max(1, point.flex_before_ms + point.flex_after_ms)
    distance_penalty = abs(cue.time_ms - point.project_time_ms) / span
    semantic_bonus = (
        1.25
        if semantic_preferences
        and cue.cue_id in semantic_preferences.get(point.visual_event_id, set())
        else 0.0
    )
    return (
        1.0
        + _visual_weight(point.priority)
        * _preset_cue_weight(preset, cue)
        * (0.5 + cue.strength * 0.5)
        + semantic_bonus
        - distance_penalty
    )


def _schedule(
    points: list[VisualSyncPoint],
    cues: list[LockedMusicCue],
    preset: Literal["narrative", "balanced", "montage"],
    semantic_preferences: dict[str, set[str]] | None = None,
) -> dict[int, int]:
    @lru_cache(maxsize=None)
    def solve(point_index: int, cue_index: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if point_index >= len(points):
            return 0.0, ()
        if cue_index >= len(cues):
            penalty = sum(
                3.0 if point.priority == VisualSyncPriority.HARD else 0.25
                for point in points[point_index:]
            )
            return -penalty, ()
        skip_point_score, skip_point_pairs = solve(point_index + 1, cue_index)
        if points[point_index].priority == VisualSyncPriority.HARD:
            skip_point_score -= 3.0
        elif points[point_index].priority == VisualSyncPriority.PREFERRED:
            skip_point_score -= 0.25
        skip_cue_score, skip_cue_pairs = solve(point_index, cue_index + 1)
        options = [
            (skip_point_score, skip_point_pairs),
            (skip_cue_score, skip_cue_pairs),
        ]
        match_score = _candidate_score(
            points[point_index],
            cues[cue_index],
            preset,
            semantic_preferences,
        )
        if match_score is not None:
            future_score, future_pairs = solve(point_index + 1, cue_index + 1)
            options.append(
                (
                    match_score + future_score,
                    ((point_index, cue_index), *future_pairs),
                )
            )
        return max(options, key=lambda item: (item[0], len(item[1])))

    _, pairs = solve(0, 0)
    return dict(pairs)


def plan_music_cues(
    music_lock: MusicMapLock,
    visual_map: VisualSyncMap,
    *,
    music_lock_path: Path,
    visual_sync_map_path: Path,
    preset: Literal["narrative", "balanced", "montage"] = "balanced",
    semantic_pairing: SemanticMusicPairingProposal | None = None,
    semantic_pairing_path: Path | None = None,
) -> CuePlanProposal:
    resolved_music = music_lock_path.expanduser().resolve(strict=True)
    resolved_visual = visual_sync_map_path.expanduser().resolve(strict=True)
    saved_music_lock = MusicMapLock.model_validate(read_json(resolved_music))
    saved_visual_map = VisualSyncMap.model_validate(read_json(resolved_visual))
    if saved_music_lock != music_lock:
        raise ValueError("in-memory MusicMap lock differs from the saved lock artifact")
    if saved_visual_map != visual_map:
        raise ValueError("in-memory VisualSyncMap differs from the saved map artifact")
    if music_lock.review.decision != "approved":
        raise ValueError("cue scheduling requires an approved MusicMap lock")
    semantic_preferences: dict[str, set[str]] | None = None
    semantic_sync_modes: dict[str, Literal["hard", "soft", "structural"]] = {}
    resolved_semantic: Path | None = None
    if bool(semantic_pairing) != bool(semantic_pairing_path):
        raise ValueError(
            "semantic pairing proposal and saved artifact path must be supplied together"
        )
    if semantic_pairing is not None and semantic_pairing_path is not None:
        resolved_semantic = semantic_pairing_path.expanduser().resolve(strict=True)
        saved_semantic = SemanticMusicPairingProposal.model_validate(
            read_json(resolved_semantic)
        )
        if saved_semantic != semantic_pairing:
            raise ValueError(
                "in-memory semantic pairing differs from the saved proposal artifact"
            )
        if (
            semantic_pairing.music_id != music_lock.music_id
            or semantic_pairing.music_definition_sha256
            != music_lock.definition_sha256
            or semantic_pairing.visual_sync_map_sha256
            != sha256_file(resolved_visual)
        ):
            raise ValueError("semantic pairing does not match the locked music and visual map")
        known_visual_ids = {point.visual_event_id for point in visual_map.points}
        known_cue_ids = {cue.cue_id for cue in music_lock.cues}
        unknown_visual = sorted(
            {
                pairing.visual_event_id
                for pairing in semantic_pairing.pairings
                if pairing.visual_event_id not in known_visual_ids
            }
        )
        unknown_cues = sorted(
            {
                cue_id
                for pairing in semantic_pairing.pairings
                for cue_id in pairing.preferred_cue_ids
                if cue_id not in known_cue_ids
            }
        )
        if unknown_visual or unknown_cues:
            raise ValueError(
                "semantic pairing referenced unknown IDs: "
                f"visual={unknown_visual}, cues={unknown_cues}"
            )
        semantic_preferences = {
            pairing.visual_event_id: set(pairing.preferred_cue_ids)
            for pairing in semantic_pairing.pairings
        }
        semantic_sync_modes = {
            pairing.visual_event_id: pairing.sync_mode
            for pairing in semantic_pairing.pairings
        }
    selected = _schedule(
        visual_map.points, music_lock.cues, preset, semantic_preferences
    )
    alignments: list[CueAlignment] = []
    for index, point in enumerate(visual_map.points):
        cue_index = selected.get(index)
        if cue_index is None:
            alignments.append(
                CueAlignment(
                    visual_event_id=point.visual_event_id,
                    status="unmatched",
                    sync_mode=semantic_sync_modes.get(
                        point.visual_event_id, point.sync_mode
                    ),
                    original_project_time_ms=point.project_time_ms,
                    within_authorized_window=False,
                    reason=(
                        "No compatible locked music cue exists inside the "
                        "authorized visual timing window."
                    ),
                    evidence_refs=point.evidence_refs,
                )
            )
            continue
        cue = music_lock.cues[cue_index]
        score = _candidate_score(point, cue, preset, semantic_preferences)
        assert score is not None
        alignments.append(
            CueAlignment(
                visual_event_id=point.visual_event_id,
                status="aligned",
                sync_mode=semantic_sync_modes.get(
                    point.visual_event_id, point.sync_mode
                ),
                original_project_time_ms=point.project_time_ms,
                proposed_project_time_ms=cue.time_ms,
                delta_ms=cue.time_ms - point.project_time_ms,
                music_cue_id=cue.cue_id,
                music_cue_kind=cue.kind,
                music_sample_index=cue.sample_index,
                alignment_score=round(score, 6),
                within_authorized_window=True,
                reason=(
                    "Global order-preserving scheduler selected this cue inside "
                    "the explicit timing window. No edit has been applied."
                ),
                evidence_refs=point.evidence_refs,
            )
        )
    aligned_count = sum(row.status == "aligned" for row in alignments)
    hard_by_id = {
        point.visual_event_id
        for point in visual_map.points
        if point.priority == VisualSyncPriority.HARD
    }
    hard_unmatched = sum(
        row.status == "unmatched" and row.visual_event_id in hard_by_id
        for row in alignments
    )
    identity = {
        "music_definition_sha256": music_lock.definition_sha256,
        "visual_sync_map_sha256": sha256_file(resolved_visual),
        "semantic_pairing_sha256": (
            sha256_file(resolved_semantic) if resolved_semantic is not None else None
        ),
        "preset": preset,
    }
    plan_hash = _canonical_hash(identity)
    return CuePlanProposal(
        plan_id=f"cue-plan-{plan_hash[:12]}",
        preset=preset,
        music_lock_path=str(resolved_music),
        music_lock_sha256=sha256_file(resolved_music),
        music_definition_sha256=music_lock.definition_sha256,
        visual_sync_map_path=str(resolved_visual),
        visual_sync_map_sha256=sha256_file(resolved_visual),
        semantic_pairing_path=(
            str(resolved_semantic) if resolved_semantic is not None else None
        ),
        semantic_pairing_sha256=(
            sha256_file(resolved_semantic) if resolved_semantic is not None else None
        ),
        semantic_pairing_used=resolved_semantic is not None,
        project_duration_ms=visual_map.project_duration_ms,
        music_duration_ms=music_lock.duration_ms,
        alignments=alignments,
        aligned_count=aligned_count,
        unmatched_count=len(alignments) - aligned_count,
        hard_unmatched_count=hard_unmatched,
        uncertainties=[
            "CuePlan is a proposal and never mutates approved source trims, identity locks, geometry, or rendered media.",
            "Applying a non-zero delta requires downstream validation of action completeness, source handles, shot boundaries, geometry, and final hold duration.",
        ],
        generated_at=utc_now(),
    )


def review_cue_plan(
    plan: CuePlanProposal,
    *,
    cue_plan_path: Path,
    reviewer: str,
    decision: Literal["approved", "rejected"],
    notes: str = "",
) -> tuple[CuePlanReview, CuePlanLock | None]:
    resolved = cue_plan_path.expanduser().resolve(strict=True)
    digest = sha256_file(resolved)
    saved_plan = CuePlanProposal.model_validate(read_json(resolved))
    if saved_plan != plan:
        raise ValueError("in-memory CuePlan differs from the saved proposal artifact")
    review = CuePlanReview(
        cue_plan_sha256=digest,
        reviewer=reviewer,
        reviewed_at=utc_now(),
        decision=decision,
        notes=notes,
    )
    if decision == "rejected":
        return review, None
    definition = {
        "cue_plan_sha256": digest,
        "review": review.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
    }
    return review, CuePlanLock(
        cue_plan_path=str(resolved),
        cue_plan_sha256=digest,
        review=review,
        plan=plan,
        definition_sha256=_canonical_hash(definition),
    )


def render_cue_review(
    *,
    music_path: Path,
    video_path: Path | None,
    visual_map: VisualSyncMap,
    plan: CuePlanProposal,
    output_path: Path,
) -> Path:
    music_uri = music_path.expanduser().resolve(strict=True).as_uri()
    video = (
        f'<video controls preload="metadata" src="{html.escape(video_path.expanduser().resolve(strict=True).as_uri())}"></video>'
        if video_path is not None
        else "<p>No review video supplied.</p>"
    )
    points = {point.visual_event_id: point for point in visual_map.points}
    rows = []
    for alignment in plan.alignments:
        point = points[alignment.visual_event_id]
        proposed = (
            f"{alignment.proposed_project_time_ms / 1000:.3f}s"
            if alignment.proposed_project_time_ms is not None
            else "—"
        )
        delta = f"{alignment.delta_ms:+d}ms" if alignment.delta_ms is not None else "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(point.visual_event_id)}</td>"
            f"<td>{html.escape(point.phase)}</td>"
            f"<td>{point.project_time_ms / 1000:.3f}s</td>"
            f"<td>{proposed}</td>"
            f"<td>{delta}</td>"
            f"<td>{html.escape(alignment.music_cue_kind or 'unmatched')}</td>"
            f"<td>{html.escape(point.priority.value)}</td>"
            f"<td>{html.escape(alignment.reason)}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>Music Cue Review</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#111;color:#eee;margin:24px}}
audio,video{{width:100%;max-width:1000px;background:#000;margin:8px 0 20px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #444;padding:8px;text-align:left}}
th{{background:#242424}}.warning{{color:#ffd166}}
</style></head><body>
<h1>Music Cue Review</h1>
<p class="warning">這是卡點 proposal；尚未改動已核准剪輯或渲染影片。</p>
<p>Preset: {html.escape(plan.preset)} · aligned {plan.aligned_count} · unmatched {plan.unmatched_count} · hard unmatched {plan.hard_unmatched_count}</p>
<h2>Music</h2><audio controls preload="metadata" src="{html.escape(music_uri)}"></audio>
<h2>Picture edit</h2>{video}
<h2>Proposed alignment</h2>
<table><thead><tr><th>ID</th><th>visual phase</th><th>original</th><th>proposed</th><th>delta</th><th>music cue</th><th>priority</th><th>reason</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path.resolve()
