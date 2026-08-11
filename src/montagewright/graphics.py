"""Editorial graphics as an independent, editable timeline track.

The words, the design and the timing are different decisions.  A model may
suggest that a product name belongs over a shot; it may not silently turn its
memory of that product into copy.  ``CopyFact`` keeps the exact words and
their provenance, ``GraphicCue`` points at those words, and this module turns
an approved plan into deterministic pixels.

The clean master is never overwritten.  Graphics are a second deliverable in
the same way burned subtitles are: removable until somebody explicitly asks
to composite them.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from montagewright.schema import Local

SourceKind = Literal[
    "user", "brief_exact", "brief_candidate", "onscreen", "transcript",
    "model_draft",
]
GraphicKind = Literal[
    "opening_title", "chapter", "product_name", "feature", "callout",
    "end_card",
]
MotionPreset = Literal["none", "fade", "rise", "slide_left", "slide_right"]
PositionHint = Literal[
    "auto", "manual", "top", "upper_left", "upper_right", "lower_left",
    "lower_right", "center",
]
CompositionIntent = Literal[
    "auto", "negative_space", "avoid_subject", "overlap_subject",
    "foreground_plate",
]
BackgroundTreatment = Literal["auto", "none", "plate"]
AdaptiveMode = Literal["auto", "strict"]
CueStatus = Literal["draft", "approved"]
TextAlignment = Literal["auto", "left", "center", "right"]
StylePreset = Literal[
    "custom", "clean", "outlined", "soft_shadow", "colour_label",
    "tech_frame", "bold_pop", "editorial_minimal", "youtube_pop",
    "magazine_story", "social_sticker", "broadcast_info",
    "cinematic_title", "sports_energy", "soft_lifestyle",
]
SurfaceTreatment = Literal[
    "template", "solid", "pill", "split", "ribbon", "sticker",
    "highlight", "outline", "glass", "editorial",
]

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
GRAPHICS_RENDERER_VERSION = 9
GRAPHIC_PRESET_REGISTRY_VERSION = 1


# A family is a curated starting point, never a sealed template. The model,
# Web editor and API all materialise these same defaults and may then override
# any individual renderer-safe field. Keeping this registry in Python avoids
# the old split where the browser knew what a preset meant but CLI/API did not.
GRAPHIC_PRESET_SPECS: dict[str, dict] = {
    "clean": {
        "label": "純字", "description": "無底板的安靜純字",
        "style": {"stroke_width": 0, "shadow_opacity": 0,
                  "plate_border_width": 0},
        "cue": {"background": "none"},
    },
    "outlined": {
        "label": "台灣 YouTube 描邊", "description": "高辨識粗描邊口播重點",
        "style": {"stroke_width": 5, "stroke_color": "#000000",
                  "shadow_opacity": 0, "plate_border_width": 0},
        "cue": {"background": "none"},
    },
    "soft_shadow": {
        "label": "柔陰影", "description": "不搶畫面的柔和浮字",
        "style": {"stroke_width": 1, "stroke_color": "#000000",
                  "shadow_opacity": 185, "shadow_blur": 9,
                  "shadow_offset_x": 3, "shadow_offset_y": 5,
                  "plate_border_width": 0},
        "cue": {"background": "none"},
    },
    "colour_label": {
        "label": "彩色標籤", "description": "緊湊的產品與人物標籤",
        "style": {"stroke_width": 0, "shadow_opacity": 0,
                  "plate_border_width": 0, "plate_alpha": 232,
                  "corner_radius": 18, "padding_x": 32, "padding_y": 20},
        "cue": {"background": "plate"},
    },
    "tech_frame": {
        "label": "科技細框", "description": "清晰的規格與介面資訊框",
        "style": {"stroke_width": 2, "stroke_color": "#07101F",
                  "shadow_opacity": 140, "shadow_blur": 8,
                  "shadow_offset_y": 4, "plate_alpha": 128,
                  "plate_border_width": 2, "corner_radius": 8},
        "cue": {"background": "plate"},
    },
    "bold_pop": {
        "label": "粗框立體字", "description": "娛樂感強烈的重點字",
        "style": {"stroke_width": 7, "stroke_color": "#111111",
                  "shadow_opacity": 210, "shadow_blur": 0,
                  "shadow_offset_x": 6, "shadow_offset_y": 7,
                  "primary_scale": 1.18, "plate_border_width": 0},
        "cue": {"background": "none"},
    },
    "editorial_minimal": {
        "label": "極簡編輯", "description": "留白、細線與克制層級",
        "best_for": "紀錄片、旅遊、章節與安靜的產品敘事",
        "style": {"surface": "editorial", "primary_color": "#171717",
                  "secondary_color": "#303030",
                  "surface_secondary_color": "#D59A21", "padding_x": 34,
                  "padding_y": 22, "primary_scale": 1.02,
                  "secondary_scale": .9, "line_spacing": 1.08,
                  "block_gap_scale": 1.15,
                  "shadow_opacity": 0, "entrance_seconds": .32,
                  "exit_seconds": .22, "motion_distance": 18},
        "cue": {"background": "none",
                "motion": "fade", "composition": "negative_space"},
    },
    "youtube_pop": {
        "label": "YouTube 強調", "description": "粗描邊與螢光筆重點",
        "best_for": "口播重點、教學結論與娛樂節奏",
        "style": {"surface": "highlight", "primary_color": "#111111",
                  "secondary_color": "#111111", "plate_color": "#FFE35A",
                  "surface_secondary_color": "#FFFFFF", "plate_alpha": 245,
                  "stroke_width": 0, "primary_scale": 1.14,
                  "padding_x": 24, "padding_y": 16,
                  "entrance_seconds": .2, "exit_seconds": .16,
                  "motion_distance": 24},
        "cue": {"background": "none",
                "motion": "rise", "composition": "negative_space"},
    },
    "magazine_story": {
        "label": "雜誌專題", "description": "暖紙底、清楚層級與編輯感",
        "best_for": "人物、文化、生活風格與專題敘事",
        "style": {"surface": "solid", "primary_color": "#181716",
                  "secondary_color": "#4B4741", "plate_color": "#F3EFE7",
                  "surface_secondary_color": "#A75C3B", "plate_alpha": 242,
                  "corner_radius": 5, "primary_scale": 1.08,
                  "secondary_scale": .82, "line_spacing": 1.16,
                  "block_gap_scale": 1.35,
                  "padding_x": 38, "padding_y": 28,
                  "max_width_scale": .92, "shadow_opacity": 75,
                  "shadow_blur": 7, "shadow_offset_y": 3,
                  "entrance_seconds": .38, "exit_seconds": .28,
                  "motion_distance": 30},
        "cue": {"background": "none",
                "motion": "slide_right", "composition": "negative_space"},
    },
    "social_sticker": {
        "label": "社群貼紙", "description": "白貼紙、硬陰影與活潑層次",
        "best_for": "短影音反應、人物標註與輕鬆內容",
        "style": {"surface": "sticker", "primary_color": "#171717",
                  "secondary_color": "#171717", "plate_color": "#FFFFFF",
                  "surface_secondary_color": "#171717", "plate_alpha": 255,
                  "plate_border_width": 3, "plate_border_color": "#171717",
                  "corner_radius": 24, "shadow_opacity": 0,
                  "primary_scale": 1.12, "entrance_seconds": .22,
                  "exit_seconds": .16, "motion_distance": 24},
        "cue": {"background": "plate",
                "motion": "rise", "composition": "negative_space"},
    },
    "broadcast_info": {
        "label": "資訊下標", "description": "中性雙層資訊板，不綁特定產業配色",
        "best_for": "姓名職稱、地點、產品名稱與兩層資料",
        "style": {"surface": "split", "primary_color": "#FFFFFF",
                  "secondary_color": "#1C1B19", "plate_color": "#1D2024",
                  "surface_secondary_color": "#EEEAE2", "plate_alpha": 245,
                  "corner_radius": 10, "padding_x": 36, "padding_y": 25,
                  "primary_scale": 1.04, "secondary_scale": .78,
                  "line_spacing": 1.12, "block_gap_scale": .75,
                  "plate_border_width": 1,
                  "plate_border_color": "#FFFFFF",
                  "entrance_seconds": .28, "exit_seconds": .2,
                  "motion_distance": 32},
        "cue": {"background": "plate",
                "motion": "slide_right", "composition": "negative_space"},
    },
    "cinematic_title": {
        "label": "電影標題", "description": "大留白、慢淡入與低調陰影",
        "best_for": "開場、情緒轉折、預告與章節標題",
        "style": {"surface": "template", "primary_scale": 1.2,
                  "secondary_scale": .72, "line_spacing": 1.24,
                  "block_gap_scale": 1.45,
                  "max_width_scale": 1.08, "shadow_opacity": 170,
                  "shadow_blur": 12, "shadow_offset_y": 4,
                  "stroke_width": 2, "stroke_color": "#000000",
                  "entrance_seconds": .7, "exit_seconds": .45,
                  # Fade ignores distance; keeping a restrained non-zero
                  # token lets Gemini/Web freely switch this family to rise
                  # or slide without declaring motion that cannot move.
                  "motion_distance": 18},
        "cue": {"background": "none",
                "motion": "fade", "position": "center",
                "composition": "auto"},
    },
    "sports_energy": {
        "label": "運動娛樂", "description": "斜角色塊與快速方向感",
        "best_for": "比分、倒數、挑戰、活動與高能量重點",
        "style": {"surface": "ribbon", "primary_color": "#FFFFFF",
                  "secondary_color": "#FFFFFF", "plate_color": "#A72B31",
                  "plate_alpha": 248, "surface_angle": 8,
                  "primary_scale": 1.13, "secondary_scale": .78,
                  "block_gap_scale": .75,
                  "padding_x": 38, "padding_y": 22,
                  "entrance_seconds": .18, "exit_seconds": .14,
                  "motion_distance": 42},
        "cue": {"background": "plate",
                "motion": "slide_left", "composition": "negative_space"},
    },
    "soft_lifestyle": {
        "label": "柔和生活", "description": "半透明霧面板與緩慢上浮",
        "best_for": "美食、居家、人物心情與生活片段",
        "style": {"surface": "glass", "primary_color": "#282521",
                  "secondary_color": "#575149", "plate_color": "#F3EDE3",
                  "plate_alpha": 176, "plate_border_width": 1,
                  "plate_border_color": "#FFFFFF", "corner_radius": 22,
                  "stroke_width": 0, "stroke_color": "#000000",
                  "primary_scale": 1.04, "secondary_scale": .8,
                  "block_gap_scale": 1.2,
                  "shadow_opacity": 120, "shadow_blur": 10,
                  "shadow_offset_y": 3, "entrance_seconds": .52,
                  "exit_seconds": .38, "motion_distance": 22},
        "cue": {"background": "plate",
                "motion": "rise", "composition": "negative_space"},
    },
}
CURATED_GRAPHIC_FAMILY_IDS = {
    "editorial_minimal", "youtube_pop", "magazine_story",
    "social_sticker", "broadcast_info", "cinematic_title",
    "sports_energy", "soft_lifestyle",
}


class CopyFact(Local):
    fact_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    exact_text: str = Field(min_length=1, max_length=240)
    source_kind: SourceKind
    source_reference: str = ""
    source_sha256: str = ""
    text_sha256: str = ""
    allowed_kinds: list[GraphicKind] = Field(default_factory=list)
    confidence: Literal["certain", "likely", "uncertain"] = "certain"
    approved: bool = False
    approved_by: Literal["user_brief", "human_review"] | None = None

    @model_validator(mode="after")
    def a_model_draft_is_not_approved_copy(self) -> "CopyFact":
        expected = hashlib.sha256(self.exact_text.encode("utf-8")).hexdigest()
        if self.text_sha256 and self.text_sha256 != expected:
            raise ValueError("copy changed after its approval digest was made")
        if self.approved and self.approved_by is None:
            raise ValueError("approved copy must name its approval authority")
        if not self.approved and self.approved_by is not None:
            raise ValueError("unapproved copy cannot name an approval authority")
        if self.source_kind == "model_draft" and self.approved:
            raise ValueError(
                "model-draft copy cannot be approved in place; human review "
                "must create a user copy"
            )
        return self


class BrandKit(Local):
    """Legacy name for optional project defaults, never inferred branding."""

    font_path: str = ""
    display_font_path: str = ""
    foreground: str = "#FFFFFF"
    secondary: str = "#D9DCE3"
    accent: str = "#FFFFFF"
    plate: str = "#111318"
    plate_alpha: int = Field(default=205, ge=0, le=255)
    corner_radius: float = Field(default=0.18, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def colours_are_six_digit_hex(self) -> "BrandKit":
        for field in ("foreground", "secondary", "accent", "plate"):
            if not HEX.match(getattr(self, field)):
                raise ValueError(f"{field} must be #RRGGBB")
        return self


class GraphicStyle(Local):
    """Per-cue visual overrides expressed in renderer-safe units.

    Pixel values are authored against a 1080px-tall frame and scaled for the
    actual output. Empty colours inherit the brand kit. Keeping these values
    on the cue makes a Web adjustment deterministic at render time instead of
    relying on browser-only CSS.
    """

    preset: StylePreset = "custom"
    surface: SurfaceTreatment = "template"
    surface_secondary_color: str = ""
    surface_angle: float = Field(default=6.0, ge=-18.0, le=18.0)
    contrast_mode: AdaptiveMode = "auto"
    primary_color: str = ""
    secondary_color: str = ""
    accent_color: str = ""
    plate_color: str = ""
    plate_alpha: int | None = Field(default=None, ge=0, le=255)
    align: TextAlignment = "auto"
    primary_scale: float = Field(default=1.0, ge=0.55, le=1.8)
    secondary_scale: float = Field(default=1.0, ge=0.55, le=1.8)
    line_spacing: float = Field(default=1.0, ge=0.65, le=2.0)
    block_gap_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    max_width_scale: float = Field(default=1.0, ge=0.45, le=1.25)
    padding_x: float = Field(default=28.0, ge=0.0, le=160.0)
    padding_y: float = Field(default=19.0, ge=0.0, le=120.0)
    corner_radius: float | None = Field(default=None, ge=0.0, le=160.0)
    stroke_width: float = Field(default=0.0, ge=0.0, le=24.0)
    stroke_color: str = "#000000"
    shadow_color: str = "#000000"
    shadow_opacity: int = Field(default=0, ge=0, le=255)
    shadow_blur: float = Field(default=0.0, ge=0.0, le=40.0)
    shadow_offset_x: float = Field(default=0.0, ge=-40.0, le=40.0)
    shadow_offset_y: float = Field(default=0.0, ge=-40.0, le=40.0)
    plate_border_width: float = Field(default=0.0, ge=0.0, le=24.0)
    plate_border_color: str = "#FFFFFF"
    emphasis_text: str = Field(default="", max_length=80)
    emphasis_color: str = ""
    entrance_seconds: float | None = Field(default=None, ge=0.05, le=2.0)
    exit_seconds: float | None = Field(default=None, ge=0.05, le=2.0)
    motion_distance: float | None = Field(default=None, ge=0.0, le=240.0)

    @model_validator(mode="before")
    @classmethod
    def legacy_visual_overrides_are_not_silently_adaptive(cls, value):
        if not isinstance(value, dict):
            return value
        has_contrast_mode = "contrast_mode" in value
        preset = str(value.get("preset") or "")
        # The six original Web presets were metadata-only in sparse v1 API
        # payloads.  Materialise only the new, versioned design families here;
        # the Web sends explicit values for every registry entry it applies.
        preset_style = (
            (GRAPHIC_PRESET_SPECS.get(preset) or {}).get("style")
            if preset in CURATED_GRAPHIC_FAMILY_IDS else None
        )
        if preset_style:
            value = {**preset_style, **value}
        if has_contrast_mode:
            return value
        if not value:
            return {"contrast_mode": "auto"}
        # Surface fields did not exist in v1. Their presence proves this is a
        # current authored payload, so the documented auto default applies.
        if any(key in value for key in (
            "surface", "surface_secondary_color", "surface_angle",
        )):
            return {**value, "contrast_mode": "auto"}
        # A serialized v1 style was authored before automatic visual
        # mutation existed. Its intent cannot be inferred from whether each
        # value happens to equal today's default, so preserve it wholesale.
        # A cue with no style field still uses default_factory and opts into
        # the new automatic contract.
        return {**value, "contrast_mode": "strict"}

    @model_validator(mode="after")
    def colours_are_empty_or_hex(self) -> "GraphicStyle":
        for field in (
            "primary_color", "secondary_color", "accent_color",
            "plate_color", "emphasis_color", "surface_secondary_color",
        ):
            value = getattr(self, field)
            if value and not HEX.match(value):
                raise ValueError(f"{field} must be empty or #RRGGBB")
        for field in ("stroke_color", "shadow_color", "plate_border_color"):
            if not HEX.match(getattr(self, field)):
                raise ValueError(f"{field} must be #RRGGBB")
        return self


class GraphicTransform(Local):
    """User-authored final-picture transform in resolution-independent units."""

    x: float = Field(default=0.5, ge=0.0, le=1.0)
    y: float = Field(default=0.5, ge=0.0, le=1.0)
    scale: float = Field(default=1.0, ge=0.25, le=4.0)
    rotation_degrees: float = Field(default=0.0, ge=-180.0, le=180.0)
    locked: bool = False


class GraphicCue(Local):
    graphic_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    kind: GraphicKind
    primary_fact_id: str
    secondary_fact_id: str = ""
    # The absolute time is what the finished timeline renders.  The anchor is
    # retained so a later recut can re-resolve it rather than pretending an
    # old clock time still belongs to the same shot.
    anchor_clip_id: str = ""
    # Stable identity in the original selection. anchor_clip_id is the
    # current reel position used to address its crop/segment.
    anchor_selection_index: int | None = Field(default=None, ge=0)
    anchor_offset_seconds: float = Field(default=0.0, ge=0.0)
    at_seconds: float = Field(default=0.0, ge=0.0)
    duration_seconds: float = Field(default=2.5, gt=0.0, le=30.0)
    template: str = "editorial_rule"
    position: PositionHint = "auto"
    composition: CompositionIntent = "auto"
    background: BackgroundTreatment = "auto"
    motion: MotionPreset = "rise"
    music_sync: Literal["none", "accent", "downbeat"] = "none"
    z_index: int | None = Field(default=None, ge=-100, le=100)
    collision_policy: Literal["avoid", "allow"] = "avoid"
    transform: GraphicTransform = Field(default_factory=GraphicTransform)
    editor_note: str = Field(default="", max_length=500)
    style: GraphicStyle = Field(default_factory=GraphicStyle)
    status: CueStatus = "draft"

    @model_validator(mode="before")
    @classmethod
    def legacy_background_choice_is_strict(cls, value):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        style = payload.get("style")
        preset = (
            style.preset if isinstance(style, GraphicStyle)
            else str((style or {}).get("preset") or "")
            if isinstance(style, dict) else ""
        )
        cue_defaults = (
            (GRAPHIC_PRESET_SPECS.get(preset) or {}).get("cue") or {}
            if preset in CURATED_GRAPHIC_FAMILY_IDS else {}
        )
        for field, default in cue_defaults.items():
            if field == "template" and payload.get("kind"):
                spec = globals().get("TEMPLATES", {}).get(str(default))
                if spec is not None and payload["kind"] not in spec.kinds:
                    continue
            payload.setdefault(field, default)
        if "style" not in payload and payload.get("background") in {"none", "plate"}:
            payload["style"] = {"contrast_mode": "strict"}
        return payload


class GraphicsPlan(Local):
    version: Literal[
        "montagewright-graphics-v1", "montagewright-graphics-v2"
    ] = "montagewright-graphics-v2"
    revision: int = Field(default=0, ge=0)
    brand: BrandKit = Field(default_factory=BrandKit)
    facts: list[CopyFact] = Field(default_factory=list)
    cues: list[GraphicCue] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_v1_overlap_semantics(cls, value):
        if not isinstance(value, dict) or value.get("version") != "montagewright-graphics-v1":
            return value
        cues = [
            (
                {**cue, "collision_policy": "allow"}
                if isinstance(cue, dict) and "collision_policy" not in cue
                else cue
            )
            for cue in value.get("cues", [])
        ]
        return {
            **value, "version": "montagewright-graphics-v2", "cues": cues,
        }

    @model_validator(mode="after")
    def references_are_unique_and_real(self) -> "GraphicsPlan":
        fact_ids = [fact.fact_id for fact in self.facts]
        cue_ids = [cue.graphic_id for cue in self.cues]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("copy fact ids must be unique")
        if len(cue_ids) != len(set(cue_ids)):
            raise ValueError("graphic ids must be unique")
        by_id = {fact.fact_id: fact for fact in self.facts}
        for cue in self.cues:
            wanted = [cue.primary_fact_id]
            if cue.secondary_fact_id:
                wanted.append(cue.secondary_fact_id)
            missing = [fact_id for fact_id in wanted if fact_id not in by_id]
            if missing:
                raise ValueError(
                    f"{cue.graphic_id} cites missing copy: {missing}"
                )
            if cue.status == "approved":
                unapproved = [
                    fact_id for fact_id in wanted if not by_id[fact_id].approved
                ]
                if unapproved:
                    raise ValueError(
                        f"{cue.graphic_id} uses unapproved copy: {unapproved}"
                    )
                disallowed = [
                    fact_id for fact_id in wanted
                    if by_id[fact_id].allowed_kinds
                    and cue.kind not in by_id[fact_id].allowed_kinds
                ]
                if disallowed:
                    raise ValueError(
                        f"{cue.graphic_id} uses copy not approved for "
                        f"{cue.kind}: {disallowed}"
                    )
        return self

    def fact(self, fact_id: str) -> CopyFact:
        return next(fact for fact in self.facts if fact.fact_id == fact_id)


def validate_brief_authority(
    plan: GraphicsPlan, approved_facts: list[CopyFact]
) -> None:
    """Reject client claims of brief approval that the server cannot prove."""

    authority = {fact.fact_id: fact for fact in approved_facts}
    for fact in plan.facts:
        if fact.approved_by != "user_brief":
            continue
        trusted = authority.get(fact.fact_id)
        if trusted is None or fact.model_dump() != trusted.model_dump():
            raise ValueError(
                f"{fact.fact_id}: brief approval does not match the "
                "server-side approved-copy manifest"
            )


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    label: str
    kinds: tuple[GraphicKind, ...]
    align: Literal["left", "center"]
    width: float
    title_height: float
    secondary_height: float
    plate: bool
    rule: bool = False
    default_position: PositionHint = "upper_left"


TEMPLATES: dict[str, TemplateSpec] = {
    "editorial_rule": TemplateSpec(
        "editorial_rule", "編輯線條", tuple(GraphicKind.__args__),
        "left", 0.72, 0.055, 0.025, False, rule=True,
    ),
    "product_plate": TemplateSpec(
        "product_plate", "產品銘牌", ("product_name", "feature", "callout"),
        "left", 0.68, 0.048, 0.023, True, default_position="lower_left",
    ),
    "hero_center": TemplateSpec(
        "hero_center", "主視覺標題", ("opening_title", "chapter", "end_card"),
        "center", 0.84, 0.075, 0.031, False, default_position="center",
    ),
    "stat_badge": TemplateSpec(
        "stat_badge", "規格徽章", ("feature", "callout"),
        "center", 0.42, 0.060, 0.022, True, default_position="upper_right",
    ),
    "center_stack": TemplateSpec(
        "center_stack", "置中多行",
        ("opening_title", "chapter", "feature", "callout", "end_card"),
        "center", 0.86, 0.060, 0.034, False, default_position="center",
    ),
    "spec_stack": TemplateSpec(
        "spec_stack", "多行規格", ("product_name", "feature", "callout"),
        "left", 0.74, 0.048, 0.031, True, default_position="upper_left",
    ),
    "end_roster": TemplateSpec(
        "end_roster", "多行結尾卡", ("end_card",),
        "center", 0.90, 0.060, 0.029, True, default_position="center",
    ),
}


def templates_for_editor() -> list[dict]:
    return [
        {
            "template_id": spec.template_id,
            "label": spec.label,
            "kinds": list(spec.kinds),
            "default_position": spec.default_position,
            "plate": spec.plate,
            "width": spec.width,
            "title_height": spec.title_height,
            "secondary_height": spec.secondary_height,
            "align": spec.align,
        }
        for spec in TEMPLATES.values()
    ]


def graphic_presets_for_editor() -> list[dict]:
    """Versioned registry shared by Gemini materialisation and Web editing."""

    return [
        {
            "preset_id": preset_id,
            "label": spec["label"],
            "description": spec["description"],
            "best_for": spec.get("best_for", ""),
            "style": dict(spec.get("style") or {}),
            "cue": dict(spec.get("cue") or {}),
            "curated_family": preset_id in CURATED_GRAPHIC_FAMILY_IDS,
        }
        for preset_id, spec in GRAPHIC_PRESET_SPECS.items()
    ]


def curated_graphic_family_ids() -> tuple[str, ...]:
    return tuple(
        item["preset_id"] for item in graphic_presets_for_editor()
        if item["curated_family"]
    )


def graphic_preset_defaults(preset_id: str) -> tuple[dict, dict]:
    spec = GRAPHIC_PRESET_SPECS.get(preset_id) or {}
    return dict(spec.get("style") or {}), dict(spec.get("cue") or {})


def graphic_family_prompt() -> str:
    """Small semantic menu; the model may still override surface and motion."""

    return "\n".join(
        f"- `{item['preset_id']}`：{item['description']}"
        + (f"；適合 {item['best_for']}" if item.get("best_for") else "")
        for item in graphic_presets_for_editor()
        if item["curated_family"]
    )


def recommended_graphic_family(kind: GraphicKind) -> str:
    return {
        "opening_title": "cinematic_title",
        "chapter": "magazine_story",
        "product_name": "broadcast_info",
        "feature": "editorial_minimal",
        "callout": "youtube_pop",
        "end_card": "cinematic_title",
    }[kind]


def minimum_read_seconds(text: str, kind: GraphicKind) -> float:
    """A conservative glance-time floor, before entrance and exit motion."""

    compact = "".join(text.split())
    cjk = sum("\u3400" <= char <= "\u9fff" for char in compact)
    latin_words = len(re.findall(r"[A-Za-z0-9]+", text))
    punctuation = len(re.findall(r"[，。！？：；,.!?:;]", text))
    base = 0.75 + cjk / 6.0 + latin_words / 3.0 + punctuation * 0.08
    if kind in {"opening_title", "chapter", "end_card"}:
        base = max(base, 2.0)
    return round(max(1.2, base), 2)


def validate_for_render(
    plan: GraphicsPlan,
    *,
    duration_seconds: float,
    subtitle_windows: list[tuple[float, float]] | None = None,
) -> list[str]:
    """Return every local delivery fault; rendering requires an empty list."""

    problems: list[str] = []
    active = [cue for cue in plan.cues if cue.status == "approved"]
    for cue in active:
        if cue.template not in TEMPLATES:
            problems.append(f"{cue.graphic_id}: unknown template {cue.template}")
            continue
        if cue.kind not in TEMPLATES[cue.template].kinds:
            problems.append(
                f"{cue.graphic_id}: {cue.template} does not support {cue.kind}"
            )
        end = cue.at_seconds + cue.duration_seconds
        if end > duration_seconds + 0.02:
            problems.append(
                f"{cue.graphic_id}: ends at {end:.2f}s past a "
                f"{duration_seconds:.2f}s film"
            )
        text = plan.fact(cue.primary_fact_id).exact_text
        if cue.secondary_fact_id:
            text += " " + plan.fact(cue.secondary_fact_id).exact_text
        floor = minimum_read_seconds(text, cue.kind)
        if cue.duration_seconds + 1e-6 < floor:
            problems.append(
                f"{cue.graphic_id}: {cue.duration_seconds:.2f}s is below the "
                f"{floor:.2f}s reading floor"
            )
        if subtitle_windows and cue.position in {
            "lower_left", "lower_right"
        }:
            if any(
                cue.at_seconds < sub_end
                and cue.at_seconds + cue.duration_seconds > sub_start
                for sub_start, sub_end in subtitle_windows
            ):
                problems.append(
                    f"{cue.graphic_id}: lower graphic overlaps subtitles"
                )
    events = sorted(
        [
            (time, edge, cue.graphic_id)
            for cue in active
            for time, edge in (
                (cue.at_seconds, 1),
                (cue.at_seconds + cue.duration_seconds, -1),
            )
        ],
        # End before start makes windows half-open: [start, end).
        key=lambda event: (event[0], event[1]),
    )
    visible: set[str] = set()
    for _, edge, graphic_id in events:
        if edge < 0:
            visible.discard(graphic_id)
        else:
            visible.add(graphic_id)
            if len(visible) > 2:
                problems.append(
                    "more than two graphics are visible together: "
                    + ", ".join(sorted(visible))
                )
                break
    return problems


def _rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def _font(size: int, text: str, preferred: str = ""):
    from PIL import ImageFont
    from montagewright import subtitles

    if preferred:
        try:
            face = ImageFont.truetype(preferred, size)
            if subtitles._can_draw(face, text):  # noqa: SLF001
                return face
        except OSError:
            pass
    # The subtitle engine already selects the correct TC/HK/SC face and
    # verifies glyph coverage. This becomes a shared typography module once
    # both tracks have settled on the same public surface.
    return subtitles._face(size, text=text)  # noqa: SLF001


def _fit(text: str, *, asked: int, room: int, font_path: str):
    size = asked
    for _ in range(16):
        face = _font(max(12, size), text, font_path)
        if max((face.getbbox(line)[2] for line in text.splitlines()), default=0) <= room:
            return face
        size = round(size * 0.94)
    face = _font(max(12, size), text, font_path)
    if max((face.getbbox(line)[2] for line in text.splitlines()), default=0) > room:
        raise ValueError(
            "graphic copy is too long for this one-line template; shorten "
            "it or choose a wider template"
        )
    return face


@dataclass(frozen=True)
class DrawnGraphic:
    path: Path
    left: int
    top: int
    width: int
    height: int
    text_mask_path: Path | None = None
    backing_path: Path | None = None
    text_run_mask_paths: tuple[Path, ...] = ()
    # The intended glyph colours without stroke, shadow or background.  The
    # contrast audit must not infer a foreground colour from the composited
    # card: after a small card is resampled, a black outline legitimately
    # bleeds into white glyph pixels and looks like black-on-black evidence.
    foreground_path: Path | None = None


def _save_png_atomic(image, destination: Path) -> None:
    """Publish one complete PNG or leave the previous file untouched."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}-",
            suffix=".png",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        image.save(temporary_path, format="PNG")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class LayoutEvidence:
    """Subject geometry in final-picture coordinates, supplied by any tracker."""

    # Normalised x, y, width, height boxes across the cue duration. This
    # deliberately does not import SAM: SAM, Vision and a human correction
    # can all provide the same neutral evidence.
    subject_boxes: tuple[tuple[float, float, float, float], ...] = ()
    source: str = "visual_complexity_proxy"


def _apply_authored_transform(
    cue: GraphicCue, card: DrawnGraphic, width: int, height: int
) -> DrawnGraphic:
    """Apply scale/rotation to all renderer evidence, then resolve manual XY."""

    from PIL import Image

    transform = cue.transform
    paths = [
        card.path, card.text_mask_path, card.backing_path,
        card.foreground_path, *card.text_run_mask_paths,
    ]
    mask_indexes = {1, *range(4, len(paths))}
    images = [
        Image.open(path).convert("L" if index in mask_indexes else "RGBA")
        if path is not None else None
        for index, path in enumerate(paths)
    ]
    if transform.scale < 1:
        images = [
            image.resize(
                (
                    max(1, round(image.width * transform.scale)),
                    max(1, round(image.height * transform.scale)),
                ),
                resample=Image.Resampling.LANCZOS,
            ) if image is not None else None
            for image in images
        ]
    if transform.rotation_degrees:
        images = [
            image.rotate(
                -transform.rotation_degrees, expand=True,
                resample=Image.Resampling.BICUBIC,
                fillcolor=(
                    0 if index in mask_indexes else (0, 0, 0, 0)
                ),
            ) if image is not None else None
            for index, image in enumerate(images)
        ]
    for path, image in zip(paths, images):
        if path is not None and image is not None:
            _save_png_atomic(image, path)
    rendered = images[0]
    assert rendered is not None
    left, top = card.left, card.top
    if cue.position == "manual":
        left = round(transform.x * width - rendered.width / 2)
        top = round(transform.y * height - rendered.height / 2)
    else:
        spec = TEMPLATES[cue.template]
        position = cue.position if cue.position != "auto" else spec.default_position
        left, top = _placements(
            width, height, rendered.width, rendered.height
        ).get(position, _placements(
            width, height, rendered.width, rendered.height
        )[spec.default_position])
    return replace(
        card, left=left, top=top,
        width=rendered.width, height=rendered.height,
    )


def _placements(
    width: int, height: int, card_width: int, card_height: int
) -> dict[str, tuple[int, int]]:
    side = round(width * 0.075)
    top_safe = round(height * 0.09)
    bottom_safe = round(height * (0.25 if width < height else 0.10))
    positions = {
        "top": ((width - card_width) // 2, top_safe),
        "upper_left": (side, top_safe),
        "upper_right": (width - side - card_width, top_safe),
        "lower_left": (side, height - bottom_safe - card_height),
        "lower_right": (
            width - side - card_width, height - bottom_safe - card_height
        ),
        "center": (
            (width - card_width) // 2, (height - card_height) // 2
        ),
    }
    # A manually enlarged card must never be clipped by the delivery frame.
    return {
        name: (
            max(0, min(left, width - card_width)),
            max(0, min(top, height - card_height)),
        )
        for name, (left, top) in positions.items()
    }


def resolve_auto_position(
    cue: GraphicCue,
    card: DrawnGraphic,
    frames: list,
    *,
    frame_width: int,
    frame_height: int,
    evidence: LayoutEvidence | None = None,
    forbidden_positions: set[str] | None = None,
    keepout_rects: list[tuple[int, int, int, int]] | None = None,
) -> tuple[DrawnGraphic, dict[str, float]]:
    """Choose a stable low-detail region over several final-picture frames."""

    from PIL import ImageFilter, ImageStat

    if not frames:
        raise ValueError(f"{cue.graphic_id}: no frames for automatic layout")
    proof = evidence or LayoutEvidence()
    if cue.composition in {"avoid_subject", "overlap_subject"} and not proof.subject_boxes:
        raise ValueError(
            f"{cue.graphic_id}: {cue.composition} with auto position needs "
            "subject evidence; choose a fixed position or provide tracking"
        )
    positions = _placements(
        frame_width, frame_height, card.width, card.height
    )
    all_candidates = (
        "top", "upper_left", "upper_right", "lower_left", "lower_right",
        "center",
    )
    candidates = tuple(
        name for name in all_candidates
        if name not in (forbidden_positions or set())
    )
    if not candidates:
        raise ValueError(f"{cue.graphic_id}: no safe automatic position")
    scores: dict[str, float] = {}
    for name in candidates:
        left, top = positions[name]
        motion_safe = _place_for_motion_safe(
            cue, replace(card, left=left, top=top),
            frame_width, frame_height,
        )
        left, top = motion_safe.left, motion_safe.top
        right = min(frame_width, left + card.width)
        bottom = min(frame_height, top + card.height)
        details: list[float] = []
        brightness: list[float] = []
        for frame in frames:
            roi = frame.crop((max(0, left), max(0, top), right, bottom))
            if roi.width <= 0 or roi.height <= 0:
                details.append(1000.0)
                brightness.append(255.0)
                continue
            gray = roi.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            details.append(
                float(ImageStat.Stat(edges).mean[0])
                + float(ImageStat.Stat(gray).stddev[0])
            )
            brightness.append(float(ImageStat.Stat(gray).mean[0]))
        # The worst sampled frame matters: a card cannot be legible only in
        # the middle of its cue. Transparent designs also prefer dark space.
        plate = (
            cue.background == "plate"
            or (cue.background == "auto" and TEMPLATES[cue.template].plate)
            or cue.composition == "foreground_plate"
        )
        score = max(details) + (0.0 if plate else max(brightness) * 0.10)
        candidate = _swept_rect(
            cue, replace(card, left=left, top=top),
            frame_width, frame_height,
        )
        keepout_overlap = max(
            (_rect_overlap(candidate, rect) for rect in (keepout_rects or [])),
            default=0.0,
        )
        score += keepout_overlap * 5000.0
        # Sampling frequency and cue duration must not make the same subject
        # geometrically more important. Use worst-frame occupancy, not a sum.
        overlap = max(
            (_box_overlap(candidate, box, frame_width, frame_height)
             for box in proof.subject_boxes),
            default=0.0,
        )
        if cue.composition in {"auto", "avoid_subject"}:
            score += overlap * 500.0
        elif cue.composition == "overlap_subject":
            score -= overlap * 500.0
        if name == TEMPLATES[cue.template].default_position:
            score -= 0.01  # deterministic tie-break that respects the design
        scores[name] = round(score, 4)
    chosen = min(candidates, key=lambda name: (scores[name], candidates.index(name)))
    left, top = positions[chosen]
    motion_safe = _place_for_motion_safe(
        cue, replace(card, left=left, top=top), frame_width, frame_height
    )
    left, top = motion_safe.left, motion_safe.top
    chosen_rect = _swept_rect(
        cue, replace(card, left=left, top=top),
        frame_width, frame_height,
    )
    if any(_rect_overlap(chosen_rect, rect) > 0.01 for rect in (keepout_rects or [])):
        raise ValueError(
            f"{cue.graphic_id}: no automatic position clears subtitle/safe-area keepouts"
        )
    return replace(card, left=max(0, left), top=max(0, top)), scores


def _box_overlap(
    candidate: tuple[int, int, int, int],
    subject: tuple[float, float, float, float],
    width: int,
    height: int,
) -> float:
    left, top, wide, tall = candidate
    sx, sy, sw, sh = subject
    subject_left, subject_top = sx * width, sy * height
    subject_right, subject_bottom = (sx + sw) * width, (sy + sh) * height
    overlap_width = max(0.0, min(left + wide, subject_right) - max(left, subject_left))
    overlap_height = max(0.0, min(top + tall, subject_bottom) - max(top, subject_top))
    return overlap_width * overlap_height / max(1.0, wide * tall)


def _rect_overlap(
    candidate: tuple[int, int, int, int], other: tuple[int, int, int, int]
) -> float:
    left, top, wide, tall = candidate
    o_left, o_top, o_wide, o_tall = other
    overlap_width = max(0, min(left + wide, o_left + o_wide) - max(left, o_left))
    overlap_height = max(0, min(top + tall, o_top + o_tall) - max(top, o_top))
    intersection = overlap_width * overlap_height
    return intersection / max(1, min(wide * tall, o_wide * o_tall))


def _motion_distance_pixels(
    cue: GraphicCue, frame_width: int, frame_height: int
) -> int:
    if cue.style.motion_distance is not None:
        return round(cue.style.motion_distance * frame_height / 1080.0)
    if cue.motion == "rise":
        return round(frame_height * 0.025)
    if cue.motion in {"slide_left", "slide_right"}:
        return round(frame_width * 0.06)
    return 0


def _feasible_graphic_motion(
    cue: GraphicCue,
    card: DrawnGraphic,
    frame_width: int,
    frame_height: int,
) -> tuple[str, int]:
    """Resolve a family entrance that can exist inside the delivery safe area.

    Curated families are starting points, so an automatic card may shorten a
    travel or become a fade when its text nearly fills the safe width/height.
    A manually positioned, locked, or strict card is authored geometry: do not
    silently reinterpret it; the normal safe-area check will explain why it
    cannot render.
    """

    requested = _motion_distance_pixels(cue, frame_width, frame_height)
    if cue.motion not in {"rise", "slide_left", "slide_right"}:
        return cue.motion, 0
    if (
        cue.position == "manual"
        or cue.transform.locked
        or cue.style.contrast_mode == "strict"
    ):
        return cue.motion, requested
    side = round(frame_width * 0.075)
    top_safe = round(frame_height * 0.09)
    bottom = frame_height - round(
        frame_height * (0.25 if frame_width < frame_height else 0.10)
    )
    capacity = (
        bottom - top_safe - card.height
        if cue.motion == "rise"
        else frame_width - side * 2 - card.width
    )
    capacity = max(0, capacity)
    if requested <= capacity:
        return cue.motion, requested
    # A two-pixel nudge is visually a fade with rounding noise. Calling it a
    # fade also makes Web/FFmpeg reports honest about what will be delivered.
    minimum_travel = max(2, round(frame_height * 0.008))
    if capacity < minimum_travel:
        return "fade", 0
    return cue.motion, capacity


def _swept_rect(
    cue: GraphicCue, card: DrawnGraphic,
    frame_width: int, frame_height: int,
) -> tuple[int, int, int, int]:
    motion, distance = _feasible_graphic_motion(
        cue, card, frame_width, frame_height
    )
    left, top, wide, tall = card.left, card.top, card.width, card.height
    if motion == "rise":
        tall += distance
    elif motion == "slide_left":
        wide += distance
    elif motion == "slide_right":
        left -= distance
        wide += distance
    return left, top, wide, tall


def _place_for_motion_safe(
    cue: GraphicCue, card: DrawnGraphic, width: int, height: int
) -> DrawnGraphic:
    side = round(width * 0.075)
    top_safe = round(height * 0.09)
    bottom = height - round(height * (0.25 if width < height else 0.10))
    motion, distance = _feasible_graphic_motion(cue, card, width, height)
    min_left, max_left = side, width - side - card.width
    min_top, max_top = top_safe, bottom - card.height
    if motion == "rise":
        max_top -= distance
    elif motion == "slide_left":
        max_left -= distance
    elif motion == "slide_right":
        min_left += distance
    if min_left > max_left or min_top > max_top:
        raise ValueError(
            f"{cue.graphic_id}: card plus entrance motion cannot fit inside "
            "the delivery safe area; reduce its size or motion distance"
        )
    if cue.position == "manual" and (
        card.left < min_left or card.left > max_left
        or card.top < min_top or card.top > max_top
    ):
        raise ValueError(
            f"{cue.graphic_id}: manual position or its entrance motion falls "
            "outside the delivery safe area; move or resize it"
        )
    return replace(
        card,
        left=max(min_left, min(card.left, max_left)),
        top=max(min_top, min(card.top, max_top)),
    )


def resolve_graphic_animation(
    cue: GraphicCue, card: DrawnGraphic, width: int, height: int
) -> dict:
    enter = (
        min(cue.style.entrance_seconds, cue.duration_seconds * 0.4)
        if cue.style.entrance_seconds is not None
        else min(0.45, max(0.16, cue.duration_seconds * 0.12))
    )
    leave = (
        min(cue.style.exit_seconds, cue.duration_seconds * 0.3)
        if cue.style.exit_seconds is not None
        else min(0.32, max(0.12, cue.duration_seconds * 0.08))
    )
    motion, distance = _feasible_graphic_motion(cue, card, width, height)
    requested_distance = _motion_distance_pixels(cue, width, height)
    from_left, from_top = card.left, card.top
    if motion == "rise":
        from_top += distance
    elif motion == "slide_left":
        from_left += distance
    elif motion == "slide_right":
        from_left -= distance
    return {
        "start_seconds": cue.at_seconds,
        "end_seconds": cue.at_seconds + cue.duration_seconds,
        "enter_seconds": enter,
        "leave_seconds": leave,
        "from_left": from_left,
        "from_top": from_top,
        "settled_left": card.left,
        "settled_top": card.top,
        "fade": motion != "none",
        "requested_motion": cue.motion,
        "resolved_motion": motion,
        "requested_motion_distance": requested_distance,
        "resolved_motion_distance": distance,
        "easing": "linear",
    }


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    channels = []
    for value in rgb:
        channel = value / 255.0
        channels.append(
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
        )
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722


def _contrast_ratio(one: tuple[int, int, int], other: tuple[int, int, int]) -> float:
    bright, dark = sorted((_relative_luminance(one), _relative_luminance(other)), reverse=True)
    return (bright + 0.05) / (dark + 0.05)


def measured_text_contrast(
    card: DrawnGraphic,
    frames: list,
    *,
    cue: GraphicCue | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    outline_width: float = 0.0,
    outline_rgb: tuple[int, int, int] = (0, 0, 0),
) -> float | None:
    """Low-percentile glyph/background contrast across sampled final frames."""

    from PIL import Image

    if not frames or card.text_mask_path is None or card.backing_path is None:
        return None
    glyph = Image.open(card.path).convert("RGBA")
    foreground = (
        Image.open(card.foreground_path).convert("RGBA")
        if card.foreground_path is not None else None
    )
    backing = Image.open(card.backing_path).convert("RGBA")
    mask_paths = (
        card.text_run_mask_paths
        if card.text_run_mask_paths else (card.text_mask_path,)
    )
    point_groups: list[list[tuple[int, int]]] = []
    for mask_path in mask_paths:
        mask = Image.open(mask_path).convert("L")
        maximum_ink = mask.getextrema()[1]
        if maximum_ink <= 0:
            raise ValueError("graphic text mask contains no auditable glyph pixels")
        ink_threshold = max(8, round(maximum_ink * 0.75))
        points = [
            (x, y)
            for y in range(card.height)
            for x in range(card.width)
            if mask.getpixel((x, y)) >= ink_threshold
        ]
        if not points:
            return None
        # Bound work independently for each text run. A short secondary line
        # must not disappear inside thousands of primary-title pixels.
        stride = max(1, len(points) // 4000)
        point_groups.append(points[::stride])
    ratio_groups: list[list[float]] = [[] for _ in point_groups]
    sample_shares = _layout_sample_shares(cue) if cue is not None else (0.03, 0.5, 0.97)
    animation = (
        resolve_graphic_animation(
            cue, card, frame_width or frames[0].width,
            frame_height or frames[0].height,
        ) if cue is not None else None
    )
    for frame_index, frame in enumerate(frames):
        sample_share = sample_shares[
            min(frame_index, len(sample_shares) - 1)
        ]
        at_left, at_top = card.left, card.top
        if animation is not None:
            elapsed = cue.duration_seconds * sample_share
            progress = min(1.0, max(
                0.0, elapsed / max(0.001, animation["enter_seconds"])
            ))
            at_left = round(
                animation["settled_left"]
                + (animation["from_left"] - animation["settled_left"])
                * (1 - progress)
            )
            at_top = round(
                animation["settled_top"]
                + (animation["from_top"] - animation["settled_top"])
                * (1 - progress)
            )
        picture = frame.convert("RGB")
        for ratios, points in zip(ratio_groups, point_groups):
            for x, y in points:
                px, py = at_left + x, at_top + y
                if not (0 <= px < picture.width and 0 <= py < picture.height):
                    ratios.append(1.0)
                    continue
                base = picture.getpixel((px, py))
                under = backing.getpixel((x, y))
                alpha = under[3] / 255.0
                background = tuple(
                    round(under[index] * alpha + base[index] * (1 - alpha))
                    for index in range(3)
                )
                # New cards carry the authored fill on a separate evidence
                # layer.  Reading the composited RGBA here made a downsampled
                # outline count as the glyph colour and rejected accessible
                # white-on-black fallbacks.  Keep the old read path only for
                # callers constructing a legacy DrawnGraphic by hand.
                intended = (
                    foreground.getpixel((x, y))[:3]
                    if foreground is not None else glyph.getpixel((x, y))[:3]
                )
                direct = _contrast_ratio(intended, background)
                if outline_width >= 1.5:
                    outline_chain = min(
                        _contrast_ratio(intended, outline_rgb),
                        _contrast_ratio(outline_rgb, background),
                    )
                    direct = max(direct, outline_chain)
                ratios.append(direct)
    percentiles = []
    for ratios in ratio_groups:
        ratios.sort()
        # Isolated antialias pixels should not fail an otherwise readable
        # line, while the weakest substantial patch must still be legible.
        percentiles.append(
            ratios[max(0, int(len(ratios) * 0.05) - 1)]
        )
    return round(min(percentiles), 3)


def _layout_sample_shares(cue: GraphicCue) -> tuple[float, float, float]:
    enter = (
        min(cue.style.entrance_seconds, cue.duration_seconds * 0.4)
        if cue.style.entrance_seconds is not None
        else min(0.45, max(0.16, cue.duration_seconds * 0.12))
    )
    # Always inspect halfway through the actual entrance. A percentage such
    # as 3% misses a 0.45s slide on a long 30s title entirely.
    entrance_midpoint = min(0.49, enter * 0.5 / cue.duration_seconds)
    return entrance_midpoint, 0.5, 0.97


def _layout_frames(
    picture: Path,
    cue: GraphicCue,
    *,
    cache_dir: Path | None = None,
) -> list:
    from io import BytesIO
    from PIL import Image

    picture_stat = picture.stat()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for share in _layout_sample_shares(cue):
        at = cue.at_seconds + cue.duration_seconds * share
        cached = None
        if cache_dir is not None:
            frame_digest = hashlib.sha256(
                json.dumps({
                    "picture": str(picture.resolve()),
                    "size": picture_stat.st_size,
                    "mtime_ns": picture_stat.st_mtime_ns,
                    "at": round(at, 6),
                }, sort_keys=True).encode("utf-8")
            ).hexdigest()
            cached = cache_dir / f"{frame_digest}.png"
        try:
            payload = cached.read_bytes() if cached and cached.is_file() else b""
        except OSError:
            payload = b""
        if not payload:
            made = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{at:.3f}", "-i", str(picture),
                    "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
                ],
                capture_output=True,
            )
            payload = made.stdout if made.returncode == 0 else b""
            if payload and cached is not None:
                # Another preview may be compiling the same cue. Publish a
                # complete PNG atomically so neither request can read a
                # half-written cache entry.
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=cache_dir, suffix=".png", delete=False
                    ) as temporary:
                        temporary.write(payload)
                        temporary_path = Path(temporary.name)
                    temporary_path.replace(cached)
                except OSError:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except (OSError, UnboundLocalError):
                        pass
        if payload:
            frames.append(Image.open(BytesIO(payload)).convert("RGB"))
    return frames


def draw_graphic(
    cue: GraphicCue,
    plan: GraphicsPlan,
    *,
    width: int,
    height: int,
    into: Path,
    render_scale: float = 1.0,
) -> DrawnGraphic:
    """Lay out the fully visible hero frame before any motion is applied."""

    from PIL import Image, ImageDraw, ImageFilter
    from montagewright.subtitles import cannot_spell

    spec = TEMPLATES.get(cue.template)
    if spec is None:
        raise ValueError(f"unknown graphic template: {cue.template}")
    style = cue.style
    align = spec.align if style.align == "auto" else style.align
    unit = height / 1080.0 * render_scale
    foreground = style.primary_color or plan.brand.foreground
    secondary_colour = style.secondary_color or plan.brand.secondary
    accent = style.accent_color or plan.brand.accent
    plate_colour = style.plate_color or plan.brand.plate
    emphasis_colour = style.emphasis_color or accent
    primary = plan.fact(cue.primary_fact_id).exact_text.strip()
    secondary = (
        plan.fact(cue.secondary_fact_id).exact_text.strip()
        if cue.secondary_fact_id else ""
    )
    unknown = cannot_spell(primary + secondary)
    if unknown:
        raise ValueError(f"no glyph for {unknown}")

    side_safe = round(width * 0.075)
    base_card_width = min(
        round(width * spec.width * style.max_width_scale),
        width - side_safe * 2,
    )
    card_width = round(base_card_width * render_scale)
    if card_width > width - side_safe * 2:
        raise ValueError(
            f"{cue.graphic_id}: uniform scale needs a {card_width}px card but "
            f"only {width - side_safe * 2}px fits inside the safe frame"
        )
    pad_x = round(style.padding_x * unit)
    pad_y = round(style.padding_y * unit)
    # A sticker's hard shadow belongs outside its foreground plate. Reserve
    # that room before fitting/placing text so the last line cannot sit on
    # the bottom/right shadow and trigger a misleading contrast fallback.
    sticker_offset = (
        max(2, round(7 * unit)) if style.surface == "sticker" else 0
    )
    content_width = card_width - sticker_offset
    stroke_width = round(style.stroke_width * unit)
    shadow_blur = round(style.shadow_blur * unit)
    shadow_x = round(style.shadow_offset_x * unit)
    shadow_y = round(style.shadow_offset_y * unit)
    shadow_room = (
        shadow_blur * 2 + max(abs(shadow_x), abs(shadow_y))
        if style.shadow_opacity else 0
    )
    text_room = content_width - (pad_x + stroke_width + shadow_room) * 2
    title = _fit(
        primary, asked=round(
            height * spec.title_height * style.primary_scale * render_scale
        ),
        room=text_room,
        font_path=plan.brand.display_font_path or plan.brand.font_path,
    )
    secondary_face = (
        _fit(
            secondary,
            asked=round(
                height * spec.secondary_height * style.secondary_scale
                * render_scale
            ),
            room=text_room, font_path=plan.brand.font_path,
        )
        if secondary else None
    )
    title_spacing = round(
        height * 0.009 * style.line_spacing * render_scale
    )
    secondary_spacing = round(
        height * 0.007 * style.line_spacing * render_scale
    )
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    title_box = measure.multiline_textbbox(
        (0, 0), primary, font=title, spacing=title_spacing,
        align=align, stroke_width=stroke_width,
    )
    title_height = title_box[3] - title_box[1]
    secondary_height = (
        measure.multiline_textbbox(
            (0, 0), secondary, font=secondary_face,
            spacing=secondary_spacing, align=align,
            stroke_width=stroke_width,
        )[3]
        - measure.multiline_textbbox(
            (0, 0), secondary, font=secondary_face,
            spacing=secondary_spacing, align=align,
            stroke_width=stroke_width,
        )[1]
        if secondary_face else 0
    )
    gap = (
        round(height * 0.010 * style.block_gap_scale * render_scale)
        if secondary else 0
    )
    rule = round(height * 0.005 * render_scale) if spec.rule else 0
    card_height = (
        pad_y * 2 + title_height + secondary_height + gap + rule
        + shadow_room * 2 + stroke_width * 2
    )
    card_height += sticker_offset
    top_safe = round(height * 0.09)
    bottom_safe = round(height * (0.25 if width < height else 0.10))
    if card_height > height - top_safe - bottom_safe:
        raise ValueError(
            f"{cue.graphic_id}: graphic is {card_height}px tall but only "
            f"{height - top_safe - bottom_safe}px fits inside the safe frame"
        )
    canvas = Image.new("RGBA", (card_width, card_height), (0, 0, 0, 0))
    text_mask = Image.new("L", (card_width, card_height), 0)
    foreground_evidence = Image.new(
        "RGBA", (card_width, card_height), (0, 0, 0, 0)
    )
    line_masks = []
    pen = ImageDraw.Draw(canvas)
    mask_pen = ImageDraw.Draw(text_mask)
    foreground_pen = ImageDraw.Draw(foreground_evidence)
    y = (
        pad_y + shadow_room + stroke_width + rule
        + (round(height * 0.010 * render_scale) if rule else 0)
    )
    title_width = title_box[2] - title_box[0]
    if align == "center":
        x = (content_width - title_width) // 2
    elif align == "right":
        x = content_width - pad_x - shadow_room - stroke_width - title_width
    else:
        x = pad_x + shadow_room + stroke_width

    secondary_box = None
    secondary_width = 0
    secondary_x = 0
    secondary_y = y + title_height + gap
    if secondary_face:
        secondary_box = pen.multiline_textbbox(
            (0, 0), secondary, font=secondary_face,
            spacing=secondary_spacing, align=align,
            stroke_width=stroke_width,
        )
        secondary_width = secondary_box[2] - secondary_box[0]
        secondary_x = (
            (content_width - secondary_width) // 2
            if align == "center" else (
                content_width - pad_x - shadow_room - stroke_width
                - secondary_width if align == "right"
                else pad_x + shadow_room + stroke_width
            )
        )

    plate = spec.plate if cue.background == "auto" else cue.background != "none"
    if cue.composition == "foreground_plate":
        plate = True
    surface = style.surface
    if surface == "template":
        surface = "solid" if plate else "none"
    if cue.composition == "foreground_plate" and surface == "none":
        surface = "solid"
    alpha = (
        style.plate_alpha
        if style.plate_alpha is not None else plan.brand.plate_alpha
    )
    radius = (
        round(style.corner_radius * unit)
        if style.corner_radius is not None
        else round(card_height * plan.brand.corner_radius)
    )
    border_width = round(style.plate_border_width * unit)
    surface_secondary = style.surface_secondary_color or accent
    edge = (0, 0, card_width - 1, card_height - 1)
    outline = (
        (*_rgb(style.plate_border_color), 255)
        if border_width else None
    )

    if surface == "solid":
        pen.rounded_rectangle(
            edge, radius=radius, fill=(*_rgb(plate_colour), alpha),
            outline=outline, width=border_width,
        )
    elif surface == "pill":
        pen.rounded_rectangle(
            edge, radius=card_height // 2,
            fill=(*_rgb(plate_colour), alpha), outline=outline,
            width=border_width,
        )
    elif surface == "split":
        pen.rounded_rectangle(
            edge, radius=radius, fill=(*_rgb(plate_colour), alpha),
            outline=outline, width=border_width,
        )
        split_at = max(
            1, min(card_height - 1,
                   secondary_y - max(1, gap // 2) if secondary_face
                   else round(card_height * 0.64)),
        )
        # A large authored radius can leave no straight section between the
        # split and the rounded bottom.  Pillow rejects an inverted rectangle,
        # so only paint that bridge when it actually exists; the rounded fill
        # below still covers the complete lower surface.
        straight_bottom = card_height - radius
        if straight_bottom >= split_at:
            pen.rectangle(
                (0, split_at, card_width - 1, straight_bottom),
                fill=(*_rgb(surface_secondary), alpha),
            )
        pen.rounded_rectangle(
            (0, split_at, card_width - 1, card_height - 1),
            radius=radius, fill=(*_rgb(surface_secondary), alpha),
        )
        pen.rectangle(
            (0, split_at, card_width - 1,
             min(card_height - 1, split_at + radius)),
            fill=(*_rgb(surface_secondary), alpha),
        )
        if outline:
            pen.rounded_rectangle(
                edge, radius=radius, outline=outline, width=border_width,
            )
    elif surface == "ribbon":
        slant = min(
            max(2, round(card_height * abs(style.surface_angle) / 45.0)),
            max(2, pad_x),
        )
        points = (
            [(slant, 0), (card_width - 1, 0),
             (card_width - slant - 1, card_height - 1), (0, card_height - 1)]
            if style.surface_angle >= 0 else
            [(0, 0), (card_width - slant - 1, 0),
             (card_width - 1, card_height - 1), (slant, card_height - 1)]
        )
        pen.polygon(points, fill=(*_rgb(plate_colour), alpha))
        if border_width:
            pen.line(points + [points[0]], fill=outline, width=border_width,
                     joint="curve")
    elif surface == "sticker":
        offset = sticker_offset or max(2, round(7 * unit))
        sticker_edge = (0, 0, card_width - offset - 1, card_height - offset - 1)
        pen.rounded_rectangle(
            (offset, offset, card_width - 1, card_height - 1),
            radius=radius, fill=(0, 0, 0, 150),
        )
        pen.rounded_rectangle(
            sticker_edge, radius=radius,
            fill=(*_rgb(plate_colour), alpha),
            outline=outline or (*_rgb(surface_secondary), 255),
            width=border_width or max(1, round(3 * unit)),
        )
    elif surface == "highlight":
        extra_x = max(2, round(10 * unit))
        extra_y = max(1, round(4 * unit))
        slant = round(card_height * style.surface_angle / 90.0)

        def highlight(left: int, top: int, wide: int, tall: int, colour: str) -> None:
            pen.polygon(
                [(left - extra_x + slant, top - extra_y),
                 (left + wide + extra_x, top - extra_y),
                 (left + wide + extra_x - slant, top + tall + extra_y),
                 (left - extra_x, top + tall + extra_y)],
                fill=(*_rgb(colour), alpha),
            )

        highlight(x, y, title_width, title_height, plate_colour)
        if secondary_face and secondary_box:
            highlight(
                secondary_x, secondary_y, secondary_width, secondary_height,
                surface_secondary,
            )
    elif surface == "outline":
        pen.rounded_rectangle(
            edge, radius=radius, fill=None,
            outline=outline or (*_rgb(surface_secondary), 255),
            width=border_width or max(1, round(3 * unit)),
        )
    elif surface == "glass":
        glass_alpha = style.plate_alpha if style.plate_alpha is not None else 105
        pen.rounded_rectangle(
            edge, radius=radius, fill=(*_rgb(plate_colour), glass_alpha),
            outline=outline or (255, 255, 255, 180),
            width=border_width or max(1, round(2 * unit)),
        )
        shine = max(1, round(2 * unit))
        pen.line(
            (radius, shine, card_width - radius, shine),
            fill=(255, 255, 255, 125), width=shine,
        )
    elif surface == "editorial":
        rail = max(2, round(7 * unit))
        pen.rounded_rectangle(
            (0, 0, rail, card_height - 1), radius=rail // 2,
            fill=(*_rgb(surface_secondary), 255),
        )
        pen.line(
            (pad_x, card_height - 1, card_width - pad_x, card_height - 1),
            fill=(*_rgb(surface_secondary), 210),
            width=max(1, round(2 * unit)),
        )
    if spec.rule:
        pen.rectangle(
            (pad_x, pad_y, pad_x + round(card_width * 0.13), pad_y + rule),
            fill=(*_rgb(accent), 255),
        )
    backing = canvas.copy()

    def paint_text(
        text: str, face, *, at_x: int, at_y: int, spacing: int,
        colour: str, emphasis: str = "", layout_width: int,
    ) -> None:
        target = (at_x, at_y)
        if style.shadow_opacity:
            shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            shadow_pen = ImageDraw.Draw(shadow)
            shadow_pen.multiline_text(
                (at_x + shadow_x, at_y + shadow_y), text, font=face,
                spacing=spacing, align=align,
                fill=(*_rgb(style.shadow_color), style.shadow_opacity),
                stroke_width=stroke_width,
                stroke_fill=(*_rgb(style.shadow_color), style.shadow_opacity),
            )
            if shadow_blur:
                shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
            canvas.alpha_composite(shadow)
        pen.multiline_text(
            target, text, font=face, spacing=spacing, align=align,
            fill=(*_rgb(colour), 255), stroke_width=stroke_width,
            stroke_fill=(*_rgb(style.stroke_color), 255),
        )
        foreground_pen.multiline_text(
            target, text, font=face, spacing=spacing, align=align,
            fill=(*_rgb(colour), 255),
        )
        # Interior glyph pixels only. The contrast auditor deliberately does
        # not let a large translucent plate hide unreadable foreground text.
        mask_pen.multiline_text(
            target, text, font=face, spacing=spacing, align=align, fill=255,
        )
        lines = text.split("\n")
        widths = [pen.textlength(line, font=face) for line in lines]
        max_line_width = max(widths, default=0)
        line_advance = (
            pen.textbbox((0, 0), "A", font=face)[3] + spacing
        )
        line_y = at_y
        for line, line_width in zip(lines, widths):
            line_mask = Image.new("L", canvas.size, 0)
            line_pen = ImageDraw.Draw(line_mask)
            if align == "center":
                line_x = at_x + (max_line_width - line_width) / 2
            elif align == "right":
                line_x = at_x + max_line_width - line_width
            else:
                line_x = at_x
            line_pen.text(
                (line_x, line_y), line, font=face, fill=255, anchor="la",
            )
            line_masks.append(line_mask)
            line_y += line_advance
        if emphasis:
            line_y = at_y
            for line in text.splitlines() or [text]:
                box = pen.textbbox(
                    (0, 0), line, font=face, stroke_width=stroke_width
                )
                line_width = box[2] - box[0]
                if align == "center":
                    line_x = at_x + (layout_width - line_width) / 2
                elif align == "right":
                    line_x = at_x + layout_width - line_width
                else:
                    line_x = at_x
                start = line.find(emphasis)
                if start >= 0:
                    prefix = line[:start]
                    pen.text(
                        (line_x + pen.textlength(prefix, font=face), line_y),
                        emphasis, font=face, fill=(*_rgb(emphasis_colour), 255),
                        stroke_width=stroke_width,
                        stroke_fill=(*_rgb(style.stroke_color), 255),
                    )
                    foreground_pen.text(
                        (line_x + pen.textlength(prefix, font=face), line_y),
                        emphasis, font=face,
                        fill=(*_rgb(emphasis_colour), 255),
                    )
                line_box = pen.textbbox(
                    (0, 0), line or " ", font=face,
                    stroke_width=stroke_width,
                )
                line_y += line_box[3] - line_box[1] + spacing

    paint_text(
        primary, title, at_x=x - title_box[0], at_y=y - title_box[1],
        spacing=title_spacing, colour=foreground,
        emphasis=style.emphasis_text, layout_width=title_width,
    )
    y += title_height + gap
    if secondary_face:
        assert secondary_box is not None
        x = secondary_x
        paint_text(
            secondary, secondary_face,
            at_x=x - secondary_box[0], at_y=y - secondary_box[1],
            spacing=secondary_spacing, colour=secondary_colour,
            emphasis=style.emphasis_text, layout_width=secondary_width,
        )

    position = cue.position if cue.position != "auto" else spec.default_position
    positions = _placements(width, height, card_width, card_height)
    left, top = positions.get(position, positions[spec.default_position])
    mask_path = into.with_name(f"{into.stem}-text-mask.png")
    backing_path = into.with_name(f"{into.stem}-backing.png")
    foreground_path = into.with_name(f"{into.stem}-foreground.png")
    _save_png_atomic(canvas, into)
    _save_png_atomic(text_mask, mask_path)
    _save_png_atomic(backing, backing_path)
    _save_png_atomic(foreground_evidence, foreground_path)
    run_mask_paths = []
    for index, line_mask in enumerate(line_masks):
        line_mask_path = into.with_name(f"{into.stem}-line-{index}.png")
        _save_png_atomic(line_mask, line_mask_path)
        run_mask_paths.append(line_mask_path)
    return DrawnGraphic(
        into, max(0, left), max(0, top), card_width, card_height,
        mask_path, backing_path, tuple(run_mask_paths), foreground_path,
    )


def compile_graphic(
    cue: GraphicCue,
    plan: GraphicsPlan,
    *,
    width: int,
    height: int,
    into: Path,
    frames: list | None = None,
    evidence: LayoutEvidence | None = None,
    forbidden_positions: set[str] | None = None,
    keepout_rects: list[tuple[int, int, int, int]] | None = None,
) -> tuple[DrawnGraphic, dict]:
    """Compile pixels and placement once for both preview and delivery."""

    if not frames or len(frames) < 3:
        raise ValueError(
            f"{cue.graphic_id}: only {len(frames or [])}/3 picture samples "
            "were decoded; contrast cannot be verified, so rendering is stopped"
        )
    requested_template = cue.template
    try:
        card = draw_graphic(
            cue, plan, width=width, height=height, into=into,
            # Small cards are typeset at full resolution then reduced once;
            # this keeps the font fitter's minimum legible size while
            # preserving a true uniform transform. Enlarged cards are
            # rerasterized sharply.
            render_scale=max(1.0, cue.transform.scale),
        )
    except ValueError as error:
        # A curated family is a semantic request, not permission to ship an
        # impossible card.  When its inherited copy structure outgrows a
        # compact template, choose the widest compatible role locally.  An
        # explicitly locked/strict card remains authored and fails closed.
        may_reflow = (
            cue.style.preset in CURATED_GRAPHIC_FAMILY_IDS
            and cue.style.contrast_mode == "auto"
            and not cue.transform.locked
            and "too long" in str(error)
        )
        alternatives = sorted(
            (
                spec for spec in TEMPLATES.values()
                if cue.kind in spec.kinds and spec.template_id != cue.template
            ),
            key=lambda spec: spec.width,
            reverse=True,
        )
        if not may_reflow or not alternatives:
            raise
        last_error: ValueError = error
        for alternative in alternatives:
            candidate = cue.model_copy(update={"template": alternative.template_id})
            try:
                card = draw_graphic(
                    candidate, plan, width=width, height=height, into=into,
                    render_scale=max(1.0, candidate.transform.scale),
                )
                cue = candidate
                break
            except ValueError as candidate_error:
                last_error = candidate_error
        else:
            raise last_error
    card = _apply_authored_transform(cue, card, width, height)
    report = {
        "resolved_left": card.left, "resolved_top": card.top,
        "resolved_position": (
            cue.position if cue.position != "auto"
            else TEMPLATES[cue.template].default_position
        ),
        "frame_width": width, "frame_height": height,
        "card_width": card.width, "card_height": card.height,
        "scores": {}, "fallback_plate": False,
        "contrast_ratio": None, "contrast_adjustments": [],
        "evidence": evidence.source if evidence else "visual_complexity_proxy",
    }
    scores: dict[str, float] = {}
    if cue.position == "auto":
        card, scores = resolve_auto_position(
            cue, card, frames or [], frame_width=width, frame_height=height,
            evidence=evidence, forbidden_positions=forbidden_positions or set(),
            keepout_rects=keepout_rects,
        )
    card = _place_for_motion_safe(cue, card, width, height)
    ratio = measured_text_contrast(
        card, frames or [], cue=cue, frame_width=width, frame_height=height,
        outline_width=round(
            cue.style.stroke_width * height / 1080.0 * cue.transform.scale
        ),
        outline_rgb=_rgb(cue.style.stroke_color),
    )
    adjustments: list[str] = []
    fallback_plate = False
    if ratio is None:
        raise ValueError(
            f"{cue.graphic_id}: text contrast could not be audited"
        )
    if ratio is not None and ratio < 4.5:
        if cue.style.contrast_mode != "auto":
            raise ValueError(
                f"{cue.graphic_id}: strict colours measure only "
                f"{ratio:.2f}:1 contrast; change the text, outline or plate"
            )
        def compile_accessible_variant(
            variant_style: GraphicStyle, *, background: str,
        ) -> tuple[DrawnGraphic, GraphicCue, dict[str, float], float | None]:
            variant_cue = cue.model_copy(update={
                "background": background, "style": variant_style,
            })
            variant = draw_graphic(
                variant_cue, plan, width=width, height=height, into=into,
                render_scale=max(1.0, variant_cue.transform.scale),
            )
            variant = _apply_authored_transform(
                variant_cue, variant, width, height
            )
            variant_scores: dict[str, float] = {}
            if cue.position == "auto":
                variant, variant_scores = resolve_auto_position(
                    variant_cue, variant, frames, frame_width=width,
                    frame_height=height, evidence=evidence,
                    forbidden_positions=forbidden_positions or set(),
                    keepout_rects=keepout_rects,
                )
            variant = _place_for_motion_safe(
                variant_cue, variant, width, height
            )
            variant_ratio = measured_text_contrast(
                variant, frames or [], cue=variant_cue,
                frame_width=width, frame_height=height,
                outline_width=round(
                    variant_style.stroke_width * height / 1080.0
                    * variant_cue.transform.scale
                ),
                outline_rgb=_rgb(variant_style.stroke_color),
            )
            return variant, variant_cue, variant_scores, variant_ratio

        # Curated families get a first chance to remain themselves.  A
        # cinematic title should become a restrained translucent scrim, and
        # a lifestyle card should deepen its frosted plate, before the global
        # emergency treatment turns either one into outlined YouTube copy.
        family_variants: list[tuple[str, GraphicStyle, str]] = []
        family_surface = str(
            ((GRAPHIC_PRESET_SPECS.get(cue.style.preset) or {}).get("style") or {})
            .get("surface", "")
        )
        family_surface_is_inherited = cue.style.surface == family_surface
        if cue.style.preset == "editorial_minimal" and family_surface_is_inherited:
            family_variants.append((
                "light_editorial_palette",
                cue.style.model_copy(update={
                    "primary_color": "#F7F5F0",
                    "secondary_color": "#D8D5CE",
                    "stroke_width": 0.0,
                }),
                cue.background,
            ))
        elif cue.style.preset == "cinematic_title" and family_surface_is_inherited:
            family_variants.append((
                "cinematic_scrim",
                cue.style.model_copy(update={
                    "surface": "solid", "primary_color": "#FFFFFF",
                    "secondary_color": "#F1EEE8", "plate_color": "#090A0D",
                    "plate_alpha": 125, "stroke_width": 0.0,
                    "shadow_opacity": 185, "shadow_blur": 12,
                    "shadow_offset_y": 4,
                }),
                "plate",
            ))
        elif cue.style.preset == "soft_lifestyle" and family_surface_is_inherited:
            family_variants.extend([
                (
                    "deeper_frosted_plate",
                    cue.style.model_copy(update={
                        "surface": "glass", "primary_color": "#24211D",
                        "secondary_color": "#514A43", "plate_color": "#F5EFE5",
                        "plate_alpha": alpha, "stroke_width": 0.0,
                    }),
                    "plate",
                )
                for alpha in (210, 235)
            ])
        for name, family_style, family_background in family_variants:
            candidate, _candidate_cue, candidate_scores, candidate_ratio = (
                compile_accessible_variant(
                    family_style, background=family_background,
                )
            )
            if candidate_ratio is not None and candidate_ratio >= 4.5:
                card, scores, ratio = candidate, candidate_scores, candidate_ratio
                adjustments.append(name)
                break

        # Preserve the authored surface first. A controlled black outline is
        # enough on many mixed backgrounds and keeps an editorial line,
        # ribbon or highlight looking like itself instead of turning every
        # adaptive title into the same black rectangle.
        outline_style = cue.style.model_copy(update={
            "primary_color": "#FFFFFF",
            "secondary_color": "#FFFFFF",
            "emphasis_color": "#FFFFFF",
            "stroke_color": "#000000",
            "stroke_width": 3.0,
        })
        if ratio < 4.5:
            try:
                outlined, _outlined_cue, outlined_scores, outlined_ratio = (
                    compile_accessible_variant(
                        outline_style, background=cue.background,
                    )
                )
            except ValueError as error:
                # An outline consumes horizontal room.  It is only the first
                # automatic accessibility option, so copy that no longer fits
                # must continue to the plate fallback instead of making an
                # otherwise valid family/template fail rendering.
                if "too long" not in str(error):
                    raise
                outlined_ratio = None
        else:
            outlined_ratio = None
        if ratio < 4.5 and outlined_ratio is not None and outlined_ratio >= 4.5:
            card, scores, ratio = outlined, outlined_scores, outlined_ratio
            adjustments.extend(["white_text", "outline"])
        elif ratio < 4.5:
            # A neutral final fallback, not an inferred brand. It is local
            # and deterministic so preview and export agree.
            fallback_style = cue.style.model_copy(update={
                "surface": "solid",
                "primary_color": "#FFFFFF",
                "secondary_color": "#FFFFFF",
                "emphasis_color": "#FFFFFF",
                "plate_color": "#000000",
                "plate_alpha": 235,
                "stroke_color": "#000000",
                # The fallback owns its accessibility treatment. Reusing an
                # authored extreme outline can swallow thin CJK glyphs and
                # turn nominal white-on-black copy into black-on-black.
                # White copy on the owned opaque plate needs no outline.  A
                # zero-width fallback also preserves the fitter's available
                # room for long CJK/Latin copy.
                "stroke_width": 0.0,
            })
            card, fallback_cue, scores, ratio = compile_accessible_variant(
                fallback_style, background="plate",
            )
            fallback_plate = True
            adjustments.extend(["white_text", "dark_plate", "outline"])
        if ratio is None:
            raise ValueError(
                f"{cue.graphic_id}: automatic contrast fallback could not be audited"
            )
        if ratio is not None and ratio < 4.5:
            raise ValueError(
                f"{cue.graphic_id}: automatic contrast fallback still "
                f"measures only {ratio:.2f}:1"
            )
    card = replace(
        card,
        left=max(0, min(card.left, width - card.width)),
        top=max(0, min(card.top, height - card.height)),
    )
    swept = _swept_rect(cue, card, width, height)
    if any(_rect_overlap(swept, rect) > 0.01 for rect in (keepout_rects or [])):
        raise ValueError(
            f"{cue.graphic_id}: graphic or its entrance motion crosses a subtitle/safe-area keepout"
        )
    report.update({
        "resolved_left": card.left, "resolved_top": card.top,
        "resolved_position": (
            min(scores, key=lambda name: scores[name])
            if scores else cue.position
        ),
        "card_width": card.width, "card_height": card.height,
        "scores": scores, "fallback_plate": fallback_plate,
        "contrast_ratio": ratio, "contrast_adjustments": adjustments,
        "authored_transform": cue.transform.model_dump(mode="json"),
        "authored_style": cue.style.model_dump(mode="json"),
        "requested_template": requested_template,
        "resolved_template": cue.template,
        "animation": resolve_graphic_animation(cue, card, width, height),
    })
    return card, report


def burn_graphics(
    picture: Path,
    plan: GraphicsPlan,
    destination: Path,
    *,
    work: Path,
    subtitle_windows: list[tuple[float, float]] | None = None,
    subtitle_boxes: list[tuple[float, float, int, int, int, int]] | None = None,
    subtitle_overlays: list | None = None,
    layout_evidence: dict[str, LayoutEvidence] | None = None,
) -> Path:
    """Composite approved graphics and keep the clean picture untouched."""

    from montagewright.measure.media import probe_video
    from montagewright.renderer import probe_duration

    shape = probe_video(picture).video
    width, height = int(shape.display_width), int(shape.display_height)
    rate = shape.average_frame_rate or shape.real_frame_rate
    output_fps = (
        f"{rate.numerator}/{rate.denominator}"
        if rate is not None and rate.denominator else "30/1"
    )
    duration = probe_duration(picture)
    problems = validate_for_render(
        plan, duration_seconds=duration,
        subtitle_windows=(None if subtitle_boxes is not None else subtitle_windows),
    )
    if problems:
        raise ValueError("; ".join(problems))
    cues = [cue for cue in plan.cues if cue.status == "approved"]
    if not cues:
        raise ValueError("no approved graphics to render")

    plan_order = {cue.graphic_id: index for index, cue in enumerate(plan.cues)}
    layout_cues = sorted(
        cues, key=lambda item: (item.position == "auto", plan_order[item.graphic_id])
    )
    drawn = []
    layout_report: dict[str, dict] = {}
    for cue in layout_cues:
        overlaps_subtitles = bool(subtitle_windows and any(
            cue.at_seconds < sub_end
            and cue.at_seconds + cue.duration_seconds > sub_start
            for sub_start, sub_end in subtitle_windows
        ))
        evidence = (layout_evidence or {}).get(cue.graphic_id)
        cue_keepouts = [
            (left, top, wide, tall)
            for sub_start, sub_end, left, top, wide, tall in (subtitle_boxes or [])
            if cue.at_seconds < sub_end
            and cue.at_seconds + cue.duration_seconds > sub_start
        ]
        overlapping_graphics = [
            (other, other_card)
            for other, other_card in drawn
            if cue.at_seconds < other.at_seconds + other.duration_seconds
            and cue.at_seconds + cue.duration_seconds > other.at_seconds
            and (
                cue.collision_policy == "avoid"
                or other.collision_policy == "avoid"
            )
        ]
        cue_keepouts.extend(
            _swept_rect(other, other_card, width, height)
            for other, other_card in overlapping_graphics
        )
        card, compiled = compile_graphic(
            cue, plan, width=width, height=height,
            into=work / f"{cue.graphic_id}.png",
            frames=_layout_frames(picture, cue),
            evidence=evidence,
            forbidden_positions=(
                {"lower_left", "lower_right"}
                if overlaps_subtitles and subtitle_boxes is None else set()
            ),
            keepout_rects=cue_keepouts,
        )
        compiled.update({
            "z_index": (
                cue.z_index if cue.z_index is not None
                else plan_order[cue.graphic_id]
            ),
            "avoids_graphic_ids": [
                other.graphic_id for other, _ in overlapping_graphics
            ],
        })
        layout_report[cue.graphic_id] = compiled
        drawn.append((cue, card))
    drawn.sort(key=lambda item: (
        item[0].z_index
        if item[0].z_index is not None
        else plan_order[item[0].graphic_id],
        plan_order[item[0].graphic_id],
    ))
    work.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(picture)]
    for _, card in drawn:
        command += [
            "-loop", "1", "-framerate", output_fps,
            "-i", str(card.path),
        ]
    for overlay in subtitle_overlays or []:
        command += ["-i", str(overlay.path)]

    filters: list[str] = []
    tag = "0:v"
    for index, (cue, card) in enumerate(drawn):
        animation = resolve_graphic_animation(cue, card, width, height)
        since = animation["start_seconds"]
        until = animation["end_seconds"]
        enter = animation["enter_seconds"]
        leave = animation["leave_seconds"]
        overlay_tag = f"g{index}"
        if cue.motion == "none":
            filters.append(f"[{index + 1}:v]format=rgba[{overlay_tag}]")
        else:
            filters.append(
                f"[{index + 1}:v]format=rgba,"
                f"fade=t=in:st={since:.9f}:d={enter:.9f}:alpha=1,"
                f"fade=t=out:st={until - leave:.9f}:d={leave:.9f}:alpha=1"
                f"[{overlay_tag}]"
            )
        progress = (
            f"min(max((t-{since:.9f})/{enter:.9f},0),1)"
        )
        dx = animation["from_left"] - animation["settled_left"]
        dy = animation["from_top"] - animation["settled_top"]
        x = f"{animation['settled_left']}+{dx}*(1-{progress})"
        y = f"{animation['settled_top']}+{dy}*(1-{progress})"
        next_tag = f"v{index}"
        filters.append(
            f"[{tag}][{overlay_tag}]overlay=x='{x}':y='{y}':"
            f"enable='gte(t,{since:.9f})*lt(t,{until:.9f})'[{next_tag}]"
        )
        tag = next_tag
    for index, overlay in enumerate(subtitle_overlays or []):
        input_index = 1 + len(drawn) + index
        overlay_tag = f"s{index}"
        filters.append(f"[{input_index}:v]format=rgba[{overlay_tag}]")
        next_tag = f"sv{index}"
        filters.append(
            f"[{tag}][{overlay_tag}]overlay={overlay.left}:{overlay.top}:"
            f"enable='gte(t,{overlay.starts_seconds:.9f})*"
            f"lt(t,{overlay.ends_seconds:.9f})'[{next_tag}]"
        )
        tag = next_tag
    command += [
        "-filter_complex", ";".join(filters),
        "-map", f"[{tag}]", "-map", "0:a?", "-t", f"{duration:.6f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-r", output_fps,
        "-fps_mode", "cfr", "-c:a", "copy", "-movflags", "+faststart",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}-", suffix=destination.suffix,
        dir=destination.parent, delete=False,
    ) as handle:
        temporary = Path(handle.name)
    command.append(str(temporary))
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            lines = completed.stderr.strip().splitlines()
            raise RuntimeError(
                lines[-1] if lines else "ffmpeg graphics compositor failed"
            )
        temporary.replace(destination)
        from montagewright.measure.storage import write_json

        write_json(work / "layout.json", layout_report)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
