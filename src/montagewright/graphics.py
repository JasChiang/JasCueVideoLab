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
    "tech_frame", "bold_pop",
]

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
GRAPHICS_RENDERER_VERSION = 6


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
        if not isinstance(value, dict) or "contrast_mode" in value:
            return value
        if not value:
            return {"contrast_mode": "auto"}
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
            "plate_color", "emphasis_color",
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
        if not isinstance(value, dict) or "style" in value:
            return value
        if value.get("background") in {"none", "plate"}:
            return {**value, "style": {"contrast_mode": "strict"}}
        return value


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
        "center_stack", "置中多行", ("opening_title", "chapter", "feature", "end_card"),
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
    paths = [card.path, card.text_mask_path, card.backing_path]
    images = [
        Image.open(path).convert("L" if index == 1 else "RGBA")
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
                resample=Image.Resampling.BICUBIC, fillcolor=(0 if index == 1 else (0, 0, 0, 0)),
            ) if image is not None else None
            for index, image in enumerate(images)
        ]
    for path, image in zip(paths, images):
        if path is not None and image is not None:
            image.save(path)
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


def _swept_rect(
    cue: GraphicCue, card: DrawnGraphic,
    frame_width: int, frame_height: int,
) -> tuple[int, int, int, int]:
    distance = _motion_distance_pixels(cue, frame_width, frame_height)
    left, top, wide, tall = card.left, card.top, card.width, card.height
    if cue.motion == "rise":
        tall += distance
    elif cue.motion == "slide_left":
        wide += distance
    elif cue.motion == "slide_right":
        left -= distance
        wide += distance
    return left, top, wide, tall


def _place_for_motion_safe(
    cue: GraphicCue, card: DrawnGraphic, width: int, height: int
) -> DrawnGraphic:
    side = round(width * 0.075)
    top_safe = round(height * 0.09)
    bottom = height - round(height * (0.25 if width < height else 0.10))
    distance = _motion_distance_pixels(cue, width, height)
    min_left, max_left = side, width - side - card.width
    min_top, max_top = top_safe, bottom - card.height
    if cue.motion == "rise":
        max_top -= distance
    elif cue.motion == "slide_left":
        max_left -= distance
    elif cue.motion == "slide_right":
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
    distance = _motion_distance_pixels(cue, width, height)
    from_left, from_top = card.left, card.top
    if cue.motion == "rise":
        from_top += distance
    elif cue.motion == "slide_left":
        from_left += distance
    elif cue.motion == "slide_right":
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
        "fade": cue.motion != "none",
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
    mask = Image.open(card.text_mask_path).convert("L")
    backing = Image.open(card.backing_path).convert("RGBA")
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
    # A deterministic even sample bounds work for long multi-line cards.
    stride = max(1, len(points) // 4000)
    points = points[::stride]
    ratios: list[float] = []
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
        for x, y in points:
            px, py = at_left + x, at_top + y
            if not (0 <= px < picture.width and 0 <= py < picture.height):
                ratios.append(1.0)
                continue
            base = picture.getpixel((px, py))
            under = backing.getpixel((x, y))
            alpha = under[3] / 255.0
            background = tuple(round(under[index] * alpha + base[index] * (1 - alpha)) for index in range(3))
            foreground = glyph.getpixel((x, y))[:3]
            direct = _contrast_ratio(foreground, background)
            if outline_width >= 1.5:
                outline_chain = min(
                    _contrast_ratio(foreground, outline_rgb),
                    _contrast_ratio(outline_rgb, background),
                )
                direct = max(direct, outline_chain)
            ratios.append(direct)
    ratios.sort()
    # Isolated antialias pixels should not fail an otherwise readable title,
    # while the weakest substantial patch still must be legible.
    return round(ratios[max(0, int(len(ratios) * 0.05) - 1)], 3)


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


def _layout_frames(picture: Path, cue: GraphicCue) -> list:
    from io import BytesIO
    from PIL import Image

    frames = []
    for share in _layout_sample_shares(cue):
        at = cue.at_seconds + cue.duration_seconds * share
        made = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-ss", f"{at:.3f}", "-i", str(picture),
                "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-",
            ],
            capture_output=True,
        )
        if made.returncode == 0 and made.stdout:
            frames.append(Image.open(BytesIO(made.stdout)).convert("RGB"))
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
    stroke_width = round(style.stroke_width * unit)
    shadow_blur = round(style.shadow_blur * unit)
    shadow_x = round(style.shadow_offset_x * unit)
    shadow_y = round(style.shadow_offset_y * unit)
    shadow_room = (
        shadow_blur * 2 + max(abs(shadow_x), abs(shadow_y))
        if style.shadow_opacity else 0
    )
    text_room = card_width - (pad_x + stroke_width + shadow_room) * 2
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
    gap = round(height * 0.010 * render_scale) if secondary else 0
    rule = round(height * 0.005 * render_scale) if spec.rule else 0
    card_height = (
        pad_y * 2 + title_height + secondary_height + gap + rule
        + shadow_room * 2 + stroke_width * 2
    )
    top_safe = round(height * 0.09)
    bottom_safe = round(height * (0.25 if width < height else 0.10))
    if card_height > height - top_safe - bottom_safe:
        raise ValueError(
            f"{cue.graphic_id}: graphic is {card_height}px tall but only "
            f"{height - top_safe - bottom_safe}px fits inside the safe frame"
        )
    canvas = Image.new("RGBA", (card_width, card_height), (0, 0, 0, 0))
    text_mask = Image.new("L", (card_width, card_height), 0)
    pen = ImageDraw.Draw(canvas)
    mask_pen = ImageDraw.Draw(text_mask)

    plate = spec.plate if cue.background == "auto" else cue.background != "none"
    if cue.composition == "foreground_plate":
        plate = True
    if plate:
        radius = (
            round(style.corner_radius * unit)
            if style.corner_radius is not None
            else round(card_height * plan.brand.corner_radius)
        )
        border_width = round(style.plate_border_width * unit)
        pen.rounded_rectangle(
            (0, 0, card_width - 1, card_height - 1),
            radius=radius,
            fill=(*_rgb(plate_colour), (
                style.plate_alpha
                if style.plate_alpha is not None else plan.brand.plate_alpha
            )),
            outline=(*_rgb(style.plate_border_color), 255)
            if border_width else None,
            width=border_width,
        )
    if spec.rule:
        pen.rectangle(
            (pad_x, pad_y, pad_x + round(card_width * 0.13), pad_y + rule),
            fill=(*_rgb(accent), 255),
        )
    backing = canvas.copy()
    y = (
        pad_y + shadow_room + stroke_width + rule
        + (round(height * 0.010 * render_scale) if rule else 0)
    )
    title_width = title_box[2] - title_box[0]
    if align == "center":
        x = (card_width - title_width) // 2
    elif align == "right":
        x = card_width - pad_x - shadow_room - stroke_width - title_width
    else:
        x = pad_x + shadow_room + stroke_width

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
        # Interior glyph pixels only. The contrast auditor deliberately does
        # not let a large translucent plate hide unreadable foreground text.
        mask_pen.multiline_text(
            target, text, font=face, spacing=spacing, align=align, fill=255,
        )
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
        secondary_box = pen.multiline_textbbox(
            (0, 0), secondary, font=secondary_face,
            spacing=secondary_spacing, align=align,
            stroke_width=stroke_width,
        )
        secondary_width = secondary_box[2] - secondary_box[0]
        x = (
            (card_width - secondary_width) // 2
            if align == "center" else (
                card_width - pad_x - shadow_room - stroke_width
                - secondary_width if align == "right"
                else pad_x + shadow_room + stroke_width
            )
        )
        paint_text(
            secondary, secondary_face,
            at_x=x - secondary_box[0], at_y=y - secondary_box[1],
            spacing=secondary_spacing, colour=secondary_colour,
            emphasis=style.emphasis_text, layout_width=secondary_width,
        )

    position = cue.position if cue.position != "auto" else spec.default_position
    positions = _placements(width, height, card_width, card_height)
    left, top = positions.get(position, positions[spec.default_position])
    into.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(into)
    mask_path = into.with_name(f"{into.stem}-text-mask.png")
    backing_path = into.with_name(f"{into.stem}-backing.png")
    text_mask.save(mask_path)
    backing.save(backing_path)
    return DrawnGraphic(
        into, max(0, left), max(0, top), card_width, card_height,
        mask_path, backing_path,
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
    card = draw_graphic(
        cue, plan, width=width, height=height, into=into,
        # Small cards are typeset at full resolution then reduced once; this
        # keeps the font fitter's minimum legible size while preserving a
        # true uniform transform. Enlarged cards are rerasterized sharply.
        render_scale=max(1.0, cue.transform.scale),
    )
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
        # A neutral accessibility fallback, not an inferred brand. It is
        # intentionally local and deterministic so preview and export agree.
        fallback_style = cue.style.model_copy(update={
            "primary_color": "#FFFFFF",
            "secondary_color": "#FFFFFF",
            "emphasis_color": "#FFFFFF",
            "plate_color": "#000000",
            "plate_alpha": 235,
            "stroke_color": "#000000",
            "stroke_width": max(2.0, cue.style.stroke_width),
        })
        fallback_cue = cue.model_copy(update={
            "background": "plate", "style": fallback_style,
        })
        fallback = draw_graphic(
            fallback_cue, plan, width=width, height=height, into=into,
            render_scale=max(1.0, fallback_cue.transform.scale),
        )
        fallback = _apply_authored_transform(
            fallback_cue, fallback, width, height
        )
        card = fallback
        if cue.position == "auto":
            card, scores = resolve_auto_position(
                fallback_cue, card, frames, frame_width=width,
                frame_height=height, evidence=evidence,
                forbidden_positions=forbidden_positions or set(),
                keepout_rects=keepout_rects,
            )
        card = _place_for_motion_safe(cue, card, width, height)
        ratio = measured_text_contrast(
            card, frames or [], cue=fallback_cue,
            frame_width=width, frame_height=height,
            outline_width=round(
                fallback_style.stroke_width * height / 1080.0
                * fallback_cue.transform.scale
            ),
            outline_rgb=_rgb(fallback_style.stroke_color),
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
