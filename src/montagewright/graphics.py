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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from montagewright.schema import Local

SourceKind = Literal[
    "user", "brief_exact", "onscreen", "transcript", "model_draft"
]
GraphicKind = Literal[
    "opening_title", "chapter", "product_name", "feature", "callout",
    "end_card",
]
MotionPreset = Literal["none", "fade", "rise", "slide_left", "slide_right"]
PositionHint = Literal[
    "auto", "top", "upper_left", "upper_right", "lower_left",
    "lower_right", "center",
]
CompositionIntent = Literal[
    "auto", "negative_space", "avoid_subject", "overlap_subject",
    "foreground_plate",
]
BackgroundTreatment = Literal["auto", "none", "plate"]
CueStatus = Literal["draft", "approved"]

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


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
    """A small design-token surface, not arbitrary CSS from the planner."""

    font_path: str = ""
    display_font_path: str = ""
    foreground: str = "#FFFFFF"
    secondary: str = "#D9DCE3"
    accent: str = "#FFD65A"
    plate: str = "#111318"
    plate_alpha: int = Field(default=205, ge=0, le=255)
    corner_radius: float = Field(default=0.18, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def colours_are_six_digit_hex(self) -> "BrandKit":
        for field in ("foreground", "secondary", "accent", "plate"):
            if not HEX.match(getattr(self, field)):
                raise ValueError(f"{field} must be #RRGGBB")
        return self


class GraphicCue(Local):
    graphic_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    kind: GraphicKind
    primary_fact_id: str
    secondary_fact_id: str = ""
    # The absolute time is what the finished timeline renders.  The anchor is
    # retained so a later recut can re-resolve it rather than pretending an
    # old clock time still belongs to the same shot.
    anchor_clip_id: str = ""
    anchor_offset_seconds: float = Field(default=0.0, ge=0.0)
    at_seconds: float = Field(default=0.0, ge=0.0)
    duration_seconds: float = Field(default=2.5, gt=0.0, le=30.0)
    template: str = "editorial_rule"
    position: PositionHint = "auto"
    composition: CompositionIntent = "auto"
    background: BackgroundTreatment = "auto"
    motion: MotionPreset = "rise"
    music_sync: Literal["none", "accent", "downbeat"] = "none"
    status: CueStatus = "draft"


class GraphicsPlan(Local):
    version: Literal["montagewright-graphics-v1"] = "montagewright-graphics-v1"
    brand: BrandKit = Field(default_factory=BrandKit)
    facts: list[CopyFact] = Field(default_factory=list)
    cues: list[GraphicCue] = Field(default_factory=list)

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
    ordered = sorted(active, key=lambda cue: cue.at_seconds)
    for index, cue in enumerate(ordered):
        simultaneous = [
            other for other in ordered[index + 1:]
            if other.at_seconds < cue.at_seconds + cue.duration_seconds
        ]
        if len(simultaneous) > 1:
            problems.append(
                f"{cue.graphic_id}: more than two graphics are visible together"
            )
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
        if face.getbbox(text)[2] <= room:
            return face
        size = round(size * 0.94)
    face = _font(max(12, size), text, font_path)
    if face.getbbox(text)[2] > room:
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


@dataclass(frozen=True)
class LayoutEvidence:
    """Subject geometry in final-picture coordinates, supplied by any tracker."""

    # Normalised x, y, width, height boxes across the cue duration. This
    # deliberately does not import SAM: SAM, Vision and a human correction
    # can all provide the same neutral evidence.
    subject_boxes: tuple[tuple[float, float, float, float], ...] = ()
    source: str = "visual_complexity_proxy"


def _placements(
    width: int, height: int, card_width: int, card_height: int
) -> dict[str, tuple[int, int]]:
    side = round(width * 0.075)
    top_safe = round(height * 0.09)
    bottom_safe = round(height * (0.25 if width < height else 0.10))
    return {
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


def resolve_auto_position(
    cue: GraphicCue,
    card: DrawnGraphic,
    frames: list,
    *,
    frame_width: int,
    frame_height: int,
    evidence: LayoutEvidence | None = None,
    forbidden_positions: set[str] | None = None,
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
        candidate = (left, top, right - left, bottom - top)
        overlap = sum(_box_overlap(candidate, box, frame_width, frame_height)
                      for box in proof.subject_boxes)
        if cue.composition in {"auto", "avoid_subject"}:
            score += overlap * 500.0
        elif cue.composition == "overlap_subject":
            score -= overlap * 500.0
        if name == TEMPLATES[cue.template].default_position:
            score -= 0.01  # deterministic tie-break that respects the design
        scores[name] = round(score, 4)
    chosen = min(candidates, key=lambda name: (scores[name], candidates.index(name)))
    left, top = positions[chosen]
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


def _layout_frames(picture: Path, cue: GraphicCue) -> list:
    from io import BytesIO
    from PIL import Image

    frames = []
    for share in (0.12, 0.5, 0.88):
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
) -> DrawnGraphic:
    """Lay out the fully visible hero frame before any motion is applied."""

    from PIL import Image, ImageDraw
    from montagewright.subtitles import cannot_spell

    spec = TEMPLATES.get(cue.template)
    if spec is None:
        raise ValueError(f"unknown graphic template: {cue.template}")
    primary = plan.fact(cue.primary_fact_id).exact_text.strip()
    secondary = (
        plan.fact(cue.secondary_fact_id).exact_text.strip()
        if cue.secondary_fact_id else ""
    )
    unknown = cannot_spell(primary + secondary)
    if unknown:
        raise ValueError(f"no glyph for {unknown}")

    card_width = round(width * spec.width)
    pad_x = round(height * 0.026)
    pad_y = round(height * 0.018)
    title = _fit(
        primary, asked=round(height * spec.title_height),
        room=card_width - pad_x * 2,
        font_path=plan.brand.display_font_path or plan.brand.font_path,
    )
    secondary_face = (
        _fit(
            secondary, asked=round(height * spec.secondary_height),
            room=card_width - pad_x * 2, font_path=plan.brand.font_path,
        )
        if secondary else None
    )
    title_box = title.getbbox(primary)
    title_height = title_box[3] - title_box[1]
    secondary_height = (
        secondary_face.getbbox(secondary)[3] - secondary_face.getbbox(secondary)[1]
        if secondary_face else 0
    )
    gap = round(height * 0.010) if secondary else 0
    rule = round(height * 0.005) if spec.rule else 0
    card_height = pad_y * 2 + title_height + secondary_height + gap + rule
    canvas = Image.new("RGBA", (card_width, card_height), (0, 0, 0, 0))
    pen = ImageDraw.Draw(canvas)

    plate = spec.plate if cue.background == "auto" else cue.background != "none"
    if cue.composition == "foreground_plate":
        plate = True
    if plate:
        radius = round(card_height * plan.brand.corner_radius)
        pen.rounded_rectangle(
            (0, 0, card_width - 1, card_height - 1),
            radius=radius,
            fill=(*_rgb(plan.brand.plate), plan.brand.plate_alpha),
        )
    if spec.rule:
        pen.rectangle(
            (pad_x, pad_y, pad_x + round(card_width * 0.13), pad_y + rule),
            fill=(*_rgb(plan.brand.accent), 255),
        )
    y = pad_y + rule + (round(height * 0.010) if rule else 0)
    title_width = title.getbbox(primary)[2]
    x = (card_width - title_width) // 2 if spec.align == "center" else pad_x
    pen.text((x, y - title_box[1]), primary, font=title,
             fill=(*_rgb(plan.brand.foreground), 255))
    y += title_height + gap
    if secondary_face:
        secondary_box = secondary_face.getbbox(secondary)
        secondary_width = secondary_box[2]
        x = (
            (card_width - secondary_width) // 2
            if spec.align == "center" else pad_x
        )
        pen.text(
            (x, y - secondary_box[1]), secondary, font=secondary_face,
            fill=(*_rgb(plan.brand.secondary), 255),
        )

    position = cue.position if cue.position != "auto" else spec.default_position
    positions = _placements(width, height, card_width, card_height)
    left, top = positions.get(position, positions[spec.default_position])
    into.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(into)
    return DrawnGraphic(into, max(0, left), max(0, top), card_width, card_height)


def burn_graphics(
    picture: Path,
    plan: GraphicsPlan,
    destination: Path,
    *,
    work: Path,
    subtitle_windows: list[tuple[float, float]] | None = None,
    layout_evidence: dict[str, LayoutEvidence] | None = None,
) -> Path:
    """Composite approved graphics and keep the clean picture untouched."""

    from montagewright.measure.media import probe_video
    from montagewright.renderer import probe_duration

    shape = probe_video(picture).video
    width, height = int(shape.display_width), int(shape.display_height)
    duration = probe_duration(picture)
    problems = validate_for_render(
        plan, duration_seconds=duration, subtitle_windows=subtitle_windows
    )
    if problems:
        raise ValueError("; ".join(problems))
    cues = [cue for cue in plan.cues if cue.status == "approved"]
    if not cues:
        raise ValueError("no approved graphics to render")

    drawn = []
    layout_report: dict[str, dict] = {}
    for cue in cues:
        card = draw_graphic(
            cue, plan, width=width, height=height,
            into=work / f"{cue.graphic_id}.png",
        )
        if cue.position == "auto":
            overlaps_subtitles = bool(subtitle_windows and any(
                cue.at_seconds < sub_end
                and cue.at_seconds + cue.duration_seconds > sub_start
                for sub_start, sub_end in subtitle_windows
            ))
            card, scores = resolve_auto_position(
                cue, card, _layout_frames(picture, cue),
                frame_width=width, frame_height=height,
                evidence=(layout_evidence or {}).get(cue.graphic_id),
                forbidden_positions=(
                    {"lower_left", "lower_right"}
                    if overlaps_subtitles else set()
                ),
            )
            fallback_plate = (
                min(scores.values()) > 55.0
                and cue.background == "auto"
                and not TEMPLATES[cue.template].plate
            )
            if fallback_plate:
                plated = draw_graphic(
                    cue.model_copy(update={"background": "plate"}), plan,
                    width=width, height=height,
                    into=work / f"{cue.graphic_id}.png",
                )
                card = replace(plated, left=card.left, top=card.top)
            layout_report[cue.graphic_id] = {
                "resolved_left": card.left, "resolved_top": card.top,
                "resolved_position": min(
                    scores, key=lambda name: scores[name]
                ),
                "frame_width": width, "frame_height": height,
                "card_width": card.width, "card_height": card.height,
                "scores": scores,
                "fallback_plate": fallback_plate,
                "evidence": (
                    (layout_evidence or {}).get(cue.graphic_id).source
                    if (layout_evidence or {}).get(cue.graphic_id)
                    else "visual_complexity_proxy"
                ),
            }
        drawn.append((cue, card))
    work.mkdir(parents=True, exist_ok=True)
    (work / "layout.json").write_text(
        json.dumps(layout_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(picture)]
    for _, card in drawn:
        command += ["-loop", "1", "-framerate", "30", "-i", str(card.path)]

    filters: list[str] = []
    tag = "0:v"
    for index, (cue, card) in enumerate(drawn):
        since = cue.at_seconds
        until = cue.at_seconds + cue.duration_seconds
        enter = min(0.45, max(0.16, cue.duration_seconds * 0.12))
        leave = min(0.32, max(0.12, cue.duration_seconds * 0.08))
        overlay_tag = f"g{index}"
        if cue.motion == "none":
            filters.append(f"[{index + 1}:v]format=rgba[{overlay_tag}]")
        else:
            filters.append(
                f"[{index + 1}:v]format=rgba,"
                f"fade=t=in:st={since:.3f}:d={enter:.3f}:alpha=1,"
                f"fade=t=out:st={until - leave:.3f}:d={leave:.3f}:alpha=1"
                f"[{overlay_tag}]"
            )
        progress = (
            f"min(max((t-{since:.3f})/{enter:.3f},0),1)"
        )
        x, y = str(card.left), str(card.top)
        if cue.motion == "rise":
            y = f"{card.top}+{round(height * .025)}*(1-{progress})"
        elif cue.motion == "slide_left":
            x = f"{card.left}+{round(width * .06)}*(1-{progress})"
        elif cue.motion == "slide_right":
            x = f"{card.left}-{round(width * .06)}*(1-{progress})"
        next_tag = f"v{index}"
        filters.append(
            f"[{tag}][{overlay_tag}]overlay=x='{x}':y='{y}':"
            f"enable='between(t,{since:.3f},{until:.3f})'[{next_tag}]"
        )
        tag = next_tag
    command += [
        "-filter_complex", ";".join(filters),
        "-map", f"[{tag}]", "-map", "0:a?", "-t", f"{duration:.6f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip().splitlines()[-1])
    return destination
