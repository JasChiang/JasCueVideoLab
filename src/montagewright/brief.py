"""A creative brief plus the copy it explicitly approves for the screen."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from montagewright.graphics import (
    TEMPLATES,
    CopyFact,
    GraphicCue,
    GraphicStyle,
    GraphicsPlan,
    curated_graphic_family_ids,
    graphic_preset_defaults,
    recommended_graphic_family,
)

APPROVED_COPY_FENCE = re.compile(
    r"```montagewright-approved-copy\s*\n(?P<body>.*?)\n```",
    re.DOTALL,
)


@dataclass(frozen=True)
class BriefDocument:
    raw: str
    creative_brief: str
    approved_copy: tuple[CopyFact, ...]
    candidates: tuple["BriefCandidate", ...]
    instructions: tuple["BriefInstruction", ...]
    sha256: str

    def graphics_candidates(self) -> tuple["BriefCandidate", ...]:
        """Server-issued copy choices Gemini may place on the timeline."""

        projected: list[BriefCandidate] = list(self.candidates)
        for fact in self.approved_copy:
            kind = fact.allowed_kinds[0] if fact.allowed_kinds else "callout"
            template = {
                "opening_title": "hero_center",
                "chapter": "center_stack",
                "product_name": "product_plate",
                "feature": "spec_stack" if len(fact.exact_text) > 12 else "stat_badge",
                "callout": "editorial_rule",
                "end_card": "end_roster",
            }[kind]
            projected.append(BriefCandidate(
                candidate_id=f"approved.{fact.fact_id}",
                primary_text=fact.exact_text,
                secondary_text="",
                kind=kind,
                template=template,
                instruction="使用者在 Brief 中明確核准的原文",
                source_reference=fact.source_reference,
                authority_fact_id=fact.fact_id,
            ))
        return tuple(projected)

    @classmethod
    def from_legacy(cls, raw: str) -> "BriefDocument":
        candidates, instructions = extract_brief_candidates(raw)
        return cls(
            raw=raw,
            creative_brief=raw,
            approved_copy=(),
            candidates=candidates,
            instructions=instructions,
            sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )

    def candidates_json(self) -> dict:
        facts = self.candidate_facts()
        ids = {fact.fact_id: fact for fact in facts}
        return {
            "brief_sha256": self.sha256,
            "candidates": [
                {
                    **asdict(candidate),
                    "primary_fact_id": self.candidate_fact_id(candidate, "primary"),
                    "secondary_fact_id": (
                        self.candidate_fact_id(candidate, "secondary")
                        if candidate.secondary_text else ""
                    ),
                }
                for candidate in self.candidates
            ],
            "facts": [fact.model_dump(mode="json") for fact in ids.values()],
            "instructions": [asdict(note) for note in self.instructions],
        }

    def candidate_facts(self) -> tuple[CopyFact, ...]:
        """Unapproved copy suggestions extracted from ordinary Brief prose.

        The server can prove where these words came from, but extracting a
        phrase is not the same thing as the user approving it for the screen.
        Only the explicit approved-copy manifest has that authority.
        """

        facts: list[CopyFact] = []
        for candidate in self.candidates:
            for which, text in (
                ("primary", candidate.primary_text),
                ("secondary", candidate.secondary_text),
            ):
                if not text:
                    continue
                facts.append(CopyFact(
                    fact_id=self.candidate_fact_id(candidate, which),
                    exact_text=text,
                    source_kind="brief_candidate",
                    source_reference=f"{candidate.source_reference}/{which}",
                    source_sha256=self.sha256,
                    text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    allowed_kinds=[candidate.kind],
                    confidence="certain",
                    approved=False,
                    approved_by=None,
                ))
        return tuple(facts)

    def candidate_fact_id(
        self, candidate: "BriefCandidate", which: str
    ) -> str:
        """A server-reserved fact id that cannot collide with copy ids."""

        return (
            f"brief_candidate.{self.sha256[:12]}."
            f"{candidate.candidate_id}.{which}"
        )


@dataclass(frozen=True)
class CandidateVariant:
    label: str
    primary_text: str
    secondary_text: str = ""


@dataclass(frozen=True)
class BriefCandidate:
    candidate_id: str
    primary_text: str
    secondary_text: str
    kind: str
    template: str
    position: str = "auto"
    instruction: str = ""
    source_reference: str = ""
    variants: tuple[CandidateVariant, ...] = ()
    # Non-empty only for immutable copy from the fenced approved-copy block.
    authority_fact_id: str = ""


@dataclass(frozen=True)
class BriefInstruction:
    source_reference: str
    text: str
    kind: str


PARENTHETICAL = re.compile(r"^[（(](.*)[）)]$")
EDITORIAL_PREFIX = re.compile(r"^(串場|剪輯|畫面|備註)\s*[:：]")
SPECISH = re.compile(
    r"(?:\d|MP\b|mm\b|m\b|吋|螢幕|相機|處理器|認證|邊框|錄影)",
    re.IGNORECASE,
)


def _variants_from_note(
    note: str, primary: str, secondary: str
) -> tuple[CandidateVariant, ...]:
    variants: list[CandidateVariant] = []
    compact = re.search(r"\b\d+(?:\.\d+)?m\b", note, re.IGNORECASE)
    if compact:
        variants.append(CandidateVariant("短版", primary, compact.group(0)))
    alternatives = re.findall(
        r"SAMSUNG\s+Galaxy(?:\s+[A-Za-z0-9| ]+|\s*系列)", note
    )
    for index, text in enumerate(dict.fromkeys(one.strip() for one in alternatives)):
        variants.append(CandidateVariant(
            "替代版" if index == 0 else f"替代版 {index + 1}", text, ""
        ))
    return tuple(variants)


def extract_brief_candidates(
    text: str,
) -> tuple[tuple[BriefCandidate, ...], tuple[BriefInstruction, ...]]:
    """Turn ordinary Markdown stanzas into reviewable copy, never approval."""

    paragraphs = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    candidates: list[BriefCandidate] = []
    instructions: list[BriefInstruction] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        raw_lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        lines = [
            line for line in raw_lines
            if not line.startswith("#") and not set(line) <= {"—", "-"}
        ]
        if not lines:
            continue
        reference = f"/paragraphs/{paragraph_index}"
        if EDITORIAL_PREFIX.match(lines[0]):
            instructions.append(BriefInstruction(reference, " ".join(lines), "editorial"))
            continue
        notes = []
        body = []
        for line in lines:
            parenthetical = PARENTHETICAL.match(line)
            if parenthetical:
                notes.append(parenthetical.group(1).strip())
            else:
                body.append(line)
        if not body:
            instructions.extend(
                BriefInstruction(reference, note, "layout") for note in notes
            )
            continue
        note = "；".join(notes)
        first, rest = body[0], body[1:]
        is_first_card = not candidates
        if first.startswith("就在") or first == "SAMSUNG":
            kind, template, position = "end_card", "end_roster", "center"
        elif "Galaxy" in first and rest and not SPECISH.search(rest[0]):
            kind = "opening_title" if is_first_card else "product_name"
            template = "hero_center" if is_first_card else "product_plate"
            position = "center" if is_first_card else "auto"
        elif len(body) >= 3:
            kind, template, position = "feature", "center_stack", "center"
        elif len(body) == 2 and (SPECISH.search(first) or SPECISH.search(rest[0])):
            kind, template, position = "feature", "spec_stack", "auto"
        elif len(body) == 1 and SPECISH.search(first):
            kind, template, position = "feature", "stat_badge", "auto"
        else:
            kind, template, position = "callout", "editorial_rule", "auto"
        secondary = "\n".join(rest)
        candidates.append(BriefCandidate(
            candidate_id=f"brief.p{paragraph_index:02d}",
            primary_text=first,
            secondary_text=secondary,
            kind=kind,
            template=template,
            position=position,
            instruction=note,
            source_reference=reference,
            variants=_variants_from_note(note, first, secondary),
        ))
        instructions.extend(
            BriefInstruction(reference, one, "layout") for one in notes
        )
    return tuple(candidates), tuple(instructions)


def parse_brief_markdown(raw: str) -> BriefDocument:
    """Parse one explicit copy manifest; prose remains creative direction."""

    matches = list(APPROVED_COPY_FENCE.finditer(raw))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if not matches:
        return BriefDocument.from_legacy(raw)
    if len(matches) > 1:
        raise ValueError("brief may contain only one approved-copy block")
    match = matches[0]
    try:
        payload = json.loads(match.group("body"))
    except json.JSONDecodeError as error:
        raise ValueError(f"approved-copy block is not valid JSON: {error}")
    if payload.get("version") != 1 or not isinstance(payload.get("items"), list):
        raise ValueError("approved-copy block needs version 1 and an items list")
    facts: list[CopyFact] = []
    for index, item in enumerate(payload["items"]):
        if not isinstance(item, dict):
            raise ValueError(f"approved-copy item {index} is not an object")
        copy_id = str(item.get("copy_id") or "").strip()
        text = str(item.get("text") or "")
        allowed = item.get("allowed_kinds") or []
        if not copy_id or not text:
            raise ValueError(f"approved-copy item {index} needs copy_id and text")
        if copy_id.startswith("brief_candidate."):
            raise ValueError(
                f"approved-copy item {index} uses a reserved copy_id prefix"
            )
        facts.append(CopyFact(
            fact_id=copy_id,
            exact_text=text,
            source_kind="brief_exact",
            source_reference=f"/items/{index}/text",
            source_sha256=digest,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            allowed_kinds=allowed,
            approved=True,
            approved_by="user_brief",
        ))
    creative = (raw[:match.start()] + raw[match.end():]).strip()
    candidates, instructions = extract_brief_candidates(creative)
    return BriefDocument(
        raw=raw,
        creative_brief=creative,
        approved_copy=tuple(facts),
        candidates=candidates,
        instructions=instructions,
        sha256=digest,
    )


def load_brief(path: Path | None) -> BriefDocument:
    if path is None:
        return BriefDocument.from_legacy("")
    return parse_brief_markdown(path.read_text(encoding="utf-8"))


def initial_graphics_plan(
    document: BriefDocument,
    selection: dict,
    *,
    shot_durations: list[float],
) -> GraphicsPlan:
    """Materialise Gemini's selection-time overlay decisions locally.

    The model may choose *whether* and *where* a card helps. It may only cite
    candidate ids that came from this exact Brief snapshot; pixels and copy
    authority remain deterministic local work.
    """

    candidates = {
        one.candidate_id: one for one in document.graphics_candidates()
    }
    facts = [*document.approved_copy, *document.candidate_facts()]
    cues: list[GraphicCue] = []
    starts: list[float] = []
    cursor = 0.0
    for duration in shot_durations:
        starts.append(cursor)
        cursor += duration
    used: set[str] = set()
    for coverage in selection.get("covered") or []:
        if not coverage.get("show_as_graphic"):
            continue
        candidate_id = str(coverage.get("graphic_candidate_id") or "")
        candidate = candidates.get(candidate_id)
        indexes = coverage.get("shot_indexes") or []
        if candidate is None or candidate_id in used or not indexes:
            continue
        shot_index = int(coverage.get("graphic_shot_index", indexes[0]))
        if shot_index not in indexes or not 0 <= shot_index < len(shot_durations):
            continue
        duration = shot_durations[shot_index]
        if duration < 0.6:
            continue
        inset = min(0.2, duration * 0.08)
        shown = max(0.4, min(duration - inset, 4.2))
        requested_family = str(
            coverage.get("graphic_design_family") or "auto"
        )
        family = (
            requested_family
            if requested_family in curated_graphic_family_ids()
            else recommended_graphic_family(candidate.kind)
        )
        style_defaults, cue_defaults = graphic_preset_defaults(family)
        surface = str(coverage.get("graphic_surface") or "inherit")
        if surface not in {"inherit", "auto"}:
            style_defaults["surface"] = surface
        style = GraphicStyle.model_validate({
            **style_defaults, "preset": family, "contrast_mode": "auto",
        })
        # The Brief parser's template is driven by copy structure (one-line
        # badge, spec stack, centered roster).  A visual family may restyle it
        # but must not silently destroy that authored hierarchy.
        template = candidate.template
        if template not in TEMPLATES or candidate.kind not in TEMPLATES[template].kinds:
            template = next(
                name for name, spec in TEMPLATES.items()
                if candidate.kind in spec.kinds
            )
        motion = str(coverage.get("graphic_motion") or "inherit")
        if motion == "inherit":
            motion = str(cue_defaults.get("motion") or (
                "fade" if candidate.kind in {"opening_title", "end_card"}
                else "rise"
            ))
        requested_composition = str(
            coverage.get("graphic_composition") or "inherit"
        )
        composition = requested_composition
        if composition == "inherit":
            composition = str(cue_defaults.get("composition") or "auto")
        position = str(
            cue_defaults.get("position") or candidate.position or "auto"
        )
        if candidate.position != "auto":
            position = candidate.position
        # A semantic composition override needs the local layout solver.  A
        # fixed family position would otherwise preserve the value in JSON
        # while making avoid/overlap/auto ineffective in rendered pixels.
        if requested_composition != "inherit":
            position = "auto"
        authoritative = (
            next(
                (fact for fact in document.approved_copy
                 if fact.fact_id == candidate.authority_fact_id),
                None,
            )
            if candidate.authority_fact_id else None
        )
        cues.append(GraphicCue(
            graphic_id=f"brief-{len(cues):02d}",
            kind=candidate.kind,
            primary_fact_id=(
                authoritative.fact_id if authoritative is not None
                else document.candidate_fact_id(candidate, "primary")
            ),
            secondary_fact_id=(
                document.candidate_fact_id(candidate, "secondary")
                if candidate.secondary_text else ""
            ),
            anchor_clip_id=f"k{shot_index:02d}",
            anchor_selection_index=shot_index,
            anchor_offset_seconds=inset,
            at_seconds=starts[shot_index] + inset,
            duration_seconds=shown,
            template=template,
            position=position,
            composition=composition,
            background=str(cue_defaults.get("background") or "auto"),
            motion=motion,
            style=style,
            editor_note="；".join(filter(None, (
                candidate.instruction,
                str(coverage.get("graphic_reason") or ""),
            ))),
            # Gemini chooses the editorial opportunity and timing. It cannot
            # promote ordinary Brief prose into approved on-screen copy.
            status="approved" if authoritative is not None else "draft",
        ))
        used.add(candidate_id)
    return GraphicsPlan(facts=facts, cues=cues)
