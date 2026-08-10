"""A creative brief plus the copy it explicitly approves for the screen."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from montagewright.graphics import CopyFact

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
        return {
            "brief_sha256": self.sha256,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "instructions": [asdict(note) for note in self.instructions],
        }


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
