"""A creative brief plus the copy it explicitly approves for the screen."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
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
    sha256: str

    @classmethod
    def from_legacy(cls, raw: str) -> "BriefDocument":
        return cls(
            raw=raw,
            creative_brief=raw,
            approved_copy=(),
            sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )


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
    return BriefDocument(
        raw=raw,
        creative_brief=creative,
        approved_copy=tuple(facts),
        sha256=digest,
    )


def load_brief(path: Path | None) -> BriefDocument:
    if path is None:
        return BriefDocument.from_legacy("")
    return parse_brief_markdown(path.read_text(encoding="utf-8"))
