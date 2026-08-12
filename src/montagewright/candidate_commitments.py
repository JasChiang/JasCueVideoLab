"""Content commitments that precede rhythm and camera treatment.

A commitment answers "what must this picture prove?" and keeps several
material spans capable of proving it.  Music may choose among those spans and
shape their timing; it may not turn an unrelated or incomplete picture into a
valid answer merely because it lands on a beat.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from montagewright.planning_state import canonical_json
from montagewright.spans import seconds_of


COMMITMENT_VERSION = "candidate-commitment-v1"
PictureRole = Literal[
    "speaker", "primary_action", "illustrative_broll", "reaction",
    "establishing", "transition", "punchline_hold", "end_hold",
    "title_read", "music_montage",
]
PresentationIntent = Literal[
    "complete_hold", "centered_hold", "reveal_endpoint",
    "partial_reveal", "transition_pass",
]
MotionPreference = Literal["native_first", "virtual_allowed", "hold"]
Tier = Literal["primary", "alternate"]


class CommitmentError(ValueError):
    """A proposal cannot be proven by the material catalog."""


class StrictFrozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class CandidateOption(StrictFrozen):
    commitment_id: str = Field(min_length=1, max_length=128)
    purpose: str = Field(min_length=1, max_length=800)
    required: bool
    picture_role: PictureRole
    span_id: str = Field(min_length=1, max_length=512)
    tier: Tier
    min_supported_seconds: float = Field(gt=0.0)
    presentation_intent: PresentationIntent
    motion_preference: MotionPreference
    target_id: str = Field(min_length=1, max_length=256)
    why: str = Field(min_length=1, max_length=1200)


class CandidateCommitments(StrictFrozen):
    contract_version: Literal["candidate-commitment-v1"]
    material_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    direction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_aspect: str = Field(min_length=1, max_length=16)
    target_seconds: float = Field(gt=0.0)
    grounding_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    options: tuple[CandidateOption, ...] = Field(min_length=1)
    deferred_span_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def option_contract_is_coherent(self) -> "CandidateCommitments":
        pairs = [(one.commitment_id, one.span_id) for one in self.options]
        if len(pairs) != len(set(pairs)):
            raise ValueError("commitment/span options must be unique")
        deferred = set(self.deferred_span_ids)
        if len(deferred) != len(self.deferred_span_ids):
            raise ValueError("deferred_span_ids must be unique")
        offered = {one.span_id for one in self.options}
        if deferred & offered:
            raise ValueError("offered and deferred spans must be disjoint")
        grouped: dict[str, list[CandidateOption]] = {}
        for option in self.options:
            grouped.setdefault(option.commitment_id, []).append(option)
        for commitment_id, options in grouped.items():
            if len({one.purpose for one in options}) != 1:
                raise ValueError(f"{commitment_id} has conflicting purposes")
            if len({one.required for one in options}) != 1:
                raise ValueError(f"{commitment_id} has conflicting required flags")
            primaries = [one for one in options if one.tier == "primary"]
            if len(primaries) != 1:
                raise ValueError(
                    f"{commitment_id} must have exactly one primary option"
                )
        return self

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self).encode("utf-8")).hexdigest()

    @property
    def required_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            option.commitment_id for option in self.options if option.required
        ))


def provider_commitment_schema(
    span_ids: Sequence[str], grounding_target_ids: Sequence[str]
) -> dict[str, Any]:
    """Flat provider grammar; local facts and hashes are deliberately absent."""

    return {
        "type": "array",
        "minItems": 1,
        "description": (
            "先定義每個畫面必須履行的內容承諾，再為同一承諾保留 primary "
            "與可替換的 alternate。這不是最後時間軸；節奏只能在這些能履約的"
            "候選中選擇。每個 commitment 是一個預計只出現一次的鏡頭 slot，"
            "恰好一個 primary；需要多顆完成同一故事目的時建立多個 commitment。"
        ),
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "commitment_id", "purpose", "required", "picture_role",
                "span_id", "tier", "min_supported_seconds",
                "presentation_intent", "motion_preference", "target_id", "why",
            ],
            "properties": {
                "commitment_id": {"type": "string"},
                "purpose": {"type": "string"},
                "required": {"type": "boolean"},
                "picture_role": {"type": "string", "enum": list(PictureRole.__args__)},
                "span_id": {"type": "string", "enum": list(span_ids)},
                "tier": {"type": "string", "enum": ["primary", "alternate"]},
                "min_supported_seconds": {
                    "type": "string",
                    "description": "完成這個可見承諾至少要多久，MM:SS。",
                },
                "presentation_intent": {
                    "type": "string", "enum": list(PresentationIntent.__args__)
                },
                "motion_preference": {
                    "type": "string", "enum": list(MotionPreference.__args__)
                },
                "target_id": {
                    "type": "string", "enum": ["none", *grounding_target_ids]
                },
                "why": {"type": "string"},
            },
        },
    }


def resolve_candidate_commitments(
    direction: dict[str, Any],
    material: Sequence[Any],
    *,
    material_digest: str,
    aspect: str,
    target_seconds: float,
    grounding_target_ids: Sequence[str] = (),
    grounding_sha256: str | None = None,
    excluded_source_ids: Sequence[str] = (),
) -> CandidateCommitments:
    """Bind provider options to immutable spans and local motion facts."""

    span_index = {
        span.span_id: span for item in material for span in item.spans
    }
    item_index = {str(item.source_id): item for item in material}
    known_targets = {"none", *grounding_target_ids}
    excluded = set(excluded_source_ids)
    options: list[CandidateOption] = []
    faults: list[str] = []
    for index, raw in enumerate(direction.get("candidate_options") or []):
        span_id = str(raw.get("span_id") or "")
        span = span_index.get(span_id)
        if span is None:
            faults.append(f"option {index} names unknown span {span_id!r}")
            continue
        if str(span.source_id) in excluded:
            faults.append(
                f"option {index} names {span_id}, whose source was ruled broken"
            )
            continue
        seconds = seconds_of(raw.get("min_supported_seconds"))
        if seconds is None or seconds <= 0:
            faults.append(f"option {index} has invalid minimum duration")
            continue
        available = float(span.ends_seconds) - float(span.starts_seconds)
        if seconds > available + 0.001:
            faults.append(
                f"option {index} needs {seconds:.3f}s but {span_id} has "
                f"{available:.3f}s"
            )
            continue
        preference = str(raw.get("motion_preference") or "")
        if preference == "native_first" and span.motion_role not in {
            "authored", "subject_follow"
        }:
            faults.append(
                f"option {index} requests native motion from {span_id}, "
                f"whose role is {span.motion_role}"
            )
            continue
        from montagewright.coverage import visual_supported_max

        supported = visual_supported_max(
            item_index.get(str(span.source_id)),
            role=str(raw.get("picture_role") or ""),
            source_start=float(span.starts_seconds),
            available_seconds=available,
            motion_role=(
                str(span.motion_role or "")
                if preference == "native_first" else ""
            ),
        )
        if seconds > supported + 0.001:
            faults.append(
                f"option {index} needs {seconds:.3f}s but {span_id} has only "
                f"{supported:.3f}s of locally supported "
                f"{raw.get('picture_role') or 'visual'} evidence"
            )
            continue
        target_id = str(raw.get("target_id") or "none")
        if target_id not in known_targets:
            faults.append(f"option {index} names unknown target {target_id!r}")
            continue
        try:
            options.append(CandidateOption(
                commitment_id=raw.get("commitment_id"),
                purpose=raw.get("purpose"),
                required=raw.get("required"),
                picture_role=raw.get("picture_role"),
                span_id=span_id,
                tier=raw.get("tier"),
                min_supported_seconds=seconds,
                presentation_intent=raw.get("presentation_intent"),
                motion_preference=preference,
                target_id=target_id,
                why=raw.get("why"),
            ))
        except Exception as error:
            faults.append(f"option {index} is invalid: {error}")
    if faults:
        raise CommitmentError("; ".join(faults))
    offered = {one.span_id for one in options}
    deferred = tuple(sorted(set(span_index) - offered))
    grouped: dict[str, list[CandidateOption]] = {}
    for option in options:
        grouped.setdefault(option.commitment_id, []).append(option)
    warnings = tuple(
        f"{commitment_id} has a single point of failure"
        for commitment_id, group in grouped.items()
        if group[0].required and len(group) == 1
    )
    try:
        return CandidateCommitments(
            contract_version=COMMITMENT_VERSION,
            material_digest=material_digest,
            direction_sha256=hashlib.sha256(
                canonical_json(direction).encode("utf-8")
            ).hexdigest(),
            target_aspect=aspect,
            target_seconds=target_seconds,
            grounding_sha256=grounding_sha256,
            options=tuple(options),
            deferred_span_ids=deferred,
            warnings=warnings,
        )
    except ValidationError as error:
        raise CommitmentError(
            f"commitment proposal is internally inconsistent: {error}"
        ) from error


def describe_commitments(commitments: CandidateCommitments) -> str:
    lines = []
    for option in commitments.options:
        lines.append(
            f"- {option.commitment_id} [{option.tier}] span={option.span_id}; "
            f"role={option.picture_role}; minimum={option.min_supported_seconds:g}s; "
            f"presentation={option.presentation_intent}; motion="
            f"{option.motion_preference}; target={option.target_id}; "
            f"purpose={option.purpose}"
        )
    return "\n".join(lines)


def validate_selection_commitments(
    shots: Sequence[dict[str, Any]], commitments: CandidateCommitments
) -> list[str]:
    """Return executable membership faults without mutating a selection."""

    allowed: dict[str, set[str]] = {}
    for option in commitments.options:
        allowed.setdefault(option.commitment_id, set()).add(option.span_id)
    seen: dict[str, int] = {}
    faults: list[str] = []
    for index, shot in enumerate(shots):
        commitment_id = str(shot.get("commitment_id") or "")
        span_id = str(shot.get("span_id") or "")
        if commitment_id not in allowed:
            faults.append(f"shot {index} names unknown commitment {commitment_id!r}")
            continue
        if span_id not in allowed[commitment_id]:
            faults.append(
                f"shot {index} uses {span_id} outside commitment {commitment_id}"
            )
        seen[commitment_id] = seen.get(commitment_id, 0) + 1
        option = next((
            one for one in commitments.options
            if one.commitment_id == commitment_id and one.span_id == span_id
        ), None)
        if option is None:
            continue
        seconds = float(shot.get("seconds_needed") or 0.0)
        if seconds + 1e-6 < option.min_supported_seconds:
            faults.append(
                f"shot {index} gives {commitment_id} {seconds:.3f}s but its "
                f"content needs {option.min_supported_seconds:.3f}s"
            )
        intent = str(shot.get("camera_intent") or "hold")
        if option.motion_preference == "native_first" and intent != "use_source_motion":
            faults.append(
                f"shot {index} must use the authored source motion for {commitment_id}"
            )
        if option.motion_preference == "hold" and intent != "hold":
            faults.append(f"shot {index} must hold for {commitment_id}")
        looks = list(shot.get("looks") or [])
        presentations = {
            str(look.get("presentation_intent") or "") for look in looks
        }
        if looks and option.presentation_intent not in presentations:
            faults.append(
                f"shot {index} does not carry presentation intent "
                f"{option.presentation_intent} for {commitment_id}"
            )
        if option.target_id != "none" and option.target_id not in {
            str(look.get("entity_id") or "none") for look in looks
        }:
            faults.append(
                f"shot {index} does not bind target {option.target_id} for "
                f"{commitment_id}"
            )
    for commitment_id in commitments.required_ids:
        count = seen.get(commitment_id, 0)
        if count != 1:
            faults.append(
                f"required commitment {commitment_id} appears {count} times"
            )
    for commitment_id, count in seen.items():
        if count > 1:
            faults.append(f"commitment {commitment_id} appears {count} times")
    return faults


def validate_replacement_commitments(
    failing: Sequence[tuple[int, dict[str, Any], str]],
    replacements: Sequence[dict[str, Any]],
    commitments: CandidateCommitments,
) -> list[str]:
    """Keep a reviewed shot's promise while permitting a different candidate.

    A bounded replan may replace the span or camera treatment, but it cannot
    quietly change what that position in the story was required to prove.
    Structural commitment changes require a new commitment revision instead.
    """

    expected = {
        f"k{index:02d}": str(shot.get("commitment_id") or "")
        for index, shot, _ in failing
    }
    allowed: dict[str, set[str]] = {}
    for option in commitments.options:
        allowed.setdefault(option.commitment_id, set()).add(option.span_id)
    faults: list[str] = []
    seen: set[str] = set()
    for replacement in replacements:
        clip_id = str(replacement.get("replace_clip_id") or "")
        if clip_id not in expected:
            faults.append(f"replacement names unexpected clip {clip_id!r}")
            continue
        if clip_id in seen:
            faults.append(f"replacement names {clip_id} more than once")
            continue
        seen.add(clip_id)
        expected_id = expected[clip_id]
        actual_id = str(replacement.get("commitment_id") or "")
        span_id = str(replacement.get("span_id") or "")
        if actual_id != expected_id:
            faults.append(
                f"{clip_id} changed commitment {expected_id!r} to {actual_id!r}"
            )
            continue
        if span_id not in allowed.get(expected_id, set()):
            faults.append(
                f"{clip_id} uses {span_id} outside commitment {expected_id}"
            )
    missing = set(expected) - seen
    if missing:
        faults.append("missing replacements for " + ", ".join(sorted(missing)))
    return faults


def minimum_supported_seconds_for_shot(
    shot: dict[str, Any], commitments: CandidateCommitments | None
) -> float:
    """Resolve the immutable candidate minimum selected by ``shot``."""

    if commitments is None:
        return 0.0
    commitment_id = str(shot.get("commitment_id") or "")
    span_id = str(shot.get("span_id") or "")
    option = next((
        one for one in commitments.options
        if one.commitment_id == commitment_id and one.span_id == span_id
    ), None)
    return float(option.min_supported_seconds) if option is not None else 0.0
