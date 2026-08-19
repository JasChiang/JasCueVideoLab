"""Content commitments that precede rhythm and camera treatment.

A commitment answers "what must this picture prove?" and keeps several
material spans capable of proving it.  Music may choose among those spans and
shape their timing; it may not turn an unrelated or incomplete picture into a
valid answer merely because it lands on a beat.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Sequence, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from montagewright.clipcard import (
    GEOMETRY_BASIS_REFERRING,
    GEOMETRY_BASIS_TRACKED,
)
from montagewright.planning_state import canonical_json
from montagewright.spans import seconds_of


COMMITMENT_VERSION = "candidate-commitment-v6-outcome-endpoint"
PictureRole = Literal[
    "speaker", "primary_action", "illustrative_broll", "reaction",
    "establishing", "transition", "punchline_hold", "end_hold",
    "title_read", "music_montage",
]
PresentationIntent = Literal[
    "complete_hold", "centered_hold", "reveal_endpoint",
    "sequential_read", "partial_reveal", "transition_pass",
]
MotionPreference = Literal["native_first", "virtual_allowed", "hold"]
CameraTreatment = Literal[
    "hold", "use_source_motion", "follow_subject", "reveal", "compare",
    "push_in", "pull_out", "multi_stop",
]
SuggestedMove = Literal[
    "hold", "source_motion", "pan", "tilt", "diagonal", "push_in",
    "pull_out", "follow", "compound",
]
Tier = Literal["primary", "alternate"]
ContentPolicy = Literal[
    "complete_action",
    "representative_excerpt",
    "result_hold",
    "continuous_process",
    "static_display",
]
VisualRelationship = Literal[
    "single", "simultaneous", "ordered", "action_sequence",
]


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
    content_policy: ContentPolicy = "static_display"
    # The semantic action Direction chose for this content promise.  This is
    # provider-authored as an id, then resolved to immutable source-clock
    # facts below.  Selection must not freely pair a purpose with a different
    # nearby action merely because both happen to be in the same take.
    content_action_id: str = "none"
    content_action_start_seconds: float | None = None
    content_action_complete_seconds: float | None = None
    required_visuals: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    outcome_evidence: str | None = None
    visual_relationship: VisualRelationship = "single"
    motion_preference: MotionPreference
    target_id: str = Field(min_length=1, max_length=256)
    why: str = Field(min_length=1, max_length=1200)
    # Direction owns the editorial idea.  These fields are advice carried to
    # Selection, never geometry truth: the local feasibility menu below is
    # still the authority on what the renderer can actually deliver.
    direction_treatment: CameraTreatment = "hold"
    direction_suggested_move: SuggestedMove = "hold"
    direction_camera_route: str = Field(default="keep the useful framing", max_length=800)
    direction_motion_reason: str = Field(default="no directional advice", max_length=1200)
    direction_fallback_treatment: CameraTreatment = "hold"
    # Local feasibility, derived from the selected source/span rather than
    # invented by the provider.  Selection and replan see the menu in ranked
    # order, so "hold" is no longer the only obviously safe answer when the
    # take has measured room for a useful treatment.
    feasible_treatments: tuple[CameraTreatment, ...] = ("hold",)
    preferred_treatment: CameraTreatment = "hold"
    minimum_camera_seconds: float = Field(default=0.0, ge=0.0)
    feasibility_reason: str = "local fallback: hold"
    identity_status: Literal["eligible", "needs_review"] = "eligible"
    identity_issue: str = ""


# The smallest gap between two landings worth crossing, shared with the crop
# compiler's own deadband. Stated here because this module decides which
# treatments are offered and the compiler decides whether they moved, and the
# two must agree about what counts as travel.
READABLE_MARGIN = 0.02


def readable_extent(
    item: Any, geometry: Any, required_visuals: "Sequence[str]",
) -> float | None:
    """How wide this option's content is, edge to edge, in source widths.

    A treatment that carries the eye across something needs something wider
    than one delivery crop to carry it across. That fact is measured, sits in
    the card, and belongs to the option rather than to any later stage.

    Returns ``None`` when the card cannot place every named visual. Unknown
    geometry is not evidence that a move is impossible, and withholding a
    treatment on it would be a new way to lose moves the footage supports.
    """

    entries = tuple(geometry or ())
    wanted = set(required_visuals or ())
    if not entries or not wanted:
        return None
    placed = {
        f"v{at:02d}": entry
        for at, entry in enumerate(entries, start=1)
        if f"v{at:02d}" in wanted
    }
    if len(placed) != len(wanted):
        return None
    left = min(float(one[2]) - float(one[4]) / 2.0 for one in placed.values())
    right = max(float(one[2]) + float(one[4]) / 2.0 for one in placed.values())
    return max(0.0, right - left)


def extent_basis(geometry: Any, required_visuals: "Sequence[str]") -> str:
    """Tracked only when every named visual has been measured.

    One unmeasured landing makes the whole span an estimate: the distance a
    read is priced on is between its ends, and an end that is still a phrase
    moves once something measures it.
    """

    entries = tuple(geometry or ())
    wanted = set(required_visuals or ())
    rows = [
        entry for at, entry in enumerate(entries, start=1)
        if f"v{at:02d}" in wanted
    ]
    if not rows or len(rows) != len(wanted):
        return GEOMETRY_BASIS_REFERRING
    if all(len(one) > 6 and one[6] == GEOMETRY_BASIS_TRACKED for one in rows):
        return GEOMETRY_BASIS_TRACKED
    return GEOMETRY_BASIS_REFERRING


def _camera_treatments(
    item: Any,
    span: Any,
    *,
    preference: MotionPreference,
    target_id: str,
    content_extent: float | None = None,
    content_basis: str = GEOMETRY_BASIS_REFERRING,
) -> tuple[tuple[CameraTreatment, ...], CameraTreatment, float, str]:
    """Rank treatments using facts that exist before the camera is planned."""

    seconds = max(0.0, float(span.ends_seconds) - float(span.starts_seconds))
    role = str(getattr(span, "motion_role", "") or "locked")
    pan_room = float(getattr(item, "pan_room", 0.0) or 0.0)
    tilt_room = float(getattr(item, "tilt_room", 0.0) or 0.0)
    push_room = float(getattr(item, "push_room", 1.0) or 1.0)
    crop_width = float(getattr(item, "crop_width", 1.0) or 1.0)
    treatments: list[CameraTreatment] = []
    reasons: list[str] = []
    minimum = 0.0

    if preference == "native_first" and role in {"authored", "subject_follow"}:
        treatments.append("use_source_motion")
        reasons.append(f"span has {role} source motion")
    # Direction's preference ranks storytelling options; it must not erase
    # measured abilities before Selection has watched the actual clip. That
    # mistake made `hold` self-fulfilling: a conservative preference removed
    # every pan/push treatment from the schema, so Gemini could never choose
    # the movement the footage clearly supported.
    travel_room = max(pan_room, tilt_room)
    # Frame room is the only thing this layer can honestly rank on. Whether
    # there is anything worth crossing it for depends on what the shot ends
    # up looking at, and Selection is free to name landings this contract
    # never listed -- 7 of 28 looks in one delivered film, 26 of 26 in
    # another. Withholding a treatment here on the content contract's own
    # span therefore removes moves from shots that were going to look
    # somewhere else. The narrow case is real, but it is exact only once the
    # looks are written and measured; it is answered by the compiler, which
    # already reports landings that resolve to one place.
    if travel_room > 0.02 and seconds >= 1.0:
        treatments.extend(("reveal", "compare"))
        minimum = max(minimum, 1.0)
        reasons.append(f"crop has {travel_room:.0%} measured travel room")
    if push_room > 1.02 and seconds >= 1.0:
        treatments.extend(("push_in", "pull_out"))
        minimum = max(minimum, 1.0)
        reasons.append(f"resolution permits up to {push_room:.2f}x push")
    if content_extent is not None:
        # Stated with its provenance, never enforced. An extent still made of
        # referring boxes describes the phrases, not the objects a crop will
        # follow, and the two differ on exactly the axis a read is measured
        # along -- so the number is offered and its standing is offered with
        # it, rather than one being quietly used as the other.
        reasons.append(
            f"named content spans {content_extent:.0%} against a "
            f"{crop_width:.0%} crop ("
            + ("measured" if content_basis == GEOMETRY_BASIS_TRACKED
               else "estimated from referring boxes")
            + ")"
        )
    if travel_room > 0.02 and seconds >= 1.8:
        treatments.append("multi_stop")
        minimum = max(minimum, 1.8)
    # A named target makes following possible, not necessarily useful. Put
    # it after moves supported by measured crop room: ranking follow first
    # for every static product shot told Selection to track a subject that
    # was not moving, and the renderer correctly collapsed that to hold.
    if target_id != "none":
        treatments.append("follow_subject")
        reasons.append("named subject can be tracked locally if it moves")
    treatments.append("hold")
    ranked = tuple(dict.fromkeys(treatments))
    preferred = ranked[0]
    return ranked, preferred, minimum, "; ".join(reasons) or "no measured move room"


class CandidateCommitments(StrictFrozen):
    contract_version: Literal[
        "candidate-commitment-v1", "candidate-commitment-v2-content-policy",
        "candidate-commitment-v3-action-bound-content-policy",
        "candidate-commitment-v4-sequential-read",
        "candidate-commitment-v5-visual-relationships",
        "candidate-commitment-v6-outcome-endpoint",
    ]
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


def _role_budget_sentence() -> str:
    """The ceilings, said in the schema that is judged against them.

    The planner was asked for a role and a length, told nothing about how
    long each role can carry, and then had its answer refused locally --
    three paid corrections in a row proposing a three-second `transition`,
    which is an ordinary connective shot in a music cut and a contract
    violation here. Generated from the table that does the refusing, so the
    two cannot drift apart.
    """

    from montagewright.coverage import VISUAL_ONLY_LIMITS

    return "、".join(
        f"{role} {limit:.2f}" if limit is not None else f"{role} 由內容決定"
        for role, limit in VISUAL_ONLY_LIMITS.items()
    )


def provider_commitment_schema(
    span_ids: Sequence[str], grounding_target_ids: Sequence[str],
    action_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Flat provider grammar; local facts and hashes are deliberately absent."""

    return {
        "type": "array",
        "minItems": 1,
        "description": (
            "先定義每個畫面必須履行的內容承諾，再為同一承諾保留 primary "
            "與可替換的 alternate。這不是最後時間軸；節奏只能在這些能履約的"
            "候選中選擇。每個 commitment 是一個「故事點」，可以由一顆或多顆"
            "鏡頭覆蓋（例如全景＋推近＋細節共同證明同一件事）；恰好一個 "
            "primary。同一故事點若給多顆，每顆必須明顯不同——不同來源、"
            "景別差一級以上、或拍帶中不重疊的另一時刻；同一段源時間絕不放兩次。"
            "primary 與 alternate 若視覺上幾乎一樣（例如同一面招牌的兩個角度），"
            "別把兩個都當鏡頭用，挑最好的一支。"
        ),
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "commitment_id", "purpose", "required", "picture_role",
                "span_id", "tier", "min_supported_seconds",
                "presentation_intent", "content_policy", "content_action_id",
                "required_visuals", "required_evidence", "outcome_evidence",
                "visual_relationship",
                "target_id", "why",
                "recommended_treatment", "suggested_move", "camera_route",
                "motion_reason", "fallback_treatment",
            ],
            "properties": {
                "commitment_id": {"type": "string"},
                "purpose": {"type": "string"},
                "required": {"type": "boolean"},
                "picture_role": {
                    "type": "string",
                    "enum": list(PictureRole.__args__),
                    "description": (
                        "這顆為什麼要被看見。每個角色能自己撐住的畫面時間有"
                        "上限（秒）：" + _role_budget_sentence() + "。"
                        "沒有旁白時整顆的長度都要由畫面自己證明，所以這些"
                        "上限就是實際可用秒數；該區段有實測到的動作或運鏡時"
                        "才可以超過。"
                    ),
                },
                "span_id": {"type": "string", "enum": list(span_ids)},
                "tier": {"type": "string", "enum": ["primary", "alternate"]},
                "min_supported_seconds": {
                    "type": "string",
                    "description": (
                        "完成這個可見承諾至少要多久，MM:SS。不能超過所選 "
                        "picture_role 的上限；要更長就換一個撐得住的角色"
                        "（例如 establishing 或 music_montage），不要用 "
                        "transition 去要三秒。"
                    ),
                },
                "presentation_intent": {
                    "type": "string", "enum": list(PresentationIntent.__args__),
                    "description": (
                        "畫面如何被讀懂。寬文字、橫向 UI、產品列在直式比例"
                        "無法同幀看全，但可以從一端讀到另一端時，選 "
                        "sequential_read 並建議 reveal 或 multi_stop；本機會依"
                        "主體實測範圍產生兩到三個落點並在落點短暫停留。只有"
                        "真的必須同一瞬間完整看見時才用 must_be_whole 的"
                        "完整型意圖並換較寬素材。"
                    ),
                },
                "content_policy": {
                    "type": "string", "enum": list(ContentPolicy.__args__),
                    "description": (
                        "這顆如何履行內容：complete_action 必須從具名動作開始"
                        "看到完成；representative_excerpt 可有理由地取代表片段；"
                        "result_hold 是結果已出現後的可讀停留；continuous_process "
                        "是瀏覽、旋轉、舞蹈等沒有唯一終點的持續過程；"
                        "static_display 是建立場景、產品細節或靜態資訊。需要先操作"
                        "再看結果時，不要把長流程硬塞一顆：建立兩個相鄰 commitment。"
                        "目前執行器沒有任意 speed-ramp 契約；不要用較短秒數假裝"
                        "快轉完整動作。冗長但無唯一終點的中段可選 "
                        "representative_excerpt／continuous_process，具因果的起點與"
                        "結果則拆成相鄰 commitment，直到時間映射能被逐段驗證。"
                    ),
                },
                "content_action_id": {
                    "type": "string",
                    "enum": ["none", *dict.fromkeys(action_ids)],
                    "description": (
                        "content_policy 所指的素材卡具名動作。complete_action "
                        "必須選一個能在 span 內完整開始並完成的 action id；"
                        "result_hold 只有在這個 span 仍包含動作完成點、且完成後"
                        "留得下 min_supported_seconds 時才填該 id，否則填 none；"
                        "representative_excerpt／continuous_process 只有刻意在該"
                        "動作中取樣時才填 id；static_display 一律填 none。"
                    ),
                },
                "required_visuals": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {"type": "string"},
                    "description": (
                        "Copy only the stable v01/v02... ids shown beside this "
                        "source's 可框住的主體. Never translate or rewrite their "
                        "labels. Include every spatial participant in an interaction: "
                        "actor/hand, tool/product, target/document, and result "
                        "when it has a v-id. This applies equally to tutorials, "
                        "people holding products, comparisons, cooking, UI, "
                        "performances and scene reveals."
                    ),
                },
                "required_evidence": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {"type": "string"},
                    "description": (
                        "Semantic proof that has no v-id/box: an interface state, "
                        "recognition result, completed gesture or causal outcome. "
                        "Do not put it in required_visuals. Named source actions "
                        "remain bound separately by content_action_id."
                    ),
                },
                "outcome_evidence": {
                    "type": "string",
                    "description": (
                        "For complete_action/result_hold, copy exactly one item "
                        "from required_evidence that must be visible at the final "
                        "stable landing after the action completes. Write none "
                        "when there is no causal visual result. Do not name the "
                        "carrier object here; it remains in required_visuals."
                    ),
                },
                "visual_relationship": {
                    "type": "string",
                    "enum": list(VisualRelationship.__args__),
                    "description": (
                        "single: one subject carries the beat; simultaneous: "
                        "all required visuals must be readable in the same "
                        "landing; ordered: the camera may read them in order; "
                        "action_sequence: their setup/action/result must be "
                        "preserved across the selected action window."
                    ),
                },
                "recommended_treatment": {
                    "type": "string", "enum": list(CameraTreatment.__args__),
                    "description": (
                        "依敘事目的與目標輸出比例建議的運鏡 treatment。這是"
                        "創作建議，不是本機能力宣告；不要因為不確定就一律 hold。"
                    ),
                },
                "suggested_move": {
                    "type": "string", "enum": list(SuggestedMove.__args__),
                    "description": (
                        "建議的實體畫框運動。寬內容進直式畫面時可考慮 pan，"
                        "高度方向可考慮 tilt；景別變化用 push/pull；主體自己動"
                        "才用 follow；多段敘事可用 compound。"
                    ),
                },
                "camera_route": {
                    "type": "string",
                    "description": (
                        "用畫面中的可辨識落點描述起點、終點與必要停留，例如"
                        "由左側標誌讀到右側產品後停住。不要寫座標或小數秒。"
                    ),
                },
                "motion_reason": {
                    "type": "string",
                    "description": (
                        "說明這個運鏡為何適合原素材、故事目的與目標比例；若"
                        "原生運鏡應保留，也要明說。"
                    ),
                },
                "fallback_treatment": {
                    "type": "string", "enum": list(CameraTreatment.__args__),
                    "description": "首選幾何不可交付時，仍保留同一敘事目的的替代 treatment。",
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
        # Direction decides what the shot must prove, not its final camera
        # move. Derive only the source-motion fact here; Selection watches the
        # actual clip and chooses camera_intent directly. Older cached
        # Directions may still contain motion_preference and are intentionally
        # ignored so a stale `hold` cannot veto a later push or reveal.
        preference = cast(MotionPreference, (
            "native_first"
            if str(span.motion_role or "") in {"authored", "subject_follow"}
            else "virtual_allowed"
        ))
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
            presentation_intent=str(raw.get("presentation_intent") or ""),
            target_id=str(raw.get("target_id") or "none"),
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
        content_policy = str(raw.get("content_policy") or "static_display")
        content_action_id = str(raw.get("content_action_id") or "none")
        local_item = item_index.get(str(span.source_id))
        requested_visuals = tuple(dict.fromkeys(
            str(value).strip()
            for value in (raw.get("required_visuals") or ())
            if str(value).strip()
        ))
        requested_evidence = tuple(dict.fromkeys(
            str(value).strip()
            for value in (raw.get("required_evidence") or ())
            if str(value).strip()
        ))
        requested_outcome = str(raw.get("outcome_evidence") or "none").strip()
        relationship = str(raw.get("visual_relationship") or "single")
        geometry = tuple(getattr(local_item, "subject_geometry", ()) or ())
        visual_ids = {
            f"v{at:02d}": str(entry[0])
            for at, entry in enumerate(geometry, start=1)
        }
        ids_by_label = {label: visual_id for visual_id, label in visual_ids.items()}
        # v5 provider answers use stable ids. A cached v4-era Direction from
        # the first paid attempt used the exact card label; migrate that
        # locally so resuming does not buy Direction again. Anything it named
        # that never had geometry is semantic evidence, not a phantom bbox.
        is_stable_answer = "required_evidence" in raw
        required_visuals: list[str] = []
        migrated_evidence = list(requested_evidence)
        unknown_visuals: list[str] = []
        for value in requested_visuals:
            if value in visual_ids:
                required_visuals.append(value)
            elif value in ids_by_label:
                required_visuals.append(ids_by_label[value])
            elif is_stable_answer:
                unknown_visuals.append(value)
            else:
                migrated_evidence.append(value)
        if unknown_visuals:
            faults.append(
                f"option {index} names unknown visual ids for "
                f"{span.source_id}: {', '.join(unknown_visuals)}"
            )
            continue
        if requested_outcome == "none" and (
            "outcome_evidence" not in raw
            and content_policy in {"complete_action", "result_hold"}
            and len(migrated_evidence) == 1
        ):
            # Safe migration for cached v5 Directions such as the Pixel LED
            # run. Multiple semantic facts are ambiguous and are never
            # guessed into an endpoint.
            requested_outcome = migrated_evidence[0]
        outcome_evidence = (
            requested_outcome if requested_outcome != "none" else None
        )
        if outcome_evidence is not None and outcome_evidence not in migrated_evidence:
            faults.append(
                f"option {index} outcome_evidence must copy one item from "
                "required_evidence"
            )
            continue
        if relationship == "simultaneous" and len(required_visuals) < 2:
            faults.append(
                f"option {index} says simultaneous but names fewer than two visuals"
            )
            continue
        if relationship == "action_sequence" and content_action_id == "none":
            faults.append(
                f"option {index} says action_sequence but binds no content action"
            )
            continue
        actions_reaching_span = tuple(
            (str(action_id), float(start), float(end))
            for action_id, start, end in (
                getattr(local_item, "action_windows", ()) or ()
            )
            if float(start) <= float(span.ends_seconds) + 1e-6
            and float(end) >= float(span.starts_seconds) - 1e-6
        )
        actions_inside_span = tuple(
            action for action in actions_reaching_span
            if action[1] >= float(span.starts_seconds) - 1e-6
            and action[2] <= float(span.ends_seconds) + 1e-6
        )
        selected_action = next((
            action for action in actions_reaching_span
            if action[0] == content_action_id
            or action[0] == content_action_id.rsplit(":", 1)[-1]
        ), None)
        if content_action_id != "none" and selected_action is None:
            faults.append(
                f"option {index} binds action {content_action_id!r}, which is "
                f"not reachable inside {span_id}"
            )
            continue
        if content_policy == "complete_action":
            if content_action_id == "none" or selected_action not in actions_inside_span:
                faults.append(
                    f"option {index} promises complete_action but does not bind "
                    f"a locally timed action that starts and completes inside {span_id}; "
                    "name content_action_id or choose another content policy"
                )
                continue
        elif content_policy == "result_hold" and selected_action is not None:
            remaining = float(span.ends_seconds) - selected_action[2]
            if remaining + 1e-6 < seconds:
                faults.append(
                    f"option {index} binds result_hold to {content_action_id!r}, "
                    f"but only {remaining:.3f}s remains after completion inside "
                    f"{span_id}; use content_action_id=none for an already-visible "
                    "result span or choose a later span"
                )
                continue
        elif content_policy == "static_display" and content_action_id != "none":
            faults.append(
                f"option {index} marks static_display but binds action "
                f"{content_action_id!r}; use none"
            )
            continue
        effective_seconds = seconds
        if content_policy == "complete_action" and selected_action is not None:
            effective_seconds = max(
                seconds, selected_action[2] - selected_action[1]
            )
        if effective_seconds > available + 0.001:
            faults.append(
                f"option {index} needs {effective_seconds:.3f}s after binding "
                f"{content_action_id!r}, but {span_id} has {available:.3f}s"
            )
            continue
        if effective_seconds > supported + 0.001:
            faults.append(
                f"option {index} needs {effective_seconds:.3f}s after binding "
                f"{content_action_id!r}, but {span_id} has only "
                f"{supported:.3f}s of locally supported evidence"
            )
            continue
        # A source the screen found the identity absent from may still be
        # promised it here, and that disagreement is not this stage's to
        # settle. The screen reads a 640-pixel proxy at a frame a second and
        # is wrong in exactly the way that resolution predicts: it called a
        # table of three handsets absent because the two rings on the middle
        # one are small at that size. Direction watched the same clip and
        # said the product was there -- and it was. Refusing the option
        # spent two paid corrections arguing with a model that was right,
        # and then killed the run. The promotion happens before this, in the
        # caller; what is left here catches a source that was never offered
        # for promotion at all.
        try:
            # One measurement, two decisions: whether the eye has to be
            # carried across this content at all, and -- below -- whether a
            # centred hold could ever have shown it. They disagreed while
            # each computed the span itself, so a treatment could be offered
            # for content the very next block proved a crop already contains.
            content_extent = readable_extent(
                local_item, geometry, required_visuals
            )
            content_basis = extent_basis(geometry, required_visuals)
            treatments, preferred, camera_floor, feasibility_reason = (
                _camera_treatments(
                    item_index.get(str(span.source_id)),
                    span,
                    preference=preference,
                    target_id=target_id,
                    content_extent=content_extent,
                    content_basis=content_basis,
                )
            )
            direction_treatment = cast(CameraTreatment, str(
                raw.get("recommended_treatment") or "hold"
            ))
            presentation_intent = str(
                raw.get("presentation_intent") or "centered_hold"
            )
            # A wide required visual or ordered group cannot be made readable
            # by centring one narrow delivery crop. Promote that promise to
            # an edge-to-edge sequential read using source geometry already
            # measured for this exact aspect. This is content-agnostic: it
            # applies equally to signage, a UI, people, or a product row.
            if (
                relationship in {"single", "ordered"}
                and required_visuals
                # A wide carrier does not imply that the viewer must read the
                # whole carrier. When Direction names a causal result without
                # its own box, that result -- not the carrier edge -- is the
                # endpoint of the shot.
                and not (
                    outcome_evidence
                    and content_policy in {"complete_action", "result_hold"}
                )
                and local_item is not None
                and presentation_intent not in {
                    "sequential_read", "partial_reveal", "transition_pass",
                }
            ):
                if content_extent is not None and content_extent > float(
                    getattr(local_item, "crop_width", 1.0)
                ) + READABLE_MARGIN:
                    presentation_intent = "sequential_read"
            if presentation_intent == "sequential_read":
                readable = tuple(
                    treatment for treatment in (
                        "reveal", "multi_stop", "use_source_motion"
                    )
                    if treatment in treatments
                )
                if not readable:
                    # Direction promised a read this crop has no room to
                    # perform. The promise cannot be kept, but the shot can:
                    # dropping the option would shrink the pool over a framing
                    # detail and can cost a commitment its only take. Keep it
                    # as the composition it actually is.
                    presentation_intent = "centered_hold"
                    feasibility_reason = (
                        f"{feasibility_reason}; Direction asked for a "
                        f"sequential read and this crop has no travel room, "
                        f"so this is a held composition"
                    )
            if presentation_intent == "sequential_read":
                if direction_treatment in readable:
                    preferred = direction_treatment
                else:
                    # Geometry is a local fact discovered while binding the
                    # Direction answer.  If that fact promotes a centred hold
                    # into a sequential read, the provider's earlier hold,
                    # push, or follow preference is advice about a contract
                    # that no longer exists.  Select the least elaborate
                    # locally executable reader instead of buying correction
                    # calls merely to have Gemini repeat that fact.
                    preferred = (
                        "multi_stop"
                        if (
                            relationship == "ordered"
                            and len(required_visuals) > 2
                            and "multi_stop" in readable
                        )
                        else "reveal"
                        if "reveal" in readable
                        else readable[0]
                    )
                    feasibility_reason = (
                        f"{feasibility_reason}; local aspect geometry promoted "
                        f"the content to sequential_read and selected {preferred} "
                        f"instead of advisory {direction_treatment}"
                    )
            elif direction_treatment in treatments:
                preferred = direction_treatment
            options.append(CandidateOption(
                commitment_id=raw.get("commitment_id"),
                purpose=raw.get("purpose"),
                required=raw.get("required"),
                picture_role=raw.get("picture_role"),
                span_id=span_id,
                tier=raw.get("tier"),
                min_supported_seconds=effective_seconds,
                presentation_intent=presentation_intent,
                # Legacy cached/test Directions predate the field. They are
                # treated as static rather than being granted permission to
                # cut a named action; every new provider answer is required
                # by the v2 schema to make the policy explicit.
                content_policy=content_policy,
                content_action_id=content_action_id,
                content_action_start_seconds=(
                    selected_action[1] if selected_action is not None else None
                ),
                content_action_complete_seconds=(
                    selected_action[2] if selected_action is not None else None
                ),
                required_visuals=tuple(dict.fromkeys(required_visuals)),
                required_evidence=tuple(dict.fromkeys(migrated_evidence)),
                outcome_evidence=outcome_evidence,
                visual_relationship=cast(VisualRelationship, relationship),
                motion_preference=preference,
                target_id=target_id,
                why=raw.get("why"),
                direction_treatment=direction_treatment,
                direction_suggested_move=raw.get("suggested_move") or "hold",
                direction_camera_route=(
                    raw.get("camera_route") or "keep the useful framing"
                ),
                direction_motion_reason=(
                    raw.get("motion_reason") or "legacy direction gave no motion advice"
                ),
                direction_fallback_treatment=(
                    raw.get("fallback_treatment") or "hold"
                ),
                feasible_treatments=treatments,
                preferred_treatment=preferred,
                minimum_camera_seconds=camera_floor,
                feasibility_reason=feasibility_reason,
            ))
        except Exception as error:
            faults.append(f"option {index} is invalid: {error}")
    if faults and not options:
        raise CommitmentError("; ".join(faults))
    # The model owns which options exist; the local contract owns that they
    # are internally coherent. Direction is a model answer that does not
    # always keep that coherence -- a commitment can come back with two
    # options for one span, with its options disagreeing on purpose or
    # required, or with no primary among them. Each is a bookkeeping slip a
    # single deterministic pass can settle, and none is a reason to throw away
    # a paid direction and stop the film. The model's structural invariant
    # stays as the last line of defence; it now holds by construction.
    repairs: list[str] = []

    seen_pairs: set[tuple[str, str]] = set()
    deduped: list[CandidateOption] = []
    for option in options:
        pair = (option.commitment_id, option.span_id)
        if pair in seen_pairs:
            repairs.append(
                f"{option.commitment_id} offered {option.span_id} twice; "
                "kept the first"
            )
            continue
        seen_pairs.add(pair)
        deduped.append(option)
    options = deduped

    grouped: dict[str, list[CandidateOption]] = {}
    for option in options:
        grouped.setdefault(option.commitment_id, []).append(option)

    changed = bool(repairs)
    for commitment_id, group in grouped.items():
        purpose = group[0].purpose
        required = group[0].required
        if len({one.purpose for one in group}) != 1:
            repairs.append(
                f"{commitment_id} options disagreed on purpose; used the "
                "primary's"
            )
        if len({one.required for one in group}) != 1:
            repairs.append(
                f"{commitment_id} options disagreed on required; used the "
                "primary's"
            )
        primaries = [one for one in group if one.tier == "primary"]
        if len(primaries) != 1:
            repairs.append(
                f"{commitment_id} had {len(primaries)} primary options; kept "
                "one primary and made the rest alternates"
            )
        for position, option in enumerate(group):
            updates: dict[str, Any] = {}
            if option.purpose != purpose:
                updates["purpose"] = purpose
            if option.required != required:
                updates["required"] = required
            want = "primary" if position == 0 else "alternate"
            if len(primaries) != 1 and option.tier != want:
                updates["tier"] = want
            if updates:
                group[position] = option.model_copy(update=updates)
                changed = True

    if changed:
        # Rebuild in the original order, drawing each commitment's repaired
        # options in sequence, so only the fields above changed.
        cursors = {commitment_id: 0 for commitment_id in grouped}
        rebuilt: list[CandidateOption] = []
        for option in options:
            group = grouped[option.commitment_id]
            rebuilt.append(group[cursors[option.commitment_id]])
            cursors[option.commitment_id] += 1
        options = rebuilt

    offered = {one.span_id for one in options}
    deferred = tuple(sorted(set(span_index) - offered))

    warnings = tuple(dict.fromkeys([
        *faults,
        *repairs,
        *(
            f"{commitment_id} has a single point of failure"
            for commitment_id, group in grouped.items()
            if group[0].required and len(group) == 1
        ),
    ]))
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
    import collections

    # Which distinct sources back each commitment. When a commitment is down
    # to one take -- often because identity confirmation removed its other
    # option -- Selection must be told, or it fills the beat by cutting the
    # same window twice, which reads as a repeated image. The editorial choice
    # (vary the framing, use a different moment, or spend one shot) is the
    # model's; the fact is local, and belongs in the brief it reads.
    sources: dict[str, set[str]] = collections.defaultdict(set)
    for option in commitments.options:
        sources[option.commitment_id].add(option.span_id.split(":", 1)[0])
    single_source = sorted(
        cid for cid, srcs in sources.items() if len(srcs) == 1
    )
    lines = []
    if single_source:
        lines.append(
            "只有一支來源的 commitment（若要給它多顆鏡頭，兩顆必須明顯不同："
            "景別差一級以上、或用拍帶中不重疊的另一時刻；絕不可把同一段源時間"
            "放兩次，那讀起來是重複畫面。素材真的只夠一顆時，就給一顆）："
            + "、".join(single_source)
        )
    for option in commitments.options:
        lines.append(
            f"- {option.commitment_id} [{option.tier}] span={option.span_id}; "
            f"role={option.picture_role}; minimum={option.min_supported_seconds:g}s; "
            f"content_policy={option.content_policy}; "
            + (
                f"content_action={option.content_action_id} "
                f"({option.content_action_start_seconds:g}–"
                f"{option.content_action_complete_seconds:g}s source clock; "
                f"complete duration="
                f"{option.content_action_complete_seconds - option.content_action_start_seconds:g}s); "
                if option.content_action_id != "none"
                and option.content_action_start_seconds is not None
                and option.content_action_complete_seconds is not None
                else "content_action=none; "
            )
            + f"presentation={option.presentation_intent}; motion="
            f"{option.motion_preference}（Direction 的偏好，不是限制；"
            "Selection 在本機量測的可行清單內選 camera_intent）; "
            f"Direction recommends treatment={option.direction_treatment}, "
            f"move={option.direction_suggested_move}, "
            f"route={option.direction_camera_route}, "
            f"reason={option.direction_motion_reason}, "
            f"fallback={option.direction_fallback_treatment}; "
            f"local camera menu={','.join(option.feasible_treatments)}; "
            f"local preferred={option.preferred_treatment}; "
            "camera timing=Selection chooses readable dwell, then local card "
            "geometry prices actual travel at the chosen energy before the "
            "answer is saved; unknown look geometry is marked conservative; "
            f"feasibility={option.feasibility_reason}; "
            f"target={option.target_id}; "
            f"visuals={','.join(option.required_visuals) or 'legacy-unspecified'}; "
            f"evidence={','.join(option.required_evidence) or 'none'}; "
            f"relationship={option.visual_relationship}; "
            f"purpose={option.purpose}"
        )
    return "\n".join(lines)


def _allowed_action_treatments(option: Any) -> set[str] | None:
    """Treatments implied by Direction's content/action pair.

    Legacy fixtures do not carry the new fields and keep their historical
    validator behaviour.  Production v3 commitments always do.
    """

    policy = getattr(option, "content_policy", None)
    if policy is None:
        return None
    action_id = str(getattr(option, "content_action_id", "none") or "none")
    if policy == "complete_action":
        return {"complete_here"}
    if policy == "result_hold":
        return {"after_completion"} if action_id != "none" else {"none"}
    if policy in {"representative_excerpt", "continuous_process"}:
        return (
            {"intentional_cut", "complete_here"}
            if action_id != "none" else {"none"}
        )
    return {"none"}


def _visual_relationship_fault(
    *, label: str, looks: Sequence[dict[str, Any]], option: CandidateOption,
) -> str | None:
    """Prove Selection did not drop a required visual participant."""

    required_visuals = tuple(
        getattr(option, "required_visuals", ()) or ()
    )
    required = set(required_visuals)
    if not required or not looks:
        return None
    per_look: list[set[str]] = []
    for look in looks:
        visible = {str(one) for one in (look.get("includes") or ())}
        at = str(look.get("at") or "")
        visible.update(one for one in required if one == at or one in at)
        per_look.append(visible)
    relationship = str(getattr(option, "visual_relationship", "single"))
    if relationship == "simultaneous":
        if not any(required <= visible for visible in per_look):
            return (
                f"{label} must keep {', '.join(required_visuals)} "
                "visible together in one landing"
            )
    else:
        covered = set().union(*per_look) if per_look else set()
        missing = required - covered
        if missing:
            return (
                f"{label} drops required visual participants: "
                f"{', '.join(sorted(missing))}"
            )
    return None


def _outcome_visual_fault(
    *, label: str, looks: Sequence[dict[str, Any]], option: CandidateOption,
) -> str | None:
    """Prove a named action result survives as the final stable landing."""

    if getattr(option, "content_policy", None) not in {
        "complete_action", "result_hold",
    }:
        return None
    evidence = str(getattr(option, "outcome_evidence", None) or "").strip()
    if not evidence:
        return None
    stable = [
        look for look in looks
        if str(look.get("presentation_intent") or "") != "transition_pass"
    ]
    if not stable:
        return f"{label} drops required action outcome: {evidence}"
    endpoint = stable[-1]
    if str(endpoint.get("geometry_query") or "") != evidence:
        return (
            f"{label} does not bind its final landing to action outcome: "
            f"{evidence}"
        )
    after = endpoint.get("geometry_after_source_seconds")
    if (
        option.content_action_complete_seconds is not None
        and (after is None or float(after) + 1e-6 < float(
            option.content_action_complete_seconds
        ))
    ):
        return f"{label} may measure action outcome before it exists: {evidence}"
    return None


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
        action_treatment = str(shot.get("action_treatment") or "none")
        content_policy = getattr(option, "content_policy", None)
        allowed_action_treatments = _allowed_action_treatments(option)
        if (
            allowed_action_treatments is not None
            and action_treatment not in allowed_action_treatments
        ):
            faults.append(
                f"shot {index} uses action treatment {action_treatment!r}, but "
                f"content policy {content_policy!r} allows only "
                f"{','.join(sorted(allowed_action_treatments))}"
            )
        expected_action = str(
            getattr(option, "content_action_id", "none") or "none"
        )
        selected_action = str(shot.get("action_id") or "none")
        if content_policy is not None and selected_action != expected_action:
            faults.append(
                f"shot {index} selects action {selected_action!r}, but "
                f"{option.span_id} content policy {content_policy!r} is bound "
                f"to {expected_action!r}"
            )
        # Direction's preference is editorial advice, but the locally derived
        # treatment menu is a physical capability contract.  Keeping those
        # concepts separate lets Selection choose a push for a static lineup
        # without allowing a move for which the crop has no room.
        camera_intent = str(shot.get("camera_intent") or "hold")
        if camera_intent not in option.feasible_treatments:
            faults.append(
                f"shot {index} camera treatment {camera_intent!r} is not "
                f"locally feasible for {option.span_id}; allowed="
                f"{','.join(option.feasible_treatments)}"
            )
        # Duration is validated once, with the selected looks and the source
        # card's measured positions, by planner.camera_duration_disagreements.
        # A second simplified table here was the reason Selection accepted a
        # move which the canonical executor rejected before Rhythm.
        looks = list(shot.get("looks") or [])
        matching_looks = [
            look for look in looks
            if option.target_id == "none"
            or str(look.get("entity_id") or "none") == option.target_id
        ]
        expected_presentation = option.presentation_intent
        if looks and expected_presentation not in {
            str(look.get("presentation_intent") or "")
            for look in matching_looks
        }:
            faults.append(
                f"shot {index} does not carry presentation intent "
                f"{expected_presentation} on target {option.target_id} "
                f"for {commitment_id}"
            )
        if looks and option.target_id != "none" and not matching_looks:
            faults.append(
                f"shot {index} does not bind target {option.target_id} for "
                f"{commitment_id}"
            )
        visual_fault = _visual_relationship_fault(
            label=f"shot {index}", looks=looks, option=option,
        )
        if visual_fault:
            faults.append(visual_fault)
        outcome_fault = _outcome_visual_fault(
            label=f"shot {index}", looks=looks, option=option,
        )
        if outcome_fault:
            faults.append(outcome_fault)
    # A story point may take more than one shot -- that is coverage, the way a
    # wide, a push and a detail together prove one idea. So a commitment used
    # by several shots is legal now; what is not is showing the same frames
    # twice, and that is caught for the whole film by the overlapping-window
    # audit rather than by a blanket count here. A required point must still be
    # answered, so it must appear at least once.
    for commitment_id in commitments.required_ids:
        if seen.get(commitment_id, 0) < 1:
            faults.append(
                f"required commitment {commitment_id} is never covered"
            )
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
            continue
        option = next((
            one for one in commitments.options
            if one.commitment_id == actual_id and one.span_id == span_id
        ), None)
        if option is None:
            continue
        action_treatment = str(replacement.get("action_treatment") or "none")
        content_policy = getattr(option, "content_policy", None)
        allowed_action_treatments = _allowed_action_treatments(option)
        if (
            allowed_action_treatments is not None
            and action_treatment not in allowed_action_treatments
        ):
            faults.append(
                f"{clip_id} uses action treatment {action_treatment!r}, but "
                f"content policy {content_policy!r} allows only "
                f"{','.join(sorted(allowed_action_treatments))}"
            )
        expected_action = str(
            getattr(option, "content_action_id", "none") or "none"
        )
        selected_action = str(replacement.get("action_id") or "none")
        if content_policy is not None and selected_action != expected_action:
            faults.append(
                f"{clip_id} selects action {selected_action!r}, but "
                f"{option.span_id} content policy {content_policy!r} is bound "
                f"to {expected_action!r}"
            )
        camera_intent = str(replacement.get("camera_intent") or "hold")
        if camera_intent not in option.feasible_treatments:
            faults.append(
                f"{clip_id} camera treatment {camera_intent!r} is not locally "
                f"feasible for {span_id}; allowed="
                f"{','.join(option.feasible_treatments)}"
            )
        # The replacement is subjected to the same canonical execution audit
        # before Rhythm/render; do not maintain a competing floor table here.
        looks = list(replacement.get("looks") or [])
        matching_looks = [
            look for look in looks
            if option.target_id == "none"
            or str(look.get("entity_id") or "none") == option.target_id
        ]
        if looks and option.target_id != "none" and not matching_looks:
            faults.append(
                f"{clip_id} does not bind target {option.target_id} for "
                f"{expected_id}"
            )
        elif looks and option.presentation_intent not in {
            str(look.get("presentation_intent") or "")
            for look in matching_looks
        }:
            faults.append(
                f"{clip_id} does not carry presentation intent "
                f"{option.presentation_intent} on target {option.target_id}"
            )
        visual_fault = _visual_relationship_fault(
            label=clip_id, looks=looks, option=option,
        )
        if visual_fault:
            faults.append(visual_fault)
        outcome_fault = _outcome_visual_fault(
            label=clip_id, looks=looks, option=option,
        )
        if outcome_fault:
            faults.append(outcome_fault)
    missing = set(expected) - seen
    if missing:
        faults.append("missing replacements for " + ", ".join(sorted(missing)))
    return faults


def bind_selection_content_contracts(
    shots: Sequence[dict[str, Any]], commitments: CandidateCommitments,
    material: Sequence[Any] = (),
) -> None:
    """Project immutable candidate policy onto normalized selected shots.

    These keys are local facts, not provider-authored schema fields. Keeping
    them on the shot lets EDL construction survive cache/resume without
    reopening Direction prose or guessing from ``why``.
    """

    options = {
        (option.commitment_id, option.span_id): option
        for option in commitments.options
    }
    visual_labels = {
        str(getattr(item, "source_id", "")): {
            f"v{at:02d}": str(entry[0])
            for at, entry in enumerate(
                getattr(item, "subject_geometry", ()) or (), start=1,
            )
        }
        for item in material
    }
    for shot in shots:
        option = options.get((
            str(shot.get("commitment_id") or ""),
            str(shot.get("span_id") or ""),
        ))
        if option is None:
            continue
        shot["content_policy"] = option.content_policy
        shot["content_min_seconds"] = option.min_supported_seconds
        shot["content_purpose"] = option.purpose
        shot["content_required_visuals"] = list(option.required_visuals)
        shot["content_required_evidence"] = list(option.required_evidence)
        shot["content_visual_relationship"] = option.visual_relationship
        shot["content_action_start_seconds"] = (
            option.content_action_start_seconds
        )
        shot["content_action_complete_seconds"] = (
            option.content_action_complete_seconds
        )
        shot["direction_motion_advice"] = {
            "treatment": option.direction_treatment,
            "move": option.direction_suggested_move,
            "route": option.direction_camera_route,
            "reason": option.direction_motion_reason,
            "fallback": option.direction_fallback_treatment,
            "locally_feasible": list(option.feasible_treatments),
        }
        if option.identity_status == "needs_review":
            shot["identity_status"] = "needs_review"
            shot["identity_issue"] = option.identity_issue
        # Direction owns how this commitment is meant to be read. Selection
        # chooses the source window and executable treatment; asking it to
        # copy the same presentation enum into a nested look created paid
        # repair loops when it otherwise selected the right commitment. Keep
        # the provider's subject/box wording, but project the immutable
        # commitment intent onto the matching look locally.
        looks = list(shot.get("looks") or [])
        matching = [
            look for look in looks
            if option.target_id == "none"
            or str(look.get("entity_id") or "none") == option.target_id
        ]
        if matching:
            primary = matching[0]
            if option.visual_relationship == "simultaneous":
                primary["includes"] = list(option.required_visuals)
            elif len(option.required_visuals) == 1:
                # ``required_visuals`` uses stable local IDs while Gemini's
                # ``look.at`` deliberately remains readable card language.
                # Project the one unambiguous ID locally instead of asking
                # the provider to copy an implementation identifier.
                primary["includes"] = list(dict.fromkeys([
                    *list(primary.get("includes") or []),
                    option.required_visuals[0],
                ]))
            elif (
                option.visual_relationship == "ordered"
                and len(matching) == len(option.required_visuals)
            ):
                # Direction already fixed the participant order and
                # Selection supplied the same number of semantic landings.
                # Pairing by order is therefore deterministic; without this
                # bridge the local validator falsely reports that every
                # human-readable look dropped v01/v02.
                for look, visual_id in zip(
                    matching, option.required_visuals, strict=True,
                ):
                    look["includes"] = list(dict.fromkeys([
                        *list(look.get("includes") or []), visual_id,
                    ]))
            # ``sequential_read`` is the content promise: every required
            # region must become readable in order.  The executor used to
            # erase that promise when Selection chose native motion by
            # rewriting it as a generic endpoint reveal.  A tiny authored
            # drift could then pass while most of a wide sign or UI remained
            # outside the vertical crop.  Keep the promise intact; the
            # selection audit below proves whether the selected source window
            # actually travels far enough, or asks for a designed reveal.
            effective_presentation = option.presentation_intent
            if effective_presentation in {
                "sequential_read", "partial_reveal", "transition_pass",
            }:
                primary.update(
                    presentation_intent=effective_presentation,
                    must_be_whole=False,
                )
            else:
                primary["presentation_intent"] = effective_presentation

            # Semantic evidence without a card bbox is not decorative prose.
            # Bind it to the final stable landing and action-completion clock.
            # Identity remains the carrier contract and is never replaced by
            # this smaller detail box.
            outcome = str(option.outcome_evidence or "").strip()
            if outcome and option.content_policy in {
                "complete_action", "result_hold",
            }:
                endpoint = next((
                    look for look in reversed(matching)
                    if str(look.get("presentation_intent") or "")
                    != "transition_pass"
                ), matching[-1])
                endpoint["geometry_query"] = outcome
                endpoint["geometry_after_source_seconds"] = (
                    option.content_action_complete_seconds
                )
                shot["outcome_visual_contract"] = {
                    "action_id": option.content_action_id,
                    "evidence": outcome,
                    "carrier_visual_ids": list(option.required_visuals),
                    "carrier_entity_id": (
                        option.target_id if option.target_id != "none" else None
                    ),
                    "visible_after_source_seconds": (
                        option.content_action_complete_seconds
                    ),
                    "endpoint": "final_stable_look",
                }

        # ``includes`` is the stable local visual contract; ``at`` is model
        # display copy.  They used to part company here: a shot promised the
        # phone through v01/v03 while ``at`` said "the green vase", and the
        # runtime detector faithfully centred the vase.  Once Direction has
        # bound stable ids, resolve the actual geometry query from those ids
        # instead of letting later prose override the contract.  Keep the
        # provider wording only when the card has no local label for every id.
        labels = visual_labels.get(str(shot.get("source_id") or ""), {})
        for look in looks:
            included = tuple(dict.fromkeys(
                str(one) for one in (look.get("includes") or ())
            ))
            canonical = [labels.get(one) for one in included]
            if included and all(canonical) and not look.get("geometry_query"):
                look["at"] = " + ".join(dict.fromkeys(
                    str(one) for one in canonical if one
                ))
        action_id = str(option.content_action_id or "none")
        shot["action_id"] = action_id
        if option.content_policy == "complete_action":
            shot["action_treatment"] = "complete_here"
        elif option.content_policy == "result_hold":
            shot["action_treatment"] = (
                "after_completion" if action_id != "none" else "none"
            )
        elif option.content_policy == "static_display" or action_id == "none":
            shot["action_treatment"] = "none"


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
