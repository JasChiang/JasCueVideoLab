"""Bounded, hash-bound authority for unattended feature delivery.

Gemini may propose or observe.  Only these application-owned contracts can
authorize an executable decision or a final delivery state.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Mapping

from pydantic import Field, model_validator

from .models import FrozenStrictModel
from .storage import utc_now


AUTONOMOUS_POLICY_VERSION = "autonomous-edit-policy-v1"
DECISION_AUTHORITY_VERSION = "decision-authority-v2"
DEGRADATION_MANIFEST_VERSION = "autonomous-degradation-manifest-v1"

Aspect = Literal["16:9", "9:16"]
MediaResolution = Literal["low", "medium", "high"]
ThinkingLevel = Literal["minimal", "low", "medium", "high"]


class AutonomousExecutionProfile(StrEnum):
    STRICT = "autonomous_strict"
    BEST_EFFORT = "autonomous_best_effort"


class AutonomousOutcome(StrEnum):
    DELIVERY_ELIGIBLE = "delivery_eligible"
    BEST_EFFORT_COMPLETE = "best_effort_complete"
    BLOCKED = "blocked"


class DurationPolicy(FrozenStrictModel):
    target_ms: int = Field(ge=30_000, le=120_000)
    min_ms: int = Field(ge=30_000, le=120_000)
    max_ms: int = Field(ge=30_000, le=120_000)

    @model_validator(mode="after")
    def validate_range(self) -> "DurationPolicy":
        if not self.min_ms <= self.target_ms <= self.max_ms:
            raise ValueError("duration must satisfy min_ms <= target_ms <= max_ms")
        return self


class BudgetPolicy(FrozenStrictModel):
    max_gemini_cost_usd: float = Field(gt=0.0, le=100.0)
    # Cold Base Clip Card ingest is deliberately separate from the warm-edit
    # allowance. ``None`` is fail-closed: a warm run may reuse current cards,
    # but it may not silently spend money refreshing stale or missing cards.
    max_cold_ingest_cost_usd: float | None = Field(
        default=None,
        gt=0.0,
        le=100.0,
    )
    max_paid_interactions: int = Field(ge=1, le=100)
    # A scoped replan is an exception path after all bounded local candidate
    # facts are known. It is disabled by default; a policy must explicitly
    # reserve the single allowed replan before production may dispatch it.
    max_semantic_replans: Literal[0, 1] = 0
    max_final_qa_passes: int = Field(default=2, ge=1, le=2)
    reserved_recovery_fraction: float = Field(default=0.20, ge=0.20, le=0.50)


class MediaResolutionPolicy(FrozenStrictModel):
    # These stages do not yet have variable-resolution execution paths. Keep
    # their signed values fixed instead of accepting ignored configuration.
    base_clip_card: Literal["low"] = "low"
    candidate_reel_plan: MediaResolution = "low"
    bounded_event_video: Literal["low"] = "low"
    exact_event_image: Literal["high"] = "high"
    exact_frame_grounding_image: Literal["high"] = "high"
    final_video_qa: MediaResolution = "low"
    # V1 resolves text/UI uncertainty with high-resolution exact stills. A
    # whole-video text-heavy escalation route is deliberately not executable.
    text_heavy_video: Literal["high"] = "high"


class GeminiOperationLimit(FrozenStrictModel):
    # The upper bound tracks what the model will actually accept, so a policy
    # can budget for a verbose answer instead of being capped below it. A
    # twelve-chapter Top-K plan has been measured at 24.5k output tokens, and
    # a run that exceeds its ceiling loses the whole paid response. Per
    # operation defaults stay where they are; raising them would inflate every
    # budget reservation whether or not the tokens are used.
    max_output_tokens: int = Field(ge=256, le=65_536)
    thinking_level: ThinkingLevel


class GeminiOperationLimits(FrozenStrictModel):
    base_clip_card: GeminiOperationLimit = GeminiOperationLimit(
        max_output_tokens=4_096,
        thinking_level="low",
    )
    candidate_reel_plan: GeminiOperationLimit = GeminiOperationLimit(
        max_output_tokens=12_288,
        thinking_level="low",
    )
    exact_event_group: GeminiOperationLimit = GeminiOperationLimit(
        max_output_tokens=2_048,
        thinking_level="low",
    )
    multi_target_grounding: GeminiOperationLimit = GeminiOperationLimit(
        max_output_tokens=2_048,
        thinking_level="low",
    )
    final_qa: GeminiOperationLimit = GeminiOperationLimit(
        max_output_tokens=8_192,
        thinking_level="low",
    )
    scoped_replan: GeminiOperationLimit = GeminiOperationLimit(
        max_output_tokens=4_096,
        thinking_level="low",
    )
    semantic_negotiation: GeminiOperationLimit = GeminiOperationLimit(
        max_output_tokens=2_048,
        thinking_level="low",
    )
    text_only_schema_repair: GeminiOperationLimit = GeminiOperationLimit(
        max_output_tokens=4_096,
        thinking_level="minimal",
    )


class PresentationPolicy(FrozenStrictModel):
    allow_two_panel_layout: bool = True
    allowed_panel_modes: tuple[
        Literal["top_bottom", "side_by_side", "context_detail"], ...
    ] = ("top_bottom", "side_by_side", "context_detail")
    max_panels: Literal[2] = 2
    max_panel_runtime_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    allow_solid_matte_fit: bool = True
    allow_blurred_background: Literal[False] = False
    allow_intentional_freeze: bool = True
    max_intentional_freeze_ms: int = Field(default=1_500, ge=1, le=5_000)
    allow_synthetic_motion_without_motivation: Literal[False] = False

    @model_validator(mode="after")
    def validate_panel_policy(self) -> "PresentationPolicy":
        if self.allow_two_panel_layout and not self.allowed_panel_modes:
            raise ValueError("enabled two-panel layout requires an allowed mode")
        if len(set(self.allowed_panel_modes)) != len(self.allowed_panel_modes):
            raise ValueError("allowed panel modes must be unique")
        return self


class EditorialPolicy(FrozenStrictModel):
    allow_optional_beat_omission: bool = True
    # V1 has no executable preferred-beat substitution compiler. Do not grant
    # authority that the production path cannot use and audit.
    allow_preferred_beat_substitution: Literal[False] = False
    allow_hard_evidence_omission: Literal[False] = False
    allow_source_reuse: tuple[
        Literal[
            "distinct_interval",
            "alternate_presentation",
            "editorial_reprise",
        ],
        ...,
    ] = ("distinct_interval", "alternate_presentation", "editorial_reprise")
    # This is an editorial limit, not an implementation convenience.  It is
    # hash-bound so the candidate-route solver, runtime reuse preflight and
    # final audit all prove the same permission boundary.
    max_editorial_reprise_overlap_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )


class SyncPolicy(FrozenStrictModel):
    hard_tolerance_frames: int = Field(default=2, ge=0, le=12)
    preferred_tolerance_frames: int = Field(default=4, ge=0, le=24)

    @model_validator(mode="after")
    def validate_tolerances(self) -> "SyncPolicy":
        if self.preferred_tolerance_frames < self.hard_tolerance_frames:
            raise ValueError("preferred tolerance cannot be tighter than hard")
        return self


class RecoveryPolicy(FrozenStrictModel):
    max_candidate_attempts_per_beat: int = Field(default=3, ge=1, le=5)
    # Dispatch journals make a failed/ambiguous paid request resumable, but V1
    # deliberately has no automatic request retry. Retrying must be a new,
    # explicit run after inspecting preserved artifacts.
    max_request_failure_retries: Literal[0] = 0
    # Semantic final QA remains a hard V1 delivery requirement.
    allow_deterministic_delivery_when_semantic_qa_unavailable: Literal[
        False
    ] = False


class SemanticNegotiationPolicy(FrozenStrictModel):
    """Bound Gemini tool use without turning planning into an agent loop."""

    enabled: bool = True
    max_global_negotiations: Literal[0, 1] = 1
    max_repair_negotiations: Literal[0, 1] = 1
    max_tool_result_rounds: int = Field(default=2, ge=1, le=2)
    max_parallel_read_only_calls: int = Field(default=3, ge=1, le=3)
    max_preview_options: int = Field(default=3, ge=2, le=3)
    automatic_function_calling: Literal[False] = False
    allowed_tools: tuple[
        Literal[
            "inspect_edit_evidence",
            "request_temporal_evidence",
            "get_music_structure",
            "enumerate_presentation_options",
            "preview_presentation_options",
            "propose_edit_decision",
        ],
        ...,
    ] = (
        "inspect_edit_evidence",
        "request_temporal_evidence",
        "get_music_structure",
        "enumerate_presentation_options",
        "preview_presentation_options",
        "propose_edit_decision",
    )

    @model_validator(mode="after")
    def validate_tools(self) -> "SemanticNegotiationPolicy":
        if len(set(self.allowed_tools)) != len(self.allowed_tools):
            raise ValueError("semantic negotiation tools must be unique")
        if self.enabled and "propose_edit_decision" not in self.allowed_tools:
            raise ValueError(
                "enabled semantic negotiation requires propose_edit_decision"
            )
        return self


class WorkerPolicy(FrozenStrictModel):
    """Maximum concurrency ceilings, not requested pool sizes.

    Selected-window production is currently sequential, so its observed
    concurrency of one is within every accepted ceiling.
    """

    ffmpeg_workers: int = Field(default=2, ge=1, le=8)
    proxy_workers: int = Field(default=2, ge=1, le=8)
    sam_workers: int = Field(default=1, ge=1, le=2)
    cold_clip_card_workers: int = Field(default=2, ge=1, le=4)
    allow_selected_edit_speculative_paid_calls: Literal[False] = False


class AutonomousEditPolicy(FrozenStrictModel):
    contract_version: Literal["autonomous-edit-policy-v1"] = (
        AUTONOMOUS_POLICY_VERSION
    )
    model_id: Literal["gemini-3.7-flash"] = "gemini-3.7-flash"
    execution_profile: AutonomousExecutionProfile
    content_mode: Literal["music_led_feature", "visual_demo"]
    requested_aspects: tuple[Aspect, ...] = Field(min_length=1, max_length=2)
    duration: DurationPolicy
    budget: BudgetPolicy
    media_resolution: MediaResolutionPolicy = MediaResolutionPolicy()
    gemini_limits: GeminiOperationLimits = GeminiOperationLimits()
    presentation: PresentationPolicy = PresentationPolicy()
    editorial: EditorialPolicy = EditorialPolicy()
    sync: SyncPolicy = SyncPolicy()
    recovery: RecoveryPolicy = RecoveryPolicy()
    semantic_negotiation: SemanticNegotiationPolicy = (
        SemanticNegotiationPolicy()
    )
    workers: WorkerPolicy = WorkerPolicy()

    @model_validator(mode="after")
    def validate_policy(self) -> "AutonomousEditPolicy":
        if len(set(self.requested_aspects)) != len(self.requested_aspects):
            raise ValueError("requested aspects must be unique")
        if self.gemini_limits.scoped_replan != (
            GeminiOperationLimits().scoped_replan
        ):
            raise ValueError(
                "scoped semantic replan call limits are reserved in V1 and "
                "cannot be customized"
            )
        return self

    def definition_sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode="json"))

    @property
    def policy_reference(self) -> str:
        return f"sha256:{self.definition_sha256()}"


class DecisionAuthorityType(StrEnum):
    AUTO_POLICY = "auto_policy"


class DecisionAuthorityV2(FrozenStrictModel):
    contract_version: Literal["decision-authority-v2"] = (
        DECISION_AUTHORITY_VERSION
    )
    authority_type: Literal["auto_policy"] = "auto_policy"
    decision_scope: Literal[
        "music_map",
        "cue_plan",
        "trim_intent",
        "reframe",
        "scoped_semantic_replan",
        "execution_binding_repair",
        "feature_cut",
        "final_delivery",
    ]
    policy_reference: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_artifact_hashes: tuple[str, ...] = Field(min_length=1)
    deterministic_gate_results: Mapping[str, Literal["passed"]]
    gemini_interaction_ids: tuple[str, ...] = ()
    decision_codes: tuple[str, ...] = Field(min_length=1)
    generated_at: str

    @model_validator(mode="after")
    def validate_authority(self) -> "DecisionAuthorityV2":
        if len(set(self.input_artifact_hashes)) != len(
            self.input_artifact_hashes
        ):
            raise ValueError("authority input artifact hashes must be unique")
        if len(set(self.decision_codes)) != len(self.decision_codes):
            raise ValueError("authority decision codes must be unique")
        for digest in self.input_artifact_hashes:
            if not digest.startswith("sha256:") or len(digest) != 71:
                raise ValueError(
                    "authority input hashes must use sha256:<64 lowercase hex>"
                )
            int(digest[7:], 16)
        if not self.deterministic_gate_results:
            raise ValueError("automatic authority requires deterministic gates")
        return self


class DegradationRecord(FrozenStrictModel):
    beat_id: str
    action: Literal[
        "optional_beat_omitted",
        "preferred_beat_substituted",
        "alternate_candidate_used",
        "two_panel_layout_used",
        "solid_matte_fit_used",
        "target_duration_shortened",
        "contextual_visual_substitution",
    ]
    reason_code: str
    copy_suppression_codes: tuple[str, ...] = ()
    original_artifact_hashes: tuple[str, ...] = ()
    replacement_artifact_hashes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_copy_suppression(self) -> "DegradationRecord":
        if len(set(self.copy_suppression_codes)) != len(
            self.copy_suppression_codes
        ):
            raise ValueError("copy-suppression codes must be unique")
        if (
            self.action == "contextual_visual_substitution"
            and "specific_claim_copy_suppressed"
            not in self.copy_suppression_codes
        ):
            raise ValueError(
                "contextual substitution must suppress specific claim copy"
            )
        return self


class AutonomousDegradationManifest(FrozenStrictModel):
    contract_version: Literal["autonomous-degradation-manifest-v1"] = (
        DEGRADATION_MANIFEST_VERSION
    )
    policy_reference: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    records: tuple[DegradationRecord, ...] = ()
    hard_evidence_omitted: Literal[False] = False
    aspect: Literal["16:9", "9:16"] | None = None
    generated_at: str


def omissions_are_policy_authorized(
    policy: AutonomousEditPolicy,
    records: tuple[DegradationRecord, ...],
) -> bool:
    """Validate editorial omissions without treating presentations as omissions."""

    for record in records:
        if (
            record.action == "optional_beat_omitted"
            and not policy.editorial.allow_optional_beat_omission
        ):
            return False
        if (
            record.action == "preferred_beat_substituted"
            and not policy.editorial.allow_preferred_beat_substitution
        ):
            return False
    return True


def sync_tolerance_for_priority(
    policy: AutonomousEditPolicy,
    priority: Literal["hard", "preferred", "optional"],
) -> int:
    """Return the policy ceiling for an editorial visual-event contract."""

    if priority == "hard":
        return policy.sync.hard_tolerance_frames
    return policy.sync.preferred_tolerance_frames


def authorize_decision(
    policy: AutonomousEditPolicy,
    *,
    decision_scope: Literal[
        "music_map",
        "cue_plan",
        "trim_intent",
        "reframe",
        "scoped_semantic_replan",
        "execution_binding_repair",
        "feature_cut",
        "final_delivery",
    ],
    input_artifact_hashes: tuple[str, ...],
    deterministic_gate_results: Mapping[str, Literal["passed"]],
    decision_codes: tuple[str, ...],
    gemini_interaction_ids: tuple[str, ...] = (),
) -> DecisionAuthorityV2:
    """Create application-owned authority from immutable evidence and gates."""

    return DecisionAuthorityV2(
        decision_scope=decision_scope,
        policy_reference=policy.policy_reference,
        input_artifact_hashes=input_artifact_hashes,
        deterministic_gate_results=dict(deterministic_gate_results),
        gemini_interaction_ids=gemini_interaction_ids,
        decision_codes=decision_codes,
        generated_at=utc_now(),
    )


def validate_authority_binding(
    authority: DecisionAuthorityV2,
    policy: AutonomousEditPolicy,
) -> None:
    if authority.policy_reference != policy.policy_reference:
        raise ValueError("decision authority is not bound to this policy")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
