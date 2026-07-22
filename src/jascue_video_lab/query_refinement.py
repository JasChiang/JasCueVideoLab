from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from .media import sha256_file

from .models import (
    Confidence,
    DenseFrame,
    DenseFrameCatalog,
    EvidenceQueryLockV2,
    MatchStatus,
    ModelProvenance,
    PredicateRequiredAt,
    PredicateStatus,
    StrictModel,
)
from .storage import read_json, write_json
from .schema import gemini_response_schema


QUERY_TEMPORAL_PROTOCOL_VERSION = "query-temporal-refinement-v1"
QUERY_TEMPORAL_TASK_INSTRUCTIONS = """本任務只做 temporal evidence refinement，不做 bbox、crop、追蹤或重新選擇 target。
只能逐字引用下方 DF frame ID；不得輸出、推算或改寫來源時間碼、毫秒、PTS 或 frame number。
圖片中的每個 DF 是離散抽樣證據；未提供的影格一律不能主張已驗證。
只有 identity 與 predicate 都直接受到影像支持，才能回傳 matched+satisfied。
若 identity 不明、相似實例無法區分、predicate 缺少必要狀態或證據不足，使用 ambiguous、target_mismatch、not_visible、insufficient_evidence、not_satisfied 或 indeterminate，且不得選 frame。

## required_at 語意
candidate：只選一張支持候選資格的 candidate_frame_id，coverage_claim=single_frame_only。
seed：只選一張 identity 清楚且 predicate 成立的 seed_frame_id，coverage_claim=single_frame_only。
transition：必須分別選 precondition/apex/postcondition 三張不同 DF，coverage_claim=transition_samples_only。三張 evidence 的 transition_phase 必須依序為 precondition、apex、postcondition，且 predicate_observed=true 只代表該張直接支持相應 phase，不代表未提供影格。
interval：positive 結果必須回傳至少兩張、時間連續且不跳號的一段合法 DF ID；選定範圍內每一張 catalog sample 都必須支持 locked identity 與 predicate。coverage_claim=sampled_frames_only 仍只代表離散抽樣點，絕不代表中間未抽樣影格。
每一筆 evidence 的 identity_status_by_target 必須逐一包含所有 predicate participant target ID，不得用單一總體布林值代替多實例身份證據。positive frame 的所有 participant 都必須是 observed。
非 matched+satisfied 結果必須 coverage_claim=no_positive_claim，所有 selection frame 欄位為 null/空陣列。
"""
QUERY_TEMPORAL_GENERATION_CONFIG = {
    "thinking_level": "low",
    "max_output_tokens": 8192,
}

TARGET_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"


def _canonical_sha256(value: Any) -> str:
    """Hash JSON without depending on host paths or JSON formatting."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dense_catalog_evidence_sha256(catalog: DenseFrameCatalog) -> str:
    """Content fingerprint for the evidence represented by a dense catalog.

    Local image paths and generation time are deliberately excluded.  Exact
    source PTS and content hashes are included, so moving an artifact does not
    invalidate evidence while replacing any frame does.
    """

    return _canonical_sha256(
        {
            "source_asset_id": catalog.source_asset_id,
            "event_id": catalog.event_id,
            "sampling_fps": catalog.sampling_fps,
            "source_start_ms": catalog.source_start_ms,
            "source_end_ms": catalog.source_end_ms,
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "event_id": frame.event_id,
                    "requested_time_ms": frame.requested_time_ms,
                    "frame_time_ms": frame.frame_time_ms,
                    "frame_pts": frame.frame_pts,
                    "frame_hash": frame.frame_hash,
                    "width": frame.width,
                    "height": frame.height,
                    "transport_image_hash": frame.transport_image_hash,
                }
                for frame in catalog.frames
            ],
            "contact_sheet_hashes": catalog.contact_sheet_hashes,
        }
    )


class QueryTemporalFingerprint(StrictModel):
    """Split fingerprints for exact, auditable temporal-refinement reuse."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal["query-temporal-refinement-v1"] = (
        QUERY_TEMPORAL_PROTOCOL_VERSION
    )
    source_asset_id: str
    event_id: str
    query_id: str
    grounding_target_id: str = Field(pattern=TARGET_ID_PATTERN)
    required_at: PredicateRequiredAt
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_instruction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_instruction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def build_query_temporal_fingerprint(
    *,
    query_lock: EvidenceQueryLockV2,
    grounding_target_id: str,
    catalog: DenseFrameCatalog,
    model_id: str,
    prompt_template: str,
    system_instruction: str,
    response_schema: dict[str, Any],
) -> QueryTemporalFingerprint:
    """Build an exact cache key without coupling temporal work to framing.

    The framing contract, approval metadata, and local paths do not affect this
    operation.  Identity, predicate, sampled evidence, model, instructions, and
    the actual API response schema do.
    """

    predicate = query_lock.predicate
    if predicate is None:
        raise ValueError("QueryLock temporal refinement requires a predicate")
    components = query_lock.component_hashes()
    known_target_ids = {target.target_id for target in query_lock.identity.targets}
    if grounding_target_id not in known_target_ids:
        raise ValueError("temporal grounding target is absent from QueryLock identity")
    if grounding_target_id not in predicate.participant_target_ids:
        raise ValueError("temporal grounding target must be a predicate participant")
    values: dict[str, Any] = {
        "protocol_version": QUERY_TEMPORAL_PROTOCOL_VERSION,
        "source_asset_id": catalog.source_asset_id,
        "event_id": catalog.event_id,
        "query_id": query_lock.query_id,
        "grounding_target_id": grounding_target_id,
        "required_at": predicate.required_at.value,
        "identity_sha256": components["identity_sha256"],
        "predicate_sha256": components["predicate_sha256"],
        "catalog_sha256": dense_catalog_evidence_sha256(catalog),
        "prompt_sha256": hashlib.sha256(prompt_template.encode("utf-8")).hexdigest(),
        "system_instruction_sha256": hashlib.sha256(
            system_instruction.encode("utf-8")
        ).hexdigest(),
        "response_schema_sha256": _canonical_sha256(response_schema),
        "task_instruction_sha256": hashlib.sha256(
            QUERY_TEMPORAL_TASK_INSTRUCTIONS.encode("utf-8")
        ).hexdigest(),
        "generation_config_sha256": _canonical_sha256(
            QUERY_TEMPORAL_GENERATION_CONFIG
        ),
        "model_id": model_id,
    }
    return QueryTemporalFingerprint(
        **values,
        request_sha256=_canonical_sha256(values),
    )


def query_temporal_contract_sha256(
    query_lock: EvidenceQueryLockV2,
    grounding_target_id: str,
) -> str:
    """Hash only lock fields that can change temporal evidence semantics."""

    components = query_lock.component_hashes()
    return _canonical_sha256(
        {
            "query_id": query_lock.query_id,
            "grounding_target_id": grounding_target_id,
            "identity_sha256": components["identity_sha256"],
            "predicate_sha256": components["predicate_sha256"],
        }
    )


def write_query_temporal_consumer_lineage(
    run_dir: Path,
    *,
    query_lock: EvidenceQueryLockV2,
    grounding_target_id: str,
    request_sha256: str,
) -> Path:
    """Record every full QueryLock revision consuming a shared temporal result."""

    definition_sha256 = query_lock.definition_sha256()
    path = run_dir / f"query_temporal.consumer-lock-{definition_sha256[:16]}.json"
    payload = {
        "contract_version": "query-temporal-consumer-lineage-v1",
        "query_id": query_lock.query_id,
        "grounding_target_id": grounding_target_id,
        "query_lock_definition_sha256": definition_sha256,
        "component_hashes": query_lock.component_hashes(),
        "approval": query_lock.approval.model_dump(mode="json"),
        "temporal_contract_sha256": query_temporal_contract_sha256(
            query_lock, grounding_target_id
        ),
        "request_sha256": request_sha256,
    }
    if path.exists():
        if read_json(path) != payload:
            raise ValueError("query temporal consumer lineage artifact was modified")
    else:
        write_json(path, payload)
    return path


class QueryTemporalFrameObservation(StrictModel):
    """A model observation tied only to an input DF identifier."""

    frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    identity_status_by_target: dict[
        str, Literal["observed", "not_visible", "ambiguous", "mismatch"]
    ]
    predicate_observed: bool | None = None
    transition_phase: Literal["precondition", "apex", "postcondition"] | None = None
    observation: str = Field(min_length=1)


CoverageClaim = Literal[
    "no_positive_claim",
    "single_frame_only",
    "transition_samples_only",
    "sampled_frames_only",
]


class QueryTemporalSelection(StrictModel):
    """Structured Gemini output containing frame IDs, never source time."""

    source_asset_id: str
    event_id: str
    query_id: str
    grounding_target_id: str = Field(pattern=TARGET_ID_PATTERN)
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_at: PredicateRequiredAt
    match_status: MatchStatus
    predicate_status: PredicateStatus
    coverage_claim: CoverageClaim
    candidate_frame_id: str | None = Field(
        default=None, pattern=r"^DF[0-9]{6}$"
    )
    seed_frame_id: str | None = Field(default=None, pattern=r"^DF[0-9]{6}$")
    precondition_frame_id: str | None = Field(
        default=None, pattern=r"^DF[0-9]{6}$"
    )
    apex_frame_id: str | None = Field(default=None, pattern=r"^DF[0-9]{6}$")
    postcondition_frame_id: str | None = Field(
        default=None, pattern=r"^DF[0-9]{6}$"
    )
    interval_sample_frame_ids: list[str] = Field(default_factory=list)
    evidence: list[QueryTemporalFrameObservation] = Field(default_factory=list)
    observable_evidence_summary: str = Field(min_length=1)
    uncertainties: list[str] = Field(default_factory=list)
    confidence: Confidence
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_stage_shape(self) -> "QueryTemporalSelection":
        frame_ids = [
            self.candidate_frame_id,
            self.seed_frame_id,
            self.precondition_frame_id,
            self.apex_frame_id,
            self.postcondition_frame_id,
        ]
        positive = (
            self.match_status == MatchStatus.MATCHED
            and self.predicate_status == PredicateStatus.SATISFIED
        )
        if self.predicate_status == PredicateStatus.NOT_APPLICABLE:
            raise ValueError("temporal refinement requires an applicable predicate")
        positive_ids: list[str] = []
        if not positive:
            if any(frame_id is not None for frame_id in frame_ids) or self.interval_sample_frame_ids:
                raise ValueError("non-positive temporal results cannot select frames")
            if self.coverage_claim != "no_positive_claim":
                raise ValueError("non-positive temporal results cannot make a coverage claim")
        elif self.required_at == PredicateRequiredAt.CANDIDATE:
            if self.candidate_frame_id is None or any(
                value is not None for value in frame_ids[1:]
            ) or self.interval_sample_frame_ids:
                raise ValueError("candidate predicate requires only candidate_frame_id")
            if self.coverage_claim != "single_frame_only":
                raise ValueError("candidate evidence is a single-frame claim")
            positive_ids = [self.candidate_frame_id]
        elif self.required_at == PredicateRequiredAt.SEED:
            if self.seed_frame_id is None or self.candidate_frame_id is not None or any(
                value is not None for value in frame_ids[2:]
            ) or self.interval_sample_frame_ids:
                raise ValueError("seed predicate requires only seed_frame_id")
            if self.coverage_claim != "single_frame_only":
                raise ValueError("seed evidence is a single-frame claim")
            positive_ids = [self.seed_frame_id]
        elif self.required_at == PredicateRequiredAt.TRANSITION:
            transition = [
                self.precondition_frame_id,
                self.apex_frame_id,
                self.postcondition_frame_id,
            ]
            if (
                self.candidate_frame_id is not None
                or self.seed_frame_id is not None
                or any(value is None for value in transition)
                or len(set(transition)) != 3
                or self.interval_sample_frame_ids
            ):
                raise ValueError("transition predicate requires distinct pre/apex/post IDs")
            if self.coverage_claim != "transition_samples_only":
                raise ValueError("transition evidence is limited to three sampled frames")
            positive_ids = [value for value in transition if value is not None]
        elif self.required_at == PredicateRequiredAt.INTERVAL:
            if any(value is not None for value in frame_ids):
                raise ValueError("interval predicate uses only interval_sample_frame_ids")
            if len(self.interval_sample_frame_ids) < 2:
                raise ValueError("interval predicate requires at least two sampled frames")
            if len(self.interval_sample_frame_ids) != len(
                set(self.interval_sample_frame_ids)
            ):
                raise ValueError("interval sample frame IDs must be unique")
            if self.coverage_claim != "sampled_frames_only":
                raise ValueError("interval evidence never proves unsampled frames")
            positive_ids = list(self.interval_sample_frame_ids)
        evidence_ids = [item.frame_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("temporal evidence frame IDs must be unique")
        for item in self.evidence:
            if not item.identity_status_by_target:
                raise ValueError("temporal evidence must report participant identity status")
            if any(
                not target_id
                or not re.fullmatch(TARGET_ID_PATTERN, target_id)
                for target_id in item.identity_status_by_target
            ):
                raise ValueError("temporal evidence contains an invalid target ID")
        missing_evidence = set(positive_ids) - set(evidence_ids)
        if missing_evidence:
            raise ValueError(
                "selected temporal frames require observations: "
                f"{sorted(missing_evidence)}"
            )
        evidence_by_id = {item.frame_id: item for item in self.evidence}
        if positive:
            for frame_id in positive_ids:
                observation = evidence_by_id[frame_id]
                if any(
                    status != "observed"
                    for status in observation.identity_status_by_target.values()
                ):
                    raise ValueError(
                        "positive temporal frames must observe every reported participant"
                    )
                if observation.predicate_observed is not True:
                    raise ValueError("positive temporal frames must observe the predicate")
            if self.required_at == PredicateRequiredAt.TRANSITION:
                expected_phases = {
                    self.precondition_frame_id: "precondition",
                    self.apex_frame_id: "apex",
                    self.postcondition_frame_id: "postcondition",
                }
                for frame_id, phase in expected_phases.items():
                    if evidence_by_id[str(frame_id)].transition_phase != phase:
                        raise ValueError(
                            "transition observations must identify precondition/apex/postcondition"
                        )
        if any(not value.strip() for value in self.uncertainties):
            raise ValueError("uncertainties must be non-empty")
        return self


class ResolvedDenseFrame(StrictModel):
    """A DF identifier resolved locally to exact source geometry and PTS."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_id: str = Field(pattern=r"^DF[0-9]{6}$")
    requested_time_ms: int = Field(ge=0)
    frame_time_ms: int = Field(ge=0)
    frame_pts: int
    frame_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ResolvedQueryTemporalObservation(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    frame: ResolvedDenseFrame
    identity_status_by_target: dict[
        str, Literal["observed", "not_visible", "ambiguous", "mismatch"]
    ]
    predicate_observed: bool | None = None
    transition_phase: Literal["precondition", "apex", "postcondition"] | None = None
    observation: str


class QueryTemporalDecision(StrictModel):
    """Locally verified decision with model IDs mapped to source PTS."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_asset_id: str
    event_id: str
    query_id: str
    grounding_target_id: str = Field(pattern=TARGET_ID_PATTERN)
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_at: PredicateRequiredAt
    match_status: MatchStatus
    predicate_status: PredicateStatus
    coverage_claim: CoverageClaim
    candidate_frame: ResolvedDenseFrame | None = None
    seed_frame: ResolvedDenseFrame | None = None
    precondition_frame: ResolvedDenseFrame | None = None
    apex_frame: ResolvedDenseFrame | None = None
    postcondition_frame: ResolvedDenseFrame | None = None
    interval_sample_frames: tuple[ResolvedDenseFrame, ...] = ()
    evidence: tuple[ResolvedQueryTemporalObservation, ...] = ()
    observable_evidence_summary: str
    uncertainties: tuple[str, ...] = ()
    confidence: Confidence
    model_provenance: ModelProvenance

    @model_validator(mode="after")
    def validate_resolved_order(self) -> "QueryTemporalDecision":
        positive = (
            self.match_status == MatchStatus.MATCHED
            and self.predicate_status == PredicateStatus.SATISFIED
        )
        frame_values = (
            self.candidate_frame,
            self.seed_frame,
            self.precondition_frame,
            self.apex_frame,
            self.postcondition_frame,
        )
        positive_frames: tuple[ResolvedDenseFrame, ...] = ()
        if not positive:
            if any(frame is not None for frame in frame_values) or self.interval_sample_frames:
                raise ValueError("non-positive temporal decisions cannot select frames")
            if self.coverage_claim != "no_positive_claim":
                raise ValueError("non-positive temporal decisions cannot claim coverage")
        elif self.required_at == PredicateRequiredAt.CANDIDATE:
            if self.candidate_frame is None or any(
                frame is not None for frame in frame_values[1:]
            ) or self.interval_sample_frames:
                raise ValueError("resolved candidate decision requires only candidate_frame")
            if self.coverage_claim != "single_frame_only":
                raise ValueError("resolved candidate decision has invalid coverage")
            positive_frames = (self.candidate_frame,)
        elif self.required_at == PredicateRequiredAt.SEED:
            if self.seed_frame is None or self.candidate_frame is not None or any(
                frame is not None for frame in frame_values[2:]
            ) or self.interval_sample_frames:
                raise ValueError("resolved seed decision requires only seed_frame")
            if self.coverage_claim != "single_frame_only":
                raise ValueError("resolved seed decision has invalid coverage")
            positive_frames = (self.seed_frame,)
        elif self.required_at == PredicateRequiredAt.TRANSITION:
            transition = (
                self.precondition_frame,
                self.apex_frame,
                self.postcondition_frame,
            )
            if (
                self.candidate_frame is not None
                or self.seed_frame is not None
                or any(frame is None for frame in transition)
                or self.interval_sample_frames
            ):
                raise ValueError("resolved transition is incomplete")
            if self.coverage_claim != "transition_samples_only":
                raise ValueError("resolved transition decision has invalid coverage")
            pts = [frame.frame_pts for frame in transition if frame is not None]
            if not pts[0] < pts[1] < pts[2]:
                raise ValueError("resolved transition must satisfy pre < apex < post PTS")
            positive_frames = tuple(
                frame for frame in transition if frame is not None
            )
        elif self.required_at == PredicateRequiredAt.INTERVAL:
            if any(frame is not None for frame in frame_values):
                raise ValueError("resolved interval decision uses only interval samples")
            if len(self.interval_sample_frames) < 2:
                raise ValueError("resolved interval decision needs at least two samples")
            if self.coverage_claim != "sampled_frames_only":
                raise ValueError("resolved interval decision has invalid coverage")
            pts = [frame.frame_pts for frame in self.interval_sample_frames]
            if any(current >= following for current, following in zip(pts, pts[1:])):
                raise ValueError("resolved interval samples must have strictly increasing PTS")
            positive_frames = self.interval_sample_frames
        evidence_by_id = {item.frame.frame_id: item for item in self.evidence}
        if len(evidence_by_id) != len(self.evidence):
            raise ValueError("resolved temporal evidence frame IDs must be unique")
        missing = {
            frame.frame_id for frame in positive_frames
        } - set(evidence_by_id)
        if missing:
            raise ValueError(
                f"resolved selected frames require observations: {sorted(missing)}"
            )
        if positive:
            for frame in positive_frames:
                observation = evidence_by_id[frame.frame_id]
                if observation.frame != frame:
                    raise ValueError(
                        "resolved temporal observation does not match its selected frame"
                    )
                if any(
                    status != "observed"
                    for status in observation.identity_status_by_target.values()
                ):
                    raise ValueError(
                        "positive temporal frames must observe every predicate participant"
                    )
                if observation.predicate_observed is not True:
                    raise ValueError("positive temporal frames must observe the predicate")
            if self.required_at == PredicateRequiredAt.TRANSITION:
                expected_phases = {
                    self.precondition_frame.frame_id: "precondition",
                    self.apex_frame.frame_id: "apex",
                    self.postcondition_frame.frame_id: "postcondition",
                }
                for frame_id, phase in expected_phases.items():
                    if evidence_by_id[frame_id].transition_phase != phase:
                        raise ValueError(
                            "transition observations must identify precondition/apex/postcondition"
                        )
        return self


class QueryTemporalEvidenceBundle(StrictModel):
    """Content-addressed envelope required before consuming a saved decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["query-temporal-evidence-bundle-v2"] = (
        "query-temporal-evidence-bundle-v2"
    )
    temporal_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grounding_target_id: str = Field(pattern=TARGET_ID_PATTERN)
    request_fingerprint: QueryTemporalFingerprint
    request_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_snapshot_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_interaction_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_template_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_schema_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _resolved_frame(frame: DenseFrame) -> ResolvedDenseFrame:
    return ResolvedDenseFrame(
        frame_id=frame.frame_id,
        requested_time_ms=frame.requested_time_ms,
        frame_time_ms=frame.frame_time_ms,
        frame_pts=frame.frame_pts,
        frame_hash=frame.frame_hash,
        width=frame.width,
        height=frame.height,
    )


def resolve_query_temporal_selection(
    *,
    selection: QueryTemporalSelection,
    query_lock: EvidenceQueryLockV2,
    catalog: DenseFrameCatalog,
    fingerprint: QueryTemporalFingerprint,
) -> QueryTemporalDecision:
    """Reject invented IDs/metadata, then resolve accepted IDs using local PTS."""

    predicate = query_lock.predicate
    if predicate is None:
        raise ValueError("QueryLock temporal refinement requires a predicate")
    expected = {
        "source_asset_id": catalog.source_asset_id,
        "event_id": catalog.event_id,
        "query_id": query_lock.query_id,
        "grounding_target_id": fingerprint.grounding_target_id,
        "identity_sha256": fingerprint.identity_sha256,
        "predicate_sha256": fingerprint.predicate_sha256,
        "catalog_sha256": fingerprint.catalog_sha256,
        "request_sha256": fingerprint.request_sha256,
        "required_at": predicate.required_at,
    }
    mismatches = {
        key: {"expected": value, "actual": getattr(selection, key)}
        for key, value in expected.items()
        if getattr(selection, key) != value
    }
    if mismatches:
        raise ValueError(f"temporal refinement immutable metadata mismatch: {mismatches}")

    frames = {frame.frame_id: frame for frame in catalog.frames}
    selected_ids = [
        selection.candidate_frame_id,
        selection.seed_frame_id,
        selection.precondition_frame_id,
        selection.apex_frame_id,
        selection.postcondition_frame_id,
        *selection.interval_sample_frame_ids,
        *(item.frame_id for item in selection.evidence),
    ]
    unknown = sorted({frame_id for frame_id in selected_ids if frame_id is not None} - set(frames))
    if unknown:
        raise ValueError(f"temporal refinement referenced unknown DF IDs: {unknown}")
    expected_participants = set(predicate.participant_target_ids)
    for item in selection.evidence:
        reported_participants = set(item.identity_status_by_target)
        if reported_participants != expected_participants:
            raise ValueError(
                "temporal evidence participant identities do not match the locked predicate: "
                f"expected={sorted(expected_participants)} "
                f"actual={sorted(reported_participants)}"
            )
    if (
        selection.match_status == MatchStatus.MATCHED
        and selection.predicate_status == PredicateStatus.SATISFIED
        and selection.required_at == PredicateRequiredAt.INTERVAL
    ):
        catalog_ids = [frame.frame_id for frame in catalog.frames]
        selected = selection.interval_sample_frame_ids
        first_index = catalog_ids.index(selected[0])
        expected_contiguous = catalog_ids[first_index : first_index + len(selected)]
        if selected != expected_contiguous:
            raise ValueError(
                "positive interval decisions must select a contiguous catalog run"
            )

    def resolve(frame_id: str | None) -> ResolvedDenseFrame | None:
        return None if frame_id is None else _resolved_frame(frames[frame_id])

    return QueryTemporalDecision(
        source_asset_id=selection.source_asset_id,
        event_id=selection.event_id,
        query_id=selection.query_id,
        grounding_target_id=selection.grounding_target_id,
        identity_sha256=selection.identity_sha256,
        predicate_sha256=selection.predicate_sha256,
        catalog_sha256=selection.catalog_sha256,
        request_sha256=selection.request_sha256,
        required_at=selection.required_at,
        match_status=selection.match_status,
        predicate_status=selection.predicate_status,
        coverage_claim=selection.coverage_claim,
        candidate_frame=resolve(selection.candidate_frame_id),
        seed_frame=resolve(selection.seed_frame_id),
        precondition_frame=resolve(selection.precondition_frame_id),
        apex_frame=resolve(selection.apex_frame_id),
        postcondition_frame=resolve(selection.postcondition_frame_id),
        interval_sample_frames=tuple(
            resolve(frame_id) for frame_id in selection.interval_sample_frame_ids
        ),
        evidence=tuple(
            ResolvedQueryTemporalObservation(
                frame=resolve(item.frame_id),
                identity_status_by_target=item.identity_status_by_target,
                predicate_observed=item.predicate_observed,
                transition_phase=item.transition_phase,
                observation=item.observation,
            )
            for item in selection.evidence
        ),
        observable_evidence_summary=selection.observable_evidence_summary,
        uncertainties=tuple(selection.uncertainties),
        confidence=selection.confidence,
        model_provenance=selection.model_provenance,
    )


def validate_query_temporal_evidence_bundle(
    decision_path: Path,
    *,
    query_lock: EvidenceQueryLockV2,
    expected_system_instruction: str,
    expected_model_id: str,
    expected_prompt_template: str,
) -> QueryTemporalDecision:
    """Re-resolve saved model IDs and reject stale, detached, or edited bundles."""

    run_dir = decision_path.parent
    paths = {
        "request_file_sha256": run_dir / "query_temporal.request.json",
        "selection_file_sha256": run_dir / "query_temporal.selection.json",
        "decision_file_sha256": decision_path,
        "catalog_snapshot_file_sha256": (
            run_dir / "query_temporal.catalog.snapshot.json"
        ),
        "raw_interaction_file_sha256": (
            run_dir / "query_temporal.raw_interaction.json"
        ),
        "prompt_template_file_sha256": (
            run_dir / "query_temporal.prompt_template.json"
        ),
        "response_schema_file_sha256": (
            run_dir / "query_temporal.response_schema.json"
        ),
    }
    bundle = QueryTemporalEvidenceBundle.model_validate(
        read_json(run_dir / "query_temporal.bundle.json")
    )
    expected_temporal_contract = query_temporal_contract_sha256(
        query_lock, bundle.grounding_target_id
    )
    if bundle.temporal_contract_sha256 != expected_temporal_contract:
        raise ValueError("temporal bundle identity/predicate contract does not match QueryLock")
    for field_name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"temporal evidence bundle file is missing: {path.name}")
        if sha256_file(path) != getattr(bundle, field_name):
            raise ValueError(f"temporal evidence bundle hash mismatch: {path.name}")

    request = read_json(paths["request_file_sha256"])
    request_fingerprint = QueryTemporalFingerprint.model_validate(
        request.get("fingerprint")
    )
    if request_fingerprint != bundle.request_fingerprint:
        raise ValueError("temporal request fingerprint differs from bundle")
    if bundle.grounding_target_id != request_fingerprint.grounding_target_id:
        raise ValueError("temporal bundle target differs from request fingerprint")
    recorded_request_values = {
        "model": request.get("model"),
        "system_instruction": request.get("system_instruction"),
        "generation_config": request.get("generation_config"),
        "response_schema": (
            (request.get("response_format") or {}).get("schema")
        ),
    }
    expected_request_values = {
        "model": expected_model_id,
        "system_instruction": expected_system_instruction,
        "generation_config": QUERY_TEMPORAL_GENERATION_CONFIG,
        "response_schema": gemini_response_schema(QueryTemporalSelection),
    }
    if recorded_request_values != expected_request_values:
        raise ValueError("temporal recorded API request differs from current contract")
    current_fixed_hashes = {
        "system_instruction_sha256": hashlib.sha256(
            expected_system_instruction.encode("utf-8")
        ).hexdigest(),
        "response_schema_sha256": _canonical_sha256(
            gemini_response_schema(QueryTemporalSelection)
        ),
        "task_instruction_sha256": hashlib.sha256(
            QUERY_TEMPORAL_TASK_INSTRUCTIONS.encode("utf-8")
        ).hexdigest(),
        "generation_config_sha256": _canonical_sha256(
            QUERY_TEMPORAL_GENERATION_CONFIG
        ),
        "model_id": expected_model_id,
        "prompt_sha256": hashlib.sha256(
            expected_prompt_template.encode("utf-8")
        ).hexdigest(),
    }
    stale_fixed_inputs = {
        key: {
            "expected": value,
            "actual": getattr(request_fingerprint, key),
        }
        for key, value in current_fixed_hashes.items()
        if getattr(request_fingerprint, key) != value
    }
    if stale_fixed_inputs:
        raise ValueError(
            "temporal evidence was created under stale fixed runtime inputs: "
            f"{stale_fixed_inputs}"
        )
    prompt_template_payload = read_json(paths["prompt_template_file_sha256"])
    prompt_template = prompt_template_payload.get("prompt_template")
    if not isinstance(prompt_template, str) or not prompt_template:
        raise ValueError("temporal prompt template artifact is invalid")
    if (
        hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()
        != request_fingerprint.prompt_sha256
    ):
        raise ValueError("temporal prompt template differs from request fingerprint")
    saved_schema = read_json(paths["response_schema_file_sha256"])
    if _canonical_sha256(saved_schema) != request_fingerprint.response_schema_sha256:
        raise ValueError("temporal response schema differs from request fingerprint")
    catalog = DenseFrameCatalog.model_validate(
        read_json(paths["catalog_snapshot_file_sha256"])
    )
    if dense_catalog_evidence_sha256(catalog) != request_fingerprint.catalog_sha256:
        raise ValueError("temporal catalog evidence hash differs from request")
    if request.get("frame_ids_in_order") != [
        frame.frame_id for frame in catalog.frames
    ]:
        raise ValueError("temporal request frame order differs from catalog")
    recorded_input = request.get("input")
    if not isinstance(recorded_input, list) or not recorded_input:
        raise ValueError("temporal recorded input is missing")
    first_item = recorded_input[0]
    if (
        not isinstance(first_item, dict)
        or first_item.get("type") != "text"
        or not isinstance(first_item.get("text"), str)
        or not first_item["text"].startswith(expected_prompt_template)
    ):
        raise ValueError("temporal task prompt does not begin with the current template")
    recorded_image_hashes = [
        item.get("sha256")
        for item in recorded_input
        if isinstance(item, dict) and item.get("type") == "image"
    ]
    if recorded_image_hashes != list(catalog.contact_sheet_hashes):
        raise ValueError("temporal recorded image evidence differs from catalog")
    selection = QueryTemporalSelection.model_validate(
        read_json(paths["selection_file_sha256"])
    )
    raw_interaction = read_json(paths["raw_interaction_file_sha256"])
    raw_output_text = raw_interaction.get("output_text")
    if isinstance(raw_output_text, str):
        raw_selection = QueryTemporalSelection.model_validate_json(raw_output_text)
        normalized_saved_selection = selection.model_copy(
            update={
                "model_provenance": selection.model_provenance.model_copy(
                    update={
                        "interaction_id": raw_selection.model_provenance.interaction_id
                    }
                )
            }
        )
        if normalized_saved_selection != raw_selection:
            raise ValueError("temporal parsed selection differs from raw model output")
        raw_interaction_id = raw_interaction.get("id")
        if (
            raw_interaction_id is not None
            and selection.model_provenance.interaction_id != raw_interaction_id
        ):
            raise ValueError("temporal interaction ID lineage differs from raw response")
    resolved = resolve_query_temporal_selection(
        selection=selection,
        query_lock=query_lock,
        catalog=catalog,
        fingerprint=request_fingerprint,
    )
    saved = QueryTemporalDecision.model_validate(
        read_json(paths["decision_file_sha256"])
    )
    if saved != resolved:
        raise ValueError("saved temporal decision differs from re-resolved evidence")
    return saved
