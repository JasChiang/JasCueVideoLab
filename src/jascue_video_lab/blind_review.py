from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from .billing import summarize_usage_and_list_price
from .gemini import GeminiLabClient
from .geometry import box_iou, center_distance
from .media import create_analysis_proxy, extract_frame, probe_video
from .models import (
    DirectMomentMap,
    DirectVideoGroundingProposal,
    EvidenceApprovalSource,
    EvidenceClaimSource,
    EvidenceFramingObligationsV2,
    EvidenceIdentityContractV2,
    EvidencePredicateContractV2,
    EvidencePredicatePhasesV2,
    EvidenceQueryApprovalProvenance,
    EvidenceQueryLockV2,
    EvidenceQueryProposalV2,
    EvidenceQueryProvenanceV2,
    EvidenceTargetIdentityV2,
    GroundingProposal,
    MediaInfo,
    PredicateRequiredAt,
    StrictModel,
    TargetCandidateMap,
    TargetIdentityScope,
    approve_evidence_query_proposal_v2,
)
from .overlay import draw_blind_review_overlay, draw_grounding_overlay
from .storage import append_error, read_json, utc_now, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_ROOT = PROJECT_ROOT / "prompts"
DEFAULT_APP_DATA_ROOT = PROJECT_ROOT / "artifacts" / "blind-review-app"
DEFAULT_FILE_CACHE_ROOT = PROJECT_ROOT / "artifacts" / "blind-review-file-cache"
NormalizedCoordinate = Annotated[int, Field(ge=0, le=1000)]


class ReviewVerdict(StrEnum):
    CORRECT = "correct"
    WRONG_OBJECT = "wrong_object"
    TOO_LARGE = "box_too_large"
    TOO_SMALL = "box_too_small"
    TARGET_NOT_VISIBLE = "target_not_visible"
    SAMPLE_FRAME_MISMATCH = "sample_frame_mismatch"
    UNABLE_TO_JUDGE = "unable_to_judge"


class TargetSelection(StrictModel):
    asset_id: str
    target_id: str = Field(min_length=1)
    target_description: str = Field(min_length=1)
    selected_by: Literal["human_candidate_selection", "human_manual_description"]
    source_candidate_id: str | None = None
    selected_at: str
    contract_target_id: str | None = None
    query_status: Literal[
        "legacy_selection", "proposal_pending_approval", "query_locked"
    ] = "legacy_selection"
    query_proposal: EvidenceQueryProposalV2 | None = None
    query_proposal_hashes: dict[str, str] | None = None
    query_lock_id: str | None = None
    query_lock_hashes: dict[str, str] | None = None


class QueryProposalOptions(StrictModel):
    """Optional, domain-neutral refinements for a selected target proposal."""

    editorial_goal: str | None = None
    identity_scope: TargetIdentityScope = TargetIdentityScope.WHOLE_INSTANCE
    parent_target_id: str | None = None
    parent_target_description: str | None = None
    parent_identity_cues: tuple[str, ...] = ()
    identity_cues: tuple[str, ...] = ()
    context_cues: tuple[str, ...] = ()
    stable_exclusions: tuple[str, ...] = ()
    observable_predicate: str | None = None
    predicate_required_at: PredicateRequiredAt = PredicateRequiredAt.CANDIDATE
    predicate_precondition: str | None = None
    predicate_apex: str | None = None
    predicate_postcondition: str | None = None
    predicate_required_evidence: tuple[str, ...] = ()
    predicate_disqualifying_conditions: tuple[str, ...] = ()
    framing_required_target_ids: tuple[str, ...] = ()
    framing_preferred_target_ids: tuple[str, ...] = ()
    framing_sacrificable_target_ids: tuple[str, ...] = ()
    framing_overlay_keepout_target_ids: tuple[str, ...] = ()
    framing_intent: str | None = None

    @model_validator(mode="after")
    def validate_predicate_options(self) -> "QueryProposalOptions":
        if self.identity_scope == TargetIdentityScope.WHOLE_INSTANCE:
            if self.parent_target_id or self.parent_target_description or self.parent_identity_cues:
                raise ValueError("whole_instance proposals cannot define a parent target")
        elif not self.parent_target_id or not self.parent_target_description:
            raise ValueError(
                "subpart and visible_region proposals require parent target ID and description"
            )
        phase_values = (
            self.predicate_precondition,
            self.predicate_apex,
            self.predicate_postcondition,
        )
        populated_phase_count = sum(bool(value and value.strip()) for value in phase_values)
        if populated_phase_count not in {0, 3}:
            raise ValueError("predicate phases must be omitted or supplied as a complete triplet")
        if any(phase_values) and not self.observable_predicate:
            raise ValueError("predicate phases require observable_predicate")
        if self.observable_predicate and self.predicate_required_at == PredicateRequiredAt.TRANSITION:
            if not all(value and value.strip() for value in phase_values):
                raise ValueError(
                    "transition predicates require precondition, apex, and postcondition"
                )
        if not self.observable_predicate and (
            self.predicate_required_evidence
            or self.predicate_disqualifying_conditions
        ):
            raise ValueError("predicate evidence rules require observable_predicate")
        return self


class HumanReviewAnnotation(StrictModel):
    annotation_id: str
    session_id: str
    review_id: str
    asset_id: str
    reviewer_type: Literal["human"]
    reviewer_name: str | None = None
    target_id: str
    target_description: str
    grounding_method: Literal["exact_frame_image", "direct_video_unknown_sample"]
    bbox_reference_frame: Literal["exact_ffmpeg_frame", "unknown_gemini_video_sample"]
    requested_timestamp_mmss: str
    requested_time_ms: int = Field(ge=0)
    frame_pts: int
    frame_time_ms: int = Field(ge=0)
    frame_hash: str
    verdict: ReviewVerdict
    notes: str = ""
    corrected_box_2d: tuple[
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
        NormalizedCoordinate,
    ] | None = None
    model_details_revealed_before_annotation: Literal[False]
    annotated_at: str

    @model_validator(mode="after")
    def validate_corrected_box(self) -> "HumanReviewAnnotation":
        if self.corrected_box_2d is not None:
            x_min, y_min, x_max, y_max = self.corrected_box_2d
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("corrected_box_2d must satisfy xmin < xmax and ymin < ymax")
        return self


def _mmss_to_ms(value: str) -> int:
    minutes, seconds = (int(part) for part in value.split(":"))
    return (minutes * 60 + seconds) * 1000


def _prompt(name: str) -> str:
    return (PROMPTS_ROOT / name).read_text(encoding="utf-8")


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if 1 < len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    return ".mp4"


_CONTRACT_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]*$")


def _contract_id(value: str, *, prefix: str) -> str:
    """Preserve valid legacy IDs and deterministically map permissive old IDs."""

    stripped = value.strip()
    if _CONTRACT_ID.fullmatch(stripped):
        return stripped
    digest = hashlib.sha256(stripped.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _definition_sha256(value: Any) -> str:
    payload = value.model_dump(mode="json", exclude_none=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _query_artifact_hashes(
    value: EvidenceQueryProposalV2 | EvidenceQueryLockV2,
) -> dict[str, str]:
    return {
        **value.component_hashes(),
        "composite_sha256": value.composite_sha256(),
        "definition_sha256": _definition_sha256(value),
    }


def _remap_target_ids(
    values: tuple[str, ...], *, aliases: dict[str, str]
) -> tuple[str, ...]:
    return tuple(aliases.get(value, value) for value in values)


def _build_query_proposal(
    *,
    target_id: str,
    target_description: str,
    candidate_features: str | None,
    claim_source: EvidenceClaimSource,
    created_by: str,
    source_reference: str | None,
    options: QueryProposalOptions,
) -> EvidenceQueryProposalV2:
    contract_target_id = _contract_id(target_id, prefix="target")
    aliases = {target_id: contract_target_id, contract_target_id: contract_target_id}
    parent_contract_id: str | None = None
    targets: list[EvidenceTargetIdentityV2] = []
    if options.parent_target_id is not None:
        parent_contract_id = _contract_id(options.parent_target_id, prefix="parent")
        aliases.update(
            {
                options.parent_target_id: parent_contract_id,
                parent_contract_id: parent_contract_id,
            }
        )
        parent_description = options.parent_target_description or options.parent_target_id
        targets.append(
            EvidenceTargetIdentityV2(
                target_id=parent_contract_id,
                target_description=parent_description,
                scope=TargetIdentityScope.WHOLE_INSTANCE,
                identity_cues=(
                    options.parent_identity_cues
                    or (parent_description,)
                ),
            )
        )
    identity_cues = options.identity_cues
    if not identity_cues:
        identity_cues = tuple(
            item
            for item in (candidate_features, target_description)
            if item and item.strip()
        )
    targets.append(
        EvidenceTargetIdentityV2(
            target_id=contract_target_id,
            target_description=target_description,
            scope=options.identity_scope,
            parent_target_id=parent_contract_id,
            identity_cues=identity_cues,
            context_cues=options.context_cues,
            stable_exclusions=options.stable_exclusions,
        )
    )
    predicate: EvidencePredicateContractV2 | None = None
    if options.observable_predicate:
        phase_values = (
            options.predicate_precondition,
            options.predicate_apex,
            options.predicate_postcondition,
        )
        predicate = EvidencePredicateContractV2(
            predicate_id=f"{contract_target_id}:predicate",
            statement=options.observable_predicate,
            participant_target_ids=(contract_target_id,),
            required_at=options.predicate_required_at,
            phases=(
                EvidencePredicatePhasesV2(
                    precondition=phase_values[0],
                    apex=phase_values[1],
                    postcondition=phase_values[2],
                )
                if all(phase_values)
                else None
            ),
            required_evidence=options.predicate_required_evidence,
            disqualifying_conditions=options.predicate_disqualifying_conditions,
        )
    framing_roles = (
        options.framing_required_target_ids,
        options.framing_preferred_target_ids,
        options.framing_sacrificable_target_ids,
    )
    if not any(framing_roles):
        framing_required = (contract_target_id,)
    else:
        framing_required = _remap_target_ids(
            options.framing_required_target_ids, aliases=aliases
        )
    framing = EvidenceFramingObligationsV2(
        required_target_ids=framing_required,
        preferred_target_ids=_remap_target_ids(
            options.framing_preferred_target_ids, aliases=aliases
        ),
        sacrificable_target_ids=_remap_target_ids(
            options.framing_sacrificable_target_ids, aliases=aliases
        ),
        overlay_keepout_target_ids=_remap_target_ids(
            options.framing_overlay_keepout_target_ids, aliases=aliases
        ),
        framing_intent=(
            options.framing_intent
            or "Preserve the selected evidence target for the intended edit."
        ),
    )
    return EvidenceQueryProposalV2(
        proposal_id=f"query-proposal-{uuid.uuid4().hex}",
        revision=1,
        editorial_goal=(
            options.editorial_goal
            or "Ground and review the selected visual evidence target."
        ),
        identity=EvidenceIdentityContractV2(targets=tuple(targets)),
        predicate=predicate,
        framing=framing,
        claim_source=claim_source,
        provenance=EvidenceQueryProvenanceV2(
            created_at=utc_now(),
            created_by=created_by,
            source_reference=source_reference,
        ),
    )


class BlindReviewService:
    """Durable local workflow. Model details stay inaccessible until a human annotation exists."""

    def __init__(
        self,
        *,
        data_root: Path = DEFAULT_APP_DATA_ROOT,
        file_cache_root: Path = DEFAULT_FILE_CACHE_ROOT,
        client_factory: Any = GeminiLabClient,
    ) -> None:
        self.data_root = data_root
        self.file_cache_root = file_cache_root
        self.client_factory = client_factory
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.file_cache_root.mkdir(parents=True, exist_ok=True)
        self._upload_locks: dict[str, threading.Lock] = {}
        self._approval_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _session_dir(self, session_id: str) -> Path:
        try:
            normalized = uuid.UUID(session_id).hex
        except ValueError as error:
            raise FileNotFoundError("invalid session id") from error
        path = self.data_root / normalized
        if not path.exists():
            raise FileNotFoundError(f"unknown session {normalized}")
        return path

    def _session(self, session_id: str) -> dict[str, Any]:
        return read_json(self._session_dir(session_id) / "session.json")

    def _save_session(self, session: dict[str, Any]) -> None:
        session["updated_at"] = utc_now()
        write_json(self.data_root / session["session_id"] / "session.json", session)

    def create_session_from_file(
        self,
        source_path: Path,
        *,
        original_filename: str,
        use_analysis_proxy: bool = True,
        move_source: bool = False,
    ) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        session_dir = self.data_root / session_id
        session_dir.mkdir(parents=True, exist_ok=False)
        destination = session_dir / f"source{_safe_suffix(original_filename)}"
        if move_source:
            os.replace(source_path, destination)
        else:
            shutil.copy2(source_path, destination)
        try:
            media = probe_video(destination)
        except Exception as error:
            append_error(session_dir, "probe_uploaded_video", error)
            raise
        write_json(session_dir / "media.json", media)
        session: dict[str, Any] = {
            "session_id": session_id,
            "stage": "uploaded",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "original_filename": original_filename,
            "source_path": str(destination.resolve()),
            "use_analysis_proxy": use_analysis_proxy,
            "asset_id": media.asset_id,
            "current_candidate_map": None,
            "current_selection": None,
            "current_query_proposal": None,
            "current_query_proposal_manifest": None,
            "current_query_lock": None,
            "current_query_lock_manifest": None,
            "current_moment_map": None,
            "reviews": [],
        }
        self._save_session(session)
        return self.session_view(session_id)

    def _media(self, session: dict[str, Any]) -> MediaInfo:
        return MediaInfo.model_validate(read_json(self.data_root / session["session_id"] / "media.json"))

    def _analysis_source(self, session: dict[str, Any]) -> tuple[Path, MediaInfo, dict[str, Any] | None]:
        source = Path(session["source_path"])
        if not session["use_analysis_proxy"]:
            return source, self._media(session), None
        session_dir = self.data_root / session["session_id"]
        proxy_path = session_dir / "analysis-proxy.mp4"
        record_path = session_dir / "analysis_proxy.json"
        if proxy_path.exists() and record_path.exists():
            return proxy_path, probe_video(proxy_path), read_json(record_path)
        try:
            proxy_media, record = create_analysis_proxy(source, proxy_path)
        except Exception as error:
            append_error(session_dir, "create_analysis_proxy", error)
            raise
        write_json(record_path, record)
        return proxy_path, proxy_media, record

    def _lock_for_upload(self, asset_hash: str) -> threading.Lock:
        with self._locks_guard:
            return self._upload_locks.setdefault(asset_hash, threading.Lock())

    def _lock_for_approval(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._approval_locks.setdefault(session_id, threading.Lock())

    def _adopt_legacy_upload(self, upload_media: MediaInfo, cache_dir: Path) -> str | None:
        initial = cache_dir / "file_upload_initial.json"
        if initial.exists():
            return None
        artifacts_root = PROJECT_ROOT / "artifacts"
        for record_path in artifacts_root.rglob("analysis_proxy.json"):
            if cache_dir in record_path.parents:
                continue
            try:
                record = read_json(record_path)
                if record.get("proxy_media", {}).get("asset_id") != upload_media.asset_id:
                    continue
                legacy_upload = record_path.parent / "upload"
                if not (legacy_upload / "file_upload_initial.json").exists():
                    continue
                for filename in ("file_upload_initial.json", "file_upload_final.json"):
                    source = legacy_upload / filename
                    if source.exists():
                        write_json(cache_dir / filename, read_json(source))
                write_json(
                    cache_dir / "registry_adoption.json",
                    {"adopted_from": str(legacy_upload), "adopted_at": utc_now()},
                )
                return str(legacy_upload)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        for identity_path in artifacts_root.rglob("upload_source_identity.json"):
            if cache_dir in identity_path.parents:
                continue
            try:
                identity = read_json(identity_path)
                if identity.get("upload_asset_id") != upload_media.asset_id:
                    continue
                legacy_upload = identity_path.parent / "upload"
                if not (legacy_upload / "file_upload_initial.json").exists():
                    continue
                for filename in ("file_upload_initial.json", "file_upload_final.json"):
                    source = legacy_upload / filename
                    if source.exists():
                        write_json(cache_dir / filename, read_json(source))
                write_json(
                    cache_dir / "registry_adoption.json",
                    {"adopted_from": str(legacy_upload), "adopted_at": utc_now()},
                )
                return str(legacy_upload)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return None

    def _ensure_upload(
        self, client: GeminiLabClient, session: dict[str, Any]
    ) -> tuple[Any, bool, MediaInfo]:
        upload_source, upload_media, proxy_record = self._analysis_source(session)
        cache_dir = self.file_cache_root / upload_media.sha256 / "upload"
        with self._lock_for_upload(upload_media.sha256):
            adopted_from = self._adopt_legacy_upload(upload_media, cache_dir)
            uploaded, reused = client.ensure_video_upload(upload_source, cache_dir)
        cache_record = {
            "upload_asset_id": upload_media.asset_id,
            "cache_dir": str(cache_dir),
            "reused": reused,
            "adopted_from": adopted_from,
            "analysis_proxy": proxy_record,
            "checked_at": utc_now(),
        }
        write_json(self.data_root / session["session_id"] / "file_api_cache.json", cache_record)
        return uploaded, reused, upload_media

    def suggest_targets(
        self, session_id: str, *, runs: int = 1
    ) -> dict[str, Any]:
        if not 1 <= runs <= 5:
            raise ValueError("runs must be between 1 and 5")
        session = self._session(session_id)
        media = self._media(session)
        batch_dir = self._session_dir(session_id) / "candidates" / f"batch-{uuid.uuid4().hex[:8]}"
        client = self.client_factory()
        started = monotonic()
        maps: list[TargetCandidateMap] = []
        try:
            uploaded, reused, _ = self._ensure_upload(client, session)
            for index in range(1, runs + 1):
                run_dir = batch_dir / f"run-{index:02d}"
                candidate_map = client.suggest_targets(
                    media=media,
                    uploaded=uploaded,
                    prompt_template=_prompt("target_candidates_mmss_zh-TW.txt"),
                    run_id=f"blind-candidates-{index:02d}-{uuid.uuid4().hex[:8]}",
                    run_dir=run_dir,
                )
                maps.append(candidate_map)
        finally:
            client.close()
        session["stage"] = "candidates_ready"
        session["current_candidate_map"] = str(
            (batch_dir / "run-01" / "target_candidates.json").relative_to(self._session_dir(session_id))
        )
        session["current_selection"] = None
        session["current_query_proposal"] = None
        session["current_query_proposal_manifest"] = None
        session["current_query_lock"] = None
        session["current_query_lock_manifest"] = None
        session["current_moment_map"] = None
        self._save_session(session)
        pricing = summarize_usage_and_list_price(batch_dir)
        result = {
            "session_id": session_id,
            "file_api_object_reused": reused,
            "elapsed_seconds": round(monotonic() - started, 6),
            "pricing": pricing,
            "runs": [candidate_map.model_dump(mode="json") for candidate_map in maps],
        }
        write_json(batch_dir / "summary.json", result)
        return result

    def select_target(
        self,
        session_id: str,
        *,
        candidate_id: str | None = None,
        target_id: str | None = None,
        target_description: str | None = None,
        query_options: QueryProposalOptions | None = None,
    ) -> TargetSelection:
        session = self._session(session_id)
        media = self._media(session)
        options = query_options or QueryProposalOptions()
        candidate_features: str | None = None
        if candidate_id:
            if target_id or target_description:
                raise ValueError("candidate selection and manual target are mutually exclusive")
            candidate_path = session.get("current_candidate_map")
            if not candidate_path:
                raise ValueError("generate candidates before selecting a candidate")
            candidate_map = TargetCandidateMap.model_validate(
                read_json(self._session_dir(session_id) / candidate_path)
            )
            candidate = next(
                (item for item in candidate_map.candidates if item.candidate_id == candidate_id), None
            )
            if candidate is None:
                raise ValueError(f"unknown candidate_id {candidate_id}")
            selected_target_id = candidate.candidate_id
            selected_description = candidate.target_description
            candidate_features = candidate.distinguishing_features
            selected_by = "human_candidate_selection"
            source_candidate_id = candidate.candidate_id
            claim_source = EvidenceClaimSource.MODEL_PROPOSAL
            created_by = "gemini_target_candidate"
            source_reference = (
                f"{candidate_map.model_provenance.interaction_id}:"
                f"{candidate.candidate_id}"
            )
        else:
            if not target_id or not target_description:
                raise ValueError("manual target_id and target_description are required together")
            selected_target_id = target_id
            selected_description = target_description
            selected_by = "human_manual_description"
            source_candidate_id = None
            claim_source = EvidenceClaimSource.USER_BRIEF
            created_by = "human_manual_description"
            source_reference = f"session:{session_id}:manual-target"
        proposal = _build_query_proposal(
            target_id=selected_target_id,
            target_description=selected_description,
            candidate_features=candidate_features,
            claim_source=claim_source,
            created_by=created_by,
            source_reference=source_reference,
            options=options,
        )
        proposal_hashes = _query_artifact_hashes(proposal)
        selection = TargetSelection(
            asset_id=media.asset_id,
            target_id=selected_target_id,
            target_description=selected_description,
            selected_by=selected_by,
            source_candidate_id=source_candidate_id,
            selected_at=utc_now(),
            contract_target_id=proposal.identity.targets[-1].target_id,
            query_status="proposal_pending_approval",
            query_proposal=proposal,
            query_proposal_hashes=proposal_hashes,
        )
        write_json(self._session_dir(session_id) / "target_selection.json", selection)
        proposal_dir = Path("queries") / proposal.proposal_id
        proposal_path = proposal_dir / "proposal.json"
        proposal_manifest_path = proposal_dir / "proposal.manifest.json"
        write_json(self._session_dir(session_id) / proposal_path, proposal)
        proposal_manifest = {
            "artifact_kind": "evidence_query_proposal_v2",
            "proposal_id": proposal.proposal_id,
            "hashes": proposal_hashes,
            "approval_status": "pending",
            "written_at": utc_now(),
        }
        write_json(
            self._session_dir(session_id) / proposal_manifest_path,
            proposal_manifest,
        )
        session["stage"] = "query_proposal_ready"
        session["current_selection"] = "target_selection.json"
        session["current_query_proposal"] = str(proposal_path)
        session["current_query_proposal_manifest"] = str(proposal_manifest_path)
        session["current_query_lock"] = None
        session["current_query_lock_manifest"] = None
        session["current_moment_map"] = None
        self._save_session(session)
        return selection

    def approve_query_proposal(
        self,
        session_id: str,
        *,
        approved_by: str,
        expected_proposal_id: str,
        expected_proposal_definition_sha256: str,
        approval_source: EvidenceApprovalSource = EvidenceApprovalSource.HUMAN_REVIEW,
        query_id: str | None = None,
        source_reference: str | None = None,
        policy_reference: str | None = None,
    ) -> dict[str, Any]:
        """Atomically compare-and-approve one reviewed proposal per session."""

        with self._lock_for_approval(session_id):
            return self._approve_query_proposal_locked(
                session_id,
                approved_by=approved_by,
                expected_proposal_id=expected_proposal_id,
                expected_proposal_definition_sha256=(
                    expected_proposal_definition_sha256
                ),
                approval_source=approval_source,
                query_id=query_id,
                source_reference=source_reference,
                policy_reference=policy_reference,
            )

    def _approve_query_proposal_locked(
        self,
        session_id: str,
        *,
        approved_by: str,
        expected_proposal_id: str,
        expected_proposal_definition_sha256: str,
        approval_source: EvidenceApprovalSource = EvidenceApprovalSource.HUMAN_REVIEW,
        query_id: str | None = None,
        source_reference: str | None = None,
        policy_reference: str | None = None,
    ) -> dict[str, Any]:
        """Explicitly promote the current proposal without rewriting its claim source."""

        if isinstance(approval_source, str):
            approval_source = EvidenceApprovalSource(approval_source)
        if approval_source != EvidenceApprovalSource.HUMAN_REVIEW:
            raise ValueError("Blind Review approval must be human_review")
        if not approved_by.strip():
            raise ValueError("approved_by is required")
        if policy_reference is not None:
            raise ValueError("Blind Review human approval cannot claim an auto policy")
        session = self._session(session_id)
        if session.get("current_query_lock"):
            raise ValueError(
                "the current proposal is already approved; create a new proposal to revise it"
            )
        proposal_path = session.get("current_query_proposal")
        if not proposal_path:
            raise ValueError("select a target and create a proposal before approval")
        proposal = EvidenceQueryProposalV2.model_validate(
            read_json(self._session_dir(session_id) / proposal_path)
        )
        proposal_hashes = _query_artifact_hashes(proposal)
        if proposal.proposal_id != expected_proposal_id:
            raise ValueError(
                "approval refers to a stale proposal_id; review the current proposal"
            )
        if (
            proposal_hashes["definition_sha256"]
            != expected_proposal_definition_sha256
        ):
            raise ValueError(
                "approval refers to a stale proposal definition; review the current proposal"
            )
        approval = EvidenceQueryApprovalProvenance(
            approved_at=utc_now(),
            approved_by=approved_by,
            approval_source=approval_source,
            source_reference=(
                source_reference or f"session:{session_id}:proposal-review"
            ),
            policy_reference=policy_reference,
        )
        lock = approve_evidence_query_proposal_v2(
            proposal,
            query_id=(query_id or f"query-{uuid.uuid4().hex}"),
            approval=approval,
        )
        lock_hashes = _query_artifact_hashes(lock)
        lock_path = Path(proposal_path).parent / "lock.json"
        lock_manifest_path = Path(proposal_path).parent / "lock.manifest.json"
        write_json(self._session_dir(session_id) / lock_path, lock)
        lock_manifest = {
            "artifact_kind": "evidence_query_lock_v2",
            "query_id": lock.query_id,
            "source_proposal_id": proposal.proposal_id,
            "source_proposal_definition_sha256": proposal_hashes["definition_sha256"],
            "hashes": lock_hashes,
            "approval": approval.model_dump(mode="json"),
            "written_at": utc_now(),
        }
        write_json(
            self._session_dir(session_id) / lock_manifest_path,
            lock_manifest,
        )
        proposal_manifest_path = session.get("current_query_proposal_manifest")
        if proposal_manifest_path:
            proposal_manifest = read_json(
                self._session_dir(session_id) / proposal_manifest_path
            )
            proposal_manifest["approval_status"] = "approved"
            proposal_manifest["approved_query_id"] = lock.query_id
            proposal_manifest["approved_at"] = approval.approved_at
            write_json(
                self._session_dir(session_id) / proposal_manifest_path,
                proposal_manifest,
            )
        selection_path = session.get("current_selection")
        if selection_path:
            selection = TargetSelection.model_validate(
                read_json(self._session_dir(session_id) / selection_path)
            ).model_copy(
                update={
                    "query_status": "query_locked",
                    "query_lock_id": lock.query_id,
                    "query_lock_hashes": lock_hashes,
                }
            )
            write_json(self._session_dir(session_id) / selection_path, selection)
        session["stage"] = "query_locked"
        session["current_query_lock"] = str(lock_path)
        session["current_query_lock_manifest"] = str(lock_manifest_path)
        self._save_session(session)
        return {
            "session_id": session_id,
            "query_lock": lock.model_dump(mode="json"),
            "hashes": lock_hashes,
            "source_proposal": {
                "proposal_id": proposal.proposal_id,
                "claim_source": proposal.claim_source,
                "definition_sha256": proposal_hashes["definition_sha256"],
            },
        }

    def _locked_query_material(
        self,
        session_id: str,
        session: dict[str, Any],
        selection: TargetSelection,
    ) -> tuple[str, str, str | None]:
        """Return approved identity material, preserving legacy saved sessions."""

        if selection.query_status == "legacy_selection":
            return selection.target_id, selection.target_description, None
        if selection.query_status != "query_locked":
            raise ValueError(
                "the current Query proposal must be explicitly approved before "
                "moment analysis or Grounding"
            )
        lock_path = session.get("current_query_lock")
        if not lock_path:
            raise ValueError("query_locked selection is missing its lock artifact")
        lock = EvidenceQueryLockV2.model_validate(
            read_json(self._session_dir(session_id) / lock_path)
        )
        actual_hashes = _query_artifact_hashes(lock)
        if selection.query_lock_hashes != actual_hashes:
            raise ValueError("saved QueryLock hashes do not match the selected target")
        contract_target_id = selection.contract_target_id
        if contract_target_id is None:
            raise ValueError("query_locked selection is missing contract_target_id")
        target = next(
            (
                candidate
                for candidate in lock.identity.targets
                if candidate.target_id == contract_target_id
            ),
            None,
        )
        if target is None:
            raise ValueError("QueryLock does not contain the selected contract target")
        identity_parts = [
            target.target_description,
            f"identity scope: {target.scope.value}",
        ]
        if target.identity_cues:
            identity_parts.append("identity cues: " + "; ".join(target.identity_cues))
        if target.context_cues:
            identity_parts.append(
                "auxiliary context only: " + "; ".join(target.context_cues)
            )
        if target.stable_exclusions:
            identity_parts.append(
                "must exclude: " + "; ".join(target.stable_exclusions)
            )
        for ancestor in lock.identity.ancestors(target.target_id):
            parent_parts = [
                f"parent instance disambiguator {ancestor.target_id}: "
                + ancestor.target_description
            ]
            if ancestor.identity_cues:
                parent_parts.append("identity cues=" + "; ".join(ancestor.identity_cues))
            if ancestor.context_cues:
                parent_parts.append("context only=" + "; ".join(ancestor.context_cues))
            if ancestor.stable_exclusions:
                parent_parts.append("exclude=" + "; ".join(ancestor.stable_exclusions))
            identity_parts.append("; ".join(parent_parts))
        if lock.identity.ancestors(target.target_id):
            identity_parts.append(
                "Parent identity only disambiguates the requested subpart; Grounding "
                "must still return the child target geometry."
            )
        predicate_material: str | None = None
        if lock.predicate is not None:
            predicate_parts = [
                f"observable predicate ({lock.predicate.required_at.value}): "
                + lock.predicate.statement
            ]
            if lock.predicate.phases is not None:
                predicate_parts.extend(
                    [
                        "precondition: " + lock.predicate.phases.precondition,
                        "apex: " + lock.predicate.phases.apex,
                        "postcondition: " + lock.predicate.phases.postcondition,
                    ]
                )
            if lock.predicate.required_evidence:
                predicate_parts.append(
                    "required evidence: "
                    + "; ".join(lock.predicate.required_evidence)
                )
            if lock.predicate.disqualifying_conditions:
                predicate_parts.append(
                    "disqualifying conditions: "
                    + "; ".join(lock.predicate.disqualifying_conditions)
                )
            predicate_material = "\n".join(predicate_parts)
        return contract_target_id, "\n".join(identity_parts), predicate_material

    def analyze_moments(
        self, session_id: str, *, runs: int = 1
    ) -> dict[str, Any]:
        if not 1 <= runs <= 5:
            raise ValueError("runs must be between 1 and 5")
        session = self._session(session_id)
        if not session.get("current_selection"):
            raise ValueError("a human must select or enter a target first")
        selection = TargetSelection.model_validate(
            read_json(self._session_dir(session_id) / session["current_selection"])
        )
        locked_target_id, identity_description, predicate_material = (
            self._locked_query_material(session_id, session, selection)
        )
        moment_search_description = identity_description
        if predicate_material is not None:
            moment_search_description += "\n" + predicate_material
        media = self._media(session)
        batch_dir = self._session_dir(session_id) / "moments" / f"batch-{uuid.uuid4().hex[:8]}"
        client = self.client_factory()
        started = monotonic()
        maps: list[DirectMomentMap] = []
        try:
            uploaded, reused, _ = self._ensure_upload(client, session)
            for index in range(1, runs + 1):
                run_dir = batch_dir / f"run-{index:02d}"
                moment_map = client.analyze_direct_moments(
                    media=media,
                    uploaded=uploaded,
                    prompt_template=_prompt("target_moments_mmss_zh-TW.txt"),
                    run_id=f"blind-moments-{index:02d}-{uuid.uuid4().hex[:8]}",
                    run_dir=run_dir,
                    locked_target_id=locked_target_id,
                    locked_target_description=moment_search_description,
                )
                maps.append(moment_map)
        finally:
            client.close()
        session["stage"] = "moments_ready"
        session["current_moment_map"] = str(
            (batch_dir / "run-01" / "direct_moments.json").relative_to(self._session_dir(session_id))
        )
        self._save_session(session)
        pricing = summarize_usage_and_list_price(batch_dir)
        result = {
            "session_id": session_id,
            "file_api_object_reused": reused,
            "elapsed_seconds": round(monotonic() - started, 6),
            "pricing": pricing,
            "runs": [moment_map.model_dump(mode="json") for moment_map in maps],
        }
        write_json(batch_dir / "summary.json", result)
        return result

    def ground_moment(
        self,
        session_id: str,
        *,
        moment_id: str,
        mode: Literal["exact_frame", "direct_video"] = "exact_frame",
    ) -> dict[str, Any]:
        session = self._session(session_id)
        if not session.get("current_moment_map") or not session.get("current_selection"):
            raise ValueError("generate target-locked moments before Grounding")
        selection = TargetSelection.model_validate(
            read_json(self._session_dir(session_id) / session["current_selection"])
        )
        locked_target_id, identity_description, _predicate_material = (
            self._locked_query_material(session_id, session, selection)
        )
        if selection.query_status == "query_locked":
            lock_path = session.get("current_query_lock")
            if not lock_path:
                raise ValueError("query_locked selection is missing its lock artifact")
            lock = EvidenceQueryLockV2.model_validate(
                read_json(self._session_dir(session_id) / lock_path)
            )
            if lock.predicate is not None:
                raise ValueError(
                    "Blind Review Web has only produced a coarse MM:SS candidate; "
                    "formal QueryLock predicate refinement is required before Grounding. "
                    "Export the session and run refine-query-predicate, or approve an "
                    "identity-only QueryLock for geometry review."
                )
        moment_map = DirectMomentMap.model_validate(
            read_json(self._session_dir(session_id) / session["current_moment_map"])
        )
        moment = next((item for item in moment_map.moments if item.moment_id == moment_id), None)
        if moment is None:
            raise ValueError(f"unknown moment_id {moment_id}")
        grounding_context = (
            "Exact-frame identity Grounding only. Use the supplied image pixels and "
            "locked identity cues; do not infer the event predicate or target position "
            "from the requested time."
            if selection.query_status == "query_locked"
            else f"{moment.label}；{moment.observable_evidence}"
        )
        review_id = f"review-{uuid.uuid4().hex[:12]}"
        review_dir = self._session_dir(session_id) / "reviews" / review_id
        requested_ms = _mmss_to_ms(moment.timestamp_mmss)
        frame = extract_frame(Path(session["source_path"]), requested_ms, review_dir / "frame.png")
        write_json(review_dir / "frame.json", frame)
        media = self._media(session)
        client = self.client_factory()
        started = monotonic()
        try:
            if mode == "exact_frame":
                proposal: GroundingProposal | DirectVideoGroundingProposal = client.ground_frame(
                    media=media,
                    frame=frame,
                    event_id=moment.moment_id,
                    event_description=grounding_context,
                    entity_id=locked_target_id,
                    target_description=identity_description,
                    prompt_template=_prompt("grounding_native_yxyx_zh-TW.txt"),
                    run_id=f"blind-grounding-{uuid.uuid4().hex[:8]}",
                    output_dir=review_dir / "grounding",
                )
                projection = proposal
                proposal_path = "grounding/grounding.json"
                proposal_type = "exact_frame_grounding"
                grounding_method = "exact_frame_image"
                bbox_reference_frame = "exact_ffmpeg_frame"
            else:
                uploaded, _, _ = self._ensure_upload(client, session)
                proposal = client.ground_video_at_moment(
                    media=media,
                    uploaded=uploaded,
                    requested_timestamp_mmss=moment.timestamp_mmss,
                    event_id=moment.moment_id,
                    event_description=grounding_context,
                    entity_id=locked_target_id,
                    target_description=identity_description,
                    prompt_template=_prompt("direct_video_grounding_native_yxyx_zh-TW.txt"),
                    run_id=f"blind-direct-video-{uuid.uuid4().hex[:8]}",
                    output_dir=review_dir / "direct-video-grounding",
                )
                projection = GroundingProposal(
                    asset_id=media.asset_id,
                    event_id=moment.moment_id,
                    entity_id=locked_target_id,
                    frame_pts=frame.frame_pts,
                    frame_time_ms=frame.frame_time_ms,
                    frame_hash=frame.frame_hash,
                    source_width=frame.width,
                    source_height=frame.height,
                    visible=proposal.visible,
                    occlusion=proposal.occlusion,
                    visibility_reason=(
                        "Projection only; model reference is an unknown video sample. "
                        + proposal.visibility_reason
                    ),
                    candidates=proposal.candidates,
                    model_provenance=proposal.model_provenance,
                )
                write_json(review_dir / "direct-video-projection.json", projection)
                proposal_path = "direct-video-grounding/direct_video_grounding.json"
                proposal_type = "direct_video_grounding"
                grounding_method = "direct_video_unknown_sample"
                bbox_reference_frame = "unknown_gemini_video_sample"
        finally:
            client.close()
        blind_path = draw_blind_review_overlay(frame.path, projection, review_dir / "blind.png")
        draw_grounding_overlay(frame.path, projection, review_dir / "revealed.png")
        manifest = {
            "review_id": review_id,
            "status": "pending_human_review",
            "created_at": utc_now(),
            "target_id": locked_target_id,
            "target_description": identity_description,
            "grounding_method": grounding_method,
            "bbox_reference_frame": bbox_reference_frame,
            "proposal_type": proposal_type,
            "moment_id": moment.moment_id,
            "requested_timestamp_mmss": moment.timestamp_mmss,
            "requested_time_ms": requested_ms,
            "frame_pts": frame.frame_pts,
            "frame_time_ms": frame.frame_time_ms,
            "frame_hash": frame.frame_hash,
            "blind_image": str(blind_path.relative_to(review_dir)),
            "proposal_path": proposal_path,
            "annotation_path": None,
            "model_details_revealed": False,
            "grounding_seconds": round(monotonic() - started, 6),
        }
        write_json(review_dir / "review.json", manifest)
        session["stage"] = "blind_review_pending"
        session["reviews"].append(review_id)
        self._save_session(session)
        return {
            "session_id": session_id,
            "review_id": review_id,
            "status": manifest["status"],
            "target_id": locked_target_id,
            "target_description": identity_description,
            "grounding_method": grounding_method,
            "bbox_reference_frame": bbox_reference_frame,
            "requested_timestamp_mmss": moment.timestamp_mmss,
            "requested_time_ms": requested_ms,
            "frame_pts": frame.frame_pts,
            "frame_time_ms": frame.frame_time_ms,
            "frame_hash": frame.frame_hash,
            "source_width": frame.width,
            "source_height": frame.height,
            "blind_image_url": f"/api/sessions/{session_id}/reviews/{review_id}/blind-image",
        }

    def submit_review(
        self,
        session_id: str,
        review_id: str,
        *,
        verdict: ReviewVerdict,
        notes: str = "",
        reviewer_name: str | None = None,
        corrected_box_2d: tuple[int, int, int, int] | None = None,
    ) -> dict[str, Any]:
        session = self._session(session_id)
        review_dir = self._session_dir(session_id) / "reviews" / review_id
        manifest_path = review_dir / "review.json"
        if not manifest_path.exists() or review_id not in session["reviews"]:
            raise FileNotFoundError(f"unknown review {review_id}")
        manifest = read_json(manifest_path)
        if manifest["status"] != "pending_human_review":
            raise ValueError("this review has already been submitted")
        annotation = HumanReviewAnnotation(
            annotation_id=f"human-{uuid.uuid4().hex}",
            session_id=session_id,
            review_id=review_id,
            asset_id=session["asset_id"],
            reviewer_type="human",
            reviewer_name=reviewer_name.strip() if reviewer_name and reviewer_name.strip() else None,
            target_id=manifest["target_id"],
            target_description=manifest["target_description"],
            grounding_method=manifest["grounding_method"],
            bbox_reference_frame=manifest["bbox_reference_frame"],
            requested_timestamp_mmss=manifest["requested_timestamp_mmss"],
            requested_time_ms=manifest["requested_time_ms"],
            frame_pts=manifest["frame_pts"],
            frame_time_ms=manifest["frame_time_ms"],
            frame_hash=manifest["frame_hash"],
            verdict=verdict,
            notes=notes,
            corrected_box_2d=corrected_box_2d,
            model_details_revealed_before_annotation=False,
            annotated_at=utc_now(),
        )
        write_json(review_dir / "human_annotation.json", annotation)
        manifest["status"] = "reviewed"
        manifest["annotation_path"] = "human_annotation.json"
        manifest["model_details_revealed"] = True
        manifest["reviewed_at"] = annotation.annotated_at
        write_json(manifest_path, manifest)
        if all(
            read_json(self._session_dir(session_id) / "reviews" / item / "review.json")["status"]
            == "reviewed"
            for item in session["reviews"]
        ):
            session["stage"] = "reviewed"
        self._save_session(session)
        return self.reveal_review(session_id, review_id)

    def reveal_review(self, session_id: str, review_id: str) -> dict[str, Any]:
        review_dir = self._session_dir(session_id) / "reviews" / review_id
        manifest_path = review_dir / "review.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"unknown review {review_id}")
        manifest = read_json(manifest_path)
        if manifest["status"] != "reviewed" or not manifest["model_details_revealed"]:
            raise PermissionError("submit a human verdict before revealing model details")
        annotation = HumanReviewAnnotation.model_validate(
            read_json(review_dir / "human_annotation.json")
        )
        proposal_payload = read_json(review_dir / manifest["proposal_path"])
        if manifest["proposal_type"] == "direct_video_grounding":
            proposal = DirectVideoGroundingProposal.model_validate(proposal_payload)
        else:
            proposal = GroundingProposal.model_validate(proposal_payload)
        result = {
            "annotation": annotation.model_dump(mode="json"),
            "proposal": proposal.model_dump(mode="json"),
            "revealed_image_url": (
                f"/api/sessions/{session_id}/reviews/{review_id}/revealed-image"
            ),
        }
        comparison = self._review_method_comparison(session_id, review_id)
        if comparison is not None:
            result["method_comparison"] = comparison
        return result

    def _review_method_comparison(
        self, session_id: str, review_id: str
    ) -> dict[str, Any] | None:
        session = self._session(session_id)
        current_dir = self._session_dir(session_id) / "reviews" / review_id
        current = read_json(current_dir / "review.json")
        if current["status"] != "reviewed":
            return None
        for sibling_id in session["reviews"]:
            if sibling_id == review_id:
                continue
            sibling_dir = self._session_dir(session_id) / "reviews" / sibling_id
            sibling = read_json(sibling_dir / "review.json")
            if (
                sibling["status"] != "reviewed"
                or sibling["moment_id"] != current["moment_id"]
                or sibling["target_id"] != current["target_id"]
                or sibling["grounding_method"] == current["grounding_method"]
            ):
                continue
            proposals: dict[str, list[Any]] = {}
            for manifest, directory in ((current, current_dir), (sibling, sibling_dir)):
                payload = read_json(directory / manifest["proposal_path"])
                proposals[manifest["grounding_method"]] = payload.get("candidates") or []
            exact = proposals.get("exact_frame_image", [])
            direct = proposals.get("direct_video_unknown_sample", [])
            if not exact or not direct:
                return {
                    "comparable": False,
                    "reason": "one method returned no visible candidate",
                }
            exact_box = exact[0]["box_2d"]
            direct_box = direct[0]["box_2d"]
            return {
                "comparable": True,
                "warning": "Direct-video bbox reference frame is unknown; this is diagnostic only.",
                "exact_frame_box_2d": exact_box,
                "direct_video_box_2d": direct_box,
                "bbox_iou": round(box_iou(exact_box, direct_box), 6),
                "bbox_center_distance": round(center_distance(exact_box, direct_box), 6),
            }
        return None

    def session_view(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        media = self._media(session)
        view: dict[str, Any] = {
            key: value
            for key, value in session.items()
            if key
            not in {
                "source_path",
                "current_candidate_map",
                "current_selection",
                "current_query_proposal",
                "current_query_proposal_manifest",
                "current_query_lock",
                "current_query_lock_manifest",
                "current_moment_map",
            }
        }
        view["media"] = media.model_dump(mode="json")
        view["video_url"] = f"/api/sessions/{session_id}/video"
        if session.get("current_candidate_map"):
            view["candidate_map"] = read_json(
                self._session_dir(session_id) / session["current_candidate_map"]
            )
        if session.get("current_selection"):
            view["selection"] = read_json(
                self._session_dir(session_id) / session["current_selection"]
            )
        if session.get("current_query_proposal"):
            view["query_proposal"] = read_json(
                self._session_dir(session_id) / session["current_query_proposal"]
            )
        if session.get("current_query_proposal_manifest"):
            view["query_proposal_manifest"] = read_json(
                self._session_dir(session_id)
                / session["current_query_proposal_manifest"]
            )
        if session.get("current_query_lock"):
            view["query_lock"] = read_json(
                self._session_dir(session_id) / session["current_query_lock"]
            )
        if session.get("current_query_lock_manifest"):
            view["query_lock_manifest"] = read_json(
                self._session_dir(session_id)
                / session["current_query_lock_manifest"]
            )
        if session.get("current_moment_map"):
            view["moment_map"] = read_json(
                self._session_dir(session_id) / session["current_moment_map"]
            )
        view["review_states"] = [
            read_json(self._session_dir(session_id) / "reviews" / review_id / "review.json")
            for review_id in session["reviews"]
        ]
        cache_path = self._session_dir(session_id) / "file_api_cache.json"
        if cache_path.exists():
            view["file_api_cache"] = read_json(cache_path)
        return view

    def export_session(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        annotations: list[dict[str, Any]] = []
        revealed_proposals: list[dict[str, Any]] = []
        pending_reviews: list[str] = []
        for review_id in session["reviews"]:
            review_dir = self._session_dir(session_id) / "reviews" / review_id
            manifest = read_json(review_dir / "review.json")
            if manifest["status"] != "reviewed":
                pending_reviews.append(review_id)
                continue
            annotations.append(read_json(review_dir / "human_annotation.json"))
            revealed_proposals.append(read_json(review_dir / manifest["proposal_path"]))
        return {
            "exported_at": utc_now(),
            "session": self.session_view(session_id),
            "human_annotations": annotations,
            "revealed_grounding_proposals": revealed_proposals,
            "pending_review_ids": pending_reviews,
        }
