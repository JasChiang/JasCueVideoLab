from __future__ import annotations

import pytest
from pydantic import ValidationError

from montagewright.measure.models import (
    EvidenceAnchor,
    EvidenceApprovalSource,
    EvidenceAspectConstraint,
    EvidenceClaimSource,
    EvidenceFramingObligations,
    EvidenceIdentityContract,
    EvidencePredicateContract,
    EvidencePredicatePhases,
    EvidenceQueryApprovalProvenance,
    EvidenceQueryLock,
    EvidenceQueryProposal,
    EvidenceQueryProvenance,
    EvidenceTargetIdentity,
    PredicateRequiredAt,
    TargetIdentityScope,
    approve_evidence_query_proposal,
)


def _identity() -> EvidenceIdentityContract:
    return EvidenceIdentityContract(
        targets=(
            EvidenceTargetIdentity(
                target_id="subject.primary",
                target_description="the reviewer-selected foreground subject",
                identity_cues=("distinctive outline", "persistent surface pattern"),
                context_cues=("near the active participant at the seed moment",),
                positive_anchors=(
                    EvidenceAnchor(frame_id="RF000120", crop_sha256="a" * 64),
                ),
                stable_exclusions=("background depiction", "reflected image"),
                negative_anchors=(
                    EvidenceAnchor(frame_id="RF000120", crop_sha256="b" * 64),
                ),
            ),
            EvidenceTargetIdentity(
                target_id="subject.primary.detail",
                target_description="a visible detail belonging to the selected subject",
                scope=TargetIdentityScope.VISIBLE_REGION,
                parent_target_id="subject.primary",
                identity_cues=("bounded visible detail",),
            ),
        )
    )


def _predicate() -> EvidencePredicateContract:
    return EvidencePredicateContract(
        predicate_id="predicate.transition",
        statement="the selected subject changes into the requested observable state",
        participant_target_ids=("subject.primary", "subject.primary.detail"),
        required_at=PredicateRequiredAt.TRANSITION,
        phases=EvidencePredicatePhases(
            precondition="the requested state is not yet visible",
            apex="the observable change is in progress",
            postcondition="the requested state is directly visible",
        ),
        required_evidence=("the same selected instance is visible across the change",),
        disqualifying_conditions=("only a depiction of the instance is visible",),
    )


def _framing() -> EvidenceFramingObligations:
    return EvidenceFramingObligations(
        required_target_ids=("subject.primary",),
        preferred_target_ids=("subject.primary.detail",),
        overlay_keepout_target_ids=("subject.primary.detail",),
        framing_intent="Keep the selected instance complete and preserve its visible detail.",
        editing_uses=("demonstration", "portrait_reframe"),
        aspect_constraints=(
            EvidenceAspectConstraint(
                aspect_ratio="9:16",
                required_target_ids=("subject.primary",),
                constraint="the selected instance must remain recognizable",
            ),
        ),
    )


def _proposal() -> EvidenceQueryProposal:
    return EvidenceQueryProposal(
        proposal_id="proposal:001",
        revision=1,
        editorial_goal="Show the selected observable transition without changing identity.",
        identity=_identity(),
        predicate=_predicate(),
        framing=_framing(),
        claim_source=EvidenceClaimSource.MODEL_PROPOSAL,
        provenance=EvidenceQueryProvenance(
            created_at="2026-07-22T00:00:00Z",
            created_by="planner-run:001",
            source_reference="clip-card-library:001",
        ),
    )


def _human_approval() -> EvidenceQueryApprovalProvenance:
    return EvidenceQueryApprovalProvenance(
        approved_at="2026-07-22T00:01:00Z",
        approved_by="reviewer:001",
        approval_source=EvidenceApprovalSource.HUMAN_REVIEW,
        source_reference="review:001",
    )


def test_keeps_identity_predicate_and_framing_as_separate_contracts() -> None:
    proposal = _proposal()
    assert proposal.identity.targets[0].context_cues
    assert proposal.predicate is not None
    assert proposal.predicate.required_at is PredicateRequiredAt.TRANSITION
    assert proposal.framing.overlay_keepout_target_ids == (
        "subject.primary.detail",
    )


def test_hashes_are_stable_and_component_scoped() -> None:
    proposal = _proposal()
    reparsed = EvidenceQueryProposal.model_validate_json(proposal.model_dump_json())
    assert reparsed.component_hashes() == proposal.component_hashes()
    assert reparsed.composite_sha256() == proposal.composite_sha256()

    changed = proposal.model_copy(
        update={
            "framing": proposal.framing.model_copy(
                update={"framing_intent": "Prefer a wider observable context."}
            )
        }
    )
    assert (
        changed.component_hashes()["identity_sha256"]
        == proposal.component_hashes()["identity_sha256"]
    )
    assert (
        changed.component_hashes()["predicate_sha256"]
        == proposal.component_hashes()["predicate_sha256"]
    )
    assert (
        changed.component_hashes()["framing_sha256"]
        != proposal.component_hashes()["framing_sha256"]
    )
    assert changed.composite_sha256() != proposal.composite_sha256()


def test_approval_creates_frozen_lock_without_relabeling_claim_source() -> None:
    proposal = _proposal()
    lock = approve_evidence_query_proposal(
        proposal,
        query_id="query:001",
        approval=_human_approval(),
    )
    assert lock.claim_source is EvidenceClaimSource.MODEL_PROPOSAL
    assert lock.approval.approval_source is EvidenceApprovalSource.HUMAN_REVIEW
    assert lock.composite_sha256() == proposal.composite_sha256()
    assert len(lock.definition_sha256()) == 64
    with pytest.raises(ValidationError, match="frozen"):
        lock.revision = 2
    with pytest.raises(ValidationError, match="frozen"):
        lock.identity.targets[0].target_description = "different target"
    with pytest.raises(ValidationError, match="frozen"):
        lock.provenance.created_by = "different creator"


def test_auto_policy_approval_requires_named_policy() -> None:
    with pytest.raises(ValidationError, match="requires policy_reference"):
        EvidenceQueryApprovalProvenance(
            approved_at="2026-07-22T00:01:00Z",
            approved_by="automation",
            approval_source=EvidenceApprovalSource.AUTO_POLICY,
        )
    approval = EvidenceQueryApprovalProvenance(
        approved_at="2026-07-22T00:01:00Z",
        approved_by="automation",
        approval_source=EvidenceApprovalSource.AUTO_POLICY,
        policy_reference="policy:bounded-auto-v1",
    )
    assert approval.policy_reference == "policy:bounded-auto-v1"


def test_subparts_require_a_known_parent_and_parent_graph_is_acyclic() -> None:
    with pytest.raises(ValidationError, match="require parent_target_id"):
        EvidenceTargetIdentity(
            target_id="detail",
            target_description="selected detail",
            scope=TargetIdentityScope.SUBPART,
            identity_cues=("observable edge",),
        )
    payload = _proposal().model_dump(mode="json")
    payload["identity"]["targets"][1]["parent_target_id"] = "unknown"
    with pytest.raises(ValidationError, match="unknown targets"):
        EvidenceQueryProposal.model_validate(payload)


def test_positive_and_negative_anchors_cannot_be_the_same_crop() -> None:
    anchor = {"frame_id": "RF000001", "crop_sha256": "c" * 64}
    with pytest.raises(ValidationError, match="must not overlap"):
        EvidenceTargetIdentity(
            target_id="subject",
            target_description="selected subject",
            positive_anchors=[anchor],
            negative_anchors=[anchor],
        )

    with pytest.raises(ValidationError, match="same crop bytes"):
        EvidenceTargetIdentity(
            target_id="subject",
            target_description="selected subject",
            positive_anchors=[anchor],
            negative_anchors=[
                {"frame_id": "RF000002", "crop_sha256": "c" * 64}
            ],
        )


def test_transition_predicate_requires_all_three_observable_phases() -> None:
    with pytest.raises(ValidationError, match="require pre/apex/post phases"):
        EvidencePredicateContract(
            predicate_id="predicate",
            statement="an observable change occurs",
            participant_target_ids=("subject",),
            required_at=PredicateRequiredAt.TRANSITION,
        )


def test_rejects_unknown_predicate_and_framing_targets() -> None:
    payload = _proposal().model_dump(mode="json")
    payload["predicate"]["participant_target_ids"] = ["unknown"]
    with pytest.raises(ValidationError, match="unknown participant targets"):
        EvidenceQueryProposal.model_validate(payload)

    payload = _proposal().model_dump(mode="json")
    payload["framing"]["required_target_ids"] = ["unknown"]
    with pytest.raises(ValidationError, match="references unknown targets"):
        EvidenceQueryProposal.model_validate(payload)


def test_primary_framing_roles_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="must be disjoint"):
        EvidenceFramingObligations(
            required_target_ids=("subject",),
            sacrificable_target_ids=("subject",),
            framing_intent="Preserve the evidence hierarchy.",
        )


def test_models_forbid_unversioned_extra_fields() -> None:
    payload = _proposal().model_dump(mode="json")
    payload["approved_by_model"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvidenceQueryProposal.model_validate(payload)


def test_only_grounding_query_v1_wire_formats_are_accepted() -> None:
    proposal = _proposal()
    assert proposal.contract_version == "grounding-query-proposal-v1"

    lock = approve_evidence_query_proposal(
        proposal,
        query_id="query:001",
        approval=_human_approval(),
    )
    assert lock.contract_version == "grounding-query-lock-v1"

    proposal_payload = proposal.model_dump(mode="json")
    proposal_payload["contract_version"] = "evidence-query-proposal-v2"
    with pytest.raises(ValidationError, match="grounding-query-proposal-v1"):
        EvidenceQueryProposal.model_validate(proposal_payload)

    lock_payload = lock.model_dump(mode="json")
    lock_payload["contract_version"] = "evidence-query-lock-v2"
    with pytest.raises(ValidationError, match="grounding-query-lock-v1"):
        EvidenceQueryLock.model_validate(lock_payload)
