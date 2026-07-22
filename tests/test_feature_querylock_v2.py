from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import jascue_video_lab.feature_cut as feature_cut
from jascue_video_lab.feature_cut import (
    _bind_evidence_query_lock_v2_lineage,
    _ground_tracking_seed,
    _load_or_create_feature_candidate_query_lock_v2,
    _query_lock_v2_runtime_geometry_lineage,
    approve_feature_query_proposal_v2_for_auto,
    evidence_query_lock_v2_lineage,
    feature_vertical_candidate_to_query_proposal_v2,
)
from jascue_video_lab.models import (
    EvidenceApprovalSource,
    EvidencePredicateContractV2,
    EvidenceQueryApprovalProvenance,
    FeatureVerticalCandidate,
    FramingRegionIntent,
    PredicateRequiredAt,
)


def _candidate() -> FeatureVerticalCandidate:
    return FeatureVerticalCandidate(
        candidate_id="candidate-1",
        rank=1,
        source_asset_id="sha256:" + "a" * 64,
        event_id="event-1",
        frame_id="RF000123",
        observed_visual_evidence=(
            "The selected instance is visible while an observable state changes."
        ),
        selection_reason="The state change supports the edit beat.",
        strategy="tracked_crop",
        regions=[
            FramingRegionIntent(
                region_id="main",
                entity_id="target.main",
                target_description="the selected foreground instance",
                role="required",
                observable_relations=["beside the selected context region"],
                exclusions=["background depiction"],
            ),
            FramingRegionIntent(
                region_id="context",
                entity_id="target.context",
                target_description="the selected contextual region",
                role="preferred",
                minimum_visible_fraction=0.75,
            ),
            FramingRegionIntent(
                region_id="overlay-zone",
                entity_id="target.overlay",
                target_description="the selected overlay-protected region",
                role="avoid_overlay",
            ),
        ],
        confidence=0.9,
    )


def _proposal(*, predicate: EvidencePredicateContractV2 | None = None):
    return feature_vertical_candidate_to_query_proposal_v2(
        _candidate(),
        editorial_goal="Show the selected evidence while preserving identity.",
        created_at="2026-07-22T00:00:00Z",
        created_by="full-auto-planner:test",
        eligible_predicate=predicate,
    )


def _auto_approval() -> EvidenceQueryApprovalProvenance:
    return EvidenceQueryApprovalProvenance(
        approved_at="2026-07-22T00:00:01Z",
        approved_by="full-auto-runtime:test",
        approval_source=EvidenceApprovalSource.AUTO_POLICY,
        policy_reference="policy:querylock-v2-bounded-auto-v1",
    )


def _auto_lock():
    return approve_feature_query_proposal_v2_for_auto(
        _proposal(),
        query_id="query:feature:candidate-1",
        approval=_auto_approval(),
    )


def test_candidate_adapter_separates_identity_predicate_and_framing() -> None:
    candidate = _candidate()
    before = candidate.model_dump(mode="json")

    proposal = feature_vertical_candidate_to_query_proposal_v2(
        candidate,
        editorial_goal="Show the selected evidence while preserving identity.",
        created_at="2026-07-22T00:00:00Z",
        created_by="full-auto-planner:test",
    )

    assert candidate.model_dump(mode="json") == before
    assert proposal.predicate is None
    assert [target.target_id for target in proposal.identity.targets] == [
        "target.main",
        "target.context",
        "target.overlay",
    ]
    assert proposal.identity.targets[0].identity_cues == (
        "the selected foreground instance",
    )
    assert proposal.identity.targets[0].context_cues == (
        "beside the selected context region",
    )
    assert proposal.identity.targets[0].stable_exclusions == (
        "background depiction",
    )
    assert proposal.framing.required_target_ids == ("target.main",)
    assert proposal.framing.preferred_target_ids == ("target.context",)
    assert proposal.framing.overlay_keepout_target_ids == ("target.overlay",)
    assert proposal.framing.aspect_constraints[0].aspect_ratio == "9:16"
    assert proposal.framing.aspect_constraints[0].required_target_clipping_policy == (
        "forbid"
    )
    visibility = {
        item.target_id: item.minimum_visible_fraction
        for item in proposal.framing.aspect_constraints[0].target_visibility_constraints
    }
    assert visibility == {"target.main": 1.0, "target.context": 0.75}
    assert "state change" not in proposal.identity.targets[0].target_description


def test_framing_hash_captures_visibility_floor_and_crop_policy() -> None:
    baseline = _proposal()
    higher_floor_candidate = _candidate().model_copy(
        update={
            "regions": [
                region.model_copy(update={"minimum_visible_fraction": 0.99})
                if region.region_id == "context"
                else region
                for region in _candidate().regions
            ]
        }
    )
    higher_floor = feature_vertical_candidate_to_query_proposal_v2(
        higher_floor_candidate,
        editorial_goal="Show the selected evidence while preserving identity.",
        created_at="2026-07-22T00:00:00Z",
        created_by="full-auto-planner:test",
    )
    controlled = feature_vertical_candidate_to_query_proposal_v2(
        _candidate().model_copy(update={"crop_mode": "primary_center"}),
        editorial_goal="Show the selected evidence while preserving identity.",
        created_at="2026-07-22T00:00:00Z",
        created_by="full-auto-planner:test",
    )

    baseline_hash = baseline.component_hashes()["framing_sha256"]
    assert higher_floor.component_hashes()["framing_sha256"] != baseline_hash
    assert controlled.component_hashes()["framing_sha256"] != baseline_hash


def test_candidate_adapter_is_deterministic_and_only_accepts_explicit_predicate() -> None:
    first = _proposal()
    second = _proposal()
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.composite_sha256() == second.composite_sha256()

    predicate = EvidencePredicateContractV2(
        predicate_id="predicate:candidate-state",
        statement="the selected instance shows the explicitly requested state",
        participant_target_ids=("target.main",),
        required_at=PredicateRequiredAt.CANDIDATE,
        required_evidence=("the requested state is directly visible",),
    )
    with_predicate = _proposal(predicate=predicate)
    assert with_predicate.predicate == predicate

    unknown_target = predicate.model_copy(
        update={"participant_target_ids": ("target.unknown",)}
    )
    with pytest.raises(ValidationError, match="unknown participant targets"):
        _proposal(predicate=unknown_target)


def test_fit_only_candidate_without_identity_does_not_create_a_query() -> None:
    candidate = FeatureVerticalCandidate(
        candidate_id="fit-only",
        rank=1,
        source_asset_id="sha256:" + "b" * 64,
        event_id="event-fit",
        frame_id="RF000124",
        observed_visual_evidence="The source frame remains usable without tracking.",
        selection_reason="A fitted composition preserves the complete frame.",
        strategy="fit_with_background",
        confidence=0.8,
    )
    with pytest.raises(ValueError, match="does not require an evidence query"):
        feature_vertical_candidate_to_query_proposal_v2(
            candidate,
            editorial_goal="Preserve the complete source frame.",
            created_at="2026-07-22T00:00:00Z",
            created_by="full-auto-planner:test",
        )


def test_full_auto_lock_requires_explicit_named_auto_policy() -> None:
    human_approval = EvidenceQueryApprovalProvenance(
        approved_at="2026-07-22T00:00:01Z",
        approved_by="reviewer:test",
        approval_source=EvidenceApprovalSource.HUMAN_REVIEW,
        source_reference="review:test",
    )
    with pytest.raises(ValueError, match="requires auto_policy"):
        approve_feature_query_proposal_v2_for_auto(
            _proposal(),
            query_id="query:feature:candidate-1",
            approval=human_approval,
        )

    lock = _auto_lock()
    lineage = evidence_query_lock_v2_lineage(
        lock,
        target_id="target.main",
        target_description="the selected foreground instance",
    )
    assert lineage["identity_sha256"] == lock.component_hashes()["identity_sha256"]
    assert lineage["predicate_sha256"] == lock.component_hashes()["predicate_sha256"]
    assert lineage["framing_sha256"] == lock.component_hashes()["framing_sha256"]
    assert lineage["composite_sha256"] == lock.composite_sha256()
    assert lineage["definition_sha256"] == lock.definition_sha256()
    assert lineage["approval"]["approval_source"] == "auto_policy"
    assert lineage["approval"]["policy_reference"] == (
        "policy:querylock-v2-bounded-auto-v1"
    )


def test_lineage_binding_does_not_mutate_legacy_payload() -> None:
    payload = {"contract_version": "bbox-seed-v2-exact-pts", "seed": [1, 2, 3, 4]}
    legacy = _bind_evidence_query_lock_v2_lineage(
        payload,
        lock=None,
        target_id="target.main",
        target_description="the selected foreground instance",
    )
    bound = _bind_evidence_query_lock_v2_lineage(
        payload,
        lock=_auto_lock(),
        target_id="target.main",
        target_description="the selected foreground instance",
    )

    assert payload == {"contract_version": "bbox-seed-v2-exact-pts", "seed": [1, 2, 3, 4]}
    assert legacy == payload
    assert "evidence_query_v2" not in legacy
    assert bound["evidence_query_v2"]["definition_sha256"] == (
        _auto_lock().definition_sha256()
    )


def test_grounding_request_key_binds_query_hashes_without_a_live_model_call(
    tmp_path, monkeypatch
) -> None:
    exact_frame = SimpleNamespace(
        frame_hash="c" * 64,
        frame_pts=9000,
        frame_time_ms=1000,
        width=1920,
        height=1080,
    )
    monkeypatch.setattr(feature_cut, "extract_frame", lambda *_args, **_kwargs: exact_frame)
    monkeypatch.setattr(
        feature_cut,
        "probe_video",
        lambda *_args, **_kwargs: SimpleNamespace(asset_id="sha256:" + "d" * 64),
    )
    frame = SimpleNamespace(requested_time_ms=1000, frame_id="RF000123")
    clip = SimpleNamespace(path=str(tmp_path / "source.mp4"))

    common = dict(
        client=object(),
        clip=clip,
        frame=frame,
        start_ms=0,
        end_ms=2000,
        feature_id="feature-1",
        event_description="Show the selected evidence.",
        entity_id="target.main",
        target_description="the selected foreground instance",
        grounding_prompt="Ground only the named target.",
        run_id="test-run",
        model_request_block_reason="test blocks external model calls",
    )
    legacy_dir = tmp_path / "legacy"
    with pytest.raises(RuntimeError, match="circuit breaker"):
        _ground_tracking_seed(output_dir=legacy_dir, **common)
    bound_dir = tmp_path / "bound"
    with pytest.raises(RuntimeError, match="circuit breaker"):
        _ground_tracking_seed(
            output_dir=bound_dir,
            query_lock_v2=_auto_lock(),
            **common,
        )

    legacy_key = json.loads(next(legacy_dir.glob("grounding/*/request-key.json")).read_text())
    bound_key = json.loads(next(bound_dir.glob("grounding/*/request-key.json")).read_text())
    assert "evidence_query_v2" not in legacy_key
    assert bound_key["evidence_query_v2_identity"]["target_id"] == "target.main"
    assert bound_key["evidence_query_v2_identity"]["identity_sha256"] == (
        _auto_lock().component_hashes()["identity_sha256"]
    )
    bound_lineage = json.loads(
        next(bound_dir.glob("grounding/*/query-lineage-*.json")).read_text()
    )
    assert bound_lineage["definition_sha256"] == _auto_lock().definition_sha256()
    assert bound_key["request_fingerprint"] != legacy_key["request_fingerprint"]


def test_auto_candidate_lock_is_persisted_and_reused(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    first = _load_or_create_feature_candidate_query_lock_v2(
        _candidate(),
        feature_id="feature-1",
        output_dir=output_dir,
    )
    second = _load_or_create_feature_candidate_query_lock_v2(
        _candidate(),
        feature_id="feature-1",
        output_dir=output_dir,
    )

    assert first == second
    assert first.approval.approval_source == EvidenceApprovalSource.AUTO_POLICY
    assert first.approval.policy_reference == (
        "policy:full-auto-topk-lazy-geometry-querylock-v2:v1"
    )
    query_parent = output_dir / "query-lock-v2"
    variants = list(query_parent.glob("variant-*"))
    assert len(variants) == 1
    assert (variants[0] / "proposal.json").exists()
    assert (variants[0] / "lock.json").exists()
    assert (variants[0] / "manifest.json").exists()
    other_feature = _load_or_create_feature_candidate_query_lock_v2(
        _candidate(),
        feature_id="feature-2",
        output_dir=output_dir,
    )
    assert other_feature.provenance.source_reference != first.provenance.source_reference
    assert len(list(query_parent.glob("variant-*"))) == 2


def test_framing_only_change_reuses_identity_grounding_fingerprint(
    tmp_path, monkeypatch
) -> None:
    proposal = _proposal()
    changed_proposal = proposal.model_copy(
        update={
            "framing": proposal.framing.model_copy(
                update={"framing_intent": "Use a different downstream layout."}
            )
        }
    )
    first_lock = approve_feature_query_proposal_v2_for_auto(
        proposal,
        query_id="query:feature:framing-a",
        approval=_auto_approval(),
    )
    second_lock = approve_feature_query_proposal_v2_for_auto(
        changed_proposal,
        query_id="query:feature:framing-b",
        approval=_auto_approval(),
    )
    assert first_lock.component_hashes()["identity_sha256"] == (
        second_lock.component_hashes()["identity_sha256"]
    )
    assert first_lock.component_hashes()["framing_sha256"] != (
        second_lock.component_hashes()["framing_sha256"]
    )
    exact_frame = SimpleNamespace(
        frame_hash="f" * 64,
        frame_pts=9000,
        frame_time_ms=1000,
        width=1920,
        height=1080,
    )
    monkeypatch.setattr(feature_cut, "extract_frame", lambda *_args, **_kwargs: exact_frame)
    monkeypatch.setattr(
        feature_cut,
        "probe_video",
        lambda *_args, **_kwargs: SimpleNamespace(asset_id="sha256:" + "d" * 64),
    )
    common = dict(
        client=object(),
        clip=SimpleNamespace(path=str(tmp_path / "source.mp4")),
        frame=SimpleNamespace(requested_time_ms=1000, frame_id="RF000123"),
        start_ms=0,
        end_ms=2000,
        feature_id="feature-1",
        event_description="Planner prose must not enter identity Grounding.",
        entity_id="target.main",
        target_description="the selected foreground instance",
        grounding_prompt="Ground only the named target.",
        run_id="test-run",
        model_request_block_reason="test blocks external model calls",
    )
    shared_output = tmp_path / "shared"
    for _name, lock in (("first", first_lock), ("second", second_lock)):
        with pytest.raises(RuntimeError, match="circuit breaker"):
            _ground_tracking_seed(
                output_dir=shared_output,
                query_lock_v2=lock,
                **common,
            )
    request_keys = list(shared_output.glob("grounding/*/request-key.json"))
    assert len(request_keys) == 1
    lineages = [
        json.loads(path.read_text())
        for path in shared_output.glob("grounding/*/query-lineage-*.json")
    ]
    assert {item["definition_sha256"] for item in lineages} == {
        first_lock.definition_sha256(),
        second_lock.definition_sha256(),
    }


def test_runtime_geometry_lineage_binds_seed_track_and_query_hashes(tmp_path) -> None:
    seed_path = tmp_path / "seed-selection.json"
    track_path = tmp_path / "segmentation-track.json"
    seed_path.write_text('{"seed":"immutable"}', encoding="utf-8")
    track_path.write_text('{"track":"immutable"}', encoding="utf-8")

    class FakeTrack:
        def model_dump(self, *, mode: str):
            assert mode == "json"
            return {"samples": [{"time_ms": 0, "box": [1, 2, 3, 4]}]}

    lock = _auto_lock()
    lineage = _query_lock_v2_runtime_geometry_lineage(
        lock=lock,
        target_id="target.main",
        target_description="the selected foreground instance",
        seed_fingerprint="e" * 64,
        seed_manifest_path=seed_path,
        track_path=track_path,
        track=FakeTrack(),  # type: ignore[arg-type]
    )

    assert lineage["seed_fingerprint"] == "e" * 64
    assert len(lineage["seed_selection_sha256"]) == 64
    assert len(lineage["track_sha256"]) == 64
    assert len(lineage["track_geometry_sha256"]) == 64
    assert lineage["evidence_query_v2"]["definition_sha256"] == (
        lock.definition_sha256()
    )
