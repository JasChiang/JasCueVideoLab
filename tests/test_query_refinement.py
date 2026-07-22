from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import jascue_video_lab.gemini as gemini_module
from jascue_video_lab.gemini import GeminiLabClient, VISUAL_EVIDENCE_SYSTEM_INSTRUCTION
from jascue_video_lab.models import (
    DenseFrame,
    DenseFrameCatalog,
    EvidenceApprovalSource,
    EvidenceClaimSource,
    EvidenceFramingObligationsV2,
    EvidenceIdentityContractV2,
    EvidencePredicateContractV2,
    EvidencePredicatePhasesV2,
    EvidenceQueryApprovalProvenance,
    EvidenceQueryLockV2,
    EvidenceQueryProvenanceV2,
    EvidenceTargetIdentityV2,
    ModelProvenance,
    PredicateRequiredAt,
)
from jascue_video_lab.query_refinement import (
    QueryTemporalDecision,
    QueryTemporalSelection,
    build_query_temporal_fingerprint,
    dense_catalog_evidence_sha256,
    resolve_query_temporal_selection,
    validate_query_temporal_evidence_bundle,
)
from jascue_video_lab.schema import gemini_response_schema


def _provenance(interaction_id: str | None = None) -> ModelProvenance:
    return ModelProvenance(
        model_id="gemini-test",
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version="2.3.0",
        interaction_id=interaction_id,
        run_id="run-query-temporal",
        generated_at="2026-07-22T00:00:00Z",
    )


def _lock(required_at: PredicateRequiredAt) -> EvidenceQueryLockV2:
    phases = (
        EvidencePredicatePhasesV2(
            precondition="the requested state is not yet visible",
            apex="the directly observable change is underway",
            postcondition="the requested state is directly visible",
        )
        if required_at == PredicateRequiredAt.TRANSITION
        else None
    )
    return EvidenceQueryLockV2(
        query_id="query:generic",
        revision=1,
        editorial_goal="Show the selected instance in the requested observable state.",
        identity=EvidenceIdentityContractV2(
            targets=(
                EvidenceTargetIdentityV2(
                    target_id="subject.primary",
                    target_description="the reviewer-selected foreground instance",
                    identity_cues=("persistent outline", "distinctive surface detail"),
                    context_cues=("near the active participant when selected",),
                    stable_exclusions=("background depiction", "reflection"),
                ),
            )
        ),
        predicate=EvidencePredicateContractV2(
            predicate_id="predicate:generic",
            statement="the selected instance enters the requested observable state",
            participant_target_ids=("subject.primary",),
            required_at=required_at,
            phases=phases,
            required_evidence=("the same instance remains identifiable",),
        ),
        framing=EvidenceFramingObligationsV2(
            required_target_ids=("subject.primary",),
            framing_intent="Keep the selected instance recognizable.",
        ),
        claim_source=EvidenceClaimSource.HUMAN_REVIEW,
        provenance=EvidenceQueryProvenanceV2(
            created_at="2026-07-22T00:00:00Z",
            created_by="reviewer:generic",
        ),
        approval=EvidenceQueryApprovalProvenance(
            approved_at="2026-07-22T00:01:00Z",
            approved_by="reviewer:generic",
            approval_source=EvidenceApprovalSource.HUMAN_REVIEW,
        ),
    )


def _catalog(tmp_path: Path, *, page_name: str = "contact.jpg") -> DenseFrameCatalog:
    page = tmp_path / page_name
    page.write_bytes(b"immutable contact sheet bytes")
    frames = [
        DenseFrame(
            frame_id=f"DF{index:06d}",
            event_id="event:generic",
            requested_time_ms=index * 250,
            frame_time_ms=index * 250,
            frame_pts=index * 30,
            frame_hash=f"{index:064x}",
            width=960,
            height=540,
            image_path=str(tmp_path / f"source-{index}.png"),
            transport_image_path=str(tmp_path / f"transport-{index}.jpg"),
            transport_image_hash=f"{index + 10:064x}",
        )
        for index in range(1, 6)
    ]
    return DenseFrameCatalog(
        source_asset_id="sha256:" + "a" * 64,
        event_id="event:generic",
        sampling_fps=4,
        source_start_ms=250,
        source_end_ms=1600,
        frames=frames,
        contact_sheet_paths=[str(page)],
        contact_sheet_hashes=[hashlib.sha256(page.read_bytes()).hexdigest()],
        generated_at="2026-07-22T00:00:00Z",
    )


def _fingerprint(
    lock: EvidenceQueryLockV2, catalog: DenseFrameCatalog
):
    return build_query_temporal_fingerprint(
        query_lock=lock,
        grounding_target_id="subject.primary",
        catalog=catalog,
        model_id="gemini-test",
        prompt_template="Choose only directly supported DF identifiers.",
        system_instruction=VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
        response_schema=gemini_response_schema(QueryTemporalSelection),
    )


def _selection_payload(
    lock: EvidenceQueryLockV2,
    catalog: DenseFrameCatalog,
    *,
    selected_ids: tuple[str, ...],
) -> dict[str, Any]:
    fingerprint = _fingerprint(lock, catalog)
    required_at = lock.predicate.required_at
    payload: dict[str, Any] = {
        "source_asset_id": catalog.source_asset_id,
        "event_id": catalog.event_id,
        "query_id": lock.query_id,
        "grounding_target_id": "subject.primary",
        "identity_sha256": fingerprint.identity_sha256,
        "predicate_sha256": fingerprint.predicate_sha256,
        "catalog_sha256": fingerprint.catalog_sha256,
        "request_sha256": fingerprint.request_sha256,
        "required_at": required_at.value,
        "match_status": "matched",
        "predicate_status": "satisfied",
        "candidate_frame_id": None,
        "seed_frame_id": None,
        "precondition_frame_id": None,
        "apex_frame_id": None,
        "postcondition_frame_id": None,
        "interval_sample_frame_ids": [],
        "evidence": [
            {
                "frame_id": frame_id,
                "identity_status_by_target": {"subject.primary": "observed"},
                "predicate_observed": True,
                "transition_phase": (
                    ("precondition", "apex", "postcondition")[index]
                    if required_at == PredicateRequiredAt.TRANSITION
                    else None
                ),
                "observation": "the locked instance and requested state are visible",
            }
            for index, frame_id in enumerate(selected_ids)
        ],
        "observable_evidence_summary": "Direct observations in supplied DF samples.",
        "uncertainties": [],
        "confidence": 0.8,
        "model_provenance": _provenance().model_dump(mode="json"),
    }
    if required_at == PredicateRequiredAt.CANDIDATE:
        payload["candidate_frame_id"] = selected_ids[0]
        payload["coverage_claim"] = "single_frame_only"
    elif required_at == PredicateRequiredAt.SEED:
        payload["seed_frame_id"] = selected_ids[0]
        payload["coverage_claim"] = "single_frame_only"
    elif required_at == PredicateRequiredAt.TRANSITION:
        (
            payload["precondition_frame_id"],
            payload["apex_frame_id"],
            payload["postcondition_frame_id"],
        ) = selected_ids
        payload["coverage_claim"] = "transition_samples_only"
    else:
        payload["interval_sample_frame_ids"] = list(selected_ids)
        payload["coverage_claim"] = "sampled_frames_only"
    return payload


def test_fingerprint_is_component_scoped_and_path_independent(tmp_path: Path) -> None:
    lock = _lock(PredicateRequiredAt.SEED)
    catalog = _catalog(tmp_path)
    first = _fingerprint(lock, catalog)

    moved = catalog.model_copy(
        update={
            "generated_at": "later",
            "contact_sheet_paths": [str(tmp_path / "different-host-path.jpg")],
            "frames": [
                frame.model_copy(
                    update={
                        "image_path": f"/other/source/{frame.frame_id}.png",
                        "transport_image_path": f"/other/proxy/{frame.frame_id}.jpg",
                    }
                )
                for frame in catalog.frames
            ],
        }
    )
    framing_changed = lock.model_copy(
        update={
            "framing": lock.framing.model_copy(
                update={"framing_intent": "Prefer more context around the instance."}
            )
        }
    )
    assert dense_catalog_evidence_sha256(moved) == first.catalog_sha256
    assert _fingerprint(framing_changed, moved).request_sha256 == first.request_sha256

    predicate_changed = lock.model_copy(
        update={
            "predicate": lock.predicate.model_copy(
                update={"statement": "a different directly observable state is visible"}
            )
        }
    )
    assert _fingerprint(predicate_changed, catalog).request_sha256 != first.request_sha256


def test_model_response_contract_has_frame_ids_but_no_source_time_fields() -> None:
    schema = gemini_response_schema(QueryTemporalSelection)
    properties = schema["properties"]
    assert "seed_frame_id" in properties
    assert "precondition_frame_id" in properties
    assert "interval_sample_frame_ids" in properties
    assert not {
        "frame_pts",
        "frame_time_ms",
        "requested_time_ms",
        "start_ms",
        "end_ms",
        "timestamp",
    }.intersection(properties)


@pytest.mark.parametrize(
    ("required_at", "ids", "expected_field"),
    [
        (PredicateRequiredAt.CANDIDATE, ("DF000002",), "candidate_frame"),
        (PredicateRequiredAt.SEED, ("DF000003",), "seed_frame"),
        (
            PredicateRequiredAt.TRANSITION,
            ("DF000001", "DF000003", "DF000005"),
            "apex_frame",
        ),
        (
            PredicateRequiredAt.INTERVAL,
            (
                "DF000002",
                "DF000003",
                "DF000004",
            ),
            "interval_sample_frames",
        ),
    ],
)
def test_local_resolution_enforces_each_required_at_semantics(
    tmp_path: Path,
    required_at: PredicateRequiredAt,
    ids: tuple[str, ...],
    expected_field: str,
) -> None:
    lock = _lock(required_at)
    catalog = _catalog(tmp_path)
    selection = QueryTemporalSelection.model_validate(
        _selection_payload(lock, catalog, selected_ids=ids)
    )
    decision = resolve_query_temporal_selection(
        selection=selection,
        query_lock=lock,
        catalog=catalog,
        fingerprint=_fingerprint(lock, catalog),
    )
    assert getattr(decision, expected_field)
    if required_at == PredicateRequiredAt.INTERVAL:
        assert decision.coverage_claim == "sampled_frames_only"
        assert [frame.frame_pts for frame in decision.interval_sample_frames] == [
            60,
            90,
            120,
        ]


def test_transition_must_be_ordered_by_resolved_source_pts(tmp_path: Path) -> None:
    lock = _lock(PredicateRequiredAt.TRANSITION)
    catalog = _catalog(tmp_path)
    selection = QueryTemporalSelection.model_validate(
        _selection_payload(
            lock,
            catalog,
            selected_ids=("DF000005", "DF000003", "DF000001"),
        )
    )
    with pytest.raises(ValidationError, match="pre < apex < post PTS"):
        resolve_query_temporal_selection(
            selection=selection,
            query_lock=lock,
            catalog=catalog,
            fingerprint=_fingerprint(lock, catalog),
        )


def test_unknown_df_and_hash_mismatch_are_rejected_locally(tmp_path: Path) -> None:
    lock = _lock(PredicateRequiredAt.SEED)
    catalog = _catalog(tmp_path)
    payload = _selection_payload(lock, catalog, selected_ids=("DF999999",))
    selection = QueryTemporalSelection.model_validate(payload)
    with pytest.raises(ValueError, match="unknown DF IDs"):
        resolve_query_temporal_selection(
            selection=selection,
            query_lock=lock,
            catalog=catalog,
            fingerprint=_fingerprint(lock, catalog),
        )

    payload = _selection_payload(lock, catalog, selected_ids=("DF000002",))
    payload["identity_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="immutable metadata mismatch"):
        resolve_query_temporal_selection(
            selection=QueryTemporalSelection.model_validate(payload),
            query_lock=lock,
            catalog=catalog,
            fingerprint=_fingerprint(lock, catalog),
        )


def test_non_positive_response_cannot_leak_a_selected_frame(tmp_path: Path) -> None:
    lock = _lock(PredicateRequiredAt.SEED)
    catalog = _catalog(tmp_path)
    payload = _selection_payload(lock, catalog, selected_ids=("DF000002",))
    payload.update(
        {
            "match_status": "ambiguous",
            "predicate_status": "indeterminate",
            "coverage_claim": "no_positive_claim",
        }
    )
    with pytest.raises(ValidationError, match="cannot select frames"):
        QueryTemporalSelection.model_validate(payload)


class _Interaction:
    def __init__(self, output_text: str) -> None:
        self.id = "interaction-query-temporal"
        self.output_text = output_text

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "id": self.id,
            "output_text": self.output_text,
            "usage": {"total_input_tokens": 123, "total_output_tokens": 45},
        }


class _Interactions:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **request: Any) -> _Interaction:
        self.calls.append(request)
        return _Interaction(self.output_text)


def _client(output_text: str) -> tuple[GeminiLabClient, _Interactions]:
    interactions = _Interactions(output_text)
    client = object.__new__(GeminiLabClient)
    client.client = SimpleNamespace(interactions=interactions)
    client.model_id = "gemini-test"
    return client, interactions


def test_interactions_call_is_single_shot_auditable_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(PredicateRequiredAt.TRANSITION)
    catalog = _catalog(tmp_path)
    payload = _selection_payload(
        lock,
        catalog,
        selected_ids=("DF000001", "DF000003", "DF000005"),
    )
    client, interactions = _client(json.dumps(payload))
    monkeypatch.setattr(
        gemini_module,
        "_provenance",
        lambda run_id, interaction_id=None, model_id="gemini-test": _provenance(
            interaction_id
        ),
    )
    run_dir = tmp_path / "run"
    decision = client.refine_query_lock_frames(
        query_lock=lock,
        grounding_target_id="subject.primary",
        catalog=catalog,
        prompt_template="Choose only directly supported DF identifiers.",
        run_id="run-query-temporal",
        run_dir=run_dir,
    )

    assert len(interactions.calls) == 1
    assert decision.apex_frame is not None
    assert decision.apex_frame.frame_pts == 90
    api_images = [
        item for item in interactions.calls[0]["input"] if item["type"] == "image"
    ]
    assert api_images and all("data" in item for item in api_images)

    saved = json.loads(
        (run_dir / "query_temporal.request.json").read_text(encoding="utf-8")
    )
    saved_images = [item for item in saved["input"] if item["type"] == "image"]
    assert saved_images and all("data" not in item for item in saved_images)
    serialized = json.dumps(saved, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert not any(
        forbidden in serialized
        for forbidden in ("source-1.png", "transport-1.jpg", "contact.jpg")
    )
    assert saved["fingerprint"]["identity_sha256"] == lock.component_hashes()[
        "identity_sha256"
    ]
    validation = json.loads(
        (run_dir / "query_temporal.schema_validation.json").read_text()
    )
    assert validation == {
        "api_call_count": 1,
        "errors": [],
        "ok": True,
        "repair_attempted": False,
    }
    assert (run_dir / "query_temporal.raw_interaction.json").exists()
    assert (run_dir / "query_temporal.raw_output.json").exists()
    assert (run_dir / "query_temporal.response_schema.json").exists()
    assert (run_dir / "query_temporal.catalog.snapshot.json").exists()
    assert (run_dir / "query_temporal.bundle.json").exists()
    assert json.loads((run_dir / "query_temporal.usage.json").read_text())["usage"]
    assert validate_query_temporal_evidence_bundle(
        run_dir / "query_temporal.decision.json",
        query_lock=lock,
        expected_system_instruction=VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
        expected_model_id="gemini-test",
        expected_prompt_template="Choose only directly supported DF identifiers.",
    ) == decision


def test_invalid_output_is_not_repaired_or_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(PredicateRequiredAt.SEED)
    catalog = _catalog(tmp_path)
    client, interactions = _client("{}")
    monkeypatch.setattr(
        gemini_module,
        "_provenance",
        lambda run_id, interaction_id=None, model_id="gemini-test": _provenance(
            interaction_id
        ),
    )
    run_dir = tmp_path / "invalid"
    with pytest.raises(ValidationError):
        client.refine_query_lock_frames(
            query_lock=lock,
            grounding_target_id="subject.primary",
            catalog=catalog,
            prompt_template="Choose only directly supported DF identifiers.",
            run_id="run-query-temporal",
            run_dir=run_dir,
        )
    assert len(interactions.calls) == 1
    validation = json.loads(
        (run_dir / "query_temporal.schema_validation.json").read_text()
    )
    assert validation["ok"] is False
    assert validation["repair_attempted"] is False
    assert validation["api_call_count"] == 1


def test_hand_edited_positive_decision_without_required_frame_fails_closed() -> None:
    with pytest.raises(ValidationError, match="resolved seed decision"):
        QueryTemporalDecision.model_validate(
            {
                "source_asset_id": "sha256:" + "a" * 64,
                "event_id": "event:generic",
                "query_id": "query:generic",
                "grounding_target_id": "subject.primary",
                "identity_sha256": "b" * 64,
                "predicate_sha256": "c" * 64,
                "catalog_sha256": "d" * 64,
                "request_sha256": "e" * 64,
                "required_at": "seed",
                "match_status": "matched",
                "predicate_status": "satisfied",
                "coverage_claim": "single_frame_only",
                "observable_evidence_summary": "direct evidence",
                "confidence": 0.8,
                "model_provenance": _provenance().model_dump(mode="json"),
            }
        )


def test_positive_selected_frame_requires_identity_and_predicate_evidence(
    tmp_path: Path,
) -> None:
    lock = _lock(PredicateRequiredAt.SEED)
    catalog = _catalog(tmp_path)
    payload = _selection_payload(lock, catalog, selected_ids=("DF000002",))
    payload["evidence"][0]["identity_status_by_target"] = {
        "subject.primary": "mismatch"
    }
    with pytest.raises(ValidationError, match="every reported participant"):
        QueryTemporalSelection.model_validate(payload)


def test_positive_interval_cannot_cherry_pick_catalog_samples(tmp_path: Path) -> None:
    lock = _lock(PredicateRequiredAt.INTERVAL)
    catalog = _catalog(tmp_path)
    selection = QueryTemporalSelection.model_validate(
        _selection_payload(
            lock,
            catalog,
            selected_ids=("DF000001", "DF000005"),
        )
    )
    with pytest.raises(ValueError, match="contiguous catalog run"):
        resolve_query_temporal_selection(
            selection=selection,
            query_lock=lock,
            catalog=catalog,
            fingerprint=_fingerprint(lock, catalog),
        )


def test_temporal_bundle_rejects_tampered_saved_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _lock(PredicateRequiredAt.SEED)
    catalog = _catalog(tmp_path)
    payload = _selection_payload(lock, catalog, selected_ids=("DF000002",))
    client, _interactions = _client(json.dumps(payload))
    monkeypatch.setattr(
        gemini_module,
        "_provenance",
        lambda run_id, interaction_id=None, model_id="gemini-test": _provenance(
            interaction_id
        ),
    )
    run_dir = tmp_path / "tampered"
    client.refine_query_lock_frames(
        query_lock=lock,
        grounding_target_id="subject.primary",
        catalog=catalog,
        prompt_template="Choose only directly supported DF identifiers.",
        run_id="run-query-temporal",
        run_dir=run_dir,
    )
    decision_path = run_dir / "query_temporal.decision.json"
    decision_path.write_text(
        decision_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bundle hash mismatch"):
        validate_query_temporal_evidence_bundle(
            decision_path,
            query_lock=lock,
            expected_system_instruction=VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
            expected_model_id="gemini-test",
            expected_prompt_template="Choose only directly supported DF identifiers.",
        )
