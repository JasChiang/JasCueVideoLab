from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from jascue_video_lab.models import (
    BoundaryPrecision,
    CardOpportunity,
    Entity,
    EntityKind,
    EvidenceModality,
    FeatureChapterBrief,
    FeatureEditBrief,
    FullClipCard,
    FullClipEvent,
    FullClipGroundingTarget,
    ModelProvenance,
    RushClip,
    RushFrame,
    RushesCatalog,
)
from jascue_video_lab.feature_cut import write_external_feature_plan_projection
from jascue_video_lab.gemini import MODEL_ID
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import write_json
from scripts.plan_clip_card_feature_cut import (
    ClipCardFeatureCandidate,
    ClipCardFeatureCandidateV3,
    ClipCardFeaturePlan,
    ClipCardFeaturePlanV2,
    ClipCardFeaturePlanV3,
    ClipCardFeatureSelect,
    ClipCardFeatureSelectV2,
    ClipCardFeatureSelectV3,
    ResolvedEntityRef,
    ResolvedFramingRegion,
    SelectedClipCardEvidence,
    _assert_fresh_feature_namespace_empty,
    _resolve_feature_reuse_artifacts,
    _verified_feature_raw_output_text,
    _write_feature_normalization_artifacts,
    build_selected_clip_card_evidence,
    canonicalize_feature_plan_output,
    compact_card,
    compact_card_v3,
    project_feature_contracts,
    project_feature_contracts_v3,
    reproject_external_feature_plan,
    reproject_external_feature_plan_v2,
    reproject_external_feature_plan_v3,
    validate_plan_contract,
    validate_plan_contract_v3,
    main as feature_planner_main,
)


ASSET_ID = "sha256:" + "a" * 64


def _provenance() -> ModelProvenance:
    return ModelProvenance(
        model_id="gemini-3.6-flash",
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version="1.0",
        interaction_id="interaction-1",
        run_id="run-1",
        generated_at="2026-07-22T00:00:00+00:00",
    )


def _card() -> FullClipCard:
    return FullClipCard(
        source_asset_id=ASSET_ID,
        proxy_asset_id="sha256:" + "b" * 64,
        duration_ms=10_000,
        summary="A visible subject demonstrates an object beside a sign.",
        content_type="generic demonstration",
        entities=[
            Entity(
                entity_id="subject-1",
                kind=EntityKind.PERSON,
                label="visible subject",
                distinguishing_features="standing at frame center",
                evidence="visible throughout the event",
            ),
            Entity(
                entity_id="sign-1",
                kind=EntityKind.TEXT_REGION,
                label="foreground sign",
                distinguishing_features="wide line of visible text",
                evidence="visible along the lower edge",
            ),
        ],
        events=[
            FullClipEvent(
                event_id="demo",
                start_mmss="00:00",
                end_mmss="00:10",
                recommended_keyframe_mmss="00:02",
                label="demonstration",
                description="The subject demonstrates an object beside a sign.",
                observable_evidence="A subject and a foreground sign are visible.",
                evidence_modalities=EvidenceModality.VISUAL,
                entity_ids=["subject-1", "sign-1"],
                primary_entity_ids=["subject-1"],
                required_entity_ids=["subject-1", "sign-1"],
                optional_entity_ids=[],
                avoid_overlay_entity_ids=["sign-1"],
                keyframe_reason="Both required regions are clear.",
                boundary_precision=BoundaryPrecision.SECOND_LEVEL,
                confidence=0.8,
                action_completeness="complete",
                editing_uses=["demo"],
                quality_risks=[],
                framing_intent="Keep the subject and sign visible.",
                card_opportunities=[
                    CardOpportunity(
                        kind="object_callout",
                        rationale="The sign can be referenced without covering it.",
                        entity_ids=["sign-1"],
                    )
                ],
                dense_refinement="not_needed",
                dense_refinement_reasons=[],
                grounding_targets=[
                    FullClipGroundingTarget(
                        entity_id="subject-1",
                        target_kind=EntityKind.PERSON,
                        target_description="the visible subject",
                        purpose="reframe",
                    )
                ],
            )
        ],
        clip_uses=["demo"],
        portrait_reframe_feasibility="good",
        uncertainties=[],
        model_provenance=_provenance(),
    )


def _catalog() -> RushesCatalog:
    return RushesCatalog(
        catalog_id="catalog-1",
        source_directory="/source",
        sample_interval_ms=1_000,
        total_duration_ms=10_000,
        clips=[
            RushClip(
                clip_id="clip-1",
                path="/source/clip.mp4",
                sha256="a" * 64,
                duration_ms=10_000,
                width=1920,
                height=1080,
                frame_rate="30/1",
                size_bytes=1,
            )
        ],
        frames=[
            RushFrame(
                frame_id="RF000001",
                clip_id="clip-1",
                requested_time_ms=2_000,
                image_path="/frames/1.jpg",
            ),
            RushFrame(
                frame_id="RF000002",
                clip_id="clip-1",
                requested_time_ms=3_000,
                image_path="/frames/2.jpg",
            ),
        ],
        analysis_reel_path="/analysis.mp4",
        generated_at="2026-07-22T00:00:00+00:00",
    )


def _brief() -> FeatureEditBrief:
    return FeatureEditBrief(
        project_id="project-1",
        title="Generic feature cut",
        target_duration_seconds=60,
        chapters=[
            FeatureChapterBrief(
                feature_id="feature-1",
                title="Visible demonstration",
                detail_lines=["Show the directly observable demonstration."],
                target_duration_seconds=6,
                vertical_primary_target_description="the visible subject",
            )
        ],
    )


def _region(
    *,
    region_id: str,
    entity_id: str,
    event_relation: str,
    constraint_role: str = "hard_core",
) -> ResolvedFramingRegion:
    return ResolvedFramingRegion.model_validate(
        {
            "region_id": region_id,
            "target_description": f"visible region for {entity_id}",
            "kind": "text_region" if entity_id == "sign-1" else "subject",
            "constraint_role": constraint_role,
            "composition": "atomic",
            "entity_refs": [
                {"entity_id": entity_id, "event_relation": event_relation}
            ],
            "observable_relation": "The region is directly visible in the event.",
        }
    )


def _candidate(
    candidate_id: str, frame_id: str, *, vertical_strategy: str
) -> ClipCardFeatureCandidate:
    resolved_regions = (
        [
            _region(
                region_id=f"{candidate_id}-subject",
                entity_id="subject-1",
                event_relation="required",
            ),
            _region(
                region_id=f"{candidate_id}-sign",
                entity_id="sign-1",
                event_relation="avoid_overlay",
                constraint_role="overlay_keepout",
            ),
        ]
        if vertical_strategy == "tracked_crop"
        else []
    )
    return ClipCardFeatureCandidate(
        candidate_id=candidate_id,
        source_asset_id=ASSET_ID,
        event_id="demo",
        frame_id=frame_id,
        observed_visual_evidence="The subject and sign are both visible.",
        selection_reason="Complete visible action with auditable regions.",
        quality_risks=[],
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_target_description=None,
        vertical_strategy=vertical_strategy,
        vertical_target_description=(
            "the directly visible subject"
            if vertical_strategy == "tracked_crop"
            else None
        ),
        resolved_regions=resolved_regions,
        confidence=0.8,
    )


def _v2_plan() -> ClipCardFeaturePlanV2:
    first = _candidate(
        "candidate-a", "RF000001", vertical_strategy="tracked_crop"
    )
    second = _candidate(
        "candidate-b", "RF000002", vertical_strategy="fit_with_background"
    )
    return ClipCardFeaturePlanV2(
        contract_version="clip-card-feature-cut-v2",
        project_id="project-1",
        catalog_id="catalog-1",
        title="Generic feature cut",
        strategy_summary="Preserve alternatives and evidence-bound regions.",
        chapters=[
            ClipCardFeatureSelectV2(
                feature_id="feature-1",
                evidence_status="supported",
                horizontal_source_asset_id=ASSET_ID,
                horizontal_event_id="demo",
                horizontal_frame_id="RF000001",
                vertical_source_asset_id=ASSET_ID,
                vertical_event_id="demo",
                vertical_frame_id="RF000002",
                observed_visual_evidence="The subject and sign are visible.",
                selection_reason="Each aspect uses a validated candidate.",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                # A brief primary target no longer forces tracked_crop.
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                quality_risks=[],
                confidence=0.8,
                candidates=[first, second],
                horizontal_candidate_id="candidate-a",
                vertical_candidate_id="candidate-b",
            )
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )


def _v3_plan() -> ClipCardFeaturePlanV3:
    first = ClipCardFeatureCandidateV3(
        candidate_id="candidate-a",
        source_asset_id=ASSET_ID,
        event_id="demo",
        frame_id="RF000001",
        observed_visual_evidence="The subject and sign are both visible.",
        selection_reason="Complete action with enough room for a portrait crop.",
        quality_risks=[],
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_focus_entity_id=None,
        vertical_strategy="tracked_crop",
        vertical_crop_mode="strict",
        framing_intent="Prioritize the subject; retain the sign as useful context.",
        required_entity_ids=["subject-1"],
        preferred_entity_ids=["sign-1"],
        sacrificable_entity_ids=[],
        confidence=0.8,
    )
    second = ClipCardFeatureCandidateV3(
        candidate_id="candidate-b",
        source_asset_id=ASSET_ID,
        event_id="demo",
        frame_id="RF000002",
        observed_visual_evidence="The subject is clear and the sign is peripheral.",
        selection_reason="Stable wide composition can be fit without tracking.",
        quality_risks=["The sign is near the edge."],
        horizontal_strategy="tracked_reframe",
        horizontal_zoom_intent="subtle",
        horizontal_focus_entity_id="subject-1",
        vertical_strategy="fit_with_background",
        vertical_crop_mode="primary_center",
        framing_intent="Preserve the whole source; the sign may be sacrificed.",
        required_entity_ids=["subject-1"],
        preferred_entity_ids=[],
        sacrificable_entity_ids=["sign-1"],
        confidence=0.7,
    )
    return ClipCardFeaturePlanV3(
        contract_version="clip-card-feature-cut-v3",
        project_id="project-1",
        catalog_id="catalog-1",
        title="Generic feature cut",
        strategy_summary="Select alternatives; derive geometry from local evidence.",
        chapters=[
            ClipCardFeatureSelectV3(
                feature_id="feature-1",
                evidence_status="supported",
                candidates=[first, second],
                horizontal_candidate_id="candidate-a",
                vertical_candidate_id="candidate-b",
            )
        ],
        uncertainties=[],
        model_provenance=_provenance(),
    )


def test_feature_output_canonicalization_is_narrow_ordered_and_auditable() -> None:
    payload = _v3_plan().model_dump(mode="json")
    first, second = payload["chapters"][0]["candidates"]
    first.update(
        {
            "horizontal_strategy": "original",
            "horizontal_zoom_intent": "detail",
            "horizontal_focus_entity_id": "subject-1",
        }
    )
    second.update(
        {
            "horizontal_strategy": "original",
            "horizontal_zoom_intent": "none",
            "horizontal_focus_entity_id": "subject-1",
        }
    )
    original = json.dumps(payload)

    canonical_text, changes = canonicalize_feature_plan_output(original)
    canonical = json.loads(canonical_text)

    assert original == json.dumps(payload)
    assert canonical["chapters"][0]["candidates"][0]["horizontal_strategy"] == "original"
    assert canonical["chapters"][0]["candidates"][0]["horizontal_zoom_intent"] == "none"
    assert canonical["chapters"][0]["candidates"][0][
        "horizontal_focus_entity_id"
    ] is None
    assert canonical["chapters"][0]["candidates"][1][
        "horizontal_focus_entity_id"
    ] is None
    assert [change["rule"] for change in changes] == [
        "explicit_original_strategy_disables_zoom",
        "explicit_original_strategy_has_no_focus_entity",
        "explicit_original_strategy_has_no_focus_entity",
    ]
    ClipCardFeaturePlanV3.model_validate_json(canonical_text)


def test_feature_reuse_rejects_mismatched_raw_response_copies() -> None:
    with pytest.raises(ValueError, match="does not exactly match"):
        _verified_feature_raw_output_text(
            raw_output={"output_text": "first"},
            raw_interaction={"output_text": "second"},
        )


def test_fresh_feature_run_refuses_existing_paid_namespace(tmp_path: Path) -> None:
    write_json(
        tmp_path / "clip-card-feature-plan.attempt-01.raw_output.json",
        {"output_text": "already paid"},
    )
    with pytest.raises(FileExistsError, match="new output directory"):
        _assert_fresh_feature_namespace_empty(tmp_path)


def test_feature_raw_reuse_resolves_complete_set_and_preserves_paid_artifact(
    tmp_path: Path,
) -> None:
    stem = "clip-card-feature-plan.attempt-01"
    paths = {
        "request": tmp_path / f"{stem}.request.json",
        "raw_output": tmp_path / f"{stem}.raw_output.json",
        "raw_interaction": tmp_path / f"{stem}.raw_interaction.json",
    }
    write_json(paths["request"], {"model": MODEL_ID})
    raw_payload = _v3_plan().model_dump(mode="json")
    raw_payload["chapters"][0]["candidates"][0].update(
        {
            "horizontal_strategy": "original",
            "horizontal_zoom_intent": "subtle",
            "horizontal_focus_entity_id": "subject-1",
        }
    )
    write_json(paths["raw_output"], {"output_text": json.dumps(raw_payload)})
    write_json(paths["raw_interaction"], {"model": MODEL_ID, "id": "paid-1"})
    original_bytes = paths["raw_output"].read_bytes()

    resolved = _resolve_feature_reuse_artifacts(tmp_path)
    canonical_text, canonical_path, audit_path = _write_feature_normalization_artifacts(
        output_dir=tmp_path,
        artifact_stem="clip-card-feature-plan",
        raw_output_path=resolved["raw_output"],
        raw_output_text=json.loads(original_bytes)["output_text"],
    )

    assert resolved["kind"] == "attempt-01"
    assert paths["raw_output"].read_bytes() == original_bytes
    assert canonical_path.exists() and audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["change_count"] == 2
    assert audit["raw_output_artifact_sha256"] == hashlib.sha256(
        original_bytes
    ).hexdigest()
    ClipCardFeaturePlanV3.model_validate_json(canonical_text)


def test_feature_reuse_binding_keeps_original_paid_request_as_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "catalog.json"
    brief_path = tmp_path / "brief.json"
    library = tmp_path / "library"
    output_dir = tmp_path / "plan"
    card_path = (
        library / "clips" / ("a" * 16) / "gemini" / "clip-card" / "clip_card.json"
    )
    write_json(catalog_path, _catalog())
    write_json(brief_path, _brief())
    write_json(card_path, _card())
    output_dir.mkdir()

    payload = _v3_plan().model_dump(mode="json")
    payload["chapters"][0]["candidates"][0].update(
        {
            "horizontal_strategy": "original",
            "horizontal_zoom_intent": "subtle",
            "horizontal_focus_entity_id": "subject-1",
        }
    )
    output_text = json.dumps(payload)
    stem = output_dir / "clip-card-feature-plan.attempt-01"
    paid_request_path = Path(f"{stem}.request.json")
    paid_raw_output_path = Path(f"{stem}.raw_output.json")
    paid_raw_interaction_path = Path(f"{stem}.raw_interaction.json")
    write_json(
        paid_request_path,
        {
            "model": MODEL_ID,
            "system_instruction": "Use only supplied evidence.",
            "input": [{"type": "text", "text": "Paid request."}],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(ClipCardFeaturePlanV3),
            },
        },
    )
    write_json(paid_raw_output_path, {"output_text": output_text})
    write_json(
        paid_raw_interaction_path,
        {
            "model": MODEL_ID,
            "id": "interaction-1",
            "output_text": output_text,
        },
    )
    original_request_hash = hashlib.sha256(paid_request_path.read_bytes()).hexdigest()
    original_raw_bytes = paid_raw_output_path.read_bytes()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_clip_card_feature_cut.py",
            str(catalog_path),
            str(brief_path),
            str(library),
            str(output_dir),
            "--reuse-raw-output",
        ],
    )

    assert feature_planner_main() == 0

    pointer = json.loads(
        (output_dir / "feature-plan.external-projection.json").read_text(
            encoding="utf-8"
        )
    )
    record = json.loads(
        (output_dir / pointer["record_path"]).read_text(encoding="utf-8")
    )
    assert record["source_request_sha256"] == original_request_hash
    assert paid_raw_output_path.read_bytes() == original_raw_bytes

    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-used")
    monkeypatch.setattr(
        "scripts.plan_clip_card_feature_cut.genai.Client",
        lambda **_: pytest.fail("fresh rerun must fail before constructing an API client"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "plan_clip_card_feature_cut.py",
            str(catalog_path),
            str(brief_path),
            str(library),
            str(output_dir),
        ],
    )
    with pytest.raises(FileExistsError, match="new output directory"):
        feature_planner_main()
def test_compact_card_preserves_entities_roles_and_relations() -> None:
    compact = compact_card(_card())

    assert [entity["entity_id"] for entity in compact["entities"]] == [
        "subject-1",
        "sign-1",
    ]
    event = compact["events"][0]
    assert event["primary_entity_ids"] == ["subject-1"]
    assert event["required_entity_ids"] == ["subject-1", "sign-1"]
    assert event["optional_entity_ids"] == []
    assert event["avoid_overlay_entity_ids"] == ["sign-1"]
    relations = {
        item["entity_id"]: item["relations"] for item in event["entity_relations"]
    }
    assert relations["subject-1"] == [
        "event_member",
        "primary",
        "required",
        "grounding_target",
    ]
    assert relations["sign-1"] == ["event_member", "required", "avoid_overlay"]


def test_v1_source_schema_remains_the_exact_single_selection_shape() -> None:
    legacy_schema = gemini_response_schema(ClipCardFeaturePlan)
    current_schema = gemini_response_schema(ClipCardFeaturePlanV2)
    legacy_schema_hash = hashlib.sha256(
        json.dumps(
            legacy_schema,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    # Canonical schema from commit 7a3d686. Existing v1 raw request sidecars
    # are checked by exact schema equality, so this fingerprint is contractual.
    assert legacy_schema_hash == (
        "112584f2b89bee80869d81b63597bf68fe1552edf9c11e7ef4ecb6197145797c"
    )
    assert "contract_version" not in legacy_schema["properties"]
    assert "contract_version" in current_schema["properties"]
    legacy_chapter = legacy_schema["$defs"]["ClipCardFeatureSelect"]["properties"]
    current_chapter = current_schema["$defs"]["ClipCardFeatureSelectV2"][
        "properties"
    ]
    assert "candidates" not in legacy_chapter
    assert "horizontal_candidate_id" not in legacy_chapter
    assert "vertical_candidate_id" not in legacy_chapter
    assert "candidates" in current_chapter


def test_v2_preserves_top_k_and_projects_legacy_feature_plan() -> None:
    plan = _v2_plan()

    validate_plan_contract(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        cards={ASSET_ID: _card()},
    )
    projected = project_feature_contracts(plan, brief=_brief(), catalog=_catalog())

    assert len(plan.chapters[0].candidates) == 2
    assert projected.chapters[0].horizontal_frame_id == "RF000001"
    assert projected.chapters[0].vertical_frame_id == "RF000002"
    assert projected.chapters[0].vertical_strategy == "fit_with_background"
    assert [candidate.candidate_id for candidate in projected.chapters[0].horizontal_candidates] == [
        "candidate-a",
        "candidate-b",
    ]
    assert [candidate.candidate_id for candidate in projected.chapters[0].vertical_candidates] == [
        "candidate-b",
        "candidate-a",
    ]
    assert projected.chapters[0].horizontal_candidates[0].rank == 1
    assert projected.chapters[0].vertical_candidates[0].rank == 1
    assert projected.chapters[0].vertical_candidates[1].regions[0].entity_id == "subject-1"
    assert projected.chapters[0].vertical_candidates[1].regions[0].role == "required"


def test_legacy_single_selection_json_remains_readable() -> None:
    payload = _v2_plan().model_dump(mode="json")
    payload.pop("contract_version")
    for chapter in payload["chapters"]:
        chapter.pop("candidates")
        chapter.pop("horizontal_candidate_id")
        chapter.pop("vertical_candidate_id")

    legacy = ClipCardFeaturePlan.model_validate(payload)
    projected = project_feature_contracts(legacy, brief=_brief(), catalog=_catalog())

    assert not hasattr(legacy, "contract_version")
    assert not hasattr(legacy.chapters[0], "candidates")
    assert projected.chapters[0].vertical_frame_id == "RF000002"
    assert projected.chapters[0].horizontal_candidates == []
    assert projected.chapters[0].vertical_candidates == []


def test_projection_entrypoints_keep_v1_candidate_free_and_v2_top_k() -> None:
    v2 = _v2_plan()
    payload = v2.model_dump(mode="json")
    payload.pop("contract_version")
    for chapter in payload["chapters"]:
        chapter.pop("candidates")
        chapter.pop("horizontal_candidate_id")
        chapter.pop("vertical_candidate_id")
    legacy = ClipCardFeaturePlan.model_validate(payload)

    _, legacy_projection = reproject_external_feature_plan(
        source_plan=legacy,
        catalog=_catalog(),
        brief=_brief(),
        source_artifacts={},
    )
    _, v2_projection = reproject_external_feature_plan_v2(
        source_plan=v2,
        catalog=_catalog(),
        brief=_brief(),
        source_artifacts={},
    )

    assert legacy_projection.chapters[0].horizontal_candidates == []
    assert legacy_projection.chapters[0].vertical_candidates == []
    assert [
        candidate.candidate_id
        for candidate in v2_projection.chapters[0].horizontal_candidates
    ] == ["candidate-a", "candidate-b"]
    assert [
        candidate.candidate_id
        for candidate in v2_projection.chapters[0].vertical_candidates
    ] == ["candidate-b", "candidate-a"]


def test_projection_entrypoints_reject_cross_version_source_plans() -> None:
    v2 = _v2_plan()
    payload = v2.model_dump(mode="json")
    payload.pop("contract_version")
    for chapter in payload["chapters"]:
        chapter.pop("candidates")
        chapter.pop("horizontal_candidate_id")
        chapter.pop("vertical_candidate_id")
    legacy = ClipCardFeaturePlan.model_validate(payload)

    with pytest.raises(ValueError, match="v1 requires its exact legacy source schema"):
        reproject_external_feature_plan(
            source_plan=v2,
            catalog=_catalog(),
            brief=_brief(),
            source_artifacts={},
        )
    with pytest.raises(ValueError, match="v2 requires a clip-card-feature-cut-v2"):
        reproject_external_feature_plan_v2(
            source_plan=legacy,
            catalog=_catalog(),
            brief=_brief(),
            source_artifacts={},
        )


@pytest.mark.parametrize(
    ("projection_contract_id", "source_model"),
    [
        ("clip-card-feature-cut-v1", ClipCardFeaturePlan),
        ("clip-card-feature-cut-v2", ClipCardFeaturePlanV2),
    ],
)
def test_projection_sidecar_validates_both_exact_source_schemas(
    tmp_path: Path,
    projection_contract_id: str,
    source_model: type[ClipCardFeaturePlan] | type[ClipCardFeaturePlanV2],
) -> None:
    v2 = _v2_plan()
    if source_model is ClipCardFeaturePlan:
        payload = v2.model_dump(mode="json")
        payload.pop("contract_version")
        for chapter in payload["chapters"]:
            chapter.pop("candidates")
            chapter.pop("horizontal_candidate_id")
            chapter.pop("vertical_candidate_id")
        source_plan = ClipCardFeaturePlan.model_validate(payload)
    else:
        source_plan = v2
    feature_plan = project_feature_contracts(
        source_plan,
        brief=_brief(),
        catalog=_catalog(),
    )
    plan_dir = tmp_path / "gemini-plan"
    catalog_path = tmp_path / "catalog.json"
    brief_path = tmp_path / "brief.json"
    feature_plan_path = plan_dir / "feature_edit_plan.json"
    source_plan_path = plan_dir / "source-plan.json"
    source_request_path = plan_dir / "source.request.json"
    raw_output_path = plan_dir / "source.raw_output.json"
    raw_interaction_path = plan_dir / "source.raw_interaction.json"
    write_json(catalog_path, _catalog())
    write_json(brief_path, _brief())
    write_json(feature_plan_path, feature_plan)
    write_json(source_plan_path, source_plan)
    write_json(
        source_request_path,
        {
            "model": MODEL_ID,
            "system_instruction": "Use only the supplied evidence.",
            "input": [{"type": "text", "text": "Select auditable evidence."}],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(source_model),
            }
        },
    )
    write_json(raw_output_path, {"output_text": source_plan.model_dump_json()})
    write_json(raw_interaction_path, {"id": "interaction-1"})

    pointer = write_external_feature_plan_projection(
        plan_dir=plan_dir,
        projection_contract_id=projection_contract_id,
        catalog_path=catalog_path,
        brief_path=brief_path,
        feature_plan_path=feature_plan_path,
        source_plan_path=source_plan_path,
        source_request_path=source_request_path,
        source_artifacts={
            "source_raw_output": raw_output_path,
            "source_raw_interaction": raw_interaction_path,
        },
    )

    assert pointer.name == "feature-plan.external-projection.json"


def test_v2_rejects_single_candidate() -> None:
    payload = _v2_plan().model_dump(mode="json")
    chapter = payload["chapters"][0]
    chapter["candidates"] = chapter["candidates"][:1]
    chapter["vertical_candidate_id"] = "candidate-a"
    chapter["vertical_frame_id"] = "RF000001"
    chapter["vertical_strategy"] = "tracked_crop"
    chapter["vertical_target_description"] = "the directly visible subject"

    with pytest.raises(ValidationError, match="Top-K 2-4"):
        ClipCardFeaturePlanV2.model_validate(payload)


def test_selected_tracked_crop_can_use_resolved_hard_core_without_fuzzy_target() -> None:
    payload = _v2_plan().model_dump(mode="json")
    chapter = payload["chapters"][0]
    first = chapter["candidates"][0]
    first["vertical_target_description"] = None
    chapter["vertical_candidate_id"] = "candidate-a"
    chapter["vertical_frame_id"] = "RF000001"
    chapter["vertical_strategy"] = "tracked_crop"
    chapter["vertical_target_description"] = None
    plan = ClipCardFeaturePlanV2.model_validate(payload)

    validate_plan_contract(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        cards={ASSET_ID: _card()},
    )
    projected = project_feature_contracts(plan, brief=_brief(), catalog=_catalog())

    rank_one = projected.chapters[0].vertical_candidates[0]
    assert rank_one.target_description is None
    assert rank_one.regions[0].role == "required"
    assert rank_one.regions[0].entity_id == "subject-1"


@pytest.mark.parametrize(
    ("entity_id", "event_relation", "error"),
    [
        ("missing-entity", "event_member", "unknown entity"),
        ("subject-1", "optional", "relation is not backed"),
    ],
)
def test_local_validation_rejects_unverifiable_region_lineage(
    entity_id: str, event_relation: str, error: str
) -> None:
    payload = _v2_plan().model_dump(mode="json")
    ref = payload["chapters"][0]["candidates"][0]["resolved_regions"][0][
        "entity_refs"
    ][0]
    ref["entity_id"] = entity_id
    ref["event_relation"] = event_relation
    plan = ClipCardFeaturePlanV2.model_validate(payload)

    with pytest.raises(ValueError, match=error):
        validate_plan_contract(
            plan,
            brief=_brief(),
            catalog=_catalog(),
            cards={ASSET_ID: _card()},
        )


def test_resolved_atomic_region_requires_one_entity() -> None:
    with pytest.raises(ValidationError, match="exactly one entity"):
        ResolvedFramingRegion(
            region_id="invalid",
            target_description="two entities incorrectly presented as atomic",
            kind="subject",
            constraint_role="hard_core",
            composition="atomic",
            entity_refs=[
                ResolvedEntityRef(entity_id="subject-1", event_relation="required"),
                ResolvedEntityRef(entity_id="sign-1", event_relation="required"),
            ],
            observable_relation="Both are visible.",
        )


def test_v3_model_schema_contains_choices_not_projection_mirrors() -> None:
    schema = gemini_response_schema(ClipCardFeaturePlanV3)
    chapter = schema["$defs"]["ClipCardFeatureSelectV3"]["properties"]
    candidate = schema["$defs"]["ClipCardFeatureCandidateV3"]["properties"]

    assert "candidates" in chapter
    assert "horizontal_frame_id" not in chapter
    assert "vertical_frame_id" not in chapter
    assert "observed_visual_evidence" not in chapter
    assert "resolved_regions" not in candidate
    assert "horizontal_target_description" not in candidate
    assert "vertical_target_description" not in candidate
    assert "required_entity_ids" in candidate
    assert "preferred_entity_ids" in candidate
    assert "sacrificable_entity_ids" in candidate
    assert "framing_intent" in candidate


def test_v3_compact_card_omits_redundant_relation_expansion() -> None:
    compact = compact_card_v3(_card())
    event = compact["events"][0]

    assert "evidence" not in compact["entities"][0]
    assert "entity_relations" not in event
    assert "card_opportunities" not in event
    assert event["primary_entity_ids"] == ["subject-1"]
    assert event["required_entity_ids"] == ["subject-1", "sign-1"]
    assert event["grounding_target_entity_ids"] == ["subject-1"]


def test_v3_projects_local_descriptions_and_regions_from_selected_evidence() -> None:
    plan = _v3_plan()
    cards = {ASSET_ID: _card()}
    validate_plan_contract_v3(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        cards=cards,
    )
    evidence = build_selected_clip_card_evidence(plan, cards=cards)
    projected = project_feature_contracts_v3(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        selected_evidence=evidence,
    )

    chapter = projected.chapters[0]
    assert chapter.horizontal_frame_id == "RF000001"
    assert chapter.vertical_frame_id == "RF000002"
    assert chapter.vertical_strategy == "fit_with_background"
    assert [item.candidate_id for item in chapter.horizontal_candidates] == [
        "candidate-a",
        "candidate-b",
    ]
    assert [item.candidate_id for item in chapter.vertical_candidates] == [
        "candidate-b",
        "candidate-a",
    ]
    assert chapter.horizontal_candidates[1].target_description == "the visible subject"
    tracked = chapter.vertical_candidates[1]
    assert tracked.target_description == "the visible subject"
    assert [(region.entity_id, region.role) for region in tracked.regions] == [
        ("subject-1", "required"),
        ("sign-1", "preferred"),
    ]
    assert tracked.regions[0].target_description == "the visible subject"
    assert tracked.regions[1].target_description.startswith("foreground sign;")
    assert any(
        relation.startswith("editorial_framing_intent=")
        for relation in tracked.regions[0].observable_relations
    )


def test_v3_projection_is_reproducible_from_hash_bound_evidence(
    tmp_path: Path,
) -> None:
    plan = _v3_plan()
    evidence = build_selected_clip_card_evidence(plan, cards={ASSET_ID: _card()})
    evidence_path = tmp_path / "selected-evidence.json"
    write_json(evidence_path, evidence)
    expected = project_feature_contracts_v3(
        plan,
        brief=_brief(),
        catalog=_catalog(),
        selected_evidence=evidence,
    )

    _, reproduced = reproject_external_feature_plan_v3(
        source_plan=plan,
        catalog=_catalog(),
        brief=_brief(),
        source_artifacts={"selected_clip_card_evidence": evidence_path},
    )

    assert reproduced.model_dump(mode="json") == expected.model_dump(mode="json")
    assert SelectedClipCardEvidence.model_validate_json(
        evidence.model_dump_json()
    ) == evidence


def test_v3_rejects_unbacked_or_unclassified_entity_decisions() -> None:
    payload = _v3_plan().model_dump(mode="json")
    payload["chapters"][0]["candidates"][0]["required_entity_ids"] = [
        "missing-entity"
    ]
    plan = ClipCardFeaturePlanV3.model_validate(payload)
    with pytest.raises(ValueError, match="not backed by its event"):
        validate_plan_contract_v3(
            plan,
            brief=_brief(),
            catalog=_catalog(),
            cards={ASSET_ID: _card()},
        )

    payload = _v3_plan().model_dump(mode="json")
    payload["chapters"][0]["candidates"][0]["preferred_entity_ids"] = []
    plan = ClipCardFeaturePlanV3.model_validate(payload)
    with pytest.raises(ValueError, match="did not classify important event entities"):
        validate_plan_contract_v3(
            plan,
            brief=_brief(),
            catalog=_catalog(),
            cards={ASSET_ID: _card()},
        )
