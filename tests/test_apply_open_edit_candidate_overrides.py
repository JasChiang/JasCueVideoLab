from __future__ import annotations

from types import SimpleNamespace

from scripts.apply_open_edit_candidate_overrides import (
    CandidateOverride,
    CandidateOverridePatch,
    apply_candidate_overrides,
    reproject_external_feature_plan as reproject_override_v1,
    reproject_external_feature_plan_v2 as reproject_override_v2,
)
from scripts.plan_clip_card_open_edit import (
    OpenEditCandidate,
    OpenEditPlan,
    OpenEditShot,
    VerticalOverflowProposal,
    project_feature_contracts,
    reproject_external_feature_plan,
    reproject_external_feature_plan_v2,
)
import pytest
from jascue_video_lab.models import FramingRegionIntent, ModelProvenance


def candidate(candidate_id: str, frame_id: str) -> OpenEditCandidate:
    return OpenEditCandidate(
        candidate_id=candidate_id,
        source_asset_id="sha256:" + candidate_id,
        event_id="event",
        frame_id=frame_id,
        observed_visual_evidence="visible evidence",
        selection_reason="reason",
        quality_risks=[],
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_target_description=None,
        vertical_strategy="tracked_crop",
        vertical_target_description="visible subject",
        vertical_crop_mode="strict",
        confidence=0.9,
    )


def test_candidate_override_changes_only_requested_aspect() -> None:
    shots = []
    for index in range(10):
        role = "hook" if index == 0 else "closing" if index == 9 else "action"
        shots.append(
            OpenEditShot(
                feature_id=f"scene_{index}",
                title="scene",
                editorial_role=role,
                intended_effect="progress",
                target_duration_seconds=6,
                candidates=[
                    candidate(f"a{index}", f"RF{index * 2 + 1:06d}"),
                    candidate(f"b{index}", f"RF{index * 2 + 2:06d}"),
                ],
                horizontal_candidate_id=f"a{index}",
                vertical_candidate_id=f"a{index}",
            )
        )
    plan = OpenEditPlan(
        project_id="project",
        catalog_id="catalog",
        inferred_title="title",
        inferred_theme="theme",
        intended_audience_hypothesis="audience",
        story_arc="arc",
        shots=shots,
        excluded_patterns=[],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id="gemini-3.5-flash",
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    patch = CandidateOverridePatch(
        interpretation="human_reviewed_candidate_replacement",
        overrides=[
            CandidateOverride(
                feature_id="scene_1",
                aspect="vertical",
                candidate_id="b1",
                reason="better portrait composition",
            )
        ],
    )

    revised = apply_candidate_overrides(plan, patch)
    assert revised.shots[1].horizontal_candidate_id == "a1"
    assert revised.shots[1].vertical_candidate_id == "b1"
    assert plan.shots[1].vertical_candidate_id == "a1"

    mismatch = SimpleNamespace(catalog_id="another-catalog")
    for projector in (reproject_override_v1, reproject_override_v2):
        with pytest.raises(ValueError, match="differs from projection catalog"):
            projector(
                source_plan=revised,
                catalog=mismatch,
                brief=object(),
                source_artifacts={},
            )


def test_open_edit_projection_entrypoints_keep_v1_and_v2_semantics() -> None:
    shots = []
    for index in range(10):
        role = "hook" if index == 0 else "closing" if index == 9 else "action"
        shots.append(
            OpenEditShot(
                feature_id=f"scene_{index}",
                title="scene",
                editorial_role=role,
                intended_effect="progress",
                target_duration_seconds=6,
                candidates=[
                    candidate(f"a{index}", f"RF{index * 2 + 1:06d}"),
                    candidate(f"b{index}", f"RF{index * 2 + 2:06d}"),
                ],
                horizontal_candidate_id=f"a{index}",
                vertical_candidate_id=f"b{index}",
            )
        )
    plan = OpenEditPlan(
        project_id="project",
        catalog_id="catalog",
        inferred_title="title",
        inferred_theme="theme",
        intended_audience_hypothesis="audience",
        story_arc="arc",
        shots=shots,
        excluded_patterns=[],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id="gemini-3.6-flash",
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )
    catalog = SimpleNamespace(catalog_id="catalog")

    _, legacy = reproject_external_feature_plan(
        source_plan=plan,
        catalog=catalog,  # type: ignore[arg-type]
        brief=object(),  # type: ignore[arg-type]
        source_artifacts={},
    )
    _, current = reproject_external_feature_plan_v2(
        source_plan=plan,
        catalog=catalog,  # type: ignore[arg-type]
        brief=object(),  # type: ignore[arg-type]
        source_artifacts={},
    )

    assert legacy.chapters[0].horizontal_candidates == []
    assert legacy.chapters[0].vertical_candidates == []
    assert [item.candidate_id for item in current.chapters[0].horizontal_candidates] == [
        "a0",
        "b0",
    ]
    assert [item.candidate_id for item in current.chapters[0].vertical_candidates] == [
        "b0",
        "a0",
    ]


def test_region_only_candidate_projects_deterministic_composite_target() -> None:
    shots = []
    for index in range(10):
        item = candidate(f"a{index}", f"RF{index + 1:06d}")
        if index == 0:
            item_payload = item.model_dump(mode="json")
            item_payload.update(
                {
                    "vertical_target_description": None,
                    "vertical_regions": [
                        FramingRegionIntent(
                            region_id="region_z",
                            target_description="the complete visible information panel",
                            kind="text_region",
                        ).model_dump(mode="json"),
                        FramingRegionIntent(
                            region_id="region_a",
                            target_description="the foreground participant",
                            kind="subject",
                        ).model_dump(mode="json"),
                    ],
                }
            )
            item = OpenEditCandidate.model_validate(item_payload)
        role = "hook" if index == 0 else "closing" if index == 9 else "action"
        shots.append(
            OpenEditShot(
                feature_id=f"scene_{index}",
                title="scene",
                editorial_role=role,
                intended_effect="progress",
                target_duration_seconds=6,
                candidates=[
                    item,
                    candidate(f"b{index}", f"RF{index + 11:06d}"),
                ],
                horizontal_candidate_id=f"a{index}",
                vertical_candidate_id=f"a{index}",
            )
        )
    plan = OpenEditPlan(
        project_id="project",
        catalog_id="catalog",
        inferred_title="title",
        inferred_theme="theme",
        intended_audience_hypothesis="audience",
        story_arc="arc",
        shots=shots,
        excluded_patterns=[],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id="gemini-3.5-flash",
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    brief, feature_plan, _ = project_feature_contracts(plan)

    expected = (
        "Preserve the union of all required framing regions: "
        "region_id=region_a, kind=subject, target=the foreground participant; "
        "region_id=region_z, kind=text_region, "
        "target=the complete visible information panel"
    )
    assert feature_plan.chapters[0].vertical_target_description == expected
    assert brief.chapters[0].vertical_primary_target_description == expected
    assert [region.region_id for region in brief.chapters[0].vertical_regions] == [
        "region_z",
        "region_a",
    ]


def test_model_candidate_can_only_propose_required_region_clipping() -> None:
    with pytest.raises(ValueError, match="vertical_overflow_policy"):
        candidate("unsafe", "RF000001").model_copy(
            update={"vertical_overflow_policy": "controlled_clip"}
        ).model_validate(
            {
                **candidate("unsafe", "RF000001").model_dump(mode="json"),
                "vertical_overflow_policy": "controlled_clip",
            }
        )

    proposed = candidate("a0", "RF000001").model_copy(
        update={
            "vertical_overflow_proposal": VerticalOverflowProposal(
                proposed_policy="controlled_clip",
                proposed_edge_priority="preserve_end",
                rationale="one peripheral region may be sacrificed after human review",
            )
        }
    )
    shots = []
    for index in range(10):
        role = "hook" if index == 0 else "closing" if index == 9 else "action"
        first = proposed if index == 0 else candidate(f"a{index}", f"RF{index + 1:06d}")
        shots.append(
            OpenEditShot(
                feature_id=f"scene_{index}",
                title="scene",
                editorial_role=role,
                intended_effect="progress",
                target_duration_seconds=6,
                candidates=[first, candidate(f"b{index}", f"RF{index + 11:06d}")],
                horizontal_candidate_id=f"a{index}",
                vertical_candidate_id=f"a{index}",
            )
        )
    plan = OpenEditPlan(
        project_id="project",
        catalog_id="catalog",
        inferred_title="title",
        inferred_theme="theme",
        intended_audience_hypothesis="audience",
        story_arc="arc",
        shots=shots,
        excluded_patterns=[],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id="gemini-3.5-flash",
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test",
            generated_at="test",
        ),
    )

    brief, feature_plan, audit = project_feature_contracts(plan)
    assert brief.chapters[0].vertical_overflow_policy == "preserve_all"
    assert brief.chapters[0].vertical_edge_priority == "balanced"
    assert "model_proposed_controlled_clip_requires_human_policy" in (
        feature_plan.chapters[0].quality_risks
    )
    assert audit["chapters"][0]["model_vertical_overflow_proposal"][
        "proposed_edge_priority"
    ] == "preserve_end"
