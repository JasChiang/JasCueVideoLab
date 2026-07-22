from __future__ import annotations

from pathlib import Path

import pytest

from jascue_video_lab.media import sha256_file
from jascue_video_lab.feature_cut import run_feature_cut_experiment
from jascue_video_lab.models import (
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    ModelProvenance,
    ReframePolicyBinding,
    RushesCatalog,
)
from jascue_video_lab.reframe_policy import (
    ReframePolicyPatch,
    apply_policy,
    build_brief_policy_binding,
    build_policy_sidecar,
    build_reused_plan_binding,
    validate_reframe_policy_bundle,
    write_immutable_policy_sidecar,
)
from jascue_video_lab.storage import read_json, write_json


def _brief() -> FeatureEditBrief:
    return FeatureEditBrief(
        project_id="generic-project",
        title="Generic edit",
        target_duration_seconds=60,
        chapters=[
            FeatureChapterBrief(
                feature_id="scene_a",
                title="Scene A",
                detail_lines=[],
                target_duration_seconds=3,
            )
        ],
    )


def _binding() -> ReframePolicyBinding:
    digest = "a" * 64
    return ReframePolicyBinding(
        binding_version="human-reframe-policy-binding-v1",
        policy_id="review-1",
        reviewer="reviewer",
        sidecar_path="/artifact/reframe-policy-sidecars/policy.json",
        sidecar_sha256=digest,
        source_brief_path="/source/brief.json",
        source_brief_sha256=digest,
        source_feature_plan_path="/source/feature-plan.json",
        source_feature_plan_sha256=digest,
        source_plan_binding_path="/source/feature-plan.binding.json",
        source_plan_binding_sha256=digest,
        catalog_path="/source/catalog.json",
        catalog_sha256=digest,
        selection_fingerprint=digest,
    )


def _policy(feature_id: str = "scene_a") -> ReframePolicyPatch:
    return ReframePolicyPatch.model_validate(
        {
            "policy_id": "review-1",
            "interpretation": "human_reviewed_reframe_policy_not_ground_truth",
            "chapters": [
                {
                    "feature_id": feature_id,
                    "decision_reason": "reviewer accepts preserving the trailing text edge",
                    "vertical_regions": [
                        {
                            "region_id": "heading",
                            "target_description": "the complete heading on the sign",
                            "kind": "text_region",
                            "role": "required",
                        }
                    ],
                    "vertical_overflow_policy": "controlled_clip",
                    "vertical_edge_priority": "preserve_end",
                }
            ],
        }
    )


def test_apply_reframe_policy_requires_binding_for_controlled_clip() -> None:
    with pytest.raises(ValueError, match="immutable human reframe policy binding"):
        apply_policy(_brief(), _policy())

    revised = apply_policy(_brief(), _policy(), binding=_binding())
    chapter = revised.chapters[0]
    assert chapter.vertical_regions[0].kind == "text_region"
    assert chapter.vertical_edge_priority == "preserve_end"
    assert revised.reframe_policy_binding == _binding()


def test_apply_reframe_policy_rejects_unknown_chapter() -> None:
    with pytest.raises(ValueError, match="unknown feature"):
        apply_policy(_brief(), _policy("missing"), binding=_binding())


def _write_source_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    catalog_path = tmp_path / "source" / "catalog.json"
    brief_path = tmp_path / "source" / "brief.json"
    plan_path = tmp_path / "source" / "gemini-plan" / "feature_edit_plan.json"
    binding_path = plan_path.parent / "feature-plan.binding.json"
    request_path = plan_path.parent / "feature_edit_plan.request.json"
    catalog = RushesCatalog(
        catalog_id="catalog-1",
        source_directory="/source",
        sample_interval_ms=2000,
        total_duration_ms=60000,
        clips=[],
        frames=[],
        analysis_reel_path="/source/reel.mp4",
        generated_at="test",
    )
    plan = FeatureEditPlan(
        project_id="generic-project",
        catalog_id="catalog-1",
        title="Generic edit",
        chapters=[
            FeatureChapterSelect(
                feature_id="scene_a",
                evidence_status="supported",
                horizontal_frame_id="RF000001",
                vertical_frame_id="RF000001",
                observed_visual_evidence="visible subject",
                selection_reason="selected",
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_strategy="tracked_crop",
                vertical_target_description="visible subject",
                quality_risks=[],
                confidence=0.9,
            )
        ],
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
    write_json(catalog_path, catalog)
    write_json(brief_path, _brief())
    write_json(plan_path, plan)
    write_json(request_path, {"request": "immutable"})
    write_json(
        binding_path,
        {
            "binding_version": "feature-plan-binding-v1",
            "origin": "generated",
            "catalog_path": str(catalog_path.resolve()),
            "catalog_sha256": sha256_file(catalog_path),
            "brief_path": str(brief_path.resolve()),
            "brief_sha256": sha256_file(brief_path),
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": sha256_file(plan_path),
            "plan_prompt_sha256": "1" * 64,
            "system_instruction_sha256": "2" * 64,
            "model_id": "gemini-3.5-flash",
            "model_id_sha256": "3" * 64,
            "response_schema_sha256": "4" * 64,
            "request_path": str(request_path.resolve()),
            "request_sha256": sha256_file(request_path),
            "created_at": "test",
        },
    )
    return catalog_path, brief_path, plan_path, binding_path


def test_policy_bundle_binds_selection_and_detects_sidecar_tampering(
    tmp_path: Path,
) -> None:
    catalog_path, source_brief_path, source_plan_path, source_binding_path = (
        _write_source_bundle(tmp_path)
    )
    output_dir = tmp_path / "reviewed"
    sidecar = build_policy_sidecar(
        policy=_policy(),
        reviewer="human-reviewer",
        review_note="reviewed the required region tradeoff",
        catalog_path=catalog_path,
        source_brief_path=source_brief_path,
        source_feature_plan_path=source_plan_path,
        source_plan_binding_path=source_binding_path,
    )
    sidecar_path, sidecar_sha256 = write_immutable_policy_sidecar(output_dir, sidecar)
    policy_binding = build_brief_policy_binding(
        sidecar=sidecar,
        sidecar_path=sidecar_path,
        sidecar_sha256=sidecar_sha256,
    )
    output_brief_path = output_dir / "brief.json"
    output_plan_path = output_dir / "gemini-plan" / "feature_edit_plan.json"
    write_json(
        output_brief_path,
        apply_policy(_brief(), _policy(), binding=policy_binding),
    )
    source_plan = FeatureEditPlan.model_validate(read_json(source_plan_path))
    write_json(output_plan_path, source_plan)
    source_binding = read_json(source_binding_path)
    saved_binding = build_reused_plan_binding(
        catalog_path=catalog_path,
        brief_path=output_brief_path,
        feature_plan_path=output_plan_path,
        source_plan_binding=source_binding,
        policy_binding=policy_binding,
    )

    validated = validate_reframe_policy_bundle(
        catalog_path=catalog_path,
        brief_path=output_brief_path,
        feature_plan_path=output_plan_path,
        saved_plan_binding=saved_binding,
    )
    assert validated["selection_fingerprint"] == sidecar.selection_fingerprint

    tampered = read_json(sidecar_path)
    tampered["review_note"] = "changed after approval"
    write_json(sidecar_path, tampered)
    with pytest.raises(ValueError, match="sidecar hash is invalid"):
        validate_reframe_policy_bundle(
            catalog_path=catalog_path,
            brief_path=output_brief_path,
            feature_plan_path=output_plan_path,
            saved_plan_binding=saved_binding,
        )


def test_renderer_rejects_unbound_controlled_clip_before_media_or_model_work(
    tmp_path: Path,
) -> None:
    catalog_path, source_brief_path, _, _ = _write_source_bundle(tmp_path)
    unsafe = _brief().model_dump(mode="json")
    unsafe["chapters"][0]["vertical_overflow_policy"] = "controlled_clip"
    unsafe_path = tmp_path / "unsafe-brief.json"
    write_json(unsafe_path, unsafe)

    with pytest.raises(ValueError, match="immutable human reframe policy sidecar"):
        run_feature_cut_experiment(
            catalog_path=catalog_path,
            brief_path=unsafe_path,
            checkpoint_path=tmp_path / "missing-checkpoint.pt",
            output_dir=tmp_path / "render",
            plan_prompt="plan",
            grounding_prompt="ground",
        )


def test_renderer_rejects_policy_brief_with_non_policy_plan_origin(
    tmp_path: Path,
) -> None:
    catalog_path, _, source_plan_path, _ = _write_source_bundle(tmp_path)
    output_dir = tmp_path / "render"
    policy_brief_path = tmp_path / "policy-brief.json"
    write_json(
        policy_brief_path,
        apply_policy(_brief(), _policy(), binding=_binding()),
    )
    write_json(
        output_dir / "gemini-plan" / "feature_edit_plan.json",
        read_json(source_plan_path),
    )
    write_json(
        output_dir / "gemini-plan" / "feature-plan.binding.json",
        {"binding_version": "feature-plan-binding-v1", "origin": "generated"},
    )

    with pytest.raises(ValueError, match="human_reframe_policy plan binding"):
        run_feature_cut_experiment(
            catalog_path=catalog_path,
            brief_path=policy_brief_path,
            checkpoint_path=tmp_path / "missing-checkpoint.pt",
            output_dir=output_dir,
            plan_prompt="plan",
            grounding_prompt="ground",
            reuse_feature_plan=True,
        )
