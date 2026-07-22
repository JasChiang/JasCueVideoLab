from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from scripts.plan_clip_card_open_edit import (
    OpenEditCandidate,
    OpenEditPlan,
    OpenEditShot,
    project_feature_contracts,
)

from jascue_video_lab.feature_cut import (
    _chapter_bounds_with_approved_trim,
    _cover_transform,
    _current_feature_plan_binding,
    _current_external_projection_binding,
    _has_complete_cached_primary_track,
    _horizontal_filter_from_track,
    _is_exhausted_model_quota_error,
    _is_non_retryable_spending_cap_error,
    _load_trim_decisions,
    _migrate_legacy_feature_plan_binding,
    _piecewise_expression,
    _concat_segments,
    _render_source_segment,
    _render_text_layer,
    _required_track_union,
    _resolve_vertical_candidate_intent,
    _segment_variant_fingerprint,
    _soft_extent_visibility_audit,
    _summarize_automatic_reframe,
    _tracking_seed_request_ms,
    _tracked_crop_geometry,
    _usable_track_centers,
    _validate_feature_plan_binding,
    _validate_shared_sam_session_cache,
    _vertical_crop_geometry,
    _vertical_center_crop_filter,
    _vertical_filter_from_track,
    _vertical_fit_filter,
    _vertical_runtime_candidate_options,
    _vertical_target_fits_crop,
    _write_incremental_pricing,
    write_external_feature_plan_projection,
)
from jascue_video_lab.gemini import (
    EDITORIAL_SYSTEM_INSTRUCTION,
    MODEL_ID,
    VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
)
from jascue_video_lab.models import (
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    FeatureVerticalCandidate,
    FramingRegionIntent,
    ModelProvenance,
    TrimIntentDecision,
    RushClip,
    RushFrame,
    RushesCatalog,
    SegmentationModelProvenance,
    SegmentationSample,
    SegmentationTrack,
    SemanticIdentityStatus,
    SharedSam21AnalysisFrame,
    SharedSam21BBoxSeed,
    SharedSam21SessionManifest,
    SharedSam21SessionTarget,
    SharedSam21SessionTiming,
    TrackingState,
)
from jascue_video_lab.media import sha256_file
from jascue_video_lab.sam_tracking import (
    SAM21_IMPLEMENTATION_REVISION,
    SAM21_TINY_MODEL_ID,
    pad_normalized_box,
)
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.shots import ShotManifest, ShotSegment
from jascue_video_lab.storage import read_json, write_json


def test_feature_plan_binding_rejects_changed_causal_inputs(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    brief = tmp_path / "brief.json"
    plan = tmp_path / "plan.json"
    request = tmp_path / "request.json"
    catalog.write_text('{"catalog":"original"}\n', encoding="utf-8")
    brief.write_text('{"brief":"original"}\n', encoding="utf-8")
    plan.write_text('{"plan":"original"}\n', encoding="utf-8")
    request.write_text('{"request":"original"}\n', encoding="utf-8")

    saved = _current_feature_plan_binding(
        catalog_path=catalog,
        brief_path=brief,
        plan_path=plan,
        plan_prompt="generic editorial prompt",
        request_path=request,
        created_at="2026-01-01T00:00:00+00:00",
        origin="generated",
    )
    current = _current_feature_plan_binding(
        catalog_path=catalog,
        brief_path=brief,
        plan_path=plan,
        plan_prompt="generic editorial prompt",
        request_path=request,
        created_at="2026-01-02T00:00:00+00:00",
        origin="generated",
    )
    _validate_feature_plan_binding(saved, current)
    causal_hashes = {
        "catalog_sha256",
        "brief_sha256",
        "plan_prompt_sha256",
        "system_instruction_sha256",
        "model_id_sha256",
        "response_schema_sha256",
        "plan_sha256",
        "request_sha256",
    }
    assert causal_hashes <= saved.keys()

    for key in causal_hashes:
        changed = dict(current)
        changed[key] = "0" * 64
        with pytest.raises(ValueError, match=key):
            _validate_feature_plan_binding(saved, changed)


def test_legacy_feature_plan_reuse_migrates_without_overwriting_evidence(
    tmp_path: Path,
) -> None:
    plan_dir = tmp_path / "gemini-plan"
    plan_dir.mkdir()
    catalog = tmp_path / "catalog.json"
    brief = tmp_path / "brief.json"
    plan = plan_dir / "feature_edit_plan.json"
    prompt = "Use direct evidence to select reusable footage."
    catalog.write_text('{"catalog":"v1"}\n', encoding="utf-8")
    brief.write_text('{"brief":"v1"}\n', encoding="utf-8")
    plan.write_text('{"plan":"v1"}\n', encoding="utf-8")
    request = {
        "model": MODEL_ID,
        "system_instruction": EDITORIAL_SYSTEM_INSTRUCTION,
        "input": [
            {"type": "video", "uri": "files/example", "mime_type": "video/mp4"},
            {
                "type": "text",
                "text": prompt + "\n\n## 本次不可變 metadata\nproject_id: test",
            },
        ],
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": gemini_response_schema(FeatureEditPlan),
        },
    }
    write_json(plan_dir / "feature_edit_plan.request.json", request)
    legacy = {
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "current_catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "current_brief_sha256": hashlib.sha256(brief.read_bytes()).hexdigest(),
        "current_plan_prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "model_id": MODEL_ID,
        # This was the known legacy bug; the original request proves the
        # actual editorial instruction before migration is accepted.
        "system_instruction_sha256": hashlib.sha256(
            VISUAL_EVIDENCE_SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
    }
    legacy_path = plan_dir / "feature-plan.reuse.json"
    write_json(legacy_path, legacy)
    original_legacy_bytes = legacy_path.read_bytes()

    migrated = _migrate_legacy_feature_plan_binding(
        plan_dir=plan_dir,
        catalog_path=catalog,
        brief_path=brief,
        plan_path=plan,
        plan_prompt=prompt,
    )

    assert migrated["origin"] == "migrated_legacy_reuse"
    assert migrated["system_instruction_sha256"] == hashlib.sha256(
        EDITORIAL_SYSTEM_INSTRUCTION.encode("utf-8")
    ).hexdigest()
    assert legacy_path.read_bytes() == original_legacy_bytes

    legacy["current_catalog_sha256"] = "f" * 64
    write_json(legacy_path, legacy)
    with pytest.raises(ValueError, match="current_catalog_sha256"):
        _migrate_legacy_feature_plan_binding(
            plan_dir=plan_dir,
            catalog_path=catalog,
            brief_path=brief,
            plan_path=plan,
            plan_prompt=prompt,
        )


@pytest.mark.parametrize(
    ("projection_contract_id", "preserve_runtime_candidates"),
    [
        ("clip-card-open-edit-v1", False),
        ("clip-card-open-edit-v2", True),
    ],
)
def test_external_projection_binding_verifies_source_request_plan_and_artifacts(
    tmp_path: Path,
    projection_contract_id: str,
    preserve_runtime_candidates: bool,
) -> None:
    raw_provenance = ModelProvenance(
        model_id=MODEL_ID,
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version="test",
        run_id="run-test",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    source_provenance = raw_provenance.model_copy(
        update={"interaction_id": "interaction-test"}
    )
    catalog = RushesCatalog(
        catalog_id="catalog-test",
        source_directory=str(tmp_path / "sources"),
        sample_interval_ms=1000,
        total_duration_ms=20000,
        clips=[
            RushClip(
                clip_id="clip-1",
                path=str(tmp_path / "source.mp4"),
                sha256="a" * 64,
                duration_ms=20000,
                width=1920,
                height=1080,
                frame_rate="30/1",
                size_bytes=1,
            )
        ],
        frames=[
            RushFrame(
                frame_id=f"RF{index:06d}",
                clip_id="clip-1",
                requested_time_ms=(index - 1) * 500,
                image_path=str(tmp_path / f"frame-{index}.jpg"),
            )
            for index in range(1, 21)
        ],
        analysis_reel_path=str(tmp_path / "reel.mp4"),
        generated_at="2026-01-01T00:00:00+00:00",
    )
    shots: list[OpenEditShot] = []
    for index in range(10):
        candidates = [
            OpenEditCandidate(
                candidate_id=f"candidate-{index}-{offset}",
                source_asset_id="sha256:" + "a" * 64,
                event_id=f"event-{index}",
                frame_id=f"RF{index * 2 + offset + 1:06d}",
                observed_visual_evidence="One directly visible subject.",
                selection_reason="Clear representative state.",
                quality_risks=[],
                horizontal_strategy="original",
                horizontal_zoom_intent="none",
                horizontal_target_description=None,
                vertical_strategy="fit_with_background",
                vertical_target_description=None,
                vertical_crop_mode="strict",
                confidence=0.8,
            )
            for offset in range(2)
        ]
        shots.append(
            OpenEditShot(
                feature_id=f"scene_{index}",
                title=f"Scene {index}",
                editorial_role=(
                    "hook" if index == 0 else "closing" if index == 9 else "action"
                ),
                intended_effect="Maintain visible narrative progression.",
                target_duration_seconds=6,
                candidates=candidates,
                horizontal_candidate_id=candidates[0].candidate_id,
                vertical_candidate_id=candidates[0].candidate_id,
            )
        )
    raw_source_plan = OpenEditPlan(
        project_id="project-test",
        catalog_id=catalog.catalog_id,
        inferred_title="Generic edit",
        inferred_theme="Observable sequence",
        intended_audience_hypothesis="General audience",
        story_arc="Opening, progression, closing",
        shots=shots,
        excluded_patterns=[],
        uncertainties=[],
        model_provenance=raw_provenance,
    )
    source_plan = raw_source_plan.model_copy(
        update={"model_provenance": source_provenance}
    )
    # Exercise both deterministic projection generations.  v1 predates
    # runtime Top-K candidates; v2 preserves them for automatic recovery.
    brief, feature_plan, _ = project_feature_contracts(
        source_plan,
        preserve_runtime_candidates=preserve_runtime_candidates,
    )
    catalog_path = tmp_path / "catalog.json"
    brief_path = tmp_path / "brief.json"
    plan_dir = tmp_path / "gemini-plan"
    feature_plan_path = plan_dir / "feature_edit_plan.json"
    source_plan_path = tmp_path / "source-plan.json"
    request_path = tmp_path / "source.request.json"
    raw_output_path = tmp_path / "source.raw_output.json"
    write_json(catalog_path, catalog)
    write_json(brief_path, brief)
    write_json(feature_plan_path, feature_plan)
    write_json(source_plan_path, source_plan)
    write_json(
        request_path,
        {
            "model": MODEL_ID,
            "system_instruction": "Use only the supplied media evidence.",
            "input": [{"type": "text", "text": "Select a coherent sequence."}],
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_response_schema(OpenEditPlan),
            },
        },
    )
    raw_interaction_path = tmp_path / "source.raw_interaction.json"
    write_json(raw_output_path, {"output_text": raw_source_plan.model_dump_json()})
    write_json(raw_interaction_path, {"id": "interaction-test"})

    invalid_request = read_json(request_path)
    invalid_request["response_format"]["schema"] = {"type": "object"}
    write_json(request_path, invalid_request)
    with pytest.raises(ValueError, match="registered model"):
        write_external_feature_plan_projection(
            plan_dir=plan_dir,
                projection_contract_id=projection_contract_id,
            catalog_path=catalog_path,
            brief_path=brief_path,
            feature_plan_path=feature_plan_path,
            source_plan_path=source_plan_path,
            source_request_path=request_path,
            source_artifacts={
                "source_raw_output": raw_output_path,
                "source_raw_interaction": raw_interaction_path,
            },
        )
    invalid_request["response_format"]["schema"] = gemini_response_schema(
        OpenEditPlan
    )
    write_json(request_path, invalid_request)

    fabricated_source = source_plan.model_copy(
        update={"inferred_title": "Fabricated source title"}
    )
    write_json(source_plan_path, fabricated_source)
    with pytest.raises(ValueError, match="raw model output"):
        write_external_feature_plan_projection(
            plan_dir=plan_dir,
                projection_contract_id=projection_contract_id,
            catalog_path=catalog_path,
            brief_path=brief_path,
            feature_plan_path=feature_plan_path,
            source_plan_path=source_plan_path,
            source_request_path=request_path,
            source_artifacts={
                "source_raw_output": raw_output_path,
                "source_raw_interaction": raw_interaction_path,
            },
        )
    write_json(source_plan_path, source_plan)

    fabricated_plan = feature_plan.model_copy(update={"title": "Fabricated title"})
    write_json(feature_plan_path, fabricated_plan)
    with pytest.raises(ValueError, match="deterministic projector"):
        write_external_feature_plan_projection(
            plan_dir=plan_dir,
                projection_contract_id=projection_contract_id,
            catalog_path=catalog_path,
            brief_path=brief_path,
            feature_plan_path=feature_plan_path,
            source_plan_path=source_plan_path,
            source_request_path=request_path,
            source_artifacts={
                "source_raw_output": raw_output_path,
                "source_raw_interaction": raw_interaction_path,
            },
        )
    write_json(feature_plan_path, feature_plan)

    pointer_path = write_external_feature_plan_projection(
        plan_dir=plan_dir,
        projection_contract_id=projection_contract_id,
        catalog_path=catalog_path,
        brief_path=brief_path,
        feature_plan_path=feature_plan_path,
        source_plan_path=source_plan_path,
        source_request_path=request_path,
        source_artifacts={
            "source_raw_output": raw_output_path,
            "source_raw_interaction": raw_interaction_path,
        },
    )
    binding = _current_external_projection_binding(
        plan_dir=plan_dir,
        catalog_path=catalog_path,
        brief_path=brief_path,
        plan_path=feature_plan_path,
        created_at="2026-01-02T00:00:00+00:00",
    )

    assert pointer_path.name == "feature-plan.external-projection.json"
    assert binding["origin"] == "external_projection"
    assert binding["external_projection_contract_id"] == projection_contract_id
    _validate_feature_plan_binding(binding, dict(binding))

    write_json(raw_output_path, {"output_text": '{"changed":true}'})
    with pytest.raises(ValueError, match="source artifact changed"):
        _current_external_projection_binding(
            plan_dir=plan_dir,
            catalog_path=catalog_path,
            brief_path=brief_path,
            plan_path=feature_plan_path,
            created_at="2026-01-02T00:00:00+00:00",
        )


def test_incremental_pricing_names_changed_error_artifacts_honestly(
    tmp_path: Path,
) -> None:
    write_json(
        tmp_path / "grounding" / "grounding.raw_interaction.json",
        {
            "model": MODEL_ID,
            "usage": {
                "total_input_tokens": 100,
                "total_output_tokens": 10,
                "total_thought_tokens": 2,
            }
        },
    )
    write_json(tmp_path / "grounding" / "errors.json", [{"message": "review"}])

    result = _write_incremental_pricing(
        output_dir=tmp_path,
        prior_interaction_hashes={},
        prior_error_hashes={},
    )

    assert result["request_count"] == 1
    assert result["changed_error_artifact_count"] == 1
    assert "failed_request_artifact_count" not in result
    assert (tmp_path / "pricing.incremental.json").exists()


def test_feature_brief_requires_unique_chapter_ids() -> None:
    with pytest.raises(ValidationError, match="unique"):
        FeatureEditBrief(
            project_id="test",
            title="test",
            target_duration_seconds=60,
            chapters=[
                FeatureChapterBrief(
                    feature_id="same",
                    title="one",
                    detail_lines=[],
                    target_duration_seconds=4,
                ),
                FeatureChapterBrief(
                    feature_id="same",
                    title="two",
                    detail_lines=[],
                    target_duration_seconds=4,
                ),
            ],
        )


def test_segment_cache_key_changes_with_source_or_tracking_geometry() -> None:
    base = {
        "source_sha256": "a" * 64,
        "start_ms": 1000,
        "end_ms": 3000,
        "filter_graph": "crop=1080:1920:x=100:y=0",
        "geometry": {"applied_strategy": "tracked_crop"},
        "track_fingerprint": "b" * 64,
    }
    original = _segment_variant_fingerprint(**base)
    assert original != _segment_variant_fingerprint(
        **{**base, "track_fingerprint": "c" * 64}
    )
    assert original != _segment_variant_fingerprint(
        **{**base, "source_sha256": "d" * 64}
    )


def test_vertical_crop_geometry_preserves_rendered_x_audit_keyframes() -> None:
    x_values, audit = _vertical_crop_geometry(
        [0.0, 1.0, 2.0],
        [200.0, 500.0, 900.0],
        [[100, 100, 300, 900], [400, 100, 600, 900], [800, 100, 1000, 900]],
    )

    assert len(x_values) == 3
    assert [item["crop_x_pixels"] for item in audit["crop_keyframes"]] == [
        round(value, 3) for value in x_values
    ]
    coordinate_space = audit["crop_coordinate_space"]
    assert coordinate_space["contract_version"] == "aspect-preserving-cover-v1"
    assert coordinate_space["orientation_basis"] == "ffmpeg_autorotated_display"
    assert coordinate_space["scale_policy"] == "aspect_preserving_cover"
    assert coordinate_space["source_display_width"] == 1920
    assert coordinate_space["source_display_height"] == 1080
    assert coordinate_space["scaled_width"] == 3414
    assert coordinate_space["scaled_height"] == 1920
    assert coordinate_space["crop_width"] == 1080
    assert coordinate_space["crop_height"] == 1920
    assert coordinate_space["active_pan_axes"] == ["x"]
    assert audit["crop_width_normalized"] == pytest.approx(316.3445)
    assert audit["max_target_width_normalized"] == 200
    assert x_values == sorted(x_values)


def test_soft_extent_visibility_is_measured_without_relaxing_hard_containment() -> None:
    _, _, crop_audit = _tracked_crop_geometry(
        [0.0, 0.5],
        [500.0, 500.0],
        [[450, 200, 550, 800], [450, 200, 550, 800]],
        source_width=1920,
        source_height=1080,
        output_width=1080,
        output_height=1920,
    )
    soft_track = SimpleNamespace(
        analysis_start_ms=0,
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=time_ms,
                tracking_state=TrackingState.TRACKED,
                derived_tracking_box=[0, 200, 400, 800],
            )
            for time_ms in (0, 500)
        ],
    )
    permissive = FramingRegionIntent(
        region_id="context",
        target_description="visible surrounding context",
        role="preferred",
        minimum_visible_fraction=0.1,
    )
    strict = permissive.model_copy(update={"minimum_visible_fraction": 0.9})

    accepted = _soft_extent_visibility_audit(
        tracks=[soft_track],  # type: ignore[list-item]
        regions=[permissive],
        crop_audit=crop_audit,
    )
    rejected = _soft_extent_visibility_audit(
        tracks=[soft_track],  # type: ignore[list-item]
        regions=[strict],
        crop_audit=crop_audit,
    )

    assert crop_audit["containment_failure_count"] == 0
    assert accepted["soft_extent_visibility_passed"] is True
    assert rejected["soft_extent_visibility_passed"] is False
    assert rejected["soft_extent_regions"][0]["minimum_visible_area_fraction"] < 0.9


def test_ranked_candidate_intent_is_not_overridden_by_generic_brief_target() -> None:
    regions, target = _resolve_vertical_candidate_intent(
        option_regions=[],
        option_target_description="the rightmost visible instance beside the sign",
        selected_target_description="the leftmost selected instance",
        brief_primary_target_description="the main object",
        brief_regions=[],
        inherit_reviewed_brief_intent=False,
    )

    assert regions == []
    assert target == "the rightmost visible instance beside the sign"


def test_legacy_or_human_candidate_can_inherit_reviewed_brief_regions() -> None:
    reviewed = FramingRegionIntent(
        region_id="reviewed-core",
        target_description="reviewed visible core",
        role="required",
    )

    regions, target = _resolve_vertical_candidate_intent(
        option_regions=[],
        option_target_description=None,
        selected_target_description=None,
        brief_primary_target_description="reviewed target",
        brief_regions=[reviewed],
        inherit_reviewed_brief_intent=True,
    )

    assert regions == [reviewed]
    assert target == "reviewed visible core"


def test_runtime_candidates_preserve_rank_but_human_binding_disables_switching() -> None:
    candidates = [
        FeatureVerticalCandidate(
            candidate_id=f"take-{rank}",
            rank=rank,
            source_asset_id="sha256:" + ("a" if rank == 1 else "b") * 64,
            event_id=f"event-{rank}",
            frame_id=f"RF{rank:06d}",
            observed_visual_evidence=f"Visible evidence {rank}",
            selection_reason=f"Reason {rank}",
            strategy="fit_with_background",
            target_description=None,
            confidence=0.8,
        )
        for rank in (1, 2)
    ]
    selected = FeatureChapterSelect(
        feature_id="scene",
        evidence_status="supported",
        horizontal_frame_id="RF000001",
        vertical_frame_id="RF000001",
        observed_visual_evidence="Visible evidence 1",
        selection_reason="Reason 1",
        horizontal_strategy="original",
        horizontal_zoom_intent="none",
        horizontal_target_description=None,
        vertical_strategy="fit_with_background",
        vertical_target_description=None,
        quality_risks=[],
        confidence=0.8,
        vertical_candidates=candidates,
    )

    automatic = _vertical_runtime_candidate_options(
        selected, human_policy_binding_present=False
    )
    reviewed = _vertical_runtime_candidate_options(
        selected, human_policy_binding_present=True
    )

    assert [item["candidate_id"] for item in automatic] == ["take-1", "take-2"]
    assert reviewed[0]["candidate_id"] == "legacy-primary"
    assert reviewed[0]["frame_id"] == "RF000001"
    assert reviewed[0]["target_description"] is None


@pytest.mark.parametrize(
    (
        "source_dimensions",
        "output_dimensions",
        "expected_scaled_dimensions",
        "expected_pan_axes",
    ),
    [
        ((1440, 1080), (1080, 1920), (2560, 1920), ["x"]),
        ((1080, 2400), (1080, 1920), (1080, 2400), ["y"]),
        ((1080, 1920), (1080, 1920), (1080, 1920), []),
        ((1440, 1080), (1920, 1080), (1920, 1440), ["y"]),
        ((2560, 1080), (1920, 1080), (2560, 1080), ["x"]),
    ],
)
def test_cover_transform_preserves_source_aspect_for_general_source_shapes(
    source_dimensions: tuple[int, int],
    output_dimensions: tuple[int, int],
    expected_scaled_dimensions: tuple[int, int],
    expected_pan_axes: list[str],
) -> None:
    transform = _cover_transform(*source_dimensions, *output_dimensions)

    assert (transform["scaled_width"], transform["scaled_height"]) == (
        expected_scaled_dimensions
    )
    assert transform["active_pan_axes"] == expected_pan_axes
    assert transform["aspect_ratio_relative_error"] < 0.001
    assert transform["normalized_track_space"] == (
        "orientation_corrected_source_0_1000"
    )


def test_portrait_source_uses_y_crop_and_preserves_required_region() -> None:
    boxes = [[400, 650, 600, 850], [400, 100, 600, 300]]
    x_values, y_values, audit = _tracked_crop_geometry(
        [0.0, 1.0],
        [500.0, 500.0],
        boxes,
        source_width=1080,
        source_height=2400,
        output_width=1080,
        output_height=1920,
    )

    assert x_values == [0.0, 0.0]
    assert y_values[0] > y_values[1]
    assert audit["crop_coordinate_space"]["active_pan_axes"] == ["y"]
    assert audit["crop_width_normalized"] == 1000
    assert audit["crop_height_normalized"] == 800
    assert audit["containment_failure_count"] == 0
    assert all(
        keyframe["required_union_contained"]
        for keyframe in audit["crop_keyframes"]
    )


def test_four_by_three_source_uses_x_crop_without_stretching() -> None:
    boxes = [[80, 200, 280, 800], [720, 200, 920, 800]]
    x_values, y_values, audit = _tracked_crop_geometry(
        [0.0, 1.0],
        [180.0, 820.0],
        boxes,
        source_width=1440,
        source_height=1080,
        output_width=1080,
        output_height=1920,
    )

    assert x_values[0] < x_values[1]
    assert y_values == [0.0, 0.0]
    coordinate_space = audit["crop_coordinate_space"]
    assert (coordinate_space["scaled_width"], coordinate_space["scaled_height"]) == (
        2560,
        1920,
    )
    assert coordinate_space["active_pan_axes"] == ["x"]
    assert coordinate_space["aspect_ratio_relative_error"] == 0
    assert audit["containment_failure_count"] == 0


def test_preserve_all_rejects_required_region_too_tall_for_viewport() -> None:
    _, _, audit = _tracked_crop_geometry(
        [0.0, 1.0],
        [500.0, 500.0],
        [[400, 50, 600, 950], [400, 50, 600, 950]],
        source_width=1080,
        source_height=2400,
        output_width=1080,
        output_height=1920,
    )

    assert audit["crop_height_normalized"] == 800
    assert audit["full_containment_feasible"] is False
    assert audit["geometry_feasible"] is False
    assert audit["containment_failure_count"] == 2


def test_projected_vertical_crop_contains_fast_moving_required_region() -> None:
    boxes = [
        [80, 100, 220, 900],
        [720, 100, 860, 900],
        [120, 100, 260, 900],
    ]
    x_values, audit = _vertical_crop_geometry(
        [0.0, 0.5, 1.0],
        [150.0, 790.0, 190.0],
        boxes,
        safety_multiplier=1.08,
    )

    assert len(x_values) == len(boxes)
    assert audit["geometry_feasible"] is True
    assert audit["full_containment_feasible"] is True
    assert audit["containment_failure_count"] == 0
    assert all(
        keyframe["required_union_contained"]
        for keyframe in audit["crop_keyframes"]
    )


def test_vertical_crop_audit_flags_source_boundary_contact() -> None:
    _, audit = _vertical_crop_geometry(
        [0.0, 0.5],
        [500.0, 500.0],
        [[100, 0, 300, 900], [700, 100, 1000, 900]],
        overflow_policy="controlled_clip",
    )

    assert audit["source_boundary_contact_count"] == 2
    assert audit["source_x_edge_contact_count"] == 1
    assert audit["source_y_edge_contact_count"] == 1
    assert audit["source_boundary_contact_ratio"] == 1.0


def test_cached_primary_track_requires_both_grounding_and_track(tmp_path: Path) -> None:
    root = tmp_path / "primary"
    assert _has_complete_cached_primary_track(root) is False
    grounding = root / "grounding" / "bbox-a" / "grounding.json"
    grounding.parent.mkdir(parents=True)
    grounding.write_text("{}", encoding="utf-8")
    assert _has_complete_cached_primary_track(root) is False
    track = root / "sam21" / "bbox-b" / "segmentation-track.json"
    track.parent.mkdir(parents=True)
    track.write_text("{}", encoding="utf-8")
    assert _has_complete_cached_primary_track(root) is True


def test_shared_sam_cache_revalidates_hashed_track_and_frame_lineage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    session_dir = tmp_path / "session"
    frames_dir = session_dir / "analysis-frames"
    frames_dir.mkdir(parents=True)
    frame_records: list[SharedSam21AnalysisFrame] = []
    for index, time_ms in enumerate((0, 500)):
        path = frames_dir / f"{index:06d}.jpg"
        path.write_bytes(f"frame-{index}".encode())
        frame_records.append(
            SharedSam21AnalysisFrame(
                sample_index=index,
                analysis_sample_time_ms=time_ms,
                source_pts=time_ms,
                path=f"analysis-frames/{path.name}",
                sha256=sha256_file(path),
            )
        )
    frames_manifest_path = session_dir / "analysis-frames-manifest.json"
    write_json(
        frames_manifest_path,
        {
            "timing_basis": "decoded_source_pts",
            "frames": [frame.model_dump(mode="json") for frame in frame_records],
        },
    )
    frames_manifest_sha256 = sha256_file(frames_manifest_path)
    checkpoint_sha256 = "c" * 64
    provenance = SegmentationModelProvenance(
        model_id=SAM21_TINY_MODEL_ID,
        implementation="facebookresearch/sam2",
        implementation_revision=SAM21_IMPLEMENTATION_REVISION,
        checkpoint_sha256=checkpoint_sha256,
        device="cpu",
        torch_version="test",
        generated_at="2026-07-22T00:00:00Z",
    )
    seeds = [
        SharedSam21BBoxSeed(
            target_id=f"region-{index}",
            target_description=f"required region {index}",
            seed_source=f"grounding-{index}.json",
            seed_time_ms=0,
            seed_frame_pts=0,
            seed_frame_sha256=str(index) * 64,
            seed_source_width=1920,
            seed_source_height=1080,
            seed_box_2d=box,
        )
        for index, box in enumerate(
            ([100, 100, 300, 800], [600, 100, 800, 800]), start=1
        )
    ]
    members: list[SharedSam21SessionTarget] = []
    for seed in seeds:
        samples = [
            SegmentationSample(
                sample_index=index,
                analysis_sample_time_ms=time_ms,
                source_pts=time_ms,
                timing_basis="decoded_source_pts",
                mask_path=f"masks/{index:06d}.png",
                mask_sha256="d" * 64,
                mask_area_pixels=100,
                mask_area_ratio=0.01,
                connected_components=1,
                derived_tracking_box=seed.seed_box_2d,
                center_2d=[
                    (seed.seed_box_2d[0] + seed.seed_box_2d[2]) / 2,
                    (seed.seed_box_2d[1] + seed.seed_box_2d[3]) / 2,
                ],
                mean_positive_probability=0.9,
                scene_cut_score=None,
                shot_boundary=False,
                tracking_state=TrackingState.TRACKED,
                state_reasons=[],
                semantic_identity_status=SemanticIdentityStatus.NOT_REVALIDATED,
            )
            for index, time_ms in enumerate((0, 500))
        ]
        track = SegmentationTrack(
            method="bbox_seed_sam2_video_mask_propagation",
            asset_id="sha256:" + "a" * 64,
            video_path=str(source.resolve()),
            target_description=seed.target_description,
            seed_source=seed.seed_source,
            seed_time_ms=seed.seed_time_ms,
            seed_sample_index=0,
            seed_frame_pts=seed.seed_frame_pts,
            seed_frame_sha256=seed.seed_frame_sha256,
            seed_source_width=seed.seed_source_width,
            seed_source_height=seed.seed_source_height,
            semantic_seed_box=seed.seed_box_2d,
            seed_prompt_type="box",
            sam_prompt_box=pad_normalized_box(seed.seed_box_2d, 0.04),
            sam_prompt_mask_polygon_xy=None,
            seed_box_padding_ratio=0.04,
            refined_seed_mask_path="masks/000000.png",
            analysis_fps=2,
            analysis_width=320,
            analysis_height=180,
            analysis_start_ms=0,
            analysis_end_ms=1000,
            source_start_pts=0,
            source_time_base={"numerator": 1, "denominator": 1000},
            timing_warning="test",
            semantic_warning="test",
            total_samples=2,
            state_counts={TrackingState.TRACKED: 2},
            elapsed_seconds=0,
            effective_fps=2,
            model_provenance=provenance,
            samples=samples,
            target_id=seed.target_id,
            shared_session_id="session-1",
            analysis_frames_manifest_sha256=frames_manifest_sha256,
        )
        track_path = (
            session_dir / "targets" / seed.target_id / "segmentation-track.json"
        )
        track_path.parent.mkdir(parents=True)
        write_json(track_path, track)
        members.append(
            SharedSam21SessionTarget(
                target_id=seed.target_id,
                target_description=seed.target_description,
                seed_time_ms=seed.seed_time_ms,
                seed_sample_index=0,
                seed_frame_pts=seed.seed_frame_pts,
                seed_frame_sha256=seed.seed_frame_sha256,
                seed_source_width=seed.seed_source_width,
                seed_source_height=seed.seed_source_height,
                track_path=str(track_path.relative_to(session_dir)),
                track_sha256=sha256_file(track_path),
                state_counts={TrackingState.TRACKED: 2},
            )
        )
    mismatched_state_counts = track.model_dump(mode="json")
    mismatched_state_counts["state_counts"] = {"low_confidence": 2}
    with pytest.raises(
        ValidationError, match="state_counts must match sample tracking_state values"
    ):
        SegmentationTrack.model_validate(mismatched_state_counts)
    manifest = SharedSam21SessionManifest(
        artifact_type="shared_sam21_multi_object_tracking_session",
        method="bbox_seed_shared_sam2_video_mask_propagation",
        session_id="session-1",
        asset_id="sha256:" + "a" * 64,
        video_path=str(source.resolve()),
        shot_id="shot-1",
        analysis_fps=2,
        analysis_width=320,
        analysis_height=180,
        analysis_start_ms=0,
        analysis_end_ms=1000,
        source_start_pts=0,
        source_time_base={"numerator": 1, "denominator": 1000},
        analysis_frames_path=frames_manifest_path.name,
        analysis_frames_manifest_sha256=frames_manifest_sha256,
        analysis_frames=frame_records,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        target_count=2,
        targets=members,
        model_provenance=provenance,
        timing=SharedSam21SessionTiming(
            shot_detection_seconds=0,
            analysis_frame_extraction_seconds=0,
            predictor_initialization_seconds=0,
            prompt_seconds=0,
            forward_propagation_seconds=0,
            reverse_propagation_seconds=0,
            target_artifact_seconds=0,
            total_seconds=0,
        ),
        warning="test",
        generated_at="2026-07-22T00:00:00Z",
    )

    tracks = _validate_shared_sam_session_cache(
        manifest=manifest,
        session_dir=session_dir,
        video_path=source,
        asset_id=manifest.asset_id,
        start_ms=0,
        end_ms=1000,
        analysis_fps=2,
        analysis_max_side=960,
        checkpoint_sha256=checkpoint_sha256,
        seeds=seeds,
        seed_box_padding_ratio=0.04,
    )
    assert [track.target_id for track in tracks] == ["region-1", "region-2"]

    first_track_path = session_dir / manifest.targets[0].track_path
    first_track_path.write_bytes(first_track_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="track hash mismatch"):
        _validate_shared_sam_session_cache(
            manifest=manifest,
            session_dir=session_dir,
            video_path=source,
            asset_id=manifest.asset_id,
            start_ms=0,
            end_ms=1000,
            analysis_fps=2,
            analysis_max_side=960,
            checkpoint_sha256=checkpoint_sha256,
            seeds=seeds,
            seed_box_padding_ratio=0.04,
        )


def test_only_non_retryable_spending_cap_errors_trip_geometry_circuit_breaker() -> None:
    assert _is_non_retryable_spending_cap_error(
        RuntimeError("project exceeded its monthly spending cap")
    )
    assert not _is_non_retryable_spending_cap_error(
        RuntimeError("429 transient requests per minute quota")
    )
    assert _is_exhausted_model_quota_error(
        RuntimeError("429 transient requests per minute quota")
    )
    assert _is_exhausted_model_quota_error(
        RuntimeError("RESOURCE_EXHAUSTED: quota exceeded")
    )
    assert not _is_exhausted_model_quota_error(
        RuntimeError("the selected entity is not visible in this frame")
    )


def test_controlled_clip_can_preserve_trailing_edge_without_claiming_containment() -> None:
    x_values, audit = _vertical_crop_geometry(
        [0.0, 0.5],
        [500.0, 500.0],
        [[100, 100, 900, 900], [100, 100, 900, 900]],
        overflow_policy="controlled_clip",
        edge_priority="preserve_end",
    )
    crop_width = audit["crop_width_normalized"]
    crop_right = x_values[0] * 1000 / 3414 + crop_width

    assert crop_right == pytest.approx(900, abs=0.01)
    assert audit["controlled_clip_applied"] is True
    assert audit["full_containment_feasible"] is False
    assert audit["containment_failure_count"] == 2
    assert 0 < audit["minimum_visible_required_width_fraction"] < 1


def test_incomplete_tracking_can_hold_grounded_seed_without_centering_the_source() -> None:
    track = SimpleNamespace(
        seed_time_ms=500,
        semantic_seed_box=[300, 100, 730, 800],
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=960,
        analysis_height=540,
        analysis_start_ms=0,
        analysis_end_ms=2000,
        analysis_fps=2.0,
        target_description="the complete visible required region",
        state_counts={"drift_suspected": 4},
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=TrackingState.DRIFT_SUSPECTED,
                derived_tracking_box=[300, 100, 730, 800],
            )
            for index in range(4)
        ],
    )

    filter_graph, audit = _vertical_filter_from_track(  # type: ignore[arg-type]
        [track],
        allow_subject_clipping=True,
        overflow_policy="controlled_clip",
        edge_priority="preserve_end",
        fallback_strategy="center_crop",
    )

    expected_crop_left = 730 - audit["crop_width_normalized"]
    actual_crop_left = (
        audit["crop_keyframes"][0]["crop_x_pixels"] * 1000 / 3414
    )
    assert "x='" in filter_graph
    assert audit["applied_strategy"] == "seed_anchor_crop"
    assert audit["coverage_passed"] is False
    assert audit["requires_gemini_review"] is True
    assert "motion_outside_seed_unverified" in audit["risk_codes"]
    assert actual_crop_left == pytest.approx(expected_crop_left, abs=0.01)


def test_required_track_union_combines_independent_regions_and_flags_missing_samples() -> None:
    def track(target: str, boxes: list[list[int] | None]) -> SimpleNamespace:
        samples = []
        for index, box in enumerate(boxes):
            samples.append(
                SimpleNamespace(
                    analysis_sample_time_ms=index * 500,
                    tracking_state="tracked" if box is not None else "lost",
                    derived_tracking_box=box,
                )
            )
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=2000,
            analysis_fps=2.0,
            target_description=target,
            state_counts={"tracked": sum(box is not None for box in boxes)},
            samples=samples,
        )

    left = track("left performer", [[100, 100, 250, 900]] * 4)
    right = track("right performer", [[600, 100, 760, 900]] * 4)
    times, centers, boxes, coverage = _required_track_union(  # type: ignore[arg-type]
        [left, right], region_ids=["left", "right"]
    )
    assert times == pytest.approx([0.0, 0.5, 1.0, 1.5])
    assert centers == [430.0] * 4
    assert boxes == [[100, 100, 760, 900]] * 4
    assert coverage["coverage_passed"] is True
    assert coverage["expected_sample_interval_ms"] == 500.0

    missing = track(
        "right performer",
        [[600, 100, 760, 900], None, [600, 100, 760, 900], None],
    )
    _, _, _, failed = _required_track_union(  # type: ignore[arg-type]
        [left, missing], region_ids=["left", "right"]
    )
    assert failed["coverage_passed"] is False
    assert failed["unavailable_required_sample_count"] == 2


def test_required_track_union_fails_closed_on_any_low_confidence_sample() -> None:
    samples = [
        SimpleNamespace(
            analysis_sample_time_ms=index * 500,
            tracking_state=(
                TrackingState.LOW_CONFIDENCE
                if index == 5
                else TrackingState.TRACKED
            ),
            derived_tracking_box=[300, 100, 600, 900],
        )
        for index in range(10)
    ]
    track = SimpleNamespace(
        analysis_start_ms=0,
        analysis_end_ms=5000,
        analysis_fps=2.0,
        seed_time_ms=0,
        semantic_seed_box=[300, 100, 600, 900],
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=960,
        analysis_height=540,
        target_description="the required visible region",
        state_counts={"tracked": 9, "low_confidence": 1},
        samples=samples,
    )

    _, _, _, coverage = _required_track_union([track])  # type: ignore[arg-type]
    _, audit = _vertical_filter_from_track([track])  # type: ignore[list-item]

    assert coverage["unavailable_required_sample_ratio"] == pytest.approx(0.1)
    assert coverage["tracking_confidence_gate_passed"] is False
    assert coverage["low_confidence_required_sample_count"] == 1
    assert coverage["coverage_passed"] is False
    assert audit["applied_strategy"] == "fit_with_background"
    assert audit["fallback_reason"] == "required_region_tracking_confidence_failed"
    assert "required_region_low_confidence" in audit["risk_codes"]
    assert audit["requires_gemini_review"] is True


def test_primary_center_relaxes_margin_but_never_clips_primary_target() -> None:
    strict_fits, strict_margin = _vertical_target_fits_crop(
        310.0, 316.3445, primary_center=False
    )
    primary_fits, primary_margin = _vertical_target_fits_crop(
        310.0, 316.3445, primary_center=True
    )
    too_wide, _ = _vertical_target_fits_crop(
        320.0, 316.3445, primary_center=True
    )

    assert strict_fits is False
    assert strict_margin == 1.08
    assert primary_fits is True
    assert primary_margin == 1.0
    assert too_wide is False


def test_tracking_seed_moves_inside_a_trim_that_excludes_catalog_anchor() -> None:
    frame = RushFrame(
        frame_id="RF000001",
        clip_id="clip-1",
        requested_time_ms=7500,
        image_path="/tmp/frame.jpg",
    )

    assert _tracking_seed_request_ms(frame, 1000, 4000) == (2500, "trim_midpoint")
    assert _tracking_seed_request_ms(frame, 7000, 8000) == (7500, "catalog_anchor")


def test_feature_cut_refuses_unreviewed_trim_decision(tmp_path) -> None:
    path = tmp_path / "proposed.json"
    decision = TrimIntentDecision(
        source_asset_id="sha256:" + "a" * 64,
        event_id="event-1",
        shot_id="shot-0001",
        usable=False,
        first_included_frame=None,
        last_included_frame=None,
        exclusive_out_frame=None,
        hold_start_frame=None,
        hold_end_frame=None,
        source_in_ms=None,
        source_out_ms=None,
        source_in_pts=None,
        source_out_pts=None,
        handle_in_ms=None,
        handle_out_ms=None,
        tail_intent="uncertain",
        proposal_path="/tmp/proposal.json",
        catalog_path="/tmp/catalog.json",
    )
    write_json(path, decision)

    with pytest.raises(ValueError, match="human-approved"):
        _load_trim_decisions([path])


def test_feature_cut_preview_flag_still_refuses_unusable_proposal(tmp_path) -> None:
    path = tmp_path / "proposed.json"
    proposal = TrimIntentDecision(
        source_asset_id="sha256:" + "a" * 64,
        event_id="event-1",
        shot_id="shot-0001",
        usable=False,
        first_included_frame=None,
        last_included_frame=None,
        exclusive_out_frame=None,
        hold_start_frame=None,
        hold_end_frame=None,
        source_in_ms=None,
        source_out_ms=None,
        source_in_pts=None,
        source_out_pts=None,
        handle_in_ms=None,
        handle_out_ms=None,
        tail_intent="uncertain",
        proposal_path="/tmp/proposal.json",
        catalog_path="/tmp/catalog.json",
    )
    write_json(path, proposal)

    with pytest.raises(ValueError, match="unreviewed proposed"):
        _load_trim_decisions([path], allow_proposed_preview=True)


def test_feature_cut_applies_only_matching_approved_trim_bounds(tmp_path) -> None:
    clip = RushClip(
        clip_id="clip-1",
        path="/tmp/source.mp4",
        sha256="a" * 64,
        duration_ms=10_000,
        width=1920,
        height=1080,
        frame_rate="30/1",
        size_bytes=1,
    )
    frame = RushFrame(
        frame_id="RF000001",
        clip_id=clip.clip_id,
        requested_time_ms=5000,
        image_path="/tmp/frame.jpg",
    )
    evidence = {
        "frame_id": "DF000001",
        "requested_time_ms": 3000,
        "frame_time_ms": 3003,
        "frame_pts": 90,
        "frame_hash": "b" * 64,
    }
    decision = TrimIntentDecision.model_validate(
        {
            "source_asset_id": "sha256:" + clip.sha256,
            "event_id": "event-1",
            "shot_id": "shot-0001",
            "usable": True,
            "first_included_frame": evidence,
            "last_included_frame": {**evidence, "frame_id": "DF000002", "frame_time_ms": 7007},
            "exclusive_out_frame": {
                **evidence,
                "frame_id": "DF000003",
                "frame_time_ms": 7250,
                "frame_pts": 220,
            },
            "hold_start_frame": None,
            "hold_end_frame": None,
            "source_in_ms": 3003,
            "source_out_ms": 7250,
            "source_in_pts": 90,
            "source_out_pts": 220,
            "handle_in_ms": 2250,
            "handle_out_ms": 8250,
            "tail_intent": "natural_pause",
            "approval_status": "approved",
            "requires_human_review": False,
            "human_review": {
                "reviewer": "reviewer",
                "reviewed_at": "2026-07-21T00:00:00Z",
                "decision": "approved",
                "notes": "verified",
            },
            "proposal_path": "/tmp/proposal.json",
            "catalog_path": "/tmp/catalog.json",
        }
    )
    proposed = TrimIntentDecision.model_validate(
        decision.model_copy(
            update={
                "approval_status": "proposed",
                "requires_human_review": True,
                "human_review": None,
            }
        ).model_dump(mode="json")
    )
    proposed_path = tmp_path / "usable-proposed.json"
    write_json(proposed_path, proposed)
    accepted = _load_trim_decisions(
        [proposed_path],
        allow_proposed_preview=True,
    )
    assert accepted[0][1].approval_status == "proposed"
    shot_cache = {
        clip.clip_id: ShotManifest(
            video_path=clip.path,
            duration_ms=clip.duration_ms,
            detector="test",
            threshold=4,
            generated_at="2026-07-21T00:00:00Z",
            boundaries=[],
            shots=[
                ShotSegment(
                    shot_id="shot-0001",
                    start_time_ms=0,
                    end_time_ms=10_000,
                    start_frame_pts=0,
                    boundary_source="video_start",
                    boundary_score=None,
                )
            ],
        )
    }

    start_ms, end_ms, shot_id, audit = _chapter_bounds_with_approved_trim(
        frame,
        clip,
        2.0,
        shot_cache,
        tmp_path,
        4.0,
        [(tmp_path / "approved.json", decision)],
    )

    assert (start_ms, end_ms, shot_id) == (3003, 7250, "shot-0001")
    assert audit["trim_method"] == "human_approved_frame_id_pts"
    assert audit["trim_event_id"] == "event-1"


def test_trim_decision_can_select_a_better_range_away_from_catalog_anchor(tmp_path) -> None:
    clip = RushClip(
        clip_id="clip-1",
        path="/tmp/source.mp4",
        sha256="a" * 64,
        duration_ms=10_000,
        width=1920,
        height=1080,
        frame_rate="30/1",
        size_bytes=1,
    )
    frame = RushFrame(
        frame_id="RF000001",
        clip_id=clip.clip_id,
        requested_time_ms=7500,
        image_path="/tmp/frame.jpg",
    )
    evidence = {
        "frame_id": "DF000001",
        "requested_time_ms": 1000,
        "frame_time_ms": 1001,
        "frame_pts": 30,
        "frame_hash": "b" * 64,
    }
    decision = TrimIntentDecision.model_validate(
        {
            "source_asset_id": "sha256:" + clip.sha256,
            "event_id": "event-1",
            "shot_id": "shot-0001",
            "usable": True,
            "first_included_frame": evidence,
            "last_included_frame": None,
            "exclusive_out_frame": {
                **evidence,
                "frame_id": "DF000002",
                "requested_time_ms": 4000,
                "frame_time_ms": 4004,
                "frame_pts": 120,
            },
            "hold_start_frame": None,
            "hold_end_frame": None,
            "source_in_ms": 1001,
            "source_out_ms": 4004,
            "source_in_pts": 30,
            "source_out_pts": 120,
            "handle_in_ms": 0,
            "handle_out_ms": 5000,
            "tail_intent": "natural_pause",
            "approval_status": "approved",
            "requires_human_review": False,
            "human_review": {
                "reviewer": "reviewer",
                "reviewed_at": "2026-07-21T00:00:00Z",
                "decision": "approved",
                "notes": "representative select precedes the coarse catalog anchor",
            },
            "proposal_path": "/tmp/proposal.json",
            "catalog_path": "/tmp/catalog.json",
        }
    )
    shot_cache = {
        clip.clip_id: ShotManifest(
            video_path=clip.path,
            duration_ms=clip.duration_ms,
            detector="test",
            threshold=4,
            generated_at="2026-07-21T00:00:00Z",
            boundaries=[],
            shots=[
                ShotSegment(
                    shot_id="shot-0001",
                    start_time_ms=0,
                    end_time_ms=10_000,
                    start_frame_pts=0,
                    boundary_source="video_start",
                    boundary_score=None,
                )
            ],
        )
    }

    start_ms, end_ms, _, _ = _chapter_bounds_with_approved_trim(
        frame,
        clip,
        2.0,
        shot_cache,
        tmp_path,
        4.0,
        [(tmp_path / "approved.json", decision)],
    )

    assert (start_ms, end_ms) == (1001, 4004)


def test_feature_brief_can_disable_titles_and_choose_primary_center_crop() -> None:
    brief = FeatureEditBrief(
        project_id="clean-cut",
        title="clean",
        target_duration_seconds=60,
        render_title_overlays=False,
        chapters=[
            FeatureChapterBrief(
                feature_id="hero",
                title="hero",
                detail_lines=[],
                target_duration_seconds=6,
                vertical_primary_target_description="reviewer-selected foreground subject",
                vertical_crop_mode="primary_center",
            )
        ],
    )
    assert brief.render_title_overlays is False
    assert brief.chapters[0].vertical_crop_mode == "primary_center"


def test_feature_brief_supports_generic_required_text_and_subject_regions() -> None:
    brief = FeatureChapterBrief(
        feature_id="mixed_scene",
        title="Preserve evidence",
        detail_lines=[],
        target_duration_seconds=4,
        vertical_regions=[
            FramingRegionIntent(
                region_id="speaker",
                target_description="the presenter nearest the lectern",
                kind="subject",
            ),
            FramingRegionIntent(
                region_id="heading",
                target_description="the complete visible heading on the sign",
                kind="text_region",
            ),
        ],
    )
    assert [region.kind for region in brief.vertical_regions] == [
        "subject",
        "text_region",
    ]

    with pytest.raises(ValidationError, match="edge priority"):
        FeatureChapterBrief(
            feature_id="invalid",
            title="invalid",
            detail_lines=[],
            target_duration_seconds=4,
            vertical_edge_priority="preserve_end",
        )


def test_feature_brief_can_forbid_blurred_vertical_fallback() -> None:
    brief = FeatureEditBrief(
        project_id="clean-cut",
        title="clean",
        target_duration_seconds=60,
        vertical_fallback_strategy="center_crop",
        chapters=[
            FeatureChapterBrief(
                feature_id="hero",
                title="hero",
                detail_lines=[],
                target_duration_seconds=6,
            )
        ],
    )
    assert brief.vertical_fallback_strategy == "center_crop"


def test_tracked_reframe_requires_target_and_nonzero_intent() -> None:
    payload = {
        "feature_id": "ui",
        "evidence_status": "supported",
        "horizontal_frame_id": "RF000001",
        "vertical_frame_id": "RF000002",
        "observed_visual_evidence": "selected subject remains visible",
        "selection_reason": "clear",
        "horizontal_strategy": "tracked_reframe",
        "horizontal_zoom_intent": "none",
        "horizontal_target_description": None,
        "vertical_strategy": "fit_with_background",
        "vertical_target_description": None,
        "quality_risks": [],
        "confidence": 0.9,
    }
    with pytest.raises(ValidationError, match="requires a zoom intent"):
        FeatureChapterSelect.model_validate(payload)


def test_piecewise_expression_is_ffmpeg_escaped() -> None:
    expression = _piecewise_expression([0.0, 0.5, 1.0], [100.0, 150.0, 130.0])
    assert "lt(t\\,0.500)" in expression
    assert "if(" in expression


def test_track_centers_are_rebased_and_exclude_low_confidence_geometry() -> None:
    track = SimpleNamespace(
        analysis_start_ms=5000,
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=5100,
                tracking_state="tracked",
                center_2d=[300.0, 500.0],
                derived_tracking_box=[200, 200, 400, 800],
            ),
            SimpleNamespace(
                analysis_sample_time_ms=6100,
                tracking_state="low_confidence",
                center_2d=[500.0, 500.0],
                derived_tracking_box=[400, 200, 600, 800],
            ),
        ],
    )
    times, centers, boxes = _usable_track_centers(track)  # type: ignore[arg-type]
    assert times == pytest.approx([0.1])
    assert centers == [300.0]
    assert boxes == [[200, 200, 400, 800]]


def test_horizontal_reframe_fails_closed_on_low_confidence_geometry() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=960,
        analysis_height=540,
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=(
                    TrackingState.LOW_CONFIDENCE
                    if index == 1
                    else TrackingState.TRACKED
                ),
                center_2d=[500.0, 500.0],
                derived_tracking_box=[400, 300, 600, 700],
            )
            for index in range(3)
        ],
    )

    filter_graph, audit = _horizontal_filter_from_track(  # type: ignore[arg-type]
        track, "subtle"
    )

    assert "scale=1920:1080" in filter_graph
    assert audit["applied_zoom"] == 1.0
    assert audit["fallback_reason"] == "tracking_confidence_gate_failed"
    assert audit["tracking_confidence_gate_passed"] is False
    assert audit["low_confidence_sample_count"] == 1
    assert audit["risk_codes"] == [
        "tracking_low_confidence",
        "requested_tracked_reframe_not_applied",
    ]
    assert audit["requires_gemini_review"] is True


def test_horizontal_reframe_fails_closed_on_source_lineage_mismatch() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        seed_source_width=1920,
        seed_source_height=1080,
        analysis_width=640,
        analysis_height=480,
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=TrackingState.TRACKED,
                center_2d=[500.0, 500.0],
                derived_tracking_box=[400, 300, 600, 700],
            )
            for index in range(3)
        ],
    )

    _, audit = _horizontal_filter_from_track(track, "subtle")  # type: ignore[arg-type]

    assert audit["applied_zoom"] == 1.0
    assert audit["source_geometry_lineage_passed"] is False
    assert audit["fallback_reason"].endswith("analysis_aspect_disagrees")
    assert "track_source_geometry_mismatch" in audit["risk_codes"]
    assert audit["requires_gemini_review"] is True


def test_horizontal_four_by_three_reframe_tracks_in_both_crop_axes() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        seed_source_width=1440,
        seed_source_height=1080,
        analysis_width=640,
        analysis_height=480,
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=TrackingState.TRACKED,
                center_2d=[500.0, 350.0 + index * 100],
                derived_tracking_box=[400, 250 + index * 100, 600, 450 + index * 100],
            )
            for index in range(3)
        ],
    )

    filter_graph, audit = _horizontal_filter_from_track(  # type: ignore[arg-type]
        track, "subtle"
    )

    assert "scale=2152:1614" in filter_graph
    assert ":x='" in filter_graph and ":y='" in filter_graph
    assert audit["fallback_reason"] is None
    assert audit["full_containment_feasible"] is True
    assert audit["crop_coordinate_space"]["source_display_width"] == 1440
    assert audit["crop_coordinate_space"]["active_pan_axes"] == ["x", "y"]


def test_vertical_portrait_reframe_uses_track_driven_y_crop() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        analysis_end_ms=1000,
        analysis_fps=2.0,
        seed_time_ms=0,
        semantic_seed_box=[400, 650, 600, 850],
        seed_source_width=1080,
        seed_source_height=2400,
        analysis_width=432,
        analysis_height=960,
        target_description="required visible region",
        state_counts={"tracked": 2},
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=TrackingState.TRACKED,
                derived_tracking_box=(
                    [400, 650, 600, 850]
                    if index == 0
                    else [400, 100, 600, 300]
                ),
            )
            for index in range(2)
        ],
    )

    filter_graph, audit = _vertical_filter_from_track(  # type: ignore[arg-type]
        [track]
    )

    assert "scale=1080:2400" in filter_graph
    assert ":x='" in filter_graph and ":y='" in filter_graph
    assert audit["applied_strategy"] == "tracked_crop"
    assert audit["crop_coordinate_space"]["active_pan_axes"] == ["y"]
    assert audit["crop_keyframes"][0]["crop_y_pixels"] > (
        audit["crop_keyframes"][1]["crop_y_pixels"]
    )
    assert audit["containment_failure_count"] == 0


def test_vertical_multi_region_reframe_rejects_disagreeing_seed_dimensions() -> None:
    def track(
        target: str,
        source_dimensions: tuple[int, int],
        analysis_dimensions: tuple[int, int],
        box: list[int],
    ) -> SimpleNamespace:
        return SimpleNamespace(
            analysis_start_ms=0,
            analysis_end_ms=1000,
            analysis_fps=2.0,
            seed_time_ms=0,
            semantic_seed_box=box,
            seed_source_width=source_dimensions[0],
            seed_source_height=source_dimensions[1],
            analysis_width=analysis_dimensions[0],
            analysis_height=analysis_dimensions[1],
            target_description=target,
            state_counts={"tracked": 2},
            samples=[
                SimpleNamespace(
                    analysis_sample_time_ms=index * 500,
                    tracking_state=TrackingState.TRACKED,
                    derived_tracking_box=box,
                )
                for index in range(2)
            ],
        )

    _, audit = _vertical_filter_from_track(  # type: ignore[list-item]
        [
            track("left", (1920, 1080), (960, 540), [100, 100, 300, 900]),
            track("right", (1440, 1080), (640, 480), [600, 100, 800, 900]),
        ],
        region_ids=["left", "right"],
    )

    assert audit["applied_strategy"] == "fit_with_background"
    assert audit["source_geometry_lineage_passed"] is False
    assert audit["fallback_reason"].endswith("required_tracks_disagree")
    assert audit["risk_codes"] == ["track_source_geometry_mismatch"]
    assert audit["requires_gemini_review"] is True


def test_vertical_fallback_filters_are_aspect_preserving_on_tall_sources() -> None:
    fit_filter = _vertical_fit_filter()
    center_filter = _vertical_center_crop_filter()

    assert "force_original_aspect_ratio=decrease" in fit_filter
    assert "[foreground_source]scale=1080:1920" in fit_filter
    assert "y=(ih-oh)/2" in center_filter


def test_non_square_pixel_source_fails_closed_to_sar_normalized_static_reframe() -> None:
    track = SimpleNamespace(
        analysis_start_ms=0,
        analysis_end_ms=1000,
        analysis_fps=2.0,
        seed_time_ms=0,
        semantic_seed_box=[350, 200, 650, 800],
        seed_source_width=720,
        seed_source_height=576,
        analysis_width=720,
        analysis_height=576,
        target_description="required visible region",
        state_counts={"tracked": 2},
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 500,
                tracking_state=TrackingState.TRACKED,
                center_2d=[500.0, 500.0],
                derived_tracking_box=[350, 200, 650, 800],
            )
            for index in range(2)
        ],
    )

    horizontal_filter, horizontal_audit = _horizontal_filter_from_track(  # type: ignore[arg-type]
        track,
        "subtle",
        display_sample_aspect_ratio=16 / 15,
    )
    vertical_filter, vertical_audit = _vertical_filter_from_track(  # type: ignore[list-item]
        [track],
        fallback_strategy="center_crop",
        display_sample_aspect_ratio=16 / 15,
    )

    for filter_graph in (horizontal_filter, vertical_filter):
        assert "iw*sar" in filter_graph
        assert "setsar=1" in filter_graph
    for audit in (horizontal_audit, vertical_audit):
        assert audit["fallback_reason"] == (
            "non_square_pixel_aspect_ratio_requires_static_reframe"
        )
        assert audit["risk_codes"][0] == (
            "non_square_pixel_aspect_ratio_requires_static_reframe"
        )
        assert audit["requires_gemini_review"] is True
        assert audit["source_display_sample_aspect_ratio"] == pytest.approx(16 / 15)
        assert audit["sample_aspect_ratio_normalized_by_ffmpeg"] is True


def test_four_by_three_tracked_filter_renders_without_aspect_stretch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "four-by-three.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:r=10:d=0.3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    track = SimpleNamespace(
        analysis_start_ms=0,
        analysis_end_ms=300,
        analysis_fps=10.0,
        seed_time_ms=0,
        semantic_seed_box=[400, 200, 600, 800],
        seed_source_width=320,
        seed_source_height=240,
        analysis_width=320,
        analysis_height=240,
        target_description="required visible region",
        state_counts={"tracked": 3},
        samples=[
            SimpleNamespace(
                analysis_sample_time_ms=index * 100,
                tracking_state=TrackingState.TRACKED,
                derived_tracking_box=[400, 200, 600, 800],
            )
            for index in range(3)
        ],
    )
    filter_graph, audit = _vertical_filter_from_track(  # type: ignore[arg-type]
        [track]
    )
    output = tmp_path / "four-by-three-vertical.mp4"

    _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=300,
        overlay_path=None,
        base_filter=filter_graph,
        output_path=output,
        source_has_audio=False,
    )
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    video = json.loads(completed.stdout)["streams"][0]

    assert (video["width"], video["height"]) == (1080, 1920)
    assert "scale=2560:1920" in filter_graph
    assert audit["crop_coordinate_space"]["aspect_ratio_relative_error"] == 0
    assert audit["containment_failure_count"] == 0


def test_dynamic_crop_filter_renders_video_and_audio(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=30:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    chapter = FeatureChapterBrief(
        feature_id="demo",
        title="動態安全裁切",
        detail_lines=["保留指定主體"],
        target_duration_seconds=3,
    )
    overlay = tmp_path / "overlay.png"
    _render_text_layer(chapter, overlay, dimensions=(1080, 1920))
    expression = _piecewise_expression([0.0, 1.0, 2.0], [400.0, 900.0, 500.0])
    output = tmp_path / "vertical.mp4"
    _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=2000,
        overlay_path=overlay,
        base_filter=(
            "[0:v]fps=30,scale=3414:1920,"
            f"crop=1080:1920:x='{expression}':y=0,setsar=1[base]"
        ),
        output_path=output,
    )
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout)["streams"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(stream["codec_type"] == "audio" for stream in streams)

    clean_output = tmp_path / "vertical-clean.mp4"
    _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=500,
        overlay_path=None,
        base_filter=(
            "[0:v]fps=30,scale=3414:1920,"
            "crop=1080:1920:x=500:y=0,setsar=1[base]"
        ),
        output_path=clean_output,
    )
    assert clean_output.exists()
    assert not (tmp_path / ".vertical-clean.partial.mp4").exists()


def test_concat_decodes_each_mp4_instead_of_stream_copy(tmp_path: Path) -> None:
    segments: list[Path] = []
    for index, frequency in enumerate((440, 660)):
        segment = tmp_path / f"segment-{index}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={'red' if index == 0 else 'blue'}:s=320x180:r=30:d=1",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={frequency}:sample_rate=48000:duration=1",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-pix_fmt",
                "yuv420p",
                str(segment),
            ],
            check=True,
        )
        segments.append(segment)
    output = tmp_path / "joined.mp4"
    _concat_segments(segments, output)
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(completed.stdout) == pytest.approx(2.0, abs=0.08)


def test_video_only_source_gets_explicit_synthetic_silence(tmp_path: Path) -> None:
    source = tmp_path / "video-only.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x180:r=30:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    output = tmp_path / "review-segment.mp4"
    audio_origin = _render_source_segment(
        source_path=source,
        start_ms=0,
        end_ms=1000,
        overlay_path=None,
        base_filter="[0:v]fps=30,scale=320:180,setsar=1[base]",
        output_path=output,
    )
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = {stream["codec_type"] for stream in json.loads(completed.stdout)["streams"]}
    assert audio_origin == "synthetic_silence"
    assert stream_types == {"video", "audio"}


def test_automatic_reframe_summary_preserves_switch_and_failure_audit() -> None:
    summary = _summarize_automatic_reframe(
        [
            {
                "feature_id": "opening",
                "applied_strategy": "tracked_crop",
                "requires_gemini_review": False,
                "automatic_candidate_selection": {
                    "enabled": True,
                    "selected_candidate_id": "take-b",
                    "selected_candidate_rank": 2,
                    "attempts": [
                        {
                            "candidate_id": "take-a",
                            "failure_codes": ["hard_core_not_fully_retained"],
                        },
                        {"candidate_id": "take-b", "failure_codes": []},
                    ],
                },
            },
            {
                "feature_id": "closing",
                "applied_strategy": "policy_blocked_preview_fit",
                "requires_gemini_review": True,
                "automatic_candidate_selection": {
                    "enabled": True,
                    "selected_candidate_id": "take-c",
                    "selected_candidate_rank": 1,
                    "attempts": [
                        {
                            "candidate_id": "take-c",
                            "failure_codes": ["track_coverage_below_minimum"],
                        }
                    ],
                },
            },
        ]
    )

    assert summary["candidate_attempt_count"] == 3
    assert summary["candidate_switch_count"] == 1
    assert summary["policy_blocked_chapter_count"] == 1
    assert summary["review_required_chapter_count"] == 1
    assert summary["failure_code_counts"] == {
        "hard_core_not_fully_retained": 1,
        "track_coverage_below_minimum": 1,
    }
    assert len(summary["summary_sha256"]) == 64
