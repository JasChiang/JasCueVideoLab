from __future__ import annotations

from pathlib import Path

from jascue_video_lab.models import ModelProvenance
from scripts.reconcile_open_edit_budget import BudgetPlan, SegmentDecision
from scripts.reuse_open_edit_budget import build_budgeted_manifest, resolve_budget_segments


def test_budget_sequence_reuse_is_deterministic_and_checks_aspect_duration(
    tmp_path: Path,
) -> None:
    budget = BudgetPlan(
        project_id="generic",
        target_min_seconds=3,
        target_max_seconds=5,
        sequence=["scene_b", "scene_a"],
        decisions=[
            SegmentDecision(feature_id="scene_a", action="keep", reason="useful"),
            SegmentDecision(feature_id="scene_b", action="keep", reason="useful"),
        ],
        strategy_summary="reviewed sequence",
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
    manifest = {
        "project_id": "generic",
        "horizontal": {
            "chapters": [
                {
                    "feature_id": "scene_a",
                    "source_in_ms": 0,
                    "source_out_ms": 2000,
                    "segment_path": "/tmp/h-a.mp4",
                },
                {
                    "feature_id": "scene_b",
                    "source_in_ms": 0,
                    "source_out_ms": 2000,
                    "segment_path": "/tmp/h-b.mp4",
                },
            ]
        },
        "vertical": {
            "chapters": [
                {
                    "feature_id": "scene_a",
                    "source_in_ms": 0,
                    "source_out_ms": 2000,
                    "segment_path": "/tmp/v-a.mp4",
                },
                {
                    "feature_id": "scene_b",
                    "source_in_ms": 0,
                    "source_out_ms": 2000,
                    "segment_path": "/tmp/v-b.mp4",
                },
            ]
        },
    }
    paths, duration = resolve_budget_segments(budget, manifest)
    assert [path.name for path in paths["9x16"]] == ["v-b.mp4", "v-a.mp4"]
    assert duration == 4.0

    source_manifest_path = tmp_path / "render-manifest.json"
    source_manifest_path.write_text("{}", encoding="utf-8")
    source_budget_path = tmp_path / "budget-plan.json"
    source_budget_path.write_text("{}", encoding="utf-8")
    budgeted = build_budgeted_manifest(
        budget,
        manifest,
        {
            "16x9": "/tmp/final-h.mp4",
            "9x16": "/tmp/final-v.mp4",
        },
        {
            "16x9": {"duration_seconds": 4.0, "sha256": "h"},
            "9x16": {"duration_seconds": 4.0, "sha256": "v"},
        },
        source_budget_plan_path=source_budget_path,
        source_manifest_path=source_manifest_path,
    )
    assert [
        item["feature_id"] for item in budgeted["vertical"]["chapters"]
    ] == ["scene_b", "scene_a"]
    assert budgeted["vertical"]["output_path"] == "/tmp/final-v.mp4"
    assert budgeted["vertical"]["media"]["sha256"] == "v"
    assert budgeted["budget_reuse"]["sequence"] == ["scene_b", "scene_a"]
    assert budgeted["budget_reuse"]["source_budget_plan_sha256"]
