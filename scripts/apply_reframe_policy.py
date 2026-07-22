#!/usr/bin/env python3
"""Create a renderer-safe bundle from an explicit human reframe decision."""

from __future__ import annotations

import argparse
from pathlib import Path

from jascue_video_lab.media import sha256_file
from jascue_video_lab.models import FeatureEditBrief, FeatureEditPlan
from jascue_video_lab.reframe_policy import (
    ChapterReframeOverride,
    ReframePolicyPatch,
    apply_policy,
    build_brief_policy_binding,
    build_policy_sidecar,
    build_reused_plan_binding,
    validate_reframe_policy_bundle,
    validate_source_plan_binding,
    write_immutable_policy_sidecar,
)
from jascue_video_lab.storage import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a human-reviewed portrait policy to a preserve_all brief and "
            "reuse the exact saved feature selection through an immutable sidecar."
        )
    )
    parser.add_argument("brief_json", type=Path)
    parser.add_argument("policy_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-plan-binding", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--review-note", required=True)
    args = parser.parse_args()

    source_brief_path = args.brief_json.expanduser().resolve()
    source_plan_path = args.feature_plan.expanduser().resolve()
    catalog_path = args.catalog.expanduser().resolve()
    source_binding_path = (
        args.source_plan_binding.expanduser().resolve()
        if args.source_plan_binding is not None
        else source_plan_path.parent / "feature-plan.binding.json"
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_brief_path = output_dir / "brief.json"
    output_plan_path = output_dir / "gemini-plan" / "feature_edit_plan.json"
    output_plan_binding_path = output_plan_path.parent / "feature-plan.binding.json"
    if output_brief_path == source_brief_path or output_plan_path == source_plan_path:
        raise ValueError("human policy output must not overwrite its immutable source inputs")

    # The saved binding is deliberately validated by its original hashes rather
    # than by today's planner schema.  This permits a reviewed geometry-only
    # rerender after planner-schema upgrades without silently replanning.
    source_binding = validate_source_plan_binding(
        binding_path=source_binding_path,
        catalog_path=catalog_path,
        brief_path=source_brief_path,
        feature_plan_path=source_plan_path,
    )
    source_brief = FeatureEditBrief.model_validate(read_json(source_brief_path))
    source_plan = FeatureEditPlan.model_validate(read_json(source_plan_path))
    policy = ReframePolicyPatch.model_validate(read_json(args.policy_json))
    sidecar = build_policy_sidecar(
        policy=policy,
        reviewer=args.reviewer,
        review_note=args.review_note,
        catalog_path=catalog_path,
        source_brief_path=source_brief_path,
        source_feature_plan_path=source_plan_path,
        source_plan_binding_path=source_binding_path,
    )
    sidecar_path, sidecar_sha256 = write_immutable_policy_sidecar(
        output_dir,
        sidecar,
    )
    policy_binding = build_brief_policy_binding(
        sidecar=sidecar,
        sidecar_path=sidecar_path,
        sidecar_sha256=sidecar_sha256,
    )
    revised_brief = apply_policy(source_brief, policy, binding=policy_binding)

    write_json(output_brief_path, revised_brief)
    # Selection is copied semantically unchanged; no Gemini planning call occurs.
    write_json(output_plan_path, source_plan)
    reused_plan_binding = build_reused_plan_binding(
        catalog_path=catalog_path,
        brief_path=output_brief_path,
        feature_plan_path=output_plan_path,
        source_plan_binding=source_binding,
        policy_binding=policy_binding,
    )
    write_json(output_plan_binding_path, reused_plan_binding)
    validate_reframe_policy_bundle(
        catalog_path=catalog_path,
        brief_path=output_brief_path,
        feature_plan_path=output_plan_path,
        saved_plan_binding=reused_plan_binding,
    )
    write_json(
        output_dir / "reframe-policy.audit.json",
        {
            "interpretation": "human_policy_applied_without_replanning",
            "policy_id": policy.policy_id,
            "reviewer": args.reviewer,
            "source_brief_path": str(source_brief_path),
            "source_brief_sha256": sha256_file(source_brief_path),
            "source_feature_plan_path": str(source_plan_path),
            "source_feature_plan_sha256": sha256_file(source_plan_path),
            "policy_sidecar_path": str(sidecar_path),
            "policy_sidecar_sha256": sidecar_sha256,
            "output_brief_path": str(output_brief_path),
            "output_brief_sha256": sha256_file(output_brief_path),
            "output_feature_plan_path": str(output_plan_path),
            "output_feature_plan_sha256": sha256_file(output_plan_path),
            "selection_fingerprint": policy_binding.selection_fingerprint,
        },
    )
    print(output_brief_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
