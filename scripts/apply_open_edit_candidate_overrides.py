#!/usr/bin/env python3
"""Apply human-reviewed take replacements to an auditable no-brief plan."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jascue_video_lab.feature_cut import (
    load_external_feature_plan_projection,
    validate_external_feature_plan_projection,
    write_external_feature_plan_projection,
)
from jascue_video_lab.media import sha256_file
from jascue_video_lab.storage import read_json, utc_now, write_json
from scripts.plan_clip_card_open_edit import OpenEditPlan, project_feature_contracts


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateOverride(StrictModel):
    feature_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    aspect: Literal["horizontal", "vertical", "both"]
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    reason: str = Field(min_length=1)


class CandidateOverridePatch(StrictModel):
    interpretation: Literal["human_reviewed_candidate_replacement"]
    overrides: list[CandidateOverride] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_aspect_targets(self) -> "CandidateOverridePatch":
        keys: set[tuple[str, str]] = set()
        for override in self.overrides:
            aspects = (
                ("horizontal", "vertical")
                if override.aspect == "both"
                else (override.aspect,)
            )
            for aspect in aspects:
                key = (override.feature_id, aspect)
                if key in keys:
                    raise ValueError(f"duplicate candidate override: {key}")
                keys.add(key)
        return self


def apply_candidate_overrides(
    plan: OpenEditPlan,
    patch: CandidateOverridePatch,
) -> OpenEditPlan:
    shots = {shot.feature_id: shot for shot in plan.shots}
    unknown = sorted(
        {override.feature_id for override in patch.overrides} - set(shots)
    )
    if unknown:
        raise ValueError(f"candidate overrides reference unknown features: {unknown}")
    updates: dict[str, dict[str, str]] = {}
    for override in patch.overrides:
        shot = shots[override.feature_id]
        candidate_ids = {candidate.candidate_id for candidate in shot.candidates}
        if override.candidate_id not in candidate_ids:
            raise ValueError(
                f"unknown candidate for {override.feature_id}: {override.candidate_id}"
            )
        update = updates.setdefault(override.feature_id, {})
        if override.aspect in {"horizontal", "both"}:
            update["horizontal_candidate_id"] = override.candidate_id
        if override.aspect in {"vertical", "both"}:
            update["vertical_candidate_id"] = override.candidate_id
    revised_shots = [
        shot.model_copy(update=updates.get(shot.feature_id, {})) for shot in plan.shots
    ]
    return OpenEditPlan.model_validate(
        plan.model_copy(update={"shots": revised_shots}).model_dump(mode="json")
    )


def reproject_external_feature_plan(
    *,
    source_plan: OpenEditPlan,
    catalog: object,
    brief: object,
    source_artifacts: dict[str, Path],
) -> tuple[object, object]:
    """Recompute a reviewed override and its renderer contracts from evidence."""

    del brief
    if getattr(catalog, "catalog_id", None) != source_plan.catalog_id:
        raise ValueError("override source plan differs from projection catalog")
    input_plan_path = source_artifacts["input_open_edit_plan"]
    patch_path = source_artifacts["candidate_override_patch"]
    audit_path = source_artifacts["candidate_override_audit"]
    upstream_pointer_path = source_artifacts["upstream_projection_pointer"]
    upstream_record = validate_external_feature_plan_projection(
        upstream_pointer_path.parent
    )
    if upstream_record.get("source_plan_sha256") != sha256_file(input_plan_path):
        raise ValueError("override input plan differs from validated upstream plan")
    input_plan = OpenEditPlan.model_validate(read_json(input_plan_path))
    patch = CandidateOverridePatch.model_validate(read_json(patch_path))
    recomputed = apply_candidate_overrides(input_plan, patch)
    if recomputed.model_dump(mode="json") != source_plan.model_dump(mode="json"):
        raise ValueError("overridden source plan differs from deterministic patch result")
    audit = read_json(audit_path)
    if (
        not isinstance(audit, dict)
        or audit.get("source_plan_sha256") != sha256_file(input_plan_path)
        or audit.get("override_patch_sha256") != sha256_file(patch_path)
        or audit.get("overrides")
        != [item.model_dump(mode="json") for item in patch.overrides]
    ):
        raise ValueError("candidate override audit differs from validated patch inputs")
    projected_brief, projected_plan, _ = project_feature_contracts(
        source_plan,
        preserve_runtime_candidates=False,
    )
    return projected_brief, projected_plan


def reproject_external_feature_plan_v2(
    *,
    source_plan: OpenEditPlan,
    catalog: object,
    brief: object,
    source_artifacts: dict[str, Path],
) -> tuple[object, object]:
    """Validate the same override chain and retain runtime Top-K candidates."""

    reproject_external_feature_plan(
        source_plan=source_plan,
        catalog=catalog,
        brief=brief,
        source_artifacts=source_artifacts,
    )
    projected_brief, projected_plan, _ = project_feature_contracts(
        source_plan,
        preserve_runtime_candidates=True,
    )
    return projected_brief, projected_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("open_edit_plan", type=Path)
    parser.add_argument("override_patch", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    plan_path = args.open_edit_plan.expanduser().resolve()
    patch_path = args.override_patch.expanduser().resolve()
    upstream_plan_dir = plan_path.parent / "gemini-plan"
    upstream_record = validate_external_feature_plan_projection(upstream_plan_dir)
    upstream_pointer_path, upstream_record_path, _ = (
        load_external_feature_plan_projection(upstream_plan_dir)
    )
    if upstream_record.get("source_plan_sha256") != sha256_file(plan_path):
        raise ValueError(
            "candidate override input does not match the validated upstream source plan"
        )
    upstream_catalog_path = upstream_record.get("catalog_path")
    upstream_request_path = upstream_record.get("source_request_path")
    if not isinstance(upstream_catalog_path, str) or not isinstance(
        upstream_request_path, str
    ):
        raise ValueError("upstream projection is missing catalog/request provenance")
    plan = OpenEditPlan.model_validate(read_json(plan_path))
    patch = CandidateOverridePatch.model_validate(read_json(patch_path))
    revised = apply_candidate_overrides(plan, patch)
    brief, feature_plan, trim_plan = project_feature_contracts(revised)

    output_dir = args.output_dir.expanduser().resolve()
    plan_dir = output_dir / "gemini-plan"
    write_json(output_dir / "open-edit-plan.overridden.json", revised)
    write_json(output_dir / "brief.json", brief)
    write_json(plan_dir / "feature_edit_plan.json", feature_plan)
    write_json(plan_dir / "open-edit-trim-plan.json", trim_plan)
    write_json(
        output_dir / "candidate-override.audit.json",
        {
            "interpretation": patch.interpretation,
            "source_plan": str(plan_path),
            "source_plan_sha256": sha256_file(plan_path),
            "override_patch": str(patch_path),
            "override_patch_sha256": sha256_file(patch_path),
            "overrides": [item.model_dump(mode="json") for item in patch.overrides],
            "generated_at": utc_now(),
        },
    )
    write_external_feature_plan_projection(
        plan_dir=plan_dir,
        projection_contract_id="open-edit-candidate-overrides-v2",
        catalog_path=Path(upstream_catalog_path),
        brief_path=output_dir / "brief.json",
        feature_plan_path=plan_dir / "feature_edit_plan.json",
        source_plan_path=output_dir / "open-edit-plan.overridden.json",
        source_request_path=Path(upstream_request_path),
        source_artifacts={
            "input_open_edit_plan": plan_path,
            "candidate_override_patch": patch_path,
            "candidate_override_audit": output_dir / "candidate-override.audit.json",
            "upstream_projection_pointer": upstream_pointer_path,
            "upstream_projection_record": upstream_record_path,
        },
    )
    print((output_dir / "open-edit-plan.overridden.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
