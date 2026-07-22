from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .media import sha256_file
from .models import (
    FeatureEditBrief,
    FeatureEditPlan,
    FramingRegionIntent,
    ReframePolicyBinding,
    RushesCatalog,
    StrictModel,
)
from .storage import read_json, utc_now, write_json


REFRAME_POLICY_SIDECAR_VERSION = "human-reframe-policy-sidecar-v1"
REFRAME_POLICY_BINDING_ORIGIN = "human_reframe_policy"
FEATURE_PLAN_BINDING_VERSION = "feature-plan-binding-v1"


class ChapterReframeOverride(StrictModel):
    feature_id: str = Field(pattern=r"^[a-z0-9_-]+$")
    vertical_regions: list[FramingRegionIntent] | None = Field(
        default=None, max_length=4
    )
    vertical_overflow_policy: Literal["preserve_all", "controlled_clip"] | None = None
    vertical_edge_priority: Literal[
        "balanced", "preserve_start", "preserve_end"
    ] | None = None
    vertical_crop_mode: Literal["strict", "primary_center"] | None = None
    decision_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_overflow_contract(self) -> "ChapterReframeOverride":
        if (
            self.vertical_edge_priority not in {None, "balanced"}
            and self.vertical_overflow_policy != "controlled_clip"
        ):
            raise ValueError(
                "a non-balanced edge priority requires an explicit controlled_clip decision"
            )
        return self


class ReframePolicyPatch(StrictModel):
    policy_id: str = Field(min_length=1)
    interpretation: Literal["human_reviewed_reframe_policy_not_ground_truth"]
    chapters: list[ChapterReframeOverride] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_chapters(self) -> "ReframePolicyPatch":
        chapter_ids = [item.feature_id for item in self.chapters]
        if len(chapter_ids) != len(set(chapter_ids)):
            raise ValueError("reframe policy feature IDs must be unique")
        return self


class ReframePolicySidecar(StrictModel):
    sidecar_version: Literal["human-reframe-policy-sidecar-v1"]
    interpretation: Literal["human_reviewed_reframe_policy_not_ground_truth"]
    policy_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    review_note: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    catalog_id: str = Field(min_length=1)
    catalog_path: str = Field(min_length=1)
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_brief_path: str = Field(min_length=1)
    source_brief_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_feature_plan_path: str = Field(min_length=1)
    source_feature_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_plan_binding_path: str = Field(min_length=1)
    source_plan_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    chapters: list[ChapterReframeOverride] = Field(min_length=1)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def feature_plan_selection_fingerprint(plan: FeatureEditPlan) -> str:
    """Hash only the immutable editorial selections used by reframe policy."""

    return _sha256_json(
        {
            "project_id": plan.project_id,
            "catalog_id": plan.catalog_id,
            "chapters": [
                {
                    "feature_id": chapter.feature_id,
                    "evidence_status": chapter.evidence_status,
                    "horizontal_frame_id": chapter.horizontal_frame_id,
                    "vertical_frame_id": chapter.vertical_frame_id,
                }
                for chapter in plan.chapters
            ],
        }
    )


def apply_policy(
    brief: FeatureEditBrief,
    policy: ReframePolicyPatch,
    *,
    binding: ReframePolicyBinding | None = None,
) -> FeatureEditBrief:
    """Apply an explicit human policy without changing editorial selection."""

    known = {chapter.feature_id for chapter in brief.chapters}
    unknown = sorted({item.feature_id for item in policy.chapters} - known)
    if unknown:
        raise ValueError(f"reframe policy references unknown feature IDs: {unknown}")
    overrides = {item.feature_id: item for item in policy.chapters}
    chapters = []
    for chapter in brief.chapters:
        override = overrides.get(chapter.feature_id)
        if override is None:
            chapters.append(chapter)
            continue
        update = override.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"feature_id", "decision_reason"},
        )
        if override.vertical_regions is not None:
            update["vertical_regions"] = override.vertical_regions
        chapters.append(chapter.model_copy(update=update))
    revised = FeatureEditBrief.model_validate(
        brief.model_copy(
            update={
                "chapters": chapters,
                "reframe_policy_binding": binding,
            }
        ).model_dump(mode="json")
    )
    if any(
        chapter.vertical_overflow_policy == "controlled_clip"
        for chapter in revised.chapters
    ) and binding is None:
        raise ValueError(
            "controlled_clip requires an immutable human reframe policy binding"
        )
    return revised


def _require_resolved_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def validate_source_plan_binding(
    *,
    binding_path: Path,
    catalog_path: Path,
    brief_path: Path,
    feature_plan_path: Path,
) -> dict[str, Any]:
    """Validate the existing plan binding before layering a human policy."""

    binding_path = _require_resolved_file(binding_path)
    catalog_path = _require_resolved_file(catalog_path)
    brief_path = _require_resolved_file(brief_path)
    feature_plan_path = _require_resolved_file(feature_plan_path)
    binding = read_json(binding_path)
    if not isinstance(binding, dict):
        raise ValueError("source feature plan binding must be an object")
    if binding.get("binding_version") != FEATURE_PLAN_BINDING_VERSION:
        raise ValueError("source feature plan binding version is unsupported")
    expected = {
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "brief_path": str(brief_path),
        "brief_sha256": sha256_file(brief_path),
        "plan_path": str(feature_plan_path),
        "plan_sha256": sha256_file(feature_plan_path),
    }
    mismatches = [key for key, value in expected.items() if binding.get(key) != value]
    if mismatches:
        raise ValueError(
            "source feature plan binding differs from selected inputs: "
            + ", ".join(mismatches)
        )
    for key in (
        "plan_prompt_sha256",
        "system_instruction_sha256",
        "model_id",
        "model_id_sha256",
        "response_schema_sha256",
        "request_path",
        "request_sha256",
    ):
        if not binding.get(key):
            raise ValueError(f"source feature plan binding is missing {key}")
    request_path = _require_resolved_file(Path(str(binding["request_path"])))
    if sha256_file(request_path) != binding["request_sha256"]:
        raise ValueError("source feature plan request hash is invalid")
    if binding.get("origin") == "external_projection":
        pointer_path = feature_plan_path.parent / "feature-plan.external-projection.json"
        pointer = read_json(_require_resolved_file(pointer_path))
        if (
            not isinstance(pointer, dict)
            or sha256_file(pointer_path) != binding.get("projection_pointer_sha256")
        ):
            raise ValueError("source external projection pointer hash is invalid")
        relative_record = pointer.get("record_path")
        if not isinstance(relative_record, str) or not relative_record:
            raise ValueError("source external projection pointer has no record")
        record_root = (feature_plan_path.parent / "feature-plan-projections").resolve()
        record_path = (feature_plan_path.parent / relative_record).resolve()
        try:
            record_path.relative_to(record_root)
        except ValueError as error:
            raise ValueError("source external projection record escapes its root") from error
        if (
            not record_path.is_file()
            or sha256_file(record_path) != pointer.get("record_sha256")
            or sha256_file(record_path) != binding.get("projection_record_sha256")
        ):
            raise ValueError("source external projection record hash is invalid")
        record = read_json(record_path)
        if not isinstance(record, dict) or any(
            record.get(key) != value
            for key, value in (
                ("catalog_sha256", expected["catalog_sha256"]),
                ("brief_sha256", expected["brief_sha256"]),
                ("feature_plan_sha256", expected["plan_sha256"]),
            )
        ):
            raise ValueError("source external projection record changed its primary inputs")
    return binding


def build_policy_sidecar(
    *,
    policy: ReframePolicyPatch,
    reviewer: str,
    review_note: str,
    catalog_path: Path,
    source_brief_path: Path,
    source_feature_plan_path: Path,
    source_plan_binding_path: Path,
) -> ReframePolicySidecar:
    catalog_path = _require_resolved_file(catalog_path)
    source_brief_path = _require_resolved_file(source_brief_path)
    source_feature_plan_path = _require_resolved_file(source_feature_plan_path)
    source_plan_binding_path = _require_resolved_file(source_plan_binding_path)
    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    brief = FeatureEditBrief.model_validate(read_json(source_brief_path))
    plan = FeatureEditPlan.model_validate(read_json(source_feature_plan_path))
    if brief.reframe_policy_binding is not None or any(
        chapter.vertical_overflow_policy != "preserve_all"
        for chapter in brief.chapters
    ):
        raise ValueError(
            "human reframe policy must be applied to an unmodified preserve_all source brief"
        )
    if plan.project_id != brief.project_id or plan.catalog_id != catalog.catalog_id:
        raise ValueError("source brief, feature plan, and catalog do not match")
    if [item.feature_id for item in plan.chapters] != [
        item.feature_id for item in brief.chapters
    ]:
        raise ValueError("source feature plan chapter order differs from brief")
    validate_source_plan_binding(
        binding_path=source_plan_binding_path,
        catalog_path=catalog_path,
        brief_path=source_brief_path,
        feature_plan_path=source_feature_plan_path,
    )
    return ReframePolicySidecar(
        sidecar_version=REFRAME_POLICY_SIDECAR_VERSION,
        interpretation=policy.interpretation,
        policy_id=policy.policy_id,
        reviewer=reviewer,
        review_note=review_note,
        project_id=brief.project_id,
        catalog_id=catalog.catalog_id,
        catalog_path=str(catalog_path),
        catalog_sha256=sha256_file(catalog_path),
        source_brief_path=str(source_brief_path),
        source_brief_sha256=sha256_file(source_brief_path),
        source_feature_plan_path=str(source_feature_plan_path),
        source_feature_plan_sha256=sha256_file(source_feature_plan_path),
        source_plan_binding_path=str(source_plan_binding_path),
        source_plan_binding_sha256=sha256_file(source_plan_binding_path),
        selection_fingerprint=feature_plan_selection_fingerprint(plan),
        chapters=policy.chapters,
    )


def write_immutable_policy_sidecar(
    output_root: Path,
    sidecar: ReframePolicySidecar,
) -> tuple[Path, str]:
    output_root = output_root.expanduser().resolve()
    payload = sidecar.model_dump(mode="json")
    fingerprint = _sha256_json(payload)
    sidecar_path = (
        output_root
        / "reframe-policy-sidecars"
        / f"policy-{fingerprint}.json"
    )
    if sidecar_path.exists():
        existing = ReframePolicySidecar.model_validate(read_json(sidecar_path))
        if existing != sidecar:
            raise ValueError("content-addressed reframe policy sidecar is inconsistent")
    else:
        write_json(sidecar_path, sidecar)
    return sidecar_path, sha256_file(sidecar_path)


def build_brief_policy_binding(
    *,
    sidecar: ReframePolicySidecar,
    sidecar_path: Path,
    sidecar_sha256: str,
) -> ReframePolicyBinding:
    return ReframePolicyBinding(
        binding_version="human-reframe-policy-binding-v1",
        policy_id=sidecar.policy_id,
        reviewer=sidecar.reviewer,
        sidecar_path=str(sidecar_path.resolve()),
        sidecar_sha256=sidecar_sha256,
        source_brief_path=sidecar.source_brief_path,
        source_brief_sha256=sidecar.source_brief_sha256,
        source_feature_plan_path=sidecar.source_feature_plan_path,
        source_feature_plan_sha256=sidecar.source_feature_plan_sha256,
        source_plan_binding_path=sidecar.source_plan_binding_path,
        source_plan_binding_sha256=sidecar.source_plan_binding_sha256,
        catalog_path=sidecar.catalog_path,
        catalog_sha256=sidecar.catalog_sha256,
        selection_fingerprint=sidecar.selection_fingerprint,
    )


def build_reused_plan_binding(
    *,
    catalog_path: Path,
    brief_path: Path,
    feature_plan_path: Path,
    source_plan_binding: dict[str, Any],
    policy_binding: ReframePolicyBinding,
) -> dict[str, Any]:
    result = {
        "binding_version": FEATURE_PLAN_BINDING_VERSION,
        "origin": REFRAME_POLICY_BINDING_ORIGIN,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "brief_path": str(brief_path.resolve()),
        "brief_sha256": sha256_file(brief_path),
        "plan_path": str(feature_plan_path.resolve()),
        "plan_sha256": sha256_file(feature_plan_path),
        "plan_prompt_sha256": source_plan_binding["plan_prompt_sha256"],
        "system_instruction_sha256": source_plan_binding[
            "system_instruction_sha256"
        ],
        "model_id": source_plan_binding["model_id"],
        "model_id_sha256": source_plan_binding["model_id_sha256"],
        "response_schema_sha256": source_plan_binding[
            "response_schema_sha256"
        ],
        "request_path": source_plan_binding["request_path"],
        "request_sha256": source_plan_binding["request_sha256"],
        "reframe_policy_sidecar_path": policy_binding.sidecar_path,
        "reframe_policy_sidecar_sha256": policy_binding.sidecar_sha256,
        "source_plan_binding_path": policy_binding.source_plan_binding_path,
        "source_plan_binding_sha256": policy_binding.source_plan_binding_sha256,
        "selection_fingerprint": policy_binding.selection_fingerprint,
        "created_at": utc_now(),
    }
    return result


def validate_reframe_policy_bundle(
    *,
    catalog_path: Path,
    brief_path: Path,
    feature_plan_path: Path,
    saved_plan_binding: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild and verify the causal binding used by the renderer."""

    catalog_path = _require_resolved_file(catalog_path)
    brief_path = _require_resolved_file(brief_path)
    feature_plan_path = _require_resolved_file(feature_plan_path)
    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    brief = FeatureEditBrief.model_validate(read_json(brief_path))
    plan = FeatureEditPlan.model_validate(read_json(feature_plan_path))
    policy_binding = brief.reframe_policy_binding
    if policy_binding is None:
        raise ValueError("controlled reframe brief has no human policy binding")
    if saved_plan_binding.get("origin") != REFRAME_POLICY_BINDING_ORIGIN:
        raise ValueError("saved feature plan is not bound to a human reframe policy")

    sidecar_root = (brief_path.parent / "reframe-policy-sidecars").resolve()
    sidecar_path = Path(policy_binding.sidecar_path).expanduser().resolve()
    try:
        sidecar_path.relative_to(sidecar_root)
    except ValueError as error:
        raise ValueError("reframe policy sidecar escapes its artifact root") from error
    if not sidecar_path.is_file() or sha256_file(sidecar_path) != policy_binding.sidecar_sha256:
        raise ValueError("reframe policy sidecar hash is invalid")
    sidecar = ReframePolicySidecar.model_validate(read_json(sidecar_path))
    expected_policy_binding = build_brief_policy_binding(
        sidecar=sidecar,
        sidecar_path=sidecar_path,
        sidecar_sha256=sha256_file(sidecar_path),
    )
    if expected_policy_binding != policy_binding:
        raise ValueError("brief reframe policy binding differs from immutable sidecar")
    if sidecar.catalog_id != catalog.catalog_id or sidecar.project_id != brief.project_id:
        raise ValueError("reframe policy sidecar does not match current project/catalog")
    if str(catalog_path) != sidecar.catalog_path or sha256_file(catalog_path) != sidecar.catalog_sha256:
        raise ValueError("reframe policy catalog provenance changed")

    source_brief_path = _require_resolved_file(Path(sidecar.source_brief_path))
    source_plan_path = _require_resolved_file(Path(sidecar.source_feature_plan_path))
    source_binding_path = _require_resolved_file(Path(sidecar.source_plan_binding_path))
    for path, expected_hash, label in (
        (source_brief_path, sidecar.source_brief_sha256, "source brief"),
        (source_plan_path, sidecar.source_feature_plan_sha256, "source feature plan"),
        (source_binding_path, sidecar.source_plan_binding_sha256, "source plan binding"),
    ):
        if sha256_file(path) != expected_hash:
            raise ValueError(f"reframe policy {label} hash changed")
    source_binding = validate_source_plan_binding(
        binding_path=source_binding_path,
        catalog_path=catalog_path,
        brief_path=source_brief_path,
        feature_plan_path=source_plan_path,
    )
    source_brief = FeatureEditBrief.model_validate(read_json(source_brief_path))
    source_plan = FeatureEditPlan.model_validate(read_json(source_plan_path))
    if source_plan.model_dump(mode="json") != plan.model_dump(mode="json"):
        raise ValueError("reframe policy changed the saved editorial feature plan")
    if feature_plan_selection_fingerprint(plan) != sidecar.selection_fingerprint:
        raise ValueError("reframe policy selection fingerprint changed")
    policy = ReframePolicyPatch(
        policy_id=sidecar.policy_id,
        interpretation=sidecar.interpretation,
        chapters=sidecar.chapters,
    )
    expected_brief = apply_policy(source_brief, policy, binding=policy_binding)
    if expected_brief.model_dump(mode="json") != brief.model_dump(mode="json"):
        raise ValueError("current brief differs from the reviewed reframe policy result")
    current_binding = build_reused_plan_binding(
        catalog_path=catalog_path,
        brief_path=brief_path,
        feature_plan_path=feature_plan_path,
        source_plan_binding=source_binding,
        policy_binding=policy_binding,
    )
    required = (
        "binding_version",
        "origin",
        "catalog_sha256",
        "brief_sha256",
        "plan_sha256",
        "plan_prompt_sha256",
        "system_instruction_sha256",
        "model_id",
        "model_id_sha256",
        "response_schema_sha256",
        "request_sha256",
        "reframe_policy_sidecar_sha256",
        "source_plan_binding_sha256",
        "selection_fingerprint",
    )
    mismatches = [
        key for key in required if saved_plan_binding.get(key) != current_binding.get(key)
    ]
    if mismatches:
        raise ValueError(
            "saved human reframe binding differs from current inputs: "
            + ", ".join(mismatches)
        )
    return current_binding
