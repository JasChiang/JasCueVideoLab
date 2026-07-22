from __future__ import annotations

import hashlib
import html
import importlib
import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path
from time import monotonic
from typing import Any, Literal, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .billing import summarize_usage_and_list_price, summarize_usage_files
from .gemini import (
    EDITORIAL_SYSTEM_INSTRUCTION,
    GeminiLabClient,
    MODEL_ID,
    VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
)
from .grounding_selection import (
    require_grounding_request_match,
    require_tracking_seed_candidate,
)
from .media import extract_frame, has_audio_stream, probe_video, sha256_file
from .multi_tracking import validate_segmentation_track_alignment
from .models import (
    FeatureChapterBrief,
    FeatureChapterSelect,
    FeatureEditBrief,
    FeatureEditPlan,
    FramingRegionIntent,
    GeminiNativeGroundingProposal,
    GroundingProposal,
    MediaInfo,
    RushClip,
    RushFrame,
    RushesCatalog,
    SegmentationTrack,
    SharedSam21AnalysisFramesManifest,
    SharedSam21BBoxSeed,
    SharedSam21SessionManifest,
    TrackingState,
    TrimIntentDecision,
)
from .overlay import draw_grounding_overlay
from .reframe_policy import (
    REFRAME_POLICY_BINDING_ORIGIN,
    validate_reframe_policy_bundle,
)
from .rushes import _segment_bounds
from .sam_tracking import (
    SAM21_CONFIG,
    SAM21_IMPLEMENTATION_REVISION,
    SAM21_TINY_MODEL_ID,
    pad_normalized_box,
    require_bbox_track_request_match,
    track_bbox_sam21,
    track_bboxes_shared_sam21,
)
from .schema import gemini_response_schema
from .shots import ShotManifest, detect_shots_ffmpeg
from .storage import read_json, utc_now, write_json


_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)
_RENDER_PIPELINE_VERSION = "feature-cut-v9-sar-aware-2d-reframe"
_TRACKING_MAX_SIDE = 960
_TRACKING_DEVICE = "cpu"
_TRACKING_SEED_BOX_PADDING_RATIO = 0.04
_FEATURE_PLAN_BINDING_VERSION = "feature-plan-binding-v1"
_EXTERNAL_PROJECTION_SIDECAR_VERSION = "external-feature-plan-projection-v1"
_EXTERNAL_PROJECTION_POINTER_NAME = "feature-plan.external-projection.json"
_EXTERNAL_PROJECTION_CONTRACTS = {
    "clip-card-open-edit-v1": {
        "source": "validated open edit plan selected from Clip Card evidence",
        "transform": "project_feature_contracts maps ordered shots into brief and feature plan",
        "target": "FeatureEditPlan",
        "module": "scripts.plan_clip_card_open_edit",
        "source_model": "OpenEditPlan",
        "projector": "reproject_external_feature_plan",
        "raw_output_role": "source_raw_output",
        "required_artifact_roles": [
            "source_raw_interaction",
            "source_raw_output",
        ],
    },
    "clip-card-feature-cut-v1": {
        "source": "validated brief-aware Clip Card feature plan",
        "transform": "chapter selections are projected in order into FeatureChapterSelect",
        "target": "FeatureEditPlan",
        "module": "scripts.plan_clip_card_feature_cut",
        "source_model": "ClipCardFeaturePlan",
        "projector": "reproject_external_feature_plan",
        "raw_output_role": "source_raw_output",
        "required_artifact_roles": [
            "source_raw_interaction",
            "source_raw_output",
        ],
    },
    "open-edit-candidate-overrides-v1": {
        "source": "validated upstream open edit plan plus human-reviewed candidate patch",
        "transform": "only named aspect candidates are replaced before project_feature_contracts",
        "target": "FeatureEditPlan",
        "module": "scripts.apply_open_edit_candidate_overrides",
        "source_model": "OpenEditPlan",
        "projector": "reproject_external_feature_plan",
        "raw_output_role": None,
        "required_artifact_roles": [
            "input_open_edit_plan",
            "candidate_override_patch",
            "candidate_override_audit",
            "upstream_projection_pointer",
            "upstream_projection_record",
        ],
    },
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _external_projection_contract_sha256(contract_id: str) -> str:
    contract = _EXTERNAL_PROJECTION_CONTRACTS.get(contract_id)
    if contract is None:
        raise ValueError(f"unsupported external feature plan projection: {contract_id}")
    return _sha256_json({"contract_id": contract_id, **contract})


def _validate_external_projection_semantics(
    *,
    projection_contract_id: str,
    catalog_path: Path,
    brief_path: Path,
    feature_plan_path: Path,
    source_plan_path: Path,
    source_request_path: Path,
    source_artifacts: Mapping[str, Path],
) -> None:
    """Reparse source evidence and deterministically reproduce the projection."""

    contract = _EXTERNAL_PROJECTION_CONTRACTS.get(projection_contract_id)
    if contract is None:
        raise ValueError(
            f"unsupported external feature plan projection: {projection_contract_id}"
        )
    module_name = contract.get("module")
    source_model_name = contract.get("source_model")
    projector_name = contract.get("projector")
    if not all(
        isinstance(value, str)
        for value in (module_name, source_model_name, projector_name)
    ):
        raise ValueError("external projection registry entry is incomplete")
    required_roles = contract.get("required_artifact_roles")
    if not isinstance(required_roles, list) or not all(
        isinstance(role, str) for role in required_roles
    ):
        raise ValueError("external projection registry artifact contract is invalid")
    missing_roles = sorted(set(required_roles) - set(source_artifacts))
    if missing_roles:
        raise ValueError(
            "external projection is missing required source artifacts: "
            + ", ".join(missing_roles)
        )

    module = importlib.import_module(module_name)
    source_model = getattr(module, source_model_name, None)
    projector = getattr(module, projector_name, None)
    if source_model is None or not callable(projector):
        raise ValueError("external projection registry implementation is unavailable")
    source_plan = source_model.model_validate(read_json(source_plan_path))
    request = read_json(source_request_path)
    response_format = request.get("response_format") if isinstance(request, dict) else None
    request_schema = (
        response_format.get("schema") if isinstance(response_format, dict) else None
    )
    expected_source_schema = gemini_response_schema(source_model)
    if request_schema != expected_source_schema:
        raise ValueError(
            "external projection source request schema does not match its registered model"
        )

    raw_output_role = contract.get("raw_output_role")
    if raw_output_role is not None:
        if not isinstance(raw_output_role, str) or raw_output_role not in source_artifacts:
            raise ValueError("external projection raw output artifact is missing")
        raw_output = read_json(source_artifacts[raw_output_role])
        output_text = raw_output.get("output_text") if isinstance(raw_output, dict) else None
        if not isinstance(output_text, str):
            raise ValueError("external projection raw output has no output_text")
        raw_plan = source_model.model_validate_json(output_text)
        source_interaction_id = source_plan.model_provenance.interaction_id
        normalized_raw_plan = raw_plan.model_copy(
            update={
                "model_provenance": raw_plan.model_provenance.model_copy(
                    update={"interaction_id": source_interaction_id}
                )
            }
        )
        if normalized_raw_plan.model_dump(mode="json") != source_plan.model_dump(
            mode="json"
        ):
            raise ValueError(
                "external projection source plan differs from validated raw model output"
            )

    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    brief = FeatureEditBrief.model_validate(read_json(brief_path))
    expected_brief, expected_plan = projector(
        source_plan=source_plan,
        catalog=catalog,
        brief=brief,
        source_artifacts=dict(source_artifacts),
    )
    if not isinstance(expected_brief, FeatureEditBrief) or not isinstance(
        expected_plan, FeatureEditPlan
    ):
        raise ValueError("external projection projector returned an invalid contract")
    actual_plan = FeatureEditPlan.model_validate(read_json(feature_plan_path))
    if expected_brief.model_dump(mode="json") != brief.model_dump(mode="json"):
        raise ValueError(
            "external projection brief differs from deterministic projector output"
        )
    if expected_plan.model_dump(mode="json") != actual_plan.model_dump(mode="json"):
        raise ValueError(
            "external FeatureEditPlan differs from deterministic projector output"
        )


def _external_request_claims(request_path: Path) -> dict[str, str]:
    request = read_json(request_path)
    if not isinstance(request, dict):
        raise ValueError("external projection source request must be an object")
    model_id = request.get("model")
    system_instruction = request.get("system_instruction")
    inputs = request.get("input")
    response_format = request.get("response_format")
    response_schema = (
        response_format.get("schema") if isinstance(response_format, dict) else None
    )
    if model_id != MODEL_ID:
        raise ValueError("external projection source request used an unexpected model")
    if not isinstance(system_instruction, str) or not system_instruction:
        raise ValueError("external projection source request has no system instruction")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("external projection source request has no model input")
    if not isinstance(response_schema, dict):
        raise ValueError("external projection source request has no response schema")
    return {
        "source_request_sha256": sha256_file(request_path),
        "source_request_input_sha256": _sha256_json(inputs),
        "source_system_instruction_sha256": _sha256_text(system_instruction),
        "source_model_id": model_id,
        "source_model_id_sha256": _sha256_text(model_id),
        "source_response_schema_sha256": _sha256_json(response_schema),
    }


def _hashed_artifact(role: str, path: Path) -> dict[str, str]:
    if not role or not role.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid external projection artifact role: {role!r}")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved)}


def write_external_feature_plan_projection(
    *,
    plan_dir: Path,
    projection_contract_id: str,
    catalog_path: Path,
    brief_path: Path,
    feature_plan_path: Path,
    source_plan_path: Path,
    source_request_path: Path,
    source_artifacts: Mapping[str, Path] | None = None,
) -> Path:
    """Write immutable provenance for a deterministic external plan projection."""

    plan_dir = plan_dir.expanduser().resolve()
    catalog_path = catalog_path.expanduser().resolve()
    brief_path = brief_path.expanduser().resolve()
    feature_plan_path = feature_plan_path.expanduser().resolve()
    source_plan_path = source_plan_path.expanduser().resolve()
    source_request_path = source_request_path.expanduser().resolve()
    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    brief = FeatureEditBrief.model_validate(read_json(brief_path))
    feature_plan = FeatureEditPlan.model_validate(read_json(feature_plan_path))
    if (
        feature_plan.catalog_id != catalog.catalog_id
        or feature_plan.project_id != brief.project_id
    ):
        raise ValueError("external projected feature plan does not match catalog/brief")
    source_plan = read_json(source_plan_path)
    if not isinstance(source_plan, dict):
        raise ValueError("external projection source plan must be an object")
    request_claims = _external_request_claims(source_request_path)
    source_provenance = source_plan.get("model_provenance")
    if (
        not isinstance(source_provenance, dict)
        or source_provenance.get("model_id") != request_claims["source_model_id"]
    ):
        raise ValueError(
            "external projection source plan provenance does not match its request"
        )
    artifacts = [
        _hashed_artifact(role, path)
        for role, path in sorted((source_artifacts or {}).items())
    ]
    artifact_paths = {
        item["role"]: Path(item["path"])
        for item in artifacts
    }
    _validate_external_projection_semantics(
        projection_contract_id=projection_contract_id,
        catalog_path=catalog_path,
        brief_path=brief_path,
        feature_plan_path=feature_plan_path,
        source_plan_path=source_plan_path,
        source_request_path=source_request_path,
        source_artifacts=artifact_paths,
    )
    core: dict[str, Any] = {
        "sidecar_version": _EXTERNAL_PROJECTION_SIDECAR_VERSION,
        "origin": "external_projection",
        "projection_contract_id": projection_contract_id,
        "projection_contract_sha256": _external_projection_contract_sha256(
            projection_contract_id
        ),
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "brief_path": str(brief_path),
        "brief_sha256": sha256_file(brief_path),
        "feature_plan_path": str(feature_plan_path),
        "feature_plan_sha256": sha256_file(feature_plan_path),
        "source_plan_path": str(source_plan_path),
        "source_plan_sha256": sha256_file(source_plan_path),
        "source_request_path": str(source_request_path),
        **request_claims,
        "source_artifacts": artifacts,
        "source_artifact_set_sha256": _sha256_json(
            [
                {"role": item["role"], "sha256": item["sha256"]}
                for item in artifacts
            ]
        ),
    }
    fingerprint = _sha256_json(core)
    record_dir = plan_dir / "feature-plan-projections"
    record_path = record_dir / f"projection-{fingerprint}.json"
    if record_path.exists():
        existing = read_json(record_path)
        if not isinstance(existing, dict) or any(
            existing.get(key) != value for key, value in core.items()
        ):
            raise ValueError("existing external projection record is inconsistent")
    else:
        write_json(record_path, {**core, "created_at": utc_now()})
    pointer_path = plan_dir / _EXTERNAL_PROJECTION_POINTER_NAME
    write_json(
        pointer_path,
        {
            "sidecar_version": _EXTERNAL_PROJECTION_SIDECAR_VERSION,
            "record_path": str(record_path.relative_to(plan_dir)),
            "record_sha256": sha256_file(record_path),
        },
    )
    return pointer_path


def load_external_feature_plan_projection(
    plan_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Load a contained immutable projection record through its small pointer."""

    plan_dir = plan_dir.expanduser().resolve()
    pointer_path = plan_dir / _EXTERNAL_PROJECTION_POINTER_NAME
    pointer = read_json(pointer_path)
    if not isinstance(pointer, dict):
        raise ValueError("external projection pointer must be an object")
    if pointer.get("sidecar_version") != _EXTERNAL_PROJECTION_SIDECAR_VERSION:
        raise ValueError("external projection pointer version is unsupported")
    relative_record = pointer.get("record_path")
    if not isinstance(relative_record, str) or not relative_record:
        raise ValueError("external projection pointer has no record path")
    record_root = (plan_dir / "feature-plan-projections").resolve()
    record_path = (plan_dir / relative_record).resolve()
    try:
        record_path.relative_to(record_root)
    except ValueError as error:
        raise ValueError("external projection record escapes its artifact root") from error
    if not record_path.is_file() or sha256_file(record_path) != pointer.get(
        "record_sha256"
    ):
        raise ValueError("external projection record hash does not match pointer")
    record = read_json(record_path)
    if not isinstance(record, dict):
        raise ValueError("external projection record must be an object")
    return pointer_path, record_path, record


def _current_external_projection_binding(
    *,
    plan_dir: Path,
    catalog_path: Path,
    brief_path: Path,
    plan_path: Path,
    created_at: str,
) -> dict[str, Any]:
    """Verify every external projection artifact and derive a reusable binding."""

    pointer_path, record_path, record = load_external_feature_plan_projection(
        plan_dir
    )
    if (
        record.get("sidecar_version") != _EXTERNAL_PROJECTION_SIDECAR_VERSION
        or record.get("origin") != "external_projection"
    ):
        raise ValueError("external projection record contract is unsupported")
    contract_id = record.get("projection_contract_id")
    if (
        not isinstance(contract_id, str)
        or record.get("projection_contract_sha256")
        != _external_projection_contract_sha256(contract_id)
    ):
        raise ValueError("external projection contract hash is invalid")
    current_files = {
        "catalog_sha256": sha256_file(catalog_path),
        "brief_sha256": sha256_file(brief_path),
        "feature_plan_sha256": sha256_file(plan_path),
    }
    for key, value in current_files.items():
        if record.get(key) != value:
            raise ValueError(f"external projection {key} differs from current input")
    for prefix in ("catalog", "brief", "feature_plan", "source_plan"):
        source_path = record.get(f"{prefix}_path")
        expected_hash = record.get(f"{prefix}_sha256")
        if not isinstance(source_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"external projection has incomplete {prefix} provenance")
        resolved = Path(source_path).expanduser().resolve()
        if not resolved.is_file() or sha256_file(resolved) != expected_hash:
            raise ValueError(f"external projection {prefix} source hash is invalid")
    request_path_value = record.get("source_request_path")
    if not isinstance(request_path_value, str):
        raise ValueError("external projection has no source request path")
    request_path = Path(request_path_value).expanduser().resolve()
    request_claims = _external_request_claims(request_path)
    for key, value in request_claims.items():
        if record.get(key) != value:
            raise ValueError(f"external projection request claim changed: {key}")
    source_plan = read_json(Path(str(record["source_plan_path"])))
    source_provenance = (
        source_plan.get("model_provenance") if isinstance(source_plan, dict) else None
    )
    if (
        not isinstance(source_provenance, dict)
        or source_provenance.get("model_id") != request_claims["source_model_id"]
    ):
        raise ValueError(
            "external projection source plan provenance no longer matches its request"
        )
    artifact_claims: list[dict[str, str]] = []
    artifact_roles: set[str] = set()
    artifact_paths: dict[str, Path] = {}
    source_artifacts = record.get("source_artifacts")
    if not isinstance(source_artifacts, list):
        raise ValueError("external projection source artifacts must be a list")
    for artifact in source_artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("external projection source artifact must be an object")
        role = artifact.get("role")
        path_value = artifact.get("path")
        expected_hash = artifact.get("sha256")
        if not all(isinstance(value, str) for value in (role, path_value, expected_hash)):
            raise ValueError("external projection source artifact is incomplete")
        if role in artifact_roles:
            raise ValueError(f"external projection source artifact role is duplicated: {role}")
        artifact_roles.add(role)
        artifact_path = Path(path_value).expanduser().resolve()
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
            raise ValueError(f"external projection source artifact changed: {role}")
        artifact_claims.append({"role": role, "sha256": expected_hash})
        artifact_paths[role] = artifact_path
    artifact_set_hash = _sha256_json(artifact_claims)
    if record.get("source_artifact_set_sha256") != artifact_set_hash:
        raise ValueError("external projection source artifact set hash is invalid")
    _validate_external_projection_semantics(
        projection_contract_id=contract_id,
        catalog_path=catalog_path,
        brief_path=brief_path,
        feature_plan_path=plan_path,
        source_plan_path=Path(str(record["source_plan_path"])),
        source_request_path=request_path,
        source_artifacts=artifact_paths,
    )
    return {
        "binding_version": _FEATURE_PLAN_BINDING_VERSION,
        "origin": "external_projection",
        "external_projection_contract_id": contract_id,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": current_files["catalog_sha256"],
        "brief_path": str(brief_path.resolve()),
        "brief_sha256": current_files["brief_sha256"],
        # For external projections, the actual source-model input is the prompt
        # contract; the renderer's unused direct-video plan prompt is irrelevant.
        "plan_prompt_sha256": request_claims["source_request_input_sha256"],
        "system_instruction_sha256": request_claims[
            "source_system_instruction_sha256"
        ],
        "model_id": request_claims["source_model_id"],
        "model_id_sha256": request_claims["source_model_id_sha256"],
        "response_schema_sha256": request_claims[
            "source_response_schema_sha256"
        ],
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": current_files["feature_plan_sha256"],
        "request_path": str(request_path),
        "request_sha256": request_claims["source_request_sha256"],
        "source_plan_sha256": record["source_plan_sha256"],
        "projection_contract_sha256": record["projection_contract_sha256"],
        "projection_pointer_sha256": sha256_file(pointer_path),
        "projection_record_sha256": sha256_file(record_path),
        "source_artifact_set_sha256": artifact_set_hash,
        "created_at": created_at,
    }


def validate_external_feature_plan_projection(plan_dir: Path) -> dict[str, Any]:
    """Validate an upstream projection in place and return its immutable record."""

    _, _, record = load_external_feature_plan_projection(plan_dir)
    required_paths = {
        key: record.get(key)
        for key in (
            "catalog_path",
            "brief_path",
            "feature_plan_path",
        )
    }
    if not all(isinstance(value, str) for value in required_paths.values()):
        raise ValueError("external projection record has incomplete primary paths")
    _current_external_projection_binding(
        plan_dir=plan_dir,
        catalog_path=Path(required_paths["catalog_path"]),  # type: ignore[arg-type]
        brief_path=Path(required_paths["brief_path"]),  # type: ignore[arg-type]
        plan_path=Path(required_paths["feature_plan_path"]),  # type: ignore[arg-type]
        created_at=utc_now(),
    )
    return record


def _current_feature_plan_binding(
    *,
    catalog_path: Path,
    brief_path: Path,
    plan_path: Path,
    plan_prompt: str,
    request_path: Path | None,
    created_at: str,
    origin: Literal["generated", "migrated_legacy_reuse", "external_projection"],
) -> dict[str, Any]:
    """Build the immutable causal inputs for one saved editorial plan."""

    binding: dict[str, Any] = {
        "binding_version": _FEATURE_PLAN_BINDING_VERSION,
        "origin": origin,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "brief_path": str(brief_path.resolve()),
        "brief_sha256": sha256_file(brief_path),
        "plan_prompt_sha256": _sha256_text(plan_prompt),
        "system_instruction_sha256": _sha256_text(
            EDITORIAL_SYSTEM_INSTRUCTION
        ),
        "model_id": MODEL_ID,
        "model_id_sha256": _sha256_text(MODEL_ID),
        "response_schema_sha256": _sha256_json(
            gemini_response_schema(FeatureEditPlan)
        ),
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "created_at": created_at,
    }
    if request_path is not None:
        binding.update(
            {
                "request_path": str(request_path.resolve()),
                "request_sha256": sha256_file(request_path),
            }
        )
    return binding


def _validate_feature_plan_binding(
    saved: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Fail closed when any causal plan input differs from saved evidence."""

    required_hashes = (
        "catalog_sha256",
        "brief_sha256",
        "plan_prompt_sha256",
        "system_instruction_sha256",
        "model_id_sha256",
        "response_schema_sha256",
        "plan_sha256",
        "request_sha256",
    )
    if saved.get("origin") == "external_projection":
        required_hashes += (
            "source_plan_sha256",
            "projection_contract_sha256",
            "projection_pointer_sha256",
            "projection_record_sha256",
            "source_artifact_set_sha256",
        )
    missing = [key for key in required_hashes if not saved.get(key)]
    if saved.get("binding_version") != _FEATURE_PLAN_BINDING_VERSION:
        missing.insert(0, "binding_version")
    if missing:
        raise ValueError(
            "saved feature plan binding is incomplete or unsupported: "
            + ", ".join(missing)
        )
    mismatches = [
        key for key in required_hashes if saved[key] != current.get(key)
    ]
    if saved.get("origin") != current.get("origin"):
        mismatches.append("origin")
    if saved.get("model_id") != current.get("model_id"):
        mismatches.append("model_id")
    if mismatches:
        raise ValueError(
            "saved feature plan causal binding differs from current inputs: "
            + ", ".join(sorted(set(mismatches)))
        )


def _migrate_legacy_feature_plan_binding(
    *,
    plan_dir: Path,
    catalog_path: Path,
    brief_path: Path,
    plan_path: Path,
    plan_prompt: str,
) -> dict[str, Any]:
    """Validate old reuse evidence plus the original API request before migration.

    The legacy record used the wrong system-instruction hash.  It is accepted
    only when that value is one of the two known historical constants and the
    untouched API request independently proves the actual editorial system
    instruction, model, schema and prompt template.  The legacy file is never
    overwritten.
    """

    legacy_path = plan_dir / "feature-plan.reuse.json"
    request_path = plan_dir / "feature_edit_plan.request.json"
    if not legacy_path.exists() or not request_path.exists():
        raise ValueError(
            "saved feature plan has no immutable binding; legacy migration "
            "requires both feature-plan.reuse.json and the original request"
        )
    legacy = read_json(legacy_path)
    if not isinstance(legacy, dict):
        raise ValueError("legacy feature plan reuse record must be an object")
    expected_legacy = {
        "plan_sha256": sha256_file(plan_path),
        "current_catalog_sha256": sha256_file(catalog_path),
        "current_brief_sha256": sha256_file(brief_path),
        "current_plan_prompt_sha256": _sha256_text(plan_prompt),
        "model_id": MODEL_ID,
    }
    missing = [key for key in expected_legacy if key not in legacy]
    mismatches = [
        key
        for key, expected in expected_legacy.items()
        if key in legacy and legacy[key] != expected
    ]
    known_system_hashes = {
        _sha256_text(EDITORIAL_SYSTEM_INSTRUCTION),
        _sha256_text(VISUAL_EVIDENCE_SYSTEM_INSTRUCTION),
    }
    if legacy.get("system_instruction_sha256") not in known_system_hashes:
        mismatches.append("system_instruction_sha256")
    if missing or mismatches:
        details = sorted(set(missing + mismatches))
        raise ValueError(
            "legacy feature plan reuse evidence does not match current inputs: "
            + ", ".join(details)
        )

    request = read_json(request_path)
    if not isinstance(request, dict):
        raise ValueError("original feature plan request must be an object")
    response_format = request.get("response_format")
    inputs = request.get("input")
    text_inputs = (
        [item.get("text") for item in inputs if item.get("type") == "text"]
        if isinstance(inputs, list)
        and all(isinstance(item, dict) for item in inputs)
        else []
    )
    expected_prompt_prefix = plan_prompt + "\n\n## 本次不可變 metadata\n"
    request_schema = (
        response_format.get("schema") if isinstance(response_format, dict) else None
    )
    request_is_valid = (
        request.get("model") == MODEL_ID
        and request.get("system_instruction") == EDITORIAL_SYSTEM_INSTRUCTION
        and request_schema == gemini_response_schema(FeatureEditPlan)
        and len(text_inputs) == 1
        and isinstance(text_inputs[0], str)
        and text_inputs[0].startswith(expected_prompt_prefix)
    )
    if not request_is_valid:
        raise ValueError(
            "original feature plan request does not prove the current "
            "model/system/schema/prompt contract"
        )

    binding = _current_feature_plan_binding(
        catalog_path=catalog_path,
        brief_path=brief_path,
        plan_path=plan_path,
        plan_prompt=plan_prompt,
        request_path=request_path,
        created_at=utc_now(),
        origin="migrated_legacy_reuse",
    )
    binding["migration_source_path"] = str(legacy_path.resolve())
    binding["migration_source_sha256"] = sha256_file(legacy_path)
    return binding


def _write_incremental_pricing(
    *,
    output_dir: Path,
    prior_interaction_hashes: dict[str, str],
    prior_error_hashes: dict[str, str],
) -> dict[str, Any]:
    """Persist a best-effort cost delta without hiding the original failure."""

    try:
        incremental_interaction_paths = [
            path
            for path in output_dir.rglob("*.raw_interaction.json")
            if prior_interaction_hashes.get(str(path.relative_to(output_dir)))
            != sha256_file(path)
        ]
        result = summarize_usage_files(
            incremental_interaction_paths,
            relative_to=output_dir,
        )
        changed_error_paths = [
            path
            for path in output_dir.rglob("errors.json")
            if prior_error_hashes.get(str(path.relative_to(output_dir)))
            != sha256_file(path)
        ]
        result.update(
            {
                "scope": "new_or_changed_raw_interactions_in_this_run",
                "historical_cache_excluded": True,
                "changed_error_artifact_count": len(changed_error_paths),
                "changed_error_artifact_paths": [
                    str(path.relative_to(output_dir))
                    for path in changed_error_paths
                ],
                "changed_error_artifacts_have_no_usage_metadata": True,
                "calculation_status": "ok",
            }
        )
    except Exception as error:  # preserve an earlier render/API exception
        result = {
            "scope": "new_or_changed_raw_interactions_in_this_run",
            "historical_cache_excluded": True,
            "calculation_status": "error",
            "calculation_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    try:
        write_json(output_dir / "pricing.incremental.json", result)
    except Exception as error:  # do not replace the render/API exception
        result["persistence_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    return result


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: Any, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _render_text_layer(
    chapter: FeatureChapterBrief,
    output_path: Path,
    *,
    dimensions: tuple[int, int],
    missing_evidence: bool = False,
    opaque: bool = False,
) -> None:
    width, height = dimensions
    image = Image.new("RGBA", dimensions, (11, 14, 18, 255 if opaque else 0))
    draw = ImageDraw.Draw(image)
    title_font = _font(54 if width > height else 48)
    detail_font = _font(34 if width > height else 31)
    label_font = _font(23 if width > height else 24)
    panel_height = round(height * (0.35 if width < height else 0.30))
    top = height - panel_height
    draw.rectangle((0, top, width, height), fill=(8, 12, 16, 218 if not opaque else 255))
    draw.rectangle((0, top, 14 if width > height else 10, height), fill=(29, 196, 96, 255))
    margin = 64 if width > height else 48
    y = top + 36
    for line in _wrap_text(draw, chapter.title, title_font, width - margin * 2):
        draw.text((margin, y), line, font=title_font, fill="white")
        y += title_font.size + 9
    y += 5
    for detail in chapter.detail_lines:
        for line in _wrap_text(draw, detail, detail_font, width - margin * 2):
            draw.text((margin, y), line, font=detail_font, fill=(220, 231, 225, 255))
            y += detail_font.size + 6
    if missing_evidence:
        label = "CATALOG 中未找到直接功能示範畫面"
        box = draw.textbbox((0, 0), label, font=label_font)
        label_width = box[2] - box[0] + 28
        draw.rounded_rectangle(
            (margin, max(22, top - 58), margin + label_width, max(22, top - 58) + 42),
            radius=10,
            fill=(211, 70, 70, 235),
        )
        draw.text((margin + 14, max(22, top - 51)), label, font=label_font, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _chapter_bounds(
    frame: RushFrame,
    clip: RushClip,
    duration_seconds: float,
    shot_cache: dict[str, ShotManifest],
    shots_dir: Path,
    scdet_threshold: float,
) -> tuple[int, int, str]:
    if clip.clip_id not in shot_cache:
        shot_cache[clip.clip_id] = detect_shots_ffmpeg(
            Path(clip.path),
            threshold=scdet_threshold,
            output_path=shots_dir / f"{clip.clip_id}.json",
        )
    shot = next(
        item
        for item in shot_cache[clip.clip_id].shots
        if item.start_time_ms <= frame.requested_time_ms < item.end_time_ms
    )
    start_ms, end_ms = _segment_bounds(
        center_ms=frame.requested_time_ms,
        requested_duration_ms=round(duration_seconds * 1000),
        clip_duration_ms=clip.duration_ms,
        shot=shot,
    )
    return start_ms, end_ms, shot.shot_id


def _load_trim_decisions(
    paths: Sequence[Path],
    *,
    allow_proposed_preview: bool = False,
) -> list[tuple[Path, TrimIntentDecision]]:
    accepted: list[tuple[Path, TrimIntentDecision]] = []
    for path in paths:
        decision = TrimIntentDecision.model_validate(read_json(path))
        is_approved = (
            decision.usable
            and decision.approval_status == "approved"
            and not decision.requires_human_review
            and decision.human_review is not None
            and decision.human_review.decision == "approved"
        )
        is_proposed_preview = (
            allow_proposed_preview
            and decision.usable
            and decision.approval_status == "proposed"
            and decision.requires_human_review
            and decision.human_review is None
        )
        if not is_approved and not is_proposed_preview:
            qualifier = (
                "human-approved or, with --allow-proposed-trim-preview, "
                "an unreviewed proposed"
            )
            raise ValueError(f"feature cut only accepts {qualifier} trim decision: {path}")
        if (
            not decision.usable
            or decision.approval_status == "rejected"
        ):
            raise ValueError(f"feature cut refuses unusable or rejected trim decision: {path}")
        accepted.append((path.resolve(), decision))
    return accepted


def _chapter_bounds_with_approved_trim(
    frame: RushFrame,
    clip: RushClip,
    duration_seconds: float,
    shot_cache: dict[str, ShotManifest],
    shots_dir: Path,
    scdet_threshold: float,
    approved_decisions: Sequence[tuple[Path, TrimIntentDecision]],
) -> tuple[int, int, str, dict[str, Any]]:
    fallback_start, fallback_end, shot_id = _chapter_bounds(
        frame,
        clip,
        duration_seconds,
        shot_cache,
        shots_dir,
        scdet_threshold,
    )
    asset_id = f"sha256:{clip.sha256}"
    matches = [
        (path, decision)
        for path, decision in approved_decisions
        if decision.source_asset_id == asset_id
        and decision.shot_id == shot_id
        and decision.source_in_ms is not None
        and decision.source_out_ms is not None
    ]
    if not matches:
        return fallback_start, fallback_end, shot_id, {
            "trim_method": "keyframe_centered_requested_duration",
            "trim_decision_path": None,
            "trim_event_id": None,
            "trim_tail_intent": None,
            "trim_human_review": None,
        }
    if len(matches) > 1:
        raise ValueError(
            f"multiple trim decisions match the selected source shot for {frame.frame_id}; "
            "event mapping is ambiguous"
        )
    path, decision = matches[0]
    assert decision.source_in_ms is not None and decision.source_out_ms is not None
    shot = next(item for item in shot_cache[clip.clip_id].shots if item.shot_id == shot_id)
    if decision.shot_id != shot_id:
        raise ValueError(
            f"approved trim decision shot differs from current FFmpeg shot for {frame.frame_id}"
        )
    if not (
        shot.start_time_ms
        <= decision.source_in_ms
        < decision.source_out_ms
        <= shot.end_time_ms
    ):
        raise ValueError("approved trim decision crosses the selected shot boundary")
    approved = decision.approval_status == "approved"
    return decision.source_in_ms, decision.source_out_ms, shot_id, {
        "trim_method": (
            "human_approved_frame_id_pts"
            if approved
            else "unreviewed_proposed_frame_id_pts"
        ),
        "trim_decision_path": str(path),
        "trim_event_id": decision.event_id,
        "trim_tail_intent": decision.tail_intent,
        "trim_requires_human_review": decision.requires_human_review,
        "trim_human_review": (
            decision.human_review.model_dump(mode="json")
            if decision.human_review is not None
            else None
        ),
    }


def _run_ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _run_segment_encoder(command: list[str]) -> None:
    try:
        _run_ffmpeg(command)
    except subprocess.CalledProcessError:
        if "h264_videotoolbox" not in command:
            raise
        fallback = ["libx264" if value == "h264_videotoolbox" else value for value in command]
        _run_ffmpeg(fallback)


def _render_source_segment(
    *,
    source_path: Path,
    start_ms: int,
    end_ms: int,
    overlay_path: Path | None,
    base_filter: str,
    output_path: Path,
    source_has_audio: bool | None = None,
) -> str:
    duration = (end_ms - start_ms) / 1000
    audio_fade_out = max(0.0, duration - 0.12)
    if source_has_audio is None:
        source_has_audio = has_audio_stream(source_path)
    if overlay_path is None:
        filter_graph = base_filter + ";[base]null[v]"
        overlay_input: list[str] = []
    else:
        filter_graph = (
            base_filter
            + ";[1:v]format=rgba[card];"
            + "[base][card]overlay=0:0:shortest=1[v]"
        )
        overlay_input = ["-loop", "1", "-i", str(overlay_path)]
    if source_has_audio:
        audio_input: list[str] = []
        audio_map = "0:a:0"
        audio_origin = "source"
    else:
        silence_input_index = 2 if overlay_path is not None else 1
        audio_input = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        audio_map = f"{silence_input_index}:a:0"
        audio_origin = "synthetic_silence"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.partial.mp4")
    _run_segment_encoder(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_ms / 1000:.3f}",
            "-i",
            str(source_path),
            *overlay_input,
            *audio_input,
            "-t",
            f"{duration:.3f}",
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            audio_map,
            "-af",
            (
                "volume=0.58,afade=t=in:st=0:d=0.08,"
                f"afade=t=out:st={audio_fade_out:.3f}:d=0.12"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "8M",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]
    )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temporary_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ]
    )
    temporary_path.replace(output_path)
    return audio_origin


def _render_missing_segment(
    chapter: FeatureChapterBrief,
    output_path: Path,
    overlay_path: Path,
    dimensions: tuple[int, int],
) -> None:
    _render_text_layer(
        chapter,
        overlay_path,
        dimensions=dimensions,
        missing_evidence=True,
        opaque=True,
    )
    duration = chapter.target_duration_seconds
    _run_segment_encoder(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(overlay_path),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            f"{duration:.3f}",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "8M",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            str(output_path),
        ]
    )


def _concat_segments(segment_paths: Sequence[Path], output_path: Path) -> None:
    if not segment_paths:
        raise ValueError("cannot concatenate an empty segment list")
    inputs: list[str] = []
    filter_inputs: list[str] = []
    for index, path in enumerate(segment_paths):
        inputs.extend(["-i", str(path.resolve())])
        filter_inputs.extend([f"[{index}:v:0]", f"[{index}:a:0]"])
    filter_graph = "".join(filter_inputs) + f"concat=n={len(segment_paths)}:v=1:a=1[v][a]"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.partial.mp4")
    _run_segment_encoder(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-b:v",
            "8M",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(temporary_path),
        ]
    )
    expected_duration = sum(_probe_duration_seconds(path) for path in segment_paths)
    actual_duration = _probe_duration_seconds(temporary_path)
    if abs(actual_duration - expected_duration) > 0.25:
        raise RuntimeError(
            f"assembled duration mismatch: expected={expected_duration:.3f}s "
            f"actual={actual_duration:.3f}s"
        )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(temporary_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ]
    )
    temporary_path.replace(output_path)


def _output_media_metadata(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=codec_name,codec_type,width,height,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video")
    audio = next(
        (stream for stream in payload["streams"] if stream["codec_type"] == "audio"),
        None,
    )
    return {
        "sha256": sha256_file(path),
        "duration_seconds": float(payload["format"]["duration"]),
        "size_bytes": int(payload["format"]["size"]),
        "video_codec": video["codec_name"],
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_rate": video["r_frame_rate"],
        "video_frames": int(video["nb_frames"]),
        "has_audio": audio is not None,
        "audio_codec": audio["codec_name"] if audio is not None else None,
    }


def _probe_duration_seconds(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(completed.stdout)["format"]["duration"])


def _segment_is_valid(
    path: Path, *, expected_duration: float, dimensions: tuple[int, int]
) -> bool:
    if not path.exists():
        return False
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return False
    try:
        payload = json.loads(probe.stdout)
        stream = payload["streams"][0]
        duration = float(payload["format"]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if (stream["width"], stream["height"]) != dimensions:
        return False
    if abs(duration - expected_duration) > 0.15:
        return False
    decode = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    return decode.returncode == 0


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _track_geometry_fingerprint(track: SegmentationTrack) -> str:
    """Fingerprint every consumed tracking sample and its model/source provenance."""
    return _stable_fingerprint(track.model_dump(mode="json"))


def _segment_variant_fingerprint(
    *,
    source_sha256: str,
    start_ms: int,
    end_ms: int,
    filter_graph: str,
    geometry: dict[str, Any],
    track_fingerprint: str | None,
) -> str:
    return _stable_fingerprint(
        {
            "contract_version": "feature-segment-render-v2",
            "source_sha256": source_sha256,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "filter_graph": filter_graph,
            "geometry": geometry,
            "track_fingerprint": track_fingerprint,
        }
    )


def _usable_track_centers(
    track: SegmentationTrack,
) -> tuple[list[float], list[float], list[list[int]]]:
    """Return only geometry eligible for unattended rendering.

    LOW_CONFIDENCE masks remain in the track artifact for review, but using
    them to drive a render would turn a warning into an implicit acceptance.
    """
    usable_states = {TrackingState.TRACKED}
    times: list[float] = []
    centers: list[float] = []
    boxes: list[list[int]] = []
    for sample in track.samples:
        if (
            sample.tracking_state in usable_states
            and sample.center_2d is not None
            and sample.derived_tracking_box is not None
        ):
            times.append(
                (sample.analysis_sample_time_ms - track.analysis_start_ms) / 1000
            )
            centers.append(float(sample.center_2d[0]))
            boxes.append([int(value) for value in sample.derived_tracking_box])
    return times, centers, boxes


def _track_confidence_diagnostics(track: SegmentationTrack) -> dict[str, Any]:
    """Derive the render gate from samples instead of trusting a summary."""

    total = len(track.samples)
    low_confidence_count = sum(
        sample.tracking_state == TrackingState.LOW_CONFIDENCE
        for sample in track.samples
    )
    return {
        "tracking_sample_count": total,
        "low_confidence_sample_count": low_confidence_count,
        "low_confidence_sample_ratio": round(
            low_confidence_count / total if total else 1.0, 6
        ),
        "tracking_confidence_gate_passed": low_confidence_count == 0,
    }


def _orientation_corrected_track_dimensions(
    tracks: Sequence[SegmentationTrack],
) -> tuple[int, int, dict[str, Any]]:
    """Resolve and validate the coordinate lineage used by Grounding and SAM."""

    if not tracks:
        raise ValueError("track_source_geometry_mismatch:no_tracks")
    source_dimensions = {
        (
            getattr(track, "seed_source_width", None),
            getattr(track, "seed_source_height", None),
        )
        for track in tracks
    }
    if None in {value for dimensions in source_dimensions for value in dimensions}:
        raise ValueError("track_source_geometry_mismatch:missing_seed_dimensions")
    if len(source_dimensions) != 1:
        raise ValueError("track_source_geometry_mismatch:required_tracks_disagree")
    source_width, source_height = next(iter(source_dimensions))
    if not isinstance(source_width, int) or not isinstance(source_height, int):
        raise ValueError("track_source_geometry_mismatch:invalid_seed_dimensions")
    source_aspect = source_width / source_height
    analysis_aspect_errors: list[float] = []
    for track in tracks:
        analysis_width = getattr(track, "analysis_width", None)
        analysis_height = getattr(track, "analysis_height", None)
        if not isinstance(analysis_width, int) or not isinstance(analysis_height, int):
            raise ValueError("track_source_geometry_mismatch:missing_analysis_dimensions")
        analysis_aspect = analysis_width / analysis_height
        relative_error = abs(analysis_aspect / source_aspect - 1)
        tolerance = max(0.01, 2 / min(analysis_width, analysis_height))
        if relative_error > tolerance:
            raise ValueError(
                "track_source_geometry_mismatch:analysis_aspect_disagrees"
            )
        analysis_aspect_errors.append(relative_error)
    return source_width, source_height, {
        "source_geometry_lineage_passed": True,
        "orientation_basis": "ffmpeg_autorotated_display",
        "source_display_width": source_width,
        "source_display_height": source_height,
        "max_analysis_aspect_relative_error": round(
            max(analysis_aspect_errors, default=0.0), 9
        ),
    }


def _horizontal_reframe_failure_geometry(
    zoom_intent: str,
    *,
    fallback_reason: str,
    risk_code: str,
    diagnostics: Mapping[str, Any] | None = None,
    geometry_safe_max_zoom: float | None = None,
) -> dict[str, Any]:
    """Describe a requested tracked reframe that was not safely applied."""

    requested = {"subtle": 1.12, "detail": 1.35}[zoom_intent]
    return {
        "requested_zoom": requested,
        "geometry_safe_max_zoom": geometry_safe_max_zoom,
        "applied_zoom": 1.0,
        "fallback_reason": fallback_reason,
        "risk_codes": list(
            dict.fromkeys([risk_code, "requested_tracked_reframe_not_applied"])
        ),
        "requires_gemini_review": True,
        **dict(diagnostics or {}),
    }


def _smooth(values: Sequence[float], alpha: float = 0.34) -> list[float]:
    if not values:
        return []
    forward = [float(values[0])]
    for value in values[1:]:
        forward.append(alpha * float(value) + (1 - alpha) * forward[-1])
    backward = [forward[-1]]
    for value in reversed(forward[:-1]):
        backward.append(alpha * value + (1 - alpha) * backward[-1])
    return list(reversed(backward))


def _piecewise_expression(times: Sequence[float], values: Sequence[float]) -> str:
    if not times or len(times) != len(values):
        raise ValueError("crop expression needs aligned non-empty times and values")
    if len(times) == 1:
        return f"{values[0]:.3f}"
    expression = f"{values[-1]:.3f}"
    for index in range(len(times) - 2, -1, -1):
        t0, t1 = times[index], times[index + 1]
        x0, x1 = values[index], values[index + 1]
        delta = max(0.001, t1 - t0)
        linear = f"{x0:.3f}+({x1 - x0:.3f})*(t-{t0:.3f})/{delta:.3f}"
        expression = f"if(lt(t\\,{t1:.3f})\\,{linear}\\,{expression})"
    # Do not extrapolate before the first observed analysis frame.  FFmpeg
    # otherwise evaluates the first linear segment at negative relative time,
    # which can move the crop away from the seed before tracking evidence exists.
    return f"if(lt(t\\,{times[0]:.3f})\\,{values[0]:.3f}\\,{expression})"


def _projected_smooth(
    desired: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    iterations: int = 12,
) -> list[float]:
    """Smooth a crop path while keeping every point inside its legal interval."""
    if not desired or not (len(desired) == len(lower) == len(upper)):
        raise ValueError("projected smoothing needs aligned non-empty values")
    if any(low > high for low, high in zip(lower, upper, strict=True)):
        raise ValueError("projected smoothing received an empty legal interval")

    values = [
        max(low, min(high, float(value)))
        for value, low, high in zip(desired, lower, upper, strict=True)
    ]
    for _ in range(iterations):
        previous = list(values)
        for index, target in enumerate(desired):
            neighbors: list[float] = []
            if index:
                neighbors.append(previous[index - 1])
            if index + 1 < len(previous):
                neighbors.append(previous[index + 1])
            neighbor_mean = sum(neighbors) / len(neighbors) if neighbors else float(target)
            proposal = 0.58 * float(target) + 0.42 * neighbor_mean
            values[index] = max(lower[index], min(upper[index], proposal))
    return values


def _even_ceil(value: float) -> int:
    return max(2, int(math.ceil(value / 2) * 2))


def _cover_transform(
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    *,
    zoom: float = 1.0,
) -> dict[str, Any]:
    """Return one deterministic, aspect-preserving cover transform."""

    if min(source_width, source_height, output_width, output_height) <= 0:
        raise ValueError("cover transform dimensions must be positive")
    if zoom < 1.0:
        raise ValueError("cover transform zoom must be at least 1")
    source_aspect = source_width / source_height
    output_aspect = output_width / output_height
    if source_aspect >= output_aspect:
        scaled_height = _even_ceil(output_height * zoom)
        scaled_width = _even_ceil(scaled_height * source_aspect)
    else:
        scaled_width = _even_ceil(output_width * zoom)
        scaled_height = _even_ceil(scaled_width / source_aspect)
    if scaled_width < output_width or scaled_height < output_height:
        raise ValueError("cover transform failed to cover the output viewport")
    active_pan_axes = [
        axis
        for axis, active in (
            ("x", scaled_width > output_width),
            ("y", scaled_height > output_height),
        )
        if active
    ]
    return {
        "contract_version": "aspect-preserving-cover-v1",
        "orientation_basis": "ffmpeg_autorotated_display",
        "scale_policy": "aspect_preserving_cover",
        "source_display_width": source_width,
        "source_display_height": source_height,
        "source_aspect_ratio": round(source_aspect, 9),
        "output_aspect_ratio": round(output_aspect, 9),
        "zoom": round(zoom, 6),
        "scaled_width": scaled_width,
        "scaled_height": scaled_height,
        "crop_width": output_width,
        "crop_height": output_height,
        "origin": "top_left",
        "normalized_track_space": "orientation_corrected_source_0_1000",
        "normalized_box_order": "x_min_y_min_x_max_y_max",
        "active_pan_axes": active_pan_axes,
        "aspect_ratio_relative_error": round(
            abs((scaled_width / scaled_height) / source_aspect - 1), 9
        ),
    }


def _axis_crop_constraints(
    *,
    padded_min: float,
    padded_max: float,
    viewport_normalized: float,
    overflow_policy: Literal["preserve_all", "controlled_clip"],
    edge_priority: Literal["balanced", "preserve_start", "preserve_end"],
) -> tuple[float, float, bool, bool]:
    max_origin = max(0.0, 1000.0 - viewport_normalized)
    lower = max(0.0, padded_max - viewport_normalized)
    upper = min(max_origin, padded_min)
    if lower <= upper + 1e-6:
        return lower, upper, True, False
    if edge_priority == "preserve_start":
        aligned = max(0.0, min(max_origin, padded_min))
    elif edge_priority == "preserve_end":
        aligned = max(0.0, min(max_origin, padded_max - viewport_normalized))
    else:
        aligned = max(
            0.0,
            min(
                max_origin,
                (padded_min + padded_max) / 2 - viewport_normalized / 2,
            ),
        )
    return aligned, aligned, False, overflow_policy == "controlled_clip"


def _tracked_crop_geometry(
    times: Sequence[float],
    centers_x: Sequence[float],
    boxes: Sequence[Sequence[int]],
    *,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    zoom: float = 1.0,
    safety_multiplier: float = 1.0,
    overflow_policy: Literal["preserve_all", "controlled_clip"] = "preserve_all",
    edge_priority: Literal["balanced", "preserve_start", "preserve_end"] = "balanced",
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Return a 2D crop path projected into per-sample safety constraints.

    Boxes are required-region union boxes in the project's canonical
    ``[x_min, y_min, x_max, y_max]`` normalized coordinate system.  When a
    union can fit, every rendered keyframe contains it (plus the requested
    safety margin).  ``controlled_clip`` is explicit overflow behavior for a
    region that is wider than the portrait viewport; it never masquerades as
    full containment.
    """
    if not times or len(times) != len(centers_x) or len(times) != len(boxes):
        raise ValueError("tracked crop geometry needs aligned non-empty samples")
    if safety_multiplier < 1.0:
        raise ValueError("safety_multiplier must be at least 1")
    transform = _cover_transform(
        source_width,
        source_height,
        output_width,
        output_height,
        zoom=zoom,
    )
    scaled_width = int(transform["scaled_width"])
    scaled_height = int(transform["scaled_height"])
    crop_width = output_width
    crop_height = output_height
    crop_width_normalized = crop_width * 1000 / scaled_width
    crop_height_normalized = crop_height * 1000 / scaled_height
    source_crop_x_max_normalized = 1000 - crop_width_normalized
    source_crop_y_max_normalized = 1000 - crop_height_normalized
    centers_y: list[float] = []
    validated_boxes: list[list[float]] = []
    for box in boxes:
        if len(box) != 4:
            raise ValueError("tracked crop boxes must contain four coordinates")
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        if not 0 <= x_min < x_max <= 1000 or not 0 <= y_min < y_max <= 1000:
            raise ValueError("tracked crop box coordinates are invalid")
        validated_boxes.append([x_min, y_min, x_max, y_max])
        centers_y.append((y_min + y_max) / 2)
    smooth_centers_x = _smooth(centers_x)
    smooth_centers_y = _smooth(centers_y)
    desired_left = [
        max(
            0.0,
            min(
                source_crop_x_max_normalized,
                center - crop_width_normalized / 2,
            ),
        )
        for center in smooth_centers_x
    ]
    desired_top = [
        max(
            0.0,
            min(
                source_crop_y_max_normalized,
                center - crop_height_normalized / 2,
            ),
        )
        for center in smooth_centers_y
    ]
    legal_left_lower: list[float] = []
    legal_left_upper: list[float] = []
    legal_top_lower: list[float] = []
    legal_top_upper: list[float] = []
    full_containment_x: list[bool] = []
    full_containment_y: list[bool] = []
    controlled_clip_samples: list[bool] = []
    margins_x: list[float] = []
    margins_y: list[float] = []
    for x_min, y_min, x_max, y_max in validated_boxes:
        width = x_max - x_min
        height = y_max - y_min
        margin_x = width * (safety_multiplier - 1) / 2
        margin_y = height * (safety_multiplier - 1) / 2
        padded_x_min = max(0.0, x_min - margin_x)
        padded_x_max = min(1000.0, x_max + margin_x)
        padded_y_min = max(0.0, y_min - margin_y)
        padded_y_max = min(1000.0, y_max + margin_y)
        x_lower, x_upper, x_fits, x_controlled = _axis_crop_constraints(
            padded_min=padded_x_min,
            padded_max=padded_x_max,
            viewport_normalized=crop_width_normalized,
            overflow_policy=overflow_policy,
            edge_priority=edge_priority,
        )
        y_lower, y_upper, y_fits, y_controlled = _axis_crop_constraints(
            padded_min=padded_y_min,
            padded_max=padded_y_max,
            viewport_normalized=crop_height_normalized,
            overflow_policy=overflow_policy,
            edge_priority=edge_priority,
        )
        legal_left_lower.append(x_lower)
        legal_left_upper.append(x_upper)
        legal_top_lower.append(y_lower)
        legal_top_upper.append(y_upper)
        full_containment_x.append(x_fits)
        full_containment_y.append(y_fits)
        controlled_clip_samples.append(x_controlled or y_controlled)
        margins_x.append(margin_x)
        margins_y.append(margin_y)

    full_containment = [
        x_fits and y_fits
        for x_fits, y_fits in zip(
            full_containment_x, full_containment_y, strict=True
        )
    ]
    geometry_feasible = overflow_policy == "controlled_clip" or all(full_containment)
    if geometry_feasible:
        crop_left_normalized = _projected_smooth(
            desired_left,
            legal_left_lower,
            legal_left_upper,
        )
        crop_top_normalized = _projected_smooth(
            desired_top,
            legal_top_lower,
            legal_top_upper,
        )
    else:
        crop_left_normalized = [
            max(low, min(high, desired))
            for desired, low, high in zip(
                desired_left, legal_left_lower, legal_left_upper, strict=True
            )
        ]
        crop_top_normalized = [
            max(low, min(high, desired))
            for desired, low, high in zip(
                desired_top, legal_top_lower, legal_top_upper, strict=True
            )
        ]
    x_values = [value * scaled_width / 1000 for value in crop_left_normalized]
    y_values = [value * scaled_height / 1000 for value in crop_top_normalized]
    max_target_width = max(box[2] - box[0] for box in validated_boxes)
    max_target_height = max(box[3] - box[1] for box in validated_boxes)
    keyframes: list[dict[str, Any]] = []
    containment_failures = 0
    minimum_visible_width_fraction = 1.0
    minimum_visible_height_fraction = 1.0
    minimum_visible_area_fraction = 1.0
    for (
        time,
        center_x,
        center_y,
        smooth_center_x,
        smooth_center_y,
        crop_x,
        crop_y,
        crop_left,
        crop_top,
        box,
        left_low,
        left_high,
        top_low,
        top_high,
        margin_x,
        margin_y,
        contained_by_construction,
        controlled,
    ) in zip(
        times,
        centers_x,
        centers_y,
        smooth_centers_x,
        smooth_centers_y,
        x_values,
        y_values,
        crop_left_normalized,
        crop_top_normalized,
        validated_boxes,
        legal_left_lower,
        legal_left_upper,
        legal_top_lower,
        legal_top_upper,
        margins_x,
        margins_y,
        full_containment,
        controlled_clip_samples,
        strict=True,
    ):
        x_min, y_min, x_max, y_max = box
        padded_x_min = max(0.0, x_min - margin_x)
        padded_x_max = min(1000.0, x_max + margin_x)
        padded_y_min = max(0.0, y_min - margin_y)
        padded_y_max = min(1000.0, y_max + margin_y)
        visible_width = max(
            0.0,
            min(x_max, crop_left + crop_width_normalized) - max(x_min, crop_left),
        )
        visible_height = max(
            0.0,
            min(y_max, crop_top + crop_height_normalized) - max(y_min, crop_top),
        )
        visible_width_fraction = visible_width / (x_max - x_min)
        visible_height_fraction = visible_height / (y_max - y_min)
        visible_area_fraction = visible_width_fraction * visible_height_fraction
        minimum_visible_width_fraction = min(
            minimum_visible_width_fraction, visible_width_fraction
        )
        minimum_visible_height_fraction = min(
            minimum_visible_height_fraction, visible_height_fraction
        )
        minimum_visible_area_fraction = min(
            minimum_visible_area_fraction, visible_area_fraction
        )
        contained = (
            padded_x_min >= crop_left - 1e-6
            and padded_x_max <= crop_left + crop_width_normalized + 1e-6
            and padded_y_min >= crop_top - 1e-6
            and padded_y_max <= crop_top + crop_height_normalized + 1e-6
        )
        if not contained:
            containment_failures += 1
        keyframes.append(
            {
                "time_seconds": round(time, 6),
                "tracked_center_x_normalized": round(center_x, 4),
                "tracked_center_y_normalized": round(center_y, 4),
                "smoothed_center_x_normalized": round(smooth_center_x, 4),
                "smoothed_center_y_normalized": round(smooth_center_y, 4),
                "required_union_box": [int(value) for value in box],
                "legal_crop_left_min_normalized": round(left_low, 4),
                "legal_crop_left_max_normalized": round(left_high, 4),
                "legal_crop_top_min_normalized": round(top_low, 4),
                "legal_crop_top_max_normalized": round(top_high, 4),
                "effective_margin_x_normalized": round(margin_x, 4),
                "effective_margin_y_normalized": round(margin_y, 4),
                "effective_margin_normalized": round(margin_x, 4),
                "crop_x_pixels": round(crop_x, 3),
                "crop_y_pixels": round(crop_y, 3),
                "required_union_contained": contained,
                "full_containment_feasible": contained_by_construction,
                "controlled_clip_applied": controlled,
                "visible_required_width_fraction": round(
                    visible_width_fraction, 6
                ),
                "visible_required_height_fraction": round(
                    visible_height_fraction, 6
                ),
                "visible_required_area_fraction": round(visible_area_fraction, 6),
            }
        )

    x_velocities = []
    y_velocities = []
    combined_velocities = []
    for t0, t1, x0, x1, y0, y1 in zip(
        times[:-1],
        times[1:],
        x_values[:-1],
        x_values[1:],
        y_values[:-1],
        y_values[1:],
        strict=True,
    ):
        delta_seconds = max(0.001, t1 - t0)
        x_velocity = abs(x1 - x0) / delta_seconds
        y_velocity = abs(y1 - y0) / delta_seconds
        x_velocities.append(x_velocity)
        y_velocities.append(y_velocity)
        combined_velocities.append(math.hypot(x_velocity, y_velocity))
    accelerations = [
        abs(v1 - v0) / max(0.001, times[index + 2] - times[index + 1])
        for index, (v0, v1) in enumerate(
            zip(combined_velocities[:-1], combined_velocities[1:], strict=True)
        )
    ]
    source_x_edge_contacts = sum(
        box[0] <= 5 or box[2] >= 995 for box in boxes
    )
    source_y_edge_contacts = sum(
        box[1] <= 5 or box[3] >= 995 for box in boxes
    )
    source_boundary_contacts = sum(
        box[0] <= 5 or box[1] <= 5 or box[2] >= 995 or box[3] >= 995
        for box in boxes
    )
    return x_values, y_values, {
        "crop_width_normalized": round(crop_width_normalized, 4),
        "crop_height_normalized": round(crop_height_normalized, 4),
        "max_target_width_normalized": max_target_width,
        "max_target_height_normalized": max_target_height,
        "overflow_policy": overflow_policy,
        "edge_priority": edge_priority,
        "geometry_feasible": geometry_feasible,
        "full_containment_feasible": all(full_containment),
        "controlled_clip_applied": any(controlled_clip_samples),
        "containment_failure_count": containment_failures,
        "minimum_visible_required_width_fraction": round(
            minimum_visible_width_fraction, 6
        ),
        "minimum_visible_required_height_fraction": round(
            minimum_visible_height_fraction, 6
        ),
        "minimum_visible_required_area_fraction": round(
            minimum_visible_area_fraction, 6
        ),
        "max_crop_x_speed_pixels_per_second": round(
            max(x_velocities, default=0.0), 4
        ),
        "max_crop_y_speed_pixels_per_second": round(
            max(y_velocities, default=0.0), 4
        ),
        "max_crop_speed_pixels_per_second": round(
            max(combined_velocities, default=0.0), 4
        ),
        "max_crop_acceleration_pixels_per_second_squared": round(
            max(accelerations, default=0.0), 4
        ),
        "source_x_edge_contact_count": source_x_edge_contacts,
        "source_y_edge_contact_count": source_y_edge_contacts,
        "source_boundary_contact_count": source_boundary_contacts,
        "source_boundary_contact_ratio": round(
            source_boundary_contacts / len(boxes), 6
        ),
        "crop_coordinate_space": transform,
        "crop_x_values_pixels": x_values,
        "crop_y_values_pixels": y_values,
        "crop_keyframes": keyframes,
    }


def _vertical_crop_geometry(
    times: Sequence[float],
    centers_x: Sequence[float],
    boxes: Sequence[Sequence[int]],
    *,
    source_width: int = 1920,
    source_height: int = 1080,
    safety_multiplier: float = 1.0,
    overflow_policy: Literal["preserve_all", "controlled_clip"] = "preserve_all",
    edge_priority: Literal["balanced", "preserve_start", "preserve_end"] = "balanced",
) -> tuple[list[float], dict[str, Any]]:
    """Compatibility wrapper for a 1080x1920 tracked crop."""

    x_values, _, audit = _tracked_crop_geometry(
        times,
        centers_x,
        boxes,
        source_width=source_width,
        source_height=source_height,
        output_width=1080,
        output_height=1920,
        safety_multiplier=safety_multiplier,
        overflow_policy=overflow_policy,
        edge_priority=edge_priority,
    )
    return x_values, audit


def _vertical_target_fits_crop(
    max_target_width_normalized: float,
    crop_width_normalized: float,
    *,
    primary_center: bool,
) -> tuple[bool, float]:
    """Primary-center may relax outer margin, never clip the selected target."""
    safety_multiplier = 1.0 if primary_center else 1.08
    return (
        max_target_width_normalized * safety_multiplier <= crop_width_normalized,
        safety_multiplier,
    )


def _required_track_union(
    tracks: Sequence[SegmentationTrack],
    *,
    region_ids: Sequence[str] | None = None,
) -> tuple[list[float], list[float], list[list[int]], dict[str, Any]]:
    """Build required-region union boxes and fail-closed coverage diagnostics."""
    if not tracks:
        raise ValueError("at least one required track is needed")
    starts = {track.analysis_start_ms for track in tracks}
    ends = {track.analysis_end_ms for track in tracks}
    rates = {float(track.analysis_fps) for track in tracks}
    if len(starts) != 1 or len(ends) != 1 or len(rates) != 1:
        raise ValueError("required tracks must share one analysis interval and rate")
    start_ms = starts.pop()
    end_ms = ends.pop()
    if end_ms is None:
        raise ValueError("required tracks must have an explicit analysis_end_ms")
    analysis_fps = rates.pop()
    labels = list(region_ids or [f"region_{index + 1}" for index in range(len(tracks))])
    if len(labels) != len(tracks):
        raise ValueError("region IDs must align with required tracks")

    # LOW_CONFIDENCE geometry remains evidence, but it cannot satisfy the
    # unattended render gate. A single required low-confidence sample forces
    # the caller onto a review-required fallback.
    usable_states = {TrackingState.TRACKED}
    all_times = sorted(
        {
            sample.analysis_sample_time_ms
            for track in tracks
            for sample in track.samples
        }
    )
    usable_by_region: dict[str, dict[int, list[int]]] = {}
    per_region: list[dict[str, Any]] = []
    low_confidence_required_sample_count = 0
    required_sample_count = 0
    low_confidence_region_ids: list[str] = []
    for label, track in zip(labels, tracks, strict=True):
        diagnostics = _track_confidence_diagnostics(track)
        low_confidence_required_sample_count += int(
            diagnostics["low_confidence_sample_count"]
        )
        required_sample_count += int(diagnostics["tracking_sample_count"])
        if not diagnostics["tracking_confidence_gate_passed"]:
            low_confidence_region_ids.append(label)
        usable = {
            sample.analysis_sample_time_ms: [
                int(value) for value in sample.derived_tracking_box
            ]
            for sample in track.samples
            if sample.tracking_state in usable_states
            and sample.derived_tracking_box is not None
        }
        usable_by_region[label] = usable
        per_region.append(
            {
                "region_id": label,
                "target_description": track.target_description,
                "state_counts": {
                    str(key): value for key, value in track.state_counts.items()
                },
                "usable_sample_count": len(usable),
                "total_sample_count": len(track.samples),
                **diagnostics,
            }
        )

    common_times = [
        time_ms
        for time_ms in all_times
        if all(time_ms in usable for usable in usable_by_region.values())
    ]
    boxes: list[list[int]] = []
    for time_ms in common_times:
        members = [usable_by_region[label][time_ms] for label in labels]
        boxes.append(
            [
                min(box[0] for box in members),
                min(box[1] for box in members),
                max(box[2] for box in members),
                max(box[3] for box in members),
            ]
        )
    centers = [(box[0] + box[2]) / 2 for box in boxes]
    times = [(time_ms - start_ms) / 1000 for time_ms in common_times]

    expected_interval_ms = 1000 / analysis_fps
    head_gap_ms = common_times[0] - start_ms if common_times else end_ms - start_ms
    tail_gap_ms = end_ms - common_times[-1] if common_times else end_ms - start_ms
    internal_gaps = [
        following - current
        for current, following in zip(common_times[:-1], common_times[1:], strict=True)
    ]
    max_internal_gap_ms = max(internal_gaps, default=0)
    unavailable_count = len(all_times) - len(common_times)
    unavailable_ratio = unavailable_count / len(all_times) if all_times else 1.0
    max_edge_gap_ms = expected_interval_ms * 1.35 + 35
    max_allowed_internal_gap_ms = expected_interval_ms * 2.25 + 35
    tracking_confidence_gate_passed = low_confidence_required_sample_count == 0
    coverage_passed = (
        tracking_confidence_gate_passed
        and len(common_times) >= 2
        and unavailable_ratio <= 0.20
        and head_gap_ms <= max_edge_gap_ms
        and tail_gap_ms <= max_edge_gap_ms
        and max_internal_gap_ms <= max_allowed_internal_gap_ms
    )
    coverage = {
        "required_region_count": len(tracks),
        "required_region_ids": labels,
        "expected_sample_count": len(all_times),
        "usable_union_sample_count": len(common_times),
        "unavailable_required_sample_count": unavailable_count,
        "unavailable_required_sample_ratio": round(unavailable_ratio, 6),
        "low_confidence_required_sample_count": low_confidence_required_sample_count,
        "low_confidence_required_sample_ratio": round(
            low_confidence_required_sample_count / required_sample_count
            if required_sample_count
            else 1.0,
            6,
        ),
        "low_confidence_region_ids": low_confidence_region_ids,
        "tracking_confidence_gate_passed": tracking_confidence_gate_passed,
        "analysis_head_gap_ms": round(head_gap_ms, 3),
        "analysis_tail_gap_ms": round(tail_gap_ms, 3),
        "max_internal_gap_ms": round(max_internal_gap_ms, 3),
        "expected_sample_interval_ms": round(expected_interval_ms, 3),
        "max_allowed_edge_gap_ms": round(max_edge_gap_ms, 3),
        "max_allowed_internal_gap_ms": round(max_allowed_internal_gap_ms, 3),
        "coverage_passed": coverage_passed,
        "per_region": per_region,
    }
    return times, centers, boxes, coverage


def _horizontal_filter_from_track(
    track: SegmentationTrack,
    zoom_intent: str,
    *,
    display_sample_aspect_ratio: float = 1.0,
) -> tuple[str, dict[str, Any]]:
    diagnostics = _track_confidence_diagnostics(track)
    if not math.isclose(display_sample_aspect_ratio, 1.0, rel_tol=0, abs_tol=1e-6):
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason="non_square_pixel_aspect_ratio_requires_static_reframe",
            risk_code="non_square_pixel_aspect_ratio_requires_static_reframe",
            diagnostics={
                **diagnostics,
                "source_display_sample_aspect_ratio": round(
                    display_sample_aspect_ratio, 9
                ),
                "sample_aspect_ratio_normalized_by_ffmpeg": True,
            },
        )
    try:
        source_width, source_height, lineage = (
            _orientation_corrected_track_dimensions([track])
        )
    except ValueError as error:
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason=str(error),
            risk_code="track_source_geometry_mismatch",
            diagnostics={
                **diagnostics,
                "source_geometry_lineage_passed": False,
            },
        )
    if not diagnostics["tracking_confidence_gate_passed"]:
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason="tracking_confidence_gate_failed",
            risk_code="tracking_low_confidence",
            diagnostics={**diagnostics, **lineage},
        )
    times, centers_x, boxes = _usable_track_centers(track)
    if len(times) < 2:
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason="fewer_than_two_usable_tracking_samples",
            risk_code="insufficient_high_confidence_tracking_samples",
            diagnostics={**diagnostics, **lineage},
        )
    requested = {"subtle": 1.12, "detail": 1.35}[zoom_intent]
    base_transform = _cover_transform(
        source_width,
        source_height,
        1920,
        1080,
    )
    max_width = max(box[2] - box[0] for box in boxes)
    max_height = max(box[3] - box[1] for box in boxes)
    safe_max = min(
        2.0,
        1920
        / (max_width / 1000 * int(base_transform["scaled_width"]) * 1.45),
        1080
        / (max_height / 1000 * int(base_transform["scaled_height"]) * 1.45),
    )
    applied = max(1.0, min(requested, safe_max))
    if applied < 1.035:
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason="mask_geometry_left_no_safe_zoom_margin",
            risk_code="tracked_reframe_no_safe_zoom_margin",
            diagnostics={**diagnostics, **lineage},
            geometry_safe_max_zoom=round(safe_max, 4),
        )
    x_values, y_values, crop_audit = _tracked_crop_geometry(
        times,
        centers_x,
        boxes,
        source_width=source_width,
        source_height=source_height,
        output_width=1920,
        output_height=1080,
        zoom=applied,
        safety_multiplier=1.45,
    )
    if (
        not crop_audit["full_containment_feasible"]
        or crop_audit["containment_failure_count"] != 0
    ):
        return _horizontal_original_filter(), _horizontal_reframe_failure_geometry(
            zoom_intent,
            fallback_reason="tracked_reframe_containment_gate_failed",
            risk_code="tracked_reframe_required_region_not_contained",
            diagnostics={**diagnostics, **lineage, **crop_audit},
            geometry_safe_max_zoom=round(safe_max, 4),
        )
    coordinate_space = crop_audit["crop_coordinate_space"]
    scaled_width = int(coordinate_space["scaled_width"])
    scaled_height = int(coordinate_space["scaled_height"])
    x_expression = _piecewise_expression(times, x_values)
    y_expression = _piecewise_expression(times, y_values)
    return (
        f"[0:v]fps=30,scale={scaled_width}:{scaled_height},"
        f"crop=1920:1080:x='{x_expression}':y='{y_expression}',setsar=1[base]",
        {
            "requested_zoom": requested,
            "geometry_safe_max_zoom": round(safe_max, 4),
            "applied_zoom": round(applied, 4),
            "fallback_reason": None,
            "risk_codes": [],
            "requires_gemini_review": False,
            **diagnostics,
            **lineage,
            **crop_audit,
        },
    )


def _horizontal_original_filter() -> str:
    return (
        "[0:v]fps=30,"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        "scale=1920:1080:force_original_aspect_ratio=increase,"
        "crop=1920:1080:x=(iw-ow)/2:y=(ih-oh)/2,setsar=1[base]"
    )


def _vertical_seed_anchor_fallback(
    tracks: Sequence[SegmentationTrack],
    *,
    source_width: int,
    source_height: int,
    coverage: dict[str, Any],
    allow_subject_clipping: bool,
    overflow_policy: Literal["preserve_all", "controlled_clip"],
    edge_priority: Literal["balanced", "preserve_start", "preserve_end"],
    fallback_strategy: Literal["fit_with_background", "center_crop"],
    failure_reason: str,
) -> tuple[str, dict[str, Any]] | None:
    """Use already-grounded seed geometry when propagation is incomplete.

    This is deliberately limited to the no-background fallback path.  It does
    not claim motion coverage: the static anchor is held for the shot and the
    result remains review-required.  The policy is domain-neutral and applies
    equally to subjects, text, UI, graphics, and other visible regions.
    """
    if fallback_strategy != "center_crop" or not tracks:
        return None
    seed_times = {track.seed_time_ms for track in tracks}
    if len(seed_times) != 1:
        return None
    anchor_boxes = [list(track.semantic_seed_box) for track in tracks]
    if any(len(box) != 4 for box in anchor_boxes):
        return None
    anchor_union = [
        min(box[0] for box in anchor_boxes),
        min(box[1] for box in anchor_boxes),
        max(box[2] for box in anchor_boxes),
        max(box[3] for box in anchor_boxes),
    ]
    start_ms = tracks[0].analysis_start_ms
    end_ms = tracks[0].analysis_end_ms
    if end_ms is None or end_ms <= start_ms:
        return None
    duration_seconds = (end_ms - start_ms) / 1000
    safety_multiplier = 1.0 if allow_subject_clipping else 1.08
    x_values, y_values, crop_audit = _tracked_crop_geometry(
        [0.0, duration_seconds],
        [(anchor_union[0] + anchor_union[2]) / 2] * 2,
        [anchor_union, anchor_union],
        source_width=source_width,
        source_height=source_height,
        output_width=1080,
        output_height=1920,
        safety_multiplier=safety_multiplier,
        overflow_policy=overflow_policy,
        edge_priority=edge_priority,
    )
    if overflow_policy == "preserve_all" and (
        not crop_audit["full_containment_feasible"]
        or crop_audit["containment_failure_count"] != 0
    ):
        return None
    controlled_clip_applied = bool(crop_audit["controlled_clip_applied"])
    risk_codes = [
        "required_region_tracking_coverage_failed",
        "seed_anchor_static_hold",
        "motion_outside_seed_unverified",
    ]
    if not coverage.get("tracking_confidence_gate_passed", True):
        risk_codes.append("required_region_low_confidence")
    if controlled_clip_applied:
        risk_codes.append("controlled_required_region_clip")
    if int(crop_audit["source_boundary_contact_count"]) > 0:
        risk_codes.extend(["source_boundary_contact", "not_recoverable_by_pan"])
    x_expression = _piecewise_expression([0.0, duration_seconds], x_values)
    y_expression = _piecewise_expression([0.0, duration_seconds], y_values)
    coordinate_space = crop_audit["crop_coordinate_space"]
    return (
        "[0:v]fps=30,"
        f"scale={coordinate_space['scaled_width']}:{coordinate_space['scaled_height']},"
        f"crop=1080:1920:x='{x_expression}':y='{y_expression}',setsar=1[base]",
        {
            "applied_strategy": "seed_anchor_crop",
            "fallback_reason": f"{failure_reason}_used_static_seed_anchor",
            "seed_anchor_time_ms": next(iter(seed_times)),
            "seed_anchor_union_box_2d": anchor_union,
            "subject_clipping_allowed": controlled_clip_applied,
            "secondary_context_clipping_allowed": allow_subject_clipping,
            "target_safety_multiplier": safety_multiplier,
            "risk_codes": list(dict.fromkeys(risk_codes)),
            "requires_gemini_review": True,
            "source_geometry_lineage_passed": True,
            "orientation_basis": "ffmpeg_autorotated_display",
            "source_display_width": source_width,
            "source_display_height": source_height,
            **coverage,
            **crop_audit,
        },
    )


def _vertical_filter_from_track(
    track: SegmentationTrack | Sequence[SegmentationTrack],
    *,
    allow_subject_clipping: bool = False,
    overflow_policy: Literal["preserve_all", "controlled_clip"] = "preserve_all",
    edge_priority: Literal["balanced", "preserve_start", "preserve_end"] = "balanced",
    region_ids: Sequence[str] | None = None,
    fallback_strategy: Literal["fit_with_background", "center_crop"] = (
        "fit_with_background"
    ),
    display_sample_aspect_ratio: float = 1.0,
) -> tuple[str, dict[str, Any]]:
    fallback_filter = (
        _vertical_center_crop_filter()
        if fallback_strategy == "center_crop"
        else _vertical_fit_filter()
    )
    tracks = [track] if isinstance(track, SegmentationTrack) else list(track)
    if not math.isclose(display_sample_aspect_ratio, 1.0, rel_tol=0, abs_tol=1e-6):
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": (
                "non_square_pixel_aspect_ratio_requires_static_reframe"
            ),
            "risk_codes": [
                "non_square_pixel_aspect_ratio_requires_static_reframe"
            ],
            "requires_gemini_review": True,
            "source_display_sample_aspect_ratio": round(
                display_sample_aspect_ratio, 9
            ),
            "sample_aspect_ratio_normalized_by_ffmpeg": True,
        }
    times, centers_x, boxes, coverage = _required_track_union(
        tracks,
        region_ids=region_ids,
    )
    try:
        source_width, source_height, lineage = (
            _orientation_corrected_track_dimensions(tracks)
        )
    except ValueError as error:
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": str(error),
            "risk_codes": ["track_source_geometry_mismatch"],
            "requires_gemini_review": True,
            "source_geometry_lineage_passed": False,
            **coverage,
        }
    confidence_gate_failed = not coverage["tracking_confidence_gate_passed"]
    coverage_risk_codes = ["required_region_unavailable"]
    if confidence_gate_failed:
        coverage_risk_codes.insert(0, "required_region_low_confidence")
    if len(times) < 2:
        failure_reason = (
            "required_region_tracking_confidence_failed"
            if confidence_gate_failed
            else "fewer_than_two_usable_tracking_samples"
        )
        anchor_fallback = _vertical_seed_anchor_fallback(
            tracks,
            source_width=source_width,
            source_height=source_height,
            coverage=coverage,
            allow_subject_clipping=allow_subject_clipping,
            overflow_policy=overflow_policy,
            edge_priority=edge_priority,
            fallback_strategy=fallback_strategy,
            failure_reason=failure_reason,
        )
        if anchor_fallback is not None:
            return anchor_fallback
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": failure_reason,
            "risk_codes": coverage_risk_codes,
            "requires_gemini_review": True,
            **lineage,
            **coverage,
        }
    if not coverage["coverage_passed"]:
        failure_reason = (
            "required_region_tracking_confidence_failed"
            if confidence_gate_failed
            else "required_region_tracking_coverage_failed"
        )
        anchor_fallback = _vertical_seed_anchor_fallback(
            tracks,
            source_width=source_width,
            source_height=source_height,
            coverage=coverage,
            allow_subject_clipping=allow_subject_clipping,
            overflow_policy=overflow_policy,
            edge_priority=edge_priority,
            fallback_strategy=fallback_strategy,
            failure_reason=failure_reason,
        )
        if anchor_fallback is not None:
            return anchor_fallback
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": failure_reason,
            "risk_codes": coverage_risk_codes,
            "requires_gemini_review": True,
            **lineage,
            **coverage,
        }
    target_safety_multiplier = 1.0 if allow_subject_clipping else 1.08
    x_values, y_values, crop_audit = _tracked_crop_geometry(
        times,
        centers_x,
        boxes,
        source_width=source_width,
        source_height=source_height,
        output_width=1080,
        output_height=1920,
        safety_multiplier=target_safety_multiplier,
        overflow_policy=overflow_policy,
        edge_priority=edge_priority,
    )
    crop_width_normalized = float(crop_audit["crop_width_normalized"])
    crop_height_normalized = float(crop_audit["crop_height_normalized"])
    max_target_width = int(crop_audit["max_target_width_normalized"])
    max_target_height = int(crop_audit["max_target_height_normalized"])
    target_fits_legacy, _ = _vertical_target_fits_crop(
        max_target_width,
        crop_width_normalized,
        primary_center=allow_subject_clipping,
    )
    full_containment_feasible = bool(crop_audit["full_containment_feasible"])
    if overflow_policy == "preserve_all" and not full_containment_feasible:
        width_too_large = (
            max_target_width * target_safety_multiplier > crop_width_normalized
        )
        height_too_large = (
            max_target_height * target_safety_multiplier > crop_height_normalized
        )
        size_risk_codes = []
        if width_too_large:
            size_risk_codes.append("required_region_too_wide")
        if height_too_large:
            size_risk_codes.append("required_region_too_tall")
        if not size_risk_codes:
            size_risk_codes.append("required_region_not_containable")
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": "required_region_union_too_large_for_safe_9x16_crop",
            "subject_clipping_allowed": False,
            "secondary_context_clipping_allowed": allow_subject_clipping,
            "target_safety_multiplier": target_safety_multiplier,
            "legacy_max_width_gate_passed": target_fits_legacy,
            "risk_codes": size_risk_codes,
            "requires_gemini_review": True,
            **lineage,
            **coverage,
            **crop_audit,
        }
    if (
        overflow_policy == "preserve_all"
        and int(crop_audit["containment_failure_count"]) != 0
    ):
        return fallback_filter, {
            "applied_strategy": fallback_strategy,
            "fallback_reason": "required_region_containment_gate_failed",
            "subject_clipping_allowed": False,
            "secondary_context_clipping_allowed": allow_subject_clipping,
            "target_safety_multiplier": target_safety_multiplier,
            "legacy_max_width_gate_passed": target_fits_legacy,
            "risk_codes": ["required_region_not_contained"],
            "requires_gemini_review": True,
            **lineage,
            **coverage,
            **crop_audit,
        }
    x_expression = _piecewise_expression(times, x_values)
    y_expression = _piecewise_expression(times, y_values)
    controlled_clip_applied = bool(crop_audit["controlled_clip_applied"])
    risk_codes = ["controlled_required_region_clip"] if controlled_clip_applied else []
    edge_hold_warning_ms = float(coverage["expected_sample_interval_ms"]) * 0.55 + 35
    if (
        float(coverage["analysis_head_gap_ms"]) > edge_hold_warning_ms
        or float(coverage["analysis_tail_gap_ms"]) > edge_hold_warning_ms
    ):
        risk_codes.append("analysis_edge_hold_long")
    if int(crop_audit["source_boundary_contact_count"]) > 0:
        risk_codes.extend(["source_boundary_contact", "not_recoverable_by_pan"])
    if float(crop_audit["max_crop_speed_pixels_per_second"]) > 720:
        risk_codes.append("crop_motion_fast")
    if float(crop_audit["max_crop_acceleration_pixels_per_second_squared"]) > 1800:
        risk_codes.append("crop_motion_acceleration_high")
    coordinate_space = crop_audit["crop_coordinate_space"]
    return (
        "[0:v]fps=30,"
        f"scale={coordinate_space['scaled_width']}:{coordinate_space['scaled_height']},"
        f"crop=1080:1920:x='{x_expression}':y='{y_expression}',setsar=1[base]",
        {
            "applied_strategy": "tracked_crop",
            "fallback_reason": None,
            "subject_clipping_allowed": controlled_clip_applied,
            "secondary_context_clipping_allowed": allow_subject_clipping,
            "target_safety_multiplier": target_safety_multiplier,
            "legacy_max_width_gate_passed": target_fits_legacy,
            "risk_codes": risk_codes,
            "requires_gemini_review": bool(risk_codes),
            **lineage,
            **coverage,
            **crop_audit,
        },
    )


def _vertical_fit_filter() -> str:
    return (
        "[0:v]fps=30,"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        "split=2[background_source][foreground_source];"
        "[background_source]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920:x=(iw-ow)/2:y=(ih-oh)/2,gblur=sigma=28[background];"
        "[foreground_source]scale=1080:1920:force_original_aspect_ratio=decrease"
        "[foreground];"
        "[background][foreground]overlay=(W-w)/2:(H-h)/2,setsar=1[base]"
    )


def _vertical_center_crop_filter() -> str:
    return (
        "[0:v]fps=30,"
        "scale='max(2,trunc(iw*sar/2)*2)':ih,setsar=1,"
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920:x=(iw-ow)/2:y=(ih-oh)/2,setsar=1[base]"
    )


def _tracking_seed_request_ms(
    frame: RushFrame,
    start_ms: int,
    end_ms: int,
) -> tuple[int, str]:
    if start_ms <= frame.requested_time_ms < end_ms:
        return frame.requested_time_ms, "catalog_anchor"
    return start_ms + (end_ms - start_ms) // 2, "trim_midpoint"


def _has_complete_cached_primary_track(output_dir: Path) -> bool:
    """A degraded composite fallback may only reuse complete local evidence."""

    return (
        any(output_dir.glob("grounding/bbox-*/grounding.json"))
        and any(output_dir.glob("sam21/bbox-*/segmentation-track.json"))
    )


def _is_non_retryable_spending_cap_error(error: Exception) -> bool:
    message = str(error).lower()
    return "spending cap" in message or "monthly spend" in message


def _ground_tracking_seed(
    *,
    client: GeminiLabClient,
    clip: RushClip,
    frame: RushFrame,
    start_ms: int,
    end_ms: int,
    feature_id: str,
    event_description: str,
    entity_id: str,
    target_description: str,
    grounding_prompt: str,
    output_dir: Path,
    run_id: str,
    model_request_block_reason: str | None = None,
) -> tuple[GroundingProposal, Any, Any, Any, Path, int, str]:
    """Ground one immutable semantic region on one exact decoded source frame."""
    exact_frame_path = output_dir / "evidence-frame.png"
    seed_requested_time_ms, seed_anchor_source = _tracking_seed_request_ms(
        frame,
        start_ms,
        end_ms,
    )
    exact_frame = extract_frame(
        Path(clip.path),
        seed_requested_time_ms,
        exact_frame_path,
    )
    media = probe_video(Path(clip.path))
    grounding_key = {
        "contract_version": (
            "exact-frame-grounding-v2"
            if entity_id == "reframe_subject"
            else "exact-frame-grounding-v3-region-intent"
        ),
        "model_id": MODEL_ID,
        "source_asset_id": media.asset_id,
        "temperature": client.temperature,
        "feature_id": feature_id,
        "frame_hash": exact_frame.frame_hash,
        "frame_pts": exact_frame.frame_pts,
        "frame_time_ms": exact_frame.frame_time_ms,
        "source_width": exact_frame.width,
        "source_height": exact_frame.height,
        "entity_id": entity_id,
        "event_description": event_description,
        "target_description": target_description,
        "prompt_sha256": hashlib.sha256(grounding_prompt.encode("utf-8")).hexdigest(),
        "system_instruction_sha256": hashlib.sha256(
            VISUAL_EVIDENCE_SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "response_schema_sha256": hashlib.sha256(
            json.dumps(
                gemini_response_schema(GeminiNativeGroundingProposal),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "thinking_level": "low",
    }
    if seed_anchor_source != "catalog_anchor":
        grounding_key.update(
            {
                "catalog_frame_id": frame.frame_id,
                "catalog_requested_time_ms": frame.requested_time_ms,
                "seed_anchor_source": seed_anchor_source,
                "seed_requested_time_ms": seed_requested_time_ms,
            }
        )
    grounding_fingerprint = hashlib.sha256(
        json.dumps(
            grounding_key,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    grounding_dir = output_dir / "grounding" / f"bbox-{grounding_fingerprint[:16]}"
    write_json(
        grounding_dir / "request-key.json",
        {**grounding_key, "request_fingerprint": grounding_fingerprint},
    )
    grounding_path = grounding_dir / "grounding.json"
    if grounding_path.exists():
        proposal = GroundingProposal.model_validate(read_json(grounding_path))
        require_grounding_request_match(
            proposal,
            asset_id=media.asset_id,
            event_id=feature_id,
            entity_id=entity_id,
            frame_pts=exact_frame.frame_pts,
            frame_time_ms=exact_frame.frame_time_ms,
            frame_hash=exact_frame.frame_hash,
            source_width=exact_frame.width,
            source_height=exact_frame.height,
            model_id=MODEL_ID,
        )
        frame_time_ms = proposal.frame_time_ms
    else:
        if model_request_block_reason:
            raise RuntimeError(
                "Gemini Grounding request skipped by run-level circuit breaker: "
                + model_request_block_reason
            )
        proposal = client.ground_frame(
            media=media,
            frame=exact_frame,
            event_id=feature_id,
            event_description=event_description,
            entity_id=entity_id,
            target_description=target_description,
            prompt_template=grounding_prompt,
            run_id=run_id,
            output_dir=grounding_dir,
        )
        frame_time_ms = exact_frame.frame_time_ms
    debug_path = grounding_dir / "debug.png"
    if not debug_path.exists():
        draw_grounding_overlay(exact_frame_path, proposal, debug_path)
    if debug_path.exists():
        shutil.copy2(debug_path, output_dir / "grounding-debug.png")
    if not proposal.visible or not proposal.candidates:
        raise ValueError(f"Gemini could not ground required region {entity_id} for {feature_id}")
    selected_seed = require_tracking_seed_candidate(proposal)
    return (
        proposal,
        selected_seed,
        exact_frame,
        media,
        grounding_path,
        frame_time_ms,
        seed_anchor_source,
    )


def _build_track(
    *,
    client: GeminiLabClient,
    clip: RushClip,
    frame: RushFrame,
    start_ms: int,
    end_ms: int,
    feature_id: str,
    event_description: str,
    target_description: str,
    checkpoint_path: Path,
    grounding_prompt: str,
    output_dir: Path,
    run_id: str,
    analysis_fps: float,
    scdet_threshold: float,
    entity_id: str = "reframe_subject",
    model_request_block_reason: str | None = None,
) -> tuple[GroundingProposal, SegmentationTrack]:
    track_root = output_dir / "sam21"
    (
        proposal,
        selected_seed,
        exact_frame,
        media,
        _,
        frame_time_ms,
        seed_anchor_source,
    ) = _ground_tracking_seed(
        client=client,
        clip=clip,
        frame=frame,
        start_ms=start_ms,
        end_ms=end_ms,
        feature_id=feature_id,
        event_description=event_description,
        entity_id=entity_id,
        target_description=target_description,
        grounding_prompt=grounding_prompt,
        output_dir=output_dir,
        run_id=run_id,
        model_request_block_reason=model_request_block_reason,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    seed_manifest = {
        "contract_version": "bbox-seed-v2-exact-pts",
        "asset_id": proposal.asset_id,
        "event_id": proposal.event_id,
        "entity_id": proposal.entity_id,
        "target_description": target_description,
        "frame_hash": proposal.frame_hash,
        "frame_pts": proposal.frame_pts,
        "candidate_number": selected_seed.candidate_number,
        "candidate_index": selected_seed.candidate_index,
        "candidate_selection_source": selected_seed.selection_source,
        "box_2d": list(selected_seed.candidate.box_2d),
        "seed_type": "gemini_bbox",
        "source_start_ms": start_ms,
        "source_end_ms": end_ms,
        "normalized_seed_shot_start_ms": start_ms,
        "normalized_seed_shot_end_ms": end_ms,
        "analysis_fps": analysis_fps,
        "analysis_max_side": _TRACKING_MAX_SIDE,
        "ffmpeg_scdet_threshold": scdet_threshold,
        "seed_box_padding_ratio": _TRACKING_SEED_BOX_PADDING_RATIO,
        "device_request": _TRACKING_DEVICE,
        "sam_config": SAM21_CONFIG,
        "sam_implementation_revision": SAM21_IMPLEMENTATION_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
    }
    seed_fingerprint = hashlib.sha256(
        json.dumps(seed_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    track_dir = track_root / f"bbox-{seed_fingerprint[:16]}"
    seed_manifest_path = track_dir / "seed-selection.json"
    write_json(seed_manifest_path, {**seed_manifest, "seed_fingerprint": seed_fingerprint})
    track_path = track_dir / "segmentation-track.json"
    if track_path.exists():
        track = SegmentationTrack.model_validate(read_json(track_path))
    else:
        track = track_bbox_sam21(
            video_path=Path(clip.path),
            checkpoint_path=checkpoint_path,
            seed_time_ms=frame_time_ms,
            seed_box_2d=selected_seed.candidate.box_2d,
            target_description=target_description,
            output_dir=track_dir,
            seed_source=str(seed_manifest_path),
            asset_id=proposal.asset_id,
            seed_frame_pts=proposal.frame_pts,
            seed_frame_sha256=proposal.frame_hash,
            seed_source_width=proposal.source_width,
            seed_source_height=proposal.source_height,
            analysis_fps=analysis_fps,
            max_side=_TRACKING_MAX_SIDE,
            device=_TRACKING_DEVICE,
            ffmpeg_scdet_threshold=scdet_threshold,
            seed_box_padding_ratio=_TRACKING_SEED_BOX_PADDING_RATIO,
            allowed_start_ms=start_ms,
            allowed_end_ms=end_ms,
        )
    require_bbox_track_request_match(
        track,
        video_path=Path(clip.path),
        asset_id=proposal.asset_id,
        target_description=target_description,
        seed_time_ms=frame_time_ms,
        seed_box_2d=selected_seed.candidate.box_2d,
        seed_box_padding_ratio=_TRACKING_SEED_BOX_PADDING_RATIO,
        analysis_fps=analysis_fps,
        analysis_start_ms=start_ms,
        analysis_end_ms=end_ms,
        checkpoint_sha256=checkpoint_sha256,
        seed_frame_pts=proposal.frame_pts,
        seed_frame_sha256=proposal.frame_hash,
        seed_source_width=proposal.source_width,
        seed_source_height=proposal.source_height,
    )
    return proposal, track


def _contained_shared_session_artifact(
    session_dir: Path,
    artifact_path: str,
    *,
    artifact_kind: str,
) -> Path:
    """Resolve a cached shared-session artifact without permitting path escape."""
    root = session_dir.expanduser().resolve(strict=True)
    resolved = (root / artifact_path).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"cached shared SAM {artifact_kind} escapes its session: {artifact_path}"
        ) from error
    if not resolved.is_file():
        raise ValueError(
            f"cached shared SAM {artifact_kind} is not a file: {artifact_path}"
        )
    return resolved


def _validate_shared_sam_session_cache(
    *,
    manifest: SharedSam21SessionManifest,
    session_dir: Path,
    video_path: Path,
    asset_id: str,
    start_ms: int,
    end_ms: int,
    analysis_fps: float,
    analysis_max_side: int,
    checkpoint_sha256: str,
    seeds: Sequence[SharedSam21BBoxSeed],
    seed_box_padding_ratio: float,
) -> list[SegmentationTrack]:
    """Validate immutable decode, model, seed, and track lineage before cache reuse."""
    resolved_video = video_path.expanduser().resolve(strict=True)
    manifest_video = Path(manifest.video_path).expanduser().resolve(strict=True)
    expected_ids = [seed.target_id for seed in seeds]
    actual_ids = [target.target_id for target in manifest.targets]
    mismatches: list[str] = []
    if manifest.asset_id != asset_id:
        mismatches.append("asset_id")
    if manifest_video != resolved_video:
        mismatches.append("video_path")
    if manifest.analysis_start_ms != start_ms:
        mismatches.append("analysis_start_ms")
    if manifest.analysis_end_ms != end_ms:
        mismatches.append("analysis_end_ms")
    if manifest.analysis_fps != analysis_fps:
        mismatches.append("analysis_fps")
    if max(manifest.analysis_width, manifest.analysis_height) > analysis_max_side:
        mismatches.append("analysis_dimensions")
    if actual_ids != expected_ids:
        mismatches.append("target_order")
    provenance = manifest.model_provenance
    if provenance.model_id != SAM21_TINY_MODEL_ID:
        mismatches.append("model_id")
    if provenance.implementation != "facebookresearch/sam2":
        mismatches.append("implementation")
    if provenance.implementation_revision != SAM21_IMPLEMENTATION_REVISION:
        mismatches.append("implementation_revision")
    if provenance.checkpoint_sha256 != checkpoint_sha256:
        mismatches.append("checkpoint_sha256")
    if mismatches:
        raise ValueError(
            "cached shared SAM session does not match request: "
            + ", ".join(mismatches)
        )

    frames_manifest_path = _contained_shared_session_artifact(
        session_dir,
        manifest.analysis_frames_path,
        artifact_kind="analysis frame manifest",
    )
    if sha256_file(frames_manifest_path) != manifest.analysis_frames_manifest_sha256:
        raise ValueError("cached shared SAM analysis frame manifest hash mismatch")
    frames_manifest = SharedSam21AnalysisFramesManifest.model_validate(
        read_json(frames_manifest_path)
    )
    if frames_manifest.frames != manifest.analysis_frames:
        raise ValueError(
            "cached shared SAM analysis frame manifest does not match session manifest"
        )
    for frame in frames_manifest.frames:
        frame_path = _contained_shared_session_artifact(
            session_dir,
            frame.path,
            artifact_kind="analysis frame",
        )
        if sha256_file(frame_path) != frame.sha256:
            raise ValueError(f"cached shared SAM analysis frame hash mismatch: {frame.path}")

    tracks: list[SegmentationTrack] = []
    for seed, member in zip(seeds, manifest.targets, strict=True):
        track_path = _contained_shared_session_artifact(
            session_dir,
            member.track_path,
            artifact_kind="track",
        )
        if sha256_file(track_path) != member.track_sha256:
            raise ValueError(f"cached shared SAM track hash mismatch: {member.target_id}")
        track = SegmentationTrack.model_validate(read_json(track_path))
        expected_prompt_box = pad_normalized_box(
            seed.seed_box_2d, seed_box_padding_ratio
        )
        track_mismatches: list[str] = []
        expected_values = {
            "asset_id": asset_id,
            "video_path": str(resolved_video),
            "target_id": seed.target_id,
            "target_description": seed.target_description,
            "seed_source": seed.seed_source,
            "seed_time_ms": seed.seed_time_ms,
            "seed_frame_pts": seed.seed_frame_pts,
            "seed_frame_sha256": seed.seed_frame_sha256,
            "seed_source_width": seed.seed_source_width,
            "seed_source_height": seed.seed_source_height,
            "semantic_seed_box": seed.seed_box_2d,
            "seed_prompt_type": "box",
            "sam_prompt_box": expected_prompt_box,
            "seed_box_padding_ratio": seed_box_padding_ratio,
            "analysis_fps": analysis_fps,
            "analysis_width": manifest.analysis_width,
            "analysis_height": manifest.analysis_height,
            "analysis_start_ms": start_ms,
            "analysis_end_ms": end_ms,
            "source_start_pts": manifest.source_start_pts,
            "source_time_base": manifest.source_time_base,
            "total_samples": len(manifest.analysis_frames),
            "state_counts": member.state_counts,
            "shared_session_id": manifest.session_id,
            "analysis_frames_manifest_sha256": (
                manifest.analysis_frames_manifest_sha256
            ),
        }
        for field, expected in expected_values.items():
            actual = getattr(track, field)
            if field == "video_path":
                actual = str(Path(actual).expanduser().resolve(strict=True))
            if actual != expected:
                track_mismatches.append(field)
        if track.model_provenance != provenance:
            track_mismatches.append("model_provenance")
        if member.target_description != seed.target_description:
            track_mismatches.append("member.target_description")
        if member.seed_time_ms != seed.seed_time_ms:
            track_mismatches.append("member.seed_time_ms")
        if member.seed_frame_pts != seed.seed_frame_pts:
            track_mismatches.append("member.seed_frame_pts")
        if member.seed_frame_sha256 != seed.seed_frame_sha256:
            track_mismatches.append("member.seed_frame_sha256")
        if member.seed_source_width != seed.seed_source_width:
            track_mismatches.append("member.seed_source_width")
        if member.seed_source_height != seed.seed_source_height:
            track_mismatches.append("member.seed_source_height")
        if track_mismatches:
            raise ValueError(
                f"cached shared SAM track {seed.target_id!r} does not match request: "
                + ", ".join(track_mismatches)
            )
        tracks.append(track)

    validate_segmentation_track_alignment(tracks)
    return tracks


def _build_required_region_tracks(
    *,
    client: GeminiLabClient,
    clip: RushClip,
    frame: RushFrame,
    start_ms: int,
    end_ms: int,
    feature_id: str,
    event_description: str,
    regions: Sequence[FramingRegionIntent],
    checkpoint_path: Path,
    grounding_prompt: str,
    output_dir: Path,
    analysis_fps: float,
    scdet_threshold: float,
    model_request_block_reason: str | None = None,
) -> tuple[list[GroundingProposal], list[SegmentationTrack], list[Path]]:
    """Ground required regions separately and share one SAM session when possible."""
    required = [region for region in regions if region.role == "required"]
    if not required:
        raise ValueError("a tracked portrait crop needs at least one required region")
    if len(required) == 1:
        region = required[0]
        region_root = output_dir / "regions" / region.region_id
        proposal, track = _build_track(
            client=client,
            clip=clip,
            frame=frame,
            start_ms=start_ms,
            end_ms=end_ms,
            feature_id=feature_id,
            event_description=event_description,
            target_description=region.target_description,
            checkpoint_path=checkpoint_path,
            grounding_prompt=grounding_prompt,
            output_dir=region_root,
            run_id=f"feature-v-{region.region_id}-{uuid.uuid4().hex[:8]}",
            analysis_fps=analysis_fps,
            scdet_threshold=scdet_threshold,
            entity_id=f"reframe_{region.region_id}",
            model_request_block_reason=model_request_block_reason,
        )
        return [proposal], [track], [region_root / "grounding-debug.png"]

    proposals: list[GroundingProposal] = []
    seeds: list[SharedSam21BBoxSeed] = []
    debug_paths: list[Path] = []
    for region in required:
        region_root = output_dir / "regions" / region.region_id
        (
            proposal,
            selected_seed,
            _,
            _,
            grounding_path,
            frame_time_ms,
            _,
        ) = _ground_tracking_seed(
            client=client,
            clip=clip,
            frame=frame,
            start_ms=start_ms,
            end_ms=end_ms,
            feature_id=feature_id,
            event_description=event_description,
            entity_id=f"reframe_{region.region_id}",
            target_description=region.target_description,
            grounding_prompt=grounding_prompt,
            output_dir=region_root,
            run_id=f"feature-v-{region.region_id}-{uuid.uuid4().hex[:8]}",
            model_request_block_reason=model_request_block_reason,
        )
        proposals.append(proposal)
        debug_paths.append(region_root / "grounding-debug.png")
        seeds.append(
            SharedSam21BBoxSeed(
                target_id=region.region_id,
                target_description=region.target_description,
                seed_source=str(grounding_path.resolve()),
                seed_time_ms=frame_time_ms,
                seed_frame_pts=proposal.frame_pts,
                seed_frame_sha256=proposal.frame_hash,
                seed_source_width=proposal.source_width,
                seed_source_height=proposal.source_height,
                seed_box_2d=list(selected_seed.candidate.box_2d),
            )
        )

    request_key = {
        "contract_version": "feature-cut-shared-required-regions-v1",
        "asset_id": proposals[0].asset_id,
        "video_path": str(Path(clip.path).expanduser().resolve()),
        "feature_id": feature_id,
        "source_start_ms": start_ms,
        "source_end_ms": end_ms,
        "analysis_fps": analysis_fps,
        "analysis_max_side": _TRACKING_MAX_SIDE,
        "ffmpeg_scdet_threshold": scdet_threshold,
        "seed_box_padding_ratio": _TRACKING_SEED_BOX_PADDING_RATIO,
        "device_request": _TRACKING_DEVICE,
        "sam_config": SAM21_CONFIG,
        "sam_implementation_revision": SAM21_IMPLEMENTATION_REVISION,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "targets": [seed.model_dump(mode="json") for seed in seeds],
    }
    request_fingerprint = hashlib.sha256(
        json.dumps(request_key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    session_parent = output_dir / "shared-sam21"
    session_dir = session_parent / f"session-{request_fingerprint[:16]}"
    manifest_path = session_dir / "shared-session.json"
    if manifest_path.exists():
        manifest = SharedSam21SessionManifest.model_validate(read_json(manifest_path))
    else:
        if session_dir.exists() and any(session_dir.iterdir()):
            raise RuntimeError(f"incomplete shared SAM session: {session_dir}")
        manifest = track_bboxes_shared_sam21(
            video_path=Path(clip.path),
            checkpoint_path=checkpoint_path,
            targets=seeds,
            output_dir=session_dir,
            asset_id=proposals[0].asset_id,
            analysis_fps=analysis_fps,
            max_side=_TRACKING_MAX_SIDE,
            device=_TRACKING_DEVICE,
            ffmpeg_scdet_threshold=scdet_threshold,
            seed_box_padding_ratio=_TRACKING_SEED_BOX_PADDING_RATIO,
            allowed_start_ms=start_ms,
            allowed_end_ms=end_ms,
        )
    session_parent.mkdir(parents=True, exist_ok=True)
    write_json(
        session_parent / f"session-{request_fingerprint[:16]}.request.json",
        {**request_key, "request_fingerprint": request_fingerprint},
    )
    tracks = _validate_shared_sam_session_cache(
        manifest=manifest,
        session_dir=session_dir,
        video_path=Path(clip.path),
        asset_id=proposals[0].asset_id,
        start_ms=start_ms,
        end_ms=end_ms,
        analysis_fps=analysis_fps,
        analysis_max_side=_TRACKING_MAX_SIDE,
        checkpoint_sha256=request_key["checkpoint_sha256"],
        seeds=seeds,
        seed_box_padding_ratio=_TRACKING_SEED_BOX_PADDING_RATIO,
    )
    return proposals, tracks, debug_paths


def _render_review_html(
    output_dir: Path,
    brief: FeatureEditBrief,
    plan: FeatureEditPlan,
    manifest: dict[str, Any],
) -> None:
    overlay_note = (
        "成片不燒錄實驗字卡；使用者 brief 只作審查 metadata。"
        if not brief.render_title_overlays
        else "成片字卡來自使用者 editorial brief。"
    )
    rows: list[str] = []
    by_id = {chapter.feature_id: chapter for chapter in plan.chapters}
    for brief_chapter in brief.chapters:
        selected = by_id[brief_chapter.feature_id]
        vertical = next(
            item for item in manifest["vertical"]["chapters"] if item["feature_id"] == brief_chapter.feature_id
        )
        horizontal = next(
            item for item in manifest["horizontal"]["chapters"] if item["feature_id"] == brief_chapter.feature_id
        )
        debug_paths = list(vertical.get("grounding_debugs") or [])
        if not debug_paths and vertical.get("grounding_debug"):
            debug_paths = [vertical["grounding_debug"]]
        debug_links: list[str] = []
        for debug_index, debug_path in enumerate(debug_paths, start=1):
            relative_debug = Path(debug_path).relative_to(output_dir.resolve())
            debug_links.append(
                f'<a href="{html.escape(str(relative_debug))}">bbox {debug_index}</a>'
            )
        debug_link = " · ".join(debug_links) or "—"
        rows.append(
            "<tr>"
            f"<td>{html.escape(brief_chapter.title)}</td>"
            f"<td>{html.escape(selected.evidence_status)}</td>"
            f"<td>{html.escape(str(selected.horizontal_frame_id))}</td>"
            f"<td>{html.escape(str(horizontal.get('applied_zoom', 1.0)))}</td>"
            f"<td>{html.escape(str(horizontal.get('trim_method', 'not_applicable')))}</td>"
            f"<td>{html.escape(str(selected.vertical_frame_id))}</td>"
            f"<td>{html.escape(vertical['applied_strategy'])}</td>"
            f"<td>{html.escape(str(vertical.get('trim_method', 'not_applicable')))}</td>"
            f"<td>{debug_link}</td>"
            f"<td>{html.escape(selected.observed_visual_evidence)}</td>"
            f"<td>{html.escape('; '.join(selected.quality_risks) or 'none')}</td>"
            "</tr>"
        )
    (output_dir / "index.html").write_text(
        """<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Feature cut review</title>
<style>body{font:15px system-ui;background:#101214;color:#eee;max-width:1500px;margin:24px auto;padding:0 20px}section{background:#1b1f24;padding:20px;margin:20px 0;border-radius:12px}video{width:min(100%,960px);max-height:76vh;background:#000}table{border-collapse:collapse;width:100%}th,td{border:1px solid #3b424a;padding:8px;text-align:left;vertical-align:top}a{color:#71e59c}</style>
<h1>Feature cut review</h1><p>"""
        + html.escape(overlay_note)
        + " 畫面證據、frame ID、Gemini bbox、SAM tracking 與 fallback 分開保存。</p>"
        + f"<section><h2>16:9</h2><video controls src=\"{html.escape(str(Path(manifest['horizontal']['output_path']).relative_to(output_dir.resolve())))}\"></video></section>"
        + f"<section><h2>9:16</h2><video controls src=\"{html.escape(str(Path(manifest['vertical']['output_path']).relative_to(output_dir.resolve())))}\"></video></section>"
        + "<table><thead><tr><th>chapter</th><th>evidence</th><th>16:9 frame</th><th>zoom</th><th>16:9 trim</th><th>9:16 frame</th><th>vertical</th><th>9:16 trim</th><th>debug</th><th>observed evidence</th><th>risks</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></html>",
        encoding="utf-8",
    )


def run_feature_cut_experiment(
    *,
    catalog_path: Path,
    brief_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    plan_prompt: str,
    grounding_prompt: str,
    temperature: float = 0.2,
    scdet_threshold: float = 4.0,
    sam_analysis_fps: float = 2.0,
    trim_decision_paths: Sequence[Path] = (),
    allow_proposed_trim_preview: bool = False,
    reuse_feature_plan: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_interaction_hashes = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in output_dir.rglob("*.raw_interaction.json")
    }
    prior_error_hashes = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in output_dir.rglob("errors.json")
    }
    catalog = RushesCatalog.model_validate(read_json(catalog_path))
    brief = FeatureEditBrief.model_validate(read_json(brief_path))
    controlled_reframe_requested = any(
        chapter.vertical_overflow_policy == "controlled_clip"
        for chapter in brief.chapters
    )
    human_reframe_policy_requested = brief.reframe_policy_binding is not None
    if controlled_reframe_requested and brief.reframe_policy_binding is None:
        raise ValueError(
            "controlled_clip requires an immutable human reframe policy sidecar"
        )
    if human_reframe_policy_requested and not reuse_feature_plan:
        raise ValueError(
            "a human reframe policy can only reuse its bound feature plan; "
            "pass --reuse-feature-plan"
        )
    plan_dir = output_dir / "gemini-plan"
    plan_path = plan_dir / "feature_edit_plan.json"
    plan_binding_path = plan_dir / "feature-plan.binding.json"
    if human_reframe_policy_requested:
        if not plan_path.is_file() or not plan_binding_path.is_file():
            raise ValueError(
                "human reframe policy bundle requires its bound feature plan and binding"
            )
        saved_human_binding = read_json(plan_binding_path)
        if (
            not isinstance(saved_human_binding, dict)
            or saved_human_binding.get("origin") != REFRAME_POLICY_BINDING_ORIGIN
        ):
            raise ValueError(
                "human reframe policy requires a human_reframe_policy plan binding"
            )
        # Validate the complete sidecar chain before probing media or creating
        # a Gemini client. A binding-shaped object is not authorization.
        validate_reframe_policy_bundle(
            catalog_path=catalog_path,
            brief_path=brief_path,
            feature_plan_path=plan_path,
            saved_plan_binding=saved_human_binding,
        )
    frames = {frame.frame_id: frame for frame in catalog.frames}
    clips = {clip.clip_id: clip for clip in catalog.clips}
    trim_decisions = _load_trim_decisions(
        trim_decision_paths,
        allow_proposed_preview=allow_proposed_trim_preview,
    )
    brief_by_id = {chapter.feature_id: chapter for chapter in brief.chapters}
    timings: dict[str, float] = {}
    incremental_pricing: dict[str, Any] = {}
    started = monotonic()
    reel_path = Path(catalog.analysis_reel_path)
    reel_media = probe_video(reel_path)
    upload_dir = catalog_path.parent / "file-cache" / reel_media.sha256 / "upload"
    client = GeminiLabClient(temperature=temperature)
    plan_reuse_record_path: Path | None = None
    gemini_geometry_block_reason: str | None = None
    write_json(
        output_dir / "geometry-model-circuit-breaker.json",
        {
            "blocked": False,
            "reason": None,
            "interpretation": "no_non_retryable_geometry_error_seen_in_this_run",
            "started_at": utc_now(),
        },
    )

    def latch_non_retryable_geometry_error(error: Exception) -> None:
        nonlocal gemini_geometry_block_reason
        if (
            gemini_geometry_block_reason is None
            and _is_non_retryable_spending_cap_error(error)
        ):
            gemini_geometry_block_reason = f"{type(error).__name__}:{error}"
            write_json(
                output_dir / "geometry-model-circuit-breaker.json",
                {
                    "blocked": True,
                    "reason": gemini_geometry_block_reason,
                    "interpretation": (
                        "non_retryable_spending_cap_latched_remaining_uncached_"
                        "grounding_requests_are_skipped"
                    ),
                    "latched_at": utc_now(),
                },
            )
    try:
        if controlled_reframe_requested and not plan_path.exists():
            raise ValueError(
                "controlled_clip policy bundle has no bound saved feature plan"
            )
        if plan_path.exists():
            if not reuse_feature_plan:
                raise ValueError(
                    "saved feature plan exists; pass --reuse-feature-plan to reuse "
                    "that editorial decision explicitly, or choose a fresh output directory"
                )
            plan = FeatureEditPlan.model_validate(read_json(plan_path))
            expected_ids = [chapter.feature_id for chapter in brief.chapters]
            actual_ids = [chapter.feature_id for chapter in plan.chapters]
            if (
                plan.project_id != brief.project_id
                or plan.catalog_id != catalog.catalog_id
                or actual_ids != expected_ids
            ):
                raise ValueError("saved feature plan does not match the current brief/catalog")
            if plan_binding_path.exists():
                saved_binding = read_json(plan_binding_path)
                if not isinstance(saved_binding, dict):
                    raise ValueError("saved feature plan binding must be an object")
                if saved_binding.get("origin") == REFRAME_POLICY_BINDING_ORIGIN:
                    current_binding = validate_reframe_policy_bundle(
                        catalog_path=catalog_path,
                        brief_path=brief_path,
                        feature_plan_path=plan_path,
                        saved_plan_binding=saved_binding,
                    )
                elif saved_binding.get("origin") == "external_projection":
                    current_binding = _current_external_projection_binding(
                        plan_dir=plan_dir,
                        catalog_path=catalog_path,
                        brief_path=brief_path,
                        plan_path=plan_path,
                        created_at=utc_now(),
                    )
                else:
                    saved_origin = saved_binding.get("origin")
                    if saved_origin not in {"generated", "migrated_legacy_reuse"}:
                        raise ValueError("saved feature plan binding origin is unsupported")
                    current_binding = _current_feature_plan_binding(
                        catalog_path=catalog_path,
                        brief_path=brief_path,
                        plan_path=plan_path,
                        plan_prompt=plan_prompt,
                        request_path=(
                            plan_dir / "feature_edit_plan.request.json"
                            if (plan_dir / "feature_edit_plan.request.json").exists()
                            else None
                        ),
                        created_at=utc_now(),
                        origin=saved_origin,
                    )
            elif (plan_dir / _EXTERNAL_PROJECTION_POINTER_NAME).exists():
                current_binding = _current_external_projection_binding(
                    plan_dir=plan_dir,
                    catalog_path=catalog_path,
                    brief_path=brief_path,
                    plan_path=plan_path,
                    created_at=utc_now(),
                )
                saved_binding = current_binding
                write_json(plan_binding_path, saved_binding)
            else:
                current_binding = _current_feature_plan_binding(
                    catalog_path=catalog_path,
                    brief_path=brief_path,
                    plan_path=plan_path,
                    plan_prompt=plan_prompt,
                    request_path=(
                        plan_dir / "feature_edit_plan.request.json"
                        if (plan_dir / "feature_edit_plan.request.json").exists()
                        else None
                    ),
                    created_at=utc_now(),
                    origin="generated",
                )
                saved_binding = _migrate_legacy_feature_plan_binding(
                    plan_dir=plan_dir,
                    catalog_path=catalog_path,
                    brief_path=brief_path,
                    plan_path=plan_path,
                    plan_prompt=plan_prompt,
                )
                write_json(plan_binding_path, saved_binding)
                current_binding["origin"] = saved_binding["origin"]
            _validate_feature_plan_binding(saved_binding, current_binding)
            reuse_event_dir = plan_dir / "feature-plan-reuse-events"
            plan_reuse_record_path = (
                reuse_event_dir / f"reuse-{uuid.uuid4().hex}.json"
            )
            write_json(
                plan_reuse_record_path,
                {
                    "interpretation": (
                        "explicit_editorial_plan_reuse_geometry_is_recomputed"
                    ),
                    "binding_path": str(plan_binding_path.resolve()),
                    "binding_sha256": sha256_file(plan_binding_path),
                    "binding_origin": current_binding["origin"],
                    "validated_causal_hashes": {
                        key: current_binding[key]
                        for key in (
                            "catalog_sha256",
                            "brief_sha256",
                            "plan_prompt_sha256",
                            "system_instruction_sha256",
                            "model_id_sha256",
                            "response_schema_sha256",
                            "plan_sha256",
                            "request_sha256",
                            "source_plan_sha256",
                            "projection_contract_sha256",
                            "projection_pointer_sha256",
                            "projection_record_sha256",
                            "source_artifact_set_sha256",
                            "reframe_policy_sidecar_sha256",
                            "source_plan_binding_sha256",
                            "selection_fingerprint",
                        )
                        if key in current_binding
                    },
                    "reused_at": utc_now(),
                },
            )
            timings["file_api_seconds"] = 0.0
            file_api_reused: bool | None = None
            timings["gemini_plan_seconds"] = 0.0
            plan_reused = True
        else:
            stage = monotonic()
            uploaded, file_api_reused = client.ensure_video_upload(reel_path, upload_dir)
            timings["file_api_seconds"] = round(monotonic() - stage, 3)
            stage = monotonic()
            plan = client.plan_feature_edit(
                catalog=catalog,
                brief=brief,
                uploaded=uploaded,
                prompt_template=plan_prompt,
                run_id=f"feature-plan-{uuid.uuid4().hex[:8]}",
                run_dir=plan_dir,
            )
            timings["gemini_plan_seconds"] = round(monotonic() - stage, 3)
            request_path = plan_dir / "feature_edit_plan.request.json"
            binding = _current_feature_plan_binding(
                catalog_path=catalog_path,
                brief_path=brief_path,
                plan_path=plan_path,
                plan_prompt=plan_prompt,
                request_path=request_path,
                created_at=utc_now(),
                origin="generated",
            )
            write_json(plan_binding_path, binding)
            plan_reused = False
        shot_cache: dict[str, ShotManifest] = {}
        shots_dir = output_dir / "shots"
        horizontal_segments: list[Path] = []
        vertical_segments: list[Path] = []
        render_config = {
            "pipeline_version": _RENDER_PIPELINE_VERSION,
            "brief": brief.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "sam_analysis_fps": sam_analysis_fps,
            "scdet_threshold": scdet_threshold,
            "trim_decisions": [
                {
                    "path": str(path),
                    "decision": decision.model_dump(mode="json"),
                }
                for path, decision in trim_decisions
            ],
            "allow_proposed_trim_preview": allow_proposed_trim_preview,
        }
        render_key = hashlib.sha256(
            json.dumps(render_config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        render_variant = (
            f"with-titles-{render_key}"
            if brief.render_title_overlays
            else f"clean-{render_key}"
        )
        manifest: dict[str, Any] = {
            "project_id": brief.project_id,
            "catalog_id": catalog.catalog_id,
            "render_title_overlays": brief.render_title_overlays,
            "render_pipeline_version": _RENDER_PIPELINE_VERSION,
            "render_cache_key": render_key,
            "feature_plan_reused": plan_reused,
            "feature_plan_binding": str(plan_binding_path.resolve()),
            "feature_plan_reuse_record": (
                str(plan_reuse_record_path.resolve())
                if plan_reuse_record_path is not None
                else None
            ),
            "reframe_policy_binding": (
                brief.reframe_policy_binding.model_dump(mode="json")
                if brief.reframe_policy_binding is not None
                else None
            ),
            "approved_trim_decision_count": sum(
                decision.approval_status == "approved" for _, decision in trim_decisions
            ),
            "unreviewed_trim_proposal_count": sum(
                decision.approval_status == "proposed" for _, decision in trim_decisions
            ),
            "contains_unreviewed_trim_proposals": any(
                decision.approval_status == "proposed" for _, decision in trim_decisions
            ),
            "trim_decisions": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "approval_status": decision.approval_status,
                    "requires_human_review": decision.requires_human_review,
                    "event_id": decision.event_id,
                    "source_asset_id": decision.source_asset_id,
                }
                for path, decision in trim_decisions
            ],
            "horizontal": {"chapters": []},
            "vertical": {"chapters": []},
        }
        track_cache: dict[tuple[str, str, int, int], tuple[GroundingProposal, SegmentationTrack, Path]] = {}
        source_audio_cache: dict[str, bool] = {}
        source_media_cache: dict[str, MediaInfo] = {}
        stage = monotonic()
        for index, selected in enumerate(plan.chapters):
            brief_chapter = brief_by_id[selected.feature_id]
            horizontal_overlay = output_dir / "overlays" / "16x9" / f"{index:02d}.png"
            vertical_overlay = output_dir / "overlays" / "9x16" / f"{index:02d}.png"
            horizontal_segment = (
                output_dir / "segments" / render_variant / "16x9" / f"{index:02d}.mp4"
            )
            vertical_segment = (
                output_dir / "segments" / render_variant / "9x16" / f"{index:02d}.mp4"
            )
            if selected.evidence_status == "not_found":
                if not _segment_is_valid(
                    horizontal_segment,
                    expected_duration=brief_chapter.target_duration_seconds,
                    dimensions=(1920, 1080),
                ):
                    _render_missing_segment(
                        brief_chapter, horizontal_segment, horizontal_overlay, (1920, 1080)
                    )
                if not _segment_is_valid(
                    vertical_segment,
                    expected_duration=brief_chapter.target_duration_seconds,
                    dimensions=(1080, 1920),
                ):
                    _render_missing_segment(
                        brief_chapter, vertical_segment, vertical_overlay, (1080, 1920)
                    )
                horizontal_entry = {
                    "feature_id": selected.feature_id,
                    "source_frame_id": None,
                    "semantic_intent": brief_chapter.title,
                    "observed_visual_evidence": selected.observed_visual_evidence,
                    "selection_reason": selected.selection_reason,
                    "duration_ms": round(brief_chapter.target_duration_seconds * 1000),
                    "source_clip_id": None,
                    "source_in_ms": None,
                    "source_out_ms": None,
                    "segment_render_fingerprint": sha256_file(horizontal_segment),
                    "segment_path": str(horizontal_segment.resolve()),
                    "applied_zoom": 1.0,
                    "fallback_reason": "catalog_evidence_not_found",
                    "audio_origin": "synthetic_silence",
                }
                vertical_entry = {
                    "feature_id": selected.feature_id,
                    "source_frame_id": None,
                    "semantic_intent": brief_chapter.title,
                    "observed_visual_evidence": selected.observed_visual_evidence,
                    "selection_reason": selected.selection_reason,
                    "duration_ms": round(brief_chapter.target_duration_seconds * 1000),
                    "source_clip_id": None,
                    "source_in_ms": None,
                    "source_out_ms": None,
                    "segment_render_fingerprint": sha256_file(vertical_segment),
                    "segment_path": str(vertical_segment.resolve()),
                    "applied_strategy": "graphic_missing_evidence_card",
                    "fallback_reason": "catalog_evidence_not_found",
                    "audio_origin": "synthetic_silence",
                }
            else:
                horizontal_frame = frames[selected.horizontal_frame_id or ""]
                horizontal_clip = clips[horizontal_frame.clip_id]
                h_start, h_end, h_shot, horizontal_trim = _chapter_bounds_with_approved_trim(
                    horizontal_frame,
                    horizontal_clip,
                    brief_chapter.target_duration_seconds,
                    shot_cache,
                    shots_dir,
                    scdet_threshold,
                    trim_decisions,
                )
                vertical_frame = frames[selected.vertical_frame_id or ""]
                vertical_clip = clips[vertical_frame.clip_id]
                for source_clip in (horizontal_clip, vertical_clip):
                    if source_clip.sha256 not in source_audio_cache:
                        source_audio_cache[source_clip.sha256] = has_audio_stream(
                            Path(source_clip.path)
                        )
                    if source_clip.sha256 not in source_media_cache:
                        source_media_cache[source_clip.sha256] = probe_video(
                            Path(source_clip.path)
                        )
                horizontal_source_has_audio = source_audio_cache[horizontal_clip.sha256]
                vertical_source_has_audio = source_audio_cache[vertical_clip.sha256]
                horizontal_source_media = source_media_cache[horizontal_clip.sha256]
                vertical_source_media = source_media_cache[vertical_clip.sha256]
                horizontal_display_sar = (
                    horizontal_source_media.video.display_sample_aspect_ratio.numerator
                    / horizontal_source_media.video.display_sample_aspect_ratio.denominator
                )
                vertical_display_sar = (
                    vertical_source_media.video.display_sample_aspect_ratio.numerator
                    / vertical_source_media.video.display_sample_aspect_ratio.denominator
                )
                v_start, v_end, v_shot, vertical_trim = _chapter_bounds_with_approved_trim(
                    vertical_frame,
                    vertical_clip,
                    brief_chapter.target_duration_seconds,
                    shot_cache,
                    shots_dir,
                    scdet_threshold,
                    trim_decisions,
                )
                if brief.render_title_overlays:
                    _render_text_layer(
                        brief_chapter, horizontal_overlay, dimensions=(1920, 1080)
                    )
                    _render_text_layer(
                        brief_chapter, vertical_overlay, dimensions=(1080, 1920)
                    )
                horizontal_filter = _horizontal_original_filter()
                horizontal_geometry = {
                    "requested_zoom": None,
                    "geometry_safe_max_zoom": None,
                    "applied_zoom": 1.0,
                    "fallback_reason": None,
                    "risk_codes": [],
                    "requires_gemini_review": False,
                }
                horizontal_debug: Path | None = None
                horizontal_track_fingerprint: str | None = None
                if selected.horizontal_strategy == "tracked_reframe":
                    target = selected.horizontal_target_description or ""
                    cache_key = (horizontal_frame.frame_id, target, h_start, h_end)
                    track_root = output_dir / "geometry" / selected.feature_id / "horizontal"
                    try:
                        if cache_key not in track_cache:
                            proposal, track = _build_track(
                                client=client,
                                clip=horizontal_clip,
                                frame=horizontal_frame,
                                start_ms=h_start,
                                end_ms=h_end,
                                feature_id=selected.feature_id,
                                event_description=(
                                    brief_chapter.title + "；" + selected.observed_visual_evidence
                                ),
                                target_description=target,
                                checkpoint_path=checkpoint_path,
                                grounding_prompt=grounding_prompt,
                                output_dir=track_root,
                                run_id=f"feature-h-{uuid.uuid4().hex[:8]}",
                                analysis_fps=sam_analysis_fps,
                                scdet_threshold=scdet_threshold,
                                model_request_block_reason=(
                                    gemini_geometry_block_reason
                                ),
                            )
                            track_cache[cache_key] = (proposal, track, track_root)
                        _, track, track_root = track_cache[cache_key]
                        horizontal_track_fingerprint = _track_geometry_fingerprint(track)
                        horizontal_filter, horizontal_geometry = _horizontal_filter_from_track(
                            track,
                            selected.horizontal_zoom_intent,
                            display_sample_aspect_ratio=horizontal_display_sar,
                        )
                        horizontal_debug = track_root / "grounding-debug.png"
                    except Exception as error:
                        latch_non_retryable_geometry_error(error)
                        horizontal_geometry = _horizontal_reframe_failure_geometry(
                            selected.horizontal_zoom_intent,
                            fallback_reason=(
                                f"tracking_or_grounding_failed:{type(error).__name__}:{error}"
                            ),
                            risk_code="tracking_or_grounding_failed",
                        )
                horizontal_segment_fingerprint = _segment_variant_fingerprint(
                    source_sha256=horizontal_clip.sha256,
                    start_ms=h_start,
                    end_ms=h_end,
                    filter_graph=horizontal_filter,
                    geometry=horizontal_geometry,
                    track_fingerprint=horizontal_track_fingerprint,
                )
                horizontal_segment = (
                    output_dir
                    / "segments"
                    / render_variant
                    / "16x9"
                    / f"{index:02d}-{horizontal_segment_fingerprint[:12]}.mp4"
                )
                if not _segment_is_valid(
                    horizontal_segment,
                    expected_duration=(h_end - h_start) / 1000,
                    dimensions=(1920, 1080),
                ):
                    _render_source_segment(
                        source_path=Path(horizontal_clip.path),
                        start_ms=h_start,
                        end_ms=h_end,
                        overlay_path=(horizontal_overlay if brief.render_title_overlays else None),
                        base_filter=horizontal_filter,
                        output_path=horizontal_segment,
                        source_has_audio=horizontal_source_has_audio,
                    )
                vertical_fallback_strategy = brief.vertical_fallback_strategy
                vertical_filter = (
                    _vertical_center_crop_filter()
                    if vertical_fallback_strategy == "center_crop"
                    else _vertical_fit_filter()
                )
                vertical_geometry: dict[str, Any] = {
                    "applied_strategy": vertical_fallback_strategy,
                    "fallback_reason": None,
                }
                vertical_debug: Path | None = None
                vertical_debugs: list[Path] = []
                vertical_track_fingerprint: str | None = None
                vertical_primary_override = brief_chapter.vertical_primary_target_description
                vertical_regions = list(brief_chapter.vertical_regions)
                required_regions = [
                    region for region in vertical_regions if region.role == "required"
                ]
                vertical_target_description = (
                    "; ".join(region.target_description for region in required_regions)
                    if required_regions
                    else vertical_primary_override
                    or selected.vertical_target_description
                )
                if (
                    selected.vertical_strategy == "tracked_crop"
                    or vertical_primary_override
                    or required_regions
                ):
                    target = vertical_target_description or ""
                    cache_key = (vertical_frame.frame_id, target, v_start, v_end)
                    vertical_geometry_root = (
                        output_dir / "geometry" / selected.feature_id / "vertical"
                    )
                    track_root = vertical_geometry_root
                    if vertical_regions:
                        region_key = hashlib.sha256(
                            json.dumps(
                                [region.model_dump(mode="json") for region in vertical_regions],
                                ensure_ascii=False,
                                sort_keys=True,
                            ).encode("utf-8")
                        ).hexdigest()[:12]
                        track_root = track_root / f"regions-{region_key}"
                    elif vertical_primary_override:
                        target_key = hashlib.sha256(target.encode("utf-8")).hexdigest()[:10]
                        track_root = track_root / f"primary-{target_key}"
                    try:
                        if required_regions:
                            _, tracks, vertical_debugs = _build_required_region_tracks(
                                client=client,
                                clip=vertical_clip,
                                frame=vertical_frame,
                                start_ms=v_start,
                                end_ms=v_end,
                                feature_id=selected.feature_id,
                                event_description=(
                                    brief_chapter.title + "；" + selected.observed_visual_evidence
                                ),
                                regions=vertical_regions,
                                checkpoint_path=checkpoint_path,
                                grounding_prompt=grounding_prompt,
                                output_dir=track_root,
                                analysis_fps=sam_analysis_fps,
                                scdet_threshold=scdet_threshold,
                                model_request_block_reason=(
                                    gemini_geometry_block_reason
                                ),
                            )
                            track_fingerprints = [
                                _track_geometry_fingerprint(track) for track in tracks
                            ]
                            vertical_track_fingerprint = hashlib.sha256(
                                json.dumps(
                                    {
                                        "regions": [
                                            region.model_dump(mode="json")
                                            for region in required_regions
                                        ],
                                        "tracks": track_fingerprints,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ).encode("utf-8")
                            ).hexdigest()
                            vertical_filter, vertical_geometry = _vertical_filter_from_track(
                                tracks,
                                allow_subject_clipping=(
                                    brief_chapter.vertical_crop_mode == "primary_center"
                                ),
                                overflow_policy=brief_chapter.vertical_overflow_policy,
                                edge_priority=brief_chapter.vertical_edge_priority,
                                region_ids=[region.region_id for region in required_regions],
                                fallback_strategy=vertical_fallback_strategy,
                                display_sample_aspect_ratio=vertical_display_sar,
                            )
                            vertical_debug = next(
                                (path for path in vertical_debugs if path.exists()), None
                            )
                        else:
                            if cache_key not in track_cache:
                                proposal, track = _build_track(
                                    client=client,
                                    clip=vertical_clip,
                                    frame=vertical_frame,
                                    start_ms=v_start,
                                    end_ms=v_end,
                                    feature_id=selected.feature_id,
                                    event_description=(
                                        brief_chapter.title
                                        + "；"
                                        + selected.observed_visual_evidence
                                    ),
                                    target_description=target,
                                    checkpoint_path=checkpoint_path,
                                    grounding_prompt=grounding_prompt,
                                    output_dir=track_root,
                                    run_id=f"feature-v-{uuid.uuid4().hex[:8]}",
                                    analysis_fps=sam_analysis_fps,
                                    scdet_threshold=scdet_threshold,
                                    model_request_block_reason=(
                                        gemini_geometry_block_reason
                                    ),
                                )
                                track_cache[cache_key] = (proposal, track, track_root)
                            _, track, track_root = track_cache[cache_key]
                            vertical_track_fingerprint = _track_geometry_fingerprint(track)
                            vertical_filter, vertical_geometry = _vertical_filter_from_track(
                                track,
                                allow_subject_clipping=(
                                    brief_chapter.vertical_crop_mode == "primary_center"
                                ),
                                overflow_policy=brief_chapter.vertical_overflow_policy,
                                edge_priority=brief_chapter.vertical_edge_priority,
                                fallback_strategy=vertical_fallback_strategy,
                                display_sample_aspect_ratio=vertical_display_sar,
                            )
                            vertical_debug = track_root / "grounding-debug.png"
                            vertical_debugs = [vertical_debug]
                        semantic_review_reasons: list[str] = []
                        if len(required_regions) > 1:
                            semantic_review_reasons.append("multiple_required_regions")
                        if any(
                            region.kind in {"text_region", "ui_region"}
                            for region in required_regions
                        ):
                            semantic_review_reasons.append("text_or_ui_region")
                        if vertical_geometry.get("fallback_reason"):
                            semantic_review_reasons.append("fallback_applied")
                        if semantic_review_reasons:
                            vertical_geometry["requires_gemini_review"] = True
                        vertical_geometry["semantic_review_reasons"] = (
                            semantic_review_reasons
                        )
                    except Exception as error:
                        latch_non_retryable_geometry_error(error)
                        primary_target = (
                            vertical_primary_override
                            or selected.vertical_target_description
                            or ""
                        )
                        primary_key = hashlib.sha256(
                            primary_target.encode("utf-8")
                        ).hexdigest()[:10]
                        primary_root = vertical_geometry_root / f"primary-{primary_key}"
                        reused_cached_composite = False
                        cached_composite_error: Exception | None = None
                        if (
                            required_regions
                            and primary_target
                            and _has_complete_cached_primary_track(primary_root)
                        ):
                            try:
                                _, track = _build_track(
                                    client=client,
                                    clip=vertical_clip,
                                    frame=vertical_frame,
                                    start_ms=v_start,
                                    end_ms=v_end,
                                    feature_id=selected.feature_id,
                                    event_description=(
                                        brief_chapter.title
                                        + "；"
                                        + selected.observed_visual_evidence
                                    ),
                                    target_description=primary_target,
                                    checkpoint_path=checkpoint_path,
                                    grounding_prompt=grounding_prompt,
                                    output_dir=primary_root,
                                    run_id=f"feature-v-cache-{uuid.uuid4().hex[:8]}",
                                    analysis_fps=sam_analysis_fps,
                                    scdet_threshold=scdet_threshold,
                                    model_request_block_reason=(
                                        gemini_geometry_block_reason
                                    ),
                                )
                                vertical_track_fingerprint = (
                                    _track_geometry_fingerprint(track)
                                )
                                vertical_filter, vertical_geometry = (
                                    _vertical_filter_from_track(
                                        track,
                                        allow_subject_clipping=(
                                            brief_chapter.vertical_crop_mode
                                            == "primary_center"
                                        ),
                                        overflow_policy=(
                                            brief_chapter.vertical_overflow_policy
                                        ),
                                        edge_priority=(
                                            brief_chapter.vertical_edge_priority
                                        ),
                                        fallback_strategy=(
                                            vertical_fallback_strategy
                                        ),
                                        display_sample_aspect_ratio=(
                                            vertical_display_sar
                                        ),
                                    )
                                )
                                prior_fallback = vertical_geometry.get(
                                    "fallback_reason"
                                )
                                vertical_geometry["fallback_reason"] = (
                                    "required_region_grounding_failed_used_cached_"
                                    f"composite_track:{type(error).__name__}"
                                    + (
                                        f":{prior_fallback}"
                                        if prior_fallback
                                        else ""
                                    )
                                )
                                vertical_geometry["risk_codes"] = list(
                                    dict.fromkeys(
                                        list(vertical_geometry.get("risk_codes") or [])
                                        + [
                                            "cached_composite_track_fallback",
                                            "required_region_contract_not_verified",
                                        ]
                                    )
                                )
                                vertical_geometry["requires_gemini_review"] = True
                                vertical_geometry["semantic_review_reasons"] = [
                                    "required_region_grounding_failed",
                                    "cached_composite_track_fallback",
                                ]
                                vertical_debug = primary_root / "grounding-debug.png"
                                vertical_debugs = [vertical_debug]
                                reused_cached_composite = True
                            except Exception as composite_error:
                                cached_composite_error = composite_error
                                reused_cached_composite = False
                        if not reused_cached_composite:
                            composite_suffix = (
                                ";cached_composite_failed:"
                                f"{type(cached_composite_error).__name__}:"
                                f"{cached_composite_error}"
                                if cached_composite_error is not None
                                else ""
                            )
                            vertical_geometry = {
                                "applied_strategy": vertical_fallback_strategy,
                                "fallback_reason": (
                                    f"tracking_or_grounding_failed:{type(error).__name__}:{error}"
                                    + composite_suffix
                                ),
                                "risk_codes": ["tracking_or_grounding_failed"],
                                "requires_gemini_review": True,
                                "semantic_review_reasons": ["fallback_applied"],
                            }
                vertical_segment_fingerprint = _segment_variant_fingerprint(
                    source_sha256=vertical_clip.sha256,
                    start_ms=v_start,
                    end_ms=v_end,
                    filter_graph=vertical_filter,
                    geometry=vertical_geometry,
                    track_fingerprint=vertical_track_fingerprint,
                )
                vertical_segment = (
                    output_dir
                    / "segments"
                    / render_variant
                    / "9x16"
                    / f"{index:02d}-{vertical_segment_fingerprint[:12]}.mp4"
                )
                if not _segment_is_valid(
                    vertical_segment,
                    expected_duration=(v_end - v_start) / 1000,
                    dimensions=(1080, 1920),
                ):
                    _render_source_segment(
                        source_path=Path(vertical_clip.path),
                        start_ms=v_start,
                        end_ms=v_end,
                        overlay_path=(vertical_overlay if brief.render_title_overlays else None),
                        base_filter=vertical_filter,
                        output_path=vertical_segment,
                        source_has_audio=vertical_source_has_audio,
                    )
                horizontal_entry = {
                    "feature_id": selected.feature_id,
                    "semantic_intent": (
                        brief_chapter.title
                        + (" — " + "; ".join(brief_chapter.detail_lines) if brief_chapter.detail_lines else "")
                    ),
                    "observed_visual_evidence": selected.observed_visual_evidence,
                    "selection_reason": selected.selection_reason,
                    "source_frame_id": horizontal_frame.frame_id,
                    "source_clip_id": horizontal_clip.clip_id,
                    "source_in_ms": h_start,
                    "source_out_ms": h_end,
                    "duration_ms": h_end - h_start,
                    "source_shot_id": h_shot,
                    "segment_render_fingerprint": horizontal_segment_fingerprint,
                    "track_geometry_fingerprint": horizontal_track_fingerprint,
                    "segment_path": str(horizontal_segment.resolve()),
                    "audio_origin": (
                        "source" if horizontal_source_has_audio else "synthetic_silence"
                    ),
                    "source_sample_aspect_ratio": (
                        horizontal_source_media.video.sample_aspect_ratio.model_dump(
                            mode="json"
                        )
                    ),
                    "source_display_sample_aspect_ratio": (
                        horizontal_source_media.video.display_sample_aspect_ratio.model_dump(
                            mode="json"
                        )
                    ),
                    "grounding_debug": str(horizontal_debug.resolve()) if horizontal_debug else None,
                    **horizontal_trim,
                    **horizontal_geometry,
                }
                vertical_entry = {
                    "feature_id": selected.feature_id,
                    "semantic_intent": (
                        brief_chapter.title
                        + (" — " + "; ".join(brief_chapter.detail_lines) if brief_chapter.detail_lines else "")
                    ),
                    "observed_visual_evidence": selected.observed_visual_evidence,
                    "selection_reason": selected.selection_reason,
                    "source_frame_id": vertical_frame.frame_id,
                    "source_clip_id": vertical_clip.clip_id,
                    "source_in_ms": v_start,
                    "source_out_ms": v_end,
                    "duration_ms": v_end - v_start,
                    "source_shot_id": v_shot,
                    "segment_render_fingerprint": vertical_segment_fingerprint,
                    "track_geometry_fingerprint": vertical_track_fingerprint,
                    "segment_path": str(vertical_segment.resolve()),
                    "audio_origin": (
                        "source" if vertical_source_has_audio else "synthetic_silence"
                    ),
                    "source_sample_aspect_ratio": (
                        vertical_source_media.video.sample_aspect_ratio.model_dump(
                            mode="json"
                        )
                    ),
                    "source_display_sample_aspect_ratio": (
                        vertical_source_media.video.display_sample_aspect_ratio.model_dump(
                            mode="json"
                        )
                    ),
                    "target_description": vertical_target_description,
                    "primary_target_override": vertical_primary_override is not None,
                    "vertical_regions": [
                        region.model_dump(mode="json") for region in vertical_regions
                    ],
                    "vertical_overflow_policy": brief_chapter.vertical_overflow_policy,
                    "vertical_edge_priority": brief_chapter.vertical_edge_priority,
                    "vertical_crop_mode": brief_chapter.vertical_crop_mode,
                    "grounding_debug": str(vertical_debug.resolve()) if vertical_debug else None,
                    "grounding_debugs": [
                        str(path.resolve()) for path in vertical_debugs if path.exists()
                    ],
                    **vertical_trim,
                    **vertical_geometry,
                }
            horizontal_segments.append(horizontal_segment)
            vertical_segments.append(vertical_segment)
            manifest["horizontal"]["chapters"].append(horizontal_entry)
            manifest["vertical"]["chapters"].append(vertical_entry)
        timings["geometry_and_segment_render_seconds"] = round(monotonic() - stage, 3)
    finally:
        try:
            client.close()
        finally:
            incremental_pricing = _write_incremental_pricing(
                output_dir=output_dir,
                prior_interaction_hashes=prior_interaction_hashes,
                prior_error_hashes=prior_error_hashes,
            )
    try:
        output_suffix = "" if brief.render_title_overlays else "-clean"
        horizontal_output = (
            output_dir / "renders" / f"feature-cut-16x9{output_suffix}.mp4"
        )
        vertical_output = (
            output_dir / "renders" / f"feature-cut-9x16{output_suffix}.mp4"
        )
        horizontal_output.parent.mkdir(parents=True, exist_ok=True)
        stage = monotonic()
        _concat_segments(horizontal_segments, horizontal_output)
        _concat_segments(vertical_segments, vertical_output)
        timings["concat_seconds"] = round(monotonic() - stage, 3)
        timings["total_seconds"] = round(monotonic() - started, 3)
        manifest["horizontal"]["output_path"] = str(horizontal_output.resolve())
        manifest["vertical"]["output_path"] = str(vertical_output.resolve())
        manifest["horizontal"]["media"] = _output_media_metadata(horizontal_output)
        manifest["vertical"]["media"] = _output_media_metadata(vertical_output)
        manifest["generated_at"] = utc_now()
        write_json(output_dir / "render-manifest.json", manifest)
        pricing = summarize_usage_and_list_price(output_dir)
        write_json(output_dir / "pricing.json", pricing)
        incremental_pricing = _write_incremental_pricing(
            output_dir=output_dir,
            prior_interaction_hashes=prior_interaction_hashes,
            prior_error_hashes=prior_error_hashes,
        )
        write_json(
            output_dir / "timing.json",
            {
                **timings,
                "file_api_reused": file_api_reused,
                "feature_plan_reuse_explicit": reuse_feature_plan and plan_reused,
                "feature_plan_reused": plan_reused,
                "generated_at": utc_now(),
            },
        )
        _render_review_html(output_dir, brief, plan, manifest)
        result = {
            "horizontal_output": str(horizontal_output.resolve()),
            "vertical_output": str(vertical_output.resolve()),
            "review_path": str((output_dir / "index.html").resolve()),
            "plan_path": str((plan_dir / "feature_edit_plan.json").resolve()),
            "manifest_path": str((output_dir / "render-manifest.json").resolve()),
            "timing": timings,
            "pricing": pricing,
            "incremental_pricing": incremental_pricing,
        }
        write_json(output_dir / "result.json", result)
        return result
    finally:
        _write_incremental_pricing(
            output_dir=output_dir,
            prior_interaction_hashes=prior_interaction_hashes,
            prior_error_hashes=prior_error_hashes,
        )
