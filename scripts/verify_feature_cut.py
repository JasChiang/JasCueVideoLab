#!/usr/bin/env python3
"""Cost-aware, observation-only Gemini QA for a rendered 9:16 feature cut.

One 720x1280 proxy is sent per distinct render/manifest/prompt/schema/model
contract. The tool never edits a cut and deliberately excludes timestamps from
both its prompt and response schema. Frame-accurate geometry remains a local
validation responsibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jascue_video_lab.billing import summarize_usage_files
from jascue_video_lab.gemini import (
    MODEL_ID,
    VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
    GeminiLabClient,
    _raw_dump,
)
from jascue_video_lab.feature_cut import _output_media_metadata
from jascue_video_lab.media import sha256_file
from jascue_video_lab.schema import gemini_response_schema
from jascue_video_lab.storage import append_error, read_json, utc_now, write_json


PROMPT_VERSION = "feature-cut-semantic-qa-v2-required-regions"
VALIDATOR_VERSION = "feature-cut-semantic-qa-validator-v3-fail-closed-risk-gate"
DEFAULT_PROMPT_RESOURCE = "feature_cut_qa_zh-TW.txt"
QA_GENERATION_CONFIG = {
    "temperature": 0,
    "thinking_level": "minimal",
    "max_output_tokens": 4096,
}
PROXY_CONTRACT = {
    "version": "feature-cut-qa-proxy-v1",
    "width": 720,
    "height": 1280,
    "video_codec": "libx264",
    "preset": "veryfast",
    "crf": 30,
    "audio_codec": "aac",
    "audio_bitrate": "64k",
}
ATTEMPT_RECORD_VERSION = "feature-cut-semantic-qa-attempt-v1"
_CANONICAL_ATTEMPT_FILES = (
    "request.json",
    "raw_interaction.json",
    "raw_output.json",
    "validated.json",
    "schema_validation.json",
    "pricing.json",
    "timing.json",
)

# A final-video review sees the rendered result, but it cannot invalidate
# producer-side facts such as incomplete tracking coverage, an unavailable
# required region, or an explicitly controlled clip. Even apparent motion
# smoothness is not sufficient: Gemini's video sampling may skip a short crop
# excursion. Keep this allowlist deliberately empty until a risk has a proven,
# versioned final-render verification procedure. Unknown future risk codes
# therefore fail closed automatically.
FINAL_RENDER_CLEARABLE_RISK_CODES: frozenset[str] = frozenset()


def _read_prompt(path: Path | None) -> str:
    if path is not None:
        return path.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    return (
        resources.files("jascue_video_lab.prompts")
        .joinpath(DEFAULT_PROMPT_RESOURCE)
        .read_text(encoding="utf-8")
    )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SegmentQaObservation(StrictModel):
    feature_id: str
    segment_id: str
    semantic_match: Literal["match", "partial", "mismatch", "uncertain"]
    target_visibility: Literal[
        "clearly_visible",
        "partially_visible",
        "not_visible",
        "uncertain",
    ]
    important_text_status: Literal[
        "not_applicable",
        "complete",
        "partial",
        "unreadable",
        "not_visible",
        "uncertain",
    ]
    issues: list[
        Literal[
            "target_scope_loss",
            "target_clipped",
            "text_clipped",
            "text_unreadable",
            "identity_mismatch",
            "semantic_mismatch",
            "unexpected_content",
            "abrupt_transition",
            "repetition",
            "other",
        ]
    ] = Field(max_length=6)
    evidence_note: str


class FeatureCutQaResult(StrictModel):
    render_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    aspect_ratio: Literal["9:16"]
    overall_status: Literal["pass", "review", "reject"]
    segments: list[SegmentQaObservation]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _attempt_raw_interaction_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "attempts").glob("*/raw_interaction.json"))


def _snapshot_existing_canonical_attempt(run_dir: Path) -> Path | None:
    """Preserve a pre-attempt-layout response before a refresh can replace it."""

    raw_path = run_dir / "raw_interaction.json"
    if not raw_path.is_file():
        return None
    raw_hash = sha256_file(raw_path)
    for existing in _attempt_raw_interaction_paths(run_dir):
        if sha256_file(existing) == raw_hash:
            return existing.parent

    snapshot_dir = run_dir / "attempts" / f"legacy-{raw_hash[:20]}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    for filename in _CANONICAL_ATTEMPT_FILES:
        source = run_dir / filename
        if source.is_file():
            shutil.copy2(source, snapshot_dir / filename)
    write_json(
        snapshot_dir / "attempt.json",
        {
            "record_version": ATTEMPT_RECORD_VERSION,
            "kind": "legacy_canonical_snapshot",
            "raw_interaction_sha256": raw_hash,
            "snapshotted_at": utc_now(),
        },
    )
    return snapshot_dir


def _next_attempt_dir(run_dir: Path) -> Path:
    """Allocate a new directory; a retry can never reuse an earlier attempt."""

    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    existing_numbers = [
        int(path.name.removeprefix("attempt-"))
        for path in attempts_dir.glob("attempt-[0-9][0-9][0-9][0-9][0-9][0-9]")
        if path.is_dir() and path.name.removeprefix("attempt-").isdigit()
    ]
    attempt_number = max(existing_numbers, default=0) + 1
    while True:
        attempt_dir = attempts_dir / f"attempt-{attempt_number:06d}"
        try:
            attempt_dir.mkdir(exist_ok=False)
        except FileExistsError:
            attempt_number += 1
            continue
        write_json(
            attempt_dir / "attempt.json",
            {
                "record_version": ATTEMPT_RECORD_VERSION,
                "kind": "interaction_attempt",
                "attempt_number": attempt_number,
                "created_at": utc_now(),
            },
        )
        return attempt_dir


def _write_attempt_and_aggregate_pricing(run_dir: Path, attempt_dir: Path) -> None:
    attempt_raw = attempt_dir / "raw_interaction.json"
    write_json(
        attempt_dir / "pricing.json",
        summarize_usage_files(
            [attempt_raw] if attempt_raw.is_file() else [],
            relative_to=attempt_dir,
        ),
    )
    write_json(
        run_dir / "pricing.json",
        summarize_usage_files(
            _attempt_raw_interaction_paths(run_dir),
            relative_to=run_dir,
        ),
    )


def compute_cache_key(
    *,
    render_hash: str,
    manifest_hash: str,
    prompt: str,
    schema: dict[str, Any],
    model: str,
    system_instruction: str = VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
    proxy_hash: str = "",
    proxy_contract: dict[str, Any] | None = None,
    generation_config: dict[str, Any] | None = None,
    proxy_geometry: str = "720x1280",
) -> str:
    """Fingerprint every input that can change the semantic QA result."""

    payload = {
        "render_hash": render_hash,
        "manifest_hash": manifest_hash,
        "prompt": prompt,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "schema": schema,
        "model": model,
        "system_instruction": system_instruction,
        "proxy_hash": proxy_hash,
        "proxy_contract": proxy_contract or PROXY_CONTRACT,
        "generation_config": generation_config or QA_GENERATION_CONFIG,
        "proxy_geometry": proxy_geometry,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        output: list[str] = []
        for child in value:
            output.extend(_flatten_text(child))
        return output
    if isinstance(value, dict):
        output = []
        for key in (
            "text",
            "description",
            "target_description",
            "expected",
            "value",
            "label",
        ):
            output.extend(_flatten_text(value.get(key)))
        return output
    return []


def _important_text_expectations(chapter: dict[str, Any]) -> list[str]:
    """Read generic text-safety fields without depending on a product taxonomy."""

    keys = (
        "important_text",
        "important_text_description",
        "required_text",
        "required_visible_text",
        "text_requirements",
    )
    values: list[str] = []
    for key in keys:
        values.extend(_flatten_text(chapter.get(key)))

    for container_key in ("reframe_regions", "vertical_regions", "target_regions"):
        regions = chapter.get(container_key)
        if not isinstance(regions, list):
            continue
        for region in regions:
            if not isinstance(region, dict):
                continue
            kind = str(
                region.get("semantic_kind")
                or region.get("region_kind")
                or region.get("kind")
                or ""
            ).lower()
            role = str(region.get("role") or "required").lower()
            if (
                kind in {"text", "text_region", "ui_text", "signage"}
                and role == "required"
            ):
                values.extend(_flatten_text(region))

    # Preserve order while avoiding prompt inflation from duplicate aliases.
    return list(dict.fromkeys(values))


def _deterministic_review_reasons(chapter: dict[str, Any]) -> list[str]:
    """Return producer facts that one sampled final-video pass cannot waive."""

    reasons: list[str] = []
    if chapter.get("requires_gemini_review") is True:
        # Despite the historical field name, this is an upstream warning, not
        # permission for this same model call to promote the cut to release-pass.
        reasons.append("manifest_requires_gemini_review")
    if chapter.get("coverage_passed") is False:
        reasons.append("required_region_tracking_coverage_failed")
    if chapter.get("full_containment_feasible") is False:
        reasons.append("required_region_full_containment_infeasible")
    if chapter.get("controlled_clip_applied") is True:
        reasons.append("controlled_required_region_clip")
    fallback_reason = chapter.get("fallback_reason")
    if isinstance(fallback_reason, str) and fallback_reason:
        reasons.append(f"fallback:{fallback_reason}")

    risk_codes = [
        str(item).strip()
        for item in chapter.get("risk_codes") or []
        if str(item).strip()
    ]
    reasons.extend(
        f"risk:{risk_code}"
        for risk_code in risk_codes
        if risk_code not in FINAL_RENDER_CLEARABLE_RISK_CODES
    )
    return list(dict.fromkeys(reasons))


def build_segment_contract(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    vertical = manifest.get("vertical")
    if not isinstance(vertical, dict):
        raise ValueError("render manifest has no vertical section")
    chapters = vertical.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("render manifest has no vertical chapters")

    contract: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ordinal, chapter in enumerate(chapters, start=1):
        if not isinstance(chapter, dict):
            raise ValueError(f"vertical chapter {ordinal} is not an object")
        feature_id = chapter.get("feature_id")
        if not isinstance(feature_id, str) or not feature_id:
            raise ValueError(f"vertical chapter {ordinal} has no feature_id")
        segment_path = chapter.get("segment_path")
        explicit_segment_id = chapter.get("segment_id")
        if isinstance(explicit_segment_id, str) and explicit_segment_id:
            segment_id = explicit_segment_id
        elif isinstance(segment_path, str) and segment_path:
            segment_id = Path(segment_path).stem
        else:
            fingerprint = chapter.get("segment_render_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError(f"vertical chapter {ordinal} has no segment identity")
            segment_id = fingerprint[:16]
        identity = (feature_id, segment_id)
        if identity in seen:
            raise ValueError(f"duplicate vertical segment identity: {identity}")
        seen.add(identity)

        expected_semantics = (
            chapter.get("expected_semantics")
            or chapter.get("semantic_intent")
            or chapter.get("target_description")
            or "No explicit semantic expectation was recorded; observe conservatively."
        )
        target_description = chapter.get("target_description")
        duration_ms = chapter.get("duration_ms")
        if duration_ms is None and all(
            key in chapter for key in ("source_in_ms", "source_out_ms")
        ):
            duration_ms = int(chapter["source_out_ms"]) - int(chapter["source_in_ms"])
        if not isinstance(duration_ms, int) or duration_ms <= 0:
            raise ValueError(f"vertical chapter {ordinal} has no valid duration_ms")
        required_regions = [
            {
                "region_id": str(region.get("region_id") or ""),
                "kind": str(region.get("kind") or "other"),
                "target_description": str(region.get("target_description") or ""),
            }
            for region in chapter.get("vertical_regions") or []
            if isinstance(region, dict)
            and str(region.get("role") or "required") == "required"
        ]
        risk_codes = [str(item) for item in chapter.get("risk_codes") or []]
        contract.append(
            {
                "manifest_order": ordinal,
                "feature_id": feature_id,
                "segment_id": segment_id,
                "expected_semantics": str(expected_semantics),
                "target_description": (
                    str(target_description) if target_description else None
                ),
                "important_text": _important_text_expectations(chapter),
                "required_regions": required_regions,
                "requires_gemini_review": bool(
                    chapter.get("requires_gemini_review")
                ),
                "risk_codes": risk_codes,
                "deterministic_review_reasons": _deterministic_review_reasons(
                    chapter
                ),
                "duration_ms": duration_ms,
            }
        )
    return contract


def validate_render_contract(
    manifest: dict[str, Any],
    *,
    render_hash: str,
    actual_media: dict[str, Any],
) -> None:
    """Fail closed when the supplied file is not the declared 9:16 timeline."""

    vertical = manifest.get("vertical")
    if not isinstance(vertical, dict):
        raise ValueError("render manifest has no vertical section")
    declared = vertical.get("media")
    if not isinstance(declared, dict):
        raise ValueError("render manifest has no vertical media fingerprint")
    if declared.get("sha256") != render_hash:
        raise ValueError("render hash does not match manifest vertical media")
    width = int(actual_media.get("width") or 0)
    height = int(actual_media.get("height") or 0)
    if width <= 0 or height <= 0 or width * 16 != height * 9:
        raise ValueError(f"semantic QA requires a true 9:16 render, got {width}x{height}")
    if int(declared.get("width") or 0) != width or int(declared.get("height") or 0) != height:
        raise ValueError("render dimensions do not match manifest vertical media")
    actual_duration = float(actual_media.get("duration_seconds") or 0.0)
    declared_duration = float(declared.get("duration_seconds") or 0.0)
    if actual_duration <= 0 or abs(actual_duration - declared_duration) > 0.25:
        raise ValueError("render duration does not match manifest vertical media")
    chapters = vertical.get("chapters")
    if not isinstance(chapters, list) or not chapters:
        raise ValueError("render manifest has no vertical chapters")
    expected_ms = 0
    for ordinal, chapter in enumerate(chapters, start=1):
        if not isinstance(chapter, dict):
            raise ValueError(f"vertical chapter {ordinal} is not an object")
        duration_ms = chapter.get("duration_ms")
        if duration_ms is None and all(
            key in chapter for key in ("source_in_ms", "source_out_ms")
        ):
            duration_ms = int(chapter["source_out_ms"]) - int(chapter["source_in_ms"])
        if not isinstance(duration_ms, int) or duration_ms <= 0:
            raise ValueError(f"vertical chapter {ordinal} has no valid duration")
        expected_ms += duration_ms
    if abs(actual_duration - expected_ms / 1000) > 0.5:
        raise ValueError("render duration does not match the ordered chapter timeline")


def validate_result_contract(
    result: FeatureCutQaResult,
    *,
    render_hash: str,
    manifest_hash: str,
    segment_contract: list[dict[str, Any]],
) -> Literal["pass", "review", "reject"]:
    if result.render_hash != render_hash or result.manifest_hash != manifest_hash:
        raise ValueError("Gemini changed immutable render or manifest hashes")
    expected = [(item["feature_id"], item["segment_id"]) for item in segment_contract]
    actual = [(item.feature_id, item.segment_id) for item in result.segments]
    if actual != expected:
        raise ValueError(f"Gemini changed or reordered segment identities: {actual} != {expected}")
    important_text_expected = {
        (item["feature_id"], item["segment_id"]): bool(item["important_text"])
        for item in segment_contract
    }
    for observation in result.segments:
        has_expectation = important_text_expected[
            (observation.feature_id, observation.segment_id)
        ]
        if not has_expectation and observation.important_text_status != "not_applicable":
            raise ValueError(
                f"{observation.feature_id}/{observation.segment_id} invented an important-text check"
            )
        if has_expectation and observation.important_text_status == "not_applicable":
            raise ValueError(
                f"{observation.feature_id}/{observation.segment_id} skipped required important text"
            )
    critical = any(
        observation.semantic_match == "mismatch"
        or observation.target_visibility == "not_visible"
        or observation.important_text_status in {"not_visible", "unreadable"}
        or bool(
            {"identity_mismatch", "semantic_mismatch"}
            & set(observation.issues)
        )
        for observation in result.segments
    )
    clean = all(
        observation.semantic_match == "match"
        and observation.target_visibility == "clearly_visible"
        and observation.important_text_status in {"not_applicable", "complete"}
        and not observation.issues
        for observation in result.segments
    )
    deterministic_review_required = any(
        bool(item.get("deterministic_review_reasons")) for item in segment_contract
    )
    return (
        "reject"
        if critical
        else "pass"
        if clean and not deterministic_review_required
        else "review"
    )


def _render_path(manifest: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    value = (manifest.get("vertical") or {}).get("output_path")
    if not isinstance(value, str) or not value:
        raise ValueError("manifest vertical.output_path is missing; pass --render")
    return Path(value).expanduser().resolve()


def _create_proxy(source: Path, destination: Path) -> float:
    if destination.exists() and destination.stat().st_size > 0:
        return 0.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            "scale=720:1280:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "30",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )
    return round(time.monotonic() - started, 3)


def _prompt_text(
    template: str,
    *,
    render_hash: str,
    manifest_hash: str,
    segments: list[dict[str, Any]],
) -> str:
    return (
        template.rstrip()
        + "\n\n## 不可變成片識別\n"
        + f"render_hash: {render_hash}\n"
        + f"manifest_hash: {manifest_hash}\n"
        + "aspect_ratio: 9:16\n\n"
        + "## 依成片順序排列的剪輯單元契約\n"
        + json.dumps(segments, ensure_ascii=False, indent=2)
    )


def _cache_complete(run_dir: Path) -> bool:
    return all(
        (run_dir / name).exists()
        for name in (
            "cache-key.json",
            "request.json",
            "raw_interaction.json",
            "raw_output.json",
            "validated.json",
            "schema_validation.json",
            "pricing.json",
            "timing.json",
        )
    )


def _expected_cache_record(
    *,
    cache_key: str,
    render_hash: str,
    manifest_hash: str,
    proxy_hash: str,
) -> dict[str, Any]:
    return {
        "cache_key": cache_key,
        "model": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "render_hash": render_hash,
        "manifest_hash": manifest_hash,
        "proxy_hash": proxy_hash,
        "proxy_contract": PROXY_CONTRACT,
        "system_instruction_sha256": hashlib.sha256(
            VISUAL_EVIDENCE_SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
    }


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"cached {label} does not match the current contract")


def _validate_cached_run(
    run_dir: Path,
    *,
    expected_cache_record: dict[str, Any],
    prompt: str,
    schema: dict[str, Any],
    proxy_hash: str,
    render_hash: str,
    manifest_hash: str,
    segment_contract: list[dict[str, Any]],
) -> FeatureCutQaResult:
    """Reload and validate a cache entry before treating it as a hit.

    Existence alone is not evidence: interrupted writes, hand-edited fixtures,
    and validator upgrades can leave a superficially complete directory. This
    function verifies both the causal request and the locally reconciled result.
    """

    state_path = run_dir / "cache-state.json"
    if state_path.exists():
        state = read_json(state_path)
        if not isinstance(state, dict) or state.get("status") != "valid":
            raise ValueError("cached run is explicitly marked refresh-required")

    cache_record = read_json(run_dir / "cache-key.json")
    _require_equal("cache-key record", cache_record, expected_cache_record)

    request = read_json(run_dir / "request.json")
    if not isinstance(request, dict):
        raise ValueError("cached request is not an object")
    for key, expected in (
        ("cache_key", expected_cache_record["cache_key"]),
        ("proxy_hash", proxy_hash),
        ("model", MODEL_ID),
        ("system_instruction", VISUAL_EVIDENCE_SYSTEM_INSTRUCTION),
        ("store", False),
        ("generation_config", QA_GENERATION_CONFIG),
        ("segment_contract", segment_contract),
    ):
        _require_equal(f"request.{key}", request.get(key), expected)
    response_format = request.get("response_format")
    if not isinstance(response_format, dict):
        raise ValueError("cached request has no response_format object")
    _require_equal("request response schema", response_format.get("schema"), schema)
    if (
        response_format.get("type") != "text"
        or response_format.get("mime_type") != "application/json"
    ):
        raise ValueError("cached request response format changed")
    inputs = request.get("input")
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("cached request input must contain text and one video")
    _require_equal("request prompt", inputs[0], {"type": "text", "text": prompt})
    video_input = inputs[1]
    if (
        not isinstance(video_input, dict)
        or video_input.get("type") != "video"
        or not isinstance(video_input.get("uri"), str)
        or not video_input["uri"]
        or not isinstance(video_input.get("mime_type"), str)
        or not video_input["mime_type"]
    ):
        raise ValueError("cached request video input is incomplete")

    raw_interaction = read_json(run_dir / "raw_interaction.json")
    if not isinstance(raw_interaction, dict) or not raw_interaction:
        raise ValueError("cached raw interaction is empty or invalid")
    raw_output = read_json(run_dir / "raw_output.json")
    if not isinstance(raw_output, dict) or not isinstance(
        raw_output.get("output_text"), str
    ):
        raise ValueError("cached raw output has no output_text")
    raw_result = FeatureCutQaResult.model_validate_json(raw_output["output_text"])
    raw_locally_derived_status = validate_result_contract(
        raw_result,
        render_hash=render_hash,
        manifest_hash=manifest_hash,
        segment_contract=segment_contract,
    )

    validated_payload = read_json(run_dir / "validated.json")
    validated = FeatureCutQaResult.model_validate(validated_payload)
    validated_locally_derived_status = validate_result_contract(
        validated,
        render_hash=render_hash,
        manifest_hash=manifest_hash,
        segment_contract=segment_contract,
    )
    if validated.overall_status != validated_locally_derived_status:
        raise ValueError("cached validated status was not locally reconciled")
    if raw_locally_derived_status != validated.overall_status:
        raise ValueError("cached raw output and validated status disagree")
    if (
        raw_result.model_copy(update={"overall_status": validated.overall_status})
        != validated
    ):
        raise ValueError("cached validated payload changed model observations")

    validation_record = read_json(run_dir / "schema_validation.json")
    if (
        not isinstance(validation_record, dict)
        or validation_record.get("ok") is not True
    ):
        raise ValueError("cached schema validation did not succeed")
    for key, expected in (
        ("validator_version", VALIDATOR_VERSION),
        ("model_reported_overall_status", raw_result.overall_status),
        ("locally_derived_overall_status", validated.overall_status),
        (
            "status_reconciled",
            raw_result.overall_status != validated.overall_status,
        ),
    ):
        _require_equal(
            f"schema_validation.{key}", validation_record.get(key), expected
        )

    pricing = read_json(run_dir / "pricing.json")
    timing = read_json(run_dir / "timing.json")
    if not isinstance(pricing, dict):
        raise ValueError("cached pricing record is not an object")
    if not isinstance(timing, dict) or timing.get("status") != "completed":
        raise ValueError("cached timing record is not completed")
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one cached Gemini semantic QA pass over a rendered 9:16 cut."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument(
        "--prompt",
        type=Path,
        help="Optional prompt override; defaults to the installed package resource.",
    )
    args = parser.parse_args()

    invocation_started = time.monotonic()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(manifest_path)
    render_path = _render_path(manifest, args.render)
    if not render_path.is_file():
        raise FileNotFoundError(render_path)

    prompt_template = _read_prompt(args.prompt)
    render_hash = sha256_file(render_path)
    actual_media = _output_media_metadata(render_path)
    validate_render_contract(
        manifest,
        render_hash=render_hash,
        actual_media=actual_media,
    )
    manifest_hash = sha256_file(manifest_path)
    segment_contract = build_segment_contract(manifest)
    schema = gemini_response_schema(FeatureCutQaResult)
    prompt = _prompt_text(
        prompt_template,
        render_hash=render_hash,
        manifest_hash=manifest_hash,
        segments=segment_contract,
    )
    proxy_contract_hash = hashlib.sha256(
        _canonical_json(PROXY_CONTRACT).encode("utf-8")
    ).hexdigest()
    proxy_path = (
        output_dir
        / "proxies"
        / f"{render_hash[:20]}-{proxy_contract_hash[:12]}.mp4"
    )
    proxy_seconds = _create_proxy(render_path, proxy_path)
    proxy_hash = sha256_file(proxy_path)
    cache_key = compute_cache_key(
        render_hash=render_hash,
        manifest_hash=manifest_hash,
        prompt=prompt,
        schema=schema,
        model=MODEL_ID,
        system_instruction=VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
        proxy_hash=proxy_hash,
        proxy_contract=PROXY_CONTRACT,
        generation_config=QA_GENERATION_CONFIG,
    )
    run_dir = output_dir / "runs" / cache_key
    run_dir.mkdir(parents=True, exist_ok=True)
    expected_cache_record = _expected_cache_record(
        cache_key=cache_key,
        render_hash=render_hash,
        manifest_hash=manifest_hash,
        proxy_hash=proxy_hash,
    )
    write_json(
        output_dir / "latest.json",
        {
            "cache_key": cache_key,
            "run_dir": str(run_dir),
            "checked_at": utc_now(),
        },
    )
    cache_miss_reason = "cache entry is incomplete"
    if _cache_complete(run_dir):
        try:
            _validate_cached_run(
                run_dir,
                expected_cache_record=expected_cache_record,
                prompt=prompt,
                schema=schema,
                proxy_hash=proxy_hash,
                render_hash=render_hash,
                manifest_hash=manifest_hash,
                segment_contract=segment_contract,
            )
        except Exception as error:
            cache_miss_reason = f"{type(error).__name__}: {error}"
        else:
            write_json(
                run_dir / "cache-state.json",
                {
                    "status": "valid",
                    "validated_at": utc_now(),
                    "validator_version": VALIDATOR_VERSION,
                },
            )
            write_json(
                output_dir / "latest-invocation.json",
                {
                    "status": "completed",
                    "cache_hit": True,
                    "cache_key": cache_key,
                    "elapsed_seconds": round(time.monotonic() - invocation_started, 3),
                    "validated_path": str(run_dir / "validated.json"),
                    "checked_at": utc_now(),
                },
            )
            print(run_dir / "validated.json")
            return

    # The marker prevents an interrupted refresh from making stale files look
    # complete on the next invocation. A successful API + local validation pass
    # is the only operation that returns this state to "valid".
    write_json(
        run_dir / "cache-state.json",
        {
            "status": "refresh_required",
            "reason": cache_miss_reason,
            "validator_version": VALIDATOR_VERSION,
            "checked_at": utc_now(),
        },
    )
    write_json(run_dir / "cache-key.json", expected_cache_record)

    write_json(
        run_dir / "proxy.json",
        {
            "source_path": str(render_path),
            "source_hash": render_hash,
            "proxy_path": str(proxy_path),
            "proxy_hash": proxy_hash,
            "width": 720,
            "height": 1280,
        },
    )

    # A refresh must never destroy the evidence or usage from an earlier
    # completed/failed interaction. Older runs used canonical root files only;
    # snapshot those once, then allocate a fresh immutable attempt directory.
    _snapshot_existing_canonical_attempt(run_dir)
    attempt_dir = _next_attempt_dir(run_dir)

    client: GeminiLabClient | None = None
    upload_seconds: float | None = None
    upload_reused: bool | None = None
    interaction_seconds: float | None = None
    try:
        client = GeminiLabClient(temperature=0.0)
        upload_started = time.monotonic()
        uploaded, upload_reused = client.ensure_video_upload(
            proxy_path,
            output_dir / "file-api" / proxy_hash,
        )
        upload_seconds = round(time.monotonic() - upload_started, 3)
        request = {
            "model": MODEL_ID,
            "system_instruction": VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
            "store": False,
            "input": [
                {"type": "text", "text": prompt},
                {
                    "type": "video",
                    "uri": uploaded.uri,
                    "mime_type": uploaded.mime_type,
                },
            ],
            "generation_config": QA_GENERATION_CONFIG,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            },
        }
        request_record = {
            **request,
            "cache_key": cache_key,
            "proxy_hash": proxy_hash,
            "segment_contract": segment_contract,
        }
        write_json(attempt_dir / "request.json", request_record)
        interaction_started = time.monotonic()
        interaction = client.client.interactions.create(**request)
        interaction_seconds = round(time.monotonic() - interaction_started, 3)
        raw_interaction = _raw_dump(interaction)
        raw_output = {"output_text": interaction.output_text}
        write_json(attempt_dir / "raw_interaction.json", raw_interaction)
        write_json(attempt_dir / "raw_output.json", raw_output)
        try:
            result = FeatureCutQaResult.model_validate_json(interaction.output_text)
            model_reported_overall_status = result.overall_status
            locally_derived_overall_status = validate_result_contract(
                result,
                render_hash=render_hash,
                manifest_hash=manifest_hash,
                segment_contract=segment_contract,
            )
            result = result.model_copy(
                update={"overall_status": locally_derived_overall_status}
            )
        except Exception as error:
            write_json(
                attempt_dir / "schema_validation.json",
                {
                    "ok": False,
                    "errors": [{"type": type(error).__name__, "message": str(error)}],
                },
            )
            raise
        validation_record = {
            "ok": True,
            "errors": [],
            "model_reported_overall_status": model_reported_overall_status,
            "locally_derived_overall_status": locally_derived_overall_status,
            "status_reconciled": (
                model_reported_overall_status != locally_derived_overall_status
            ),
            "validator_version": VALIDATOR_VERSION,
        }
        timing_record = {
            "status": "completed",
            "cache_hit": False,
            "proxy_seconds": proxy_seconds,
            "upload_seconds": upload_seconds,
            "upload_reused": upload_reused,
            "interaction_seconds": interaction_seconds,
            "total_seconds": round(time.monotonic() - invocation_started, 3),
            "completed_at": utc_now(),
            "attempt_dir": str(attempt_dir),
        }
        write_json(attempt_dir / "validated.json", result)
        write_json(attempt_dir / "schema_validation.json", validation_record)
        write_json(attempt_dir / "timing.json", timing_record)
        _write_attempt_and_aggregate_pricing(run_dir, attempt_dir)

        # Canonical files point at the latest successful attempt for cache
        # validation. Every billed attempt remains independently preserved
        # beneath attempts/, including schema-invalid and interrupted retries.
        write_json(run_dir / "request.json", request_record)
        write_json(run_dir / "raw_interaction.json", raw_interaction)
        write_json(run_dir / "raw_output.json", raw_output)
        write_json(run_dir / "validated.json", result)
        write_json(run_dir / "schema_validation.json", validation_record)
        write_json(run_dir / "timing.json", timing_record)
        write_json(
            run_dir / "cache-state.json",
            {
                "status": "valid",
                "validated_at": utc_now(),
                "validator_version": VALIDATOR_VERSION,
            },
        )
    except BaseException as error:
        append_error(run_dir, "feature_cut_semantic_qa", error)
        append_error(attempt_dir, "feature_cut_semantic_qa", error)
        _write_attempt_and_aggregate_pricing(run_dir, attempt_dir)
        timing_record = {
            "status": "failed",
            "cache_hit": False,
            "proxy_seconds": proxy_seconds,
            "upload_seconds": upload_seconds,
            "upload_reused": upload_reused,
            "interaction_seconds": interaction_seconds,
            "total_seconds": round(time.monotonic() - invocation_started, 3),
            "completed_at": utc_now(),
            "attempt_dir": str(attempt_dir),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        write_json(attempt_dir / "timing.json", timing_record)
        write_json(run_dir / "timing.json", timing_record)
        write_json(
            output_dir / "latest-invocation.json",
            {
                "status": "failed",
                "cache_hit": False,
                "cache_key": cache_key,
                "elapsed_seconds": round(time.monotonic() - invocation_started, 3),
                "checked_at": utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
    finally:
        if client is not None:
            client.close()

    write_json(
        output_dir / "latest-invocation.json",
        {
            "status": "completed",
            "cache_hit": False,
            "cache_key": cache_key,
            "elapsed_seconds": round(time.monotonic() - invocation_started, 3),
            "validated_path": str(run_dir / "validated.json"),
            "checked_at": utc_now(),
        },
    )
    print(run_dir / "validated.json")


if __name__ == "__main__":
    main()
