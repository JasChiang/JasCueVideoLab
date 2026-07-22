from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scripts.verify_feature_cut import (
    FeatureCutQaResult,
    FINAL_RENDER_CLEARABLE_RISK_CODES,
    MODEL_ID,
    QA_GENERATION_CONFIG,
    SegmentQaObservation,
    VALIDATOR_VERSION,
    VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
    _attempt_raw_interaction_paths,
    _expected_cache_record,
    _next_attempt_dir,
    _read_prompt,
    _snapshot_existing_canonical_attempt,
    _validate_cached_run,
    _write_attempt_and_aggregate_pricing,
    build_segment_contract,
    compute_cache_key,
    validate_result_contract,
    validate_render_contract,
)
from jascue_video_lab.storage import write_json


def _schema() -> dict[str, object]:
    return FeatureCutQaResult.model_json_schema()


def test_default_prompt_is_available_as_installed_package_resource() -> None:
    prompt = _read_prompt(None)

    assert "只描述成片中直接可見或可聽的證據" in prompt
    assert "不得輸出、推算或引用任何時間戳" in prompt


def test_cache_key_is_deterministic_and_covers_every_contract_input() -> None:
    base = {
        "render_hash": "a" * 64,
        "manifest_hash": "b" * 64,
        "prompt": "observe only",
        "schema": _schema(),
        "model": "gemini-3.5-flash",
        "system_instruction": "observation only",
        "proxy_hash": "e" * 64,
    }
    first = compute_cache_key(**base)
    assert compute_cache_key(**base) == first

    for key, replacement in (
        ("render_hash", "c" * 64),
        ("manifest_hash", "d" * 64),
        ("prompt", "changed prompt"),
        ("schema", {"type": "object"}),
        ("model", "different-model"),
        ("system_instruction", "changed system instruction"),
        ("proxy_hash", "f" * 64),
    ):
        changed = dict(base)
        changed[key] = replacement
        assert compute_cache_key(**changed) != first

    assert compute_cache_key(
        **base,
        generation_config={"temperature": 0, "thinking_level": "low"},
    ) != first
    assert compute_cache_key(
        **base,
        proxy_contract={"version": "different-proxy"},
    ) != first


def test_qa_attempts_are_immutable_and_pricing_aggregates_retries(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    first = _next_attempt_dir(run_dir)
    second = _next_attempt_dir(run_dir)
    assert first.name == "attempt-000001"
    assert second.name == "attempt-000002"

    write_json(
        first / "raw_interaction.json",
        {
            "id": "first-billed-attempt",
            "usage": {"total_input_tokens": 100, "total_output_tokens": 10},
        },
    )
    write_json(
        second / "raw_interaction.json",
        {
            "id": "second-billed-attempt",
            "usage": {"total_input_tokens": 200, "total_output_tokens": 20},
        },
    )
    _write_attempt_and_aggregate_pricing(run_dir, first)
    _write_attempt_and_aggregate_pricing(run_dir, second)

    pricing = json.loads((run_dir / "pricing.json").read_text(encoding="utf-8"))
    assert pricing["request_count"] == 2
    assert pricing["total_input_tokens"] == 300
    assert pricing["total_output_tokens"] == 30
    assert _attempt_raw_interaction_paths(run_dir) == [
        first / "raw_interaction.json",
        second / "raw_interaction.json",
    ]


def test_legacy_canonical_response_is_snapshotted_only_once(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "raw_interaction.json",
        {
            "id": "legacy-billed-attempt",
            "usage": {"total_input_tokens": 50, "total_output_tokens": 5},
        },
    )
    write_json(run_dir / "request.json", {"model": MODEL_ID})
    write_json(run_dir / "timing.json", {"status": "failed"})

    snapshot = _snapshot_existing_canonical_attempt(run_dir)
    assert snapshot is not None
    assert (snapshot / "raw_interaction.json").is_file()
    assert (snapshot / "request.json").is_file()
    assert _snapshot_existing_canonical_attempt(run_dir) == snapshot
    assert len(_attempt_raw_interaction_paths(run_dir)) == 1


def test_build_segment_contract_uses_explicit_or_stable_path_identity() -> None:
    manifest = {
        "vertical": {
            "chapters": [
                {
                    "feature_id": "feature-a",
                    "segment_id": "segment-explicit",
                    "segment_path": "/tmp/ignored.mp4",
                    "target_description": "the selected moving subject",
                    "important_text": ["visible heading"],
                    "duration_ms": 2000,
                },
                {
                    "feature_id": "feature-b",
                    "segment_path": "/tmp/01-derived-id.mp4",
                    "target_description": "the foreground object",
                    "vertical_regions": [
                        {
                            "kind": "text_region",
                            "role": "required",
                            "target_description": "complete sign text",
                        }
                    ],
                    "duration_ms": 2500,
                },
                {
                    "feature_id": "feature-c",
                    "segment_path": "/tmp/02-preferred-text.mp4",
                    "semantic_intent": "preserve the main action",
                    "vertical_regions": [
                        {
                            "kind": "text_region",
                            "role": "preferred",
                            "target_description": "decorative background lettering",
                        }
                    ],
                    "duration_ms": 3000,
                },
            ]
        }
    }
    contract = build_segment_contract(manifest)
    assert [item["segment_id"] for item in contract] == [
        "segment-explicit",
        "01-derived-id",
        "02-preferred-text",
    ]
    assert contract[0]["important_text"] == ["visible heading"]
    assert contract[1]["important_text"] == ["complete sign text"]
    assert contract[2]["important_text"] == []
    assert contract[2]["expected_semantics"] == "preserve the main action"
    assert [item["duration_ms"] for item in contract] == [2000, 2500, 3000]
    assert contract[1]["required_regions"] == [
        {
            "region_id": "",
            "kind": "text_region",
            "target_description": "complete sign text",
        }
    ]


def test_build_segment_contract_fail_closes_review_flag_and_all_risk_codes() -> None:
    manifest = {
        "vertical": {
            "chapters": [
                {
                    "feature_id": "feature-risk",
                    "segment_id": "segment-risk",
                    "duration_ms": 1000,
                    "requires_gemini_review": True,
                    "risk_codes": [
                        "crop_motion_fast",
                        "source_boundary_contact",
                        "future_unrecognized_risk",
                    ],
                }
            ]
        }
    }

    assert not FINAL_RENDER_CLEARABLE_RISK_CODES
    contract = build_segment_contract(manifest)
    assert contract[0]["requires_gemini_review"] is True
    assert contract[0]["risk_codes"] == [
        "crop_motion_fast",
        "source_boundary_contact",
        "future_unrecognized_risk",
    ]
    assert contract[0]["deterministic_review_reasons"] == [
        "manifest_requires_gemini_review",
        "risk:crop_motion_fast",
        "risk:source_boundary_contact",
        "risk:future_unrecognized_risk",
    ]
    model_pass = FeatureCutQaResult(
        render_hash="a" * 64,
        manifest_hash="b" * 64,
        aspect_ratio="9:16",
        overall_status="pass",
        segments=[
            SegmentQaObservation(
                feature_id="feature-risk",
                segment_id="segment-risk",
                semantic_match="match",
                target_visibility="clearly_visible",
                important_text_status="not_applicable",
                issues=[],
                evidence_note="The sampled final video appears clean.",
            )
        ],
    )
    assert (
        validate_result_contract(
            model_pass,
            render_hash="a" * 64,
            manifest_hash="b" * 64,
            segment_contract=contract,
        )
        == "review"
    )


def test_render_contract_rejects_wrong_file_or_aspect() -> None:
    manifest = {
        "vertical": {
            "media": {
                "sha256": "a" * 64,
                "width": 1080,
                "height": 1920,
                "duration_seconds": 4.0,
            },
            "chapters": [
                {"feature_id": "one", "duration_ms": 2000},
                {"feature_id": "two", "duration_ms": 2000},
            ],
        }
    }
    media = {"width": 1080, "height": 1920, "duration_seconds": 4.0}
    validate_render_contract(manifest, render_hash="a" * 64, actual_media=media)

    with pytest.raises(ValueError, match="hash"):
        validate_render_contract(manifest, render_hash="b" * 64, actual_media=media)
    with pytest.raises(ValueError, match="true 9:16"):
        validate_render_contract(
            manifest,
            render_hash="a" * 64,
            actual_media={"width": 1920, "height": 1080, "duration_seconds": 4.0},
        )


def test_schema_forbids_unknown_status_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SegmentQaObservation(
            feature_id="feature-a",
            segment_id="segment-a",
            semantic_match="probably",  # type: ignore[arg-type]
            target_visibility="clearly_visible",
            important_text_status="not_applicable",
            issues=[],
            evidence_note="direct observation",
        )
    with pytest.raises(ValidationError):
        SegmentQaObservation.model_validate(
            {
                "feature_id": "feature-a",
                "segment_id": "segment-a",
                "semantic_match": "match",
                "target_visibility": "clearly_visible",
                "important_text_status": "not_applicable",
                "issues": [],
                "evidence_note": "direct observation",
                "timestamp": "00:01",
            }
        )


def test_local_validation_rejects_reordered_ids_and_skipped_text_check() -> None:
    contract = [
        {
            "feature_id": "feature-a",
            "segment_id": "segment-a",
            "important_text": ["required heading"],
        },
        {
            "feature_id": "feature-b",
            "segment_id": "segment-b",
            "important_text": [],
        },
    ]
    segment_a = SegmentQaObservation(
        feature_id="feature-a",
        segment_id="segment-a",
        semantic_match="match",
        target_visibility="clearly_visible",
        important_text_status="complete",
        issues=[],
        evidence_note="The requested evidence is visible.",
    )
    segment_b = SegmentQaObservation(
        feature_id="feature-b",
        segment_id="segment-b",
        semantic_match="partial",
        target_visibility="partially_visible",
        important_text_status="not_applicable",
        issues=["target_clipped"],
        evidence_note="Only part of the target remains visible.",
    )
    valid = FeatureCutQaResult(
        render_hash="a" * 64,
        manifest_hash="b" * 64,
        aspect_ratio="9:16",
        overall_status="review",
        segments=[segment_a, segment_b],
    )
    assert validate_result_contract(
        valid,
        render_hash="a" * 64,
        manifest_hash="b" * 64,
        segment_contract=contract,
    ) == "review"

    reordered = valid.model_copy(update={"segments": [segment_b, segment_a]})
    with pytest.raises(ValueError, match="reordered"):
        validate_result_contract(
            reordered,
            render_hash="a" * 64,
            manifest_hash="b" * 64,
            segment_contract=contract,
        )

    skipped = valid.model_copy(
        update={
            "segments": [
                segment_a.model_copy(update={"important_text_status": "not_applicable"}),
                segment_b,
            ]
        }
    )
    with pytest.raises(ValueError, match="skipped required important text"):
        validate_result_contract(
            skipped,
            render_hash="a" * 64,
            manifest_hash="b" * 64,
            segment_contract=contract,
        )

    inconsistent = valid.model_copy(update={"overall_status": "pass"})
    assert validate_result_contract(
        inconsistent,
        render_hash="a" * 64,
        manifest_hash="b" * 64,
        segment_contract=contract,
    ) == "review"


def test_local_geometry_gate_prevents_semantic_pass_from_becoming_release_pass() -> None:
    contract = [
        {
            "feature_id": "feature-a",
            "segment_id": "segment-a",
            "important_text": [],
            "deterministic_review_reasons": ["required_region_full_containment_infeasible"],
        }
    ]
    result = FeatureCutQaResult(
        render_hash="a" * 64,
        manifest_hash="b" * 64,
        aspect_ratio="9:16",
        overall_status="pass",
        segments=[
            SegmentQaObservation(
                feature_id="feature-a",
                segment_id="segment-a",
                semantic_match="match",
                target_visibility="clearly_visible",
                important_text_status="not_applicable",
                issues=[],
                evidence_note="The target remains recognizable.",
            )
        ],
    )

    assert validate_result_contract(
        result,
        render_hash="a" * 64,
        manifest_hash="b" * 64,
        segment_contract=contract,
    ) == "review"


def _write_valid_cache_fixture(run_dir: Path) -> dict[str, Any]:
    render_hash = "a" * 64
    manifest_hash = "b" * 64
    proxy_hash = "c" * 64
    cache_key = "d" * 64
    prompt = "Observe the final render only."
    schema = FeatureCutQaResult.model_json_schema()
    segment_contract = [
        {
            "feature_id": "feature-a",
            "segment_id": "segment-a",
            "important_text": [],
            "deterministic_review_reasons": [],
        }
    ]
    raw_result = FeatureCutQaResult(
        render_hash=render_hash,
        manifest_hash=manifest_hash,
        aspect_ratio="9:16",
        overall_status="pass",
        segments=[
            SegmentQaObservation(
                feature_id="feature-a",
                segment_id="segment-a",
                semantic_match="match",
                target_visibility="clearly_visible",
                important_text_status="not_applicable",
                issues=[],
                evidence_note="The intended subject remains visible.",
            )
        ],
    )
    expected_cache_record = _expected_cache_record(
        cache_key=cache_key,
        render_hash=render_hash,
        manifest_hash=manifest_hash,
        proxy_hash=proxy_hash,
    )
    request = {
        "model": MODEL_ID,
        "system_instruction": VISUAL_EVIDENCE_SYSTEM_INSTRUCTION,
        "store": False,
        "input": [
            {"type": "text", "text": prompt},
            {
                "type": "video",
                "uri": "https://example.invalid/cached-video",
                "mime_type": "video/mp4",
            },
        ],
        "generation_config": QA_GENERATION_CONFIG,
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": schema,
        },
        "cache_key": cache_key,
        "proxy_hash": proxy_hash,
        "segment_contract": segment_contract,
    }
    write_json(run_dir / "cache-key.json", expected_cache_record)
    write_json(run_dir / "request.json", request)
    write_json(run_dir / "raw_interaction.json", {"id": "interaction-cached"})
    write_json(run_dir / "raw_output.json", {"output_text": raw_result.model_dump_json()})
    write_json(run_dir / "validated.json", raw_result)
    write_json(
        run_dir / "schema_validation.json",
        {
            "ok": True,
            "errors": [],
            "model_reported_overall_status": "pass",
            "locally_derived_overall_status": "pass",
            "status_reconciled": False,
            "validator_version": VALIDATOR_VERSION,
        },
    )
    write_json(run_dir / "pricing.json", {"request_count": 1})
    write_json(run_dir / "timing.json", {"status": "completed"})
    write_json(run_dir / "cache-state.json", {"status": "valid"})
    return {
        "expected_cache_record": expected_cache_record,
        "prompt": prompt,
        "schema": schema,
        "proxy_hash": proxy_hash,
        "render_hash": render_hash,
        "manifest_hash": manifest_hash,
        "segment_contract": segment_contract,
    }


def test_cached_run_is_reloaded_and_fully_validated(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    contract = _write_valid_cache_fixture(run_dir)

    result = _validate_cached_run(run_dir, **contract)

    assert result.overall_status == "pass"
    assert result.segments[0].segment_id == "segment-a"


@pytest.mark.parametrize(
    ("filename", "mutate", "message"),
    [
        (
            "cache-key.json",
            lambda value: {**value, "proxy_hash": "e" * 64},
            "cache-key record",
        ),
        (
            "request.json",
            lambda value: {
                **value,
                "response_format": {
                    **value["response_format"],
                    "schema": {"type": "object"},
                },
            },
            "response schema",
        ),
        (
            "validated.json",
            lambda value: {**value, "render_hash": "f" * 64},
            "immutable render or manifest hashes",
        ),
        (
            "validated.json",
            lambda value: {**value, "overall_status": "review"},
            "not locally reconciled",
        ),
        (
            "validated.json",
            lambda value: {
                **value,
                "segments": [
                    {**value["segments"][0], "segment_id": "segment-reordered"}
                ],
            },
            "changed or reordered segment identities",
        ),
        (
            "schema_validation.json",
            lambda value: {
                **value,
                "locally_derived_overall_status": "review",
            },
            "locally_derived_overall_status",
        ),
    ],
)
def test_corrupted_cache_cannot_be_a_hit(
    tmp_path: Path,
    filename: str,
    mutate: Any,
    message: str,
) -> None:
    run_dir = tmp_path / filename.replace(".json", "")
    contract = _write_valid_cache_fixture(run_dir)
    path = run_dir / filename
    write_json(path, mutate(json.loads(path.read_text(encoding="utf-8"))))

    with pytest.raises(ValueError, match=message):
        _validate_cached_run(run_dir, **contract)


def test_refresh_required_cache_state_cannot_be_a_hit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    contract = _write_valid_cache_fixture(run_dir)
    write_json(run_dir / "cache-state.json", {"status": "refresh_required"})

    with pytest.raises(ValueError, match="refresh-required"):
        _validate_cached_run(run_dir, **contract)


def test_schema_contains_no_timestamp_field() -> None:
    payload = json.dumps(_schema())
    assert "timestamp" not in payload
    assert "frame_pts" not in payload
    assert "frame_time" not in payload
